# PufferLib's customized CleanRL PPO + LSTM implementation
# Adapted from https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy

from pdb import set_trace as T
import os
import psutil
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import pufferlib
import pufferlib.frameworks.cleanrl
import pufferlib.vectorization.multiprocessing
import pufferlib.vectorization.serial


def train(
        binding,
        agent,
        exp_name=os.path.basename(__file__),
        seed=1,
        torch_deterministic=True,
        cuda=True,
        track=False,
        wandb_project_name='cleanRL',
        wandb_entity=None,
        total_timesteps=10000000,
        learning_rate=2.5e-4,
        num_buffers=1,
        num_envs=8,
        num_agents=1,
        num_cores=psutil.cpu_count(logical=False),
        num_steps=128,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=4,
        update_epochs=4,
        norm_adv=True,
        clip_coef=0.1,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
    ):
    program_start = time.time()
    env_id = binding.env_name
    args = pufferlib.utils.dotdict(locals())
    batch_size = int(num_envs * num_agents * num_buffers * num_steps)

    run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"
    if track:
        import wandb

        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # Note: Must recompute num_envs for multiagent envs
    envs_per_worker = num_envs / num_cores
    assert envs_per_worker == int(envs_per_worker)
    assert envs_per_worker >= 1

    buffers = []
    for i in range(num_buffers):
        buffers.append(
                pufferlib.vectorization.multiprocessing.VecEnv(
                    binding,
                    num_workers=num_cores,
                    envs_per_worker=int(envs_per_worker),
                )
        )

    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_buffers, num_envs * num_agents) + binding.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_buffers, num_envs * num_agents) + binding.single_action_space.shape, dtype=int).to(device)
    logprobs = torch.zeros((num_steps, num_buffers, num_envs * num_agents)).to(device)
    rewards = torch.zeros((num_steps, num_buffers, num_envs * num_agents)).to(device)
    dones = torch.zeros((num_steps, num_buffers, num_envs * num_agents)).to(device)
    values = torch.zeros((num_steps, num_buffers, num_envs * num_agents)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, next_done, next_lstm_state = [], [], []
    for envs in buffers:
        envs.async_reset(seed=seed + int(i*num_cores*envs_per_worker*num_agents))
        o, _, _, info = envs.recv()
        next_obs.append(torch.Tensor(o).to(device))
        next_done.append(torch.zeros((num_envs * num_agents,)).to(device))
        
        next_lstm_state.append((
            torch.zeros(agent.lstm.num_layers, num_envs * num_agents, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, num_envs * num_agents, agent.lstm.hidden_size).to(device),
        ))  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    num_updates = total_timesteps // batch_size

    for update in range(1, num_updates + 1):
        epoch_lengths = []
        epoch_returns = []
        epoch_time = time.time()
        epoch_step = 0

        initial_lstm_state = [
            torch.cat([e[0].clone() for e in next_lstm_state], dim=1),
            torch.cat([e[1].clone() for e in next_lstm_state], dim=1)
        ]

        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        env_step_time = 0
        inference_time = 0
        for step in range(0, num_steps + 1):
            for buf, envs in enumerate(buffers):
                global_step += num_envs * num_agents

                # TRY NOT TO MODIFY: Receive from game and log data
                if step == 0:
                    obs[step, buf] = next_obs[buf]
                    dones[step, buf] = next_done[buf]
                else:
                    start = time.time()
                    o, r, d, i = envs.recv()
                    env_step_time += time.time() - start

                    next_obs[buf] = torch.Tensor(o).to(device)
                    next_done[buf] = torch.Tensor(d).to(device)

                    if step != num_steps:
                        obs[step, buf] = next_obs[buf]
                        dones[step, buf] = next_done[buf]


                    rewards[step - 1, buf] = torch.tensor(r).float().to(device).view(-1)

                    for item in i:
                        if "episode" in item.keys():
                            epoch_lengths.append(item["episode"]["l"])
                            epoch_returns.append(item["episode"]["r"])
                            writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)

                if step == num_steps:
                    continue

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    start = time.time()
                    action, logprob, _, value, next_lstm_state[buf] = agent.get_action_and_value(next_obs[buf], next_lstm_state[buf], next_done[buf])
                    #action, logprob, _, value, _ = agent.get_action_and_value(next_obs[buf], None, next_done[buf])
                    inference_time += time.time() - start
                    values[step, buf] = value.flatten()

                actions[step, buf] = action
                logprobs[step, buf] = logprob

                # TRY NOT TO MODIFY: execute the game
                start = time.time()
                envs.send(action.cpu().numpy(), None)
                env_step_time += time.time() - start

        # bootstrap value if not done
        with torch.no_grad():
            for buf in range(num_buffers):
                next_value = agent.get_value(
                    next_obs[buf],
                    next_lstm_state[buf],
                    next_done[buf],
                ).reshape(1, -1)

                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done[buf]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        train_time = time.time()
        assert num_envs * num_buffers % num_minibatches == 0
        agentsperbatch = num_envs * num_agents * num_buffers // num_minibatches
        agentinds = np.arange(num_envs * num_agents * num_buffers)
        flatinds = np.arange(batch_size).reshape(num_steps, num_envs * num_agents * num_buffers)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(agentinds)
            for start in range(0, num_envs * num_agents * num_buffers, agentsperbatch):
                end = start + agentsperbatch
                mbagentinds = agentinds[start:end]
                mb_inds = flatinds[:, mbagentinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbagentinds], initial_lstm_state[1][:, mbagentinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        mean_reward = float(torch.mean(rewards))
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TIMING: performance metrics to evaluate cpu/gpu usage
        epoch_step += batch_size

        train_time = time.time() - train_time
        epoch_time = time.time() - epoch_time
        extra_time = epoch_time - train_time - env_step_time

        env_sps = int(epoch_step / env_step_time)
        inference_sps = int(epoch_step / inference_time)
        train_sps = int(epoch_step / train_time)
        epoch_sps = int(epoch_step / epoch_time)

        remaining = timedelta(seconds=int((total_timesteps - global_step) / epoch_sps))
        uptime = timedelta(seconds=int(time.time() - program_start))
        completion_percentage = 100 * global_step / total_timesteps

        if len(epoch_returns) > 0:
            epoch_return = np.mean(epoch_returns)
            epoch_length = int(np.mean(epoch_lengths))
        else:
            epoch_return = 0.0
            epoch_length = 0.0

        print(
            f'Epoch: {update} - Mean Return: {epoch_return:<5.4}, Episode Length: {epoch_length}\n'
            f'\t{completion_percentage:.3}% / {global_step // 1000}K steps - {uptime} Elapsed, ~{remaining} Remaining\n'
            f'\tSteps Per Second: Overall={epoch_sps}, Env={env_sps}, Inference={inference_sps}, Train={train_sps}\n'
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("performance/env_time", env_step_time, global_step)
        writer.add_scalar("performance/inference_time", inference_time, global_step)
        writer.add_scalar("performance/train_time", train_time, global_step)
        writer.add_scalar("performance/epoch_time", epoch_time, global_step)
        writer.add_scalar("performance/extra_time", extra_time, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/reward", mean_reward, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

    envs.close()
    writer.close()

    if track:
        wandb.finish()

if __name__ == '__main__':
    import sys
    assert sys.argv[1] in 'nmmo nethack breakout atari'.split()

    cuda = torch.cuda.is_available()
    track = True # WandB tracking

    # Note: the dependencies for these demos are not currently compatible
    # Run pip install -e .[atari] or .[nethack] to install the dependencies for a specific demo 
    # This will be resolved in a future update
    if sys.argv[1] == 'nmmo':
        # Neural MMO requires the CarperAI fork of the nmmo-environment repo
        import pufferlib.registry.nmmo
        binding = pufferlib.registry.nmmo.make_binding()
        agent = pufferlib.frameworks.cleanrl.make_policy(
            pufferlib.registry.nmmo.Policy,
        )(binding)

        train(
            binding,
            agent,
            cuda=cuda,
            total_timesteps=10_000_000,
            track=track,
            num_envs=2,
            num_cores=2,
            num_minibatches=1,
            num_agents=128,
            wandb_project_name='pufferlib',
            wandb_entity='jsuarez',
        )
    elif sys.argv[1] == 'nethack':
        import pufferlib.registry.nethack
        binding = pufferlib.registry.nethack.make_binding()
        agent = pufferlib.frameworks.cleanrl.make_policy(
            pufferlib.registry.nethack.Policy,
        )(binding)

        train(
            binding,
            agent, 
            cuda=cuda,
            total_timesteps=10_000_000,
            track=track,
            num_envs=12,
            num_cores=4,
            num_minibatches=1,
            num_agents=1,
            wandb_project_name='pufferlib',
            wandb_entity='jsuarez',
        )
    else:
        import pufferlib.registry.atari
        bindings = [pufferlib.registry.atari.make_binding('BreakoutNoFrameskip-v4', framestack=1)]
        if sys.argv[1] == 'atari':
            bindings += [
                pufferlib.registry.atari.make_binding('BeamRiderNoFrameskip-v4', framestack=1),
                pufferlib.registry.atari.make_binding('PongNoFrameskip-v4', framestack=1),
                pufferlib.registry.atari.make_binding('EnduroNoFrameskip-v4', framestack=1),
                pufferlib.registry.atari.make_binding('QbertNoFrameskip-v4', framestack=1),
                pufferlib.registry.atari.make_binding('SeaquestNoFrameskip-v4', framestack=1),
                pufferlib.registry.atari.make_binding('SpaceInvadersNoFrameskip-v4', framestack=1),
            ]
 
        for binding in bindings:
            agent = pufferlib.frameworks.cleanrl.make_policy(
                pufferlib.registry.atari.Policy,
            )(binding, framestack=1)

            train(
                binding,
                agent,
                num_cores=4, 
                num_envs=4,
                num_buffers=2,
                cuda=cuda,
                total_timesteps=10_000_000,
                track=track,
                wandb_project_name='pufferlib',
                wandb_entity='jsuarez',
            )
