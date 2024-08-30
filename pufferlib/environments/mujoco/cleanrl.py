# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import configparser
import argparse
import random
import time
import ast
import os
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib.frameworks.cleanrl
from pufferlib.environments.mujoco.environment import cleanrl_env_creator
from pufferlib.environments.mujoco.policy import CleanRLPolicy, Policy


if __name__ == "__main__":
    # Simpler args parse just for this script. Configs are read from file ONLY.
    parser = argparse.ArgumentParser(description="Training arguments for cleanrl mujoco", add_help=False)
    parser.add_argument("-e", "--env-id", type=str, default="Hopper-v4")
    parser.add_argument("-c", "--config", type=str, default="config/mujoco.ini")
    # The current cleanrl script does NOT support recurrent policies
    parser.add_argument("-p", "--policy", type=str, default="cleanrl", choices=["cleanrl", "puffer"])
    parser.add_argument("--wandb-project", type=str, default="cleanrl_mujoco")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--track", action="store_true", help="Track on WandB")
    parser.add_argument("--capture-video", action="store_true", help="Capture videos")
    args = parser.parse_known_args()[0]

    if not os.path.exists(args.config):
        raise Exception(f'Config {args.config} not found')

    p = configparser.ConfigParser()
    p.read(args.config)
    assert args.env_id in p['base']['env_name'].split(), f"Env {args.env_id} not found in {args.config}"

    for section in p.sections():
        for key in p[section]:
            argparse_key = f'--{section}.{key}'.replace('_', '-')
            parser.add_argument(argparse_key, default=p[section][key])

    parsed = parser.parse_args().__dict__
    args_dict = {'env': {}, 'policy': {}, 'rnn': {}}
    env_name = parsed.pop('env_id')
    for key, value in parsed.items():
        next = args_dict
        for subkey in key.split('.'):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:
            prev[subkey] = value

    run_name = f"cleanrl_{env_name}_{args_dict['train']['seed']}_{int(time.time())}"

    # Translate puffer args to cleanrl args
    args = SimpleNamespace(**args_dict["train"])
    args.env_id = env_name
    args.policy = args_dict["policy"]
    args.wandb_project = args_dict["wandb_project"]
    args.wandb_group = args_dict["wandb_group"]
    args.track = args_dict["track"]
    args.capture_video = args_dict["capture_video"]
    args.cuda = args_dict["train"]["device"] == "cuda"
    if not hasattr(args, "target_kl"):
        args.target_kl = None

    # NOTE: These are CRITICAL (but confusing) for cleanrl to work
    args.num_steps = args.batch_size
    num_minibatches = args.minibatch_size

    # These are placeholder values for cleanrl
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    wandb = None
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            save_code=True,
            name=run_name,
            config=args,
        )

    episode_stats = {
        "episode_return": [],
        "episode_length": [],
    }

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gymnasium.vector.SyncVectorEnv(
        [
            cleanrl_env_creator(args.env_id, run_name, args.capture_video, args.gamma, i)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gymnasium.spaces.Box
    ), "only continuous action space is supported"

    if args.policy == "cleanrl":
        policy = CleanRLPolicy(envs)
    elif args.policy == "puffer":
        policy = Policy(envs)
    
    agent = pufferlib.frameworks.cleanrl.Policy(policy).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode_return" in info:
                        print(
                            f"global_step: {global_step}, episode_return: {int(info['episode_return'])}"
                        )
                        episode_stats["episode_return"].append(info["episode_return"])
                        episode_stats["episode_length"].append(info["episode_length"])

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.vf_clip_coef,
                        args.vf_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"Steps: {global_step}, SPS: {int(global_step / (time.time() - start_time))}")
        if args.track and wandb is not None:
            wandb.log(
                {
                    "0verview/agent_steps": global_step,
                    "0verview/SPS": int(global_step / (time.time() - start_time)),
                    "0verview/epoch": iteration,
                    "0verview/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "environment/episode_return": np.mean(episode_stats["episode_return"]),
                    "environment/episode_length": np.mean(episode_stats["episode_length"]),
                }
            )
        episode_stats["episode_return"].clear()
        episode_stats["episode_length"].clear()

    # if args.save_model:
    #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    #     torch.save(agent.state_dict(), model_path)
    #     print(f"model saved to {model_path}")
    #     from cleanrl_utils.evals.ppo_eval import evaluate

    #     episodic_returns = evaluate(
    #         model_path,
    #         make_env,
    #         args.env_id,
    #         eval_episodes=10,
    #         run_name=f"{run_name}-eval",
    #         Model=Agent,
    #         device=device,
    #         gamma=args.gamma,
    #     )
    #     for idx, episodic_return in enumerate(episodic_returns):
    #         writer.add_scalar("eval/episodic_return", episodic_return, idx)

    #     if args.upload_model:
    #         from cleanrl_utils.huggingface import push_to_hub

    #         repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #         repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #         push_to_hub(
    #             args,
    #             episodic_returns,
    #             repo_id,
    #             "PPO",
    #             f"runs/{run_name}",
    #             f"videos/{run_name}-eval",
    #         )

    envs.close()
    if args.track and wandb is not None:
        wandb.finish()