# PufferLib's customized CleanRL PPO + LSTM implementation
# Adapted from https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy

from collections import defaultdict
from pdb import run, set_trace as T
import os
import psutil
import random
import time
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import pufferlib
import pufferlib.utils
import pufferlib.frameworks.cleanrl
import pufferlib.vectorization.multiprocessing
import pufferlib.vectorization.serial

class CleanPuffeRL:
    def __init__(
        self,
        binding,
        agent,
        config={},
        exp_name=os.path.basename(__file__),
        seed=1,
        torch_deterministic=True,
        cuda=True,
        vec_backend=pufferlib.vectorization.multiprocessing.VecEnv,
        total_timesteps=10000000,
        learning_rate=2.5e-4,
        num_buffers=1,
        num_envs=8,
        num_cores=psutil.cpu_count(logical=False),
        num_steps=128,
        run_name=None,
        cpu_offload=True,
        verbose=True
    ):
        self.start_time = time.time()
        self.verbose = verbose

        # Note: Must recompute num_envs for multiagent envs
        envs_per_worker = num_envs / num_cores
        assert envs_per_worker == int(envs_per_worker)
        assert envs_per_worker >= 1
        envs_per_worker = int(envs_per_worker)

        # Create environments
        process = psutil.Process()
        allocated = process.memory_info().rss
        self.buffers = [
            vec_backend(
                binding,
                num_workers=num_cores,
                envs_per_worker=envs_per_worker,
            )
            for _ in range(num_buffers)
        ]

        if verbose:
            print('Allocated %.2f MB to environments. Only accurate for Serial backend.' % ((process.memory_info().rss - allocated) / 1e6))

        self.binding = binding
        self.cpu_offload = cpu_offload
        self.config = config
        self.num_buffers = num_buffers
        self.num_envs = num_envs
        self.num_agents = binding.max_agents
        self.num_cores = num_cores
        self.num_steps = num_steps
        self.envs_per_worker = envs_per_worker
        self.seed = seed

        self.batch_size = int(num_envs * self.num_agents * num_buffers * num_steps)
        self.num_updates = total_timesteps // self.batch_size

        self.global_step = 0
        self.agent_step = 0
        self.start_epoch = 0
        self.update = 0

        # Seed everything
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        # Setup agent
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.agent = agent.to(self.device)
        self.agent.is_recurrent = hasattr(self.agent, "lstm")

        # Setup optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

        # Setup logging
        self.run_name = run_name or f"{binding.env_name}__{seed}__{int(time.time())}"

        self.wandb_run_id = None
        self.wandb_initialized = False

        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.config.items()])),
        )

    def init_wandb(self, wandb_project_name, wandb_entity, wandb_run_id = None):
        if self.wandb_initialized:
            return

        import wandb
        self.wandb_run_id = self.wandb_run_id or wandb_run_id or wandb.util.generate_id()

        wandb.init(
            id=self.wandb_run_id,
            project=wandb_project_name,
            entity=wandb_entity,
            config=self.config,
            sync_tensorboard=True,
            name=self.run_name,
            monitor_gym=True,
            save_code=True,
            resume="allow",
        )
        self.wandb_initialized = True
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.config.items()])),
        )

    def resume_model(self, path):
        resume_state = torch.load(path)
        self.wandb_run_id = resume_state.get('wandb_run_id')
        self.global_step = resume_state.get('global_step', 0)
        self.agent_step = resume_state.get('agent_step', 0)
        self.update = resume_state['update']

        if self.verbose:
            print(f'Resuming from {path} with wandb_run_id={self.wandb_run_id}')

        self.agent.load_state_dict(resume_state['agent_state_dict'])
        self.optimizer.load_state_dict(resume_state['optimizer_state_dict'])

    def save_model(self, save_path, **kwargs):
      state = {
        "agent_state_dict": self.agent.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "wandb_run_id": self.wandb_run_id,
        "global_step": self.global_step,
        "agent_step": self.agent_step,
        "update": self.update,
        **kwargs
      }

      if self.verbose:
          print(f'Saving checkpoint to {save_path}')

      temp_path = os.path.join(f'{save_path}.tmp')
      torch.save(state, temp_path)
      os.rename(temp_path, save_path)

    def allocate_storage(self):
        next_obs, next_done, next_lstm_state = [], [], []
        for i, envs in enumerate(self.buffers):
            envs.async_reset(self.seed + i*self.num_envs)
            o, _, _, _ = envs.recv()
            next_obs.append(torch.Tensor(o).to(self.device))
            next_done.append(torch.zeros((self.num_envs * self.num_agents,)).to(self.device))

            if self.agent.is_recurrent:
                shape = (self.agent.lstm.num_layers, self.num_envs * self.num_agents, self.agent.lstm.hidden_size)
                next_lstm_state.append((
                    torch.zeros(shape).to(self.device),
                    torch.zeros(shape).to(self.device)
                ))
            else:
                next_lstm_state.append(None)

        common_shape = (self.num_steps, self.num_buffers, self.num_envs * self.num_agents)

        allocated = torch.cuda.memory_allocated(self.device)
        data = SimpleNamespace(
            next_obs=next_obs, next_done=next_done, next_lstm_state=next_lstm_state,
            obs=torch.zeros(common_shape + self.binding.single_observation_space.shape).to('cpu' if self.cpu_offload else self.device),
            actions=torch.zeros(common_shape + self.binding.single_action_space.shape, dtype=int).to(self.device),
            logprobs=torch.zeros(common_shape).to(self.device),
            rewards=torch.zeros(common_shape).to(self.device),
            dones=torch.zeros(common_shape).to(self.device),
            values=torch.zeros(common_shape).to(self.device),
        )

        if self.verbose:
            print('Allocated %.2f GB to storage' % ((torch.cuda.memory_allocated(self.device) - allocated) / 1e9))

        return data

    @pufferlib.utils.profile
    def evaluate(self, agent, data, max_episodes=None):
        allocated = torch.cuda.memory_allocated(self.device)

        data.initial_lstm_state = None
        if self.agent.is_recurrent:
            data.initial_lstm_state = [
                torch.cat([e[0].clone() for e in data.next_lstm_state], dim=1),
                torch.cat([e[1].clone() for e in data.next_lstm_state], dim=1)
            ]

        env_step_time = 0
        inference_time = 0
        num_episodes = 0
        for step in range(0, self.num_steps + 1):
            if max_episodes and num_episodes >= max_episodes:
                break

            for buf, envs in enumerate(self.buffers):
                self.global_step += self.num_envs * self.num_agents

                # TRY NOT TO MODIFY: Receive from game and log data
                if step == 0:
                    data.obs[step, buf] = data.next_obs[buf].to('cpu' if self.cpu_offload else self.device)
                    data.dones[step, buf] = data.next_done[buf]
                else:
                    start = time.time()
                    o, r, d, i = envs.recv()
                    env_step_time += time.time() - start

                    data.next_obs[buf] = torch.Tensor(o).to(self.device)
                    data.next_done[buf] = torch.Tensor(d).to(self.device)

                    if step != self.num_steps:
                        data.obs[step, buf] = data.next_obs[buf].to('cpu' if self.cpu_offload else self.device)
                        data.dones[step, buf] = data.next_done[buf]

                    data.rewards[step - 1, buf] = torch.tensor(r).float().to(self.device).view(-1)

                    episode_stats = defaultdict(float)
                    num_stats = 0
                    for item in i:
                        if "episode" in item.keys():
                            self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], self.global_step)
                            self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], self.global_step)

                        for agent_info in item.values():
                            if "episode_stats" in agent_info.keys():
                                num_stats += 1
                                for name, stat in agent_info["episode_stats"].items():
                                    self.writer.add_histogram(f"charts/episode_stats/{name}/hist", stat, self.global_step)
                                    episode_stats[name] += stat

                    if num_stats > 0:
                        print("End of episode:", step)
                        for name, stat in episode_stats.items():
                            self.writer.add_scalar(f"charts/episode_stats/{name}", stat / num_stats, self.global_step)
                            print("Episode stats:", name, stat / num_stats)
                        num_episodes += 1

                if step == self.num_steps:
                    continue

                if max_episodes and num_episodes >= max_episodes:
                    continue

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    start = time.time()
                    if self.agent.is_recurrent:
                        action, logprob, _, value, data.next_lstm_state[buf] = agent.get_action_and_value(data.next_obs[buf].to(self.device), data.next_lstm_state[buf], data.next_done[buf])
                    else:
                        action, logprob, _, value = agent.get_action_and_value(data.next_obs[buf].to(self.device))
                    inference_time += time.time() - start
                    data.values[step, buf] = value.flatten()

                data.actions[step, buf] = action
                data.logprobs[step, buf] = logprob

                # TRY NOT TO MODIFY: execute the game
                start = time.time()
                envs.send(action.cpu().numpy(), None)
                env_step_time += time.time() - start

        epoch_step = self.num_envs * self.num_agents * self.num_buffers * self.num_steps
        env_sps = int(epoch_step / env_step_time)
        inference_sps = int(epoch_step / inference_time)

        self.writer.add_scalar("performance/env_time", env_step_time, self.global_step)
        self.writer.add_scalar("performance/env_sps", env_sps, self.global_step)
        self.writer.add_scalar("performance/inference_time", inference_time, self.global_step)
        self.writer.add_scalar("performance/inference_sps", inference_sps, self.global_step)

        mean_reward = float(torch.mean(data.rewards))
        self.writer.add_scalar("charts/reward", mean_reward, self.global_step)

        for profile in envs.profile():
            for k, v in profile.items():
                # Added deltas to pufferlib.
                # TODO: Test that this matches the original implementation.
                self.writer.add_scalar(f'performance/env/{k}', np.mean(v.delta), self.global_step)

        uptime = timedelta(seconds=int(time.time() - self.start_time))

        if self.verbose:
            print('%.2f GB Allocated at the start of evaluation' % (allocated / 1e9))
            print('Allocated %.2f GB during evaluation\n' % ((torch.cuda.memory_allocated(self.device) - allocated) / 1e9))

        print(
            f'Epoch: {self.update} - {self.global_step // 1000}K steps - {uptime} Elapsed\n'
            f'\tSteps Per Second: Env={env_sps}, Inference={inference_sps}'
        )

        return data

    @pufferlib.utils.profile
    def train(
            self, agent, data,
            anneal_lr=True,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=4,
            update_epochs=4,
            bptt_horizon=16,
            norm_adv=True,
            clip_coef=0.1,
            clip_vloss=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
        ):
        assert self.num_steps % bptt_horizon == 0, "num_steps must be divisible by bptt_horizon"
        allocated = torch.cuda.memory_allocated(self.device)

        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (self.update - 1.0) / self.num_updates
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # bootstrap value if not done
        with torch.no_grad():
            for buf in range(self.num_buffers):
                next_value = agent.get_value(
                    data.next_obs[buf],
                    data.next_lstm_state[buf],
                    data.next_done[buf],
                ).reshape(1, -1)

                advantages = torch.zeros_like(data.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - data.next_done[buf]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - data.dones[t + 1]
                        nextvalues = data.values[t + 1]
                    delta = data.rewards[t] + gamma * nextvalues * nextnonterminal - data.values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + data.values

        #### This is the update logic
        # flatten the batch
        b_obs = data.obs.reshape((num_minibatches, bptt_horizon, -1) + self.binding.single_observation_space.shape)
        b_logprobs = data.logprobs.reshape(num_minibatches, bptt_horizon, -1)
        b_actions = data.actions.reshape((num_minibatches, bptt_horizon, -1) + self.binding.single_action_space.shape)
        b_dones = data.dones.reshape(num_minibatches, bptt_horizon, -1)
        b_values = data.values.reshape(num_minibatches, -1)
        b_advantages = advantages.reshape(num_minibatches, bptt_horizon, -1)
        b_returns = returns.reshape(num_minibatches, -1)

        # Optimizing the policy and value network
        train_time = time.time()
        clipfracs = []
        for epoch in range(update_epochs):
            initial_initial_lstm_state = data.initial_lstm_state
            for minibatch in range(num_minibatches):
                if self.agent.is_recurrent:
                    initial_lstm_state = initial_initial_lstm_state
                    _, newlogprob, entropy, newvalue, initial_lstm_state = agent.get_action_and_value(
                        b_obs[minibatch].to(self.device), initial_lstm_state, b_dones[minibatch], b_actions[minibatch])
                    initial_lstm_state = (initial_lstm_state[0].detach(), initial_lstm_state[1].detach())
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[minibatch].reshape(-1, *self.binding.single_observation_space.shape).to(self.device), action=b_actions[minibatch])

                logratio = newlogprob - b_logprobs[minibatch].reshape(-1)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[minibatch].reshape(-1)
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[minibatch]) ** 2
                    v_clipped = b_values[minibatch] + torch.clamp(
                        newvalue - b_values[minibatch],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[minibatch]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[minibatch]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                self.optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TIMING: performance metrics to evaluate cpu/gpu usage
        train_time = time.time() - train_time
        train_sps = int(self.batch_size / train_time)
        self.update += 1

        print(
            f'\tTrain={train_sps}\n'
        )

        if self.verbose:
            print('Allocated %.2f MB during training' % ((torch.cuda.memory_allocated(self.device) - allocated) / 1e6))

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar("performance/train_sps", train_sps, self.global_step)
        self.writer.add_scalar("performance/train_time", train_time, self.global_step)

        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)

    def close(self):
        self.writer.close()

        if self.wandb_initialized:
            import wandb
            wandb.finish()