# PufferLib's customized CleanRL PPO + LSTM implementation
# TODO: Testing, cleaned up metric/perf/mem logging

from pdb import run, set_trace as T
import os
import psutil
import random
import time
from datetime import timedelta
from types import SimpleNamespace
from collections import defaultdict

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import pufferlib
import pufferlib.emulation
import pufferlib.utils
import pufferlib.frameworks.cleanrl
import pufferlib.vectorization.multiprocessing
import pufferlib.vectorization.serial
import wandb

def unroll_nested_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f'{k}/{k2}', v2
        else:
            yield k, v

@dataclass
class CleanPuffeRL:
    binding: pufferlib.emulation.Binding
    agent: nn.Module
    exp_name: str = os.path.basename(__file__)
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    vec_backend: ... = pufferlib.vectorization.multiprocessing.VecEnv
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_buffers: int = 1
    num_envs: int = 8
    num_cores: int = psutil.cpu_count(logical=False)
    run_name: str = None
    cpu_offload: bool = True
    verbose: bool = True
    batch_size: int = 2**14

    def __post_init__(self, *args, **kwargs):
        self.start_time = time.time()

        self.global_step = self.agent_step = self.start_epoch = self.update = 0
        self.num_updates = self.total_timesteps // self.batch_size
        self.num_agents = self.binding.max_agents
        self.envs_per_worker = self.num_envs // self.num_cores
        assert self.num_cores * self.envs_per_worker == self.num_envs

        # Seed everything
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        # Create environments
        self.process = psutil.Process()
        allocated = self.process.memory_info().rss
        self.buffers = [
            self.vec_backend(
                self.binding,
                num_workers=self.num_cores,
                envs_per_worker=self.envs_per_worker,
            )
            for _ in range(self.num_buffers)
        ]

        if self.verbose:
            print('Allocated %.2f MB to environments. Only accurate for Serial backend.' % ((self.process.memory_info().rss - allocated) / 1e6))

        # Setup agent
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")
        self.agent = self.agent.to(self.device)
        self.agent.is_recurrent = hasattr(self.agent, "lstm")

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # Setup logging
        self.run_name = self.run_name or f"{self.binding.env_name}__{self.seed}__{int(time.time())}"
        self.wandb_run_id = None
        self.wandb_initialized = False

        # Extra logging values
        self.dead_obs = np.zeros((self.num_agents, np.prod(self.binding.single_observation_space.shape)), dtype=bool)

    def init_wandb(
            self, wandb_project_name='pufferlib', wandb_entity=None,
            wandb_run_id = None, extra_data = None):

        if self.wandb_initialized:
            return

        self.wandb_run_id = self.wandb_run_id or wandb_run_id or wandb.util.generate_id()
        extra_data = extra_data or {}

        wandb.init(
            id=self.wandb_run_id,
            project=wandb_project_name,
            entity=wandb_entity,
            config=extra_data,
            sync_tensorboard=True,
            name=self.run_name,
            monitor_gym=True,
            save_code=True,
            resume="allow",
        )
        self.wandb_initialized = True

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
            next_done.append(torch.zeros((self.num_envs * self.num_agents,)).to(self.device))
            next_obs.append([])

            if self.agent.is_recurrent:
                shape = (self.agent.lstm.num_layers, self.num_envs * self.num_agents, self.agent.lstm.hidden_size)
                next_lstm_state.append((
                    torch.zeros(shape).to(self.device),
                    torch.zeros(shape).to(self.device)
                ))
            else:
                next_lstm_state.append(None)

        allocated_torch = torch.cuda.memory_allocated(self.device)
        allocated_cpu = self.process.memory_info().rss
        data = SimpleNamespace(
            buf = 0, sort_keys = [],
            next_obs=next_obs, next_done=next_done, next_lstm_state=next_lstm_state,
            obs = torch.zeros(self.batch_size+1, *self.binding.single_observation_space.shape).to('cpu' if self.cpu_offload else self.device),
            actions=torch.zeros(self.batch_size+1, *self.binding.single_action_space.shape, dtype=int).to(self.device),
            logprobs=torch.zeros(self.batch_size+1).to(self.device),
            rewards=torch.zeros(self.batch_size+1).to(self.device),
            dones=torch.zeros(self.batch_size+1).to(self.device),
            values=torch.zeros(self.batch_size+1).to(self.device),
        )

        allocated_torch = torch.cuda.memory_allocated(self.device) - allocated_torch
        allocated_cpu = self.process.memory_info().rss - allocated_cpu
        if self.verbose:
            print('Allocated to storage - Pytorch: %.2f GB, System: %.2f GB' % (allocated_torch/1e9, allocated_cpu/1e9))

        return data

    @pufferlib.utils.profile
    def evaluate(self, agent, data):
        allocated_torch = torch.cuda.memory_allocated(self.device)
        allocated_cpu = self.process.memory_info().rss
        ptr = env_step_time = inference_time = 0

        step = 0
        stats = defaultdict(list)
        agent_stats = defaultdict(list)
        performance = defaultdict(list)

        while True:
            buf = data.buf

            step += 1
            if ptr == self.batch_size+1:
                break

            start = time.time()
            o, r, d, i = self.buffers[buf].recv()
            env_step_time += time.time() - start

            for profile in self.buffers[buf].profile():
                for k, v in profile.items():
                    performance[k].append(v['delta'])

            o = torch.Tensor(o)
            if not self.cpu_offload:
                o = o.to(self.device)

            r = torch.Tensor(r).float().to(self.device).view(-1)

            if len(d) != 0 and len(data.next_done[buf]) != 0:
                alive_mask = (data.next_done[buf].cpu() + torch.Tensor(d)) != 2
                data.next_done[buf] = torch.Tensor(d).to(self.device)
            else:
                alive_mask = [1 for _ in range(len(o))]

            # ALGO LOGIC: action logic
            start = time.time()
            with torch.no_grad():
                if self.agent.is_recurrent:
                    action, logprob, _, value, data.next_lstm_state[buf] = agent.get_action_and_value(o.to(self.device), data.next_lstm_state[buf], data.next_done[buf])
                else:
                    action, logprob, _, value = agent.get_action_and_value(o.to(self.device))
                value = value.flatten()

            inference_time += time.time() - start

            # TRY NOT TO MODIFY: execute the game
            start = time.time()
            self.buffers[buf].send(action.cpu().numpy(), None)
            env_step_time += time.time() - start
            data.buf = (data.buf + 1) % self.num_buffers

            for idx in np.where(alive_mask)[0]:
                if ptr == self.batch_size+1:
                    break

                data.obs[ptr] = o[idx]
                data.values[ptr] = value[idx]
                data.actions[ptr] = action[idx]
                data.logprobs[ptr] = logprob[idx]
                data.sort_keys.append((buf, idx, step))

                if len(d) != 0:
                    data.rewards[ptr] = r[idx]
                    data.dones[ptr] = d[idx]

                ptr += 1

            for item in i:
                for agent_id, agent_info in item.items():
                    for name, stat in unroll_nested_dict(agent_info):
                        # General stats
                        try:
                            stat = float(stat)
                            stats[name].append(stat)
                        except ValueError:
                            continue
                        # Per-agent stats
                        try:
                            agent_stat = float(stat)
                            agent_stats[f"{name}/agent_{agent_id}"].append(agent_stat)
                        except ValueError:
                            continue

        self.global_step += self.batch_size
        env_sps = int(self.batch_size / env_step_time)
        inference_sps = int(self.batch_size / inference_time)

        if self.wandb_initialized:
            wandb.log({
                "performance/env_time": env_step_time,
                "performance/env_sps": env_sps,
                "performance/inference_time": inference_time,
                "performance/inference_sps": inference_sps,
                **{f'performance/env/{k}': np.mean(v) for k, v in performance.items()},
                **{f'charts/{k}': np.mean(v) for k, v in stats.items()},
                **{f'agents/{k}': np.mean(v) for k, v in agent_stats.items()},
                "charts/reward": float(torch.mean(data.rewards)),
                "agent_steps": self.global_step,
                "global_step": self.global_step,
            })

        allocated_torch = torch.cuda.memory_allocated(self.device) - allocated_torch
        allocated_cpu = self.process.memory_info().rss - allocated_cpu
        if self.verbose:
            print('Allocated during evaluation - Pytorch: %.2f GB, System: %.2f GB' % (allocated_torch/1e9, allocated_cpu/1e9))

        uptime = timedelta(seconds=int(time.time() - self.start_time))
        print(
            f'Epoch: {self.update} - {self.global_step // 1000}K steps - {uptime} Elapsed\n'
            f'\tSteps Per Second: Env={env_sps}, Inference={inference_sps}'
        )

        return data

    @pufferlib.utils.profile
    def train(self, agent, data, batch_rows=32, update_epochs=4,
            bptt_horizon=16, gamma=0.99, gae_lambda=0.95, anneal_lr=True,
            norm_adv=True,clip_coef=0.1, clip_vloss=True, ent_coef=0.01,
            vf_coef=0.5, max_grad_norm=0.5, target_kl=None):

        #assert self.num_steps % bptt_horizon == 0, "num_steps must be divisible by bptt_horizon"
        allocated_torch = torch.cuda.memory_allocated(self.device)
        allocated_cpu = self.process.memory_info().rss

        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (self.update - 1.0) / self.num_updates
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # Sort here
        idxs = sorted(range(len(data.sort_keys)), key=data.sort_keys.__getitem__)
        data.sort_keys = []

        num_minibatches = self.batch_size // bptt_horizon // batch_rows
        b_idxs = torch.Tensor(idxs).long()[:-1].reshape(batch_rows, num_minibatches, bptt_horizon).transpose(0, 1)

        # bootstrap value if not done
        with torch.no_grad():
            advantages = torch.zeros(self.batch_size, device=self.device)
            lastgaelam = 0
            for t in reversed(range(self.batch_size)):
                i, i_nxt = idxs[t], idxs[t + 1]
                nextnonterminal = 1.0 - data.dones[i_nxt]
                nextvalues = data.values[i_nxt]
                delta = data.rewards[i] + gamma * nextvalues * nextnonterminal - data.values[i]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

        # Flatten the batch
        data.b_obs = b_obs = data.obs[b_idxs]
        b_actions=data.actions[b_idxs]
        b_logprobs=data.logprobs[b_idxs]
        b_dones = data.dones[b_idxs]
        b_values = data.values[b_idxs]
        b_advantages = advantages.reshape(batch_rows, num_minibatches, bptt_horizon).transpose(0, 1)
        b_returns = b_advantages + b_values

        # Extra debug logging
        start_dead_time = time.time()
        # Calculate percent of observations that have never received a value other than 0
        # Reduce along batch and bptt dims to get a single boolean for every observation feature for each agent
        self.dead_obs = np.logical_or(self.dead_obs, np.logical_or.reduce(b_obs.cpu().numpy(), axis=(0, 1)))
        check_dead_time = time.time() - start_dead_time

        # Optimizing the policy and value network
        train_time = time.time()
        clipfracs = []
        for epoch in range(update_epochs):
            lstm_state = None
            for mb in range(num_minibatches):
                mb_obs = b_obs[mb].to(self.device)
                mb_actions = b_actions[mb].contiguous()
                mb_values = b_values[mb].reshape(-1)
                mb_advantages = b_advantages[mb].reshape(-1)
                mb_returns = b_returns[mb].reshape(-1)

                if self.agent.is_recurrent:
                    _, newlogprob, entropy, newvalue, lstm_state = agent.get_action_and_value(mb_obs, lstm_state, b_dones[mb], mb_actions)
                    lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        mb_obs.reshape(-1, *self.binding.single_observation_space.shape), action=mb_actions)

                logratio = newlogprob - b_logprobs[mb].reshape(-1)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = mb_advantages.reshape(-1)
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

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

        allocated_torch = torch.cuda.memory_allocated(self.device) - allocated_torch
        allocated_cpu = self.process.memory_info().rss - allocated_cpu
        if self.verbose:
            print('Allocated during training - Pytorch: %.2f GB, System: %.2f GB' % (allocated_torch/1e9, allocated_cpu/1e9))

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if self.wandb_initialized:
            wandb.log({
                "performance/train_sps": train_sps,
                "performance/train_time": train_time,
                "performance/check_dead_time": check_dead_time,
                "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                "charts/dead_obs_percent": np.count_nonzero(self.dead_obs == 0) / np.prod(self.dead_obs.shape) * 100,
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "agent_steps": self.global_step,
                "global_step": self.global_step,
            })

    def close(self):
        if self.wandb_initialized:
            wandb.finish()