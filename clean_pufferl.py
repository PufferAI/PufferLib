from pdb import set_trace as T
import numpy as np

import os
import random
import time
import uuid

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib
import pufferlib.utils
import pufferlib.emulation
import pufferlib.vectorization
import pufferlib.frameworks.cleanrl
import pufferlib.policy_pool


@pufferlib.dataclass
class Profile:
    uptime = 0
    epoch = 0
    epoch_sps = 0
    agent_steps = 0
    train_time = 0
    eval_time = 0
    env_time = 0
    misc_time = 0
    alloc_time = 0
    backward_time = 0

    def __init__(self, config):
        self.start = time.time()
        env = pufferlib.utils.Profiler()
        inference = pufferlib.utils.Profiler()
        misc = pufferlib.utils.Profiler()

    @property
    def epoch_time(self):
        return self.train_time + self.eval_time

    def update(self):
        self.uptime = time.time() - self.start
        self.eval_time = self.misc.elapsed
        self.inference_time = self.inference.elapsed
        self.env_time = self.env.elapsed
        self.misc_time = self.misc.elapsed

    def update_train_perf():
        # Shorten or details
        uptime = int(time.time() - data.start_time)
        SPS = int(data.padded_steps_collected / perf.epoch_time)
        estimated_duration = int(config.total_timesteps / profile.SPS)


@pufferlib.dataclass
class Losses:
    policy_loss = 0
    value_loss = 0
    entropy = 0
    old_approx_kl = 0
    approx_kl = 0
    clipfrac = 0
    explained_variance = 0

    def update(self, policy, value, entropy, old_approx_kl,
            approx_kl, clipfrac, explained_variance):
        self.policy_loss = torch.stack(policy).cpu().mean().item()
        self.value_loss = torch.stack(value).cpu().mean().item()
        self.entropy = torch.stack(entropy).cpu().mean().item()
        self.old_approx_kl = torch.stack(old_approx_kl).cpu().mean().item()
        self.approx_kl = torch.stack(approx_kl).cpu().mean().item()
        self.clipfrac = torch.stack(clipfrac).cpu().mean().item()
        self.explained_variance = explained_variance

class Experience:
    @pufferlib.utils.profile
    def __init__(self, config, policy, obs_shape, atn_shape, total_agents):
        # Storage buffers
        obs=torch.zeros(config.batch_size + 1, *obs_shape)
        actions=torch.zeros(config.batch_size + 1, *atn_shape, dtype=int)
        logprobs=torch.zeros(config.batch_size + 1)
        rewards=torch.zeros(config.batch_size + 1)
        dones=torch.zeros(config.batch_size + 1)
        truncateds=torch.zeros(config.batch_size + 1)
        values=torch.zeros(config.batch_size + 1)

        # Numpy views into storage for faster indexing
        obs_ary = np.asarray(obs)
        actions_ary = np.asarray(actions)
        logprobs_ary = np.asarray(logprobs)
        rewards_ary = np.asarray(rewards)
        dones_ary = np.asarray(dones)
        truncateds_ary = np.asarray(truncateds)
        values_ary = np.asarray(values)

        if hasattr(policy, 'lstm'):
            shape = (policy.lstm.num_layers, total_agents, policy.lstm.hidden_size)
            self.next_lstm_state = (
                torch.zeros(shape).to(config.device),
                torch.zeros(shape).to(config.device),
            )

        self.ptr = 0
        self.step = 0
        self.sort_keys = [],
        self.batch_size = config.batch_size
        self.bppt_horizon = config.bptt_horizon
        self.num_minibatches = config.batch_size // config.bptt_horizon // config.batch_rows

        self.mb_obs_buffer = torch.zeros(config.batch_rows, config.bptt_horizon,
            obs_shape, pin_memory=(config.device=="cuda"))

    @property
    def full(self):
        return self.ptr >= self.config.batch_size

    def store(self, obs, value, action, logprob, reward, done, env_id, mask):
        # Mask learner and Ensure indices do not exceed batch size
        indices = torch.where(mask)[0][:self.batch_size - self.ptr + 1].numpy()

        ptr = self.ptr
        end = ptr + len(indices)
        self.ptr = end
 
        self.obs_ary[ptr:end] = obs.cpu().numpy()[indices]
        self.values_ary[ptr:end] = value.cpu().numpy()[indices]
        self.actions_ary[ptr:end] = action[indices]
        self.logprobs_ary[ptr:end] = logprob.cpu().numpy()[indices]
        self.rewards_ary[ptr:end] = reward.cpu().numpy()[indices]
        self.dones_ary[ptr:end] = done.cpu().numpy()[indices]
        self.sort_keys.extend([(env_id[i], self.step) for i in indices])
        self.step += 1

    def sort_training_data(self):
        start_time = time.time()
        sort_keys = self.sort_keys
        idxs = sorted(range(len(sort_keys)), key=sort_keys.__getitem__)
        self.sort_keys = []
        self.ptr = 0
        self.step = 0

        return torch.Tensor(idxs).long()[:-1].reshape(
            self.batch_rows, self.num_minibatches, self.bptt_horizon).transpose(0, 1)

    def flatten_batch(self, b_idxs, advantages):
        self.b_obs = torch.as_tensor(self.obs_ary[b_idxs])
        self.b_actions = torch.as_tensor(self.actions_ary[b_idxs]
            ).to(self.device, non_blocking=True)
        self.b_logprobs = torch.as_tensor(self.logprobs_ary[b_idxs]
            ).to(self.device, non_blocking=True)
        self.b_dones = torch.as_tensor(self.dones_ary[b_idxs]
            ).to(self.device, non_blocking=True)
        self.b_values = torch.as_tensor(self.values_ary[b_idxs]
            ).to(self.device, non_blocking=True)
        self.b_advantages = advantages.reshape(
            self.batch_rows, self.num_minibatches, self.config.bptt_horizon
        ).transpose(0, 1)
        self.b_returns = self.b_advantages + self.b_values


    def minibatches(self):
        for mb in range(self.num_minibatches):
            self.mb_obs_buffer.copy_(self.b_obs[mb], non_blocking=True)
            mb_obs = self.mb_obs_buffer.to(self.device, non_blocking=True)
            mb_actions = self.b_actions[mb].contiguous()
            mb_values = self.b_values[mb].reshape(-1)
            mb_advantages = self.b_advantages[mb].reshape(-1)
            mb_returns = self.b_returns[mb].reshape(-1)
            yield mb_obs, mb_actions, mb_values, mb_advantages, mb_returns

def print_size(policy):
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f'Model Size: {n_params//1000} K parameters')

def save_checkpoint(data):
    path = os.path.join(data.config.data_dir, data.exp_name)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f'model_{data.update:06d}.pt'
    model_path = os.path.join(path, model_name)

    # Already saved
    if os.path.exists(model_path):
        return model_path

    torch.save(data.uncompiled_agent, model_path)

    state = {
        "optimizer_state_dict": data.optimizer.state_dict(),
        "global_step": data.global_step,
        "agent_step": data.global_step,
        "update": data.update,
        "model_name": model_name,
    }

    if data.wandb:
        state['exp_name'] = data.exp_name

    state_path = os.path.join(path, 'trainer_state.pt')
    torch.save(state, state_path + '.tmp')
    os.rename(state_path + '.tmp', state_path)

    return model_path

def try_load_checkpoint(data, epoch=None):
    config = data.config
    resume_state = {}
    path = os.path.join(config.data_dir, config.exp_name)

    if not os.path.exists(path):
        print('No checkpoints found. Assuming new experiment')
        return

    trainer_path = os.path.join(path, 'trainer_state.pt')
    resume_state = torch.load(trainer_path)
    model_path = os.path.join(path, resume_state["model_name"])
    data.policy.uncompiled.load_state_dict(model_path, map_location=config.device)
    data.optimizer.load_state_dict(resume_state["optimizer_state_dict"])
    print(f'Resumed from update {resume_state["update"]} '
          f'with policy {resume_state["model_name"]}')

def create(config, policy, vecenv, optimizer=None, wandb=None,
        policy_selector=pufferlib.policy_pool.random_selector):
    seed_everything(config.seed, config.torch_deterministic)
    vecenv.async_reset(config.seed)

    obs_shape = vecenv.single_observation_space.shape
    atn_shape = vecenv.single_action_space.shape,
    total_agents = vecenv.num_envs * vecenv.agents_per_env

    experience = Experience(config, policy, obs_shape, atn_shape, total_agents)
    profile = Profile(config, vecenv, policy)
    losses = Losses()

    #self.total_updates = config.total_timesteps // config.batch_size

    print_size(policy)

    uncompiled_policy = policy
    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode)

    optimizer = optim.Adam(policy.parameters(),
        lr=config.learning_rate, eps=1e-5)

    # Wraps the policy for self-play. Disregard for single-agent
    policy = pufferlib.policy_pool.PolicyPool(
        policy=policy,
        total_agents=vecenv.agents_per_batch,
        atn_shape=vecenv.single_action_space.shape,
        device=config.device,
        data_dir=config.data_dir,
        kernel=config.pool_kernel,
        policy_selector=pufferlib.policy_pool.random_selector,
        global_step=0,
    )

    return pufferlib.namespace(config=config, policy=policy,
        optimizer=optimizer, vecenv=vecenv, wandb=wandb,)

@pufferlib.utils.profile
def evaluate(data):
    config, profile, experience = data.config, data.profile, data.experience

    if data.wandb is not None and data.performance.total_uptime > 0:
        data.wandb.log({
            'SPS': profile.SPS,
            'global_step': data.global_step,
            'learning_rate': data.optimizer.param_groups[0]["lr"],
            **{f'losses/{k}': v for k, v in data.losses.items()},
            **{f'performance/{k}': v
                for k, v in data.performance.items()},
            **{f'stats/{k}': v for k, v in data.stats.items()},
            **{f'skillrank/{policy}': elo
                for policy, elo in data.policy_pool.ranker.ratings.items()},
        })

    data.policy.update_policies()

    ptr = steps = agent_steps = 0
    infos = defaultdict(lambda: defaultdict(list))
    while not experience.full:
        with profile.env:
            o, r, d, t, i, env_id, mask = data.pool.recv()

        with profile.misc:
            i = data.policy_pool.update_scores(i, "return")
            # TODO: Update this for policy pool
            for ii, ee  in zip(i['learner'], env_id):
                ii['env_id'] = ee

            steps += sum(mask) # May contain padding
            agent_steps += len(mask)

        with profile.inference, torch.no_grad():
            o = torch.as_tensor(o)
            r = torch.as_tensor(r)
            d = torch.as_tensor(d)

            # Multiple policies will not work with new envpool
            next_lstm_state = data.next_lstm_state
            if next_lstm_state is not None:
                next_lstm_state = (
                    next_lstm_state[0][:, env_id],
                    next_lstm_state[1][:, env_id],
                )

            actions, logprob, value, next_lstm_state = data.policy_pool.forwards(
                    o.to(data.device), next_lstm_state)

            if next_lstm_state is not None:
                h, c = next_lstm_state
                data.next_lstm_state[0][:, env_id] = h
                data.next_lstm_state[1][:, env_id] = c

            value = value.flatten()

        with profile.misc:
            actions = actions.cpu().numpy()
            mask = torch.as_tensor(mask * data.policy_pool.mask)
            experience.store(o, value, actions, logprob, r, d, env_id, mask)

            for policy_name, policy_i in i.items():
                for agent_i in policy_i:
                    for name, dat in unroll_nested_dict(agent_i):
                        infos[policy_name][name].append(dat)

        with profile.env:
            data.pool.send(actions)

    data.stats = {}
    infos = infos['learner']

    # Moves into models... maybe. Definitely moves. You could also just return infos and have it in demo
    if 'pokemon_exploration_map' in infos:
        for idx, pmap in zip(infos['env_id'], infos['pokemon_exploration_map']):
            if not hasattr(data, 'pokemon'):
                import pokemon_red_eval
                data.map_updater = pokemon_red_eval.map_updater()
                data.map_buffer = np.zeros((data.config.num_envs, *pmap.shape))

            data.map_buffer[idx] = pmap

        pokemon_map = np.sum(data.map_buffer, axis=0)
        rendered = data.map_updater(pokemon_map)
        data.stats['Media/exploration_map'] = data.wandb.Image(rendered)

    data.global_step += steps
    data.padded_steps_collected = agent_steps
    data.reward = float(torch.mean(data.rewards))

    profile.update_eval_perf()

    return data.stats, infos

@pufferlib.utils.profile
def train(data):
    config, profile, experience = data.config, data.profile, data.experience
    losses = data.losses

    if done_training(data):
        raise RuntimeError(
            f"Max training updates {data.total_updates} already reached")

    config = data.config
    # assert data.num_steps % bptt_horizon == 0, "num_steps must be divisible by bptt_horizon"

    if config.anneal_lr:
        frac = 1.0 - (data.update - 1.0) / data.total_updates
        lrnow = frac * config.learning_rate
        data.optimizer.param_groups[0]["lr"] = lrnow

    # Details
    idxs = experience.sort_training_data()
    start_time = time.time()

    # bootstrap value if not done
    dones_np = data.dones.numpy()
    values_np = data.values.numpy()
    rewards_np = data.rewards.numpy()
    with torch.no_grad():
        advantages = np.zeros(config.batch_size)
        lastgaelam = 0
        for t in reversed(range(config.batch_size)):
            i, i_nxt = idxs[t], idxs[t + 1]

            nextnonterminal = 1.0 - dones_np[i_nxt]
            nextvalues = values_np[i_nxt]
            delta = (
                rewards_np[i_nxt]
                + config.gamma * nextvalues * nextnonterminal
                - values_np[i]
            )
            advantages[t] = lastgaelam = (
                delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            )

    advantages = torch.from_numpy(advantages).to(data.device)
    experience.flatten_batch(idxs, advantages)

    # Optimizing the policy and value network
    pg_losses, entropy_losses, v_losses, clipfracs, old_kls, kls = [], [], [], [], [], []

    for epoch in range(config.update_epochs):
        lstm_state = None
        for obs, atn, val, adv, ret, log_probs in experience.minibatches():
            if hasattr(data.agent, 'lstm'):
                _, newlogprob, entropy, newvalue, lstm_state = data.agent(
                    obs, state=lstm_state, action=atn)
                lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
            else:
                _, newlogprob, entropy, newvalue = data.agent(
                    obs.reshape(-1, *data.pool.single_observation_space.shape),
                    action=atn,
                )

            logratio = newlogprob - log_probs.reshape(-1)
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                old_kls.append(old_approx_kl)
                approx_kl = ((ratio - 1) - logratio).mean()
                kls.append(approx_kl)
                clipfracs += [
                    ((ratio - 1.0).abs() > config.clip_coef).float().mean()
                ]

            mb_advantages = mb_advantages.reshape(-1)
            if config.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - config.clip_coef, 1 + config.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            pg_losses.append(pg_loss)#.item())

            # Value loss
            newvalue = newvalue.view(-1)
            if config.clip_vloss:
                v_loss_unclipped = (newvalue - ret) ** 2
                v_clipped = val + torch.clamp(
                    newvalue - val,
                    -config.vf_clip_coef,
                    config.vf_clip_coef,
                )
                v_loss_clipped = (v_clipped - ret) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

            v_losses.append(v_loss)
            entropy_loss = entropy.mean()
            entropy_losses.append(entropy_loss)#.item())

            loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
            data.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(data.agent.parameters(), config.max_grad_norm)
            data.optimizer.step()

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    y_pred, y_true = experience.values.cpu().numpy(), experience.returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    perf = data.performance
    losses.update()
    data.update += 1

    profile.update_train()
    if data.update % config.checkpoint_interval == 0 or done_training(data):
        save_checkpoint(data)

def close(data):
    data.pool.close()

    if data.wandb is not None:
        artifact_name = f"{data.exp_name}_model"
        artifact = data.wandb.Artifact(artifact_name, type="model")
        model_path = save_checkpoint(data)
        artifact.add_file(model_path)
        data.wandb.run.log_artifact(artifact)
        data.wandb.finish()

def rollout(env_creator, env_kwargs, agent_creator, agent_kwargs,
        model_path=None, device='cuda', verbose=True):
    env = env_creator(**env_kwargs)
    if model_path is None:
        agent = agent_creator(env, **agent_kwargs)
    else:
        agent = torch.load(model_path, map_location=device)

    terminal = truncated = True
 
    while True:
        if terminal or truncated:
            if verbose:
                print('---  Reset  ---')

            ob, info = env.reset()
            state = None
            step = 0
            return_val = 0

        ob = torch.tensor(ob).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(agent, 'lstm'):
                action, _, _, _, state = agent(ob, state)
            else:
                action, _, _, _ = agent(ob)

        ob, reward, terminal, truncated, _ = env.step(action[0].item())
        return_val += reward

        chars = env.render()
        print("\033c", end="")
        print(chars)

        if verbose:
            print(f'Step: {step} Reward: {reward:.4f} Return: {return_val:.2f}')

        time.sleep(0.5)
        step += 1

def done_training(data):
    return data.global_step >= data.config.total_timesteps

def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v

import psutil
import GPUtil

import rich
from rich.console import Console
from rich.table import Table

ROUND_OPEN = rich.box.Box(
    "╭──╮\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "╰──╯\n"
)

c1 = '[bright_cyan]'
c2 = '[white]'
c3 = '[cyan]'
b1 = '[bright_cyan]'
b2 = '[bright_white]'

def abbreviate(num):
    if num < 1e3:
        return f"{num:.0f}"
    elif num < 1e6:
        return f"{num/1e3:.1f}k"
    elif num < 1e9:
        return f"{num/1e6:.1f}m"
    elif num < 1e12:
        return f"{num/1e9:.1f}b"
    else:
        return f"{num/1e12:.1f}t"

def duration(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s"


def print_dashboard(uptime, estimated_duration, epoch,
        global_step, sps, performance, losses, user):
    dashboard = Table(box=ROUND_OPEN, expand=True,
        show_header=False, border_style='bright_cyan')

    table = Table(box=None, expand=True, show_header=False)
    dashboard.add_row(table)
    cpu_percent = psutil.cpu_percent()
    dram_percent = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu_percent = gpus[0].load * 100 if gpus else 0
    vram_percent = gpus[0].memoryUtil * 100 if gpus else 0
    table.add_column(justify="left", width=30)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=12)
    table.add_column(justify="right", width=12)
    table.add_row(
        f':blowfish: {c1}PufferLib {b2}1.0.0',
        f'{c1}CPU: {c3}{cpu_percent:.1f}%',
        f'{c1}GPU: {c3}{gpu_percent:.1f}%',
        f'{c1}DRAM: {c3}{dram_percent:.1f}%',
        f'{c1}VRAM: {c3}{vram_percent:.1f}%',
    )
        
    s = Table(box=None, expand=True)
    s.add_column(f"{c1}Summary", justify='left', vertical='top', width=16)
    s.add_column(f"{c1}Value", justify='right', vertical='top', width=8)
    s.add_row(f'{c2}Uptime', f'{b2}{duration(uptime)}')
    s.add_row(f'{c2}Estim', f'{b2}{duration(estimated_duration)}')
    s.add_row(f'{c2}Time', f'{b2}{performance.epoch_time:.2f}')
    s.add_row(f'{c2}Epoch', f'{b2}{epoch}')
    s.add_row(f'{c2}Steps/sec', f'{b2}{abbreviate(sps)}')
    s.add_row(f'{c2}Agent Steps', f'{b2}{abbreviate(global_step)}')
  
    p = Table(box=None, expand=True)
    p.add_column(f"{c1}Performance", justify="left", width=16)
    p.add_column(f"{c1}Time", justify="right", width=8)
    p.add_row(f'{c2}Training', f'{b2}{performance.train_time:.3f}')
    p.add_row(f'{c2}Evaluation', f'{b2}{performance.eval_time:.3f}')
    p.add_row(f'{c2}Environment', f'{b2}{performance.env_time:.3f}')
    p.add_row(f'{c2}Forward', f'{b2}{performance.inference_time:.3f}')
    p.add_row(f'{c2}Misc', f'{b2}{performance.misc_time:.3f}')
    p.add_row(f'{c2}Allocation', f'{b2}{performance.alloc_time:.3f}')
    p.add_row(f'{c2}Backward', f'{b2}{performance.backward_time:.3f}')

    l = Table(box=None, expand=True)
    l.add_column(f'{c1}Losses', justify="left", width=16)
    l.add_column(f'{c1}Value', justify="right", width=8)
    for metric, value in losses.items():
        l.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')

    monitor = Table(box=None, expand=True, pad_edge=False)
    monitor.add_row(s, p, l)
    dashboard.add_row(monitor)

    if user:
        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0
        for metric, value in user.items():
            u = left if i % 2 == 0 else right
            u.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')
            i += 1

    console = Console()
    with console.capture() as capture:
        console.print(dashboard)

    print('\033[0;0H' + capture.get())
