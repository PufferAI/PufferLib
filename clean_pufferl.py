from pdb import set_trace as T
import numpy as np

import os
import random
import time

from collections import defaultdict

import torch

import pufferlib
import pufferlib.utils
import pufferlib.policy_pool


def create(config, vecenv, policy, optimizer=None, wandb=None,
        policy_selector=pufferlib.policy_pool.random_selector):
    seed_everything(config.seed, config.torch_deterministic)
    profile = Profile()
    losses = Losses()

    msg = f'Model Size: {abbreviate(count_params(policy))} parameters'
    print_dashboard(0, 0, profile, losses, {}, msg, clear=True)

    vecenv.async_reset(config.seed)
    obs_shape = vecenv.single_observation_space.shape
    atn_shape = vecenv.single_action_space.shape
    total_agents = vecenv.num_envs * vecenv.agents_per_env

    lstm = policy.lstm if hasattr(policy, 'lstm') else None
    experience = Experience(config.batch_size, config.bptt_horizon,
        config.batch_rows, obs_shape, atn_shape, config.device, lstm, total_agents)

    uncompiled_policy = policy
    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode)

    optimizer = torch.optim.Adam(policy.parameters(),
        lr=config.learning_rate, eps=1e-5)

    # Wraps the policy for self-play. Disregard for single-agent
    policy = pufferlib.policy_pool.PolicyPool(
        policy=policy,
        total_agents=vecenv.agents_per_batch,
        atn_shape=vecenv.single_action_space.shape,
        device=config.device,
        data_dir=config.data_dir,
        kernel=config.pool_kernel,
        policy_selector=policy_selector,
    )

    return pufferlib.namespace(
        config=config,
        vecenv=vecenv,
        policy=policy,
        uncompiled_policy=uncompiled_policy,
        optimizer=optimizer,
        experience=experience,
        profile=profile,
        losses=losses,
        wandb=wandb,
        global_step=0,
        epoch=0,
        stats={},
        msg=msg,
    )

@pufferlib.utils.profile
def evaluate(data):
    config, profile, experience = data.config, data.profile, data.experience

    with profile.eval_misc:
        policy = data.policy
        policy.update_policies()
        agent_steps = 0
        #infos = defaultdict(lambda: defaultdict(list))
        infos = defaultdict(list)
        lstm_h, lstm_c = experience.lstm_h, experience.lstm_c

    while not experience.full:
        with profile.env:
            o, r, d, t, i, env_id, mask = data.vecenv.recv()
            env_id = env_id.tolist()

        with profile.eval_misc:
            data.global_step += sum(mask)

            o = torch.as_tensor(o)
            r = torch.as_tensor(r)
            d = torch.as_tensor(d)
            #i = data.policy.update_scores(i, "return")

            # TODO: Update this for policy pool
            #for ii, ee  in zip(i['learner'], env_id):
            #    ii['env_id'] = ee

        with profile.eval_forward, torch.no_grad():
            # TODO: In place-update should be faster
            h = lstm_h[:, env_id] if lstm_h is not None else None
            c = lstm_c[:, env_id] if lstm_c is not None else None
            actions, logprob, value, (h, c) = policy.forwards(o.to(config.device), h, c)
            if lstm_h is not None:
                lstm_h[:, env_id] = h
                lstm_c[:, env_id] = c

        with profile.eval_misc:
            value = value.flatten()
            actions = actions.cpu().numpy()
            mask = torch.as_tensor(mask * policy.mask)
            experience.store(o, value, actions, logprob, r, d, env_id, mask)

            # Really neeed to look at policy pool soon
            for agent_info in i:
                for k, v in agent_info.items():
                    infos[k].append(v)

            '''
            for policy_name, policy_i in i.items():
                for agent_i in policy_i:
                    for name, dat in unroll_nested_dict(agent_i):
                        infos[policy_name][name].append(dat)
            '''

        with profile.env:
            data.vecenv.send(actions)

    with profile.eval_misc:
        data.stats = {}
        #infos = infos['learner']

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

        for k, v in infos.items():
            try: # TODO: Better checks on log data types
                data.stats[k] = np.mean(v)
            except:
                continue


    return data.stats, infos

@pufferlib.utils.profile
def train(data):
    config, profile, experience = data.config, data.profile, data.experience
    losses = data.losses

    with profile.train_misc:
        # TODO: Not a very good bootstrap implementation. Doesn't handle
        # bounds between segments
        # bootstrap value if not done
        idxs = experience.sort_training_data()
        dones_np = experience.dones_np
        values_np = experience.values_np
        rewards_np = experience.rewards_np
        advantages = np.zeros(config.batch_size)
        lastgaelam = 0
        for t in reversed(range(config.batch_size-1)):
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

        advantages = torch.from_numpy(advantages).to(config.device)
        experience.flatten_batch(advantages)

    # Optimizing the policy and value network
    pg_losses, entropy_losses, v_losses = [], [], []
    clipfracs, old_kls, kls = [], [], []
    for epoch in range(config.update_epochs):
        lstm_state = None
        for obs, atn, log_probs, val, adv, ret in experience.minibatches():
            with profile.train_forward:
                if experience.lstm_h is not None:
                    _, newlogprob, entropy, newvalue, lstm_state = data.policy.learner_policy(
                        obs, state=lstm_state, action=atn)
                    lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                else:
                    _, newlogprob, entropy, newvalue = data.policy.learner_policy(
                        obs.reshape(-1, *data.vecenv.single_observation_space.shape),
                        action=atn,
                    )

            with profile.train_misc:
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

                adv = adv.reshape(-1)
                if config.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(
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

            with profile.learn:
                data.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(data.policy.learner_policy.parameters(), config.max_grad_norm)
                data.optimizer.step()

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    with profile.train_misc:
        if config.anneal_lr:
            frac = 1.0 - data.global_step / config.total_timesteps
            lrnow = frac * config.learning_rate
            data.optimizer.param_groups[0]["lr"] = lrnow

        y_pred, y_true = experience.values.cpu().numpy(), experience.b_returns.cpu().reshape(-1).numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        losses.update(pg_losses, v_losses, entropy_losses,
            old_kls, kls, clipfracs, explained_var)
        data.epoch += 1

        done_training = data.global_step >= config.total_timesteps
        if data.epoch % config.checkpoint_interval == 0 or done_training:
            save_checkpoint(data)
            data.msg = f'Checkpoint saved at update {data.epoch}'

        if done_training:
            close(data)

        if profile.update(data) or done_training:
            print_dashboard(data.global_step, data.epoch,
                profile, data.losses, data.stats, data.msg)

            if data.wandb is not None and data.global_step > 0:
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

def close(data):
    data.vecenv.close()
    if data.wandb is not None:
        artifact_name = f"{data.exp_name}_model"
        artifact = data.wandb.Artifact(artifact_name, type="model")
        model_path = save_checkpoint(data)
        artifact.add_file(model_path)
        data.wandb.run.log_artifact(artifact)
        data.wandb.finish()

class Profile:
    SPS: ... = 0
    uptime: ... = 0
    remaining: ... = 0
    eval_time: ... = 0
    env_time: ... = 0
    eval_forward_time: ... = 0
    eval_misc_time: ... = 0
    train_time: ... = 0
    train_forward_time: ... = 0
    learn_time: ... = 0
    train_misc_time: ... = 0
    def __init__(self):
        self.start = time.time()
        self.env = pufferlib.utils.Profiler()
        self.eval_forward = pufferlib.utils.Profiler()
        self.eval_misc = pufferlib.utils.Profiler()
        self.train_forward = pufferlib.utils.Profiler()
        self.learn = pufferlib.utils.Profiler()
        self.train_misc = pufferlib.utils.Profiler()
        self.prev_steps = 0

    @property
    def epoch_time(self):
        return self.train_time + self.eval_time

    def update(self, data, interval_s=1):
        global_step = data.global_step
        if global_step == 0:
            return True

        uptime = time.time() - self.start
        if uptime - self.uptime < interval_s:
            return False

        self.SPS = (global_step - self.prev_steps) / (uptime - self.uptime)
        self.prev_steps = global_step
        self.uptime = uptime

        self.remaining = (data.config.total_timesteps - global_step) / self.SPS
        self.eval_time = data._timers['evaluate'].elapsed
        self.eval_forward_time = self.eval_forward.elapsed
        self.env_time = self.env.elapsed
        self.eval_misc_time = self.eval_misc.elapsed
        self.train_time = data._timers['train'].elapsed
        self.train_forward_time = self.train_forward.elapsed
        self.learn_time = self.learn.elapsed
        self.train_misc_time = self.train_misc.elapsed
        return True

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
    '''Flat tensor storage and array views for faster indexing'''
    def __init__(self, batch_size, bptt_horizon, batch_rows, obs_shape, atn_shape,
                 device='cuda', lstm=None, lstm_total_agents=0):
        self.obs=torch.zeros(batch_size, *obs_shape)
        self.actions=torch.zeros(batch_size, *atn_shape, dtype=int)
        self.logprobs=torch.zeros(batch_size)
        self.rewards=torch.zeros(batch_size)
        self.dones=torch.zeros(batch_size)
        self.truncateds=torch.zeros(batch_size)
        self.values=torch.zeros(batch_size)

        self.obs_np = np.asarray(self.obs)
        self.actions_np = np.asarray(self.actions)
        self.logprobs_np = np.asarray(self.logprobs)
        self.rewards_np = np.asarray(self.rewards)
        self.dones_np = np.asarray(self.dones)
        self.truncateds_np = np.asarray(self.truncateds)
        self.values_np = np.asarray(self.values)

        self.mb_obs_buffer = torch.zeros(batch_rows, bptt_horizon,
            *obs_shape, pin_memory=(device=="cuda"))

        self.lstm_h = self.lstm_c = None
        if lstm is not None:
            assert lstm_total_agents > 0
            shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
            self.lstm_h = torch.zeros(shape).to(device)
            self.lstm_c = torch.zeros(shape).to(device)

        self.num_minibatches = batch_size // bptt_horizon // batch_rows
        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.batch_rows = batch_rows
        self.device = device
        self.sort_keys = []
        self.ptr = 0
        self.step = 0

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(self, obs, value, action, logprob, reward, done, env_id, mask):
        # Mask learner and Ensure indices do not exceed batch size
        ptr = self.ptr
        indices = torch.where(mask)[0].numpy()[:self.batch_size - ptr]
        end = ptr + len(indices)
 
        self.obs_np[ptr:end] = obs.cpu().numpy()[indices]
        self.values_np[ptr:end] = value.cpu().numpy()[indices]
        self.actions_np[ptr:end] = action[indices]
        self.logprobs_np[ptr:end] = logprob.cpu().numpy()[indices]
        self.rewards_np[ptr:end] = reward.cpu().numpy()[indices]
        self.dones_np[ptr:end] = done.cpu().numpy()[indices]
        self.sort_keys.extend([(env_id[i], self.step) for i in indices])
        self.ptr = end
        self.step += 1

    def sort_training_data(self):
        idxs = sorted(range(len(self.sort_keys)), key=self.sort_keys.__getitem__)
        self.b_idxs = torch.Tensor(idxs).long().reshape(
            self.batch_rows, self.num_minibatches, self.bptt_horizon).transpose(0, 1)
        self.sort_keys = []
        self.ptr = 0
        self.step = 0
        return idxs

    def flatten_batch(self, advantages):
        b_idxs = self.b_idxs
        self.b_obs = torch.as_tensor(self.obs_np[b_idxs])
        self.b_actions = torch.as_tensor(self.actions_np[b_idxs]
            ).to(self.device, non_blocking=True)
        self.b_logprobs = torch.as_tensor(self.logprobs_np[b_idxs]
            ).to(self.device, non_blocking=True)
        self.b_dones = torch.as_tensor(self.dones_np[b_idxs]
            ).to(self.device, non_blocking=True)
        self.b_values = torch.as_tensor(self.values_np[b_idxs]
            ).to(self.device, non_blocking=True)
        self.b_advantages = advantages.reshape(self.batch_rows,
            self.num_minibatches, self.bptt_horizon).transpose(0, 1)
        self.b_returns = self.b_advantages + self.b_values

    def minibatches(self):
        for mb in range(self.num_minibatches):
            self.mb_obs_buffer.copy_(self.b_obs[mb], non_blocking=True)
            mb_obs = self.mb_obs_buffer.to(self.device, non_blocking=True)
            mb_actions = self.b_actions[mb].contiguous()
            mb_logprobs = self.b_logprobs[mb].reshape(-1)
            mb_values = self.b_values[mb].reshape(-1)
            mb_advantages = self.b_advantages[mb].reshape(-1)
            mb_returns = self.b_returns[mb].reshape(-1)
            yield mb_obs, mb_actions, mb_logprobs, mb_values, mb_advantages, mb_returns

def save_checkpoint(data):
    config = data.config
    path = os.path.join(config.data_dir, config.exp_name)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f'model_{data.epoch:06d}.pt'
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        return model_path

    torch.save(data.uncompiled_policy, model_path)

    state = {
        'optimizer_state_dict': data.optimizer.state_dict(),
        'global_step': data.global_step,
        'agent_step': data.global_step,
        'update': data.epoch,
        'model_name': model_name,
        'exp_name': config.exp_name,
    }
    state_path = os.path.join(path, 'trainer_state.pt')
    torch.save(state, state_path + '.tmp')
    os.rename(state_path + '.tmp', state_path)
    return model_path

def try_load_checkpoint(data):
    config = data.config
    path = os.path.join(config.data_dir, config.exp_name)
    if not os.path.exists(path):
        print('No checkpoints found. Assuming new experiment')
        return

    trainer_path = os.path.join(path, 'trainer_state.pt')
    resume_state = torch.load(trainer_path)
    model_path = os.path.join(path, resume_state['model_name'])
    data.policy.uncompiled.load_state_dict(model_path, map_location=config.device)
    data.optimizer.load_state_dict(resume_state['optimizer_state_dict'])
    print(f'Loaded checkpoint {resume_state["model_name"]}')

def count_params(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)

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
        return f'{b2}{num:.0f}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}k'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}m'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}b'
    else:
        return f'{b2}{num/1e12:.1f}{c2}t'

def duration(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"


def fmt_perf(name, time, uptime):
    percent = 0 if uptime == 0 else int(100*time/uptime - 1e-5)
    return f'{c1}{name}', duration(time), f'{b2}{percent:2d}%'

def print_dashboard(global_step, epoch, profile, losses, stats, msg, clear=False):
    console = Console()
    if clear:
        console.clear()

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
    s.add_row(f'{c2}Agent Steps', abbreviate(global_step))
    s.add_row(f'{c2}SPS', abbreviate(profile.SPS))
    s.add_row(f'{c2}Epoch', abbreviate(epoch))
    s.add_row(f'{c2}Uptime', duration(profile.uptime))
    s.add_row(f'{c2}Remaining', duration(profile.remaining))

    p = Table(box=None, expand=True, show_header=False)
    p.add_column(f"{c1}Performance", justify="left", width=10)
    p.add_column(f"{c1}Time", justify="right", width=8)
    p.add_column(f"{c1}%", justify="right", width=4)
    p.add_row(*fmt_perf('Evaluate', profile.eval_time, profile.uptime))
    p.add_row(*fmt_perf('  Forward', profile.eval_forward_time, profile.uptime))
    p.add_row(*fmt_perf('  Env', profile.env_time, profile.uptime))
    p.add_row(*fmt_perf('  Misc', profile.eval_misc_time, profile.uptime))
    p.add_row(*fmt_perf('Train', profile.train_time, profile.uptime))
    p.add_row(*fmt_perf('  Forward', profile.train_forward_time, profile.uptime))
    p.add_row(*fmt_perf('  Learn', profile.learn_time, profile.uptime))
    p.add_row(*fmt_perf('  Misc', profile.train_misc_time, profile.uptime))

    l = Table(box=None, expand=True, )
    l.add_column(f'{c1}Losses', justify="left", width=16)
    l.add_column(f'{c1}Value', justify="right", width=8)
    for metric, value in losses.items():
        l.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')

    monitor = Table(box=None, expand=True, pad_edge=False)
    monitor.add_row(s, p, l)
    dashboard.add_row(monitor)

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
    for metric, value in stats.items():
        u = left if i % 2 == 0 else right
        u.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')
        i += 1


    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    table.add_row(f' {c1}Message: {c2}{msg}')

    with console.capture() as capture:
        console.print(dashboard)

    print('\033[0;0H' + capture.get())
