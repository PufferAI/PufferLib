from pdb import set_trace as T
import numpy as np

import os
import random
import time
import psutil
import uuid

from collections import defaultdict
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib
import pufferlib.utils
import pufferlib.emulation
import pufferlib.vectorization
import pufferlib.frameworks.cleanrl
import pufferlib.policy_god


@pufferlib.dataclass
class Performance:
    total_uptime = 0
    total_updates = 0
    total_agent_steps = 0
    epoch_time = 0
    epoch_sps = 0
    evaluation_time = 0
    evaluation_sps = 0
    evaluation_memory = 0
    evaluation_pytorch_memory = 0
    env_time = 0
    env_sps = 0
    inference_time = 0
    inference_sps = 0
    train_time = 0
    train_sps = 0
    train_memory = 0
    train_pytorch_memory = 0

@pufferlib.dataclass
class Losses:
    policy_loss = 0
    value_loss = 0
    entropy = 0
    old_approx_kl = 0
    approx_kl = 0
    clipfrac = 0
    explained_variance = 0

@pufferlib.dataclass
class Charts:
    global_step = 0
    SPS = 0
    learning_rate = 0
    episodic_length = 0
    episodic_return = 0

def init(self: object = None, 
        # Agent
        agent: nn.Module = None,
        agent_creator: callable = None,
        agent_kwargs: dict = None,

        # Environment
        env_creator: callable = None,
        env_creator_kwargs: dict = None,
        vectorization: ... = pufferlib.vectorization.Serial,
        num_cores: int = psutil.cpu_count(logical=False),
        num_buffers: int = 1,
        num_envs: int = 8,

        # Experiment
        run_id: str = None,
        track: bool = False,
        verbose: bool = True,
        data_dir: str = 'experiments',
        checkpoint_interval: int = 1,
        total_timesteps: int = 10_000_000,
        batch_size: int = 2**14,
        learning_rate: float = 2.5e-4,

        # Hardware & Reproducibility
        device: str = torch.device("cuda") if torch.cuda.is_available() else "cpu",
        cpu_offload: bool = True,
        torch_deterministic: bool = True,
        seed: int = 1,

        # Policy God
        policy_store: pufferlib.policy_store.PolicyStore = None,
        policy_ranker: pufferlib.policy_ranker.PolicyRanker = None,
        policy_pool: pufferlib.policy_pool.PolicyPool = None,
        policy_selector: pufferlib.policy_ranker.PolicySelector = None,

        # Selfplay
        pool_learner_weight: float = 1.0,
        pool_num_policies: int = 1,
    ):
    start_time = time.time()
    seed_everything(seed, torch_deterministic)
    total_updates = total_timesteps // batch_size
    envs_per_worker = num_envs // num_cores
    obs_device = "cpu" if cpu_offload else device
    assert num_cores * envs_per_worker == num_envs

    wandb = None
    if track:
        import wandb
        assert run_id == wandb.run.id
    elif run_id is None:
        run_id = str(uuid.uuid4())[:8]

    # Create environments, agent, and optimizer
    init_profiler = pufferlib.utils.Profiler(memory=True)
    with init_profiler:
        buffers = [
            vectorization(
                env_creator,
                env_kwargs=env_creator_kwargs,
                num_workers=num_cores,
                envs_per_worker=envs_per_worker,
            )
            for _ in range(num_buffers)
        ]
        num_agents = buffers[0].num_agents
        total_agents = num_agents * num_envs
    agent = pufferlib.emulation.make_object(
        agent, agent_creator, buffers[:1], agent_kwargs)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # If data_dir is provided, load the resume state
    resume_state = {}
    if run_id is not None:
        path = os.path.join(data_dir, run_id)
        if os.path.exists(path):
            trainer_path = os.path.join(path, 'trainer_state.pt')
            resume_state = torch.load(trainer_path)
            model_path = os.path.join(path, resume_state["model_name"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            agent.load_state_dict(torch.load(model_path).state_dict())
            print(f'Resumed from update {resume_state["update"]} '
                  f'with policy {resume_state["policy_checkpoint_name"]}')
    global_step = resume_state.get("global_step", 0)
    agent_step = resume_state.get("agent_step", 0)
    update = resume_state.get("update", 0)

    # Create policy pool
    policy_god = pufferlib.policy_god.PolicyGod(
        buffers[0], agent, os.path.join(data_dir, run_id),
        resume_state, device, num_envs, num_agents,
        pool_learner_weight,pool_num_policies)
    agent = policy_god.agent

    # Allocate Storage
    storage_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True).start()
    next_done, next_lstm_state = [], []
    for i, envs in enumerate(buffers):
        envs.async_reset(seed + i)
        next_done.append(torch.zeros(total_agents).to(device))
        if not agent.is_recurrent:
            next_lstm_state.append(None)
        else:
            shape = (agent.lstm.num_layers, total_agents, agent.lstm.hidden_size)
            next_lstm_state.append((
                torch.zeros(shape).to(device),
                torch.zeros(shape).to(device),
            ))
    obs_shape = buffers[0].single_observation_space.shape
    atn_shape = buffers[0].single_action_space.shape
    obs=torch.zeros(batch_size + 1, *obs_shape).to(obs_device)
    actions=torch.zeros(batch_size + 1, *atn_shape, dtype=int).to(device)
    logprobs=torch.zeros(batch_size + 1).to(device)
    rewards=torch.zeros(batch_size + 1).to(device)
    dones=torch.zeros(batch_size + 1).to(device)
    values=torch.zeros(batch_size + 1).to(device)
    storage_profiler.stop()

    # Original CleanRL charts for comparison
    charts = Charts()
    charts.global_step = global_step
    charts.learning_rate = learning_rate

    losses = Losses()

    #"charts/actions": wandb.Histogram(b_actions.cpu().numpy()),
    init_performance = pufferlib.namespace(
        init_time = time.time() - start_time,
        init_env_time = init_profiler.elapsed,
        init_env_memory = init_profiler.memory,
        tensor_memory = storage_profiler.memory,
        tensor_pytorch_memory = storage_profiler.pytorch_memory,
    )
 
    performance = Performance()

    return pufferlib.namespace(self,
        charts = charts,
        losses = losses,
        init_performance = init_performance,
        performance = performance,

        # Agent, Optimizer, and Environment
        agent = agent,
        optimizer = optimizer,
        buffers = buffers,
        num_buffers = num_buffers,
        buf = 0,

        # Experiment
        run_id = run_id,
        verbose = verbose,
        data_dir = data_dir,
        checkpoint_interval = checkpoint_interval,
        total_updates = total_updates,
        global_step = global_step,
        batch_size = batch_size,
        learning_rate = learning_rate,

        # Storage
        sort_keys = [],
        next_done = next_done,
        next_lstm_state = next_lstm_state,
        obs = obs,
        actions = actions,
        logprobs = logprobs,
        rewards = rewards,
        dones = dones,
        values = values,

        # Selfplay
        policy_god = policy_god,
        pool_learner_weight = pool_learner_weight,
        pool_num_policies = pool_num_policies,

        # Hardware
        device = device,
        obs_device = obs_device,

        # Logging
        track = track,
        wandb = wandb,
        start_time = start_time,
        update = update,
        vectorization = vectorization,
    )

@pufferlib.utils.profile
def evaluate(data):
    # TODO: Handle update on resume
    if data.track and data.performance.total_uptime > 0:
        data.wandb.log({
            **{f'charts/{k}': v for k, v in data.charts.__dict__.items()},
            **{f'losses/{k}': v for k, v in data.losses.__dict__.items()},
            **{f'performance/{k}': v for k, v in data.performance.__dict__.items()},
            **{f'stats/{k}': v for k, v in data.stats.items()},
        })

    data.policy_god.update_policies()
    performance = defaultdict(list)
    env_profiler = pufferlib.utils.Profiler()
    inference_profiler = pufferlib.utils.Profiler()
    eval_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True).start()

    ptr = step = padded_steps_collected = agent_steps_collected = 0
    infos = defaultdict(lambda: defaultdict(list))
    stats = defaultdict(lambda: defaultdict(list))
    while True:
        buf = data.buf
        step += 1
        if ptr == data.batch_size + 1:
            break

        with env_profiler:
            o, r, d, t, i = data.buffers[buf].recv()

        '''
        for profile in data.buffers[buf].profile():
            for k, v in profile.items():
                performance[k].append(v["delta"])
        '''

        i = data.policy_god.update_scores(i, "return")
        o = torch.Tensor(o).to(data.obs_device)
        r = torch.Tensor(r).float().to(data.device).view(-1)
        if len(d) != 0 and len(data.next_done[buf]) != 0:
            alive_mask = (data.next_done[buf].cpu() + torch.Tensor(d)) != 2
            data.next_done[buf] = torch.Tensor(d).to(data.device)
        else:
            alive_mask = [1 for _ in range(len(o))]

        agent_steps_collected += sum(alive_mask)
        padded_steps_collected += len(alive_mask)
        with inference_profiler, torch.no_grad():
            actions, logprob, value, data.next_lstm_state[buf] = data.policy_god.forwards(
                o.to(data.device), data.next_lstm_state[buf], data.next_done[buf])
            value = value.flatten()

        # Index alive mask with policy pool idxs...
        # TODO: Find a way to avoid having to do this
        if data.pool_learner_weight > 0:
          alive_mask = np.array(alive_mask) * data.policy_god.policy_pool.learner_mask

        for idx in np.where(alive_mask)[0]:
            if ptr == data.batch_size + 1:
                break
            data.obs[ptr] = o[idx]
            data.values[ptr] = value[idx]
            data.actions[ptr] = actions[idx]
            data.logprobs[ptr] = logprob[idx]
            data.sort_keys.append((buf, idx, step))
            if len(d) != 0:
                data.rewards[ptr] = r[idx]
                data.dones[ptr] = d[idx]
            ptr += 1

        for policy_name, policy_i in i.items():
            for agent_i in policy_i:
                if not agent_i:
                    continue
                for name, stat in unroll_nested_dict(agent_i):
                    infos[policy_name][name].append(stat)
                    if 'Task_eval_fn' in name:
                        # Temporary hack for NMMO competition
                        continue
                    try:
                        stat = float(stat)
                        stats[policy_name][name].append(stat)
                    except:
                        continue

        with env_profiler:
            data.buffers[buf].send(actions.cpu().numpy())
        data.buf = (data.buf + 1) % data.num_buffers

    data.policy_god.update_ranks(data.global_step)
    data.global_step += data.batch_size
    eval_profiler.stop()

    charts = data.charts
    charts.reward = float(torch.mean(data.rewards))
    charts.agent_steps = data.global_step
    charts.SPS = int(padded_steps_collected / eval_profiler.elapsed)
    charts.global_step = data.global_step

    perf = data.performance
    perf.total_uptime = int(time.time() - data.start_time)
    perf.total_agent_steps = data.global_step
    perf.env_time = env_profiler.elapsed
    perf.env_sps = int(agent_steps_collected / env_profiler.elapsed)
    perf.inference_time = inference_profiler.elapsed
    perf.inference_sps = int(padded_steps_collected / inference_profiler.elapsed)
    perf.eval_time = eval_profiler.elapsed
    perf.eval_sps = int(padded_steps_collected / eval_profiler.elapsed)
    perf.eval_memory = eval_profiler.end_mem
    perf.eval_pytorch_memory = eval_profiler.end_torch_mem
    data.stats = {k: np.mean(v) for k, v in stats['learner'].items()}

    if data.verbose:
        print_dashboard(data.stats, data.init_performance, data.performance)

    return stats, infos

@pufferlib.utils.profile
def train(
    data,
    batch_rows=32,
    update_epochs=4,
    bptt_horizon=16,
    gamma=0.99,
    gae_lambda=0.95,
    anneal_lr=True,
    norm_adv=True,
    clip_coef=0.1,
    clip_vloss=True,
    vf_clip_coef=0.1,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
):
    if done_training(data):
        raise RuntimeError(
            f"Max training updates {data.total_updates} already reached")

    # assert data.num_steps % bptt_horizon == 0, "num_steps must be divisible by bptt_horizon"
    train_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True)
    train_profiler.start()

    if anneal_lr:
        frac = 1.0 - (data.update - 1.0) / data.total_updates
        lrnow = frac * data.learning_rate
        data.optimizer.param_groups[0]["lr"] = lrnow

    num_minibatches = data.batch_size // bptt_horizon // batch_rows
    idxs = sorted(range(len(data.sort_keys)), key=data.sort_keys.__getitem__)
    data.sort_keys = []
    b_idxs = (
        torch.Tensor(idxs)
        .long()[:-1]
        .reshape(batch_rows, num_minibatches, bptt_horizon)
        .transpose(0, 1)
    )

    # bootstrap value if not done
    with torch.no_grad():
        advantages = torch.zeros(data.batch_size, device=data.device)
        lastgaelam = 0
        for t in reversed(range(data.batch_size)):
            i, i_nxt = idxs[t], idxs[t + 1]
            nextnonterminal = 1.0 - data.dones[i_nxt]
            nextvalues = data.values[i_nxt]
            delta = (
                data.rewards[i_nxt]
                + gamma * nextvalues * nextnonterminal
                - data.values[i]
            )
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            )

    # Flatten the batch
    data.b_obs = b_obs = data.obs[b_idxs]
    b_actions = data.actions[b_idxs]
    b_logprobs = data.logprobs[b_idxs]
    b_dones = data.dones[b_idxs]
    b_values = data.values[b_idxs]
    b_advantages = advantages.reshape(
        batch_rows, num_minibatches, bptt_horizon
    ).transpose(0, 1)
    b_returns = b_advantages + b_values

    # Optimizing the policy and value network
    train_time = time.time()
    pg_losses, entropy_losses, v_losses, clipfracs, old_kls, kls = [], [], [], [], [], []
    for epoch in range(update_epochs):
        #shape = (data.agent.lstm.num_layers, batch_rows, data.agent.lstm.hidden_size)
        lstm_state = None
        for mb in range(num_minibatches):
            mb_obs = b_obs[mb].to(data.device)
            mb_actions = b_actions[mb].contiguous()
            mb_values = b_values[mb].reshape(-1)
            mb_advantages = b_advantages[mb].reshape(-1)
            mb_returns = b_returns[mb].reshape(-1)

            if data.agent.is_recurrent:
                _, newlogprob, entropy, newvalue, lstm_state = data.agent.get_action_and_value(
                    mb_obs, state=lstm_state, done=b_dones[mb], action=mb_actions)
                lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
            else:
                _, newlogprob, entropy, newvalue = data.agent.get_action_and_value(
                    mb_obs.reshape(-1, *data.buffers[0].single_observation_space.shape),
                    action=mb_actions,
                )

            logratio = newlogprob - b_logprobs[mb].reshape(-1)
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                old_kls.append(old_approx_kl.item())
                approx_kl = ((ratio - 1) - logratio).mean()
                kls.append(approx_kl.item())
                clipfracs += [
                    ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                ]

            mb_advantages = mb_advantages.reshape(-1)
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - clip_coef, 1 + clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            pg_losses.append(pg_loss.item())

            # Value loss
            newvalue = newvalue.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    newvalue - mb_values,
                    -vf_clip_coef,
                    vf_clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
            v_losses.append(v_loss.item())

            entropy_loss = entropy.mean()
            entropy_losses.append(entropy_loss.item())

            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
            data.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(data.agent.parameters(), max_grad_norm)
            data.optimizer.step()

        if target_kl is not None:
            if approx_kl > target_kl:
                break

    train_profiler.stop()
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    data.update += 1

    charts = data.charts
    charts.learning_rate = data.optimizer.param_groups[0]["lr"]

    losses = data.losses
    losses.policy_loss = np.mean(pg_losses)
    losses.value_loss = np.mean(v_losses)
    losses.entropy = np.mean(entropy_losses)
    losses.old_approx_kl = np.mean(old_kls)
    losses.approx_kl = np.mean(kls)
    losses.clipfrac = np.mean(clipfracs)
    losses.explained_variance = explained_var

    perf = data.performance
    perf.total_uptime = int(time.time() - data.start_time)
    perf.total_updates = data.update
    perf.train_time = time.time() - train_time
    perf.train_sps = int(data.batch_size / perf.train_time)
    perf.train_memory = train_profiler.end_mem
    perf.train_pytorch_memory = train_profiler.end_torch_mem
    perf.epoch_time = perf.eval_time + perf.train_time
    perf.epoch_sps = int(data.batch_size / perf.epoch_time)

    if data.verbose:
        print_dashboard(data.stats, data.init_performance, data.performance)

    if data.update % data.checkpoint_interval == 0 or done_training(data):
       save_checkpoint(data)

def close(data):
    for envs in data.buffers:
        envs.close()

    if data.track:
        artifact_name = f"{data.run_id}_model"
        artifact = data.wandb.Artifact(artifact_name, type="model")
        model_path = save_checkpoint(data)
        artifact.add_file(model_path)
        data.wandb.run.log_artifact(artifact)
        data.wandb.finish()

def done_training(data):
    return data.update >= data.total_updates

def save_checkpoint(data):
    path = os.path.join(data.data_dir, data.run_id)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f'model_{data.update:06d}.pt'
    model_path = os.path.join(path, model_name)

    # Already saved
    if os.path.exists(model_path):
        return model_path

    torch.save(data.agent, model_path)

    state = {
        "optimizer_state_dict": data.optimizer.state_dict(),
        "global_step": data.global_step,
        "agent_step": data.global_step,
        "update": data.update,
        "model_name": model_name,
    }
    state_path = os.path.join(path, 'trainer_state.pt')
    torch.save(state, state_path + '.tmp')
    os.rename(state_path + '.tmp', state_path)

    data.policy_god.add_policy(model_name, data.agent)
    return model_path

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

def print_dashboard(stats, init_performance, performance):
    output = []
    data = {**stats, **init_performance, **performance}
    
    grouped_data = defaultdict(dict)
    
    for k, v in data.items():
        if k == 'total_uptime':
            v = timedelta(seconds=v)
        if 'memory' in k:
            v = pufferlib.utils.format_bytes(v)
        elif 'time' in k:
            try:
                v = f"{v:.2f} s"
            except:
                pass
        
        first_word, *rest_words = k.split('_')
        rest_words = ' '.join(rest_words).title()
        
        grouped_data[first_word][rest_words] = v
    
    for main_key, sub_dict in grouped_data.items():
        output.append(f"{main_key.title()}")
        for sub_key, sub_value in sub_dict.items():
            output.append(f"    {sub_key}: {sub_value}")
    
    print("\033c", end="")
    print('\n'.join(output))

class CleanPuffeRL:
    __init__ = init
    evaluate = evaluate
    train = train
    close = close
    done_training = done_training
