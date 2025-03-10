from pdb import set_trace as T
import numpy as np

import os
import random
import psutil
import time

from threading import Thread
from collections import defaultdict, deque

import rich
from rich.console import Console
from rich.table import Table

import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch
import pufferlib.cleanrl

torch.set_float32_matmul_precision('high')

# Fast Cython GAE implementation
#import pyximport
#pyximport.install(setup_args={"include_dirs": np.get_include()})
from c_gae import compute_gae


def create(config, vecenv, policy, optimizer=None, wandb=None):
    seed_everything(config.seed, config.torch_deterministic)
    profile = Profile()
    losses = make_losses()

    utilization = Utilization()
    msg = f'Model Size: {abbreviate(count_params(policy))} parameters'
    print_dashboard(config.env, utilization, 0, 0, profile, losses, {}, msg, clear=True)

    vecenv.async_reset(config.seed)
    obs_shape = vecenv.single_observation_space.shape
    obs_dtype = vecenv.single_observation_space.dtype
    atn_shape = vecenv.single_action_space.shape
    atn_dtype = vecenv.single_action_space.dtype
    total_agents = vecenv.num_agents

    use_amp_obs = getattr(vecenv, "use_amp_obs", False)

    lstm = policy.lstm if hasattr(policy, 'lstm') else None
    experience = Experience(config.batch_size, config.bptt_horizon,
        config.minibatch_size, obs_shape, obs_dtype, atn_shape, atn_dtype,
        config.cpu_offload, config.device, lstm, total_agents,
        use_amp_obs)

    uncompiled_policy = policy

    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode)

    optimizer = torch.optim.Adam(policy.parameters(),
        lr=config.learning_rate, eps=1e-5)

    # Store initial policy weights for regenerative regularization
    # https://arxiv.org/pdf/2308.11958
    initial_params = {}
    for name, param in policy.named_parameters():
        initial_params[name] = param.detach().clone()

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
        stats=defaultdict(list),
        msg=msg,
        last_log_time=0,
        utilization=utilization,
        use_amp_obs=use_amp_obs,
        initial_params=initial_params,
    )

@pufferlib.utils.profile
def evaluate(data):
    config, profile, experience = data.config, data.profile, data.experience

    with profile.eval_misc:
        policy = data.policy
        infos = defaultdict(list)
        lstm_h, lstm_c = experience.lstm_h, experience.lstm_c

    while not experience.full:
        with profile.env:
            o, r, d, t, info, env_id, mask = data.vecenv.recv()
            env_id = env_id.tolist()

        with profile.eval_misc:
            if isinstance(mask, torch.Tensor):
                data.global_step += mask.sum().item()
            else:
                data.global_step += sum(mask)

            o = torch.as_tensor(o)
            o_device = o.to(config.device)
            r = torch.as_tensor(r)
            d = torch.as_tensor(d)
            t = torch.as_tensor(t)

        with profile.eval_forward, torch.no_grad():
            # TODO: In place-update should be faster. Leaking 7% speed max
            # Also should be using a cuda tensor to index
            if lstm_h is not None:
                # Reset the hidden states for the done/truncated envs
                reset_envs = torch.logical_or(d, t)
                if reset_envs.any():
                    lstm_h[:, reset_envs] = 0
                    lstm_c[:, reset_envs] = 0

                h = lstm_h[:, env_id]
                c = lstm_c[:, env_id]
                actions, logprob, _, value, (h, c) = policy(o_device, (h, c))
                lstm_h[:, env_id] = h
                lstm_c[:, env_id] = c
            else:
                actions, logprob, _, value = policy(o_device)

            if config.device == 'cuda':
                torch.cuda.synchronize()

        with profile.eval_misc:
            value = value.flatten()
            actions = actions.cpu().numpy()
            mask = torch.as_tensor(mask)# * policy.mask)
            o = o if config.cpu_offload else o_device

            amp_obs = data.vecenv.amp_obs if data.use_amp_obs else None
            experience.store(o, amp_obs, value, actions, logprob, r, d, t, env_id, mask)

            for i in info:
                for k, v in pufferlib.utils.unroll_nested_dict(i):
                    infos[k].append(v)

        with profile.env:
            data.vecenv.send(actions)

    with profile.eval_misc:
        for k, v in infos.items():
            if '_map' in k and data.wandb is not None:
                data.stats[f'Media/{k}'] = data.wandb.Image(v[0])
                continue

            if isinstance(v, np.ndarray):
                v = v.tolist()
            try:
                iter(v)
            except TypeError:
                data.stats[k].append(v)
            else:
                data.stats[k] += v

    # TODO: Better way to enable multiple collects
    data.experience.ptr = 0
    data.experience.step = 0
    return data.stats, infos

@pufferlib.utils.profile
def train(data):
    config, profile, experience = data.config, data.profile, data.experience
    data.losses = make_losses()
    losses = data.losses

    with profile.train_misc:
        idxs = experience.sort_training_data()
        dones_np = experience.dones_np[idxs]
        trunc_np = experience.truncateds_np[idxs]
        values_np = experience.values_np[idxs]
        rewards_np = experience.rewards_np[idxs]
        experience.flatten_batch()

        if data.use_amp_obs:
            amp_obs_demo = data.vecenv.fetch_amp_obs_demo()  # [num_envs, amp_obs_size]
            amp_minibatch_size = amp_obs_demo.shape[0]

        # Mean bound loss attribute
        mean_bound_loss = getattr(data.policy.policy, "mean_bound_loss", None)

    # Compute adversarial reward. Note: discriminator doesn't get
    # updated as often this way, but GAE is more accurate
    adversarial_reward = torch.zeros(
        experience.num_minibatches, config.minibatch_size).to(config.device)

    discriminate = getattr(data.policy.policy, "discriminate", None)
    if data.use_amp_obs and discriminate is not None:
        with torch.no_grad():
            for mb in range(experience.num_minibatches):
                disc_logits = discriminate(experience.b_amp_obs[mb]).squeeze()
                prob = 1 / (1 + torch.exp(-disc_logits))
                adversarial_reward[mb] = -torch.log(torch.maximum(
                    1 - prob, torch.tensor(0.0001, device=config.device)))

    # TODO: Nans in adversarial reward and gae
    adversarial_reward_np = adversarial_reward.cpu().numpy().ravel()

    # For motion imitation, done is True ONLY when the env is terminated early.
    # Successful replay of motions will get done=False, truncation=True
    # Since gae is using only dones, the advantages for truncated steps are 
    # computed as the same as the nonterminal steps.
    # NOTE: The imitation reward and adversarial reward are equally weighted.
    advantages_np = compute_gae(dones_np, values_np,
        rewards_np + adversarial_reward_np, config.gamma, config.gae_lambda)

    advantages = torch.as_tensor(advantages_np).to(config.device)
    experience.b_advantages = advantages.reshape(experience.minibatch_rows,
        experience.num_minibatches, experience.bptt_horizon).transpose(0, 1).reshape(
        experience.num_minibatches, experience.minibatch_size)
    experience.returns_np = advantages_np + experience.values_np
    experience.b_returns = experience.b_advantages + experience.b_values

    # Optimizing the policy and value network
    total_minibatches = experience.num_minibatches * config.update_epochs
    mean_pg_loss, mean_v_loss, mean_entropy_loss = 0, 0, 0
    mean_old_kl, mean_kl, mean_clipfrac = 0, 0, 0
    for epoch in range(config.update_epochs):
        lstm_state = None
        for mb in range(experience.num_minibatches):
            with profile.train_misc:
                obs = experience.b_obs[mb].to(config.device)
                atn = experience.b_actions[mb]
                log_probs = experience.b_logprobs[mb]
                val = experience.b_values[mb]
                adv = experience.b_advantages[mb]
                ret = experience.b_returns[mb]
                
                if data.use_amp_obs:
                    amp_obs_agent = torch.cat([
                        experience.b_amp_obs[mb][:amp_minibatch_size],
                        experience.b_amp_obs_replay[mb][:amp_minibatch_size],
                    ])

            with profile.train_forward:
                if experience.lstm_h is not None:
                    _, newlogprob, entropy, newvalue, lstm_state = data.policy(
                        obs, state=lstm_state, action=atn)
                    lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                else:
                    _, newlogprob, entropy, newvalue = data.policy(
                        obs.reshape(-1, *data.vecenv.single_observation_space.shape),
                        action=atn,
                    )

                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.train_misc:
                logratio = newlogprob - log_probs.reshape(-1)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

                adv = adv.reshape(-1)
                if config.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = ((newvalue - ret) ** 2).mean()

                # Discriminator loss
                disc_loss = 0.0
                if data.use_amp_obs:
                    disc_agent_logits = discriminate(amp_obs_agent)
                    disc_demo_logits = discriminate(amp_obs_demo)
                    disc_loss_agent = torch.nn.BCEWithLogitsLoss()(disc_agent_logits, torch.zeros_like(disc_agent_logits))
                    disc_loss_demo = torch.nn.BCEWithLogitsLoss()(disc_demo_logits, torch.ones_like(disc_demo_logits))
                    disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef + disc_loss * config.disc_coef

                if mean_bound_loss is not None:
                    loss += mean_bound_loss * 10.0 # hard coded for now

                # Regenerative regularization, https://arxiv.org/pdf/2308.11958
                l2_init_reg_loss = 0
                for name, param in data.policy.named_parameters():
                    if name in data.initial_params:
                        l2_init_reg_loss += (param - data.initial_params[name]).pow(2).mean()
                
                loss += l2_init_reg_loss * 0.001  # hard coded for now, 1e-3 in paper

            with profile.learn:
                data.optimizer.zero_grad()
                loss.backward()

                before_clip_grad_norm = 0
                for p in data.policy.parameters():
                    if p.grad is not None:
                        before_clip_grad_norm += p.grad.norm().item()

                torch.nn.utils.clip_grad_norm_(data.policy.parameters(), config.max_grad_norm)

                # after_clip_grad_norm = 0
                # for p in data.policy.parameters():
                #     if p.grad is not None:
                #         after_clip_grad_norm += p.grad.norm().item()

                data.optimizer.step()
                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.train_misc:
                losses.policy_loss += pg_loss.item() / total_minibatches
                losses.value_loss += v_loss.item() / total_minibatches
                losses.entropy += entropy_loss.item() / total_minibatches
                losses.old_approx_kl += old_approx_kl.item() / total_minibatches
                losses.approx_kl += approx_kl.item() / total_minibatches
                losses.clipfrac += clipfrac.item() / total_minibatches
                losses.before_clip_grad_norm += before_clip_grad_norm / total_minibatches
                # losses.after_clip_grad_norm += after_clip_grad_norm / total_minibatches
                losses.l2_init_reg_loss += l2_init_reg_loss.item() / total_minibatches

                if data.use_amp_obs:
                    losses.disc_loss += disc_loss.item() / total_minibatches
                    losses.disc_agent_acc += (disc_agent_logits < 0).float().mean() / total_minibatches
                    losses.disc_demo_acc += (disc_demo_logits > 0).float().mean() / total_minibatches

                if mean_bound_loss:
                    losses.mean_bound_loss += mean_bound_loss.item() / total_minibatches

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    with profile.train_misc:
        if config.anneal_lr:
            frac = 1.0 - data.global_step / config.total_timesteps
            lrnow = frac * config.learning_rate
            data.optimizer.param_groups[0]["lr"] = lrnow

        y_pred = experience.values_np
        y_true = experience.returns_np
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        losses.explained_variance = explained_var
        data.epoch += 1

        done_training = data.global_step >= config.total_timesteps
        # TODO: beter way to get episode return update without clogging dashboard
        # TODO: make this appear faster
        if done_training or profile.update(data):
            mean_and_log(data)
            print_dashboard(config.env, data.utilization, data.global_step, data.epoch,
                profile, data.losses, data.stats, data.msg)
            data.stats = defaultdict(list)

        if data.epoch % config.checkpoint_interval == 0 or done_training:
            save_checkpoint(data)
            data.msg = f'Checkpoint saved at update {data.epoch}'

def mean_and_log(data):
    for k in list(data.stats.keys()):
        v = data.stats[k]
        try:
            v = np.mean(v)
        except:
            del data.stats[k]

        data.stats[k] = v

    if data.wandb is None:
        return

    data.last_log_time = time.time()
    data.wandb.log({
        '0verview/SPS': data.profile.SPS,
        '0verview/agent_steps': data.global_step,
        '0verview/epoch': data.epoch,
        '0verview/learning_rate': data.optimizer.param_groups[0]["lr"],
        **{f'environment/{k}': v for k, v in data.stats.items()},
        **{f'losses/{k}': v for k, v in data.losses.items()},
        **{f'performance/{k}': v for k, v in data.profile},
    })

def close(data):
    data.vecenv.close()
    data.utilization.stop()
    config = data.config
    if data.wandb is not None:
        artifact_name = f"{config.exp_id}_model"
        artifact = data.wandb.Artifact(artifact_name, type="model")
        model_path = save_checkpoint(data)
        artifact.add_file(model_path)
        # NOTE: PHC model is large to save for all sweep runs
        # data.wandb.run.log_artifact(artifact)
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

    def __iter__(self):
        yield 'SPS', self.SPS
        yield 'uptime', self.uptime
        yield 'remaining', self.remaining
        yield 'eval_time', self.eval_time
        yield 'env_time', self.env_time
        yield 'eval_forward_time', self.eval_forward_time
        yield 'eval_misc_time', self.eval_misc_time
        yield 'train_time', self.train_time
        yield 'train_forward_time', self.train_forward_time
        yield 'learn_time', self.learn_time
        yield 'train_misc_time', self.train_misc_time

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

def make_losses():
    return pufferlib.namespace(
        policy_loss=0,
        value_loss=0,
        disc_loss=0,
        disc_agent_acc=0,
        disc_demo_acc=0,
        entropy=0,
        old_approx_kl=0,
        approx_kl=0,
        clipfrac=0,
        explained_variance=0,
        mean_bound_loss=0,
        before_clip_grad_norm=0,
        # after_clip_grad_norm=0,
        l2_init_reg_loss=0,
    )

class Experience:
    '''Flat tensor storage and array views for faster indexing'''
    def __init__(self, batch_size, bptt_horizon, minibatch_size, obs_shape, obs_dtype, atn_shape, atn_dtype,
                 cpu_offload=False, device='cuda', lstm=None, lstm_total_agents=0,
                 use_amp_obs=False, amp_obs_size=1960, amp_obs_update_prob=0.01):
        if minibatch_size is None:
            minibatch_size = batch_size

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        atn_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_dtype]
        pin = device == 'cuda' and cpu_offload
        obs_device = device if not pin else 'cpu'
        self.obs=torch.zeros(batch_size, *obs_shape, dtype=obs_dtype,
            pin_memory=pin, device=device if not pin else 'cpu')

        self.actions=torch.zeros(batch_size, *atn_shape, dtype=atn_dtype, pin_memory=pin)
        self.logprobs=torch.zeros(batch_size, pin_memory=pin)
        self.rewards=torch.zeros(batch_size, pin_memory=pin)
        self.dones=torch.zeros(batch_size, pin_memory=pin)
        self.truncateds=torch.zeros(batch_size, pin_memory=pin)
        self.values=torch.zeros(batch_size, pin_memory=pin)

        #self.obs_np = np.asarray(self.obs)
        self.actions_np = np.asarray(self.actions)
        self.logprobs_np = np.asarray(self.logprobs)
        self.rewards_np = np.asarray(self.rewards)
        self.dones_np = np.asarray(self.dones)
        self.truncateds_np = np.asarray(self.truncateds)
        self.values_np = np.asarray(self.values)

        self.lstm_h = self.lstm_c = None
        if lstm is not None:
            assert lstm_total_agents > 0
            shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
            self.lstm_h = torch.zeros(shape).to(device)
            self.lstm_c = torch.zeros(shape).to(device)

        num_minibatches = batch_size / minibatch_size
        self.num_minibatches = int(num_minibatches)
        if self.num_minibatches != num_minibatches:
            raise ValueError('batch_size must be divisible by minibatch_size')

        minibatch_rows = minibatch_size / bptt_horizon
        self.minibatch_rows = int(minibatch_rows)
        if self.minibatch_rows != minibatch_rows:
            raise ValueError('minibatch_size must be divisible by bptt_horizon')

        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.minibatch_size = minibatch_size
        self.device = device
        self.sort_keys = []
        self.ptr = 0
        self.step = 0

        self.use_amp_obs = use_amp_obs
        if self.use_amp_obs:
            self.amp_obs=torch.zeros(batch_size, amp_obs_size, dtype=obs_dtype,
                pin_memory=pin, device=device if not pin else 'cpu')
            self.amp_obs_replay=torch.zeros(batch_size, amp_obs_size, dtype=obs_dtype,
                pin_memory=pin, device=device if not pin else 'cpu')
            # self.demo=torch.zeros(batch_size, 358, dtype=obs_dtype,
            #     pin_memory=pin, device=device if not pin else 'cpu')
            # self.state=torch.zeros(batch_size, 358, dtype=obs_dtype,
            #     pin_memory=pin, device=device if not pin else 'cpu')

            self.amp_obs_replay_filled = False
            self.amp_obs_update_prob = amp_obs_update_prob

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(self, obs, amp_obs, value, action, logprob, reward, done, trunc, env_id, mask):
        # Mask learner and Ensure indices do not exceed batch size
        ptr = self.ptr
        indices = torch.where(mask)[0].numpy()[:self.batch_size - ptr]
        end = ptr + len(indices)
 
        self.obs[ptr:end] = obs.to(self.obs.device)[indices]
        if self.use_amp_obs:
            self.amp_obs[ptr:end] = amp_obs.to(self.amp_obs.device)[indices]

        self.values_np[ptr:end] = value.cpu().numpy()[indices]
        self.actions_np[ptr:end] = action[indices]
        self.logprobs_np[ptr:end] = logprob.cpu().numpy()[indices]
        self.rewards_np[ptr:end] = reward.cpu().numpy()[indices]
        self.dones_np[ptr:end] = done.cpu().numpy()[indices]
        self.truncateds_np[ptr:end] = trunc.cpu().numpy()[indices]
        self.sort_keys.extend([(env_id[i], self.step) for i in indices])
        self.ptr = end
        self.step += 1

    def sort_training_data(self):
        idxs = np.asarray(sorted(
            range(len(self.sort_keys)), key=self.sort_keys.__getitem__))
        self.b_idxs_obs = torch.as_tensor(idxs.reshape(
                self.minibatch_rows, self.num_minibatches, self.bptt_horizon
            ).transpose(1,0,-1)).to(self.obs.device).long()
        self.b_idxs = self.b_idxs_obs.to(self.device)
        self.b_idxs_flat = self.b_idxs.reshape(
            self.num_minibatches, self.minibatch_size)
        self.sort_keys = []
        return idxs

    def flatten_batch(self):
        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat
        self.b_actions = self.actions.to(self.device, non_blocking=True)
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)
        self.b_dones = self.dones.to(self.device, non_blocking=True)
        self.b_truncated = self.truncateds.to(self.device, non_blocking=True)
        self.b_values = self.values.to(self.device, non_blocking=True)
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.b_actions[b_idxs].contiguous()
        self.b_logprobs = self.b_logprobs[b_idxs]
        self.b_dones = self.b_dones[b_idxs]
        self.b_truncated = self.b_truncated[b_idxs]
        self.b_values = self.b_values[b_flat]

        # AMP, only used for discriminator training
        if self.use_amp_obs:
            self.b_amp_obs = self.amp_obs[b_flat]

            # Update the amp obs replay
            if not self.amp_obs_replay_filled:
                self.amp_obs_replay[:] = self.amp_obs[:]
                self.amp_obs_replay_filled = True
            else:
                # Only update the fraction of the replay buffer
                update_idx = torch.rand(self.batch_size) < self.amp_obs_update_prob
                self.amp_obs_replay[update_idx] = self.amp_obs[update_idx]

            # For the replay, the order does not matter
            rep_idx = torch.randperm(self.batch_size).reshape(
                self.num_minibatches, self.minibatch_size)
            self.b_amp_obs_replay = self.amp_obs_replay[rep_idx]

class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque(maxlen=maxlen)
        self.cpu_util = deque(maxlen=maxlen)
        self.gpu_util = deque(maxlen=maxlen)
        self.gpu_mem = deque(maxlen=maxlen)

        self.delay = delay
        self.stopped = False
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100*psutil.cpu_percent())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100*mem.active/mem.total)
            if torch.cuda.is_available():
                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100*free/total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

def save_checkpoint(data):
    config = data.config
    path = os.path.join(config.data_dir, config.exp_id)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f'model_{data.epoch:06d}.pt'
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        return model_path

    checkpoint = {
        "config": data.config,
        "state_dict": data.uncompiled_policy.state_dict()
    }
    torch.save(checkpoint, model_path)

    state = {
        'optimizer_state_dict': data.optimizer.state_dict(),
        'global_step': data.global_step,
        'agent_step': data.global_step,
        'update': data.epoch,
        'model_name': model_name,
        'exp_id': config.exp_id,
    }
    state_path = os.path.join(path, 'trainer_state.pt')
    torch.save(state, state_path + '.tmp')
    os.rename(state_path + '.tmp', state_path)
    return model_path

def try_load_checkpoint(data):
    config = data.config
    path = os.path.join(config.data_dir, config.exp_id)
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

def rollout(env_creator, env_kwargs, policy_cls, rnn_cls, agent_creator, agent_kwargs,
        backend, render_mode='auto', model_path=None, device='cuda'):

    if render_mode != 'auto':
        env_kwargs['render_mode'] = render_mode

    # We are just using Serial vecenv to give a consistent
    # single-agent/multi-agent API for evaluation
    env = pufferlib.vector.make(env_creator, env_kwargs=env_kwargs, backend=backend)

    agent = agent_creator(env, policy_cls, rnn_cls, agent_kwargs).to(device)
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(checkpoint['state_dict'])

    ob, info = env.reset()
    driver = env.driver_env
    os.system('clear')
    state = None

    frames = []
    tick = 0
    while tick <= 200000:
        if tick % 1 == 0:
            render = driver.render()
            if driver.render_mode == 'ansi':
                print('\033[0;0H' + render + '\n')
                time.sleep(0.05)
            elif driver.render_mode == 'rgb_array':
                frames.append(render)
                import cv2
                render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', render)
                cv2.waitKey(1)
                time.sleep(1/24)
            elif driver.render_mode in ('human', 'raylib') and render is not None:
                frames.append(render)

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            if hasattr(agent, 'lstm'):
                action, _, _, _, state = agent(ob, state)
            else:
                action, _, _, _ = agent(ob)

            action = action.cpu().numpy().reshape(env.action_space.shape)

        ob, reward = env.step(action)[:2]
        reward = reward.mean()
        if tick % 128 == 0:
            print(f'Reward: {reward:.4f}, Tick: {tick}')
        tick += 1

    # Save frames as gif
    if frames:
        import imageio
        os.makedirs('../docker', exist_ok=True) or imageio.mimsave('../docker/eval.gif', frames, fps=15, loop=0)

def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

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

# TODO: Add env name to print_dashboard
def print_dashboard(env_name, utilization, global_step, epoch,
        profile, losses, stats, msg, clear=False, max_stats=[0]):
    console = Console()
    if clear:
        console.clear()

    dashboard = Table(box=ROUND_OPEN, expand=True,
        show_header=False, border_style='bright_cyan')

    table = Table(box=None, expand=True, show_header=False)
    dashboard.add_row(table)
    cpu_percent = np.mean(utilization.cpu_util)
    dram_percent = np.mean(utilization.cpu_mem)
    gpu_percent = np.mean(utilization.gpu_util)
    vram_percent = np.mean(utilization.gpu_mem)
    table.add_column(justify="left", width=30)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=13)
    table.add_column(justify="right", width=13)
    table.add_row(
        f':blowfish: {c1}PufferLib {b2}2.0.0',
        f'{c1}CPU: {c3}{cpu_percent:.1f}%',
        f'{c1}GPU: {c3}{gpu_percent:.1f}%',
        f'{c1}DRAM: {c3}{dram_percent:.1f}%',
        f'{c1}VRAM: {c3}{vram_percent:.1f}%',
    )
        
    s = Table(box=None, expand=True)
    s.add_column(f"{c1}Summary", justify='left', vertical='top', width=16)
    s.add_column(f"{c1}Value", justify='right', vertical='top', width=8)
    s.add_row(f'{c2}Environment', f'{b2}{env_name}')
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
        try: # Discard non-numeric values
            int(value)
        except:
            continue

        u = left if i % 2 == 0 else right
        u.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')
        i += 1
        if i == 30:
            break

    for i in range(max_stats[0] - i):
        u = left if i % 2 == 0 else right
        u.add_row('', '')

    max_stats[0] = max(max_stats[0], i)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    table.add_row(f' {c1}Message: {c2}{msg}')

    with console.capture() as capture:
        console.print(dashboard)

    print('\033[0;0H' + capture.get())
