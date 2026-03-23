## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import os
import glob
import time
import importlib
from collections import defaultdict

import numpy as np

import torch
import torch.distributed

import pufferlib
import pufferlib.pytorch
import pufferlib.pufferl
from pufferlib.muon import Muon

from pufferlib import _C

from torch.distributions.utils import logits_to_probs

def _log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)

def _entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)

def sample_logits(logits, action=None):
    is_discrete = isinstance(logits, torch.Tensor)
    if isinstance(logits, torch.distributions.Normal):
        batch = logits.loc.shape[0]
        if action is None:
            action = logits.sample().view(batch, -1)
        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, logits_entropy
    elif is_discrete:
        logits = logits.unsqueeze(0)
    else: # multi-discrete
        logits = torch.nn.utils.rnn.pad_sequence(
            [l.transpose(0,1) for l in logits],
            batch_first=False,
            padding_value=-torch.inf
        ).permute(1,2,0)

    normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    probs = logits_to_probs(logits)

    if action is None:
        probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
        action = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, replacement=True).int()
        action = action.reshape(probs.shape[:-1])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    logprob = _log_prob(normalized_logits, action)
    logits_entropy = _entropy(normalized_logits).sum(0)

    if is_discrete:
        return action.T, logprob.squeeze(0), logits_entropy.squeeze(0)

    return action.T, logprob.sum(0), logits_entropy

_OBS_DTYPE_MAP = {
    'ByteTensor':   torch.uint8,
    'FloatTensor':  torch.float32,
}

_TORCH_TO_TYPESTR = {
    torch.uint8:   '|u1',
    torch.float32: '<f4',
}

class _CudaPtr:
    '''Wraps a raw CUDA pointer so torch.as_tensor can consume it via
    __cuda_array_interface__ without any copy or C++ torch dependency.'''
    def __init__(self, ptr, shape, dtype):
        self.__cuda_array_interface__ = {
            'data':    (ptr, False),
            'shape':   shape,
            'typestr': _TORCH_TO_TYPESTR[dtype],
            'version': 2,
        }

class _Space:
    def __init__(self, shape, dtype, nvec=None):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        if nvec is not None:
            self.nvec = np.array(nvec, dtype=np.int64)
            self.n = int(self.nvec.sum())
        else:
            self.n = shape[0] if shape else 0
            self.nvec = np.array([self.n], dtype=np.int64)

class VecEnv:
    '''Wraps _C.VecEnv. GPU buffer tensors created once at init (zero-copy).
    step(actions) advances the env; obs/rewards/terminals update in-place.'''

    def __init__(self, vec):
        self._vec = vec
        self.num_agents = vec.total_agents
        obs_dtype = _OBS_DTYPE_MAP.get(vec.obs_dtype, torch.uint8)
        np_obs_dtype = {torch.uint8: np.uint8, torch.float32: np.float32}[obs_dtype]
        self.single_observation_space = _Space((vec.obs_size,), np_obs_dtype)
        self.single_action_space = _Space((vec.num_atns,), np.float64, nvec=vec.act_sizes)
        self.driver_env = self

        self.observations = torch.as_tensor(_CudaPtr(vec.gpu_obs_ptr,
            (vec.total_agents, vec.obs_size), obs_dtype))
        self.rewards = torch.as_tensor(_CudaPtr(vec.gpu_rewards_ptr,
            (vec.total_agents,), torch.float32))
        self.terminals = torch.as_tensor(_CudaPtr(vec.gpu_terminals_ptr,
            (vec.total_agents,), torch.float32))

    def reset(self, seed=0):
        self._vec.reset()

    def step(self, actions):
        actions = actions.T if actions.dim() > 1 else actions.unsqueeze(-1)
        actions_gpu = actions.to(dtype=torch.float32, device='cuda').contiguous()
        self._vec.step(actions_gpu.data_ptr())
        torch.cuda.synchronize()
        return self.observations, self.rewards, self.terminals

    def log(self):
        return self._vec.log()

    def close(self):
        self._vec.close()

class PuffeRL:
    def __init__(self, args, vecenv, policy, verbose=True):
        config = args['train']
        device = config['device']

        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        vecenv.reset(config['seed'])
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents
        horizon = config['horizon']

        self.observations = torch.zeros(horizon, total_agents, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype], device=device)
        self.actions = torch.zeros(horizon, total_agents, *atn_space.shape, device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
        self.values = torch.zeros(horizon, total_agents, device=device)
        self.logprobs = torch.zeros(horizon, total_agents, device=device)
        self.rewards = torch.zeros(horizon, total_agents, device=device)
        self.terminals = torch.zeros(horizon, total_agents, device=device)
        self.ratio = torch.ones(total_agents, horizon, device=device)
        self.state = policy.initial_state(total_agents, device=device)

        self.batch_size = total_agents * horizon
        self.minibatch_segments = config['minibatch_size'] // horizon
        self.total_epochs = max(1, config['total_timesteps'] // self.batch_size)

        self.policy = policy
        self.optimizer = Muon(
            self.policy.parameters(),
            lr=config['learning_rate'],
            momentum=config['beta1'],
            eps=config['eps'],
        )

        self.args = args
        self.config = config
        self.vecenv = vecenv
        self.world_size = args['world_size']
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.profile = Profile()
        self.env_logs = {}
        self.losses = {}
        self.verbose = verbose

        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        if verbose:
            pufferlib.pufferl.print_dashboard(args, self.model_size, {}, clear=True)


    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def num_params(self):
        return self.model_size

    def rollouts(self):
        prof = self.profile
        config = self.config
        device = config['device']

        self.state = tuple(torch.zeros_like(s) for s in self.state) if self.state else ()

        horizon = config['horizon']
        o = self.vecenv.observations
        r = torch.zeros(self.total_agents, device=device)
        d = torch.zeros(self.total_agents, device=device)

        P = Profile
        prof.mark(0)
        for t in range(horizon):
            o_device = torch.as_tensor(o, device=device)

            prof.mark(1)
            with torch.no_grad():
                logits, value, state = self.policy.forward_eval(o_device, self.state)
                action, logprob, _ = sample_logits(logits)
            prof.mark(2)

            with torch.no_grad():
                self.state = state
                self.observations[t] = o_device
                self.actions[t] = action
                self.logprobs[t] = logprob
                self.rewards[t] = torch.as_tensor(r, device=device)
                self.terminals[t] = torch.as_tensor(d, device=device).float()
                self.values[t] = value.flatten()

            prof.mark(2)
            o, r, d = self.vecenv.step(action)
            prof.mark(3)

            prof.elapsed(P.EVAL_GPU, 1, 2)
            prof.elapsed(P.EVAL_ENV, 2, 3)

        prof.mark(1)
        prof.elapsed(P.ROLLOUT, 0, 1)

        self.global_step += self.total_agents * horizon

        self.env_logs = self.vecenv.log()

    def train(self):
        prof = self.profile
        losses = defaultdict(float)
        config = self.config
        device = config['device']

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1

        learning_rate = config['learning_rate']
        if config['anneal_lr'] and self.epoch > 0:
            lr_ratio = self.epoch / self.total_epochs
            lr_min = config['learning_rate'] * config['min_lr_ratio']
            learning_rate = lr_min + 0.5*(learning_rate - lr_min) * (1 + np.cos(np.pi * lr_ratio))
            self.optimizer.param_groups[0]['lr'] = learning_rate

        # Transpose from [horizon, agents] (contiguous writes) to [agents, horizon] (minibatch indexing)
        obs = self.observations.transpose(0, 1).contiguous()
        act = self.actions.transpose(0, 1).contiguous()
        val = self.values.T.contiguous()
        lp = self.logprobs.T.contiguous()
        rew = self.rewards.T.contiguous().clamp(-1, 1)
        ter = self.terminals.T.contiguous()

        P = Profile
        prof.mark(0)
        num_minibatches = int(config['replay_ratio'] * self.batch_size / config['minibatch_size'])
        for mb in range(num_minibatches):
            shape = val.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(val, rew,
                ter, self.ratio, advantages, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs,
                self.minibatch_segments, replacement=True)
            mb_prio = (self.total_agents*prio_probs[idx, None])**-anneal_beta

            mb_obs = obs[idx]
            mb_actions = act[idx]
            mb_logprobs = lp[idx]
            mb_values = val[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            prof.mark(1)
            logits, newvalue = self.policy(mb_obs)
            actions, newlogprob, entropy = sample_logits(logits, action=mb_actions)
            prof.mark(2)
            prof.elapsed(P.TRAIN_FORWARD, 1, 2)

            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss
            val[idx] = newvalue.detach().float()

            losses['policy_loss'] += pg_loss
            losses['value_loss'] += v_loss
            losses['entropy'] += entropy_loss
            losses['old_approx_kl'] += old_approx_kl
            losses['approx_kl'] += approx_kl
            losses['clipfrac'] += clipfrac
            losses['importance'] += ratio.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
            self.optimizer.step()
            self.optimizer.zero_grad()

        prof.mark(1)
        prof.elapsed(P.TRAIN, 0, 1)

        losses = {k: v.item() / num_minibatches for k, v in losses.items()}
        y_pred = val.flatten()
        y_true = advantages.flatten() + val.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var

        self.losses = losses
        self.epoch += 1

    def log(self):
        P = Profile
        perf = self.profile.read_and_reset()
        logs = {
            'SPS': self.sps * self.world_size,
            'agent_steps': self.global_step * self.world_size,
            'uptime': time.time() - self.start_time,
            'epoch': self.epoch,
            'env': dict(self.env_logs),
            'loss': dict(self.losses),
            'perf': {
                'rollout': perf[P.ROLLOUT],
                'eval_gpu': perf[P.EVAL_GPU],
                'eval_env': perf[P.EVAL_ENV],
                'train': perf[P.TRAIN],
                'train_misc': perf[P.TRAIN_MISC],
                'train_forward': perf[P.TRAIN_FORWARD],
            },
            'util': dict(_C.get_utilization(self.args.get('gpu_id', 0))),
        }
        self.last_log_time = time.time()
        self.last_log_step = self.global_step
        return logs

    eval_log = log

    def save_weights(self, path):
        torch.save(self.policy.state_dict(), path)

    def close(self):
        self.vecenv.close()

    @classmethod
    def create_pufferl(cls, args):
        '''Matches _C.create_pufferl(args) interface.'''
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        args['train']['device'] = device

        # DDP setup
        if 'LOCAL_RANK' in os.environ:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

        args['vec']['num_buffers'] = 1
        vecenv = load_env(args['env_name'], args)
        policy = load_policy(args, vecenv, args['env_name'])

        if 'LOCAL_RANK' in os.environ:
            torch.distributed.init_process_group(backend='nccl', world_size=world_size)
            policy = policy.to(local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                policy, device_ids=[local_rank], output_device=local_rank)
            if hasattr(policy, 'lstm'):
                model.hidden_size = policy.hidden_size
            model.forward_eval = policy.forward_eval
            model.initial_state = policy.initial_state
            policy = model.to(local_rank)

        return cls(args, vecenv, policy)

def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    num_steps, horizon = values.shape
    _C.puff_advantage(
        values.data_ptr(), rewards.data_ptr(), terminals.data_ptr(),
        ratio.data_ptr(), advantages.data_ptr(),
        num_steps, horizon,
        gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)
    return advantages

class Profile:
    '''Matches pufferlib.cu profiling: accumulate ms, report seconds.'''
    ROLLOUT, EVAL_GPU, EVAL_ENV, TRAIN, TRAIN_MISC, TRAIN_FORWARD, NUM = range(7)

    def __init__(self):
        self.accum = [0.0] * Profile.NUM
        self._events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    def mark(self, idx):
        self._events[idx].record()

    def elapsed(self, idx, start_ev, end_ev):
        self._events[end_ev].synchronize()
        self.accum[idx] += self._events[start_ev].elapsed_time(self._events[end_ev])

    def read_and_reset(self):
        out = [v / 1000.0 for v in self.accum]
        self.accum = [0.0] * Profile.NUM
        return out

def load_env(env_name, args):
    return VecEnv(_C.create_vec(args))

def load_policy(args, vecenv, env_name=''):
    import pufferlib.models
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)

    policy_kwargs = args['policy']
    network_cls = getattr(pufferlib.models, policy_kwargs['network'])
    encoder_cls = getattr(env_module.torch, 'Encoder', pufferlib.models.DefaultEncoder)
    decoder_cls = getattr(env_module.torch, 'Decoder', pufferlib.models.DefaultDecoder)
    network = network_cls(**policy_kwargs)
    encoder = encoder_cls(vecenv.driver_env, policy_kwargs['hidden_size'])
    decoder = decoder_cls(vecenv.driver_env, policy_kwargs['hidden_size'])
    policy = pufferlib.models.Policy(encoder, decoder, network)

    device = args['train']['device']
    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['wandb']:
            import wandb
            artifact = wandb.use_artifact(f'{load_id}:latest')
            data_dir = artifact.download()
            path = f'{data_dir}/{max(os.listdir(data_dir))}'
        else:
            raise pufferlib.APIUsageError('load_id requires --wandb')

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args['load_model_path']
    if load_path == 'latest':
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    return policy

