## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
import io
import sys
import glob
import ast
import time
import random
import shutil
import argparse
import importlib
import configparser
from threading import Thread
from collections import defaultdict, deque

import numpy as np
import psutil

import torch
from torch import func
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch
try:
    from pufferlib import _C
    from pufferlib import fake_tensors
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME
)
# Assume advantage kernel has been built if torch has been compiled with CUDA or HIP support
# and can find CUDA or HIP in the system
ADVANTAGE_CUDA = bool(CUDA_HOME or ROCM_HOME)

# DEBUG FLAG IS A BUG. FUCK THIS DO NOT NOT NOT ENABLE
#torch.autograd.set_detect_anomaly(True)
#torch._dynamo.config.capture_scalar_outputs = True


class PuffeRL:
    def __init__(self, config, logger=None, verbose=True):
        # Backend perf optimization
        num_envs = 8192
        self.num_envs = num_envs
        #grid_size = 11
        dummy = torch.zeros(5).cuda()

        #vecenv = CPPEnv(num_envs)
        #vecenv.reset()

        # Reproducibility
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        from gymnasium.spaces import Box, Discrete
        obs_space = Box(low=0, high=2, shape=(118,), dtype=np.float32)
        atn_space = Discrete(3)
        self.single_observation_space = obs_space
        self.single_action_space = atn_space

        # Vecenv info
        #vecenv.async_reset(seed)
        #obs_space = vecenv.single_observation_space
        #atn_space = vecenv.single_action_space
        #total_agents = vecenv.num_agents
        total_agents = num_envs
        self.total_agents = total_agents
        self.agents_per_batch = total_agents

        # Experience
        if config['batch_size'] == 'auto' and config['bptt_horizon'] == 'auto':
            raise pufferlib.APIUsageError('Must specify batch_size or bptt_horizon')
        elif config['batch_size'] == 'auto':
            config['batch_size'] = total_agents * config['bptt_horizon']
        elif config['bptt_horizon'] == 'auto':
            config['bptt_horizon'] = config['batch_size'] // total_agents

        batch_size = config['batch_size']
        horizon = config['bptt_horizon']
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(
                f'Total agents {total_agents} <= segments {segments}'
            )

        device = config['device']
        '''
        self.observations = torch.zeros(segments, horizon, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == 'cuda' and config['cpu_offload'],
            device='cpu' if config['cpu_offload'] else device)
        self.actions = torch.zeros(segments, horizon, *atn_space.shape, device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents
        '''
 
        # Minibatching & gradient accumulation
        minibatch_size = config['minibatch_size']
        max_minibatch_size = config['max_minibatch_size']
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = config['num_minibatches']
        self.minibatch_segments = self.minibatch_size // horizon 
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f'minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}'
            )

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = Logger(config)

        epochs = config['total_timesteps'] // config['batch_size']
        eta_min = config['learning_rate'] * config['min_lr_ratio']
        
        self.total_epochs = epochs

        self.num_layers = 4
        config['input_size'] = 118
        config['num_atns'] = 3
        config['hidden_size'] = 128
        config['expansion_factor'] = 1
        config['num_layers'] = self.num_layers
        config['minibatch_segments'] = self.minibatch_segments
        config['segments'] = segments
        config['horizon'] = horizon
        config['lr'] = config['learning_rate']
        config['beta1'] = config['adam_beta1']
        config['beta2'] = config['adam_beta2']
        config['eps'] = config['adam_eps']
        config['max_epochs'] = epochs
        config['total_minibatches'] = self.total_minibatches
        config['accumulate_minibatches'] = self.accumulate_minibatches
        config['num_envs'] = self.num_envs
        config['cudagraphs'] = True
        config['kernels'] = True
        config['profile'] = False
        config['num_buffers'] = 2
        self.pufferl_cpp = _C.create_pufferl(config)
        self.observations = self.pufferl_cpp.observations
        self.actions = self.pufferl_cpp.actions
        self.rewards = self.pufferl_cpp.rewards
        self.terminals = self.pufferl_cpp.terminals
        self.logprobs = self.pufferl_cpp.logprobs
        self.values = self.pufferl_cpp.values
        self.debug = self.pufferl_cpp.debug

        # Initializations
        self.config = config
        #self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}
        self.verbose = verbose

        # Dashboard
        #self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

        #self.compiled_evaluate = torch.compile(_C.compiled_evaluate)
        #self.eval_forward = torch.compile(self.pufferl_cpp.policy.forward, mode='reduce-overhead')

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)


    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)

        config = self.config
        device = config['device']

        state = _C.rollouts(self.pufferl_cpp,)

        '''
        obs, act, rew, term = _C.env_buffers(self.pufferl_cpp)

        num_buffers = 2
        block_size = int(self.num_envs / num_buffers)
        with torch.no_grad():
            for i in range(self.config['bptt_horizon']):
                buf = i % num_buffers
                h = int(i / num_buffers)
                _C.python_vec_recv(self.pufferl_cpp, buf)

                start = int(block_size * buf)
                obs_batch = obs.narrow(0, start, block_size)
                state_batch = state.narrow(1, start, block_size)
                logits, value, state_out = self.eval_forward(obs_batch.cuda(), state_batch)
                state_batch.copy_(state_out)

                logits = torch.nan_to_num(logits)
                logprobs = torch.log_softmax(logits, dim=1)
                action = torch.multinomial(logprobs.exp(), 1, True).squeeze(1).to(torch.int32)
                logprob = logprobs.gather(1, action.unsqueeze(1)).squeeze(1)

                self.observations.select(1, h).narrow(0, start, block_size).copy_(obs_batch, True)
                self.actions.select(1, h).narrow(0, start, block_size).copy_(action.to(torch.int64), True)
                self.logprobs.select(1, h).narrow(0, start, block_size).copy_(logprob.to(torch.float32), True)
                self.values.select(1, h).narrow(0, start, block_size).copy_(value.flatten().to(torch.float32), True)

                rewards_batch = rew.narrow(0, start, block_size)
                rewards_clamped = torch.clamp(rewards_batch, -1, 1)

                self.rewards.select(1, h).narrow(0, start, block_size).copy_(rewards_clamped.to(torch.float32), True)

                terminals_batch = term.narrow(0, start, block_size)
                self.terminals.select(1, h).narrow(0, start, block_size).copy_(terminals_batch.to(torch.float32), True)

                act.narrow(0, start, block_size).copy_(action.to(torch.float32), True)

                torch.cuda.synchronize()
                _C.python_vec_send(self.pufferl_cpp, buf)
        '''

        #torch.cuda.synchronize()
        logs = _C.log_environments(self.pufferl_cpp)
        if logs:
            self.stats['perf'] = [logs['perf']]
            self.stats['score'] = [logs['score']]
            self.stats['episode_return'] = [logs['episode_return']]
            self.stats['episode_length'] = [logs['episode_length']]
            self.stats['n'] = [logs['n']]

        self.global_step += config['batch_size']
        profile.end()
        return self.stats

    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)
        config = self.config
        device = config['device']

        losses = _C.train(self.pufferl_cpp)

        profile('train_misc', epoch)
        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            #self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        return logs

    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        device = config['device']
        agent_steps = int(dist_sum(self.global_step, device))
        logs = {
            'SPS': dist_sum(self.sps, device),
            'agent_steps': agent_steps,
            'uptime': time.time() - self.start_time,
            'epoch': int(dist_sum(self.epoch, device)),
            #'learning_rate': self.optimizer.param_groups[0]["lr"],
            **{f'environment/{k}': v for k, v in self.stats.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v['elapsed'] for k, v in self.profile},
            #**{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            #**{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            #**{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               self.logger.log(logs, agent_steps)
               return logs
           else:
               return None

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        os._exit(0)
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'],
            self.config["env"], f'{run_id}.pt')
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'],
            self.config["env"], run_id)
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        #torch.save(self.uncompiled_policy.state_dict(), model_path)

        state = {
            #'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'update': self.epoch,
            'model_name': model_name,
            'run_id': run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.replace(state_path + '.tmp', state_path)
        return model_path

    def print_dashboard(self, clear=False, idx=[0],
            c1='[cyan]', c2='[white]', b1='[bright_cyan]', b2='[bright_white]'):
        if not self.verbose:
            return

        config = self.config
        sps = dist_sum(self.sps, config['device'])
        agent_steps = dist_sum(self.global_step, config['device'])
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True,
            show_header=False, border_style='bright_cyan')
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f'{b1}PufferLib {b2}3.0 {idx[0]*" "}:blowfish:',
            f'{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%',
            f'{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%',
            f'{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%',
            f'{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%',
        )
        idx[0] = (idx[0] - 1) % 10
            
        s = Table(box=None, expand=True)
        remaining = f'{b2}A hair past a freckle{c2}'
        if sps != 0:
            remaining = duration((config['total_timesteps'] - agent_steps)/sps, b2, c2)

        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
        s.add_row(f'{c2}Env', f'{b2}{config["env"]}')
        #s.add_row(f'{c2}Params', abbreviate(self.model_size, b2, c2))
        s.add_row(f'{c2}Steps', abbreviate(agent_steps, b2, c2))
        s.add_row(f'{c2}SPS', abbreviate(sps, b2, c2))
        s.add_row(f'{c2}Epoch', f'{b2}{self.epoch}')
        s.add_row(f'{c2}Uptime', duration(self.uptime, b2, c2))
        s.add_row(f'{c2}Remaining', remaining)

        delta = profile.eval['buffer'] + profile.train['buffer']
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf('Evaluate', b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf('  Env', b2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf('Train', b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf('  Learn', b2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.train_misc, b2, c2))

        l = Table(box=None, expand=True, )
        l.add_column(f'{c1}Losses', justify="left", width=16)
        l.add_column(f'{c1}Value', justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')

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

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    '''CUDA kernel for puffer advantage with automatic CPU fallback. You need
    nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
    compile the fast version.'''

    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()

    torch.ops.pufferlib.compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)

    if not ADVANTAGE_CUDA:
        return advantages.to(device)

    return advantages


def abbreviate(num, b2, c2):
    if num < 1e3:
        return f'{b2}{num}{c2}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}K'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}M'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}B'
    else:
        return f'{b2}{num/1e12:.2f}{c2}T'

def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*prof['buffer']/delta_ref - 1e-5)
    return f'{color}{name}', duration(prof['elapsed'], b2, c2), f'{b2}{percent:2d}{c2}%'

def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()

def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()

class Profile:
    def __init__(self, frequency=1):
        self.reset()
        self.frequency = frequency
        self.stack = []

    def reset(self):
        self.profiles = defaultdict(lambda: defaultdict(float))

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        # Skip profiling the first few epochs, which are noisy due to setup
        if (epoch + 1) % self.frequency != 0:
            return

        #if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]['start'] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile['start']
        profile['delta'] += delta
        # Multiply delta by freq to account for skipped epochs
        profile['elapsed'] += delta * self.frequency

    def end(self):
        #if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for prof in self.profiles.values():
            if prof['delta'] > 0:
                prof['buffer'] = prof['delta']
                prof['delta'] = 0

class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100*psutil.cpu_percent()/psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100*mem.active/mem.total)
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                   time.sleep(self.delay)
                   continue

                #self.gpu_util.append(torch.cuda.utilization())
                #free, total = torch.cuda.mem_get_info()
                #self.gpu_mem.append(100*(total-free)/total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

def downsample(data_list, num_points):
    if not data_list or num_points <= 0:
        return []
    if num_points == 1:
        return [data_list[-1]]
    if len(data_list) <= num_points:
        return data_list

    last = data_list[-1]
    data_list = data_list[:-1]

    data_np = np.array(data_list)
    num_points -= 1  # one down for the last one

    n = (len(data_np) // num_points) * num_points
    data_np = data_np[-n:] if n > 0 else data_np
    downsampled = data_np.reshape(num_points, -1).mean(axis=1)

    return downsampled.tolist() + [last]

class Logger:
    def __init__(self, args):
        self.run_id = str(int(1000*time.time()))
        root = os.path.join(args['data_dir'], 'logs', args['env'])
        if not os.path.exists(root):
            os.makedirs(root)

        self.path = os.path.join(root, self.run_id + '.json')
        self.logs = {'data': []}
        for k, v in pufferlib.unroll_nested_dict(args):
            self.logs[k] = v

    # Temp hack to log full config
    def init(self, args):
        for k, v in pufferlib.unroll_nested_dict(args):
            self.logs[k] = v

    def log(self, logs, step):
        self.logs['data'].append(logs)

    def log_cost(self, cost):
        self.logs['cost'] = cost

    def close(self, model_path):
        import json
        with open(self.path, 'w') as f:
            json.dump(self.logs, f)

class NeptuneLogger:
    def __init__(self, args, load_id=None, mode='async'):
        import neptune as nept
        neptune_name = args['neptune_name']
        neptune_project = args['neptune_project']
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags = [args['tag']] if args['tag'] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)
        self.should_upload_model = not args['no_model_upload']

    def init(self, args):
        pass

    def log_cost(self, cost):
        pass

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def upload_model(self, model_path):
        self.neptune['model'].track_files(model_path)

    def close(self, model_path):
        if self.should_upload_model:
            self.upload_model(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination='artifacts')
        return f'artifacts/{self.run_id}.pt'
 
class WandbLogger:
    def __init__(self, args, load_id=None, resume='allow'):
        import wandb
        wandb.init(
            id=load_id or wandb.util.generate_id(),
            project=args['wandb_project'],
            group=args['wandb_group'],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            tags = [args['tag']] if args['tag'] is not None else [],
            settings=wandb.Settings(console="off"),  # stop sending dashboard to wandb
        )
        self.wandb = wandb
        self.run_id = wandb.run.id
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type='model')
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path):
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f'{self.run_id}:latest')
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f'{data_dir}/{model_file}'

def check(env_name):
    torch.set_printoptions(precision=16)

    args = load_config(env_name)
    args['train']['optimizer'] = 'adam'

    vecenv = load_env(env_name, args)

    torch.manual_seed(args['train']['seed'])
    policy = load_policy(args, vecenv, env_name)

    import pufferlib.python_pufferl
    train_config = dict(**args['train'])
    pufferl_python = pufferlib.python_pufferl.PuffeRL(train_config, vecenv, policy, verbose=False)

    pufferl_cpp = PuffeRL(train_config, verbose=False)

    python_params = dict(policy.named_parameters())
    for k, v in pufferl_cpp.pufferl_cpp.policy.named_parameters():
        v_python = python_params[k].data
        assert torch.allclose(v, v_python)

    torch.manual_seed(args['train']['seed'])
    pufferl_python.evaluate()
    #pufferl_python.train()
    #pufferl_python.evaluate()

    torch.manual_seed(args['train']['seed'])
    pufferl_cpp.evaluate()
    #pufferl_cpp.train()
    #pufferl_cpp.evaluate()

    # You need to determinize the env before checks
    for i in range(args['train']['bptt_horizon']):
        python_obs = pufferl_python.observations[:, i].float()
        cpp_obs = pufferl_cpp.observations[:, i]
        assert torch.allclose(pufferl_python.observations[:, i].float(), pufferl_cpp.observations[:, i]), f'Observation {i} mismatch'
        assert torch.allclose(pufferl_python.actions[:, i], pufferl_cpp.actions[:, i].long()), f'Action {i} mismatch'
        assert torch.allclose(pufferl_python.rewards[:, i], pufferl_cpp.rewards[:, i]), f'Reward {i} mismatch'
        assert torch.allclose(pufferl_python.terminals[:, i], pufferl_cpp.terminals[:, i]), f'Terminal {i} mismatch'
        assert torch.allclose(pufferl_python.logprobs[:, i], pufferl_cpp.logprobs[:, i], atol=1e-5), f'Logprob {i} mismatch'
        assert torch.allclose(pufferl_python.values[:, i], pufferl_cpp.values[:, i], atol=1e-4), f'Value {i} mismatch'

    python_params = dict(policy.named_parameters())
    for k, v in pufferl_cpp.pufferl_cpp.policy.named_parameters():
        v_python = python_params[k].data
        assert torch.allclose(v, v_python, atol=1e-5)

    print('Check passed')

def train(env_name, args=None, vecenv=None, policy=None, logger=None, verbose=True, should_stop_early=None):
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if 'LOCAL_RANK' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print("World size", world_size)
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    #vecenv = vecenv or load_env(env_name, args)
    #policy = policy or load_policy(args, vecenv, env_name)

    if 'LOCAL_RANK' in os.environ:
        args['train']['device'] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend='nccl', world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank
        )
        if hasattr(policy, 'lstm'):
            #model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args['neptune']:
        logger = NeptuneLogger(args)
    elif args['wandb']:
        logger = WandbLogger(args)

    train_config = dict(**args['train'])#, env=env_name)
    #pufferl = PuffeRL(train_config, vecenv, policy, logger, verbose)
    pufferl = PuffeRL(train_config, logger, verbose)
    pufferl.logger.init(args)

    all_logs = []
    max_cost = args['train'].get('max_cost', -1)
    while pufferl.global_step < train_config['total_timesteps']:
        if pufferl.uptime > max_cost and max_cost > 0:
            break
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
            if pufferl.global_step > 0.20*train_config['total_timesteps']:
                all_logs.append(logs)

            if should_stop_early is not None and should_stop_early(logs):
                model_path = pufferl.close()
                pufferl.logger.close(model_path)
                return all_logs

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    '''
    uptime = pufferl.uptime
    agent_steps = pufferl.global_step
    for i in range(128):  # Run eval for at least 32, but put a hard stop at 128.
        stats = pufferl.evaluate()
        if i >= 32 and stats:
            break

    logs = pufferl.mean_and_log()
    logs['uptime'] = uptime
    logs['agent_steps'] = agent_steps
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.log_cost(uptime)
    pufferl.logger.close(model_path)
    return all_logs
    '''

def sps(env_name, args=None, vecenv=None, policy=None, logger=None, verbose=True, should_stop_early=None):
    args = args or load_config(env_name)
    train_config = dict(**args['train'])#, env=env_name)
    pufferl = PuffeRL(train_config, logger, verbose)
    # Warmup
    for _ in range(3):
        _C.batched_forward(
            pufferl.pufferl_cpp,
            pufferl.observations,
            pufferl.total_minibatches,
            pufferl.minibatch_segments,
        )

    N = 100
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        _C.batched_forward(
            pufferl.pufferl_cpp,
            pufferl.observations,
            pufferl.total_minibatches,
            pufferl.minibatch_segments,
        )
    torch.cuda.synchronize()
    end = time.time()
    dt = end - start
    sps = pufferl.config['batch_size']*N/dt
    print(f'SPS: {sps/1e6:.1f}M')


def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    backend = args['vec']['backend']
    if backend != 'PufferEnv':
        backend = 'Serial'

    args['vec'] = dict(backend=backend, num_envs=1)
    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    ob, info = vecenv.reset()
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]
    device = args['train']['device']

    state = {}
    if args['train']['use_rnn']:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    frames = []
    while True:
        render = driver.render()
        if len(frames) < args['save_frames']:
            frames.append(render)

        # Screenshot Ocean envs with F12, gifs with control + F12
        if driver.render_mode == 'ansi':
            print('\033[0;0H' + render + '\n')
            time.sleep(1/args['fps'])
        elif driver.render_mode == 'rgb_array':
            pass
            #import cv2
            #render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            #cv2.imshow('frame', render)
            #cv2.waitKey(1)
            #time.sleep(1/args['fps'])

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            logits, value = policy.forward_eval(ob, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

        if isinstance(logits, torch.distributions.Normal):
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        ob = vecenv.step(action)[0]

        if len(frames) > 0 and len(frames) == args['save_frames']:
            import imageio
            imageio.mimsave(args['gif_path'], frames, fps=args['fps'], loop=0)
            print(f'Saved {len(frames)} frames to {args["gif_path"]}')

def stop_if_loss_nan(logs):
    return any("losses/" in k and np.isnan(v) for k, v in logs.items())

def _sweep_worker(env_name, q_host, q_worker, device):
    while True:
        #print("Worker waiting")
        args = q_worker.get()
        #print("Worker got data")
        args['train']['device'] = device
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            all_logs = train(env_name, args=args, verbose=False)
        except Exception:
            import traceback
            traceback.print_exc()

        #all_logs = [{'foo': 0}]
        #print("Worker ran experiment")
        q_host.put(all_logs)
        #print("Worker submitted result")

def multisweep(args=None, env_name=None):
    args = args or load_config(env_name)
    sweep_gpus = args['sweep_gpus']
    if sweep_gpus == -1:
        sweep_gpus = torch.cuda.device_count()

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'

    from multiprocessing import Process, Queue, set_start_method
    from copy import deepcopy

    host_queues = []
    worker_queues = []
    workers = []
    worker_args = []
    set_start_method('spawn')
    for i in range(sweep_gpus):
        q_host = Queue()
        q_worker = Queue()
        w = Process(
            target=_sweep_worker,
            args=(env_name, q_host, q_worker, f'cuda:{i}')
        )
        w.start()
        host_queues.append(q_host)
        worker_queues.append(q_worker)
        args = deepcopy(args)
        worker_args.append(args)

    for w in range(sweep_gpus):
        args = worker_args[w]
        sweep.suggest(args)
        total_timesteps = args['train']['total_timesteps']
        worker_queues[w].put(args)

    runs = 0

    suggestion = deepcopy(args)
    while runs < args['max_runs']:
        for w in range(sweep_gpus):
            args = worker_args[w]
            if host_queues[w].empty():
                continue

            all_logs = host_queues[w].get(timeout=0)
            if not all_logs:
                continue

            all_logs = [e for e in all_logs if target_key in e]
            scores = downsample([log[target_key] for log in all_logs], points_per_run)
            times = downsample([log['uptime'] for log in all_logs], points_per_run)
            steps = downsample([log['agent_steps'] for log in all_logs], points_per_run)
            #costs = np.stack([times, steps], axis=1)
            costs = times
            timesteps = [log['agent_steps'] for log in all_logs]
            timesteps = downsample(timesteps, points_per_run)
            for score, cost, timestep in zip(scores, costs, timesteps):
                args['train']['total_timesteps'] = timestep
                sweep.observe(args, score, cost)

            runs += 1

            sweep.suggest(args)
            worker_queues[w].put(args)

def paretosweep(args=None, env_name=None):
    args = args or load_config(env_name)
    sweep_gpus = args['sweep_gpus']
    if sweep_gpus == -1:
        sweep_gpus = torch.cuda.device_count()

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    total_timesteps = args['sweep']['train'].pop('total_timesteps')
    mmin = total_timesteps['min']
    mmax = total_timesteps['max']
    all_timesteps = np.geomspace(mmin, mmax, sweep_gpus)
    # You hardcoded buffer size to 5 instead of 10 for this
    sweeps = [sweep_cls(args['sweep']) for _ in range(sweep_gpus)]
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'

    from multiprocessing import Process, Queue, set_start_method
    from copy import deepcopy

    host_queues = []
    worker_queues = []
    workers = []
    worker_args = []
    set_start_method('spawn')
    for i in range(sweep_gpus):
        q_host = Queue()
        q_worker = Queue()
        w = Process(
            target=_sweep_worker,
            args=(env_name, q_host, q_worker, f'cuda:{i}')
        )
        w.start()
        host_queues.append(q_host)
        worker_queues.append(q_worker)
        args = deepcopy(args)
        worker_args.append(args)

    for w in range(sweep_gpus):
        args = worker_args[w]
        sweeps[w].suggest(args)
        args['train']['total_timesteps'] = all_timesteps[w]
        worker_queues[w].put(args)

    runs = 0

    suggestion = deepcopy(args)
    while runs < args['max_runs']:
        for w in range(sweep_gpus):
            args = worker_args[w]
            if host_queues[w].empty():
                continue

            all_logs = host_queues[w].get(timeout=0)
            if not all_logs:
                continue

            all_logs = [e for e in all_logs if target_key in e]
            scores = downsample([log[target_key] for log in all_logs], points_per_run)
            times = downsample([log['uptime'] for log in all_logs], points_per_run)
            steps = downsample([log['agent_steps'] for log in all_logs], points_per_run)
            #costs = np.stack([times, steps], axis=1)
            costs = times
            timesteps = [log['agent_steps'] for log in all_logs]
            timesteps = downsample(timesteps, points_per_run)
            for score, cost, timestep in zip(scores, costs, timesteps):
                args['train']['total_timesteps'] = timestep
                sweeps[w].observe(args, score, cost)

            runs += 1

            sweeps[w].suggest(args)
            args['train']['total_timesteps'] = all_timesteps[w]
            worker_queues[w].put(args)

    print('Done')

def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args['wandb'] and not args['neptune']:
        raise pufferlib.APIUsageError('Sweeps require either wandb or neptune')
    args['no_model_upload'] = True  # Uploading trained model during sweep crashed wandb

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'
    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # In the first run, skip sweep and use the train args specified in the config
        if i > 0:
            sweep.suggest(args)

        all_logs = train(env_name, args=args, should_stop_early=stop_if_loss_nan)
        all_logs = [e for e in all_logs if target_key in e]

        if not all_logs:
            sweep.observe(args, 0, 0, is_failure=True)
            continue

        total_timesteps = args['train']['total_timesteps']

        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log['agent_steps'] for log in all_logs], points_per_run)
        timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)

        if len(timesteps) > 0 and timesteps[-1] < 0.7 * total_timesteps:  # 0.7 is arbitrary
            s = scores.pop()
            c = costs.pop()
            args['train']['total_timesteps'] = timesteps.pop()
            sweep.observe(args, s, c, is_failure=True)

        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args['train']['total_timesteps'] = total_timesteps

def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config(env_name)
    #vecenv = vecenv or load_env(env_name, args)
    #policy = policy or load_policy(args, vecenv)

    #train_config = dict(**args['train'], env=args['env_name'], tag=args['tag'])
    train_config = dict(**args['train'])
    #pufferl = PuffeRL(train_config, vecenv, policy, neptune=args['neptune'], wandb=args['wandb'])
    pufferl = PuffeRL(train_config)

    # Warmup
    for _ in range(5):
        stats = pufferl.evaluate()
        pufferl.train()

    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()      # optional, clears cuBLAS heuristics
    torch.compiler.cudagraph_mark_step_begin()  # forces any pending CUDA graph/JIT work to finish
    torch.cuda.synchronize()

    pufferl.evaluate()
    pufferl.train()
    torch.cuda.synchronize()

    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity, schedule

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            skip_first=15,
            wait=5,
            warmup=10,
            active=5,
            repeat=1
        ),
    ) as prof:
        for _ in range(35):  # 15 + 5 + 10 + 5 
            with record_function("full_step"):
                pufferl.evaluate()
                pufferl.train()
            prof.step()

    '''
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function("full_step"):
            for _ in range(5):
                stats = pufferl.evaluate()
                pufferl.train()
                prof.step()
    '''

    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="self_cpu_time_total",
        row_limit=50,
    ))
    #prof.export_chrome_trace("trace.json")

def export(args=None, env_name=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    args['vec'] = dict(backend='Serial', num_envs=1)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
    path = f'{args["env_name"]}_weights.bin'
    weights = np.concatenate(weights)
    weights.tofile(path)
    print(f'Saved {len(weights)} weights to {path}')

def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    env_name = args['env_name']
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
 
def load_env(env_name, args):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    return pufferlib.vector.make(make_env, env_kwargs=args['env'], **args['vec'])

def load_policy(args, vecenv, env_name=''):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)

    device = args['train']['device']
    policy_cls = getattr(env_module.torch, args['policy_name'])
    policy = policy_cls(vecenv.driver_env, **args['policy'])

    '''
    rnn_name = args['rnn_name']
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])
        policy = rnn_cls(vecenv.driver_env, policy, **args['policy'])
    '''
    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['neptune']:
            path = NeptuneLogger(args, load_id, mode='read-only').download()
        elif args['wandb']:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError('No run id provided for eval')

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
        #state_path = os.path.join(*load_path.split('/')[:-1], 'state.pt')
        #optim_state = torch.load(state_path)['optimizer_state_dict']
        #pufferl.optimizer.load_state_dict(optim_state)

    return policy

def load_config(env_name, parser=None):
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')
    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split(): break
        else:
            raise pufferlib.APIUsageError('No config for env_name {}'.format(env_name))

    return process_config(p, parser=parser)

def load_config_file(file_path, fill_in_default=True, parser=None):
    if not os.path.exists(file_path):
        raise pufferlib.APIUsageError('No config file found')

    config_paths = [file_path]

    if fill_in_default:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
        # Process the puffer defaults first
        config_paths.insert(0, os.path.join(puffer_dir, 'config/default.ini'))

    p = configparser.ConfigParser()
    p.read(config_paths)

    return process_config(p, parser=parser)

def make_parser():
    '''Creates the argument parser with default PufferLib arguments.'''
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandb/Neptune run')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--max-runs', type=int, default=1200, help='Max number of sweep runs')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--neptune', action='store_true', help='Use neptune for logging')
    parser.add_argument('--neptune-name', type=str, default='pufferai')
    parser.add_argument('--neptune-project', type=str, default='ablations')
    parser.add_argument('--no-model-upload', action='store_true', help='Do not upload models to wandb or neptune')
    parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
    parser.add_argument('--sweep-gpus', type=int, default=-1, help='multigpu sweeps')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    return parser

def process_config(config, parser=None):
    if parser is None:
        parser = make_parser()

    parser.description = f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]' \
        ' demo options. Shows valid args for your env and policy'

    def auto_type(value):
        """Type inference for numeric args that use 'auto' as a default value"""
        if value == 'auto': return value
        if value.isnumeric(): return int(value)
        return float(value)

    for section in config.sections():
        for key in config[section]:
            try:
                value = ast.literal_eval(config[section][key])
            except:
                value = config[section][key]

            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(
                fmt.replace('_', '-'),
                default=value,
                type=auto_type if value == 'auto' else type(value)
            )

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args['train']['env'] = args['env_name'] or ''  # for trainer dashboard
    args['train']['use_rnn'] = args['rnn_name'] is not None
    return args

def main():
    err = 'Usage: puffer [train, eval, sweep, autotune, profile, export] [env_name] [optional args]. --help for more info'
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == 'train':
        train(env_name=env_name)
    elif mode == 'eval':
        eval(env_name=env_name)
    elif mode == 'sweep':
        sweep(env_name=env_name)
    elif mode == 'multisweep':
        multisweep(env_name=env_name)
    elif mode == 'paretosweep':
        paretosweep(env_name=env_name)
    elif mode == 'autotune':
        autotune(env_name=env_name)
    elif mode == 'profile':
        profile(env_name=env_name)
    elif mode == 'export':
        export(env_name=env_name)
    elif mode == 'check':
        check(env_name=env_name)
    elif mode == 'sps':
        sps(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)

if __name__ == '__main__':
    main()
