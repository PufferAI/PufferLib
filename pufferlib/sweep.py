import random
import math
import warnings
import os
import json
from copy import deepcopy
from contextlib import contextmanager

import numpy as np
import pufferlib

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, PolynomialKernel, ScaleKernel, AdditiveKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from scipy.stats.qmc import Sobol
from scipy.spatial import KDTree

EPSILON = 1e-6

@contextmanager
def default_tensor_dtype(dtype):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old_dtype)

class Space:
    def __init__(self, min, max, scale, mean, is_integer=False):
        self.min = min
        self.max = max
        self.scale = scale
        self.mean = mean # TODO: awkward to have just this normalized
        self.norm_min = self.normalize(min)
        self.norm_max = self.normalize(max)
        self.norm_mean = self.normalize(mean)
        self.is_integer = is_integer

class Linear(Space):
    def __init__(self, min, max, scale, mean, is_integer=False):
        if scale == 'auto':
            scale = 0.5

        super().__init__(min, max, scale, mean, is_integer)

    def normalize(self, value):
        #assert isinstance(value, (int, float))
        zero_one = (value - self.min)/(self.max - self.min)
        return 2*zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1)/2
        value = zero_one * (self.max - self.min) + self.min
        if self.is_integer:
            value = round(value)
        return value

class Pow2(Space):
    def __init__(self, min, max, scale, mean, is_integer=False):
        if scale == 'auto':
            scale = 0.5
            #scale = 2 / (np.log2(max) - np.log2(min))

        super().__init__(min, max, scale, mean, is_integer)

    def normalize(self, value):
        #assert isinstance(value, (int, float))
        #assert value != 0.0
        zero_one = (math.log(value, 2) - math.log(self.min, 2))/(math.log(self.max, 2) - math.log(self.min, 2))
        return 2*zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1)/2
        log_spaced = zero_one*(math.log(self.max, 2) - math.log(self.min, 2)) + math.log(self.min, 2)
        rounded = round(log_spaced)
        return 2 ** rounded

class Log(Space):
    base: int = 10

    def __init__(self, min, max, scale, mean, is_integer=False):
        if scale == 'time':
            # TODO: Set scaling param intuitively based on number of jumps from min to max
            scale = 1 / (np.log2(max) - np.log2(min))
        elif scale == 'auto':
            scale = 0.5

        super().__init__(min, max, scale, mean, is_integer)

    def normalize(self, value):
        #assert isinstance(value, (int, float))
        #assert value != 0.0
        zero_one = (math.log(value, self.base) - math.log(self.min, self.base))/(math.log(self.max, self.base) - math.log(self.min, self.base))
        return 2*zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1)/2
        log_spaced = zero_one*(math.log(self.max, self.base) - math.log(self.min, self.base)) + math.log(self.min, self.base)
        value = self.base ** log_spaced
        if self.is_integer:
            value = round(value)
        return value

class Logit(Space):
    base: int = 10

    def __init__(self, min, max, scale, mean, is_integer=False):
        if scale == 'auto':
            scale = 0.5

        super().__init__(min, max, scale, mean, is_integer)

    def normalize(self, value):
        #assert isinstance(value, (int, float))
        #assert value != 0.0
        #assert value != 1.0
        zero_one = (math.log(1-value, self.base) - math.log(1-self.min, self.base))/(math.log(1-self.max, self.base) - math.log(1-self.min, self.base))
        return 2*zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1)/2
        log_spaced = zero_one*(math.log(1-self.max, self.base) - math.log(1-self.min, self.base)) + math.log(1-self.min, self.base)
        return 1 - self.base**log_spaced

def _params_from_puffer_sweep(sweep_config):
    param_spaces = {}
    for name, param in sweep_config.items():
        if name in ('method', 'metric', 'goal', 'downsample', 'use_gpu', 'prune_pareto',
                    'state_file', 'override_file'):
            continue

        assert isinstance(param, dict)
        if any(isinstance(param[k], dict) for k in param):
            param_spaces[name] = _params_from_puffer_sweep(param)
            continue
 
        assert 'distribution' in param
        distribution = param['distribution']
        search_center = param['mean']
        kwargs = dict(
            min=param['min'],
            max=param['max'],
            scale=param['scale'],
            mean=search_center,
        )
        if distribution == 'uniform':
            space = Linear(**kwargs)
        elif distribution == 'int_uniform':
            space = Linear(**kwargs, is_integer=True)
        elif distribution == 'uniform_pow2':
            space = Pow2(**kwargs, is_integer=True)
        elif distribution == 'log_normal':
            space = Log(**kwargs)
        elif distribution == 'logit_normal':
            space = Logit(**kwargs)
        else:
            raise ValueError(f'Invalid distribution: {distribution}')

        param_spaces[name] = space

    return param_spaces

class Hyperparameters:
    def __init__(self, config, verbose=True):
        self.spaces = _params_from_puffer_sweep(config)
        self.flat_spaces = dict(pufferlib.unroll_nested_dict(self.spaces))
        self.num = len(self.flat_spaces)

        self.metric = config['metric']
        goal = config['goal']
        assert goal in ('maximize', 'minimize')
        self.optimize_direction = 1 if goal == 'maximize' else -1

        self.search_centers = np.array([
            e.norm_mean for e in self.flat_spaces.values()])
        self.min_bounds = np.array([
            e.norm_min for e in self.flat_spaces.values()])
        self.max_bounds = np.array([
            e.norm_max for e in self.flat_spaces.values()])
        self.search_scales = np.array([
            e.scale for e in self.flat_spaces.values()])

        if verbose:
            print('Min random sample:')
            for name, space in self.flat_spaces.items():
                print(f'\t{name}: {space.unnormalize(max(space.norm_mean - space.scale, space.norm_min))}')

            print('Max random sample:')
            for name, space in self.flat_spaces.items():
                print(f'\t{name}: {space.unnormalize(min(space.norm_mean + space.scale, space.norm_max))}')

    def sample(self, n, mu=None, scale=1):
        if mu is None:
            mu = self.search_centers

        if len(mu.shape) == 1:
            mu = mu[None, :]

        n_input, n_dim = mu.shape
        scale = scale * self.search_scales
        mu_idxs = np.random.randint(0, n_input, n)
        samples = scale*(2*np.random.rand(n, n_dim) - 1) + mu[mu_idxs]
        return np.clip(samples, self.min_bounds, self.max_bounds)

    def from_dict(self, params):
        flat_params = dict(pufferlib.unroll_nested_dict(params))
        values = []
        for key, space in self.flat_spaces.items():
            assert key in flat_params, f'Missing hyperparameter {key}'
            val = flat_params[key]
            normed = space.normalize(val)
            values.append(normed)

        return np.array(values)

    def to_dict(self, sample, fill=None):
        params = deepcopy(self.spaces) if fill is None else fill
        self._fill(params, self.spaces, sample)
        return params

    def _fill(self, params, spaces, flat_sample, idx=0):
        for name, space in spaces.items():
            if isinstance(space, dict):
                idx = self._fill(params[name], spaces[name], flat_sample, idx=idx)
            else:
                params[name] = spaces[name].unnormalize(flat_sample[idx])
                idx += 1

        return idx

    def get_flat_idx(self, flat_key):
        return list(self.flat_spaces.keys()).index(flat_key)


def pareto_points(observations):
    if not observations:
        return [], []

    scores = np.array([e['output'] for e in observations])
    costs = np.array([e['cost'] for e in observations])

    # Sort by cost to find non-dominated points efficiently
    sorted_indices = np.argsort(costs)
    
    pareto = []
    pareto_idxs = []
    max_score_so_far = -np.inf

    for idx in sorted_indices:
        if scores[idx] > max_score_so_far + EPSILON:
            pareto.append(observations[idx])
            pareto_idxs.append(idx)
            max_score_so_far = scores[idx]

    return pareto, pareto_idxs

def prune_pareto_front(pareto, efficiency_threshold=0.5, pruning_stop_score_fraction=0.98):
    # Prune the high-cost long tail of a pareto front
    # like (score 0.99, cost 100), (score 0.991, cost 200)
    if not pareto or len(pareto) < 2:
        return pareto

    sorted_pareto = sorted(pareto, key=lambda x: x['cost'])
    scores = np.array([e['output'] for e in sorted_pareto])
    costs = np.array([e['cost'] for e in sorted_pareto])
    score_range = max(scores.max() - scores.min(), EPSILON)
    cost_range = max(costs.max() - costs.min(), EPSILON)

    max_pareto_score = scores[-1] if scores.size > 0 else -np.inf

    for i in range(len(sorted_pareto) - 1, 0, -1):
        if scores[i] < pruning_stop_score_fraction * max_pareto_score:
            break

        norm_score_gain = (scores[i] - scores[i-1]) / score_range
        norm_cost_increase = (costs[i] - costs[i-1]) / cost_range
        efficiency = norm_score_gain / (norm_cost_increase + EPSILON)

        if efficiency < efficiency_threshold:
            sorted_pareto.pop(i)
        else:
            # Stop pruning if we find an efficient point
            break

    return sorted_pareto


class Random:
    def __init__(self,
            sweep_config,
            global_search_scale = 1,
            random_suggestions = 1024,
        ):

        self.hyperparameters = Hyperparameters(sweep_config)
        self.global_search_scale = global_search_scale
        self.random_suggestions = random_suggestions
        self.success_observations = []

    def suggest(self, fill=None):
        suggestions = self.hyperparameters.sample(self.random_suggestions)
        self.suggestion = random.choice(suggestions)
        return self.hyperparameters.to_dict(self.suggestion, fill), {}

    def observe(self, hypers, score, cost, is_failure=False):
        params = self.hyperparameters.from_dict(hypers)
        self.success_observations.append(dict(
            input=hypers,
            output=score,
            cost=cost,
            is_failure=is_failure,
        ))

class ParetoGenetic:
    def __init__(self,
            sweep_config,
            global_search_scale = 1,
            suggestions_per_pareto = 1,
            bias_cost = True,
            log_bias = False,
        ):

        self.hyperparameters = Hyperparameters(sweep_config)
        self.global_search_scale = global_search_scale
        self.suggestions_per_pareto = suggestions_per_pareto
        self.bias_cost = bias_cost
        self.log_bias = log_bias
        self.success_observations = []

    def suggest(self, fill=None):
        if len(self.success_observations) == 0:
            suggestion = self.hyperparameters.search_centers
            return self.hyperparameters.to_dict(suggestion, fill), {}

        candidates, _ = pareto_points(self.success_observations)
        pareto_costs = np.array([e['cost'] for e in candidates])

        if self.bias_cost:
            if self.log_bias:
                cost_dists = np.abs(np.log(pareto_costs[:, None]) - np.log(pareto_costs[None, :]))
            else:
                cost_dists = np.abs(pareto_costs[:, None] - pareto_costs[None, :])

            cost_dists += (np.max(pareto_costs) + 1)*np.eye(len(pareto_costs)) # mask self-distance
            idx = np.argmax(np.min(cost_dists, axis=1))
            search_centers = candidates[idx]['input']
        else:
            search_centers = np.stack([e['input'] for e in candidates])

        suggestions = self.hyperparameters.sample(
            len(candidates)*self.suggestions_per_pareto, mu=search_centers)
        suggestion = suggestions[np.random.randint(0, len(suggestions))]
        return self.hyperparameters.to_dict(suggestion, fill), {}

    def observe(self, hypers, score, cost, is_failure=False):
        params = self.hyperparameters.from_dict(hypers)
        self.success_observations.append(dict(
            input=params,
            output=score,
            cost=cost,
            is_failure=is_failure,
        ))


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, x_dim):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # Matern 3/2 kernel (equivalent to Pyro's Matern32)
        matern_kernel = MaternKernel(nu=1.5, ard_num_dims=x_dim)

        # NOTE: setting this constraint changes GP behavior, including the lengthscale
        # even though the lengthscale is well within the range ... Commenting out for now.
        # lengthscale_constraint = gpytorch.constraints.Interval(0.01, 10.0)
        # matern_kernel = MaternKernel(nu=1.5, ard_num_dims=x_dim, lengthscale_constraint=lengthscale_constraint)

        linear_kernel = PolynomialKernel(power=1)
        self.covar_module = ScaleKernel(AdditiveKernel(linear_kernel, matern_kernel))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def lengthscale_range(self):
        # Get lengthscale from MaternKernel
        lengthscale = self.covar_module.base_kernel.kernels[1].lengthscale.tolist()[0]
        return min(lengthscale), max(lengthscale)

def train_gp_model(model, likelihood, mll, optimizer, train_x, train_y, training_iter=50):
    model.train()
    likelihood.train()
    model.set_train_data(inputs=train_x, targets=train_y, strict=False)

    loss = None
    for _ in range(training_iter):
        try:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            loss = loss.detach()

        except gpytorch.utils.errors.NotPSDError:
            # It's rare but it does happen. Hope it's a transient issue.
            break

    model.eval()
    likelihood.eval()
    return loss.item() if loss is not None else 0


# TODO: Eval defaults
class Protein:
    def __init__(self,
            sweep_config,
            max_suggestion_cost = 3600,
            resample_frequency = 0,
            num_random_samples = 30,
            global_search_scale = 1,
            suggestions_per_pareto = 256,
            seed_with_search_center = True,
            expansion_rate = 0.25,
            gp_training_iter = 50,
            gp_learning_rate = 0.001,
            gp_max_obs = 750,  # gp train time jumps after 800
            infer_batch_size = 4096,            
            optimizer_reset_frequency = 50,
            use_gpu = True,
            cost_param = "train/total_timesteps",
            prune_pareto = True,
        ):
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.hyperparameters = Hyperparameters(sweep_config)
        self.global_search_scale = global_search_scale
        self.suggestions_per_pareto = suggestions_per_pareto
        self.seed_with_search_center = seed_with_search_center
        self.resample_frequency = resample_frequency
        self.max_suggestion_cost = max_suggestion_cost
        self.expansion_rate = expansion_rate
        self.gp_training_iter = gp_training_iter
        self.gp_learning_rate = gp_learning_rate
        self.optimizer_reset_frequency = optimizer_reset_frequency
        self.prune_pareto = prune_pareto

        self.success_observations = []
        self.failure_observations = []
        self.suggestion_idx = 0
        self.min_score, self.max_score = math.inf, -math.inf
        self.log_c_min, self.log_c_max = math.inf, -math.inf

        # Use Sobel seq for structured random exploration
        self.sobol = Sobol(d=self.hyperparameters.num, scramble=True)
        self.num_random_samples = num_random_samples
        # NOTE: test if sobol sampling really helps
        # points_per_run = sweep_config['downsample']
        # self.num_random_samples = 3 * points_per_run * self.hyperparameters.num

        self.cost_param_idx = self.hyperparameters.get_flat_idx(cost_param)
        self.cost_random_suggestion = self.hyperparameters.search_centers[self.cost_param_idx]

        self.gp_max_obs = gp_max_obs  # train time bumps after 800?
        self.infer_batch_size = infer_batch_size

        self.state_file = sweep_config.get('state_file', 'sweep_state.json')
        self.override_file = sweep_config.get('override_file', 'override.json')

        # Use 64 bit for GP regression
        with default_tensor_dtype(torch.float64):
            # Params taken from HEBO: https://arxiv.org/abs/2012.03826
            noise_prior = LogNormalPrior(math.log(1e-2), 0.5)

            # Create dummy data for model initialization on the selected device
            dummy_x = torch.ones((1, self.hyperparameters.num), device=self.device)
            dummy_y = torch.zeros(1, device=self.device)
            # Score GP
            self.likelihood_score = GaussianLikelihood(noise_prior=deepcopy(noise_prior)).to(self.device)
            self.gp_score = ExactGPModel(dummy_x, dummy_y, self.likelihood_score, self.hyperparameters.num).to(self.device)
            self.mll_score = ExactMarginalLogLikelihood(self.likelihood_score, self.gp_score).to(self.device)
            self.score_opt = torch.optim.Adam(self.gp_score.parameters(), lr=self.gp_learning_rate, amsgrad=True)

            # Cost GP
            self.likelihood_cost = GaussianLikelihood(noise_prior=deepcopy(noise_prior)).to(self.device)
            self.gp_cost = ExactGPModel(dummy_x, dummy_y, self.likelihood_cost, self.hyperparameters.num).to(self.device)
            self.mll_cost = ExactMarginalLogLikelihood(self.likelihood_cost, self.gp_cost).to(self.device)
            self.cost_opt = torch.optim.Adam(self.gp_cost.parameters(), lr=self.gp_learning_rate, amsgrad=True)

            # Buffers for GP training and inference
            self.gp_params_buffer = torch.empty(self.gp_max_obs, self.hyperparameters.num, device=self.device)
            self.gp_score_buffer = torch.empty(self.gp_max_obs, device=self.device)
            self.gp_cost_buffer = torch.empty(self.gp_max_obs, device=self.device)
            self.infer_batch_buffer = torch.empty(self.infer_batch_size, self.hyperparameters.num, device=self.device)

        self._load_state_if_exists()

    @staticmethod
    def _json_default(obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(f'Not JSON serializable: {type(obj)}')

    def _save_state(self):
        """Save sweep state to JSON for crash recovery."""
        state = {
            'suggestion_idx': self.suggestion_idx,
            'success_observations': self.success_observations,
            'failure_observations': self.failure_observations,
            'min_score': self.min_score if self.min_score != math.inf else None,
            'max_score': self.max_score if self.max_score != -math.inf else None,
            'log_c_min': self.log_c_min if self.log_c_min != math.inf else None,
            'log_c_max': self.log_c_max if self.log_c_max != -math.inf else None,
        }
        tmp = f'{self.state_file}.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(state, f, indent=2, default=self._json_default)
            os.replace(tmp, self.state_file)
        except OSError as e:
            print(f'[Protein] Failed to save state: {e}')
            if os.path.exists(tmp):
                os.remove(tmp)

    def _load_state_if_exists(self):
        """Load state from previous run if exists (crash recovery)."""
        tmp = f'{self.state_file}.tmp'
        if os.path.exists(tmp):
            os.remove(tmp)
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            self.suggestion_idx = state.get('suggestion_idx', 0)
            self.success_observations = state.get('success_observations', [])
            self.failure_observations = state.get('failure_observations', [])
            if state.get('min_score') is not None:
                self.min_score = state['min_score']
            if state.get('max_score') is not None:
                self.max_score = state['max_score']
            if state.get('log_c_min') is not None:
                self.log_c_min = state['log_c_min']
            if state.get('log_c_max') is not None:
                self.log_c_max = state['log_c_max']
            for obs in self.success_observations + self.failure_observations:
                if isinstance(obs['input'], list):
                    obs['input'] = np.array(obs['input'])
            print(f'[Protein] Resumed from {self.state_file}: {len(self.success_observations)} obs, idx={self.suggestion_idx}')
        except (json.JSONDecodeError, KeyError, FileNotFoundError, OSError) as e:
            print(f'[Protein] Failed to load state: {e}')

    def _check_override(self):
        """Check for override. Returns None, 'skip', or params dict."""
        if not os.path.exists(self.override_file):
            return None
        tmp = f'{self.override_file}.tmp'
        try:
            with open(self.override_file) as f:
                data = json.load(f)
            if 'suggestions' not in data or not data['suggestions']:
                os.remove(self.override_file)
                return None
            suggestion = data['suggestions'].pop(0)
            if data['suggestions']:
                with open(tmp, 'w') as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, self.override_file)
            else:
                os.remove(self.override_file)

            # Check for skip flag (spacer runs)
            if suggestion.get('skip', False):
                reason = suggestion.get('reason', 'Spacer - skip override')
                print(f'[Protein] SKIP: {reason}')
                return 'skip'

            reason = suggestion.get('reason', 'No reason provided')
            print(f'[Protein] OVERRIDE: {reason}')
            return suggestion.get('params', suggestion)
        except (json.JSONDecodeError, KeyError) as e:
            print(f'[Protein] Invalid override file: {e}')
            if os.path.exists(self.override_file):
                os.remove(self.override_file)
            return None
        except OSError as e:
            print(f'[Protein] Failed to update override file: {e}')
            if os.path.exists(tmp):
                os.remove(tmp)
            return None

    def _apply_params_to_fill(self, fill, params):
        """Apply param dict to fill in place. Modifies fill directly."""
        for key, value in pufferlib.unroll_nested_dict(params):
            parts = key.split('/')
            try:
                target = fill
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
            except KeyError:
                print(f'[Protein] Override key not found: {key}')

    def _filter_near_duplicates(self, inputs, duplicate_threshold=EPSILON):
        if len(inputs) < 2:
            return np.arange(len(inputs))

        tree = KDTree(inputs)
        to_keep = np.ones(len(inputs), dtype=bool)
        # Iterate from most recent to oldest
        for i in range(len(inputs) - 1, -1, -1):
            if to_keep[i]:
                nearby_indices = tree.query_ball_point(inputs[i], r=duplicate_threshold)
                # Exclude the point itself from being marked for removal
                nearby_indices.remove(i)
                if nearby_indices:
                    to_keep[nearby_indices] = False

        return np.where(to_keep)[0]

    def _sample_observations(self, max_size=None, recent_ratio=0.5):
        if not self.success_observations:
            return []

        # Update the stats using the full data
        y = np.array([e['output'] for e in self.success_observations])
        self.min_score, self.max_score = y.min(), y.max()

        c = np.array([e['cost'] for e in self.success_observations])
        log_c = np.log(np.maximum(c, EPSILON))
        self.log_c_min, self.log_c_max = log_c.min(), log_c.max()

        params = np.array([e['input'] for e in self.success_observations])
        dedup_indices = self._filter_near_duplicates(params)
        observations = [self.success_observations[i] for i in dedup_indices]

        if max_size is None:
            max_size = self.gp_max_obs

        if len(observations) <= max_size:
            return observations
        
        recent_size = int(recent_ratio*max_size)
        recent_obs = observations[-recent_size:]
        older_obs = observations[:-recent_size]
        num_to_sample = max_size - recent_size
        random_sample_obs = random.sample(older_obs, num_to_sample)

        return random_sample_obs + recent_obs

    def _train_gp_models(self):
        if not self.success_observations:
            return 0, 0
        
        sampled_observations = self._sample_observations(max_size=self.gp_max_obs)
        num_sampled = len(sampled_observations)

        # Prepare tensors using pre-allocated buffers
        params = np.array([e['input'] for e in sampled_observations])
        params_tensor = self.gp_params_buffer[:num_sampled]
        params_tensor.copy_(torch.from_numpy(params))

        # Normalized scores and costs
        y = np.array([e['output'] for e in sampled_observations])
        y_norm = (y - self.min_score) / (np.abs(self.max_score - self.min_score) + EPSILON)
        y_norm_tensor = self.gp_score_buffer[:num_sampled]
        y_norm_tensor.copy_(torch.from_numpy(y_norm))

        c = np.array([e['cost'] for e in sampled_observations])
        log_c = np.log(np.maximum(c, EPSILON)) # Ensure log is not taken on zero
        log_c_norm = (log_c - self.log_c_min) / (self.log_c_max - self.log_c_min + EPSILON)
        log_c_norm_tensor = self.gp_cost_buffer[:num_sampled]
        log_c_norm_tensor.copy_(torch.from_numpy(log_c_norm))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", gpytorch.utils.warnings.NumericalWarning)
            score_loss = train_gp_model(self.gp_score, self.likelihood_score, self.mll_score, self.score_opt, params_tensor, y_norm_tensor, training_iter=self.gp_training_iter)
            cost_loss = train_gp_model(self.gp_cost, self.likelihood_cost, self.mll_cost, self.cost_opt, params_tensor, log_c_norm_tensor, training_iter=self.gp_training_iter)

        return score_loss, cost_loss

    def _gp_suggest(self, fill):
        """Generate suggestion using Gaussian Process optimization."""
        info = {}

        score_loss, cost_loss = self._train_gp_models()

        if self.optimizer_reset_frequency and self.suggestion_idx % self.optimizer_reset_frequency == 0:
            print(f'Resetting GP optimizers at suggestion {self.suggestion_idx}')
            self.score_opt = torch.optim.Adam(self.gp_score.parameters(), lr=self.gp_learning_rate, amsgrad=True)
            self.cost_opt = torch.optim.Adam(self.gp_cost.parameters(), lr=self.gp_learning_rate, amsgrad=True)

        candidates, pareto_idxs = pareto_points(self.success_observations)

        if self.prune_pareto:
            candidates = prune_pareto_front(candidates)

        ### Sample suggestions
        search_centers = np.stack([e['input'] for e in candidates])
        suggestions = self.hyperparameters.sample(
            len(candidates)*self.suggestions_per_pareto, mu=search_centers)

        dedup_indices = self._filter_near_duplicates(suggestions)
        suggestions = suggestions[dedup_indices]

        if len(suggestions) == 0:
            # Fallback to search center if all suggestions are filtered
            suggestion = self.hyperparameters.search_centers
            return self.hyperparameters.to_dict(suggestion, fill), info

        ### Predict scores and costs
        # Batch predictions to avoid GPU OOM for large number of suggestions
        gp_y_norm_list, gp_log_c_norm_list = [], []

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4), warnings.catch_warnings():
            warnings.simplefilter("ignore", gpytorch.utils.warnings.NumericalWarning)

            # Create a reusable buffer on the device to avoid allocating a huge tensor
            for i in range(0, len(suggestions), self.infer_batch_size):
                batch_numpy = suggestions[i:i+self.infer_batch_size]
                current_batch_size = len(batch_numpy)

                # Use a slice of the buffer if the current batch is smaller
                batch_tensor = self.infer_batch_buffer[:current_batch_size]
                batch_tensor.copy_(torch.from_numpy(batch_numpy))

                try:
                    # Score and cost prediction
                    pred_y_mean = self.likelihood_score(self.gp_score(batch_tensor)).mean.cpu()
                    pred_c_mean = self.likelihood_cost(self.gp_cost(batch_tensor)).mean.cpu()

                except RuntimeError:
                    # Handle numerical errors during GP prediction
                    pred_y_mean = torch.zeros(current_batch_size)
                    pred_c_mean = torch.zeros(current_batch_size)

                gp_y_norm_list.append(pred_y_mean.cpu())
                gp_log_c_norm_list.append(pred_c_mean.cpu())

                del pred_y_mean, pred_c_mean

        # Concatenate results from all batches
        gp_y_norm = torch.cat(gp_y_norm_list).numpy()
        gp_log_c_norm = torch.cat(gp_log_c_norm_list).numpy()

        # Unlinearize
        gp_y = gp_y_norm*(self.max_score - self.min_score) + self.min_score
        gp_log_c = gp_log_c_norm*(self.log_c_max - self.log_c_min) + self.log_c_min
        gp_c = np.exp(gp_log_c)

        max_c_mask = gp_c < self.max_suggestion_cost

        target = (1 + self.expansion_rate)*np.random.rand()
        weight = 1 - abs(target - gp_log_c_norm)

        # NOTE: Tried upper confidence bounds, but it did more harm because gp was noisy
        score = gp_y_norm

        suggestion_scores = self.hyperparameters.optimize_direction * max_c_mask * (
                score * weight)

        best_idx = np.argmax(suggestion_scores)
        info = dict(
            cost = gp_c[best_idx].item(),
            score = gp_y[best_idx].item(),
            rating = suggestion_scores[best_idx].item(),
            score_loss = score_loss,
            cost_loss = cost_loss,
            score_lengthscale = self.gp_score.lengthscale_range,
            cost_lengthscale = self.gp_cost.lengthscale_range,
        )
        print('Predicted -- ',
            f'Score: {info["score"]:.3f}',
            f'Cost: {info["cost"]:.3f}',
            f'Rating: {info["rating"]:.3f}',
        )

        best = suggestions[best_idx]
        return self.hyperparameters.to_dict(best, fill), info

    def suggest(self, fill):
        info = {}
        self.suggestion_idx += 1
        override = self._check_override()

        # Always generate a suggestion (GP, Sobol, or search center)
        if len(self.success_observations) == 0 and self.seed_with_search_center:
            suggestion = self.hyperparameters.search_centers
            result = self.hyperparameters.to_dict(suggestion, fill)

        elif len(self.success_observations) < self.num_random_samples:
            # Sobol sequence for early exploration
            zero_one = self.sobol.random(1)[0]
            suggestion = 2*zero_one - 1  # Scale from [0, 1) to [-1, 1)
            cost_suggestion = self.cost_random_suggestion + 0.1 * np.random.randn()
            suggestion[self.cost_param_idx] = np.clip(cost_suggestion, -1, 1)  # limit the cost
            result = self.hyperparameters.to_dict(suggestion, fill)

        elif self.resample_frequency and self.suggestion_idx % self.resample_frequency == 0:
            # Resample from pareto front
            candidates, _ = pareto_points(self.success_observations)
            suggestions = np.stack([e['input'] for e in candidates])
            best_idx = np.random.randint(0, len(candidates))
            best = suggestions[best_idx]
            result = self.hyperparameters.to_dict(best, fill)

        else:
            # Full GP suggestion
            result, info = self._gp_suggest(fill)

        # Apply override ON TOP of generated suggestion
        if override == 'skip':
            info['skip'] = True
        elif override:
            self._apply_params_to_fill(result, override)
            info['override'] = True

        return result, info

    def observe(self, hypers, score, cost, is_failure=False):
        params = self.hyperparameters.from_dict(hypers)
        new_observation = dict(
            input=params,
            output=score,
            cost=cost,
            is_failure=is_failure,
        )

        if is_failure or not np.isfinite(score) or np.isnan(score):
            new_observation['is_failure'] = True
            self.failure_observations.append(new_observation)
            self._save_state()
            return

        if self.success_observations:
            success_params = np.stack([e['input'] for e in self.success_observations])
            dist = np.linalg.norm(params - success_params, axis=1)
            same = np.where(dist < EPSILON)[0]
            if len(same) > 0:
                self.success_observations[same[0]] = new_observation
                self._save_state()
                return

        # Ignore obs that are below the minimum cost
        if params[self.cost_param_idx] <= -1:
            return

        self.success_observations.append(new_observation)
        self._save_state()


def read_sweep_results(state_file, sweep_config, sort_by='score'):
    """
    Load sweep results as user/agent-readable dicts.

    Args:
        state_file: Path to {project}_sweep.json
        sweep_config: The 'sweep' section from load_config()
        sort_by: 'score' (descending), 'cost' (ascending), or None

    Returns:
        List of dicts: [{'params': {...}, 'score': float, 'cost': float}, ...]
    """
    with open(state_file) as f:
        state = json.load(f)

    hyperparams = Hyperparameters(sweep_config, verbose=False)

    results = []
    for obs in state.get('success_observations', []):
        input_vec = np.array(obs['input'])

        if len(input_vec) != hyperparams.num:
            raise ValueError(
                f"State file has {len(input_vec)} dimensions but config has {hyperparams.num}. "
                f"Config may have changed since sweep started."
            )

        params = hyperparams.to_dict(input_vec)
        flat_params = dict(pufferlib.unroll_nested_dict(params))

        results.append({
            'params': flat_params,
            'score': obs['output'],
            'cost': obs['cost'],
        })

    if sort_by == 'score':
        results.sort(key=lambda x: x['score'], reverse=True)
    elif sort_by == 'cost':
        results.sort(key=lambda x: x['cost'])

    return results


def create_override(override_file, suggestions, reason=None):
    """
    Inject hyperparams into next sweep run. Use real values, not normalized.

    Args:
        override_file: Path to write
        suggestions: List of dicts, e.g. [{'train/learning_rate': 0.001}]
        reason: List of strings (same length as suggestions), or None
    """
    if reason is None:
        reason = [None] * len(suggestions)
    if len(reason) != len(suggestions):
        raise ValueError(f"Got {len(suggestions)} suggestions but {len(reason)} reasons")

    data = {
        'suggestions': [
            {
                'params': s,
                'reason': r or 'Programmatic override'
            }
            for s, r in zip(suggestions, reason)
        ]
    }

    tmp = f'{override_file}.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, override_file)
    except OSError:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
