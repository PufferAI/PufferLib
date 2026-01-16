import random
import math
import warnings
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
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from scipy.spatial import KDTree
from sklearn.linear_model import LogisticRegression

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
    def __init__(self, min, max, scale, is_integer=False):
        self.min = min
        self.max = max
        self.scale = scale
        self.norm_min = self.normalize(min)
        self.norm_max = self.normalize(max)
        # Since min/max are normalized from -1 to 1, just use 0 as a mean
        self.norm_mean = 0
        self.is_integer = is_integer

class Linear(Space):
    def __init__(self, min, max, scale, is_integer=False):
        if scale == 'auto':
            scale = 0.5

        super().__init__(min, max, scale, is_integer)

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
    def __init__(self, min, max, scale, is_integer=False):
        if scale == 'auto':
            scale = 0.5
            #scale = 2 / (np.log2(max) - np.log2(min))

        super().__init__(min, max, scale, is_integer)

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

    def __init__(self, min, max, scale, is_integer=False):
        if scale == 'time':
            # TODO: Set scaling param intuitively based on number of jumps from min to max
            scale = 1 / (np.log2(max) - np.log2(min))
        elif scale == 'auto':
            scale = 0.5

        super().__init__(min, max, scale, is_integer)

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

    def __init__(self, min, max, scale, is_integer=False):
        if scale == 'auto':
            scale = 0.5

        super().__init__(min, max, scale, is_integer)

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

def _params_from_puffer_sweep(sweep_config, only_include=None):
    param_spaces = {}

    if 'sweep_only' in sweep_config:
        only_include = [p.strip() for p in sweep_config['sweep_only'].split(',')]

    for name, param in sweep_config.items():
        if name in ('method', 'metric', 'metric_distribution', 'goal', 'downsample', 'use_gpu', 'prune_pareto',
                    'sweep_only', 'max_suggestion_cost', 'early_stop_quantile'):
            continue

        assert isinstance(param, dict)
        if any(isinstance(param[k], dict) for k in param):
            param_spaces[name] = _params_from_puffer_sweep(param, only_include)
            continue
 
        if only_include and not any(k in name for k in only_include):
            continue

        assert 'distribution' in param
        distribution = param['distribution']
        kwargs = dict(
            min=param['min'],
            max=param['max'],
            scale=param['scale'],
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
        keys = list(self.flat_spaces.keys())
        return keys.index(flat_key) if flat_key in keys else None

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

    for i in range(len(sorted_pareto) - 1, 1, -1):
        if scores[i-1] < pruning_stop_score_fraction * max_pareto_score:
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


class RobustLogCostModel:
    """
    Fits Score ~ A + B * log(Cost) using Quantile Regression (Median)
    and provides a cost-only threshold for early stopping.
    """
    def __init__(self, quantile=0.3, min_num_samples=30):
        self.quantile = quantile  # 0.5 = Median regression
        self.min_num_samples = min_num_samples
        self.is_fitted = False
        self.A = None
        self.B = None
        self.max_score = None
        self.max_cost = None
        self.upper_cost_threshold = None

    def _quantile_loss(self, params, x, y, q):
        # Pinball loss function for quantile regression
        a, b = params
        y_pred = a + b * x
        residuals = y - y_pred
        return np.sum(np.maximum(q * residuals, (q - 1) * residuals))

    def fit(self, observations, upper_cost_threshold=None):
        self.is_fitted = False
        scores = np.array([e['output'] for e in observations])
        costs = np.array([e['cost'] for e in observations])
        self.max_score = scores.max()
        self.upper_cost_threshold = upper_cost_threshold or costs.max()

        valid_indices = (costs > EPSILON) & np.isfinite(scores)
        if np.sum(valid_indices) < self.min_num_samples:
            return

        y = scores[valid_indices]
        c = costs[valid_indices]
        x_log_c = np.log(c)

        # Initial guess using standard Polyfit (OLS) just to get in the ballpark
        try:
            b_init, a_init = np.polyfit(x_log_c, y, 1)
        except np.linalg.LinAlgError:
            # Fallback guess
            b_init, a_init = 0.0, np.mean(y)

        # Minimize the Quantile Loss (L1 for median)
        res = minimize(
            self._quantile_loss, 
            x0=[a_init, b_init], 
            args=(x_log_c, y, self.quantile),
            method='Nelder-Mead', # Robust solver for non-differentiable functions
            bounds=[(None, None), (0, None)] # B should be positive
        )
        
        self.A, self.B = res.x
        self.is_fitted = True

    def get_threshold(self, cost, min_cost_fraction=0.3, abs_min_cost=10):
        if not self.is_fitted or self.upper_cost_threshold is None:
            return -np.inf

        # NOTE: min_allowed_cost seems vary a lot from env to env, so dynamically set here
        min_allowed_cost = self.upper_cost_threshold * min_cost_fraction + abs_min_cost
        if cost < min_allowed_cost:
            return -np.inf

        # Stop long long train runs that don't do very well enough
        if cost > 1.2 * self.upper_cost_threshold:
            return 0.9 * self.max_score

        return self.A + self.B * np.log(cost)


# TODO: Eval defaults
class Protein:
    def __init__(self,
            sweep_config,
            max_suggestion_cost = 3600,
            resample_frequency = 0,
            num_random_samples = 10,
            num_keep_top_obs = 5,
            global_search_scale = 1,
            suggestions_per_pareto = 256,
            expansion_rate = 0.1,
            gp_training_iter = 50,
            gp_learning_rate = 0.001,
            gp_max_obs = 750,  # gp train time jumps after 800
            infer_batch_size = 4096,            
            optimizer_reset_frequency = 50,
            use_gpu = True,
            cost_param = "train/total_timesteps",
            prune_pareto = True,
        ):
        # Process sweep config. NOTE: sweep_config takes precedence. It's not good.
        _use_gpu = sweep_config['use_gpu'] if 'use_gpu' in sweep_config else use_gpu
        _prune_pareto = sweep_config['prune_pareto'] if 'prune_pareto' in sweep_config else prune_pareto
        _max_suggestion_cost = sweep_config['max_suggestion_cost'] if 'max_suggestion_cost' in sweep_config else max_suggestion_cost

        self.device = torch.device("cuda:0" if _use_gpu and torch.cuda.is_available() else "cpu")
        self.hyperparameters = Hyperparameters(sweep_config)
        self.metric_distribution = sweep_config['metric_distribution']
        self.global_search_scale = global_search_scale
        self.suggestions_per_pareto = suggestions_per_pareto
        self.resample_frequency = resample_frequency
        self.max_suggestion_cost = _max_suggestion_cost
        self.expansion_rate = expansion_rate
        self.gp_training_iter = gp_training_iter
        self.gp_learning_rate = gp_learning_rate
        self.optimizer_reset_frequency = optimizer_reset_frequency
        self.prune_pareto = _prune_pareto

        self.success_observations = []
        self.failure_observations = []
        self.num_keep_top_obs = num_keep_top_obs
        self.top_observations = []

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
        self.cost_random_suggestion = None
        if self.cost_param_idx is not None:
            self.cost_random_suggestion = -0.8  # In norm cost space. Make arg if necessary
        self.target_cost_ratio = []

        self.gp_max_obs = gp_max_obs  # train time bumps after 800?
        self.infer_batch_size = infer_batch_size

        # Probably useful only when downsample=1 and each run is expensive.
        self.use_success_prob = sweep_config['downsample'] == 1
        self.success_classifier = LogisticRegression(class_weight='balanced')

        # This model is conservative. Aggressive early stopping interferes with and hampers GP model learning.
        self.stop_threshold_model = RobustLogCostModel(quantile=sweep_config['early_stop_quantile'])
        self.upper_cost_threshold = -np.inf

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

        observations = self.success_observations.copy()

        # Update the stats using the full data
        y = np.array([e['output'] for e in observations])
        self.min_score, self.max_score = y.min(), y.max()

        c = np.array([e['cost'] for e in observations])
        log_c = np.log(np.maximum(c, EPSILON))
        self.log_c_min = log_c.min()
        self.log_c_max = np.quantile(log_c, 0.97)  # Make it less sensitive to outlier points

        # When the data is scare, also use failed observations
        if len(observations) < 100 and self.failure_observations:
            # Give the min score for the failed obs, so this value will keep changing.
            for e in self.failure_observations:
                e['output'] = self.min_score
            
            # NOTE: the order of obs matters since recent obs are always fed into gp training
            # So, putting the failure obs first.
            observations = self.failure_observations + observations

        params = np.array([np.append(e['input'], [e['output'], e['cost']]) for e in observations])
        dedup_indices = self._filter_near_duplicates(params)
        observations = [observations[i] for i in dedup_indices]

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

    def _get_top_obs_params(self):
        if not self.top_observations:
            return np.array([])
        
        params = np.array([e['input'] for e in self.top_observations])
        if self.cost_param_idx is None:
            return params

        # Add the same params with less cost to the search center, and not the original
        original_costs_norm = params[:, self.cost_param_idx]

        params_1 = np.copy(params)
        cost_norm_1 = original_costs_norm - (original_costs_norm - (-1)) / 2
        params_1[:, self.cost_param_idx] = cost_norm_1
        params_2 = np.copy(params)
        cost_norm_2 = original_costs_norm - (original_costs_norm - (-1)) / 3
        params_2[:, self.cost_param_idx] = cost_norm_2

        return np.vstack([params_1, params_2])

    def _sample_target_cost_ratio(self, expansion_rate, target_ratios=(0.16, 0.32, 0.48, 0.64, 0.8, 1.0)):
        if not self.target_cost_ratio:
            self.target_cost_ratio = list(target_ratios)
            random.shuffle(self.target_cost_ratio)
        target_ratio = np.clip(self.target_cost_ratio.pop() + 0.1 * np.random.randn(), 0, 1)
        return (1 + expansion_rate) * target_ratio

    def suggest(self, fill):
        info = {}
        self.suggestion_idx += 1
        
        # NOTE: Changed pufferl to use the train args, NOT the sweep hyperparam search center
        # if len(self.success_observations) == 0 and self.seed_with_search_center:
        #     suggestion = self.hyperparameters.search_centers
        #     return self.hyperparameters.to_dict(suggestion, fill), info

        if self.suggestion_idx <= self.num_random_samples:
            # Suggest the next point in the Sobol sequence
            zero_one = self.sobol.random(1)[0]
            suggestion = 2*zero_one - 1  # Scale from [0, 1) to [-1, 1)
            if self.cost_param_idx is not None:
                cost_suggestion = self.cost_random_suggestion + 0.1 * np.random.randn()
                suggestion[self.cost_param_idx] = np.clip(cost_suggestion, -1, 1)  # limit the cost
            return self.hyperparameters.to_dict(suggestion, fill), info

        elif self.resample_frequency and self.suggestion_idx % self.resample_frequency == 0:
            candidates, _ = pareto_points(self.success_observations)
            suggestions = np.stack([e['input'] for e in candidates])
            best_idx = np.random.randint(0, len(candidates))
            best = suggestions[best_idx]
            return self.hyperparameters.to_dict(best, fill), info

        score_loss, cost_loss = self._train_gp_models()

        if self.optimizer_reset_frequency and self.suggestion_idx % self.optimizer_reset_frequency == 0:
            print(f'Resetting GP optimizers at suggestion {self.suggestion_idx}')
            self.score_opt = torch.optim.Adam(self.gp_score.parameters(), lr=self.gp_learning_rate, amsgrad=True)
            self.cost_opt = torch.optim.Adam(self.gp_cost.parameters(), lr=self.gp_learning_rate, amsgrad=True)
       
        pareto_front, pareto_idxs = pareto_points(self.success_observations)
        pruned_front = prune_pareto_front(pareto_front)
        pareto_observations = pruned_front if self.prune_pareto else pareto_front

        # Use the max cost from the pruned pareto to avoid inefficiently long runs
        if self.upper_cost_threshold < 0:
            self.upper_cost_threshold = pruned_front[-1]['cost']
        # Try to change the threshold slowly
        elif self.upper_cost_threshold < pruned_front[-1]['cost']:
            self.upper_cost_threshold *= 1.01
        self.stop_threshold_model.fit(self.success_observations, self.upper_cost_threshold)

        ### Sample suggestions
        search_centers = np.stack([e['input'] for e in pareto_observations])
        if self.top_observations:
            # Add top observations by score to search centers for diversity
            search_centers = np.vstack([search_centers, self._get_top_obs_params()])

        suggestions = self.hyperparameters.sample(
            len(search_centers)*self.suggestions_per_pareto, mu=search_centers)

        dedup_indices = self._filter_near_duplicates(suggestions)
        suggestions = suggestions[dedup_indices]

        if len(suggestions) == 0:
            return self.suggest(fill) # Fallback to random if all suggestions are filtered

        ### Predict scores and costs
        # Batch predictions to avoid GPU OOM for large number of suggestions
        gp_y_norm_list, gp_log_c_norm_list = [], []

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), warnings.catch_warnings():
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
                    pred_y_mean, pred_c_mean = torch.zeros(current_batch_size)

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

        # Maximize score. (Tried upper confidence bounds, but it did more harm because gp was noisy)
        suggestion_scores = self.hyperparameters.optimize_direction * gp_y_norm

        # Then, decide the budget for this session and favor closer suggestions
        max_c_mask = gp_c < self.max_suggestion_cost
        target_cost = self._sample_target_cost_ratio(self.expansion_rate)
        weight = 1 - abs(target_cost - gp_log_c_norm)
        suggestion_scores *= max_c_mask * weight

        # Then, consider the prob of training success, only when downsample = 1
        # NOTE: Useful only in limited scenarios, where each data point is expensive. So turn it off by default.
        if self.use_success_prob and len(self.success_observations) > 9 and len(self.failure_observations) > 9:
            success_params = np.array([e['input'] for e in self.success_observations])
            failure_params = np.array([e['input'] for e in self.failure_observations])
            X_train = np.vstack([success_params, failure_params])
            y_train = np.concatenate([
                np.ones(len(success_params)),
                np.zeros(len(failure_params))
            ])
            if len(np.unique(y_train)) > 1:
                self.success_classifier.fit(X_train, y_train)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    p_success = self.success_classifier.predict_proba(suggestions)[:, 1]
                suggestion_scores *= p_success

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

    def logit_transform(self, value, epsilon=1e-9):
        value = np.clip(value, epsilon, 1 - epsilon)
        logit = math.log(value / (1 - value))
        return np.clip(logit, -5, 100)

    def observe(self, hypers, score, cost, is_failure=False):
        params = self.hyperparameters.from_dict(hypers)

        if self.metric_distribution == 'percentile':
            score = self.logit_transform(score)

        new_observation = dict(
            input=params,
            output=score,
            cost=cost,
            is_failure=is_failure,
        )

        if is_failure or not np.isfinite(score) or np.isnan(score):
            new_observation['is_failure'] = True
            self.failure_observations.append(new_observation)
            return

        if self.success_observations:
            success_params = np.stack([e['input'] for e in self.success_observations])
            dist = np.linalg.norm(params - success_params, axis=1)
            same = np.where(dist < EPSILON)[0]
            if len(same) > 0:
                self.success_observations[same[0]] = new_observation
                return

        # Ignore obs that are below the minimum cost
        if self.cost_param_idx is not None and params[self.cost_param_idx] <= -1:
            return

        self.success_observations.append(new_observation)

        # Update top_observations without sorting the full list every time
        if len(self.top_observations) < self.num_keep_top_obs:
            self.top_observations.append(new_observation)
            self.top_observations.sort(key=lambda x: x['output'], reverse=True)
        elif score > self.top_observations[-1]['output']:
            self.top_observations.pop()
            self.top_observations.append(new_observation)
            self.top_observations.sort(key=lambda x: x['output'], reverse=True)

    def get_early_stop_threshold(self, cost):
        return self.stop_threshold_model.get_threshold(cost)

    def should_stop(self, score, cost):
        threshold = self.get_early_stop_threshold(cost)

        if self.metric_distribution == 'percentile':
            score = self.logit_transform(score)

        return score < threshold
