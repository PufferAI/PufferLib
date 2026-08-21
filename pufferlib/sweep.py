import random
import math
import warnings
from collections import deque
from copy import deepcopy
from contextlib import contextmanager

import numpy as np

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, PolynomialKernel, ScaleKernel, AdditiveKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior, GammaPrior
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from scipy.spatial import KDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer

EPSILON = 1e-6

def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v

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
        self.norm_mean = 0
        self.is_integer = is_integer

class Linear(Space):
    def __init__(self, min, max, scale, is_integer=False):
        if scale == 'auto':
            scale = 0.5

        super().__init__(min, max, scale, is_integer)

    def normalize(self, value):
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

        super().__init__(min, max, scale, is_integer)

    def normalize(self, value):
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
            scale = 1 / (np.log2(max) - np.log2(min))
        elif scale == 'auto':
            scale = 0.5

        super().__init__(min, max, scale, is_integer)

    def normalize(self, value):
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
        value = max(self.min, min(value, self.max))
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
                    'sweep_only', 'max_suggestion_cost', 'early_stop_quantile', 'gpus', 'max_runs'):
            continue

        assert isinstance(param, dict), f'Param {name} is not a dict'
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
        self.flat_spaces = dict(unroll_nested_dict(self.spaces))
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
        flat_params = dict(unroll_nested_dict(params))
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

    def suggest(self, fill=None, fixed_total_timesteps=None):
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

    def early_stop(self, logs, target_key):
        if any("loss/" in k and np.isnan(v) for k, v in logs.items()):
            logs['is_loss_nan'] = True
            return True
        return False


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

    def suggest(self, fill=None, fixed_total_timesteps=None):
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

            cost_dists += (np.max(pareto_costs) + 1)*np.eye(len(pareto_costs))
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

    def early_stop(self, logs, target_key):
        if any("loss/" in k and np.isnan(v) for k, v in logs.items()):
            logs['is_loss_nan'] = True
            return True
        return False


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, x_dim, use_polynomial=True):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # Matern 5/2 (HEBO/BoTorch default) plus a GammaPrior on the
        # lengthscale to keep ARD from collapsing at low data:dim ratios.
        matern_kernel = MaternKernel(
            nu=2.5, ard_num_dims=x_dim,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
        # Direct ref so lengthscale_range works regardless of additive ordering.
        self._matern_kernel = matern_kernel
        outputscale_prior = GammaPrior(2.0, 0.15)
        if use_polynomial:
            # Cost GP only: log_cost ≈ linear in log(lr)/log(steps), so the
            # additive linear term is appropriate. Score GP drops it — the
            # unbounded polynomial extrapolation pins lr to the search-space
            # bounds whenever failures cluster at one end of a dim.
            linear_kernel = PolynomialKernel(power=1)
            self.covar_module = ScaleKernel(
                AdditiveKernel(linear_kernel, matern_kernel),
                outputscale_prior=outputscale_prior,
            )
        else:
            self.covar_module = ScaleKernel(matern_kernel, outputscale_prior=outputscale_prior)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def lengthscale_range(self):
        lengthscale = self._matern_kernel.lengthscale.tolist()[0]
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
            break

    model.eval()
    likelihood.eval()
    return loss.item() if loss is not None else 0


class RobustLogCostModel:
    """Fits Score ~ A + B * log(Cost) via Quantile Regression (Median)."""
    def __init__(self, quantile=0.3, min_num_samples=3):
        self.quantile = quantile
        self.min_num_samples = min_num_samples
        self.is_fitted = False
        self.A = None
        self.B = None
        self.max_score = None
        self.min_score = None
        self.max_cost = None
        self.upper_cost_threshold = None

    def _quantile_loss(self, params, x, y, q):
        a, b = params
        y_pred = a + b * x
        residuals = y - y_pred
        return np.sum(np.maximum(q * residuals, (q - 1) * residuals))

    def fit(self, observations, upper_cost_threshold=None):
        self.is_fitted = False
        scores = np.array([e['output'] for e in observations])
        costs = np.array([e['cost'] for e in observations])
        self.max_score = scores.max()
        self.min_score = scores.min()
        self.upper_cost_threshold = upper_cost_threshold or costs.max()

        valid_indices = (costs > EPSILON) & np.isfinite(scores)
        if np.sum(valid_indices) < self.min_num_samples:
            return

        y = scores[valid_indices]
        c = costs[valid_indices]
        x_log_c = np.log(c)

        try:
            b_init, a_init = np.polyfit(x_log_c, y, 1)
        except np.linalg.LinAlgError:
            b_init, a_init = 0.0, np.mean(y)

        res = minimize(
            self._quantile_loss,
            x0=[a_init, b_init],
            args=(x_log_c, y, self.quantile),
            method='Nelder-Mead',
            bounds=[(None, None), (0, None)]
        )

        self.A, self.B = res.x
        self.is_fitted = True

    def get_threshold(self, cost, min_cost_fraction=0.03, abs_min_cost=10):
        if not self.is_fitted or self.upper_cost_threshold is None:
            return -np.inf

        min_allowed_cost = self.upper_cost_threshold * min_cost_fraction + abs_min_cost
        if cost < min_allowed_cost:
            return -np.inf

        # The original "stop trials past 1.2x budget" branch returned
        # 0.9 * max_score, which inverts for negated losses (max_score < 0
        # ⇒ threshold > max_score ⇒ every trial gets killed). With fixed
        # train_steps it's also unreachable in normal operation. Dropped.
        return self.A + self.B * np.log(cost)


class Protein:
    def __init__(self,
            sweep_config,
            max_suggestion_cost = 3600,
            resample_frequency = 0,
            num_random_samples = 10,
            global_search_scale = 1,
            suggestions_per_pareto = 256,
            expansion_rate = 0.1,
            gp_training_iter = 100,
            gp_learning_rate = 0.01,
            gp_max_obs = 750,
            infer_batch_size = 4096,
            optimizer_reset_frequency = 50,
            use_gpu = True,
            cost_param = "train/total_timesteps",
            prune_pareto = True,
        ):
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

        self.suggestion_idx = 0
        self.min_score, self.max_score = math.inf, -math.inf
        self.log_c_min, self.log_c_max = math.inf, -math.inf

        # Failure sentinel: frozen once on first merge into the score GP.
        # Previously `e['output'] = self.min_score` ratcheted upward as new
        # successes lowered min_score, retilting the kernel toward unfailed
        # corners forever.
        self._failure_sentinel = None
        # HEBO recipe: Yeo-Johnson output warping + per-fit input rescaling.
        # Warping linearizes the response surface; per-fit rescaling restores
        # σ-driven exploration on dims where observations have clustered.
        self._y_warper_yj = PowerTransformer(method='yeo-johnson', standardize=True)
        self._y_best_warped = None
        self._x_min_obs = None
        self._x_max_obs = None

        self.sobol = Sobol(d=self.hyperparameters.num, scramble=True)
        self.num_random_samples = num_random_samples

        self.cost_param_idx = self.hyperparameters.get_flat_idx(cost_param)
        self.cost_space = None
        self.cost_random_suggestion = None
        if self.cost_param_idx is not None:
            self.cost_space = list(self.hyperparameters.flat_spaces.values())[self.cost_param_idx]
            self.cost_random_suggestion = -0.8
        self.target_cost_ratio = []
        self._running_target_buffer = deque(maxlen=30)

        self.gp_max_obs = gp_max_obs
        self.infer_batch_size = infer_batch_size

        self.use_success_prob = sweep_config['downsample'] == 1
        self.success_classifier = LogisticRegression(class_weight='balanced')

        self.stop_threshold_model = RobustLogCostModel(quantile=sweep_config['early_stop_quantile'])
        self.upper_cost_threshold = -np.inf

        with default_tensor_dtype(torch.float64):
            noise_prior = LogNormalPrior(math.log(1e-2), 0.5)
            noise_constraint = gpytorch.constraints.GreaterThan(8e-4)

            dummy_x = torch.ones((1, self.hyperparameters.num), device=self.device)
            dummy_y = torch.zeros(1, device=self.device)
            # Score GP: drops the polynomial term.
            self.likelihood_score = GaussianLikelihood(
                noise_prior=deepcopy(noise_prior),
                noise_constraint=deepcopy(noise_constraint),
            ).to(self.device)
            self.gp_score = ExactGPModel(
                dummy_x, dummy_y, self.likelihood_score,
                self.hyperparameters.num, use_polynomial=False,
            ).to(self.device)
            self.mll_score = ExactMarginalLogLikelihood(self.likelihood_score, self.gp_score).to(self.device)
            self.score_opt = torch.optim.Adam(self.gp_score.parameters(), lr=self.gp_learning_rate, amsgrad=True)

            # Cost GP: keeps the polynomial (log_cost approximately linear in
            # log(lr)/log(steps), so the linear extrapolation is appropriate).
            self.likelihood_cost = GaussianLikelihood(
                noise_prior=deepcopy(noise_prior),
                noise_constraint=deepcopy(noise_constraint),
            ).to(self.device)
            self.gp_cost = ExactGPModel(
                dummy_x, dummy_y, self.likelihood_cost,
                self.hyperparameters.num, use_polynomial=True,
            ).to(self.device)
            self.mll_cost = ExactMarginalLogLikelihood(self.likelihood_cost, self.gp_cost).to(self.device)
            self.cost_opt = torch.optim.Adam(self.gp_cost.parameters(), lr=self.gp_learning_rate, amsgrad=True)

            self.gp_params_buffer = torch.empty(self.gp_max_obs, self.hyperparameters.num, device=self.device)
            self.gp_score_buffer = torch.empty(self.gp_max_obs, device=self.device)
            self.gp_cost_buffer = torch.empty(self.gp_max_obs, device=self.device)
            self.infer_batch_buffer = torch.empty(self.infer_batch_size, self.hyperparameters.num, device=self.device)

    def to(self, device):
        self.device = torch.device(device)
        for attr in ('gp_score', 'gp_cost', 'likelihood_score', 'likelihood_cost',
                     'mll_score', 'mll_cost', 'gp_params_buffer', 'gp_score_buffer',
                     'gp_cost_buffer', 'infer_batch_buffer'):
            setattr(self, attr, getattr(self, attr).to(self.device))
        for opt in (self.score_opt, self.cost_opt):
            for state in opt.state.values():
                state.update({k: v.to(self.device) for k, v in state.items() if isinstance(v, torch.Tensor)})
        return self

    def _filter_near_duplicates(self, inputs, duplicate_threshold=EPSILON):
        if len(inputs) < 2:
            return np.arange(len(inputs))

        tree = KDTree(inputs)
        to_keep = np.ones(len(inputs), dtype=bool)
        for i in range(len(inputs) - 1, -1, -1):
            if to_keep[i]:
                nearby_indices = tree.query_ball_point(inputs[i], r=duplicate_threshold)
                nearby_indices.remove(i)
                if nearby_indices:
                    to_keep[nearby_indices] = False

        return np.where(to_keep)[0]

    def _sample_observations(self, max_size=None, recent_ratio=0.5):
        if not self.success_observations:
            return []

        observations = self.success_observations.copy()

        y = np.array([e['output'] for e in observations])
        self.min_score, self.max_score = y.min(), y.max()

        c = np.array([e['cost'] for e in observations])
        log_c = np.log(np.maximum(c, EPSILON))
        self.log_c_min = log_c.min()
        self.log_c_max = np.quantile(log_c, 0.97)

        # Failures pinned to a FROZEN sentinel below the worst observed
        # success. Without this freeze the failure floor ratchets with
        # min_score as new successes arrive, retilting the kernel slope.
        if len(observations) < 100 and self.failure_observations:
            if self._failure_sentinel is None:
                spread = max(abs(self.max_score - self.min_score), 1.0)
                self._failure_sentinel = float(self.min_score) - 0.5 * spread
            for e in self.failure_observations:
                e['output'] = self._failure_sentinel

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

        # Per-fit input rescaling to the observed box. On dims where obs
        # cluster, x_span shrinks → σ saturates at unobserved values → UCB
        # exploration dominates on that dim.
        params = np.array([e['input'] for e in sampled_observations], dtype=np.float64)
        self._x_min_obs = params.min(axis=0)
        self._x_max_obs = params.max(axis=0)
        x_span = np.maximum(self._x_max_obs - self._x_min_obs, 1e-8)
        params_scaled = 2.0 * (params - self._x_min_obs) / x_span - 1.0
        params_tensor = self.gp_params_buffer[:num_sampled]
        params_tensor.copy_(torch.from_numpy(params_scaled))

        # Output warping: signed (so GP always sees "higher = better"), then
        # Yeo-Johnson + standardize. Fall back to plain z-score on degeneracy.
        y = np.array([e['output'] for e in sampled_observations], dtype=np.float64)
        signed_y = self.hyperparameters.optimize_direction * y
        y_warped = None
        if num_sampled >= 4 and float(np.std(signed_y)) > 1e-9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    y_warped = self._y_warper_yj.fit_transform(signed_y.reshape(-1, 1)).ravel().astype(np.float64)
                except (ValueError, RuntimeError):
                    y_warped = None
        if y_warped is None:
            mu, sd = signed_y.mean(), max(float(np.std(signed_y)), EPSILON)
            y_warped = (signed_y - mu) / sd
        self._y_best_warped = float(y_warped.max())
        y_warped_tensor = self.gp_score_buffer[:num_sampled]
        y_warped_tensor.copy_(torch.from_numpy(y_warped))

        c = np.array([e['cost'] for e in sampled_observations])
        log_c = np.log(np.maximum(c, EPSILON))
        log_c_norm = (log_c - self.log_c_min) / (self.log_c_max - self.log_c_min + EPSILON)
        log_c_norm_tensor = self.gp_cost_buffer[:num_sampled]
        log_c_norm_tensor.copy_(torch.from_numpy(log_c_norm))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", gpytorch.utils.warnings.NumericalWarning)
            score_loss = train_gp_model(self.gp_score, self.likelihood_score, self.mll_score, self.score_opt, params_tensor, y_warped_tensor, training_iter=self.gp_training_iter)
            cost_loss = train_gp_model(self.gp_cost, self.likelihood_cost, self.mll_cost, self.cost_opt, params_tensor, log_c_norm_tensor, training_iter=self.gp_training_iter)

        return score_loss, cost_loss

    def _sample_target_cost_ratio(self, expansion_rate, target_ratios=(0.16, 0.32, 0.48, 0.64, 0.8, 1.0)):
        if not self.target_cost_ratio:
            self.target_cost_ratio = list(target_ratios)
            random.shuffle(self.target_cost_ratio)
        target_ratio = np.clip(self.target_cost_ratio.pop() + 0.1 * np.random.randn(), 0, 1)
        return (1 + expansion_rate) * target_ratio

    def _sobol_suggestion(self, fill, fixed_cost_norm):
        suggestion = 2 * self.sobol.random(1)[0] - 1
        if fixed_cost_norm is not None:
            suggestion[self.cost_param_idx] = fixed_cost_norm
        elif self.cost_param_idx is not None:
            cost_suggestion = self.cost_random_suggestion + 0.1 * np.random.randn()
            suggestion[self.cost_param_idx] = np.clip(cost_suggestion, -1, 1)
        return self.hyperparameters.to_dict(suggestion, fill), {}

    def suggest(self, fill, fixed_total_timesteps=None):
        info = {}
        self.suggestion_idx += 1
        fixed_cost_norm = None
        if fixed_total_timesteps is not None and self.cost_space is not None:
            fixed_cost_norm = self.cost_space.normalize(fixed_total_timesteps)

        if self.suggestion_idx <= self.num_random_samples:
            return self._sobol_suggestion(fill, fixed_cost_norm)

        if self.resample_frequency and self.suggestion_idx % self.resample_frequency == 0:
            candidates, _ = pareto_points(self.success_observations)
            if candidates:
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
        # An all-failure Sobol phase leaves success_observations empty here.
        if not pruned_front:
            return self._sobol_suggestion(fill, fixed_cost_norm)
        pareto_observations = pruned_front if self.prune_pareto else pareto_front

        if self.upper_cost_threshold < 0:
            self.upper_cost_threshold = pruned_front[-1]['cost']
        elif self.upper_cost_threshold < pruned_front[-1]['cost']:
            self.upper_cost_threshold *= 1.01
        self.stop_threshold_model.fit(self.success_observations, self.upper_cost_threshold)

        # Candidates: global Sobol (UCB σ-exploration) + tight local
        # perturbations around top-K observations (µ-injection refinement).
        # Pure Sobol breaks down at our dim: P(any sample within ~ℓ of an
        # obs) ≈ (ℓ/2)^d ≈ 10⁻¹² at d=16, so argmax-µ over Sobol never
        # lands near best-obs.
        d = self.hyperparameters.num
        n_cand = 1 << max(11, (max(2000, 200 * d) - 1).bit_length())
        cand_sobol = Sobol(d=d, scramble=True, seed=self.suggestion_idx)
        suggestions_global = 2.0 * cand_sobol.random(n_cand) - 1.0

        k_top = min(5, len(self.success_observations))
        top_centers = np.stack([e['input'] for e in
                                sorted(self.success_observations,
                                       key=lambda e: e['output'],
                                       reverse=True)[:k_top]])
        n_local = max(256 * k_top, 512)
        suggestions_local = self.hyperparameters.sample(n_local, mu=top_centers, scale=0.4)

        suggestions = np.vstack([suggestions_global, suggestions_local])

        if fixed_cost_norm is not None:
            suggestions[:, self.cost_param_idx] = fixed_cost_norm

        dedup_indices = self._filter_near_duplicates(suggestions)
        suggestions = suggestions[dedup_indices]

        if len(suggestions) == 0:
            return self.suggest(fill)

        # Apply the same per-fit rescaling to candidates before prediction.
        x_span = np.maximum(self._x_max_obs - self._x_min_obs, 1e-8)
        suggestions_scaled = 2.0 * (suggestions - self._x_min_obs) / x_span - 1.0

        gp_y_mean_list, gp_y_var_list, gp_log_c_norm_list = [], [], []

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), warnings.catch_warnings():
            warnings.simplefilter("ignore", gpytorch.utils.warnings.NumericalWarning)

            for i in range(0, len(suggestions_scaled), self.infer_batch_size):
                batch_numpy = suggestions_scaled[i:i+self.infer_batch_size]
                current_batch_size = len(batch_numpy)
                batch_tensor = self.infer_batch_buffer[:current_batch_size]
                batch_tensor.copy_(torch.from_numpy(batch_numpy))

                try:
                    posterior = self.likelihood_score(self.gp_score(batch_tensor))
                    pred_y_mean = posterior.mean.cpu()
                    pred_y_var = posterior.variance.cpu()
                    pred_c_mean = self.likelihood_cost(self.gp_cost(batch_tensor)).mean.cpu()
                except RuntimeError:
                    pred_y_mean = torch.zeros(current_batch_size)
                    pred_y_var = torch.ones(current_batch_size)
                    pred_c_mean = torch.zeros(current_batch_size)

                gp_y_mean_list.append(pred_y_mean)
                gp_y_var_list.append(pred_y_var)
                gp_log_c_norm_list.append(pred_c_mean)

        gp_y_mean = torch.cat(gp_y_mean_list).numpy()
        gp_y_var = torch.cat(gp_y_var_list).numpy()
        gp_y_std = np.sqrt(np.maximum(gp_y_var, EPSILON * EPSILON))
        gp_log_c_norm = torch.cat(gp_log_c_norm_list).numpy()
        gp_log_c = gp_log_c_norm*(self.log_c_max - self.log_c_min) + self.log_c_min
        gp_c = np.exp(gp_log_c)

        # HEBO stochastic perturbation: noise ~ N(0, 2·σ²_obs) added to µ
        # robustifies acquisition against GP miscalibration.
        noise_std = float(self.gp_score.likelihood.noise.detach().sqrt().cpu().item())
        gp_y_mu_perturbed = gp_y_mean + math.sqrt(2.0) * noise_std * np.random.randn(len(gp_y_mean))

        # UCB; κ = sqrt(2·log(n·d)) is HEBO's heuristic, grows slowly with n.
        n_obs = max(1, len(self.success_observations))
        kappa = math.sqrt(2.0 * math.log(max(n_obs * d, 2)))
        ucb_scores = gp_y_mu_perturbed + kappa * gp_y_std

        # Cost and success-prob factors kept in log-space so they compose
        # additively with the acquisition scores.
        cost_logweight = np.zeros_like(ucb_scores)
        cost_mask = np.ones_like(ucb_scores, dtype=bool)
        if fixed_cost_norm is None and self.cost_param_idx is not None:
            cost_mask = gp_c < self.max_suggestion_cost
            target_cost = self._sample_target_cost_ratio(self.expansion_rate)
            cost_logweight = np.log(np.maximum(1 - np.abs(target_cost - gp_log_c_norm), 1e-9))

        success_logweight = np.zeros_like(ucb_scores)
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
                success_logweight = np.log(np.maximum(p_success, 1e-9))

        # MACE mean injection: every 3rd GP-phase trial commits to argmax(µ)
        # to refine the currently-best mode. At κ≈3.4 pure UCB won't shrink
        # σ fast enough on a single observation. Drop the target-cost soft
        # weight here (refinement shouldn't be cost-biased), keep the hard
        # over-budget mask and the success-prob weight.
        if self.suggestion_idx % 3 == 0:
            mu_scores = gp_y_mu_perturbed + success_logweight
            mu_scores[~cost_mask] = -np.inf
            best_idx = int(np.argmax(mu_scores))
            acq_mode = 'mu_inject'
            picked_score = float(mu_scores[best_idx])
        else:
            ucb_scores = ucb_scores + cost_logweight + success_logweight
            ucb_scores[~cost_mask] = -np.inf
            best_idx = int(np.argmax(ucb_scores))
            acq_mode = 'ucb'
            picked_score = float(ucb_scores[best_idx])

        info = dict(
            acq_mode = acq_mode,
            cost = gp_c[best_idx].item(),
            score_mean_warped = float(gp_y_mean[best_idx]),
            score_std_warped = float(gp_y_std[best_idx]),
            picked_score = picked_score,
            kappa = kappa,
            f_best_warped = self._y_best_warped,
            score_loss = score_loss,
            cost_loss = cost_loss,
            score_lengthscale = self.gp_score.lengthscale_range,
            cost_lengthscale = self.gp_cost.lengthscale_range,
        )
        print('Predicted -- ',
            f'{acq_mode}: {picked_score:+.3f}',
            f'µ_w: {info["score_mean_warped"]:+.3f}',
            f'σ_w: {info["score_std_warped"]:.3f}',
            f'κ: {info["kappa"]:.2f}',
            f'f*_w: {info["f_best_warped"]:+.3f}',
            f'Cost: {info["cost"]:.0f}',
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

        if self.cost_param_idx is not None and params[self.cost_param_idx] <= -1:
            return

        self.success_observations.append(new_observation)

    def get_early_stop_threshold(self, cost):
        return self.stop_threshold_model.get_threshold(cost)

    def should_stop(self, score, cost):
        threshold = self.get_early_stop_threshold(cost)

        if self.metric_distribution == 'percentile':
            score = self.logit_transform(score)

        return score < threshold

    def early_stop(self, logs, target_key):
        for k, v in logs['loss'].items():
            if np.isnan(v):
                logs['is_loss_nan'] = True
                return True

        if 'uptime' not in logs or target_key not in logs:
            return False

        metric_val, cost = logs['env'][target_key], logs['uptime']
        self._running_target_buffer.append(metric_val)
        target_running_mean = np.mean(self._running_target_buffer)
        threshold = self.get_early_stop_threshold(cost)
        logs['early_stop_threshold'] = max(threshold, -5)
        if self.should_stop(max(target_running_mean, metric_val), cost):
            logs['is_loss_nan'] = False
            return True
        return False
