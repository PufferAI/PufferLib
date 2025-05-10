import numpy as np
import random
import math
import warnings

from copy import deepcopy

import pufferlib
import scipy.stats

import torch
import pyro
from pyro.contrib import gp as gp
from pyro.contrib.gp.kernels import Kernel
from pyro.contrib.gp.models import GPRegression


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
        if name in ('method', 'name', 'metric', 'max_score'):
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
        self.flat_spaces = dict(pufferlib.utils.unroll_nested_dict(self.spaces))
        self.num = len(self.flat_spaces)

        self.metric = config['metric']
        assert self.metric['goal'] in ['maximize', 'minimize']
        self.optimize_direction = 1 if self.metric['goal'] == 'maximize' else -1

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
        flat_params = dict(pufferlib.utils.unroll_nested_dict(params))
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


def pareto_points(observations, eps=1e-6):
    scores = np.array([e['output'] for e in observations])
    costs = np.array([e['cost'] for e in observations])
    pareto = []
    idxs = []
    for idx, obs in enumerate(observations):
        # TODO: Ties and groups
        higher_score = scores + eps > scores[idx]
        lower_cost = costs - eps < costs[idx]
        better = higher_score & lower_cost
        better[idx] = False
        if not better.any():
            pareto.append(obs)
            idxs.append(idx)

    return pareto, idxs

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
        return self.hyperparameters.to_dict(self.suggestion, fill)

    def observe(self, score, cost, is_failure=False):
        self.success_observations.append(dict(
            input=self.suggestion,
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


def create_gp(x_dim, scale_length=1.0):
    # Dummy data
    X = scale_length * torch.ones((1, x_dim))
    y = torch.zeros((1,))

    matern_kernel = gp.kernels.Matern32(input_dim=x_dim, lengthscale=X)
    linear_kernel = gp.kernels.Polynomial(x_dim, degree=1)
    kernel = gp.kernels.Sum(linear_kernel, matern_kernel)

    # Params taken from HEBO: https://arxiv.org/abs/2012.03826
    model = gp.models.GPRegression(X, y, kernel=kernel, jitter=1.0e-4)
    model.noise = pyro.nn.PyroSample(pyro.distributions.LogNormal(math.log(1e-2), 0.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, optimizer

class Protein:
    def __init__(self,
            sweep_config,
            max_suggestion_cost = None,
            resample_frequency = 5,
            num_random_samples = 10,
            global_search_scale = 1,
            random_suggestions = 1024,
            suggestions_per_pareto = 256,
            min_score = None,
            max_score = None,
        ):
        self.hyperparameters = Hyperparameters(sweep_config)

        self.min_score = min_score
        self.max_score = max_score
        if self.min_score is None:
            warnings.warn('No min_score specified. This can destabilize tuning.')
        if self.max_score is None:
            warnings.warn('No max_score specified. This can destabilize tuning.')

        self.num_random_samples = num_random_samples
        self.global_search_scale = global_search_scale
        self.random_suggestions = random_suggestions
        self.suggestions_per_pareto = suggestions_per_pareto
        self.resample_frequency = resample_frequency
        self.max_suggestion_cost = max_suggestion_cost

        self.success_observations = []
        self.failure_observations = []
        self.suggestion_idx = 0

        self.gp_score, self.score_opt = create_gp(self.hyperparameters.num)
        self.gp_cost, self.cost_opt = create_gp(self.hyperparameters.num)

    def suggest(self, fill):
        # TODO: Clip random samples to bounds so we don't get bad high cost samples
        info = {}
        #if self.suggestion_idx <= self.num_random_samples:
        #    suggestions = self.hyperparameters.sample(self.random_suggestions)
        #    best_idx = np.random.randint(0, self.random_suggestions)
        #    best = suggestions[best_idx]
        self.suggestion_idx += 1
        if len(self.success_observations) == 0:
            best = self.hyperparameters.search_centers
            return self.hyperparameters.to_dict(best, fill), info
        elif len(self.success_observations) < self.num_random_samples:
            suggestions = self.hyperparameters.sample(self.random_suggestions)
            self.suggestion = random.choice(suggestions)
            return self.hyperparameters.to_dict(self.suggestion, fill), info
        elif self.resample_frequency and self.suggestion_idx % self.resample_frequency == 0:
            candidates, _ = pareto_points(self.success_observations)
            suggestions = np.stack([e['input'] for e in candidates])
            best_idx = np.random.randint(0, len(candidates))
            best = suggestions[best_idx]
            return self.hyperparameters.to_dict(best, fill), info

        params = np.array([e['input'] for e in self.success_observations])
        params = torch.from_numpy(params)
        eps = 1e-2

        # Scores variable y
        y = np.array([e['output'] for e in self.success_observations])

        # Transformed scores
        min_score = self.min_score
        if min_score is None:
            min_score = np.min(y) - np.min(np.abs(y))

        if np.min(y) < min_score - 1e-6:
            raise ValueError(f'Min score {min_score} is less than min score in data {np.min(y)}')

        max_score = self.max_score
        if max_score is None:
            max_score = np.max(y) + np.max(np.abs(y))

        if np.max(y) > max_score + 1e-6:
            raise ValueError(f'Max score {max_score} is greater than max score in data {np.max(y)}')

        # Linearize, exp transform, linearize
        y_norm = (y - min_score) / (max_score - min_score)
        #yt = -np.log(1 - y_norm + eps)
        #yt_min = np.min(yt)
        #yt_max = np.max(yt)
        #yt_norm = (yt - yt_min) / (yt_max - yt_min)

        #self.gp_score.set_data(params, torch.from_numpy(yt_norm))
        self.gp_score.set_data(params, torch.from_numpy(y_norm))
        self.gp_score.train()
        gp.util.train(self.gp_score, self.score_opt)
        self.gp_score.eval()

        # Log costs
        c = np.array([e['cost'] for e in self.success_observations])

        log_c = np.log(c)

        # Linear input norm creates clean 1 mean fn
        log_c_min = np.min(log_c)
        log_c_max = np.max(log_c)
        log_c_norm = (log_c - log_c_min) / (log_c_max - log_c_min)

        self.gp_cost.mean_function = lambda x: 1
        self.gp_cost.set_data(params, torch.from_numpy(log_c_norm))
        self.gp_cost.train()
        gp.util.train(self.gp_cost, self.cost_opt)
        self.gp_cost.eval()

        candidates, pareto_idxs = pareto_points(self.success_observations)
        pareto_costs = np.array([e['cost'] for e in candidates])

        #cost_dists = np.abs(np.log(pareto_costs[:, None]) - np.log(pareto_costs[None, :]))
        ###cost_dists = np.abs(pareto_costs[:, None] - pareto_costs[None, :])
        #cost_dists += (np.max(pareto_costs) + 1)*np.eye(len(pareto_costs)) # mask self-distance
        #idx = np.argmax(np.min(cost_dists, axis=1))
        #search_centers = candidates[idx]['input']

        ### Sample suggestions
        search_centers = np.stack([e['input'] for e in candidates])
        suggestions = self.hyperparameters.sample(
            len(candidates)*self.suggestions_per_pareto, mu=search_centers)

        ### Predict scores and costs
        suggestions = torch.from_numpy(suggestions)
        with torch.no_grad():
            gp_y_norm, gp_y_norm_var = self.gp_score(suggestions)
            gp_log_c_norm, gp_log_c_norm_var = self.gp_cost(suggestions)

        gp_y_norm = gp_y_norm.numpy()
        gp_log_c_norm = gp_log_c_norm.numpy()

        # Unlinearize, inverse exp transform, unlinearize
        #gp_yt = gp_yt_norm*(yt_max - yt_min) + yt_min
        #gp_y_norm = -(np.exp(-gp_yt) - 1 - eps)
        #gp_y = gp_y_norm*(max_score - min_score) + min_score
        gp_y = gp_y_norm*(max_score - min_score) + min_score

        gp_log_c = gp_log_c_norm*(log_c_max - log_c_min) + log_c_min
        gp_c = np.exp(gp_log_c)

        gp_c_min = np.min(gp_c)
        gp_c_max = np.max(gp_c)
        gp_c_norm = (gp_c - gp_c_min) / (gp_c_max - gp_c_min)

        pareto_y = y[pareto_idxs]
        #pareto_yt = yt[pareto_idxs]
        #pareto_yt_norm = yt_norm[pareto_idxs]
        pareto_c = c[pareto_idxs]
        pareto_log_c_norm = log_c_norm[pareto_idxs]

        max_c = np.max(c)
        min_c = np.min(c)

        c_right = abs(pareto_log_c_norm[None, :] - gp_log_c_norm[:, None])

        #pareto_c_norm = (pareto_c - min_c) / (max_c - min_c)
        #gp_c_norm = (gp_c - min_c) / (max_c - min_c)
        #c_right = np.abs(pareto_c_norm[None, :] - gp_c_norm[:, None])

        #pareto_log_c_norm = (np.log(pareto_c) - log_c_min) / (log_c_max - log_c_min)
        #c_right = np.abs(pareto_log_c_norm[None, :] - gp_log_c_norm[:, None])

        sorted_dist = np.sort(c_right, axis=1)
        #top_k = sorted_dist[:, :5]
        #pareto_dist_weight = np.sum(top_k, axis=1) / top_k.shape[1]

        nearest_idx = np.argmin(c_right, axis=1)
        nearest_pareto_dist = np.min(c_right, axis=1)
        nearest_pareto_y = pareto_y[nearest_idx]

        #c_left = np.abs(gp_c[:, None] - pareto_c[None, :])
        #c_left[c_left < 0] = np.inf
        #nearest_idx = np.argmin(c_left, axis=1)
        #nearest_pareto_yt_norm = pareto_yt_norm[nearest_idx]

        max_c_mask = gp_c < self.max_suggestion_cost
        #suggestion_scores = self.hyperparameters.optimize_direction * max_c_mask * (
        #        gp_yt_norm - nearest_pareto_yt_norm) * nearest_pareto_dist

        #suggestion_scores = self.hyperparameters.optimize_direction * max_c_mask * (
        #        gp_yt_norm - nearest_pareto_yt_norm)# / gp_c

        #np.argwhere(gp_c > c)
        cumsum_mask = c[None, :] <= np.clip(gp_c[:, None], min_c, max_c)
        cumsum_mask = cumsum_mask * c[None, :]
        cumsum = np.sum(cumsum_mask, axis=1) / np.sum(c)
        target = gp_c_norm 
        weight = target - cumsum

        #if np.random.rand() < 0.5:
        #    score = gp_y_norm
        #else:
        #    score = gp_y_norm * weight
        #suggestion_scores = self.hyperparameters.optimize_direction * max_c_mask * (
        #        score)# / gp_c


        target = 1.25*np.random.rand()
        weight = 1 - abs(target - gp_log_c_norm)

        suggestion_scores = self.hyperparameters.optimize_direction * max_c_mask * (
                gp_y_norm*weight)# / gp_c

        #suggestion_scores = self.hyperparameters.optimize_direction * max_c_mask * (
        #        gp_y_norm*nearest_pareto_dist)# / gp_c

        #exp_scores = np.exp(suggestion_scores)
        #sum_exp_scores = np.sum(exp_scores)
        #softmax_scores = exp_scores / sum_exp_scores
        #idxs = np.arange(len(softmax_scores))
        #best_idx = np.random.choice(idxs, p=softmax_scores)

        # This works and uncovers approximate binary search when the GP is perfect
        # Can't include cost in denom because it biases this case
        # Instead, use conservative score and/or cost estimates
        # Just need to figure out why the GP is overconfident

        best_idx = np.argmax(suggestion_scores)
        #best_idx = np.argmax(gp_y_norm)
        info = dict(
            cost = gp_c[best_idx].item(),
            score = gp_y[best_idx].item(),
            nearby = nearest_pareto_y[best_idx].item(),
            dist = nearest_pareto_dist[best_idx].item(),
            rating = suggestion_scores[best_idx].item(),
        )
        print('Predicted -- ',
            f'Score: {info["score"]:.3f}',
            f'Nearby: {info["nearby"]:.3f}',
            f'Dist: {info["dist"]:.3f}',
            f'Cost: {info["cost"]:.3f}',
            f'Rating: {info["rating"]:.3f}',
        )
        '''
        if info['rating'] < 10:
            from bokeh.models import ColumnDataSource, LinearColorMapper
            from bokeh.plotting import figure, show
            from bokeh.palettes import Turbo256

            source = ColumnDataSource(data=dict(
                x=c,
                y=y,
                order=np.argsort(c),
            ))
            mapper = LinearColorMapper(
                palette=Turbo256,
                low=0,
                high=len(c)
            )

            idxs = np.argsort(pareto_c)
            pareto_source = ColumnDataSource(data=dict(
                x=pareto_c[idxs],
                y=pareto_y[idxs],
            ))

            c_sorted = sorted(c)
            cost_source = ColumnDataSource(data=dict(
                x = c_sorted,
                y = np.cumsum(c_sorted) / np.sum(c_sorted),
            ))

            #gp_pareto_source = ColumnDataSource(data=dict(
            #    x=gp_c,
            #    y=gp_y,
            #    order=np.argsort(gp_c),
            #))

            preds = [{
                'output': gp_y[i],
                'cost': gp_c[i],
            } for i in range(len(gp_c))]
            _, pareto_idxs = pareto_points(preds)

            gp_c_pareto = gp_c[pareto_idxs]
            gp_y_pareto = gp_y[pareto_idxs]
            idxs = np.argsort(gp_c_pareto)
            gp_source = ColumnDataSource(data=dict(
                x=gp_c_pareto[idxs],
                y=gp_y_pareto[idxs],
            ))

            p = figure(title='Hyperparam Test', 
                       x_axis_label='Cost', 
                       y_axis_label='Score')

            # Original data
            p.scatter(
                x='x', 
                y='y', 
                color={'field': 'order', 'transform': mapper}, 
                size=10, 
                source=source
            )

            p.line(x='x', y='y', color='red', source=pareto_source)
            p.line(x='x', y='y', color='blue', source=gp_source)
            p.line(x='x', y='y', color='green', source=cost_source)
            #p.line(x='x', y='y', color='green', source=gp_pareto_source)

            show(p)
        '''

        best = suggestions[best_idx].numpy()
        return self.hyperparameters.to_dict(best, fill), info

    def observe(self, hypers, score, cost, is_failure=False):
        params = self.hyperparameters.from_dict(hypers)
        new_observation = dict(
            input=params,
            output=score,
            cost=cost,
            is_failure=is_failure,
        )

        if len(self.success_observations) == 0:
            self.success_observations.append(new_observation)
            return

        success_params = np.stack([e['input'] for e in self.success_observations])
        dist = np.linalg.norm(params - success_params, axis=1)
        same = np.where(dist < 1e-6)[0]
        if len(same) > 0:
            self.success_observations[same[0]] = new_observation
        else:
            self.success_observations.append(new_observation)

def _carbs_params_from_puffer_sweep(sweep_config):
    from carbs import (
        Param,
        LinearSpace,
        LogSpace,
        LogitSpace,
    )

    param_spaces = {}
    for name, param in sweep_config.items():
        if name in ('method', 'name', 'metric', 'max_score'):
            continue

        assert isinstance(param, dict)
        if any(isinstance(param[k], dict) for k in param):
            param_spaces[name] = _carbs_params_from_puffer_sweep(param)
            continue
 
        assert 'distribution' in param
        distribution = param['distribution']
        search_center = param['mean']
        kwargs = dict(
            min=param['min'],
            max=param['max'],
        )
        if distribution == 'uniform':
            space = LinearSpace(**kwargs)
        elif distribution in ('int_uniform', 'uniform_pow2'):
            space = LinearSpace(**kwargs, is_integer=True)
        elif distribution == 'log_normal':
            space = LogSpace(**kwargs)
        elif distribution == 'logit_normal':
            space = LogitSpace(**kwargs)
        else:
            raise ValueError(f'Invalid distribution: {distribution}')

        param_spaces[name] = Param(
            name=name,
            space=space,
            search_center=search_center
        )

    return param_spaces

class Carbs:
    def __init__(self,
            sweep_config: dict,
            max_suggestion_cost: float = None,
            resample_frequency: int = 5,
            num_random_samples: int = 10,
        ):

        param_spaces = _carbs_params_from_puffer_sweep(sweep_config)
        flat_spaces = [e[1] for e in pufferlib.utils.unroll_nested_dict(param_spaces)]
        for e in flat_spaces:
            print(e.name, e.space)

        from carbs import (
            CARBSParams,
            CARBS,
        )

        carbs_params = CARBSParams(
            better_direction_sign=1,
            is_wandb_logging_enabled=False,
            resample_frequency=resample_frequency,
            num_random_samples=num_random_samples,
            max_suggestion_cost=max_suggestion_cost,
            is_saved_on_every_observation=False,
        )
        self.carbs = CARBS(carbs_params, flat_spaces)

    def suggest(self, args):
        self.suggestion = self.carbs.suggest().suggestion
        for k in ('train', 'env'):
            for name, param in args['sweep'][k].items():
                if name in self.suggestion:
                    args[k][name] = self.suggestion[name]

    def observe(self, hypers, score, cost, is_failure=False):
        from carbs import ObservationInParam
        self.carbs.observe(
            ObservationInParam(
                input=self.suggestion,
                output=score,
                cost=cost,
                is_failure=is_failure,
            )
        )