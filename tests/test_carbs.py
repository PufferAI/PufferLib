import numpy as np

from carbs import CARBS
from carbs import CARBSParams
from carbs import LinearSpace
from carbs import LogSpace
from carbs import LogitSpace
from carbs import ObservationInParam
from carbs import ParamDictType
from carbs import Param

import wandb


class SyntheticExperiment:
    def __init__(self, n_params, noise=0.1):
        self.n_params = n_params
        self.noise = noise

        self.param_optima = np.random.randn(n_params)

    def optimize(self, params):
        dist = (params-self.param_optima)**2
        reward = 2**(-dist)
        noise = 1 + self.noise*np.random.randn()
        return noise * np.prod(reward)

class CARBSSearch:
    def __init__(self, experiment):
        self.experiment = experiment
        self.best_reward = None
        self.best_params = None

        param_spaces = [
            Param(name=str(i),
                    space=LinearSpace(min=-10, max=10, is_integer=False),
                    search_center=0.0)
            for i in range(self.experiment.n_params)
        ]
        carbs_params = CARBSParams(
            better_direction_sign=1,
            is_wandb_logging_enabled=False,
            resample_frequency=0,
        )
        self.carbs = CARBS(carbs_params, param_spaces)

    def sample(self):
        suggestion = self.carbs.suggest().suggestion
        params = np.array([suggestion[str(i)] for i in range(self.experiment.n_params)])
        reward = self.experiment.optimize(params)
        if self.best_reward is None or reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params

        obs_out = self.carbs.observe(
            ObservationInParam(
                input=suggestion,
                output=reward,
                cost=1,#uptime,
            )
        )
        return params, reward

class GeneticAlgorithm:
    def __init__(self, experiment, mutation_rate=0.1):
        self.experiment = experiment
        self.mutation_rate = mutation_rate
        self.best_reward = None
        self.best_params = np.random.randn(self.experiment.n_params)

    def sample(self):
        mutation = self.mutation_rate*np.random.randn(self.experiment.n_params)
        params = self.best_params + mutation
        reward = self.experiment.optimize(params)
        if self.best_reward is None or reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params

        return params, reward

class WandbSearch:
    def __init__(self, experiment, method='bayes', strategy=None):
        self.experiment = experiment
        self.strategy = strategy

        self.parameters = {f'param_{i}':
            {'distribution': 'normal', 'mu': 0, 'sigma': 1}
            for i in range(10)}

        name = strategy.__class__.__name__ if strategy is not None else method
        self.sweep_id = wandb.sweep(
            sweep=dict(
                method=method,
                name=f'sweep-{name}',
                metric=dict(
                    goal='maximize',
                    name='reward',
                ),
                parameters=self.parameters,
            ),
            project="sweeping",
        )
        self.idx = 0

    def run(self):
        def main():
            self.idx += 1
            wandb.init(name=f'experiment-{self.idx}')
            wandb.config.__dict__['_locked'] = {}
            if self.strategy is not None:
                params, reward = self.strategy.sample()
            else:
                params = np.array([float(wandb.config[k]) for k in self.parameters])
                reward = self.experiment.optimize(params)

            param_dict = dict(zip(self.parameters.keys(), params))
            wandb.config.update(param_dict, allow_val_change=True)
            wandb.log({'reward': reward})

        wandb.agent(self.sweep_id, main, count=100)


if __name__ == '__main__':
    experiment = SyntheticExperiment(10)

    strategy = CARBSSearch(experiment)
    wandb_search = WandbSearch(experiment, strategy=strategy)
    wandb_search.run()

    wandb_search = WandbSearch(experiment, method='random')
    wandb_search.run()

    wandb_search = WandbSearch(experiment, method='bayes')
    wandb_search.run()

    strategy = GeneticAlgorithm(experiment)
    wandb_search = WandbSearch(experiment, strategy=strategy)
    wandb_search.run()
