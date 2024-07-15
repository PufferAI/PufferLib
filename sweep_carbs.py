import sys

from math import log, ceil, floor
def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return int(2**min(possible_results, key= lambda z: abs(x-2**z)))

def sweep_carbs(args, wandb_name, env_module, make_env):
    import wandb
    sweep_id = wandb.sweep(
        sweep=dict(args.sweep),
        project="carbs",
    )
    target_metric = args.sweep['metric']['name'].split('/')[-1]
    wandb_train_params = args.sweep.parameters['train']['parameters']
    wandb_env_params = args.sweep.parameters['env']['parameters']
    wandb_policy_params = args.sweep.parameters['policy']['parameters']

    import numpy as np
    from loguru import logger

    from carbs import CARBS
    from carbs import CARBSParams
    from carbs import LinearSpace
    from carbs import LogSpace
    from carbs import LogitSpace
    from carbs import ObservationInParam
    from carbs import ParamDictType
    from carbs import Param

    logger.remove()
    logger.add(sys.stdout, level="DEBUG", format="{message}")

    def carbs_param(name, space, wandb_params, min=None, max=None,
            search_center=None, is_integer=False, rounding_factor=1):
        wandb_param = wandb_params[name]
        if min is None:
            min = float(wandb_param['min'])
        if max is None:
            max = float(wandb_param['max'])

        if space == 'log':
            Space = LogSpace
            if search_center is None:
                search_center = 2**(np.log2(min) + np.log2(max)/2)
        elif space == 'linear':
            Space = LinearSpace
            if search_center is None:
                search_center = (min + max)/2
        elif space == 'logit':
            Space = LogitSpace
            assert min == 0
            assert max == 1
            assert search_center is not None
        else:
            raise ValueError(f'Invalid CARBS space: {space} (log/linear)')

        return Param(
            name=name,
            space=Space(
                min=min,
                max=max,
                is_integer=is_integer,
                rounding_factor=rounding_factor
            ),
            search_center=search_center,
        )

    # Must be hardcoded and match wandb sweep space for now
    param_spaces = [
        carbs_param('cnn_channels', 'linear', wandb_policy_params, search_center=32, is_integer=True),
        carbs_param('hidden_size', 'linear', wandb_policy_params, search_center=128, is_integer=True),
        #carbs_param('vision', 'linear', search_center=5, is_integer=True),
        #carbs_param('total_timesteps', 'log', wandb_train_params, search_center=1_000_000_000, is_integer=True),
        carbs_param('learning_rate', 'log', wandb_train_params, search_center=9e-4),
        carbs_param('gamma', 'logit', wandb_train_params, search_center=0.99),
        carbs_param('gae_lambda', 'logit', wandb_train_params, search_center=0.90),
        carbs_param('update_epochs', 'linear', wandb_train_params, search_center=1, is_integer=True),
        carbs_param('clip_coef', 'logit', wandb_train_params, search_center=0.1),
        carbs_param('vf_coef', 'logit', wandb_train_params, search_center=0.5),
        carbs_param('vf_clip_coef', 'logit', wandb_train_params, search_center=0.1),
        carbs_param('max_grad_norm', 'linear', wandb_train_params, search_center=0.5),
        carbs_param('ent_coef', 'log', wandb_train_params, search_center=0.07),
        #carbs_param('env_batch_size', 'linear', search_center=384,
        #    is_integer=True, rounding_factor=24),
        carbs_param('batch_size', 'log', wandb_train_params, search_center=262144, is_integer=True),
        carbs_param('minibatch_size', 'log', wandb_train_params, search_center=4096, is_integer=True),
        carbs_param('bptt_horizon', 'log', wandb_train_params, search_center=16, is_integer=True),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=0,
    )
    carbs = CARBS(carbs_params, param_spaces)

    def main():
            args.exp_name = init_wandb(args, wandb_name, id=args.exp_id)
            orig_suggestion = carbs.suggest().suggestion
            suggestion = orig_suggestion.copy()
            print('Suggestion:', suggestion)
            cnn_channels = suggestion.pop('cnn_channels')
            hidden_size = suggestion.pop('hidden_size')
            #vision = suggestion.pop('vision')
            #wandb.config.env['vision'] = vision
            wandb.config.policy['cnn_channels'] = cnn_channels
            wandb.config.policy['hidden_size'] = hidden_size
            wandb.config.train.update(suggestion)
            wandb.config.train['batch_size'] = closest_power(
                suggestion['batch_size'])
            wandb.config.train['minibatch_size'] = closest_power(
                suggestion['minibatch_size'])
            wandb.config.train['bptt_horizon'] = closest_power(
                suggestion['bptt_horizon'])
            #wandb.config.train['num_envs'] = int(
            #    3*suggestion['env_batch_size'])
            args.train.__dict__.update(dict(wandb.config.train))
            #args.env.__dict__['vision'] = vision
            args.policy.__dict__['cnn_channels'] = cnn_channels
            args.policy.__dict__['hidden_size'] = hidden_size
            args.rnn.__dict__['input_size'] = hidden_size
            args.rnn.__dict__['hidden_size'] = hidden_size
            args.track = True
            print(wandb.config.train)
            print(wandb.config.env)
            print(wandb.config.policy)
            try:
                stats, profile = train(args, env_module, make_env)
            except Exception as e:
                is_failure = True
                import traceback
                traceback.print_exc()
            else:
                observed_value = stats[target_metric]
                uptime = profile.uptime

                with open('hypers.txt', 'a') as f:
                    f.write(f'Train: {args.train.__dict__}\n')
                    f.write(f'Env: {args.env.__dict__}\n')
                    f.write(f'Policy: {args.policy.__dict__}\n')
                    f.write(f'RNN: {args.rnn.__dict__}\n')
                    f.write(f'Uptime: {uptime}\n')
                    f.write(f'Value: {observed_value}\n')

                obs_out = carbs.observe(
                    ObservationInParam(
                        input=orig_suggestion,
                        output=observed_value,
                        cost=uptime,
                    )
                )

    wandb.agent(sweep_id, main, count=500)


