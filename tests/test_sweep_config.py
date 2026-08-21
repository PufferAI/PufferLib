import ast
import configparser
from collections import defaultdict

import pufferlib.sweep


def _load_sweep_config(*paths):
    parser = configparser.ConfigParser()
    parser.read(paths)

    args = defaultdict(dict)
    for section in parser.sections():
        for key in parser[section]:
            try:
                value = ast.literal_eval(parser[section][key])
            except Exception:
                value = parser[section][key]

            name = key if section == 'base' else f'{section}.{key}'
            cursor = args
            for subkey in name.split('.'):
                previous = cursor
                cursor = cursor.setdefault(subkey, {})
            previous[subkey] = value

    return args['sweep']


def test_match_sweep_config_keys_are_not_hyperparameters():
    sweep_config = _load_sweep_config('config/default.ini', 'config/breakout.ini')
    spaces = pufferlib.sweep._params_from_puffer_sweep(sweep_config)

    assert 'match_enemy_model_path' not in spaces
    assert 'match_num_games' not in spaces
    assert 'train' in spaces
