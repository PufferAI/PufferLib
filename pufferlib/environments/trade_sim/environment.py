import functools
import numpy as np

import pufferlib

from nof1.simulation.env import TradingEnvironment

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config_path='../nof1-trading-sim/config/experiment_config_3.yaml', render_mode='human', buf=None, seed=1):
    '''Crafter creation function'''
    from nof1.utils.config_manager import ConfigManager
    from nof1.data_ingestion.historical_data_reader import HistoricalDataReader

    config_manager = ConfigManager(config_path)
    config = config_manager.config
    data_reader = HistoricalDataReader(config_manager)
    states, prices, atrs, timestamps = data_reader.preprocess_data()
    
    # Create environment
    env = TradingEnvironmentPuff(config_manager.config, states=states, prices=prices, atrs=atrs, timestamps=timestamps)
    return pufferlib.emulation.GymnasiumPufferEnv(env, buf=buf)

class TradingEnvironmentPuff(TradingEnvironment):
    def reset(self):
        obs, info = super().reset()
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if not terminated and not truncated:
            info = {}

        return obs.astype(np.float32), reward, terminated, truncated, info

