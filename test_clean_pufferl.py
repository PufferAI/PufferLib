from pdb import set_trace as T
import numpy as np
import sys

from pufferlib.registry import nmmo

import pufferlib.emulation
import pufferlib.models
import pufferlib.registry.nmmo

from tests.mock_environments import MOCK_ENVIRONMENTS
from clean_pufferl import CleanPuffeRL

# TODO: Remove this
class FeatureExtractor(pufferlib.emulation.Postprocessor):
    def features(self, obs, step):
        assert len(obs) > 0
        values = list(obs.values())

        ob1 = values[0]
        if len(obs) == 1:
            return (ob1, ob1)
        else:
            return (ob1, values[1])


for env_cls in MOCK_ENVIRONMENTS:
    binding = pufferlib.emulation.Binding(
        env_cls=env_cls,
        env_name=env_cls.__name__,
        # TODO: Handle variable team sizes via padded action space
        teams = {
            'team_1': ['agent_1', 'agent_2'],
            'team_2': ['agent_3', 'agent_4'],
            'team_3': ['agent_5', 'agent_6'],
            'team_4': ['agent_7', 'agent_8'],
            'team_5': ['agent_9', 'agent_10'],
            'team_6': ['agent_11', 'agent_12'],
            'team_7': ['agent_13', 'agent_14'],
            'team_8': ['agent_15', 'agent_16'],
        },
        #postprocessor_cls=FeatureExtractor,
    )

    device = 'cuda'

    """
    import nmmo
    binding = pufferlib.emulation.Binding(
            env_cls=nmmo.Env,
            env_name='Neural MMO',
            emulate_const_horizon=128,
            #teams={f'team_{i+1}': [i*8+j+1 for j in range(8)] for i in range(16)},
        )
    """

    envs = pufferlib.vectorization.serial.VecEnv(
        binding,
        num_workers=1,
        envs_per_worker=1,
    )

    # TODO: Better check on recurrent arg shape
    agent = pufferlib.frameworks.cleanrl.make_policy(
            pufferlib.models.Default, recurrent_args=[4, 4],
            recurrent_kwargs={'num_layers': 1}
        )(binding, 4, 4).to(device)

    # TODO: Check on num_agents
    trainer = CleanPuffeRL(binding, agent, num_envs=4, num_steps=128, num_cores=2,    
            vec_backend=pufferlib.vectorization.serial.VecEnv)
    #trainer.load_model(path)

    data = trainer.allocate_storage()

    num_updates = 2
    for update in range(trainer.update+1, num_updates + 1):
        trainer.evaluate(agent, data)
        
        #Assert that data is correct
        ob = data.obs.cpu().numpy()
        ob1 = ob[:-1]
        ob2 = ob[1:]
        diff = ob2 - ob1
        diff_mask = np.abs(diff) < 0.5
        assert_mask = diff_mask | (ob1==0) | (ob2==0)
        assert assert_mask.all()

        orig_obs = data.obs
        trainer.train(agent, data)

    trainer.close()