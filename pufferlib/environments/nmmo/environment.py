from pdb import set_trace as T
import numpy as np
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers
import pufferlib.postprocess


def env_creator(name='nmmo'):
    return functools.partial(make, name)

def make(name, *args, buf=None, **kwargs):
    '''Neural MMO creation function'''
    nmmo = pufferlib.environments.try_import('nmmo')
    env = nmmo.Env(*args, **kwargs)
    env = NMMOWrapper(env)
    env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    env = pufferlib.postprocess.MeanOverAgents(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env, buf=buf)

class NMMOWrapper(pufferlib.postprocess.PettingZooWrapper):
    '''Remove task spam'''
    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self):
        '''Quick little renderer for NMMO'''
        tiles = self.env.tile_map[:, :, 2].astype(np.uint8)
        render = np.zeros((tiles.shape[0], tiles.shape[1], 3), dtype=np.uint8)
        BROWN = (136, 69, 19)
        render[tiles == 1] = (0, 0, 255)
        render[tiles == 2] = (0, 255, 0)
        render[tiles == 3] = BROWN
        render[tiles == 4] = (64, 255, 64)
        render[tiles == 5] = (128, 128, 128)
        render[tiles == 6] = BROWN
        render[tiles == 7] = (255, 128, 128)
        render[tiles == 8] = BROWN
        render[tiles == 9] = (128, 255, 128)
        render[tiles == 10] = BROWN
        render[tiles == 11] = (128, 128, 255)
        render[tiles == 12] = BROWN
        render[tiles == 13] = (192, 255, 192)
        render[tiles == 14] = (0, 0, 255)
        render[tiles == 15] = (64, 64, 255)

        for agent in self.env.realm.players.values():
            agent_r = agent.row.val
            agent_c = agent.col.val
            render[agent_r, agent_c, :] = (255, 255, 0)

        for npc in self.env.realm.npcs.values():
            agent_r = npc.row.val
            agent_c = npc.col.val
            render[agent_r, agent_c, :] = (255, 0, 0)

        return render

    def reset(self, seed=None):
        obs, infos = self.env.reset(seed=seed)
        self.obs = obs
        return obs, infos

    def step(self, actions):
        obs, rewards, dones, truncateds, infos = self.env.step(actions)
        infos = {k: list(v['task'].values())[0] for k, v in infos.items()}
        self.obs = obs
        return obs, rewards, dones, truncateds, infos

    def close(self):
        return self.env.close()
