from pdb import set_trace as T
import pufferlib.emulation

from pufferlib.registry.pokemon_red.pokemon_red import PokemonRed as env_creator


def make_env(
        headless: bool = True,
        save_video: bool = False,
        use_screen_explore=True,
        sim_frame_dist=2_000_000.0,
        init_state='has_pokedex_nballs.state',
    ):
    '''Pokemon Red'''
    env = env_creator(headless=headless, save_video=save_video,
        use_screen_explore=use_screen_explore, sim_frame_dist=sim_frame_dist,
        init_state=init_state
    )
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)


class PokemonRedWrapper:
    def __init__(self, *args, **kwargs):
        self.env = env_creator(*args, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
