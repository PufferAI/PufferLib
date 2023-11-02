from pdb import set_trace as T
import pufferlib.emulation

from pufferlib.environments import PokemonRed as env_creator


def make_env(
        headless: bool = True,
        save_video: bool = False,
        max_steps=2048*10, 
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
