import pufferlib.new_emulation
import pufferlib.emulation
import pufferlib.registry.nmmo

import nmmo

teams = {i: [i] for i in range(1, 129)}
postprocessor_cls = pufferlib.emulation.Postprocessor
env = pufferlib.registry.nmmo.PufferNMMO(teams, postprocessor_cls)
obs = env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
obs, rewards, dones, infos = env.step(actions)