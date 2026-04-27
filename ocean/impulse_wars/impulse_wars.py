from types import SimpleNamespace

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.impulse_wars import binding


discMoveToContMove = np.array([
    [1.0, 0.707107, 0.0, -0.707107, -1.0, -0.707107, 0.0, 0.707107, 0.0],
    [0.0, 0.707107, 1.0, 0.707107, 0.0, -0.707107, -1.0, -0.707107, 0.0],
], dtype=np.float32)
discAimToContAim = np.array([
    [1.0, 0.92388, 0.707107, 0.382683, 0.0, -0.382683, -0.707107, -0.92388, -1.0, -0.92388, -0.707107, -0.382683, 0.0, 0.382683, 0.707107, 0.92388, 0.0],
    [0.0, 0.382683, 0.707107, 0.92388, 1.0, 0.92388, 0.707107, 0.382683, 0.0, -0.382683, -0.707107, -0.92388, -1.0, -0.92388, -0.707107, -0.382683, 0.0],
], dtype=np.float32)


class ImpulseWars(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs: int = 1,
        num_drones: int = 2,
        num_agents: int = 1,
        enable_teams: bool = False,
        sitting_duck: bool = False,
        continuous: bool = False,
        is_training: bool = True,
        human_control: bool = False,
        reward_win: float = 2.0,
        reward_self_kill: float = -1.0,
        reward_enemy_death: float = 1.0,
        reward_enemy_kill: float = 1.0,
        reward_death: float = -0.25,
        reward_energy_emptied: float = -0.75,
        reward_weapon_pickup: float = 0.5,
        reward_shield_break: float = 0.5,
        reward_shot_hit_coef: float = 0.005, 
        reward_explosion_hit_coef: float = 0.005,
        seed: int = 0,
        render: bool = False,
        report_interval: int = 64,
        buf = None,
    ):
        self.obsInfo = SimpleNamespace(**binding.get_consts(num_drones))

        if num_envs <= 0:
            raise ValueError("num_envs must be greater than 0")
        if num_drones > self.obsInfo.maxDrones or num_drones <= 0:
            raise ValueError(f"num_drones must greater than 0 and less than or equal to {self.obsInfo.maxDrones}")
        if num_agents > num_drones or num_agents <= 0:
            raise ValueError("num_agents must greater than 0 and less than or equal to num_drones")
        if enable_teams and (num_drones % 2 != 0 or num_drones <= 2):
            raise ValueError("enable_teams is only supported for even numbers of drones greater than 2")

        self.numDrones = num_drones
        self.continuous = continuous

        self.num_agents = num_agents * num_envs
        self.tick = 0

        # map observations are bit packed to save space, and scalar
        # observations need to be floats
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obsInfo.obsBytes,), dtype=np.uint8
        )

        if self.continuous:
            # action space is actually bounded by (-1, 1) but pufferlib
            # will check that actions are within the bounds of the action
            # space before actions get to the env, and we ensure the actions
            # are bounded there; so set bounds to (-inf, inf) here so
            # action bounds checks pass
            self.single_action_space = gymnasium.spaces.Box(
                low=float("-inf"), high=float("inf"), shape=(self.obsInfo.contActionsSize,), dtype=np.float32
            )
        else:
            self.single_action_space = gymnasium.spaces.MultiDiscrete(
                [
                    9,  # move, noop + 8 directions
                    17,  # aim, noop + 16 directions
                    2,  # shoot or not
                    2,  # brake or not
                    2,  # burst
                ]
            )

        self.report_interval = report_interval
        self.render_mode = "human" if render else None

        super().__init__(buf)
        if not self.continuous:
            self.actions = np.zeros((self.num_agents, self.obsInfo.contActionsSize), dtype=np.float32)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            num_drones=num_drones,
            num_agents=num_agents,
            map_idx=-1,
            enable_teams=enable_teams,
            sitting_duck=sitting_duck,
            is_training=is_training,
            continuous=continuous,
            reward_win=reward_win,
            reward_self_kill=reward_self_kill,
            reward_enemy_death=reward_enemy_death,
            reward_enemy_kill=reward_enemy_kill,
            reward_death=reward_death,
            reward_energy_emptied=reward_energy_emptied,
            reward_weapon_pickup=reward_weapon_pickup,
            reward_shield_break=reward_shield_break,
            reward_shot_hit_coef=reward_shot_hit_coef,
            reward_explosion_hit_coef=reward_explosion_hit_coef,
        )

        binding.shared(self.c_envs)

    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        if self.continuous:
            self.actions[:] = actions
        else:
            contMove = discMoveToContMove[:, actions[:, 0]].T
            contAim =  discAimToContAim[:, actions[:, 1]].T
            contRest = actions[:, 2:].astype(np.float32)
            self.actions[:] = np.concatenate([contMove, contAim, contRest], axis=1)

        self.tick += 1    
        binding.vec_step(self.c_envs)

        infos = []
        if self.tick % self.report_interval == 0:
            infos.append(binding.vec_log(self.c_envs))

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def testPerf(timeout, actionCache, numEnvs):
    env = ImpulseWars(numEnvs)

    import time

    np.random.seed(int(time.time()))
    actions = np.random.uniform(-1, 1, (actionCache, env.num_agents, 7))

    tick = 0
    start = time.time()
    while time.time() - start < timeout:
        action = actions[tick % actionCache]
        env.step(action)
        tick += 1

    sps = numEnvs * (tick / (time.time() - start))
    print(f"SPS: {sps:,}")
    print(f"Steps: {numEnvs * tick}")

    env.close()


if __name__ == "__main__":
    testPerf(timeout=5, actionCache=1024, numEnvs=1)
