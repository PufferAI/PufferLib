from pdb import set_trace as T
import numpy as np
from types import SimpleNamespace
import gymnasium
import pettingzoo
import time

from pufferlib.ocean.nmmo3.cy_nmmo3 import Environment, entity_dtype, reward_dtype

import pufferlib

class NMMO3(pufferlib.PufferEnv):
    def __init__(self, width=4*[512], height=4*[512], num_envs=4,
            num_players=1024, num_enemies=2048, num_resources=2048,
            num_weapons=1024, num_gems=512, tiers=5, levels=40,
            teleportitis_prob=0.001, enemy_respawn_ticks=2,
            item_respawn_ticks=100, x_window=7, y_window=5,
            reward_combat_level=1.0, reward_prof_level=1.0,
            reward_item_level=0.5, reward_market=0.01,
            reward_death=-1.0, buf=None):

        if len(width) > num_envs:
            width = width[:num_envs]
        if len(height) > num_envs:
            height = height[:num_envs]

        if not isinstance(width, list):
            width = num_envs * [width]
        if not isinstance(height, list):
            height = num_envs * [height]
        if not isinstance(num_players, list):
            num_players = num_envs * [num_players]
        if not isinstance(num_enemies, list):
            num_enemies = num_envs * [num_enemies]
        if not isinstance(num_resources, list):
            num_resources = num_envs * [num_resources]
        if not isinstance(num_weapons, list):
            num_weapons = num_envs * [num_weapons]
        if not isinstance(num_gems, list):
            num_gems = num_envs * [num_gems]
        if not isinstance(tiers, list):
            tiers = num_envs * [tiers]
        if not isinstance(levels, list):
            levels = num_envs * [levels]
        if not isinstance(teleportitis_prob, list):
            teleportitis_prob = num_envs * [teleportitis_prob]
        if not isinstance(enemy_respawn_ticks, list):
            enemy_respawn_ticks = num_envs * [enemy_respawn_ticks]
        if not isinstance(item_respawn_ticks, list):
            item_respawn_ticks = num_envs * [item_respawn_ticks]

        assert isinstance(width, list)
        assert isinstance(height, list)
        assert isinstance(num_players, list)
        assert isinstance(num_enemies, list)
        assert isinstance(num_resources, list)
        assert isinstance(num_weapons, list)
        assert isinstance(num_gems, list)
        assert isinstance(tiers, list)
        assert isinstance(levels, list)
        assert isinstance(teleportitis_prob, list)
        assert isinstance(enemy_respawn_ticks, list)
        assert isinstance(item_respawn_ticks, list)
        assert isinstance(x_window, int)
        assert isinstance(y_window, int)

        assert len(width) == num_envs
        assert len(height) == num_envs
        assert len(num_players) == num_envs
        assert len(num_enemies) == num_envs
        assert len(num_resources) == num_envs
        assert len(num_weapons) == num_envs
        assert len(num_gems) == num_envs
        assert len(tiers) == num_envs
        assert len(levels) == num_envs
        assert len(teleportitis_prob) == num_envs
        assert len(enemy_respawn_ticks) == num_envs
        assert len(item_respawn_ticks) == num_envs

        total_players = 0
        total_enemies = 0
        for idx in range(num_envs):
            assert isinstance(width[idx], int)
            assert isinstance(height[idx], int)

            if num_players[idx] is None:
                num_players[idx] = width[idx] * height[idx] // 2048
            if num_enemies[idx] is None:
                num_enemies[idx] = width[idx] * height[idx] // 512
            if num_resources[idx] is None:
                num_resources[idx] = width[idx] * height[idx] // 1024
            if num_weapons[idx] is None:
                num_weapons[idx] = width[idx] * height[idx] // 2048
            if num_gems[idx] is None:
                num_gems[idx] = width[idx] * height[idx] // 4096
            if tiers[idx] is None:
                if height[idx] <= 128:
                    tiers[idx] = 1
                elif height[idx] <= 256:
                    tiers[idx] = 2
                elif height[idx] <= 512:
                    tiers[idx] = 3
                elif height[idx] <= 1024:
                    tiers[idx] = 4
                else:
                    tiers[idx] = 5
            if levels[idx] is None:
                if height[idx] <= 128:
                    levels[idx] = 7
                elif height[idx] <= 256:
                    levels[idx] = 15
                elif height[idx] <= 512:
                    levels[idx] = 31
                elif height[idx] <= 1024:
                    levels[idx] = 63
                else:
                    levels[idx] = 99

            total_players += num_players[idx]
            total_enemies += num_enemies[idx]

        self.players_flat = np.zeros((total_players, 51+501+3), dtype=np.intc)
        self.enemies_flat = np.zeros((total_enemies, 51+501+3), dtype=np.intc)
        self.rewards_flat = np.zeros((total_players, 10), dtype=np.float32)
        #map_obs = np.zeros((total_players, 11*15 + 47 + 10), dtype=np.intc)
        #counts = np.zeros((num_envs, height, width), dtype=np.uint8)
        #terrain = np.zeros((num_envs, height, width), dtype=np.uint8)
        #rendered = np.zeros((num_envs, height, width, 3), dtype=np.uint8)
        actions = np.zeros((total_players), dtype=np.intc)
        self.actions = actions

        self.num_agents = total_players
        self.num_players = total_players
        self.num_enemies = total_enemies

        self.players = np.frombuffer(self.players_flat,
            dtype=entity_dtype()).view(np.recarray)
        self.enemies = np.frombuffer(self.enemies_flat,
            dtype=entity_dtype()).view(np.recarray)
        self.struct_rewards = np.frombuffer(self.rewards_flat,
            dtype=reward_dtype()).view(np.recarray)

        self.comb_goal_mask = np.array([1, 0, 1, 0, 1, 1, 0, 1, 1, 1])
        self.prof_goal_mask = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1])
        self.tick = 0

        self.single_observation_space = gymnasium.spaces.Box(low=0,
            high=255, shape=(11*15*10+47+10,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(26)
        self.render_mode = 'human'

        super().__init__(buf)
        self.c_env = Environment(self.observations, self.players_flat,
            self.enemies_flat, self.rewards_flat, self.actions,
            width, height, num_envs, num_players, num_enemies,
            num_resources, num_weapons, num_gems, tiers, levels,
            teleportitis_prob, enemy_respawn_ticks, item_respawn_ticks,
            reward_combat_level, reward_prof_level, reward_item_level,
            reward_market, reward_death, x_window, y_window)

    def reset(self, seed=None):
        self.struct_rewards.fill(0)
        self.rewards.fill(0)
        self.is_reset = True
        self.c_env.reset()
        return self.observations, []

    def step(self, actions):
        if not hasattr(self, 'is_reset'):
            raise Exception('Must call reset before step')
        self.rewards.fill(0)
        rewards = self.struct_rewards
        rewards.fill(0)
        self.actions[:] = actions[:]
        self.c_env.step()

        rewards = rewards.total
        infos = []
        if self.tick % 128 == 0:
            log = self.c_env.log()
            if log['episode_length'] > 0:
                infos.append(log)

            '''
            print(
                f'Comb lvl: {np.mean(self.players.comb_lvl)} (max {np.max(self.players.comb_lvl)})',
                f'Prof lvl: {np.mean(self.players.prof_lvl)} (max {np.max(self.players.prof_lvl)})',
                f'Time alive: {np.mean(self.players.time_alive)} (max {np.max(self.players.time_alive)})',
            )
            '''

        if False and self.tick % 128 == 0:
            # TODO: Log images to Wandb in latest version
            infos['nmmo3_map'] = self.render()

        self.tick += 1

        self.rewards[:] = rewards.ravel()

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def render(self):
        self.c_env.render()
        #all_maps = [e.rendered.astype(np.float32) for e in self.c_env.envs]
        #all_counts = [e.counts.astype(np.float32) for e in self.c_env.envs]

        '''
        agg_maps = np.zeros((2048, 2048, 3), dtype=np.float32)
        agg_counts = np.zeros((2048, 2048), dtype=np.float32)

        agg_maps[256:512, :1024] = all_maps[0]
        agg_counts[256:512, :1024] = all_counts[0]

        agg_maps[512:1024, :1024] = all_maps[1]
        agg_counts[512:1024, :1024] = all_counts[1]

        agg_maps[1024:2048, :1024] = all_maps[2]
        agg_counts[1024:2048, :1024] = all_counts[2]

        agg_maps[:, 1024:] = all_maps[3]
        agg_counts[:, 1024:] = all_counts[3]

        agg_maps = all_maps[0]
        agg_counts = all_counts[0]

        map = agg_maps
        counts = agg_counts.astype(np.float32)/255

        # Lerp rendered with counts
        #counts = self.c_env.counts.astype(np.float32)/255
        counts = np.clip(25*counts, 0, 1)[:, :, None]

        lerped = map * (1 - counts) + counts * np.array([0, 255, 255])

        num_players = self.num_players
        r = self.players.r
        c = self.players.c
        lerped[r[:num_players], c[:num_players]] = np.array([0, 0, 0])

        num_enemies = self.num_enemies
        r = self.enemies.r
        c = self.enemies.c
        lerped[r[:num_enemies], c[:num_enemies]] = np.array([255, 0, 0])

        lerped = lerped[::2, ::2]

        return lerped.astype(np.uint8)
        '''

    def close(self):
        self.c_envs.close()

class Overlays:
    def __init__(self, width, height):
        self.counts = np.zeros((width, height), dtype=int)
        self.value_function = np.zeros((width, height), dtype=np.float32)

def test_env_performance(env, timeout=10):
    num_agents = env.num_players

    actions = {t:
        {agent: np.random.randint(0, 6) for agent in range(1, num_agents+1)}
        for t in range(100)
    }
    actions = {t: np.random.randint(0, 6, num_agents) for t in range(100)}
    idx = 0

    import time
    start = time.time()
    num_steps = 0
    while time.time() - start < timeout:
        env.step(actions[num_steps % 100])
        num_steps += 1

    end = time.time()
    fps = num_agents * num_steps / (end - start)
    print(f"Test Environment Performance FPS: {fps:.2f}")


