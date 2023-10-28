from pdb import set_trace as T
import numpy as np

from pathlib import Path
from io import BytesIO

import hnswlib

from gymnasium import Env, spaces

from pyboy import PyBoy
from pyboy.utils import WindowEvent


# addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
# https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
HP_ADDR =  [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
PARTY_ADDR = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
OPPONENT_LEVEL_ADDR = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
BADGE_1_ADDR = 0xD356

VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B
]


if self.extra_buttons:
    self.valid_actions.extend([
        WindowEvent.PRESS_BUTTON_START,
        WindowEvent.PASS
    ])


RELEASE_ARROW = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP
]

RELEASE_BUTTON = [
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B
]

class PokemonRed(Env):
    def __init__(
            self,
            gb_path=str(Path(__file__).parent / 'pokemon_red.gb'),
            init_state=str(Path(__file__).parent / 'has_pokedex_nballs.state'),
            headless=True,
            similar_frame_dist=2000000.0,
            num_elements=20000,
            vec_dim=5760,
            act_freq=24,
            max_steps=16384,
            downsample_factor=2,
            reward_scale_healing=4,
            reward_scale_level=4,
            reward_scale_explore_threshold=22,
            reward_scale_opponent_level=0.2,
            reward_scale_death=-0.1,
            reward_scale_badge=2,
            reward_scale_knn_pre=0.004,
            reward_scale_knn_post=0.01,
            reward_scale_level_progress=100,
            reward_scale_exploration_progress=160,
        ):
        assert downsample_factor in (1, 2, 4, 8, 16)
        self.headless = headless
        self.act_freq = act_freq
        self.max_steps = max_steps
        self.downsample_factor = downsample_factor
        self.init_state = init_state

        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']

        # Reward
        self.r_healing = reward_scale_healing
        self.r_level = reward_scale_level
        self.r_explore_threshold = reward_scale_explore_threshold
        self.r_opponent_level = reward_scale_opponent_level
        self.r_death = reward_scale_death
        self.r_badge = reward_scale_badge
        self.r_knn_pre = reward_scale_knn_pre
        self.r_knn_post = reward_scale_knn_post
        self.r_level_progress = reward_scale_level_progress
        self.r_exploration_progress = reward_scale_exploration_progress
        
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.total_healing_rew = 0

        self.s_health = 0
        self.s_max_opponent_level = 0
        self.s_base_explore = 0
        self.s_levels_satisfied = False

        # KNN
        self.similar_frame_dist = similar_frame_dist
        self.num_elements = num_elements
        self.vec_dim = vec_dim

        # Gymnasium Metadata
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.action_space = spaces.Discrete(len(VALID_ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(144//downsample_factor, 160//downsample_factor),
            dtype=np.uint8
        )

        self.pyboy = PyBoy(
            gb_path,
            debugging=False,
            disable_input=True,
            window_type='headless' if headless else 'SDL2',
            hide_window=True,
        )
        self.screen = self.pyboy.botsupport_manager().screen()

        if not headless:
            self.pyboy.set_emulation_speed(6)

        self.reset()

    def update_seen_coords(self):
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)
            self.seen_coords = {}

        self.seen_coords[coord_string] = self.step_count

    def reset(self, seed=None):
        self.seed = seed

        # restart game, skipping credits
        #with open(self.init_state, "rb") as f:
        #    self.pyboy.load_state(f)

        if self.use_screen_explore:
            self.init_knn()
        else:
            self.init_map_mem
            
        self.rewards = {
            "healing" : 0,
            "event" : 0,
            "level" : 0,
            "opponent_level" : 0,
            "badges" : 0,
            "exploration" : 0
        }
        
        self.death_count = 0
        self.step_count = 0

        self.prev_health = self.read_hp_fraction()
        self.compute_rewards()
        self.total_reward = sum(self.rewards.values())

        ob = self.render()
        ob = grayscale(ob)
        return ob, {}
    
    def render(self, reduce_res=True):
        game_pixels_render = self.screen.screen_ndarray()

        if not reduce_res:
            return game_pixels_render

        s = self.downsample_factor
        game_pixels_render = game_pixels_render[::s, ::s, :]
        return 255*game_pixels_render.astype(np.uint8)
    
    def step(self, action):
        self.run_action_on_emulator(action)
        self.append_agent_stats(action)
        ob = self.render()
        ob = grayscale(ob)

        # trim off memory from frame for knn index
        if self.use_screen_explore:
            self.update_frame_knn_index(obs_flat)
        else:
            self.update_seen_coords()

        self.update_heal_reward()     
        reward, _ = self.compute_rewards()
        self.s_health = self.read_hp_fraction()

        if self.step_count >= self.max_steps:
            step_limit_reached = True
            info = self.rewards | self.get_agent_stats(action)
        else:
            step_limit_reached = False
            info = self.rewards

        self.step_count += 1
        return ob, reward, False, step_limit_reached, info

    def init_map_mem(self):
        self.seen_coords = {}

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(VALID_ACTIONS[action])
        if self.headless:
            self.pyboy._rendering(False)
            
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(RELEASE_ARROW[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(RELEASE_BUTTON[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if i == self.act_freq - 1:
                self.pyboy._rendering(True)
            self.pyboy.tick()

    def compute_rewards(self):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        self.rewards_old = self.rewards.copy()

        # adds up all event flags, exclude museum ticket
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
        0,)
    
    def get_agent_stats(self, action):
        return {
            'step': self.step_count,
            'x': self.read_m(X_POS_ADDR),
            'y': self.read_m(Y_POS_ADDR),
            'map': self.read_m(MAP_N_ADDR),
            'last_action': action,
            'pcount': self.read_m(0xD163),
            'ptypes': [self.read_m(addr) for addr in PARTY_ADDR],
            'levels': [self.read_m(a) for a in PARTY_LEVEL_ADDR],
            'hp': self.read_hp_fraction(),
            'frames': self.knn_index.get_current_count(),
            'deaths': self.death_count,
            'badge': self.get_badges(),
            'event': self.rewards["event"],
            'healr': 0,
        }

    def compute_rewards(self):
        self.rewards_old = self.rewards.copy()

        # healing reward
        curr_health = self.read_hp_fraction()
        hp_delta = curr_health - self.prev_health
        self.rewards["healing"] = self.r_healing * max(0, hp_delta)

        if curr_health <= 0:
            self.death_count += 1

        self.prev_health = curr_health

        # event reward
        curr_event_rew = max(sum([self.bit_count(self.read_m(i))
            for i in range(0xD747, 0xD886)]) - 13, 0)

        self.rewards["event"] = max(curr_event_rew, self.rewards["event"])

        # level reward
        curr_levels = [max(self.read_m(a) - 2, 0) for a in PARTY_LEVEL_ADDR]
        curr_level_sum =  max(sum(curr_levels) - 4, 0) # subtract starting pokemon level
        if curr_level_sum < self.r_explore_threshold:
            scaled_levels = curr_level_sum
        else:
            scaled_levels = (curr_level_sum - self.r_explore_threshold) / self.r_level
        self.rewards["level"] = max(scaled_levels, self.rewards["level"])

        # opponent level reward
        opp_level = max([self.read_m(a) for a in OPPONENT_LEVEL_ADDR]) - 5
        opp_level_scaled = opp_level * self.r_opponent_level
        self.rewards["opponent_level"] = max(opp_level_scaled, self.rewards["opponent_level"])

        # death penalty
        self.rewards["death"] = self.r_death * self.death_count

        # badge reward
        self.rewards["badges"] = self.r_badge * self.get_badges()

        # exploration reward
        pre_rew = self.explore_weight * 0.005
        post_rew = self.explore_weight * 0.01
        cur_size = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        curr_size = self.knn_index.get_current_count()
        if self.s_levels_satisfied:
            base = self.s_base_explore
            post = curr_size * self.r_knn_post
        else:
            base = curr_size * self.r_knn_pre
            post = 0
        self.rewards["exploration"] = base + post

        total_reward = sum(self.rewards.values())
        step_reward = total_reward - sum(self.rewards_old.values())

        return step_reward, (
            self.r_level_progress * (self.rewards["level"] - self.rewards_old["level"]),
            0,
            self.r_exploration_progress * (self.rewards["exploration"] - self.rewards_old["exploration"]),
        )

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)
   
    def get_badges(self):
        return self.bit_count(self.read_m(BADGE_1_ADDR))

    def read_hp_fraction(self):
        hp_sum = sum([self.read_hp(add) for add in HP_ADDR])
        max_hp_sum = sum([self.read_hp(add) for add in MAX_HP_ADDR])
        if max_hp_sum == 0:
            return 0
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    def bit_count(self, bits):
        '''built-in since python 3.10'''
        return bin(bits).count('1')

    def init_knn(self):
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim)
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)

    def update_frame_knn_index(self, frame):
        curr_levels = [max(self.read_m(a) - 2, 0) for a in PARTY_LEVEL_ADDR]
        curr_level_sum =  max(sum(curr_levels) - 4, 0) # subtract starting pokemon level

        if curr_level_sum >= 22 and not self.s_levels_satisfied:
            self.s_levels_satisfied = True
            self.s_base_explore = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame.ravel(), np.array([0]))
        else:
            # check for nearest frame and add if current 
            frame = frame.ravel()
            labels, distances = self.knn_index.knn_query(frame, k = 1)
            if distances[0] > self.similar_frame_dist:
                count = np.array([self.knn_index.get_current_count()])
                self.knn_index.add_items(frame, count)

def grayscale(frame):
    return np.dot(frame[...,:3], [0.299, 0.587, 0.114])
