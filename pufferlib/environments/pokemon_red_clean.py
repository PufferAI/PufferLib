from pdb import set_trace as T

import sys
import uuid 
from math import floor, sqrt
from pathlib import Path

import numpy as np
from einops import rearrange
from skimage.transform import resize
from pyboy import PyBoy
import hnswlib

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

ENV_CFG = {
    "rewards" : {
        "healing_scale" : 4,
        "level_scale" : 4,
        "explore_threshold" : 22,
        "opponent_level_scale" : 0.2,
        "death_scale" : -0.1,
        "badge_scale" : 2,
        "knn_pre_scale" :  0.004,
        "knn_post_scale" : 0.01,
        "level_progress_scale" : 100,
        "exploration_progress_scale" : 160,
        "reward_step_scale" : 0.1,
    },
    "reward_params" : {
            "max_event_rew" : 0,
            "max_level_rew" : 0,
            "total_healing_rew" : 0,
    },
    "state_params" : {
            "health" : 0,
            "max_opponent_level" : 0,
            "base_explore" : 0,
            "levels_satisfied" : False,
    },
}

VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
    WindowEvent.PASS
]

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
            vec_dim=4320,
            headless=True,
            num_elements=20000,
            init_state=str(Path(__file__).parent / 'has_pokedex_nballs.state'),
            act_freq=24,
            max_steps=16384,
            early_stopping=False,
            downsample_factor=2,
            frame_stacks=3,
            similar_frame_dist=2000000.0,
            instance_id=None,
            env_cfg=ENV_CFG,
        ):
        self.vec_dim = vec_dim
        self.headless = headless
        self.num_elements = num_elements
        self.init_state = init_state
        self.act_freq = act_freq
        self.max_steps = max_steps
        self.early_stopping = early_stopping
        self.downsample_factor = downsample_factor
        self.frame_stacks = frame_stacks
        self.similar_frame_dist = similar_frame_dist
        self.reset_count = 0 #previously passed as arg?
        self.cfg = env_cfg

        self.instance_id = instance_id
        if instance_id is None:
            self.instance_id = str(uuid.uuid4())[:8]

        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2],
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(VALID_ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        head = 'headless' if headless else 'SDL2'

        self.pyboy = PyBoy(
            gb_path,
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )

        self.screen = self.pyboy.botsupport_manager().screen()
        self.pyboy.set_emulation_speed(0 if headless else 6)
        self.reset()

    def reset(self, seed=None):
        self.seed = seed

        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_knn()

        self.recent_memory = np.zeros(
            (self.output_shape[1]*self.memory_height, 3),
            dtype=np.uint8,
        )
        self.recent_frames = np.zeros((
            self.frame_stacks, self.output_shape[0], 
            self.output_shape[1], self.output_shape[2]
            ), dtype=np.uint8,
        )

        self.rewards = {
            "healing" : 0,
            "event" : 0,
            "level" : 0,
            "opponent_level" : 0,
            #"opponent_level_raw" : 0,
            "badges" : 0,
            "exploration" : 0
        }
        
        self.death_count = 0
        self.step_count = 0
        self.reset_count += 1

        self.compute_rewards()
        self.total_reward = sum(self.rewards.values())

        return self.render(), {}
    
    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim) # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)

        if not reduce_res:
            return game_pixels_render

        game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)

        if update_mem:
            self.recent_frames[0] = game_pixels_render

        if not add_memory:
            return game_pixels_render

        pad = np.zeros(
            shape=(self.mem_padding, self.output_shape[1], 3), 
            dtype=np.uint8
        )
        game_pixels_render = np.concatenate((
            self.create_exploration_memory(), 
            pad,
            self.create_recent_memory(),
            pad,
            rearrange(self.recent_frames, 'f h w c -> (f h) w c')
        ), axis=0)

        return game_pixels_render
    
    def step(self, action):
        self.run_action_on_emulator(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = obs_memory[
            frame_start:frame_start+self.output_shape[0], ...].flatten().astype(np.float32)

        self.update_frame_knn_index(obs_flat)
        new_reward, new_prog = self.compute_rewards()
        self.cfg["state_params"]["health"] = self.read_hp_fraction()

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()
        self.step_count += 1

        info = self.rewards | self.get_agent_stats(action) if step_limit_reached else self.rewards
        return obs_memory, self.cfg["rewards"]["reward_step_scale"] * new_reward, False, step_limit_reached, info

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(VALID_ACTIONS[action])
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(RELEASE_ARROW[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(RELEASE_BUTTON[action - 4])
                if action == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            self.pyboy.tick()
    
    def get_agent_stats(self, action):
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)
        levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return {
            'step': self.step_count, 'x': x_pos, 'y': y_pos, 'map': map_n,
            'last_action': action,
            'pcount': self.read_m(0xD163), 'levels': levels, 'ptypes': self.read_party(),
            'hp': self.read_hp_fraction(),
            'frames': self.knn_index.get_current_count(),
            'deaths': self.death_count, 'badge': self.get_badges(),
            'event': self.rewards["event"], 'healr': self.cfg["reward_params"]["total_healing_rew"]
        }

    def update_frame_knn_index(self, frame_vec):
        curr_levels = [max(self.read_m(a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        curr_level_sum =  max(sum(curr_levels) - 4, 0) # subtract starting pokemon level
        if curr_level_sum >= 22 and not self.cfg["state_params"]["levels_satisfied"]:
            self.cfg["state_params"]["levels_satisfied"] = True
            self.cfg["state_params"]["base_explore"] = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()]))
        else:
            # check for nearest frame and add if current 
            labels, distances = self.knn_index.knn_query(frame_vec, k = 1)
            if distances[0] > self.similar_frame_dist:
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()]))

    def compute_rewards(self):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        self.rewards_old = self.rewards.copy()

        # healing reward
        curr_health = self.read_hp_fraction()
        self.rewards["healing"] = self.cfg["rewards"]["healing_scale"] * max(0, curr_health - self.cfg["state_params"]["health"])
        if self.cfg["state_params"]["health"] <= 0: self.death_count += 1
        self.cfg["state_params"]["health"] = curr_health

        # event reward
        curr_event_rew = max(sum([self.bit_count(self.read_m(i)) for i in range(0xD747, 0xD886)]) - 13, 0)
        self.rewards["event"] = max(curr_event_rew, self.rewards["event"])

        # level reward
        curr_levels = [max(self.read_m(a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        curr_level_sum =  max(sum(curr_levels) - 4, 0) # subtract starting pokemon level
        scaled_levels = curr_level_sum if curr_level_sum < self.cfg["rewards"]["explore_threshold"] else (curr_level_sum - self.cfg["rewards"]["explore_threshold"]) / self.cfg["rewards"]["level_scale"] + self.cfg["rewards"]["explore_threshold"]
        self.rewards["level"] = max(scaled_levels, self.rewards["level"])

        # opponent level reward
        opp_level = max([self.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
        opp_level_scaled = opp_level * self.cfg["rewards"]["opponent_level_scale"]
        self.rewards["opponent_level"] = max(opp_level_scaled, self.rewards["opponent_level"])
        #self.rewards["opponent_level_raw"] = max(opp_level, self.rewards["opponent_level_raw"])

        # death penalty
        self.rewards["death"] = self.cfg["rewards"]["death_scale"] * self.death_count

        # badge reward
        self.rewards["badges"] = self.cfg["rewards"]["badge_scale"] * self.get_badges()

        # exploration reward
        curr_size = self.knn_index.get_current_count()
        base = (self.cfg["state_params"]["base_explore"] if self.cfg["state_params"]["levels_satisfied"] else curr_size) *  self.cfg["rewards"]["knn_pre_scale"]
        post = (curr_size if self.cfg["state_params"]["levels_satisfied"] else 0) * self.cfg["rewards"]["knn_post_scale"]
        self.rewards["exploration"] = base + post

        total_reward = sum(self.rewards.values())
        step_reward = total_reward - sum(self.rewards_old.values())

        return step_reward, (
            self.cfg["rewards"]["level_progress_scale"] * (self.rewards["level"] - self.rewards_old["level"]),
            0,
            self.cfg["rewards"]["exploration_progress_scale"] * (self.rewards["exploration"] - self.rewards_old["exploration"]),
        )

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height
        
        def make_reward_channel(r_val):
            col_steps = self.col_steps
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        
        level, hp, explore = (
            self.cfg["rewards"]["level_progress_scale"] * self.rewards["level"],
            0,
            self.cfg["rewards"]["exploration_progress_scale"] * self.rewards["exploration"],
        )
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)
        
        if self.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def create_recent_memory(self):
        return rearrange(
            self.recent_memory, 
            '(w h) c -> h w c', 
            h=self.memory_height,
        )

    def check_if_done(self):
        if self.early_stopping:
            return self.step_count > 128 and self.recent_memory.sum() < (255 * 1)
        return self.step_count >= self.max_steps
            
    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]

    def read_hp_fraction(self):
        hp_sum = sum([self.read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
        max_hp_sum = sum([self.read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    def bit_count(self, bits):
        '''built-in since python 3.10'''
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    def read_money(self):
        return (
            100 * 100 * self.read_bcd(self.read_m(0xD347)) + 
            100 * self.read_bcd(self.read_m(0xD348)) +
            self.read_bcd(self.read_m(0xD349))
        )
