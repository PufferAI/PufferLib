import numpy as np
import gymnasium
import os
from raylib import rl
import heapq
import math

import pufferlib
from pufferlib.environments.ocean.highway.c_highway import CHighway, step_all
from pufferlib.environments.ocean import render


class PufferHighway:
    def __init__(self, num_envs=200, render_mode='human'):
        self.num_envs = num_envs
        self.render_mode = render_mode

        # sim hparams (to put in config file)
        self.agents_per_env = 10
        self.cars_per_env = 50
        total_agents = self.num_envs * self.agents_per_env

        self.car_width = 2  # m
        self.car_length = 5  # m

        self.dt = 0.05
        self.max_speed = 35

        # env spec
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = self.num_envs
        self.render_mode = render_mode
        self.emulated = None
        self.done = False
        self.buf = pufferlib.namespace(
            observations = np.zeros(
                (total_agents, 3), dtype=np.float32),
            rewards = np.zeros(total_agents, dtype=np.float32),
            terminals = np.zeros(total_agents, dtype=bool),
            truncations = np.zeros(total_agents, dtype=bool),
            masks = np.ones(total_agents, dtype=bool),
        )
        self.actions = np.zeros(total_agents, dtype=np.float32)

        # env storage
        # veh position is the front bumper position (back bumper is at x = front_bumper - car_length)
        self.veh_positions = np.zeros((self.num_envs, self.cars_per_env), dtype=np.float32)
        self.veh_speeds = np.zeros((self.num_envs, self.cars_per_env), dtype=np.float32)
        self.t = 0


        # render
        if render_mode == 'human':
            self.client = RaylibClient(car_width=self.car_width, car_length=self.car_length)
    
    def reset(self, seed=None):

        self.veh_speeds[:,:] = 20
        self.t = 0

        self.c_envs = []
        for i in range(self.num_envs):
            start, end = self.agents_per_env*i, self.agents_per_env*(i+1)
            self.c_envs.append(CHighway(
                self.buf.observations[start:end],
                self.actions[start:end],
                self.buf.rewards[start:end],
                self.veh_positions[i],
                self.veh_speeds[i],
            ))
            self.c_envs[i].reset()

        return self.buf.observations, {}
    
    def get_idm_accel(self, this_vel, lead_vel=None, headway=np.inf):
        v0 = 45
        T = 1.0
        a = 1.5
        b = 2.0
        delta = 4
        s0 = 1.0

        assert headway > 1e-3

        if lead_vel is None:
            s_star = 0
        else:
            s_star = s0 + max(0, this_vel * T + this_vel * (this_vel - lead_vel) / (2 * np.sqrt(a * b)))

        accel = a * (1 - (this_vel / v0) ** delta - (s_star / headway) ** 2)

        return accel

    def step(self, actions):
        self.actions[:] = actions
        step_all(self.c_envs)

        for i in range(self.num_envs):
            for j in range(self.cars_per_env):
                speed = self.veh_speeds[i, j]
                if j > 0:
                    lead_vel = self.veh_speeds[i, j - 1]
                    headway = self.veh_positions[i, j - 1] - self.veh_positions[i, j] - self.car_length
                    accel = self.get_idm_accel(speed, lead_vel, headway)
                    self.veh_speeds[i, j] += accel * self.dt
                else:
                    lead_vel = None
                    headway = np.inf
                    self.veh_speeds[i, j] = 20 + 10 * np.sin(self.t / 4)
        self.veh_speeds = np.clip(self.veh_speeds, 0, self.max_speed)

        for i in range(self.num_envs):
            for j in range(self.cars_per_env):
                self.veh_positions[i, j] += self.veh_speeds[i, j] * self.dt
        
        self.t += self.dt

        info = {}

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, info)

    def render(self):
        if self.render_mode == 'human':
            return self.client.render(self.veh_positions[0], self.veh_speeds[0], self.t)


def get_rgb(t):
    if 0 <= t <= 0.25:  # From black to red
        red = int(255 * (t / 0.25))
        return (red, 0, 0)
    elif 0.25 < t <= 0.5:  # From red to orange
        red = 255
        green = int(127 * ((t - 0.25) / 0.25))  # Mix in green for orange
        return (red, green, 0)
    elif 0.5 < t <= 0.75:  # From orange to yellow
        red = 255
        green = int(127 + 128 * ((t - 0.5) / 0.25))
        return (red, green, 0)
    elif 0.75 < t <= 1:  # From yellow to green
        red = int(255 - 255 * ((t - 0.75) / 0.25))
        green = 255
        return (red, green, 0)


class RaylibClient:
    def __init__(self, car_width, car_length):
        self.screenw = 1600
        self.screenh = 300
        rl.InitWindow(self.screenw, self.screenh * 2, "Puffer Highway".encode())
        rl.SetTargetFPS(200)

        self.car_width = car_width
        self.car_length = car_length

        self.pos_history = []
        self.speed_history = []
        self.prev_t = -1e9

    def render(self, veh_positions, veh_speeds, t):
        if rl.IsKeyDown(rl.KEY_ESCAPE):
            exit(0)

        if t >= self.prev_t + 0.4:
            self.pos_history.append(np.copy(veh_positions))
            self.speed_history.append(np.copy(veh_speeds))
            self.prev_t = t

        rl.BeginDrawing()
        rl.ClearBackground(render.PUFF_BACKGROUND)

        padding = 20  # px
        min_pos = min(veh_positions) - self.car_length  # m
        max_pos = max(veh_positions)  # m

        pixels_per_meter = (self.screenw - 2 * padding) / (max_pos - min_pos)
        # pixels_per_meter = 3

        marker_every = 100  # m
        for i in range(
            math.floor(min_pos / marker_every), math.ceil(max_pos / marker_every) + 1):
            x_abs_m = i * marker_every  # m
            x_rel_px = self.screenw - padding + pixels_per_meter * (x_abs_m - max_pos)
            rl.DrawLine(
                int(x_rel_px), 0,
                int(x_rel_px), 50,
                [255, 255, 255, 255]
            )

        rl.DrawRectangle(
            0, int(self.screenh / 2 - 1.5 * self.car_width * pixels_per_meter),
            self.screenw, int(3 * self.car_width * pixels_per_meter),
            [0, 0, 0, 255]
        )
        for pos in veh_positions:
            rl.DrawRectangle(
                int(self.screenw - padding + pixels_per_meter * ((pos - max_pos) - self.car_length)),
                int(self.screenh / 2 - self.car_width * pixels_per_meter / 2),
                int(self.car_length * pixels_per_meter),
                int(self.car_width * pixels_per_meter),
                [255, 255, 255, 255]
            )

        # TSD
        rl.DrawRectangle(0, self.screenh, self.screenw, self.screenh, [255, 255, 255, 255])

        dx = 6
        radius = 3
        for i in range(len(self.pos_history)):
            min_pos = min(self.pos_history[i]) - self.car_length  # m
            max_pos = max(self.pos_history[i])  # m
            pixels_per_meter = (self.screenh - radius) / (max_pos - min_pos)
            for speed, pos in zip(self.speed_history[i], self.pos_history[i]):
                speed_ratio = speed / 35  # 0-1. larger = green, lower = red
                rl.DrawRectangle(
                    int(dx/2 + i * dx - radius),
                    int(self.screenh - (pos - max_pos) * pixels_per_meter + radius - radius),
                    2*radius, 2*radius,
                    [*get_rgb(speed_ratio), 150]
                )

            # void DrawLineStrip(Vector2 *points, int pointCount, Color color);                                  // Draw lines sequence (using gl lines)

        
        rl.DrawFPS(10, 10)
        rl.EndDrawing()
        return render.cdata_to_numpy()


if __name__ == '__main__':
    env = PufferHighway(num_envs=1, render_mode='human')
    env.reset()
    while True:
        env.step([0] * (env.num_envs * env.agents_per_env))
        env.render()