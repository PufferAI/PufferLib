"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: This is the rocket lander simulation built on top of the gym lunar lander. It's made to be a continuous
             action problem (as opposed to discretized).
"""
import sys, math
from enum import Enum

import pufferlib

PUF_SCALE = 30

"""State array definitions"""
class State(Enum):
    x = 0
    y = 1
    x_dot = 2
    y_dot = 3
    theta = 4
    theta_dot = 5
    left_ground_contact = 6
    right_ground_contact = 7
# --------------------------------
"""Simulation Update"""
FPS = 60
UPDATE_TIME = 1/FPS

# --------------------------------
"""Simulation view, Scale and Math Conversions"""
# NOTE: Dimensions do not change linearly with Scale
SCALE = 30  # Adjusts Pixels to Units conversion, Forces, and Leg positioning. Keep at 30.

VIEWPORT_W = 1000
VIEWPORT_H = 800

SEA_CHUNKS = 25

DEGTORAD = math.pi/180

W = int(VIEWPORT_W / SCALE)
H = int(VIEWPORT_H / SCALE)

# --------------------------------
"""Rocket Relative Dimensions"""
INITIAL_RANDOM = 20000.0  # Initial random force (if enabled through simulation settings)

LANDER_CONSTANT = 1  # Constant controlling the dimensions
LANDER_LENGTH = 227 / LANDER_CONSTANT
LANDER_RADIUS = 10 / LANDER_CONSTANT
LANDER_POLY = [
    (-LANDER_RADIUS, 0), (+LANDER_RADIUS, 0),
    (+LANDER_RADIUS, +LANDER_LENGTH), (-LANDER_RADIUS, +LANDER_LENGTH)
]

NOZZLE_POLY = [
    (-LANDER_RADIUS+LANDER_RADIUS/2, 0), (+LANDER_RADIUS-LANDER_RADIUS/2, 0),
    (-LANDER_RADIUS + LANDER_RADIUS/2, +LANDER_LENGTH/8), (+LANDER_RADIUS-LANDER_RADIUS/2, +LANDER_LENGTH/8)
]

LEG_AWAY = 30 / LANDER_CONSTANT
LEG_DOWN = 0.3/LANDER_CONSTANT
LEG_W, LEG_H = 3 / LANDER_CONSTANT, LANDER_LENGTH / 8 / LANDER_CONSTANT

SIDE_ENGINE_VERTICAL_OFFSET = 5  # y-distance away from the top of the rocket
SIDE_ENGINE_HEIGHT = LANDER_LENGTH - SIDE_ENGINE_VERTICAL_OFFSET
SIDE_ENGINE_AWAY = 10.0

# --------------------------------
"""Forces, Costs, Torque, Friction"""
MAIN_ENGINE_POWER = FPS*LANDER_LENGTH / (LANDER_CONSTANT * 2.1)  # Multiply by FPS since we're using Forces not Impulses
SIDE_ENGINE_POWER = MAIN_ENGINE_POWER / 50  # Multiply by FPS since we're using Forces not Impulses

INITIAL_FUEL_MASS_PERCENTAGE = 0.2  # Allocate a % of the total initial weight of the rocket to fuel
MAIN_ENGINE_FUEL_COST = MAIN_ENGINE_POWER/SIDE_ENGINE_POWER
SIDE_ENGINE_FUEL_COST = 1

LEG_SPRING_TORQUE = LANDER_LENGTH/2
NOZZLE_TORQUE = 500 / LANDER_CONSTANT
NOZZLE_ANGLE_LIMIT = 15*DEGTORAD

BARGE_FRICTION = 2

# --------------------------------
"""Landing Calibration"""
LANDING_VERTICAL_CALIBRATION = 0.03
TERRAIN_CHUNKS = 16 # 0-20 calm seas, 20+ rough seas
BARGE_LENGTH_X1_RATIO = 0.35# 0.35#0.27 # 0 -1
BARGE_LENGTH_X2_RATIO = 0.65#0.65 #0.73 # 0 -1
# --------------------------------
"Kinematic Constants"
# NOTE: Recalculate if the dimensions of the rocket change in any way
MASS = 25.222
L1 = 3.8677
L2 = 3.7
LN = 0.1892
INERTIA = 482.2956
GRAVITY = 9.81

# ---------------------------------
"""State Reset Limits"""
THETA_LIMIT = 35*DEGTORAD

# ---------------------------------
"""State Definition"""
# Added for accessing state array in a readable manner
XX = State.x.value
YY = State.y.value
X_DOT = State.x_dot.value
Y_DOT = State.y_dot.value
THETA = State.theta.value
THETA_DOT = State.theta_dot.value
LEFT_GROUND_CONTACT = State.left_ground_contact.value
RIGHT_GROUND_CONTACT = State.right_ground_contact.value
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import numpy as np
import Box2D
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import logging
from itertools import chain
from raylib import rl, colors

def draw_poly(coords, color=colors.RED):
    coords = [e for e in coords] + [coords[0]]
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        x1 *= PUF_SCALE
        y1 *= PUF_SCALE
        x2 *= PUF_SCALE
        y2 *= PUF_SCALE
        y1 = VIEWPORT_H - y1
        y2 = VIEWPORT_H - y2
        rl.DrawLine(int(x1), int(y1), int(x2), int(y2), color)
        #print(f'Drawing rect {x} {y} {w} {h}')
        #rl.DrawRectangle(int(x), int(y), int(w), int(h), color)

def draw_circle(x, y, r, color):
    x *= PUF_SCALE
    y *= PUF_SCALE
    r *= PUF_SCALE
    y = VIEWPORT_H - y
    rl.DrawCircle(int(x), int(y), int(r), color)

# This contact detector is equivalent the one implemented in Lunar Lander
class ContactDetector(contactListener):
    """
    Creates a contact listener to check when the rocket touches down.
    """
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def begin_contact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def end_contact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

class RocketLander(pufferlib.PufferEnv):
    """
    Continuous VTOL of a rocket.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self, side_engines=True, clouds=True, vectorized_nozzle=True,
            starting_y_pos_constant=1, initial_force='random', buf=None):
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0, -GRAVITY))
        self.main_base = None
        self.barge_base = None
        self.CONTACT_FLAG = False

        self.side_engines = side_engines
        self.clouds = clouds
        self.vectorized_nozzle = vectorized_nozzle
        self.starting_y_pos_constant = starting_y_pos_constant
        self.initial_force = initial_force

        self.minimum_barge_height = 0
        self.maximum_barge_height = 0
        self.landing_coordinates = []

        self.lander = None
        self.particles = []
        self.state = []
        self.prev_shaping = None


        # [x_pos, y_pos, x_vel, y_vel, lateral_angle, angular_velocity]
        self.single_observation_space = spaces.Box(-np.inf, +np.inf, (8,))

        # Main Engine, Nozzle Angle, Left/Right Engine
        self.single_action_space = spaces.Box(-np.inf, +np.inf, (3,))
        #     np.array([0, -1, -NOZZLE_ANGLE_LIMIT]),
        #    np.array([1, 1, NOZZLE_ANGLE_LIMIT]),
        #    (3,)
        #)
        self.lander_tilt_angle_limit = THETA_LIMIT

        self.game_over = False

        self.dynamicLabels = {}
        self.staticLabels = {}

        self.impulsePos = (0, 0)

        self.untransformed_state = [0] * 6  # Non-normalized state

        self.num_agents = 1
        self.render_mode = 'human'
        super().__init__(buf)

    def reset(self, seed=None):
        self.np_random, returned_seed = seeding.np_random(seed or 42)
        self._destroy()
        self.game_over = False
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        smoothed_terrain_edges, terrain_divider_coordinates_x = self._create_terrain(TERRAIN_CHUNKS)

        self.initial_mass = 0
        self.remaining_fuel = 0
        self.prev_shaping = 0
        self.CONTACT_FLAG = False

        # Engine Stats
        self.action_history = []

        # gradient of 0.009
        # Reference y-trajectory
        self.y_pos_ref = [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
        self.y_pos_speed = [-1.9, -1.8, -1.64, -1.5, -1.5, -1.3, -1.0, -0.9]
        self.y_pos_flags = [False for _ in self.y_pos_ref]

        # Create the simulation objects
        self._create_clouds()
        self._create_barge()
        self._create_base_static_edges(TERRAIN_CHUNKS, smoothed_terrain_edges, terrain_divider_coordinates_x)

        # Adjust the initial coordinates of the rocket
        initial_coordinates = None
        if initial_coordinates is not None:
            xx, yy, randomness_degree, normalized = initial_coordinates
            x = xx * W + np.random.uniform(-randomness_degree, randomness_degree)
            y = yy * H + np.random.uniform(-randomness_degree, randomness_degree)
            if not normalized:
                x = x / W
                y = y / H
        else:
            x, y = W / 2 + np.random.uniform(-0.1, 0.1), H / self.starting_y_pos_constant
        self.initial_coordinates = (x, y)

        self._create_rocket(self.initial_coordinates)

        self.state, self.untransformed_state = self.__generate_state()  # Generate state
        self.observations[:] = self.state
        return self.observations, []

    def _destroy(self):
        if not self.main_base: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.main_base)
        self.main_base = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def step(self, action):
        action = action.ravel()
        assert len(action) == 3  # Fe, Fs, psi

        # Check for contact with the ground
        if (self.legs[0].ground_contact or self.legs[1].ground_contact) and self.CONTACT_FLAG == False:
            self.CONTACT_FLAG = True

        # Shutdown all Engines upon contact with the ground
        if self.CONTACT_FLAG:
            action = [0, 0, 0]

        if self.vectorized_nozzle:
            part = self.nozzle
            part.angle = self.lander.angle + float(action[2])  # This works better than motorSpeed
            if part.angle > NOZZLE_ANGLE_LIMIT:
                part.angle = NOZZLE_ANGLE_LIMIT
            elif part.angle < -NOZZLE_ANGLE_LIMIT:
                part.angle = -NOZZLE_ANGLE_LIMIT
            # part.joint.motorSpeed = float(action[2]) # action[2] is in radians
            # That means having a value of 2*pi will rotate at 360 degrees / second
            # A transformation can be done on the action, such as clipping the value
        else:
            part = self.lander

        # "part" is used to decide where the main engine force is applied (whether it is applied to the bottom of the
        # nozzle or the bottom of the first stage rocket

        # Main Force Calculations
        if self.remaining_fuel == 0:
            logging.info("Strictly speaking, you're out of fuel, but act anyway.")
        m_power = self.__main_engines_force_computation(action, rocketPart=part)
        s_power, engine_dir = self.__side_engines_force_computation(action)

        # Decrease the rocket mass
        self._decrease_mass(m_power, s_power)

        # State Vector
        self.previous_state = self.state  # Keep a record of the previous state
        state, self.untransformed_state = self.__generate_state()  # Generate state
        self.state = state  # Keep a record of the new state

        # Rewards for reinforcement learning
        reward = self.__compute_rewards(state, m_power, s_power,
                                        part.angle)  # part angle can be used as part of the reward

        # Check if the game is done, adjust reward based on the final state of the body
        state_reset_conditions = [
            self.game_over,  # Evaluated depending on body contact
            abs(state[XX]) >= 1.0,  # Rocket moves out of x-space
            state[YY] < 0 or state[YY] > 1.3,  # Rocket moves out of y-space or below barge
            abs(state[THETA]) > THETA_LIMIT]  # Rocket tilts greater than the "controllable" limit
        self._update_particles()

        self.observations[:] = state
        self.rewards[:] = reward

        done = False
        if any(state_reset_conditions):
            done = True
            reward = -10
            self.reset()
        if not self.lander.awake:
            done = True
            reward = +10

        self.terminals[:] = done
        info = [{'reward': reward}]
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def __main_engines_force_computation(self, action, rocketPart, *args):
        # ----------------------------------------------------------------------------
        # Nozzle Angle Adjustment

        # For readability
        sin = math.sin(rocketPart.angle)
        cos = math.cos(rocketPart.angle)

        # Random dispersion for the particles
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        # Main engine
        m_power = 0
        try:
            if (action[0] > 0.0):
                # Limits
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.3  # 0.5..1.0
                assert m_power >= 0.3 and m_power <= 1.0
                # ------------------------------------------------------------------------
                ox = sin * (4 / SCALE + 2 * dispersion[0]) - cos * dispersion[
                    1]  # 4 is move a bit downwards, +-2 for randomness
                oy = -cos * (4 / SCALE + 2 * dispersion[0]) - sin * dispersion[1]
                impulse_pos = (rocketPart.position[0] + ox, rocketPart.position[1] + oy)

                # rocketParticles are just a decoration, 3.5 is here to make rocketParticle speed adequate
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power,
                                          radius=7)

                rocketParticleImpulse = (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power)
                bodyImpulse = (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power)
                point = impulse_pos
                wake = True

                # Force instead of impulse. This enables proper scaling and values in Newtons
                p.ApplyForce(rocketParticleImpulse, point, wake)
                rocketPart.ApplyForce(bodyImpulse, point, wake)
        except:
            print("Error in main engine power.")

        return m_power

    def __side_engines_force_computation(self, action):
        # ----------------------------------------------------------------------------
        # Side engines
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
        sin = math.sin(self.lander.angle)  # for readability
        cos = math.cos(self.lander.angle)
        s_power = 0.0
        y_dir = 1 # Positioning for the side Thrusters
        engine_dir = 0
        if (self.side_engines):  # Check if side gas thrusters are enabled
            if (np.abs(action[1]) > 0.5): # Have to be > 0.5
                # Orientation engines
                engine_dir = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0

                # Positioning
                constant = (LANDER_LENGTH - SIDE_ENGINE_VERTICAL_OFFSET) / SCALE
                dx_part1 = - sin * constant  # Used as reference for dy
                dx_part2 = - cos * engine_dir * SIDE_ENGINE_AWAY / SCALE
                dx = dx_part1 + dx_part2

                dy = np.sqrt(
                    np.square(constant) - np.square(dx_part1)) * y_dir - sin * engine_dir * SIDE_ENGINE_AWAY / SCALE

                # Force magnitude
                oy = -cos * dispersion[0] - sin * (3 * dispersion[1] + engine_dir * SIDE_ENGINE_AWAY / SCALE)
                ox = sin * dispersion[0] - cos * (3 * dispersion[1] + engine_dir * SIDE_ENGINE_AWAY / SCALE)

                # Impulse Position
                impulse_pos = (self.lander.position[0] + dx,
                               self.lander.position[1] + dy)

                # Plotting purposes only
                self.impulsePos = (self.lander.position[0] + dx, self.lander.position[1] + dy)

                try:
                    p = self._create_particle(1, impulse_pos[0], impulse_pos[1], s_power, radius=3)
                    p.ApplyForce((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos,
                                 True)
                    self.lander.ApplyForce((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos, True)
                except:
                    logging.error("Error due to Nan in calculating y during sqrt(l^2 - x^2). "
                                  "x^2 > l^2 due to approximations on the order of approximately 1e-15.")

        return s_power, engine_dir

    def __generate_state(self):
        # ----------------------------------------------------------------------------
        # Update
        self.world.Step(1.0 / FPS, 6 * 30, 6 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        target = (self.initial_barge_coordinates[1][0] - self.initial_barge_coordinates[0][0]) / 2 + \
                 self.initial_barge_coordinates[0][0]
        state = [
            (pos.x - target) / (W / 2),
            (pos.y - (self.maximum_barge_height + (LEG_DOWN / SCALE))) / (W / 2) - LANDING_VERTICAL_CALIBRATION,
            # affects controller
            # self.bargeHeight includes height of helipad
            vel.x * (W / 2) / FPS,
            vel.y * (H / 2) / FPS,
            self.lander.angle,
            # self.nozzle.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
        ]

        untransformed_state = [pos.x, pos.y, vel.x, vel.y, self.lander.angle, self.lander.angularVelocity]

        return state, untransformed_state

    # ['dx','dy','x_vel','y_vel','theta','theta_dot','left_ground_contact','right_ground_contact']
    def __compute_rewards(self, state, main_engine_power, side_engine_power, part_angle):
        reward = 0
        shaping = -200 * np.sqrt(np.square(state[0]) + np.square(state[1])) \
                  - 100 * np.sqrt(np.square(state[2]) + np.square(state[3])) \
                  - 1000 * abs(state[4]) - 30 * abs(state[5]) \
                  + 20 * state[6] + 20 * state[7]

        # penalize increase in altitude
        if state[3] > 0:
            shaping = shaping - 1

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # penalize the use of engines
        reward += -main_engine_power * 0.3
        if self.side_engines:
            reward += -side_engine_power * 0.3

        return reward / 10

    # Problem specific - LINKED
    def _create_terrain(self, chunks):
        # Terrain Coordinates
        # self.helipad_x1 = W / 5
        # self.helipad_x2 = self.helipad_x1 + W / 5
        divisor_constant = 8  # Control the height of the sea
        self.helipad_y = H / divisor_constant

        # Terrain
        # height = self.np_random.uniform(0, H / 6, size=(CHUNKS + 1,))
        height = np.random.normal(H / divisor_constant, 0.5, size=(chunks + 1,))
        chunk_x = [W / (chunks - 1) * i for i in range(chunks)]
        # self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        # self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        height[chunks // 2 - 2] = self.helipad_y
        height[chunks // 2 - 1] = self.helipad_y
        height[chunks // 2 + 0] = self.helipad_y
        height[chunks // 2 + 1] = self.helipad_y
        height[chunks // 2 + 2] = self.helipad_y

        return [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(chunks)], chunk_x  # smoothed Y

    def _create_rocket(self, initial_coordinates=(W / 2, H / 1.2)):
        body_color = (1, 1, 1)
        # ----------------------------------------------------------------------------------------
        # LANDER

        initial_x, initial_y = initial_coordinates
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.lander.color1 = body_color
        self.lander.color2 = (0, 0, 0)

        if isinstance(self.initial_force, str):
            self.lander.ApplyForceToCenter((
                self.np_random.uniform(-INITIAL_RANDOM * 0.3, INITIAL_RANDOM * 0.3),
                self.np_random.uniform(-1.3 * INITIAL_RANDOM, -INITIAL_RANDOM)
            ), True)
        else:
            self.lander.ApplyForceToCenter(self.initial_force, True)

        # COG is set in the middle of the polygon by default. x = 0 = middle.
        # self.lander.mass = 25
        # self.lander.localCenter = (0, 3) # COG
        # ----------------------------------------------------------------------------------------
        # LEGS
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=5.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x005)
            )
            leg.ground_contact = False
            leg.color1 = body_color
            leg.color2 = (0, 0, 0)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(-i * 0.3 / LANDER_CONSTANT, 0),
                localAnchorB=(i * 0.5 / LANDER_CONSTANT, LEG_DOWN),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = 40 * DEGTORAD
                rjd.upperAngle = 45 * DEGTORAD
            else:
                rjd.lowerAngle = -45 * DEGTORAD
                rjd.upperAngle = -40 * DEGTORAD
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        # ----------------------------------------------------------------------------------------
        # NOZZLE
        self.nozzle = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in NOZZLE_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0040,
                maskBits=0x003,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.nozzle.color1 = (0, 0, 0)
        self.nozzle.color2 = (0, 0, 0)
        rjd = revoluteJointDef(
            bodyA=self.lander,
            bodyB=self.nozzle,
            localAnchorA=(0, 0),
            localAnchorB=(0, 0.2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=NOZZLE_TORQUE,
            motorSpeed=0,
            referenceAngle=0,
            lowerAngle=-13 * DEGTORAD,  # +- 15 degrees limit applied in practice
            upperAngle=13 * DEGTORAD
        )
        # The default behaviour of a revolute joint is to rotate without resistance.
        self.nozzle.joint = self.world.CreateJoint(rjd)
        # ----------------------------------------------------------------------------------------
        # self.drawlist = [self.nozzle] + [self.lander] + self.legs
        self.drawlist = self.legs + [self.nozzle] + [self.lander]
        self.initial_mass = self.lander.mass
        self.remaining_fuel = INITIAL_FUEL_MASS_PERCENTAGE * self.initial_mass
        return

    # Problem specific - LINKED
    def _create_barge(self):
        # Landing Barge
        # The Barge can be modified in shape and angle
        self.bargeHeight = self.helipad_y * (1 + 0.6)

        # self.landingBargeCoordinates = [(self.helipad_x1, 0.1), (self.helipad_x2, 0.1),
        #                                 (self.helipad_x2, self.bargeHeight), (self.helipad_x1, self.bargeHeight)]
        assert BARGE_LENGTH_X1_RATIO < BARGE_LENGTH_X2_RATIO, 'Barge Length X1 must be 0-1 and smaller than X2'

        x1 = BARGE_LENGTH_X1_RATIO*W
        x2 = BARGE_LENGTH_X2_RATIO*W
        self.landing_barge_coordinates = [(x1, 0.1), (x2, 0.1),
                                          (x2, self.bargeHeight), (x1, self.bargeHeight)]

        self.initial_barge_coordinates = self.landing_barge_coordinates
        self.minimum_barge_height = min(self.landing_barge_coordinates[2][1], self.landing_barge_coordinates[3][1])
        self.maximum_barge_height = max(self.landing_barge_coordinates[2][1], self.landing_barge_coordinates[3][1])
        # Used for the actual area inside the Barge
        # barge_length = self.helipad_x2 - self.helipad_x1
        # padRatio = 0.2
        # self.landingPadCoordinates = [self.helipad_x1 + barge_length * padRatio,
        #                               self.helipad_x2 - barge_length * padRatio]
        barge_length = x2 - x1
        padRatio = 0.2
        self.landing_pad_coordinates = [x1 + barge_length * padRatio,
                                        x2 - barge_length * padRatio]

        self.landing_coordinates = self.get_landing_coordinates()

    # Problem specific - LINKED
    def _create_base_static_edges(self, CHUNKS, smooth_y, chunk_x):
        # Sky
        self.sky_polys = []
        # Ground
        self.ground_polys = []
        self.sea_polys = [[] for _ in range(SEA_CHUNKS)]

        # Main Base
        self.main_base = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self._create_static_edge(self.main_base, [p1, p2], 0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

            self.ground_polys.append([p1, p2, (p2[0], 0), (p1[0], 0)])

            for j in range(SEA_CHUNKS - 1):
                k = 1 - (j + 1) / SEA_CHUNKS
                self.sea_polys[j].append([(p1[0], p1[1] * k), (p2[0], p2[1] * k), (p2[0], 0), (p1[0], 0)])

        self._update_barge_static_edges()

    def _update_barge_static_edges(self):
        if self.barge_base is not None:
            self.world.DestroyBody(self.barge_base)
        self.barge_base = None
        barge_edge_coordinates = [self.landing_barge_coordinates[2], self.landing_barge_coordinates[3]]
        self.barge_base = self.world.CreateStaticBody(shapes=edgeShape(vertices=barge_edge_coordinates))
        self._create_static_edge(self.barge_base, barge_edge_coordinates, friction=BARGE_FRICTION)

    @staticmethod
    def _create_static_edge(base, vertices, friction):
        base.CreateEdgeFixture(
            vertices=vertices,
            density=0,
            friction=friction)
        return

    def _create_particle(self, mass, x, y, ttl, radius=3):
        """
        Used for both the Main Engine and Side Engines
        :param mass: Different mass to represent different forces
        :param x: x position
        :param y:  y position
        :param ttl:
        :param radius:
        :return:
        """
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=radius / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
        )
        p.ttl = ttl  # ttl is decreased with every time step to determine if the particle should be destroyed
        self.particles.append(p)
        # Check if some particles need cleaning
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particles):
        while self.particles and (all_particles or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def _create_cloud(self, x_range, y_range, y_variance=0.1):
        self.cloud_poly = []
        numberofdiscretepoints = 3

        initial_y = (VIEWPORT_H * np.random.uniform(y_range[0], y_range[1], 1)) / SCALE
        initial_x = (VIEWPORT_W * np.random.uniform(x_range[0], x_range[1], 1)) / SCALE

        y_coordinates = np.random.normal(0, y_variance, numberofdiscretepoints)
        x_step = np.linspace(initial_x, initial_x + np.random.uniform(1, 6), numberofdiscretepoints + 1)

        for i in range(0, numberofdiscretepoints):
            self.cloud_poly.append((x_step[i], initial_y + math.sin(3.14 * 2 * i / 50) * y_coordinates[i]))

        return self.cloud_poly

    def _create_clouds(self):
        self.clouds = []
        for i in range(10):
            self.clouds.append(self._create_cloud([0.2, 0.4], [0.65, 0.7], 1))
            self.clouds.append(self._create_cloud([0.7, 0.8], [0.75, 0.8], 1))

    def _decrease_mass(self, main_engine_power, side_engine_power):
        x = np.array([float(main_engine_power), float(side_engine_power)])
        consumed_fuel = 0.009 * np.sum(x * (MAIN_ENGINE_FUEL_COST, SIDE_ENGINE_FUEL_COST)) / SCALE
        self.lander.mass = self.lander.mass - consumed_fuel
        self.remaining_fuel -= consumed_fuel
        if self.remaining_fuel < 0:
            self.remaining_fuel = 0

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                rl.CloseWindow()
                self.viewer = None
            return

        # This can be removed since the code is being update to utilize env.refresh() instead
        # Kept here for backwards compatibility purposes
        # Viewer Creation
        if self.viewer is None:  # Initial run will enter here
            self.viewer = 1
            rl.InitWindow(VIEWPORT_W, VIEWPORT_H, 'Puffer Rocket Lander'.encode())
            rl.SetTargetFPS(FPS)

        rl.BeginDrawing()
        rl.ClearBackground(colors.BLACK)

        #for p in self.sky_polys:
        #    self.viewer.draw_polygon(p, color=(0.83, 0.917, 1.0))

        # Landing Barge
        draw_poly(self.landing_barge_coordinates, colors.RED)

        for g in self.ground_polys:
            draw_poly(g, colors.GREEN)

        for i, s in enumerate(self.sea_polys):
            k = 1 - (i + 1) / SEA_CHUNKS
            for poly in s:
                draw_poly(poly, [0, int(127*k), int(255*k), 255])

        if self.clouds:
            self._render_clouds()

        # Landing Flags
        for x in self.landing_pad_coordinates:
            flagy1 = self.landing_barge_coordinates[3][1]
            flagy2 = self.landing_barge_coordinates[2][1] + 25 / SCALE

            polygon_coordinates = [(x, flagy2), (x, flagy2 - 10 / SCALE), (x + 25 / SCALE, flagy2 - 5 / SCALE)]
            draw_poly(polygon_coordinates, colors.RED)
            draw_poly([(x, flagy1), (x, flagy2)], colors.WHITE)

        # Lander and Particles
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    #t = rendering.Transform(translation=trans * f.shape.pos)
                    x, y = trans * f.shape.pos
                    rl.DrawCircle(int(x), int(y), f.shape.radius, colors.RED)
                    #rl.DrawCircle(f.shape.radius, 20, colors.RED)
                    #self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    #self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False,
                    #                        linewidth=2).add_attr(t)
                else:
                    # Lander
                    path = [trans * v for v in f.shape.vertices]
                    draw_poly(path, colors.BLUE)

        #self.draw_marker(x=self.lander.worldCenter.x, y=self.lander.worldCenter.y)  # Center of Gravity
        rl.EndDrawing()
        pass
        # self.drawMarker(x=self.impulsePos[0], y=self.impulsePos[1])              # Side Engine Forces Positions
        # self.drawMarker(x=self.lander.position[0], y=self.lander.position[1])    # (0,0) position

        # Commented out to be able to draw from outside the class using self.refresh
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _render_clouds(self):
        pass
        #for x in self.clouds:
        #    self.viewer.draw_polygon(x, color=(1.0, 1.0, 1.0))

    def _update_particles(self):
        for obj in self.particles:
            obj.ttl -= 0.1
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))

        self._clean_particles(False)

    def get_landing_coordinates(self):
        x = (self.landing_barge_coordinates[1][0] - self.landing_barge_coordinates[0][0]) / 2 + \
            self.landing_barge_coordinates[0][0]
        y = abs(self.landing_barge_coordinates[2][1] - self.landing_barge_coordinates[3][1]) / 2 + \
            min(self.landing_barge_coordinates[2][1], self.landing_barge_coordinates[3][1])
        return [x, y]

def flatten_array(the_list):
    return list(chain.from_iterable(the_list))


