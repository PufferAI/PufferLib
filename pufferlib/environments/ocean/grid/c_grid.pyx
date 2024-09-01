# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: profile=False

from libc.stdlib cimport rand

cdef:
    int EMPTY = 0
    int FOOD = 1
    int WALL = 2
    int AGENT_1 = 3
    int AGENT_2 = 4
    int AGENT_3 = 5
    int AGENT_4 = 6

    int PASS = 0
    int NORTH = 1
    int SOUTH = 2
    int EAST = 3
    int WEST = 4

cdef class Environment:
    cdef:
        int width
        int height
        int num_agents
        int horizon
        int vision_range
        float agent_speed
        bint discretize
        float food_reward
        int expected_lifespan
        int obs_size

        unsigned char[:, :] grid
        unsigned char[:, :, :] observations
        float[:] rewards
        float[:, :] agent_positions
        float[:, :] spawn_position_cands
        int[:] agent_colors

    def __init__(self, grid, agent_positions, spawn_position_cands, agent_colors,
            observations, rewards, int width, int height, int num_agents, int horizon,
            int vision_range, float agent_speed, bint discretize, float food_reward,
            int expected_lifespan):
        self.width = width 
        self.height = height
        self.num_agents = num_agents
        self.horizon = horizon
        self.vision_range = vision_range
        self.agent_speed = agent_speed
        self.discretize = discretize
        self.food_reward = food_reward
        self.expected_lifespan = expected_lifespan
        self.obs_size = 2*self.vision_range + 1

        self.grid = grid
        self.observations = observations
        self.rewards = rewards
        self.agent_positions = agent_positions
        self.spawn_position_cands = spawn_position_cands
        self.agent_colors = agent_colors

    cdef void compute_observations(self):
        cdef:
            float y
            float x
            int r
            int c
            int agent_idx

        for agent_idx in range(self.num_agents):
            y = self.agent_positions[agent_idx, 0]
            x = self.agent_positions[agent_idx, 1]
            r = int(y)
            c = int(x)
            self.observations[agent_idx, :] = self.grid[
                r-self.vision_range:r+self.vision_range+1,
                c-self.vision_range:c+self.vision_range+1
            ]

    cdef void spawn_food(self):
        cdef int r, c, tile
        while True:
            r = rand() % (self.height - 1)
            c = rand() % (self.width - 1)
            tile = self.grid[r, c]
            if tile == EMPTY:
                self.grid[r, c] = FOOD
                return

    cdef void spawn_agent(self, int agent_idx):
        cdef int old_r, old_c, r, c, tile

        # Delete agent from old position
        old_r = int(self.agent_positions[agent_idx, 0])
        old_c = int(self.agent_positions[agent_idx, 1])
        self.grid[old_r, old_c] = EMPTY

        r = rand() % (self.height - 1)
        c = rand() % (self.width - 1)
        tile = self.grid[r, c]
        if tile == EMPTY:
            # Spawn agent in new position
            self.grid[r, c] = self.agent_colors[agent_idx]
            self.agent_positions[agent_idx, 0] = r
            self.agent_positions[agent_idx, 1] = c
            return

    def reset(self, seed=0):
        # Add borders
        cdef int left = int(self.agent_speed * self.vision_range)
        cdef int right = self.width - int(self.agent_speed*self.vision_range) - 1
        cdef int bottom = self.height - int(self.agent_speed*self.vision_range) - 1
        self.grid[:left, :] = WALL
        self.grid[:, :left] = WALL
        self.grid[bottom:, :] = WALL
        self.grid[:, right:] = WALL

        # Agent spawning
        cdef:
            int spawn_idx
            float y
            float x
            int disc_y
            int disc_x
            int agent_idx = 0

        for spawn_idx in range(self.width*self.height):
            y = self.spawn_position_cands[spawn_idx, 0]
            x = self.spawn_position_cands[spawn_idx, 1]
            disc_y = int(y)
            disc_x = int(x)

            if self.grid[disc_y, disc_x] == EMPTY:
                self.grid[disc_y, disc_x] = self.agent_colors[agent_idx]
                self.agent_positions[agent_idx, 0] = y
                self.agent_positions[agent_idx, 1] = x
                agent_idx += 1
                if agent_idx == self.num_agents:
                    break

        self.compute_observations()

    def step(self, np_actions):
        cdef:
            float[:, :] actions_continuous
            unsigned int[:, :] actions_discrete
            int agent_idx
            float y
            float x
            float vel_y
            float vel_x
            int disc_y
            int disc_x
            int disc_dest_y
            int disc_dest_x

        if self.discretize:
            actions_discrete = np_actions
        else:
            actions_continuous = np_actions

        for agent_idx in range(self.num_agents):
            if self.discretize:
                # Convert [0, 1, 2] to [-1, 0, 1]
                vel_y = float(actions_discrete[agent_idx, 0]) - 1.0
                vel_x = float(actions_discrete[agent_idx, 1]) - 1.0
            else:
                vel_y = actions_continuous[agent_idx, 0]
                vel_x = actions_continuous[agent_idx, 1]

            y = self.agent_positions[agent_idx, 0]
            x = self.agent_positions[agent_idx, 1]
            dest_y = y + self.agent_speed * vel_y
            dest_x = x + self.agent_speed * vel_x

            # Discretize
            disc_y = int(y)
            disc_x = int(x)
            disc_dest_y = int(dest_y)
            disc_dest_x = int(dest_x)

            if self.grid[disc_dest_y, disc_dest_x] == FOOD:
                self.grid[disc_dest_y, disc_dest_x] = EMPTY
                self.rewards[agent_idx] = self.food_reward
                self.spawn_food()

            if self.grid[disc_dest_y, disc_dest_x] == 0:
                self.grid[disc_y, disc_x] = EMPTY
                self.grid[disc_dest_y, disc_dest_x] = self.agent_colors[agent_idx]

                # Continuous position update
                self.agent_positions[agent_idx, 0] = dest_y
                self.agent_positions[agent_idx, 1] = dest_x

            # Randomly respawn agents
            if rand() % self.expected_lifespan == 0:
                self.spawn_agent(agent_idx)

        self.compute_observations()
