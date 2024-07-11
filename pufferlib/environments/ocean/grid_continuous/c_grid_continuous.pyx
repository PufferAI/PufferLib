# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

cdef:
    int EMPTY = 0
    int AGENT = 1
    int WALL = 2

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
        int obs_size

        unsigned char[:, :] grid
        unsigned char[:, :, :] observations
        float[:, :] agent_positions
        float[:, :] spawn_position_cands

    def __init__(self, grid, agent_positions, spawn_position_cands,
            observations, int width, int height, int num_agents, int horizon,
            int vision_range, float agent_speed, bint discretize):
        self.width = width 
        self.height = height
        self.num_agents = num_agents
        self.horizon = horizon
        self.vision_range = vision_range
        self.agent_speed = agent_speed
        self.discretize = discretize
        self.obs_size = 2*self.vision_range + 1

        self.grid = grid
        self.observations = observations
        self.agent_positions = agent_positions
        self.spawn_position_cands = spawn_position_cands

    cdef void _compute_observations(self):
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

    def reset(self, seed=0):
        # Add borders
        cdef int left = self.vision_range
        cdef int right = self.width - self.vision_range - 1
        cdef int bottom = self.height - self.vision_range - 1
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
            disc_y = int(self.agent_speed * y)
            disc_x = int(self.agent_speed * x)

            if self.grid[disc_y, disc_x] == 0:
                self.grid[disc_y, disc_x] = AGENT
                self.agent_positions[agent_idx, 0] = y
                self.agent_positions[agent_idx, 1] = x
                agent_idx += 1
                if agent_idx == self.num_agents:
                    break

        self._compute_observations()


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
            dest_y = y + vel_y
            dest_x = x + vel_x

            # Discretize
            disc_y = int(self.agent_speed * y)
            disc_x = int(self.agent_speed * x)
            disc_dest_y = int(self.agent_speed * dest_y)
            disc_dest_x = int(self.agent_speed * dest_x)

            if self.grid[disc_dest_y, disc_dest_x] == 0:
                self.grid[disc_y, disc_x] = EMPTY
                self.grid[disc_dest_y, disc_dest_x] = AGENT

                # Continuous position update
                self.agent_positions[agent_idx, 0] = dest_y
                self.agent_positions[agent_idx, 1] = dest_x

        self._compute_observations()
