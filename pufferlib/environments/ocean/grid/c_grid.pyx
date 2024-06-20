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
        int map_size
        int num_agents
        int horizon
        int vision_range
        int obs_size

        unsigned char[:, :] grid
        unsigned char[:, :, :] observations
        unsigned int[:, :] agent_positions
        unsigned int[:, :] spawn_position_cands

    def __init__(self, grid, agent_positions, spawn_position_cands, observations,
            int map_size, int num_agents, int horizon, int vision_range):
        self.map_size = map_size 
        self.num_agents = num_agents
        self.horizon = horizon
        self.vision_range = vision_range
        self.obs_size = 2*self.vision_range + 1

        self.grid = grid
        self.observations = observations
        self.agent_positions = agent_positions
        self.spawn_position_cands = spawn_position_cands

    cdef void _compute_observations(self):
        cdef int r, c, agent_idx
        for agent_idx in range(self.num_agents):
            r = self.agent_positions[agent_idx, 0]
            c = self.agent_positions[agent_idx, 1]
            self.observations[agent_idx, :] = self.grid[
                r-self.vision_range:r+self.vision_range+1,
                c-self.vision_range:c+self.vision_range+1
            ]

    def reset(self, observations, seed=0):
        self.observations = observations

        # Add borders
        cdef int left = self.vision_range
        cdef int right = self.map_size - self.vision_range - 1
        self.grid[:left, :] = WALL
        self.grid[right:, :] = WALL
        self.grid[:, :left] = WALL
        self.grid[:, right:] = WALL

        # Agent spawning
        cdef int spawn_idx, r, c
        cdef int agent_idx = 0
        for spawn_idx in range(self.map_size**2):
            r = self.spawn_position_cands[spawn_idx, 0]
            c = self.spawn_position_cands[spawn_idx, 1]
            if self.grid[r, c] == 0:
                self.grid[r, c] = AGENT
                self.agent_positions[agent_idx, 0] = r
                self.agent_positions[agent_idx, 1] = c
                agent_idx += 1
                if agent_idx == self.num_agents:
                    break

        self._compute_observations()

    def step(self, np_actions):
        cdef unsigned int[:] actions = np_actions
        cdef int agent_idx, atn, r, c, dr, dc, dest_r, dest_c
        for agent_idx in range(self.num_agents):
            r = self.agent_positions[agent_idx, 0]
            c = self.agent_positions[agent_idx, 1]
            atn = actions[agent_idx]
            dr = 0
            dc = 0
            if atn == PASS:
                continue
            elif atn == NORTH:
                dr = -1
            elif atn == SOUTH:
                dr = 1
            elif atn == EAST:
                dc = 1
            elif atn == WEST:
                dc = -1
            else:
                raise ValueError(f'Invalid action: {atn}')

            dest_r = r + dr
            dest_c = c + dc

            if self.grid[dest_r, dest_c] == 0:
                self.grid[r, c] = EMPTY
                self.grid[dest_r, dest_c] = AGENT
                self.agent_positions[agent_idx, 0] = dest_r
                self.agent_positions[agent_idx, 1] = dest_c

        self._compute_observations()
