# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=True
# cython: initializedcheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: nonecheck=True
# cython: profile=False

from libc.stdlib cimport rand
cimport numpy as cnp
import numpy as np

cdef:
    int EMPTY = 0
    int TOWER = 1
    int WALL = 2
    int AGENT_1 = 3
    int AGENT_2 = 4
    int CREEP_1 = 5
    int CREEP_2 = 6

    int PASS = 0
    int NORTH = 1
    int SOUTH = 2
    int EAST = 3
    int WEST = 4

    int TOWER_VISION = 5

cdef struct Player:
    int team
    float health
    float y
    float x
    float spawn_y
    float spawn_x

cdef struct Creep:
    int team
    float health
    float y
    float x
    float spawn_y
    float spawn_x

cdef struct Tower:
    int team
    float health
    float damage
    float y
    float x

cpdef player_dtype():
    '''Make a dummy player to get the dtype'''
    cdef Player player
    return np.asarray(<Player[:1]>&player).dtype

cpdef creep_dtype():
    '''Make a dummy creep to get the dtype'''
    cdef Creep creep
    return np.asarray(<Creep[:1]>&creep).dtype

cpdef tower_dtype():
    '''Make a dummy tower to get the dtype'''
    cdef Tower tower
    return np.asarray(<Tower[:1]>&tower).dtype

cdef class Environment:
    cdef:
        int width
        int height
        int num_agents
        int num_creeps
        int num_towers
        int vision_range
        float agent_speed
        bint discretize
        int obs_size

        unsigned char[:, :] grid
        unsigned char[:, :, :] observations
        float[:] rewards
        int[:, :] pid_map
        Player* players
        Creep* creeps
        Tower* towers

    def __init__(self, grid, cnp.ndarray players, cnp.ndarray creeps,
            cnp.ndarray towers, observations, rewards,
            int vision_range, float agent_speed, bint discretize):
        self.height = grid.shape[0]
        self.width = grid.shape[1]
        self.num_agents = players.shape[0]
        self.num_creeps = creeps.shape[0]
        self.num_towers = towers.shape[0]
        self.vision_range = vision_range
        self.agent_speed = agent_speed
        self.discretize = discretize
        self.obs_size = 2*vision_range + 1

        self.grid = grid
        self.observations = observations
        self.rewards = rewards

        self.pid_map = np.zeros((self.height, self.width), dtype=np.int32) - 1

        self.players = <Player*> players.data
        self.creeps = <Creep*> creeps.data
        self.towers = <Tower*> towers.data

        cdef Player* player
        cdef int i, y, x
        for i in range(self.num_agents):
            player = self.get_player(i)
            player.health = 3
            if i < self.num_agents//2:
                player.team = 0
                player.spawn_y = 64 + i
                player.spawn_x = 62
            else:
                player.team = 1
                player.spawn_y = 64 + i
                player.spawn_x = 66

            player.y = player.spawn_y
            player.x = player.spawn_x
            y = int(player.y)
            x = int(player.x)

            self.pid_map[y, x] = i
            print(player.team, player.health, player.y, player.x)

        '''
        for i in range(self.num_agents, self.num_agents + self.num_creeps):
            creep = self.get_creep(i)
            creep.health = 1
            if i < self.num_agents//2:
                creep.team = 0
                creep.spawn_y = 64 + i
                creep.spawn_x = 62
        '''

        cdef Tower* tower
        tower = self.get_tower(0)
        tower.team = 0
        tower.health = 3
        tower.damage = 1
        tower.y = 43
        tower.x = 25

        tower = self.get_tower(1)
        tower.team = 0
        tower.health = 3
        tower.damage = 1
        tower.y = 74
        tower.x = 55

        tower = self.get_tower(2)
        tower.team = 0
        tower.health = 3
        tower.damage = 1
        tower.y = 104
        tower.x = 96

        tower = self.get_tower(3)
        tower.team = 0
        tower.health = 15
        tower.damage = 3
        tower.y = 72
        tower.x = 25

        tower = self.get_tower(4)
        tower.team = 0
        tower.health = 15
        tower.damage = 3
        tower.y = 85
        tower.x = 46

        tower = self.get_tower(5)
        tower.team = 0
        tower.health = 15
        tower.damage = 3
        tower.y = 106
        tower.x = 66

        tower = self.get_tower(6)
        tower.team = 0
        tower.health = 50
        tower.damage = 10
        tower.y = 97
        tower.x = 25

        tower = self.get_tower(7)
        tower.team = 0
        tower.health = 50
        tower.damage = 10
        tower.y = 99
        tower.x = 29

        tower = self.get_tower(8)
        tower.team = 0
        tower.health = 50
        tower.damage = 10
        tower.y = 104
        tower.x = 33

        for i in range(self.num_towers//2):
            self.get_tower(self.num_towers//2 + i).team = 1
            self.get_tower(self.num_towers//2 + i).health = self.get_tower(i).health
            self.get_tower(self.num_towers//2 + i).damage = self.get_tower(i).damage
            self.get_tower(self.num_towers//2 + i).y = 128 - self.get_tower(i).y
            self.get_tower(self.num_towers//2 + i).x = 128 - self.get_tower(i).x

        for i in range(self.num_towers):
            tower = self.get_tower(i)
            self.grid[int(tower.y), int(tower.x)] = TOWER

            

    cdef Player* get_player(self, int pid):
        #if pid < self.num_agents:
        return &self.players[pid]
        #else:
        #    return &self.creeps[pid - self.num_agents]

    cdef Tower* get_tower(self, int pid):
        return &self.towers[pid]

    cdef void compute_observations(self):
        cdef:
            float y
            float x
            int r
            int c
            int agent_idx
            Player* player

        for agent_idx in range(self.num_agents):
            player = self.get_player(agent_idx)
            y = player.y
            x = player.x
            r = int(y)
            c = int(x)
            self.observations[agent_idx, :] = self.grid[
                r-self.vision_range:r+self.vision_range+1,
                c-self.vision_range:c+self.vision_range+1
            ]

    cdef void spawn_agent(self, int agent_idx):
        cdef int old_r, old_c, r, c, tile
        cdef Player* player

        # Delete agent from old position
        player = self.get_player(agent_idx)
        old_r = int(player.y)
        old_c = int(player.x)
        self.grid[old_r, old_c] = EMPTY

        r = rand() % (self.height - 1)
        c = rand() % (self.width - 1)
        tile = self.grid[r, c]
        if tile == EMPTY:
            # Spawn agent in new position
            self.grid[r, c] = self.agent_colors[agent_idx]
            player.y = r
            player.x = c
            return

    def reset(self, seed=0):
        # Add borders
        cdef int left = int(self.agent_speed * self.vision_range)
        cdef int right = self.width - int(self.agent_speed*self.vision_range) - 1
        cdef int bottom = self.height - int(self.agent_speed*self.vision_range) - 1
        self.compute_observations()

    cdef move_to(self, int pid, float dest_y, float dest_x):
        cdef:
            Player* player = self.get_player(pid)

        self.grid[int(player.y), int(player.x)] = EMPTY
        if player.team == 0:
            self.grid[int(dest_y), int(dest_x)] = AGENT_1
        else:
            self.grid[int(dest_y), int(dest_x)] = AGENT_2
        self.pid_map[int(player.y), int(player.x)] = -1
        self.pid_map[int(dest_y), int(dest_x)] = pid
        player.y = dest_y
        player.x = dest_x

    cdef respawn(self, int pid):
        cdef Player* player = self.get_player(pid)
        self.move_to(pid, player.spawn_y, player.spawn_x)
        player.health = 3

    def step(self, np_actions):
        cdef:
            float[:, :] actions_continuous
            unsigned int[:, :] actions_discrete
            int agent_idx
            float y
            float x
            float vel_y
            float vel_x
            int attack
            int disc_y
            int disc_x
            int disc_dest_y
            int disc_dest_x
            Player* player
            Player* target
            int pid
            int target_pid
            Tower* tower
            float damage
            int dy
            int dx

        if self.discretize:
            actions_discrete = np_actions
        else:
            actions_continuous = np_actions

        for tower_idx in range(self.num_towers):
            tower = self.get_tower(tower_idx)
            damage = tower.damage
            y = tower.y
            x = tower.x

            for dy in range(-TOWER_VISION, TOWER_VISION+1):
                for dx in range(-TOWER_VISION, TOWER_VISION+1):
                    disc_y = int(y + dy)
                    disc_x = int(x + dx)

                    pid = self.pid_map[disc_y, disc_x]
                    if pid != -1:
                        target = self.get_player(pid)
                        if target.team == tower.team:
                            continue

                        target.health -= damage
                        if target.health <= 0:
                            self.respawn(pid)

        for agent_idx in range(self.num_agents):
            player = self.get_player(agent_idx)
            y = player.y
            x = player.x

            # Attacks
            attack = actions_discrete[agent_idx, 2]
            dest_y = y
            dest_x = x
            if attack == 0:
                dest_y = y + self.agent_speed
            elif attack == 1:
                dest_x = x + self.agent_speed
            elif attack == 2:
                dest_y = y - self.agent_speed
            elif attack == 3:
                dest_x = x - self.agent_speed

            disc_dest_y = int(dest_y)
            disc_dest_x = int(dest_x)
            target_pid = self.pid_map[disc_dest_y, disc_dest_x]
            if target_pid != -1:
                target = self.get_player(target_pid)
                if target.team != player.team:
                    target.health -= 1
                    if target.health <= 0:
                        self.respawn(target_pid)

            if self.discretize:
                # Convert [0, 1, 2] to [-1, 0, 1]
                vel_y = float(actions_discrete[agent_idx, 0]) - 1.0
                vel_x = float(actions_discrete[agent_idx, 1]) - 1.0
            else:
                vel_y = actions_continuous[agent_idx, 0]
                vel_x = actions_continuous[agent_idx, 1]

            dest_y = y + self.agent_speed * vel_y
            dest_x = x + self.agent_speed * vel_x

            # Discretize
            disc_y = int(y)
            disc_x = int(x)
            disc_dest_y = int(dest_y)
            disc_dest_x = int(dest_x)

            if self.grid[disc_dest_y, disc_dest_x] == 0:
                self.grid[disc_y, disc_x] = EMPTY
                if player.team == 0:
                    self.grid[disc_dest_y, disc_dest_x] = AGENT_1
                else:
                    self.grid[disc_dest_y, disc_dest_x] = AGENT_2

                # Continuous position update
                player.y = dest_y
                player.x = dest_x

        self.compute_observations()
