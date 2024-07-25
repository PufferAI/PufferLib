# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=True
# cython: initializedcheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: nonecheck=True
# cython: profile=False

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrtf
from cpython.list cimport PyList_GET_ITEM
cimport cython
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
    int NEUTRAL = 7
    int DEBUG = 8

    int PASS = 0
    int NORTH = 1
    int SOUTH = 2
    int EAST = 3
    int WEST = 4

    int TOWER_VISION = 5
    int CREEP_VISION = 5
    int NEUTRAL_VISION = 3

    int ENTITY_PLAYER = 0
    int ENTITY_CREEP = 1
    int ENTITY_NEUTRAL = 2
    int ENTITY_TOWER = 3

ctypedef struct Entity:
    int pid
    int team
    int type
    float health
    float max_health
    float mana
    float max_mana
    float y
    float x
    float spawn_y
    float spawn_x
    float damage
    int lane
    int waypoint
    float move_speed
    float move_modifier
    int stun_timer
    int move_timer
    int q_timer
    int w_timer
    int e_timer
    int basic_attack_timer
    int basic_attack_cd
    int is_hit
    int level
    int xp
    int xp_on_kill
    float reward

cdef struct EntityList:
    Entity* entity
    EntityList* next


def step_all(list envs):
    cdef:
        int n = len(envs)
        int i

    for i in range(n):
        (<Environment>PyList_GET_ITEM(envs, i)).step()
  
cpdef entity_dtype():
    '''Make a dummy entity to get the dtype'''
    cdef Entity entity
    return np.asarray(<Entity[:1]>&entity).dtype

@cython.profile(False)
cdef int xp_for_player_kill(Entity* player):
    return 100 + int(player.xp / 7.69)
 
@cython.profile(False)
cdef float clip(float x):
    # Clip to [-1, 1]
    return max(-1, min(x, 1))

@cython.profile(False)
cdef float l2_distance(float x1, float y1, float x2, float y2):
    '''Sqrt throws possible error which calls back to Python...
    Maybe some way to avoid negative sqrt check? Or just use L1'''
    cdef:
        float dx = x1 - x2
        float dy = y1 - y2

    return sqrtf(dx*dx + dy*dy)

@cython.profile(False)
cdef inline float l1_distance(float x1, float y1, float x2, float y2):
    return abs(x1 - x2) + abs(y1 - y2)

cdef class Environment:
    cdef:
        int width
        int height
        int num_agents
        int num_creeps
        int num_neutrals
        int num_towers
        int vision_range
        float agent_speed
        bint discretize
        int obs_size
        int creep_idx
        int tick
        int[:] xp_for_level

        unsigned char[:, :] grid
        unsigned char[:, :] orig_grid
        unsigned char[:, :, :, :] ai_paths
        unsigned char[:, :, :] observations_map
        unsigned char[:, :] observations_extra

        float[:] rewards
        #int[:, :] actions
        int[:] actions
        int[:, :] pid_map
        Entity[:, :] player_obs
        Entity[:] entities

        float[:, :, :] waypoints 
        dict entity_data
        float[:, :] neutral_spawns
        int[:, :] tower_spawns

        # MAX_ENTITIES x MAX_SCANNED_TARGETS
        Entity* scanned_targets[256][121]

        int rng_n
        int rng_idx
        float[:] rng

    def __init__(self, cnp.ndarray grid, cnp.ndarray ai_paths,
            cnp.ndarray pids, cnp.ndarray entities, dict entity_data,
            cnp.ndarray player_obs, cnp.ndarray observations_map, cnp.ndarray observations_extra,
            cnp.ndarray rewards, cnp.ndarray actions, int num_agents, int num_creeps, int num_neutrals,
            int num_towers, int vision_range, float agent_speed, bint discretize):
        self.height = grid.shape[0]
        self.width = grid.shape[1]
        self.num_agents = num_agents
        self.num_creeps = num_creeps
        self.num_neutrals = num_neutrals
        self.num_towers = num_towers
        self.vision_range = vision_range
        self.agent_speed = agent_speed
        self.discretize = discretize
        self.obs_size = 2*vision_range + 1
        self.creep_idx = 0

        # Hey, change the scanned_targets size to match!
        assert num_agents + num_creeps + num_neutrals + num_towers <= 256
        assert self.obs_size * self.obs_size <= 121

        self.grid = grid
        self.orig_grid = grid.copy()
        self.ai_paths = ai_paths
        self.observations_map = observations_map
        self.observations_extra = observations_extra
        self.rewards = rewards
        self.actions = actions

        self.pid_map = pids
        self.entities = entities
        self.entity_data = entity_data
        self.player_obs = player_obs

        # Preallocate RNG -1 to 1
        self.rng_n = 10000
        self.rng_idx = 0
        self.rng = 2*np.random.rand(self.rng_n).astype(np.float32) - 1

        self.xp_for_level = np.array([
            0, 240, 640, 1160, 1760, 2440, 3200, 4000, 4900, 4900, 7000, 8200,
            9500, 10900, 12400, 14000, 15700, 17500, 19400, 21400, 23600, 26000,
            28600, 31400, 34400, 38400, 43400, 49400, 56400, 63900], dtype=np.int32)

        self.waypoints = np.zeros((6, 20, 2), dtype=np.float32)
        for team in range(2):
            if team == 0:
                prefix = 'npc_dota_spawner_good_'
            else:

                prefix = 'npc_dota_spawner_bad_'

            for lane in range(3):
                suffix = ['top', 'mid', 'bot'][lane]
                key = f'{prefix}{suffix}'
                y = entity_data[key]['spawn_y']
                x = entity_data[key]['spawn_x']
                self.waypoints[3*team + lane, 0, 0] = y
                self.waypoints[3*team + lane, 0, 1] = x
                #self.grid[int(y), int(x)] = DEBUG
                waypoints = entity_data[key]['waypoints']
                for i in range(len(waypoints)):
                    y = waypoints[i]['y']
                    x = waypoints[i]['x']
                    self.waypoints[3*team + lane, i+1, 0] = y
                    self.waypoints[3*team + lane, i+1, 1] = x
                    #self.grid[int(y), int(x)] = DEBUG


        self.tower_spawns = np.zeros((num_towers, 4), dtype=np.int32)
        # y, x, team, tier

        idx = 0
        for team in range(2):
            if team == 0:
                prefix = 'dota_goodguys_tower'
            else:
                prefix = 'dota_badguys_tower'

            for tier in range(1, 5):
                for suffix in ['_top', '_mid', '_bot']:
                    if tier == 4 and suffix == '_mid':
                        continue # no mid tier 4 towers

                    tower_name = f'{prefix}{tier}{suffix}'
                    self.tower_spawns[idx, 0] = int(entity_data[tower_name]['y'])
                    self.tower_spawns[idx, 1] = int(entity_data[tower_name]['x'])
                    self.tower_spawns[idx, 2] = team
                    self.tower_spawns[idx, 3] = tier
                    idx += 1

        # Hardcode ancients
        tower_name = 'dota_goodguys_fort'
        self.tower_spawns[idx, 0] = int(entity_data[tower_name]['y'])
        self.tower_spawns[idx, 1] = int(entity_data[tower_name]['x'])
        self.tower_spawns[idx, 2] = 0
        self.tower_spawns[idx, 3] = 5

        idx += 1
        tower_name = 'dota_badguys_fort'
        self.tower_spawns[idx, 0] = entity_data[tower_name]['y']
        self.tower_spawns[idx, 1] = entity_data[tower_name]['x']
        self.tower_spawns[idx, 2] = 1
        self.tower_spawns[idx, 3] = 5

        # Num camps
        self.neutral_spawns = np.zeros((18, 2), dtype=np.float32)
        idx = 0
        for team in range(2):
            if team == 0:
                prefix = 'neutralcamp_good_'
            else:
                prefix = 'neutralcamp_evil_'

            for i in range(1, 10):
                neutral_name = f'{prefix}{i}'
                self.neutral_spawns[idx, 0] = entity_data[neutral_name]['y']
                self.neutral_spawns[idx, 1] = entity_data[neutral_name]['x']
                idx += 1

    @cython.profile(False)
    cdef int level(self, xp):
        cdef int i
        for i in range(len(self.xp_for_level)):
            if xp < self.xp_for_level[i]:
                return i + 1

        return i+1
     
    @cython.profile(False)
    cdef Entity* get_player_ob(self, int pid, int idx):
        return &self.player_obs[pid, idx]

    @cython.profile(False)
    cdef Entity* get_entity(self, int pid):
        return &self.entities[pid]

    @cython.profile(False)
    cdef Entity* get_creep(self, int idx):
        return &self.entities[idx + self.num_agents]

    @cython.profile(False)
    cdef Entity* get_neutral(self, int idx):
        return &self.entities[idx + self.num_agents + self.num_creeps]

    @cython.profile(False)
    cdef Entity* get_tower(self, int idx):
        return &self.entities[idx + self.num_agents + self.num_creeps + self.num_neutrals]

    cdef void compute_observations(self):
        cdef:
            int r
            int c
            int x
            int y
            int idx
            int pid
            int target_pid
            Entity* player
            Entity* target
            Entity* target_ob

        # TODO: Figure out how to zero data
        #self.player_obs[:, :] = 0

        for pid in range(self.num_agents):
            for idx in range(10):
                self.get_player_ob(pid, idx).pid = -1

            player = self.get_entity(pid)
            y = int(player.y)
            x = int(player.x)
            self.observations_map[pid, :] = self.grid[
                y-self.vision_range:y+self.vision_range+1,
                x-self.vision_range:x+self.vision_range+1
            ]
            self.observations_extra[pid, 0] = <unsigned char> player.x
            self.observations_extra[pid, 1] = <unsigned char> player.y
            self.observations_extra[pid, 2] = <unsigned char> (200*self.rewards[pid] + 32)

            idx = 0
            # TODO: sort by distance
            for r in range(y-self.vision_range, y+self.vision_range+1):
                for c in range(x-self.vision_range, x+self.vision_range+1):
                    target_pid = self.pid_map[r, c]
                    if target_pid == -1:
                        continue

                    target = self.get_entity(target_pid)
                    # TODO: figure out how to copy all
                    target_ob = self.get_player_ob(pid, idx)
                    target_ob.pid = target_pid
                    target_ob.y = target.y
                    target_ob.x = target.x
                    target_ob.health = target.health
                    target_ob.team = target.team
                    target_ob.type = target.type

                    idx += 1
                    if idx == 10:
                        break

                if idx == 10:
                    break

    cdef void spawn_all_towers(self):
        cdef:
            int idx

        for idx in range(self.num_towers):
            self.spawn_tower(idx, self.tower_spawns[idx, 2], self.tower_spawns[idx, 3],
                self.tower_spawns[idx, 0], self.tower_spawns[idx, 1])

    cdef void spawn_tower(self, int idx, int team, int tier, int y, int x):
        cdef Entity* tower = self.get_tower(idx)
        tower.type = ENTITY_TOWER
        tower.pid = idx + self.num_agents + self.num_creeps + self.num_neutrals
        tower.team = team
        tower.basic_attack_cd = 5

        if tier == 1:
            tower.health = 1800
            tower.max_health = 1800
            tower.damage = 100
            tower.xp_on_kill = 800
        elif tier == 2:
            tower.health = 2500
            tower.max_health = 2500
            tower.damage = 190
            tower.xp_on_kill = 1600
        elif tier == 3:
            tower.health = 2500
            tower.max_health = 2500
            tower.damage = 190
            tower.xp_on_kill = 2400
        elif tier == 4:
            # TODO: Look up damage
            tower.health = 2500
            tower.max_health = 2500
            tower.damage = 190
            tower.xp_on_kill = 3200
        elif tier == 5:
            tower.health = 4500
            tower.max_health = 4500
            tower.damage = 0
            tower.xp_on_kill = 0
        else:
            raise ValueError('Invalid tier')

        self.move_to(tower, y, x)

    def reset(self, seed=0):
        cdef:
            Entity* player
            Entity* tower
            int pid
            int y
            int x

        self.grid[:] = self.orig_grid
        self.pid_map[:] = -1

        self.spawn_all_towers()

        self.tick = 0
        for pid in range(self.num_agents):
            self.rewards[pid] = 0
            player = self.get_entity(pid)
            player.pid = pid
            player.type = ENTITY_PLAYER
            player.max_health = 500
            player.max_mana = 100
            player.move_speed = self.agent_speed
            player.move_modifier = 0
            player.move_timer = 0
            player.stun_timer = 0
            player.level = 1
            player.basic_attack_cd = 8
            player.damage = 50
            player.xp = 0

            if pid < self.num_agents//2:
                player.team = 0
            else:
                player.team = 1

            self.respawn_player(player)

        # Hardcode lanes for now
        self.get_entity(0).lane = 2
        self.get_entity(1).lane = 2
        self.get_entity(4).lane = 2
        self.get_entity(2).lane = 1
        self.get_entity(3).lane = 0

        self.get_entity(5).lane = 3
        self.get_entity(6).lane = 3
        self.get_entity(9).lane = 3
        self.get_entity(7).lane = 4
        self.get_entity(8).lane = 5

        self.compute_observations()

    @cython.profile(False)
    cdef bint move_to(self, Entity* player, float dest_y, float dest_x):
        cdef:
            int disc_y = int(player.y)
            int disc_x = int(player.x)
            int disc_dest_y = int(dest_y)
            int disc_dest_x = int(dest_x)
            int agent_type

        if (self.grid[disc_dest_y, disc_dest_x] != EMPTY and
                self.pid_map[disc_dest_y, disc_dest_x] != player.pid):
            return False

        if player.type == ENTITY_TOWER:
            agent_type = TOWER
        elif player.type == ENTITY_CREEP:
            if player.team == 0:
                agent_type = CREEP_1
            else:
                agent_type = CREEP_2
        elif player.type == ENTITY_NEUTRAL:
            agent_type = NEUTRAL
        elif player.type == ENTITY_PLAYER:
            if player.team == 0:
                agent_type = AGENT_1
            else:
                agent_type = AGENT_2

        self.grid[disc_y, disc_x] = EMPTY
        self.grid[disc_dest_y, disc_dest_x] = agent_type

        self.pid_map[disc_y, disc_x] = -1
        self.pid_map[disc_dest_y, disc_dest_x] = player.pid

        player.y = dest_y
        player.x = dest_x
        return True

    cdef bint move_near(self, Entity* entity, Entity* target):
        if entity.pid == target.pid:
            return False
        if self.move_to(entity, target.y+1.0, target.x) == 1:
            return True
        elif self.move_to(entity, target.y-1.0, target.x) == 1:
            return True
        elif self.move_to(entity, target.y, target.x+1.0) == 1:
            return True
        elif self.move_to(entity, target.y, target.x-1.0) == 1:
            return True
        elif self.move_to(entity, target.y+1.0, target.x+1.0) == 1:
            return True
        elif self.move_to(entity, target.y+1.0, target.x-1.0) == 1:
            return True
        elif self.move_to(entity, target.y-1.0, target.x+1.0) == 1:
            return True
        elif self.move_to(entity, target.y-1.0, target.x-1.0) == 1:
            return True
        else:
            return False

    cdef void kill(self, Entity* entity):
        cdef:
            int x = int(entity.x)
            int y = int(entity.y)

        self.grid[y, x] = EMPTY
        self.pid_map[y, x] = -1
        entity.pid = -1

    @cython.profile(False)
    cdef bint move_towards(self, Entity* entity, float speed, int dest_y, int dest_x):
        speed = entity.move_modifier * speed

        cdef:
            int entity_y = int(entity.y)
            int entity_x = int(entity.x)
            int atn = self.ai_paths[dest_y, dest_x, entity_y, entity_x]
            int dy = 0
            int dx = 0
            float jitter_x
            float jitter_y
            int move_to_y
            int move_to_x

        if atn == 0:
            dy = 1
        elif atn == 1:
            dy = -1
        elif atn == 2:
            dx = 1
        elif atn == 3:
            dx = -1
        elif atn == 4:
            dy = 1
            dx = -1
        elif atn == 5:
            dy = -1
            dx = -1
        elif atn == 6:
            dy = -1
            dx = 1
        elif atn == 7:
            dy = 1
            dx = 1
        #else:
        #    dy = 1
        #    dx = 0
 
        move_to_y = int(entity.y + dy*speed)
        move_to_x = int(entity.x + dx*speed)
        if (self.grid[move_to_y, move_to_x] != EMPTY and
                self.pid_map[move_to_y, move_to_x] != entity.pid):
            jitter_x = self.rng[self.rng_idx]
            jitter_y = self.rng[self.rng_idx + 1]

            self.rng_idx += 2
            if self.rng_idx >= self.rng_n - 1:
                self.rng_idx = 0

            #jitter_x = 2*(rand()/(RAND_MAX + 1.0) - 0.5)
            #jitter_y = 2*(rand()/(RAND_MAX + 1.0) - 0.5)
            return self.move_to(entity,
                entity.y + jitter_y, entity.x + jitter_x)

        return self.move_to(entity,
            entity.y + dy*speed,
            entity.x + dx*speed)

    cdef void neutral_ai(self, Entity* neutral):
        cdef:
            Entity* target
            int pid = neutral.pid

        if self.tick % 5 == 0:
            self.scan_aoe(neutral, NEUTRAL_VISION,
                exclude_friendly=True, exclude_hostile=False,
                exclude_creeps=True, exclude_neutrals=True,
                exclude_towers=True)

        if self.scanned_targets[pid][0] != NULL:
            target = self.nearest_scanned_target(neutral)
            #if l2_distance(neutral.y, neutral.x, target.y, target.x) < 2:
            if abs(neutral.y - target.y) + abs(neutral.x - target.x) < 2:
                self.basic_attack(neutral, target)
            else:
                self.move_towards(neutral, self.agent_speed, int(target.y), int(target.x))
        #elif l2_distance(neutral.y, neutral.x, neutral.spawn_y, neutral.spawn_x) > 2:
        elif abs(neutral.y - neutral.spawn_y) + abs(neutral.x - neutral.spawn_x) > 2:
            self.move_towards(neutral, self.agent_speed, int(neutral.spawn_y), int(neutral.spawn_x))

    cdef void spawn_neutral(self, int idx, int camp):
        cdef:
            int pid = idx + self.num_agents + self.num_creeps
            Entity* neutral = self.get_entity(pid)
            int dy, dx

        neutral.pid = pid
        neutral.type = ENTITY_NEUTRAL
        neutral.health = 500
        neutral.max_health = 500
        neutral.mana = 0
        neutral.max_mana = 100
        neutral.team = 2
        neutral.spawn_y = self.neutral_spawns[camp, 0]
        neutral.spawn_x = self.neutral_spawns[camp, 1]
        neutral.xp_on_kill = 35
        neutral.basic_attack_cd = 5
        neutral.damage = 22

        for i in range(10):
            dy = rand() % 7 - 3
            dx = rand() % 7 - 3
            if self.move_to(neutral, neutral.spawn_y + dy, neutral.spawn_x + dx):
                break

    cdef void creep_ai(self, Entity* creep):
        cdef:
            int waypoint = creep.waypoint
            int lane = creep.lane
            int pid = creep.pid
            float dest_y, dest_x
            float dist
            Entity* target

        if self.tick % 5 == 0:
            self.scan_aoe(creep, CREEP_VISION, exclude_friendly=True,
                exclude_hostile=False, exclude_creeps=False,
                exclude_neutrals=True, exclude_towers=False)

        if self.scanned_targets[pid][0] != NULL:
            target = self.nearest_scanned_target(creep)
            dest_y = target.y
            dest_x = target.x

            #dist = l2_distance(creep.y, creep.x, dest_y, dest_x)
            dist = abs(creep.y - dest_y) + abs(creep.x - dest_x)
            if dist < 2:
                self.basic_attack(creep, target)
            self.move_towards(creep, self.agent_speed, int(dest_y), int(dest_x))
        else:
            dest_y = self.waypoints[lane, waypoint, 0]
            dest_x = self.waypoints[lane, waypoint, 1]
            self.move_towards(creep, self.agent_speed, int(dest_y), int(dest_x))
            # TODO: Last waypoint?
            #if l2_distance(creep.y, creep.x, dest_y, dest_x) < 2 and self.waypoints[lane, waypoint+1, 0] != 0:
            if abs(creep.y - dest_y) + abs(creep.x - dest_x) < 2 and self.waypoints[lane, waypoint+1, 0] != 0:
                creep.waypoint += 1

    cdef void spawn_creep(self, int idx, int lane):
        cdef:
            int pid = idx + self.num_agents
            Entity* creep = self.get_entity(pid)
            int team 

        if lane < 3:
            team = 0
        else:
            team = 1

        creep.pid = pid
        creep.type = ENTITY_CREEP
        creep.health = 450
        creep.max_health = 450
        creep.team = team
        creep.lane = lane
        creep.waypoint = 0
        creep.xp_on_kill = 60
        creep.damage = 22
        creep.basic_attack_cd = 5

        self.respawn_creep(creep, lane)

    cdef void spawn_creep_wave(self):
        cdef int lane, creep
        for lane in range(6):
            for creep in range(5):
                self.spawn_creep(self.creep_idx, lane)
                self.creep_idx = (self.creep_idx + 1) % self.num_creeps

    cdef void respawn_player(self, Entity* entity):
        cdef:
            bint valid_pos = False
            int spawn_y
            int spawn_x
            int y, x

        if entity.team == 0:
            y = 128 - 15
            x = 12
        else:
            y = 15
            x = 128 - 12

        while not valid_pos:
            spawn_y = y + rand() % 15 - 7
            spawn_x = x + rand() % 15 - 7
            if self.grid[spawn_y, spawn_x] == EMPTY:
                valid_pos = True
                break

        self.move_to(entity, spawn_y, spawn_x)
        entity.health = entity.max_health
        entity.mana = entity.max_mana
        entity.waypoint = 0

    cdef bint respawn_creep(self, Entity* entity, int lane):
        cdef:
            bint valid_pos = False
            int spawn_y = int(self.waypoints[lane, 0, 0])
            int spawn_x = int(self.waypoints[lane, 0, 1])
            int x, y

        #self.grid[spawn_y, spawn_x] = DEBUG

        for i in range(10):
            y = spawn_y + rand() % 7 - 3
            x = spawn_x + rand() % 7 - 3
            if self.grid[y, x] == EMPTY:
                valid_pos = True
                break

        self.move_to(entity, y, x)
        entity.health = entity.max_health
        entity.waypoint = 1
        return valid_pos

    cdef bint valid_target(self, Entity* target):
        if target.pid == -1:
            return False

        if target.team != target.team:
            return False

        return True

    cdef bint attack(self, Entity* player, Entity* target, float damage):
        cdef:
            int xp = 0
            float reward

        if target.pid == -1:
            return False

        if target.team == player.team:
            return False

        target.is_hit = 1
        target.health -= damage
        if target.health > 0:
            return True

        if target.type == ENTITY_PLAYER:
            self.respawn_player(target)
        elif (target.type == ENTITY_TOWER or target.type == ENTITY_CREEP
                or target.type == ENTITY_NEUTRAL):
            self.kill(target)

        if player.type == ENTITY_PLAYER:
            if player.xp < 10000000:
                if target.type == ENTITY_PLAYER:
                    xp = xp_for_player_kill(target)
                else:
                    xp = target.xp_on_kill

                player.xp += xp
                reward = xp / 100.0
                if reward > 1:
                    reward = 1
                player.reward += reward

            player.level = self.level(player.xp)
            player.damage = 50 + 6*player.level
            player.max_health = 500 + 50*player.level
 
        return True

    @cython.profile(False)
    cdef bint basic_attack(self, Entity* player, Entity* target):
        if player.basic_attack_timer > 0:
            return False

        player.basic_attack_timer = player.basic_attack_cd
        return self.attack(player, target, player.damage)

    cdef bint heal(self, Entity* player, Entity* target, float amount):
        if target.pid == -1:
            return False

        if target.team != player.team:
            return False

        if target.type != ENTITY_PLAYER:
            return False

        target.health += amount
        if target.health > target.max_health:
            target.health = target.max_health

        return True

    @cython.profile(False)
    cdef scan_aoe(self, Entity* player, int radius,
            bint exclude_friendly, bint exclude_hostile, bint exclude_creeps,
            bint exclude_neutrals, bint exclude_towers):
        cdef:
            int player_y = int(player.y)
            int player_x = int(player.x)
            int player_team = player.team
            int pid = player.pid
            int idx = 0
            Entity* target
            int target_team, target_type
            int y, x

        for y in range(player_y-radius, player_y+radius+1):
            for x in range(player_x-radius, player_x+radius+1):
                target_pid = self.pid_map[y, x]
                if target_pid == -1:
                    continue

                target = self.get_entity(target_pid)
                target_team = target.team

                if exclude_friendly and target_team == player_team:
                    continue

                if exclude_hostile and target_team != player_team:
                    continue

                target_type = target.type
                if exclude_neutrals and target_type == ENTITY_NEUTRAL:
                    continue

                if exclude_creeps and target_type == ENTITY_CREEP:
                    continue

                if exclude_towers and target_type == ENTITY_TOWER:
                    continue

                self.scanned_targets[pid][idx] = target
                idx += 1

        self.scanned_targets[pid][idx] = NULL

    cdef void aoe_scanned(self, Entity* player,
            Entity* target, float damage, int stun):
        cdef:
            int pid = player.pid
            int idx

        for idx in range(121):
            target = self.scanned_targets[pid][idx]
            if target == NULL:
                break

            if damage < 0:
                self.heal(player, target, -damage)
                continue

            self.attack(player, target, damage)
            if stun > 0:
                target.stun_timer = stun

    cdef bint player_aoe_attack(self, Entity* player,
            Entity* target, int radius, float damage, int stun):
        cdef bint success = self.scan_aoe(player, radius,
            exclude_friendly=True, exclude_hostile=False,
            exclude_creeps=False, exclude_neutrals=False,
            exclude_towers=False)

        if not success:
            return False

        self.aoe_scanned(player, target, damage, stun)
        return True

    @cython.profile(False)
    cdef Entity* nearest_scanned_target(self, Entity* player):
        cdef:
            Entity* nearest_target = NULL
            Entity* target
            float nearest_dist = 9999999
            float player_y = player.y
            float player_x = player.x
            float dist
            int idx = 0
            int pid = player.pid

        for idx in range(121):
            target = self.scanned_targets[pid][idx]
            if target == NULL:
                break

            #dist = l2_distance(player.y, player.x, target.y, target.x)
            dist = abs(player_y - target.y) + abs(player_x - target.x)
            if dist < nearest_dist:
                nearest_target = target
                nearest_dist = dist

        return nearest_target

    cdef bint aoe_push(self, Entity* player, radius, float amount):
        cdef:
            Entity* target
            int idx = 0
            int pid = player.pid

        self.scan_aoe(player, radius, exclude_friendly=True,
            exclude_hostile=False, exclude_creeps=False,
            exclude_neutrals=False, exclude_towers=True)

        for idx in range(121):
            target = self.scanned_targets[pid][idx]
            if target == NULL:
                break

            self.push(player, target, amount)
            success = True

        return success

    cdef bint push(self, Entity* player, Entity* target, float amount):
        cdef:
            #float dist = l2_distance(target.x, target.y, player.x, player.y)
            float dist = abs(target.x - player.x) + abs(target.y - player.y)
            float dx = target.x - player.x
            float dy = target.y - player.y

        if dist == 0.0:
            return False

        # Norm to unit vector
        dx = amount * dx / dist
        dy = amount * dy / dist

        return self.move_to(target, target.y + dy, target.x + dx)

    cdef bint pull(self, Entity* player, Entity* target, float amount):
        return self.push(target, player, -amount)

    @cython.profile(False)
    cdef void update_status(self, Entity* entity):
        if entity.stun_timer > 0:
            entity.stun_timer -= 1

        if entity.move_timer > 0:
            entity.move_timer -= 1

        if entity.move_timer == 0:
            entity.move_modifier = 1.0

    @cython.profile(False)
    cdef void update_cooldowns(self, Entity* entity):
        if entity.q_timer > 0:
            entity.q_timer -= 1

        if entity.w_timer > 0:
            entity.w_timer -= 1

        if entity.e_timer > 0:
            entity.e_timer -= 1

        if entity.basic_attack_timer > 0:
            entity.basic_attack_timer -= 1

    cdef void skill_support_hook(self, Entity* player, Entity* target):
        if player.mana < 15:
            return

        self.pull(target, player, 1.0)
        player.mana -= 15

    cdef void skill_support_aoe_heal(self, Entity* player, Entity* target):
        if player.mana < 40:
            return

        if self.player_aoe_attack(player, player, 4, -200, 0):
            player.mana -= 40
            player.w_timer = 60

    cdef void skill_support_stun(self, Entity* player, Entity* target):
        if player.mana < 60:
            return

        if self.attack(player, target, 50):
            target.stun_timer = 60
            player.mana -= 60
            player.e_timer = 50

    cdef void skill_burst_nuke(self, Entity* player, Entity* target):
        if player.mana < 60:
            return

        if self.attack(player, target, 500):
            player.mana -= 60
            player.q_timer = 70

    cdef void skill_burst_aoe(self, Entity* player, Entity* target):
        if player.mana < 40:
            return

        if self.player_aoe_attack(player, target, 2, 200, 0):
            player.mana -= 40
            player.w_timer = 40

    cdef void skill_burst_aoe_stun(self, Entity* player, Entity* target):
        if player.mana < 60:
            return

        if self.player_aoe_attack(player, target, 2, 0, 40):
            player.mana -= 60
            player.e_timer = 80

    cdef void skill_tank_aoe_dot(self, Entity* player, Entity* target):
        if player.mana < 5:
            return

        if self.player_aoe_attack(player, player, 2, 20, 0):
            player.mana -= 5

    cdef void skill_tank_self_heal(self, Entity* player, Entity* target):
        if player.mana < 30:
            return

        if self.heal(player, player, 250):
            player.mana -= 30
            player.w_timer = 60

    cdef void skill_tank_engage_aoe(self, Entity* player, Entity* target):
        if player.mana < 60:
            return

        if self.move_near(player, target):
            player.mana -= 60
            player.e_timer = 70
            self.aoe_push(player, 4, 3.0)

    cdef void skill_carry_retreat_slow(self, Entity* player, Entity* target):
        for i in range(3):
            if player.mana < 20:
                return

            if self.push(target, player, 1.5):
                target.move_timer = 15
                target.move_modifier = 0.5
                player.mana -= 20
                player.w_timer = 40

    cdef void skill_carry_slow_damage(self, Entity* player, Entity* target):
        if player.mana < 40:
            return

        if self.attack(player, target, 100):
            target.move_timer = 60
            target.move_modifier = 0.5
            player.mana -= 40
            player.w_timer = 40

    cdef void skill_carry_aoe(self, Entity* player, Entity* target):
        if player.mana < 40:
            return

        if self.player_aoe_attack(player, target, 2, 200, 0):
            player.mana -= 40
            player.e_timer = 40

    cdef void skill_assassin_aoe_minions(self, Entity* player, Entity* target):
        if player.mana < 40:
            return

        # Targeted on minions, splashes to players
        if (target.type == ENTITY_CREEP or target.type == ENTITY_NEUTRAL
                ) and self.player_aoe_attack(player, target, 3, 300, 0):
            player.mana -= 40
            player.q_timer = 40

    cdef void skill_assassin_tp_damage(self, Entity* player, Entity* target):
        if player.mana < 60:
            return

        if self.move_near(player, target) == -1:
            return

        player.mana -= 60
        if self.attack(player, target, 600):
            player.w_timer = 60

    cdef void skill_assassin_move_buff(self, Entity* player, Entity* target):
        if player.mana < 5:
            return

        player.move_modifier = 2.0
        player.move_timer = 1
        player.mana -= 5

    cdef int step(self):
        cdef:
            float[:] actions_continuous
            int[:] actions_discrete
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
            Entity* player
            Entity* target
            Entity* creep
            Entity* neutral
            Entity* tower
            int pid
            int target_pid
            float damage
            int atn
            int dy
            int dx
            bint use_q
            bint use_w
            bint use_e
            bint use_basic_attack
            float prev_dist_to_ancient
            float dist_to_ancient

        for pid in range(self.num_agents + self.num_towers + self.num_creeps):
            player = self.get_entity(pid)
            player.is_hit = 0

        #if self.discretize:
        #    actions_discrete = self.actions
        #else:
        #    actions_continuous = self.actions
        actions_discrete = self.actions

        # Neutral AI
        cdef int camp, neut
        if self.tick % 600 == 0:
            for camp in range(18):
                for neut in range(4):
                    self.spawn_neutral(4*camp + neut, camp)

        for idx in range(self.num_neutrals):
            neutral = self.get_neutral(idx)
            if neutral.pid == -1:
                continue

            self.update_status(neutral)
            self.update_cooldowns(neutral)
            if neutral.stun_timer > 0:
                continue

            self.neutral_ai(neutral)

        # Creep AI
        if self.tick % 150 == 0:
            self.spawn_creep_wave()

        for idx in range(self.num_creeps):
            creep = self.get_creep(idx)
            if creep.pid == -1:
                continue

            self.update_status(creep)
            self.update_cooldowns(creep)
            if creep.stun_timer > 0:
                continue

            self.creep_ai(creep)

        # Tower AI
        for tower_idx in range(self.num_towers):
            tower = self.get_tower(tower_idx)
            if tower.pid == -1:
                continue

            self.update_cooldowns(tower)
            if tower.basic_attack_timer > 0:
                continue

            if self.tick % 3 == 0: # Is this fast enough?
                self.scan_aoe(tower, TOWER_VISION, exclude_friendly=True,
                    exclude_hostile=False, exclude_creeps=False,
                    exclude_neutrals=True, exclude_towers=True)

            target = self.nearest_scanned_target(tower)
            if target != NULL:
                self.basic_attack(tower, target)

        # Player Logic
        for pid in range(self.num_agents):
            player = self.get_entity(pid)
            player.reward = 0
            self.rewards[pid] = 0

            if player.mana < player.max_mana:
                player.mana += 1

            if player.health < player.max_health:
                player.health += 1

            self.update_status(player)
            self.update_cooldowns(player)

            if player.stun_timer > 0:
                continue

            # Attacks
            if self.discretize:
                atn = actions_discrete[pid]
                if atn == 0:
                    vel_y = -1
                    vel_x = -1
                elif atn == 1:
                    vel_y = 0
                    vel_x = -1
                elif atn == 2:
                    vel_y = 1
                    vel_x = -1
                elif atn == 3:
                    vel_y = -1
                    vel_x = 0
                elif atn == 4:
                    vel_y = 0
                    vel_x = 0
                elif atn == 5:
                    vel_y = 1
                    vel_x = 0
                elif atn == 6:
                    vel_y = -1
                    vel_x = 1
                elif atn == 7:
                    vel_y = 0
                    vel_x = 1
                elif atn == 8:
                    vel_y = 1
                    vel_x = 1
                else:
                    raise ValueError('Invalid action')

                atn = 0#actions_discrete[pid, 1]
                use_q = False
                use_w = False
                use_e = False
                use_basic_attack = False
                if atn == 0:
                    use_basic_attack = True
                elif atn == 1:
                    use_q = True
                elif atn == 2:
                    use_w = True
                elif atn == 3:
                    use_e = True
                else:
                    raise ValueError('Invalid action')
            else:
                pass
                #vel_y = actions_continuous[pid, 0]
                #vel_x = actions_continuous[pid, 1]
                #attack = int(actions_continuous[pid, 2])
                # TODO: Breaks to python
                #use_q = int(actions_continuous[pid, 3]) > 0.5
                #use_w = int(actions_continuous[pid, 4]) > 0.5
                #use_e = int(actions_continuous[pid, 5]) > 0.5

            self.scan_aoe(player, self.vision_range, exclude_friendly=True,
                exclude_hostile=False, exclude_creeps=False,
                exclude_neutrals=False, exclude_towers=False)

            target = NULL
            if self.scanned_targets[pid][0] != NULL:
                target = self.nearest_scanned_target(player)

            # This is a copy. Have to get the real one
            #target = self.get_player_ob(pid, attack)
            #target = self.get_entity(target.pid)

            if target != NULL:
                if player.pid == 0 or player.pid == 5:
                    if use_q:
                        self.skill_support_hook(player, target)
                    elif use_w:
                        self.skill_support_aoe_heal(player, target)
                    elif use_e:
                        self.skill_support_stun(player, target)
                    else:
                        self.basic_attack(player, target)
                elif player.pid == 1 or player.pid == 6:
                    if use_q:
                        self.skill_assassin_aoe_minions(player, target)
                    elif use_w:
                        self.skill_assassin_tp_damage(player, target)
                    elif use_e:
                        self.skill_assassin_move_buff(player, target)
                    else:
                        self.basic_attack(player, target)
                elif player.pid == 2 or player.pid == 7:
                    if use_q:
                        self.skill_burst_nuke(player, target)
                    elif use_w:
                        self.skill_burst_aoe(player, target)
                    elif use_e:
                        self.skill_burst_aoe_stun(player, target)
                    else:
                        self.basic_attack(player, target)
                elif player.pid == 3 or player.pid == 8:
                    if use_q:
                        self.skill_tank_aoe_dot(player, target)
                    elif use_w:
                        self.skill_tank_self_heal(player, target)
                    elif use_e:
                        self.skill_tank_engage_aoe(player, target)
                    else:
                        self.basic_attack(player, target)
                elif player.pid == 4 or player.pid == 9:
                    if use_q:
                        self.skill_carry_retreat_slow(player, target)
                    elif use_w:
                        self.skill_carry_slow_damage(player, target)
                    elif use_e:
                        self.skill_carry_aoe(player, target)
                    else:
                        self.basic_attack(player, target)

            # Reward based on distance to enemy ancient
            if player.team == 0:
                ancient = self.get_tower(23)
            else:
                ancient = self.get_tower(22)
 
            dest_y = player.y + player.move_modifier*self.agent_speed*vel_y
            dest_x = player.x + player.move_modifier*self.agent_speed*vel_x

            prev_dist_to_ancient = abs(player.y - ancient.y) + abs(player.x - ancient.x)
            self.move_to(player, dest_y, dest_x)
            dist_to_ancient = abs(player.y - ancient.y) + abs(player.x - ancient.x)
            player.reward += 0.1*(prev_dist_to_ancient - dist_to_ancient)
            self.rewards[pid] = player.reward

        self.tick += 1
        self.compute_observations()

        if self.get_tower(22).health <= 0:
            self.reset(0)
            return 1 
        if self.get_tower(23).health <= 0:
            self.reset(0)
            return 2

        return 0

