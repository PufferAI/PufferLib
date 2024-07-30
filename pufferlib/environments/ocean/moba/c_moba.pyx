# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: profile=False

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrtf
from cpython.list cimport PyList_GET_ITEM
cimport cython
cimport numpy as cnp
import numpy as np

ctypedef void(*skill)(Environment, Entity*, Entity*)

cdef:
    # Grid IDs
    int EMPTY = 0
    int WALL = 1
    int TOWER = 2
    int RADIANT_CREEP = 3
    int DIRE_CREEP = 4
    int NEUTRAL = 5
    int RADIANT_SUPPORT = 6
    int RADIANT_ASSASSIN = 7
    int RADIANT_BURST = 8
    int RADIANT_TANK = 9
    int RADIANT_CARRY = 10
    int DIRE_SUPPORT = 11
    int DIRE_ASSASSIN = 12
    int DIRE_BURST = 13
    int DIRE_TANK = 14
    int DIRE_CARRY = 15

    int TOWER_VISION = 5
    int CREEP_VISION = 5
    int NEUTRAL_VISION = 3

    int ENTITY_PLAYER = 0
    int ENTITY_CREEP = 1
    int ENTITY_NEUTRAL = 2
    int ENTITY_TOWER = 3

cdef struct Entity:
    int pid
    int entity_type
    int hero_type
    int grid_id
    int team
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
    int tier

cdef struct Reward:
    float death
    float xp
    float distance
    float tower

cpdef entity_dtype():
    '''Make a dummy entity to get the dtype'''
    cdef Entity entity
    return np.asarray(<Entity[:1]>&entity).dtype

cpdef reward_dtype():
    '''Make a dummy reward to get the dtype'''
    cdef Reward reward
    return np.asarray(<Reward[:1]>&reward).dtype

def step_all(list envs):
    cdef:
        int n = len(envs)
        int i

    for i in range(n):
        (<Environment>PyList_GET_ITEM(envs, i)).step()
  
@cython.profile(False)
cdef int xp_for_player_kill(Entity* player):
    return 100 + <int>(player.xp / 7.69)
 
@cython.profile(False)
cdef float clip(float x):
    # Clip to [-1, 1]
    return max(-1, min(x, 1))

@cython.profile(False)
cdef inline float l1_distance(float x1, float y1, float x2, float y2):
    return abs(x1 - x2) + abs(y1 - y2)

cdef class Environment:
    cdef:
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

        unsigned char[:, :] grid
        unsigned char[:, :] orig_grid
        unsigned char[:, :, :, :] ai_paths
        int[:, :] atn_map
        unsigned char[:, :, :, :] observations_map
        unsigned char[:, :] observations_extra
        int[:] xp_for_level
        int[:] actions
        int[:, :] pid_map
        Entity[:, :] player_obs
        Entity[:] entities
        Reward[:] rewards
        float[:] sum_rewards
        float[:] norm_rewards
        float[:, :, :] waypoints 

        float reward_death
        float reward_xp
        float reward_distance
        float reward_tower
        
        public int total_towers_taken

        # MAX_ENTITIES x MAX_SCANNED_TARGETS
        Entity* scanned_targets[256][121]
        skill skills[10][3]

        int rng_n
        int rng_idx
        float[:] rng

    def __init__(self, cnp.ndarray grid, cnp.ndarray ai_paths,
            cnp.ndarray pids, cnp.ndarray entities, dict entity_data,
            cnp.ndarray player_obs, cnp.ndarray observations_map, cnp.ndarray observations_extra,
            cnp.ndarray rewards, cnp.ndarray sum_rewards, cnp.ndarray norm_rewards, cnp.ndarray actions,
            int num_agents, int num_creeps, int num_neutrals,
            int num_towers, int vision_range, float agent_speed, bint discretize, float reward_death,
            float reward_xp, float reward_distance, float reward_tower):

        self.num_agents = num_agents
        self.num_creeps = num_creeps
        self.num_neutrals = num_neutrals
        self.num_towers = num_towers
        self.vision_range = vision_range
        self.agent_speed = agent_speed
        self.discretize = discretize
        self.obs_size = 2*vision_range + 1
        self.creep_idx = 0

        self.reward_death = reward_death
        self.reward_xp = reward_xp
        self.reward_distance = reward_distance
        self.reward_tower = reward_tower
        self.sum_rewards = sum_rewards
        self.norm_rewards = norm_rewards
        self.total_towers_taken = 0

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
        self.player_obs = player_obs

        # Preallocate RNG -1 to 1
        self.rng_n = 10000
        self.rng_idx = 0
        self.rng = 2*np.random.rand(self.rng_n).astype(np.float32) - 1

        self.xp_for_level = np.array([
            0, 240, 640, 1160, 1760, 2440, 3200, 4000, 4900, 4900, 7000, 8200,
            9500, 10900, 12400, 14000, 15700, 17500, 19400, 21400, 23600, 26000,
            28600, 31400, 34400, 38400, 43400, 49400, 56400, 63900], dtype=np.int32)
        self.atn_map = np.array([
            [1, -1, 0, 0, 1, -1, -1, 1],
            [0, 0, 1, -1, -1, -1, 1, 1]], dtype=np.int32)

        # Initialize Players
        cdef Entity *player
        for team in range(2):
            if team == 0:
                spawn_y = 128 - 15
                spawn_x = 12
            else:
                spawn_y = 15
                spawn_x = 128 - 12

            for pid in range(team*5, team*5 + 5):
                player = self.get_entity(pid)
                player.pid = pid
                player.entity_type = ENTITY_PLAYER
                player.team = team
                player.spawn_y = spawn_y
                player.spawn_x = spawn_x
                player.move_speed = self.agent_speed
                player.max_health = 500
                player.max_mana = 100
                player.basic_attack_cd = 8
                player.damage = 50

            pid = 5*team
            player = self.get_entity(pid)
            player.pid = pid
            player.entity_type = ENTITY_PLAYER
            player.grid_id = RADIANT_SUPPORT + team*5
            player.hero_type = 0
            player.lane = 2 + 3*team
            self.skills[pid][0] = self.skill_support_hook
            self.skills[pid][1] = self.skill_support_aoe_heal
            self.skills[pid][2] = self.skill_support_stun

            pid = 5*team + 1
            player = self.get_entity(pid)
            player.pid = pid
            player.entity_type = ENTITY_PLAYER
            player.grid_id = RADIANT_ASSASSIN + team*5
            player.hero_type = 1
            player.lane = 2 + 3*team
            self.skills[pid][0] = self.skill_assassin_aoe_minions
            self.skills[pid][1] = self.skill_assassin_tp_damage
            self.skills[pid][2] = self.skill_assassin_move_buff

            pid = 5*team + 2
            player = self.get_entity(pid)
            player.pid = pid
            player.entity_type = ENTITY_PLAYER
            player.grid_id = RADIANT_BURST + team*5
            player.hero_type = 2
            player.lane = 1 + 3*team
            self.skills[pid][0] = self.skill_burst_nuke
            self.skills[pid][1] = self.skill_burst_aoe
            self.skills[pid][2] = self.skill_burst_aoe_stun

            pid = 5*team + 3
            player = self.get_entity(pid)
            player.pid = pid
            player.entity_type = ENTITY_PLAYER
            player.grid_id = RADIANT_TANK + team*5
            player.hero_type = 3
            player.lane = 3*team
            self.skills[pid][0] = self.skill_tank_aoe_dot
            self.skills[pid][1] = self.skill_tank_self_heal
            self.skills[pid][2] = self.skill_tank_engage_aoe

            pid = 5*team + 4
            player = self.get_entity(pid)
            player.pid = pid
            player.entity_type = ENTITY_PLAYER
            player.grid_id = RADIANT_CARRY + team*5
            player.hero_type = 4
            player.lane = 2 + 3*team
            self.skills[pid][0] = self.skill_carry_retreat_slow
            self.skills[pid][1] = self.skill_carry_slow_damage
            self.skills[pid][2] = self.skill_carry_aoe

        # Load creep waypoints for each lane
        self.waypoints = np.zeros((6, 20, 2), dtype=np.float32)
        for lane in range(6):
            lane_data = entity_data['waypoints'][lane]
            self.waypoints[lane, 0, 0] = lane_data['spawn_y']
            self.waypoints[lane, 0, 1] = lane_data['spawn_x']
            waypoints = lane_data['waypoints']
            for i, waypoint in enumerate(waypoints):
                self.waypoints[lane, i+1, 0] = waypoint['y']
                self.waypoints[lane, i+1, 1] = waypoint['x']

        cdef Entity *tower
        for idx, tower_data in enumerate(entity_data['towers']):
            tower = self.get_tower(idx)
            tower.pid = idx + self.num_agents + self.num_creeps + self.num_neutrals
            tower.entity_type = ENTITY_TOWER
            tower.grid_id = TOWER
            tower.basic_attack_cd = 5
            tower.team = tower_data['team']
            tower.spawn_y = tower_data['y']
            tower.spawn_x = tower_data['x']
            tower.max_health = tower_data['health']
            tower.damage = tower_data['damage']
            tower.tier = tower_data['tier']
            tower.xp_on_kill = 800 * tower.tier

        # Load neutral data
        idx = 0
        cdef Entity* neutral
        for camp, camp_data in enumerate(entity_data['neutrals']):
            for i in range(4): # Neutrals per camp
                neutral = self.get_neutral(idx)
                neutral.pid = idx + self.num_agents + self.num_creeps
                neutral.entity_type = ENTITY_NEUTRAL
                neutral.grid_id = NEUTRAL
                neutral.max_health = 500
                neutral.team = 2
                neutral.spawn_y = camp_data['y']
                neutral.spawn_x = camp_data['x']
                neutral.xp_on_kill = 35
                neutral.basic_attack_cd = 5
                neutral.damage = 22
                idx += 1

    cdef void compute_observations(self):
        cdef:
            int x, y, r, c, dx, dy, xx, yy, idx, pid, target_pid
            Entity *player, *target
            Reward *reward

        self.observations_map[:] = 0

        # Probably safe to not clear this
        self.observations_extra[:] = 0

        for pid in range(self.num_agents):
            player = self.get_entity(pid)
            reward = self.get_reward(pid)

            y = int(player.y)
            x = int(player.x)

            self.observations_map[pid, :, :, 0] = self.grid[
                y-self.vision_range:y+self.vision_range+1,
                x-self.vision_range:x+self.vision_range+1
            ]

            # TODO: Add bounds debug checks asserts
            self.observations_extra[pid, 0] = <unsigned char> (2*player.x)
            self.observations_extra[pid, 1] = <unsigned char> (2*player.y)
            self.observations_extra[pid, 2] = <unsigned char> (255*player.level/30.0)
            self.observations_extra[pid, 3] = <unsigned char> (255*player.health/player.max_health)
            self.observations_extra[pid, 4] = <unsigned char> (255*player.mana/player.max_mana)
            self.observations_extra[pid, 5] = <unsigned char> player.damage
            self.observations_extra[pid, 6] = <unsigned char> (100*player.move_speed)
            self.observations_extra[pid, 7] = <unsigned char> (player.move_modifier*100)
            self.observations_extra[pid, 8] = <unsigned char> (2*player.stun_timer)
            self.observations_extra[pid, 9] = <unsigned char> (2*player.move_timer)
            self.observations_extra[pid, 10] = <unsigned char> (2*player.q_timer)
            self.observations_extra[pid, 11] = <unsigned char> (2*player.w_timer)
            self.observations_extra[pid, 12] = <unsigned char> (2*player.e_timer)
            self.observations_extra[pid, 13] = <unsigned char> (50*player.basic_attack_timer)
            self.observations_extra[pid, 14] = <unsigned char> (50*player.basic_attack_cd)
            self.observations_extra[pid, 15] = <unsigned char> (255*player.is_hit)
            self.observations_extra[pid, 16] = <unsigned char> (255*player.team)
            self.observations_extra[pid, 17+player.hero_type] = 255

            # Assumes scaled between -1 and 1, else overflows
            self.observations_extra[pid, 22] = <unsigned char> (127*reward.death + 128)
            self.observations_extra[pid, 23] = <unsigned char> (25*reward.xp)
            self.observations_extra[pid, 24] = <unsigned char> (127*reward.distance + 128)
            self.observations_extra[pid, 25] = <unsigned char> (70*reward.tower)

            for dy in range(-self.vision_range, self.vision_range+1):
                for dx in range(-self.vision_range, self.vision_range+1):
                    xx = x + dx
                    yy = y + dy
                    target_pid = self.pid_map[yy, xx]
                    if target_pid == -1:
                        continue

                    target = self.get_entity(target_pid)
                    xx = dx + self.vision_range
                    yy = dy + self.vision_range

                    self.observations_map[pid, xx, yy, 1] = <unsigned char> (255*target.health/target.max_health)
                    self.observations_map[pid, xx, yy, 2] = <unsigned char> (255*target.mana/target.max_mana)
                    self.observations_map[pid, xx, yy, 3] = <unsigned char> (target.level/30.0)

    @cython.profile(False)
    cdef int level(self, int xp):
        cdef int i
        for i in range(30):
            if xp < self.xp_for_level[i]:
                return i + 1

        return i+1
     
    @cython.profile(False)
    cdef Reward* get_reward(self, int pid):
        return &self.rewards[pid]

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

        self.grid[disc_y, disc_x] = EMPTY
        self.grid[disc_dest_y, disc_dest_x] = player.grid_id

        self.pid_map[disc_y, disc_x] = -1
        self.pid_map[disc_dest_y, disc_dest_x] = player.pid

        player.y = dest_y
        player.x = dest_x
        return True

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

        if atn < 8:
            dy = self.atn_map[0, atn]
            dx = self.atn_map[1, atn]

        move_to_y = int(entity.y + dy*speed)
        move_to_x = int(entity.x + dx*speed)
        if (self.grid[move_to_y, move_to_x] != EMPTY and
                self.pid_map[move_to_y, move_to_x] != entity.pid):
            jitter_x = self.rng[self.rng_idx]
            jitter_y = self.rng[self.rng_idx + 1]

            self.rng_idx += 2
            if self.rng_idx >= self.rng_n - 1:
                self.rng_idx = 0

            return self.move_to(entity,
                entity.y + jitter_y, entity.x + jitter_x)

        return self.move_to(entity,
            entity.y + dy*speed,
            entity.x + dx*speed)

    @cython.profile(False)
    cdef bint move_near(self, Entity* entity, Entity* target):
        cdef int dy, dx
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if self.move_to(entity, target.y+dy, target.x+dx):
                    return True

        return False

    @cython.profile(False)
    cdef bint attack(self, Entity* player, Entity* target, float damage):
        cdef:
            int xp = 0
            Reward* reward

        if target.pid == -1:
            return False

        if target.team == player.team:
            return False

        target.is_hit = 1
        target.health -= damage
        if target.health > 0:
            return True

        if target.entity_type == ENTITY_PLAYER:
            reward = self.get_reward(target.pid)
            reward.death = self.reward_death
            self.respawn_player(target)
        elif (target.entity_type == ENTITY_TOWER or target.entity_type == ENTITY_CREEP
                or target.entity_type == ENTITY_NEUTRAL):
            self.kill(target)

        if player.entity_type != ENTITY_PLAYER:
            return True

        if target.entity_type == ENTITY_PLAYER:
            xp = xp_for_player_kill(target)
        else:
            xp = target.xp_on_kill

        if player.xp < 10000000:
            player.xp += xp

        reward = self.get_reward(player.pid)
        reward.xp = self.reward_xp * xp
        if target.entity_type == ENTITY_TOWER:
            self.total_towers_taken += 1
            reward.tower = self.reward_tower

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

    @cython.profile(False)
    cdef bint heal(self, Entity* player, Entity* target, float amount):
        if target.pid == -1:
            return False

        if target.team != player.team:
            return False

        if target.entity_type != ENTITY_PLAYER:
            return False

        target.health += amount
        if target.health > target.max_health:
            target.health = target.max_health

        return True

    @cython.profile(False)
    cdef void kill(self, Entity* entity):
        cdef:
            int y = int(entity.y)
            int x = int(entity.x)

        self.grid[y, x] = EMPTY
        self.pid_map[y, x] = -1
        entity.pid = -1

    @cython.profile(False)
    cdef void respawn_player(self, Entity* entity):
        cdef:
            bint valid_pos = False
            int y, x

        entity.health = entity.max_health
        entity.mana = entity.max_mana
        entity.basic_attack_timer = 0
        entity.move_modifier = 0
        entity.move_timer = 0
        entity.stun_timer = 0

        while not valid_pos:
            y = <int>entity.spawn_y + rand()%15 - 7
            x = <int>entity.spawn_x + rand()%15 - 7
            if self.grid[y, x] == EMPTY:
                valid_pos = True
                break

        self.move_to(entity, y, x)

    @cython.profile(False)
    cdef void spawn_creep(self, int idx, int lane):
        cdef:
            int pid = idx + self.num_agents
            Entity* creep = self.get_entity(pid)
            int team 

        if lane < 3:
            creep.team = 0
            creep.grid_id = RADIANT_CREEP
        else:
            creep.team = 1
            creep.grid_id = DIRE_CREEP

        creep.pid = pid
        creep.entity_type = ENTITY_CREEP
        creep.health = 450
        creep.max_health = 450
        creep.lane = lane
        creep.waypoint = 0
        creep.xp_on_kill = 60
        creep.damage = 22
        creep.basic_attack_cd = 5

        self.respawn_creep(creep, lane)

    @cython.profile(False)
    cdef bint respawn_creep(self, Entity* entity, int lane):
        cdef:
            bint valid_pos = False
            int spawn_y = int(self.waypoints[lane, 0, 0])
            int spawn_x = int(self.waypoints[lane, 0, 1])
            int x, y

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

    @cython.profile(False)
    cdef void respawn_neutral(self, int idx, int camp):
        cdef:
            Entity* neutral = self.get_neutral(idx)
            int dy, dx

        neutral.health = neutral.max_health
        neutral.basic_attack_timer = 0

        for i in range(10):
            dy = rand() % 7 - 3
            dx = rand() % 7 - 3
            if self.move_to(neutral, neutral.spawn_y + dy, neutral.spawn_x + dx):
                break


    @cython.profile(False)
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

            dist = abs(creep.y - dest_y) + abs(creep.x - dest_x)
            if dist < 2:
                self.basic_attack(creep, target)
            self.move_towards(creep, self.agent_speed, int(dest_y), int(dest_x))
        else:
            dest_y = self.waypoints[lane, waypoint, 0]
            dest_x = self.waypoints[lane, waypoint, 1]
            self.move_towards(creep, self.agent_speed, int(dest_y), int(dest_x))
            # TODO: Last waypoint?
            if abs(creep.y - dest_y) + abs(creep.x - dest_x) < 2 and self.waypoints[lane, waypoint+1, 0] != 0:
                creep.waypoint += 1

    @cython.profile(False)
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
            if abs(neutral.y - target.y) + abs(neutral.x - target.x) < 2:
                self.basic_attack(neutral, target)
            else:
                self.move_towards(neutral, self.agent_speed, int(target.y), int(target.x))
        elif abs(neutral.y - neutral.spawn_y) + abs(neutral.x - neutral.spawn_x) > 2:
            self.move_towards(neutral, self.agent_speed, int(neutral.spawn_y), int(neutral.spawn_x))

    @cython.profile(False)
    cdef bint scan_aoe(self, Entity* player, int radius,
            bint exclude_friendly, bint exclude_hostile, bint exclude_creeps,
            bint exclude_neutrals, bint exclude_towers):
        cdef:
            int player_y = int(player.y)
            int player_x = int(player.x)
            int player_team = player.team
            int pid = player.pid, idx = 0
            int target_team, target_type, y, x
            Entity* target

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

                target_type = target.entity_type
                if exclude_neutrals and target_type == ENTITY_NEUTRAL:
                    continue

                if exclude_creeps and target_type == ENTITY_CREEP:
                    continue

                if exclude_towers and target_type == ENTITY_TOWER:
                    continue

                self.scanned_targets[pid][idx] = target
                idx += 1

        self.scanned_targets[pid][idx] = NULL

        if idx == 0:
            return False

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

            dist = abs(player_y - target.y) + abs(player_x - target.x)
            if dist < nearest_dist:
                nearest_target = target
                nearest_dist = dist

        return nearest_target

    @cython.profile(False)
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

    @cython.profile(False)
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
    cdef bint aoe_push(self, Entity* player, int radius, float amount):
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

    @cython.profile(False)
    cdef bint push(self, Entity* player, Entity* target, float amount):
        cdef:
            float dist = abs(target.x - player.x) + abs(target.y - player.y)
            float dx = target.x - player.x
            float dy = target.y - player.y

        if dist == 0.0:
            return False

        # Norm to unit vector
        dx = amount * dx / dist
        dy = amount * dy / dist

        return self.move_to(target, target.y + dy, target.x + dx)

    @cython.profile(False)
    cdef bint pull(self, Entity* player, Entity* target, float amount):
        return self.push(target, player, -amount)

    @cython.profile(False)
    cdef void skill_support_hook(self, Entity* player, Entity* target):
        if player.mana < 15:
            return

        self.pull(target, player, 1.0)
        player.mana -= 15

    @cython.profile(False)
    cdef void skill_support_aoe_heal(self, Entity* player, Entity* target):
        if player.mana < 40:
            return

        if self.player_aoe_attack(player, player, 4, -200, 0):
            player.mana -= 40
            player.w_timer = 60

    @cython.profile(False)
    cdef void skill_support_stun(self, Entity* player, Entity* target):
        if player.mana < 60:
            return

        if self.attack(player, target, 50):
            target.stun_timer = 60
            player.mana -= 60
            player.e_timer = 50

    @cython.profile(False)
    cdef void skill_burst_nuke(self, Entity* player, Entity* target):
        if player.mana < 60:
            return

        if self.attack(player, target, 500):
            player.mana -= 60
            player.q_timer = 70

    @cython.profile(False)
    cdef void skill_burst_aoe(self, Entity* player, Entity* target):
        if player.mana < 40:
            return

        if self.player_aoe_attack(player, target, 2, 200, 0):
            player.mana -= 40
            player.w_timer = 40

    @cython.profile(False)
    cdef void skill_burst_aoe_stun(self, Entity* player, Entity* target):
        if player.mana < 60:
            return

        if self.player_aoe_attack(player, target, 2, 0, 40):
            player.mana -= 60
            player.e_timer = 80

    @cython.profile(False)
    cdef void skill_tank_aoe_dot(self, Entity* player, Entity* target):
        if player.mana < 5:
            return

        if self.player_aoe_attack(player, player, 2, 20, 0):
            player.mana -= 5

    @cython.profile(False)
    cdef void skill_tank_self_heal(self, Entity* player, Entity* target):
        if player.mana < 30:
            return

        if self.heal(player, player, 250):
            player.mana -= 30
            player.w_timer = 60

    @cython.profile(False)
    cdef void skill_tank_engage_aoe(self, Entity* player, Entity* target):
        if player.mana < 60:
            return

        if self.move_near(player, target):
            player.mana -= 60
            player.e_timer = 70
            self.aoe_push(player, 4, 3.0)

    @cython.profile(False)
    cdef void skill_carry_retreat_slow(self, Entity* player, Entity* target):
        for i in range(3):
            if player.mana < 20:
                return

            if self.push(target, player, 1.5):
                target.move_timer = 15
                target.move_modifier = 0.5
                player.mana -= 20
                player.w_timer = 40

    @cython.profile(False)
    cdef void skill_carry_slow_damage(self, Entity* player, Entity* target):
        if player.mana < 40:
            return

        if self.attack(player, target, 100):
            target.move_timer = 60
            target.move_modifier = 0.5
            player.mana -= 40
            player.w_timer = 40

    @cython.profile(False)
    cdef void skill_carry_aoe(self, Entity* player, Entity* target):
        if player.mana < 40:
            return

        if self.player_aoe_attack(player, target, 2, 200, 0):
            player.mana -= 40
            player.e_timer = 40

    @cython.profile(False)
    cdef void skill_assassin_aoe_minions(self, Entity* player, Entity* target):
        if player.mana < 40:
            return

        # Targeted on minions, splashes to players
        if (target.entity_type == ENTITY_CREEP or target.entity_type == ENTITY_NEUTRAL
                ) and self.player_aoe_attack(player, target, 3, 300, 0):
            player.mana -= 40
            player.q_timer = 40

    @cython.profile(False)
    cdef void skill_assassin_tp_damage(self, Entity* player, Entity* target):
        if player.mana < 60:
            return

        if self.move_near(player, target) == -1:
            return

        player.mana -= 60
        if self.attack(player, target, 600):
            player.w_timer = 60

    @cython.profile(False)
    cdef void skill_assassin_move_buff(self, Entity* player, Entity* target):
        if player.mana < 5:
            return

        player.move_modifier = 2.0
        player.move_timer = 1
        player.mana -= 5

    cpdef reset(self, seed=0):
        cdef:
            Entity* player
            Entity* tower
            int pid
            int y
            int x
            int idx

        self.grid[:] = self.orig_grid
        self.pid_map[:] = -1
        self.tick = 0

        # Respawn towers
        for idx in range(self.num_towers):
            tower = self.get_tower(idx)
            tower.health = tower.max_health
            tower.basic_attack_timer = 0
            self.move_to(tower, tower.spawn_y, tower.spawn_x)

        # Respawn agents
        for i in range(self.num_agents):
            player = self.get_entity(i)
            player.xp = 0
            player.level = 1
            self.respawn_player(player)

        # TODO: Respawn creeps?
        #for i in range(self.num_creeps):
        #    self.spawn_creep(i, 0)

        self.compute_observations()

    cdef void step_neutrals(self):
        cdef int camp, neut, idx
        cdef Entity* neutral

        if self.tick % 600 == 0:
            for camp in range(18):
                for neut in range(4):
                    self.respawn_neutral(4*camp + neut, camp)

        for idx in range(self.num_neutrals):
            neutral = self.get_neutral(idx)
            if neutral.pid == -1:
                continue

            self.update_status(neutral)
            self.update_cooldowns(neutral)
            if neutral.stun_timer > 0:
                continue

            self.neutral_ai(neutral)

    cdef void step_creeps(self):
        cdef int idx, lane
        cdef Entity* creep
        # Spawn creep wave
        if self.tick % 150 == 0:
            for lane in range(6):
                for _ in range(5):
                    self.spawn_creep(self.creep_idx, lane)
                    self.creep_idx = (self.creep_idx + 1) % self.num_creeps

        # Creep AI
        for idx in range(self.num_creeps):
            creep = self.get_creep(idx)
            if creep.pid == -1:
                continue

            self.update_status(creep)
            self.update_cooldowns(creep)
            if creep.stun_timer > 0:
                continue

            self.creep_ai(creep)

    cdef void step_towers(self):
        cdef int tower_idx
        cdef Entity* tower

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

    cdef void step_players(self):
        cdef:
            float[:] actions_continuous
            int[:] actions_discrete
            float y, x, dy, dx, vel_y, vel_x
            int disc_y, disc_x, disc_dest_y, disc_dest_x
            int agent_idx, attack, pid, target_pid, damage, atn, lane
            float prev_dist_to_ancient, dist_to_ancient
            bint use_q, use_w, use_e, use_basic_attack
            Entity *player, *target, *creep, *neutral, *tower
            Reward* reward

        # Clear rewards
        self.sum_rewards[:] = 0
        for pid in range(self.num_agents):
            player = self.get_entity(pid)
            reward = self.get_reward(pid)
            reward.death = 0
            reward.xp = 0
            reward.distance = 0
            reward.tower = 0

        #if self.discretize:
        #    actions_discrete = self.actions
        #else:
        #    actions_continuous = self.actions
        actions_discrete = self.actions

        for pid in range(self.num_agents):
            player = self.get_entity(pid)
            if rand() % 1024 == 0:
                self.respawn_player(player)

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

            if target != NULL:
                if use_q:
                    self.skills[pid][0](self, player, target)
                elif use_w:
                    self.skills[pid][1](self, player, target)
                elif use_e:
                    self.skills[pid][2](self, player, target)
                else:
                    self.basic_attack(player, target)

            # Reward based on distance to enemy ancient
            if player.team == 0:
                ancient = self.get_tower(22)
            else:
                ancient = self.get_tower(23)
 
            dest_y = player.y + player.move_modifier*self.agent_speed*vel_y
            dest_x = player.x + player.move_modifier*self.agent_speed*vel_x

            prev_dist_to_ancient = abs(player.y - ancient.y) + abs(player.x - ancient.x)
            self.move_to(player, dest_y, dest_x)
            dist_to_ancient = abs(player.y - ancient.y) + abs(player.x - ancient.x)
            reward = self.get_reward(pid)
            reward.distance = self.reward_distance * (prev_dist_to_ancient - dist_to_ancient)
            self.sum_rewards[pid] = reward.death + reward.xp + reward.distance + reward.tower
            self.norm_rewards[pid] = (reward.death/self.reward_death + reward.xp/self.reward_xp
                + reward.distance/self.reward_distance + reward.tower/self.reward_tower)

    cdef int step(self):
        cdef int pid
        cdef Entity* player

        for pid in range(self.num_agents + self.num_towers + self.num_creeps):
            player = self.get_entity(pid)
            player.is_hit = 0

        self.step_neutrals()
        self.step_creeps()
        self.step_towers()
        self.step_players()

        self.tick += 1
        self.compute_observations()

        if self.get_tower(22).health <= 0:
            self.reset(0)
            return 1 
        if self.get_tower(23).health <= 0:
            self.reset(0)
            return 2

        return 0

