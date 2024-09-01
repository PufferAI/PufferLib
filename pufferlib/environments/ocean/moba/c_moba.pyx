# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: profile=True

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrtf
from cpython.list cimport PyList_GET_ITEM
cimport cython
cimport numpy as cnp
import numpy as np

ctypedef bint(*skill)(Environment, Entity*, Entity*)

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

    int XP_RANGE = 7
    int MAX_USES = 2_000_000_000

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
    int q_uses
    int w_uses
    int e_uses
    int basic_attack_uses
    int basic_attack_timer
    int basic_attack_cd
    int is_hit
    int level
    int xp
    int xp_on_kill
    float reward
    int tier
    float base_health
    float base_mana
    float base_damage
    int hp_gain_per_level
    int mana_gain_per_level
    int damage_gain_per_level
    float damage_dealt
    float damage_received
    float healing_dealt
    float healing_received
    int deaths
    int heros_killed
    int creeps_killed
    int neutrals_killed
    int towers_killed
    float last_x
    float last_y
    int target_pid
    int attack_aoe

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
        int[:, :] actions
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
        public int total_levels_gained
        public int radiant_victories
        public int dire_victories

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
        self.total_levels_gained = 0
        self.radiant_victories = 0
        self.dire_victories = 0

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
                player.basic_attack_cd = 8
                player.base_damage = 50
                player.q_uses = 0
                player.w_uses = 0
                player.e_uses = 0
                player.basic_attack_uses = 0
                player.damage_dealt = 0
                player.damage_received = 0
                player.healing_dealt = 0
                player.healing_received = 0
                player.deaths = 0
                player.heros_killed = 0
                player.creeps_killed = 0
                player.neutrals_killed = 0
                player.towers_killed = 0
                player.last_x = 0
                player.last_y = 0

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
            player.base_health = 500
            player.base_mana = 250
            player.hp_gain_per_level = 100
            player.mana_gain_per_level = 50
            player.damage_gain_per_level = 10

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
            player.base_health = 400
            player.base_mana = 300
            player.hp_gain_per_level = 100
            player.mana_gain_per_level = 65
            player.damage_gain_per_level = 10

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
            player.base_health = 400
            player.base_mana = 300
            player.hp_gain_per_level = 75
            player.mana_gain_per_level = 90
            player.damage_gain_per_level = 10

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
            player.base_health = 700
            player.base_mana = 200
            player.hp_gain_per_level = 150
            player.mana_gain_per_level = 50
            player.damage_gain_per_level = 15

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
            player.base_health = 300
            player.base_mana = 250
            player.hp_gain_per_level = 50
            player.mana_gain_per_level = 50
            player.damage_gain_per_level = 25


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
            '''
            except:
                print(f'Invalid player position: {y}, {x}, pid: {pid}')
                print(f'type: {player.entity_type}, team: {player.team}, hero type: {player.hero_type}')
                print(f'health: {player.health}, max health: {player.max_health}')
                print(f'mana: {player.mana}, max mana: {player.max_mana}')
                print(f'level: {player.level}')
                print(f'x: {player.x}, y: {player.y}')
                print(f'spawn x: {player.spawn_x}, spawn y: {player.spawn_y}')
                print(f'damage: {player.damage}')
                print(f'lane: {player.lane}')
                print(f'waypoint: {player.waypoint}')
                print(f'move speed: {player.move_speed}')
                print(f'move modifier: {player.move_modifier}')
                print(f'stun timer: {player.stun_timer}')
                print(f'move timer: {player.move_timer}')
                print(f'q timer: {player.q_timer}')
                print(f'w timer: {player.w_timer}')
                print(f'e timer: {player.e_timer}')
                print(f'basic attack timer: {player.basic_attack_timer}')
                print(f'is hit: {player.is_hit}')
                print(f'game tick: {self.tick}')
                exit(0)
            '''


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
    cdef bint spawn_at(self, Entity* entity, float y, float x):
        cdef:
            int disc_y = int(y)
            int disc_x = int(x)

        if self.grid[disc_y, disc_x] != EMPTY:
            return False

        self.grid[disc_y, disc_x] = entity.grid_id
        self.pid_map[disc_y, disc_x] = entity.pid
        entity.y = y
        entity.x = x
        return True

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

        '''
        if self.grid[disc_y, disc_x] == TOWER:
            print(f'Player type {player.entity_type} at {disc_y}, {disc_x} is a tower?')
            print(f'PID: {player.pid}, moving to {disc_dest_y}, {disc_dest_x}')
            print(f'hero type: {player.hero_type}')
            print(f'team: {player.team}')
            print(f'Tick: {self.tick}')
            exit(0)
            return False

        if self.grid[disc_dest_y, disc_dest_x] == TOWER:
            print(f'Cannot move to tower at {disc_dest_y}, {disc_dest_x}')
            return False
        '''

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
        '''
        if target.x == 0 and target.y == 0:
            print(f'MOVE NEAR Invalid target at {target.y}, {target.x}')
            print(f'entity x: {entity.x}, y: {entity.y}')
            print(f'target x: {target.x}, y: {target.y}')
            print(f'target pid: {target.pid}')
            print(f'entity pid: {entity.pid}')
            print(f'Tick: {self.tick}')
            exit(0)
        '''

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
            int level
            Reward* reward

        if target.pid == -1:
            return False

        if target.team == player.team:
            return False

        player.target_pid = target.pid
        cdef float current_health = target.health
        target.is_hit = 1
        target.health -= damage
        if target.health > 0:
            player.damage_dealt += damage
            target.damage_received += damage
            return True

        player.damage_dealt += current_health
        target.damage_received += current_health

        cdef float target_x = target.x
        cdef float target_y = target.y

        if target.entity_type == ENTITY_PLAYER:
            reward = self.get_reward(target.pid)
            reward.death = self.reward_death
            player.heros_killed += 1
            target.deaths += 1
            self.respawn_player(target)
        elif target.entity_type == ENTITY_CREEP:
            player.creeps_killed += 1
            self.kill(target)
        elif target.entity_type == ENTITY_NEUTRAL:
            player.neutrals_killed += 1
            self.kill(target)
        elif target.entity_type == ENTITY_TOWER:
            player.towers_killed += 1
            self.kill(target)

        if player.entity_type != ENTITY_PLAYER:
            return True

        if target.entity_type == ENTITY_PLAYER:
            xp = xp_for_player_kill(target)
        else:
            xp = target.xp_on_kill

        # Share xp with allies in range
        cdef int first_player_on_team = 0
        if player.team == 1:
            first_player_on_team = 5

        cdef bint[5] in_range = [False, False, False, False, False]
        cdef int num_in_range = 0
        cdef Entity* ally
        cdef int i
        for i in range(5):
            ally = self.get_entity(first_player_on_team + i)
            if ally.pid == player.pid:
                in_range[i] = True
                num_in_range += 1
                continue

            if max(abs(ally.y - target_y), abs(ally.x - target_x)) <= XP_RANGE:
                in_range[i] = True
                num_in_range += 1

        xp = xp / num_in_range

        for i in range(5):
            if not in_range[i]:
                continue

            ally = self.get_entity(first_player_on_team + i)
            if ally.xp > 10000000:
                continue

            ally.xp += xp

            reward = self.get_reward(first_player_on_team + i)
            reward.xp = self.reward_xp * xp

            level = ally.level
            ally.level = self.level(ally.xp)
            if ally.level > level:
                self.total_levels_gained += 1

            ally.max_health = ally.base_health + ally.level*ally.hp_gain_per_level
            ally.max_mana = ally.base_mana + ally.level*ally.mana_gain_per_level
            ally.damage = ally.base_damage + ally.level*ally.damage_gain_per_level

        reward = self.get_reward(player.pid)
        if target.entity_type == ENTITY_TOWER:
            self.total_towers_taken += 1
            reward.tower = self.reward_tower

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

        cdef float current_health = target.health
        target.health += amount
        if target.health > target.max_health:
            target.health = target.max_health
            player.healing_dealt += target.max_health - current_health
            target.healing_received += target.max_health - current_health
        else:
            player.healing_dealt += amount
            target.healing_received += amount

        return True

    @cython.profile(False)
    cdef void kill(self, Entity* entity):
        cdef:
            int y = int(entity.y)
            int x = int(entity.x)

        self.grid[y, x] = EMPTY
        self.pid_map[y, x] = -1
        entity.pid = -1
        entity.x = 0
        entity.y = 0

    @cython.profile(False)
    cdef void respawn_player(self, Entity* entity):
        cdef:
            bint valid_pos = False
            int pid = entity.pid
            int y, x

        self.kill(entity)
        entity.pid = pid
        entity.max_health = entity.base_health
        entity.max_mana = entity.base_mana
        entity.health = entity.max_health
        entity.mana = entity.max_mana
        entity.damage = entity.base_damage
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
        '''
        if not self.move_to(entity, y, x):
            print(f'Failed to move to {y}, {x}')
            print(f'entity type: {entity.entity_type}, team: {entity.team}, hero type: {entity.hero_type}')
            print(f'health: {entity.health}, max health: {entity.max_health}')
            print(f'mana: {entity.mana}, max mana: {entity.max_mana}')
            print(f'level: {entity.level}')
            print(f'x: {entity.x}, y: {entity.y}')
            print(f'spawn x: {entity.spawn_x}, spawn y: {entity.spawn_y}')
            print(f'damage: {entity.damage}')
            print(f'lane: {entity.lane}')
            print(f'waypoint: {entity.waypoint}')
            print(f'move speed: {entity.move_speed}')
            print(f'move modifier: {entity.move_modifier}')
            print(f'stun timer: {entity.stun_timer}')
            print(f'move timer: {entity.move_timer}')
            print(f'q timer: {entity.q_timer}')
            print(f'w timer: {entity.w_timer}')
            print(f'e timer: {entity.e_timer}')
            print(f'basic attack timer: {entity.basic_attack_timer}')
            print(f'is hit: {entity.is_hit}')
            print(f'Tick: {self.tick}')
            exit(0)
        '''

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
        #creep.x = 0
        #creep.y = 0

        self.respawn_creep(creep, lane)

    @cython.profile(False)
    cdef bint respawn_creep(self, Entity* entity, int lane):
        cdef:
            bint valid_pos = False
            int spawn_y = int(self.waypoints[lane, 0, 0])
            int spawn_x = int(self.waypoints[lane, 0, 1])
            int x, y, i

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
    cdef void respawn_neutral(self, int idx):
        cdef:
            Entity* neutral = self.get_neutral(idx)
            int spawn_y = int(neutral.spawn_y)
            int spawn_x = int(neutral.spawn_x)
            int y, x, i

        neutral.pid = idx + self.num_agents + self.num_creeps
        neutral.health = neutral.max_health
        neutral.basic_attack_timer = 0
        # TODO: Can this leak over games somehow?
        #neutral.x = 0
        #neutral.y = 0

        # TODO: Clean up spawn regions. Some might be offset and are obscured.
        # Maybe check all valid spawn spots?
        for i in range(100):
            y = spawn_y + rand() % 7 - 3
            x = spawn_x + rand() % 7 - 3
            if self.grid[y, x] == EMPTY:
                break

        if not self.move_to(neutral, y, x):
            neutral.pid = -1
            print(f'Failed to respawn neutral {idx} at spawn {spawn_y}, {spawn_x}. Grid: {self.grid[y, x]}')


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

                '''
                if (target.x == 0 and target.y == 0) or target.pid == -1:
                    print(f'Invalid target at {y}, {x}')
                    print(f'player x: {player.x}, y: {player.y}')
                    print(f'target x: {target.x}, y: {target.y}')
                    print(f'target pid: {target_pid}')
                    print(f'player pid: {player.pid}')
                    print(f'Tick: {self.tick}')
                    exit(0)
                '''

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
        cdef bint success, exclude_hostile, exclude_friendly
        if damage < 0:
            exclude_hostile = True
            exclude_friendly = False
        else:
            exclude_hostile = False
            exclude_friendly = True

        success = self.scan_aoe(player, radius, exclude_friendly=exclude_friendly,
            exclude_hostile=exclude_hostile, exclude_creeps=False,
            exclude_neutrals=False, exclude_towers=False)

        if not success:
            return False

        self.aoe_scanned(player, target, damage, stun)
        player.target_pid = target.pid
        player.attack_aoe = radius
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
        return False

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
    cdef bint skill_support_hook(self, Entity* player, Entity* target):
        if target == NULL or player.mana < 100:
            return False

        self.pull(target, player, 1.5 + 0.1*player.level)
        player.mana -= 100
        player.q_timer = 15
        return True

    @cython.profile(False)
    cdef bint skill_support_aoe_heal(self, Entity* player, Entity* target):
        if player.mana < 100:
            return False

        if self.player_aoe_attack(player, player, 5, -350 - 50*player.level, 0):
            player.mana -= 100 
            player.w_timer = 50
            return True

        return False

    @cython.profile(False)
    cdef bint skill_support_stun(self, Entity* player, Entity* target):
        if target == NULL or player.mana < 75:
            return False

        if self.attack(player, target, 50 + 20*player.level):
            target.stun_timer = 15 + (int)(0.5*player.level)
            player.mana -= 75
            player.e_timer = 60
            return True

        return False

    @cython.profile(False)
    cdef bint skill_burst_nuke(self, Entity* player, Entity* target):
        if target == NULL or player.mana < 200:
            return False

        if self.attack(player, target, 250 + 40*player.level):
            player.mana -= 200
            player.q_timer = 70
            return True
        
        return False

    @cython.profile(False)
    cdef bint skill_burst_aoe(self, Entity* player, Entity* target):
        if target == NULL or player.mana < 100:
            return False

        if self.player_aoe_attack(player, target, 2, 100 + 40*player.level, 0):
            player.mana -= 100
            player.w_timer = 40
            return True

        return False

    @cython.profile(False)
    cdef bint skill_burst_aoe_stun(self, Entity* player, Entity* target):
        if target == NULL or player.mana < 75:
            return False

        if self.player_aoe_attack(player, target, 2, 0, 10 + (int)(0.5*player.level)):
            player.mana -= 75
            player.e_timer = 50
            return True

        return False

    @cython.profile(False)
    cdef bint skill_tank_aoe_dot(self, Entity* player, Entity* target):
        if player.mana < 5:
            return False

        if self.player_aoe_attack(player, player, 2, 25 + 2.0*player.level, 0):
            player.mana -= 5
            return True

        return False

    @cython.profile(False)
    cdef bint skill_tank_self_heal(self, Entity* player, Entity* target):
        if player.mana < 100:
            return False

        if self.heal(player, player, 400 + 125*player.level):
            player.mana -= 100
            player.w_timer = 70
            return True

        return False

    @cython.profile(False)
    cdef bint skill_tank_engage_aoe(self, Entity* player, Entity* target):
        #return False # TODO: Fix teleport
        if target == NULL or player.mana < 50:
            return False

        if self.move_near(player, target):
            player.mana -= 50
            player.e_timer = 40
            self.aoe_push(player, 4, 2.0 + 0.1*player.level)
            return True

        return False

    @cython.profile(False)
    cdef bint skill_carry_retreat_slow(self, Entity* player, Entity* target):
        cdef int i
        cdef bint success = False
        for i in range(3):
            if target == NULL or player.mana < 25:
                return False

            if self.push(target, player, 1.0 + 0.05*player.level):
                target.move_timer = 15
                target.move_modifier = 0.5
                player.mana -= 25
                player.q_timer = 40
                success = True

        return success

    @cython.profile(False)
    cdef bint skill_carry_slow_damage(self, Entity* player, Entity* target):
        if target == NULL or player.mana < 150:
            return False

        if self.attack(player, target, 50 + 20*player.level):
            target.move_timer = 20 + player.level
            target.move_modifier = 0.5
            player.mana -= 150
            player.w_timer = 40
            return True

        return False

    @cython.profile(False)
    cdef bint skill_carry_aoe(self, Entity* player, Entity* target):
        if target == NULL or player.mana < 100:
            return False

        if self.player_aoe_attack(player, target, 2, 100 + 20*player.level, 0):
            player.mana -= 100
            player.e_timer = 40
            return True

        return False

    @cython.profile(False)
    cdef bint skill_assassin_aoe_minions(self, Entity* player, Entity* target):
        if target == NULL or player.mana < 100:
            return False

        # Targeted on minions, splashes to players
        if (target.entity_type == ENTITY_CREEP or target.entity_type == ENTITY_NEUTRAL
                ) and self.player_aoe_attack(player, target, 3, 100+20*player.level, 0):
            player.mana -= 100
            player.q_timer = 40
            return True

        return False

    @cython.profile(False)
    cdef bint skill_assassin_tp_damage(self, Entity* player, Entity* target):
        # TODO: Fix tp off map?
        #return False
        if target == NULL or player.mana < 150:
            return False

        if self.move_near(player, target) == -1:
            return False

        player.mana -= 150
        if self.attack(player, target, 250+50*player.level):
            player.w_timer = 60
            return True
        
        return False

    @cython.profile(False)
    cdef bint skill_assassin_move_buff(self, Entity* player, Entity* target):
        if player.mana < 100:
            return False

        player.move_modifier = 2.0
        player.move_timer = 25
        player.mana -= 100
        player.e_timer = 100
        return True

    def randomize_tower_hp(self):
        cdef int tower_idx
        cdef Entity* tower
        for tower_idx in range(self.num_towers):
            tower = self.get_tower(tower_idx)
            tower.health = rand() % tower.max_health + 1

    cpdef reset(self, seed=0):
        cdef:
            Entity* player
            Entity* tower
            Entity* creep
            Entity* neutral
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
            tower.target_pid = -1
            tower.pid = idx + self.num_agents + self.num_creeps + self.num_neutrals
            tower.health = tower.max_health
            tower.basic_attack_timer = 0
            tower.x = 0
            tower.y = 0
            self.grid[int(tower.spawn_y), int(tower.spawn_x)] = EMPTY
            self.move_to(tower, tower.spawn_y, tower.spawn_x)
            tower.last_x = tower.x
            tower.last_y = tower.y
            '''
            if not self.move_to(tower, tower.spawn_y, tower.spawn_x):
                print(f'Grid: {self.grid[int(tower.spawn_y), int(tower.spawn_x)]}')
                print(f'PID: {self.pid_map[int(tower.spawn_y), int(tower.spawn_x)]}')
                print(f'Failed to respawn tower {idx}')
            '''

        # Respawn agents
        for i in range(self.num_agents):
            player = self.get_entity(i)
            player.target_pid = -1
            player.xp = 0
            player.level = 1
            player.x = 0
            player.y = 0
            self.respawn_player(player)

        for i in range(self.num_creeps):
            creep = self.get_creep(i)
            creep.target_pid = -1
            creep.pid = -1
            creep.x = 0
            creep.y = 0

        for i in range(self.num_neutrals):
            neutral = self.get_neutral(i)
            neutral.target_pid = -1
            neutral.pid = -1
            neutral.x = 0
            neutral.y = 0

        self.compute_observations()

    cdef void step_neutrals(self):
        cdef int camp, neut, idx
        cdef Entity* neutral

        if self.tick % 600 == 0:
            for camp in range(18):
                for neut in range(4):
                    self.respawn_neutral(4*camp + neut)

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

        for pid in range(self.num_agents):
            player = self.get_entity(pid)
            if rand() % 1024 == 0:
                self.respawn_player(player)

            if player.mana < player.max_mana:
                player.mana += 2
            if player.mana > player.max_mana:
                player.mana = player.max_mana
            if player.health < player.max_health:
                player.health += 2
            if player.health > player.max_health:
                player.health = player.max_health

            self.update_status(player)
            self.update_cooldowns(player)

            if player.stun_timer > 0:
                continue

            # Attacks
            #vel_y = float(self.actions_discrete[pid, 0] - 3) / 3
            #vel_x = float(self.actions_discrete[pid, 1] - 3) / 3

            vel_y = float(self.actions[pid, 0]) / 100
            vel_x = float(self.actions[pid, 1]) / 100

            attack_target = int(self.actions[pid, 2])
            use_q = self.actions[pid, 3]
            use_w = self.actions[pid, 4]
            use_e = self.actions[pid, 5]

            '''
            if self.discretize:
                vel_y = float(self.actions_discrete[pid, 0] - 3) / 3
                vel_x = float(self.actions_discrete[pid, 1] - 3) / 3
                attack_target = int(self.actions_discrete[pid, 2])
                use_q = self.actions_discrete[pid, 3]
                use_w = self.actions_discrete[pid, 4]
                use_e = self.actions_discrete[pid, 5]
            else:
                pass
                #vel_y = actions_continuous[pid, 0]
                #vel_x = actions_continuous[pid, 1]
                #attack = int(actions_continuous[pid, 2])
                # TODO: Breaks to python
                #use_q = int(actions_continuous[pid, 3]) > 0.5
                #use_w = int(actions_continuous[pid, 4]) > 0.5
                #use_e = int(actions_continuous[pid, 5]) > 0.5
            '''

            #if attack_target == 0:
            #    pass
            # TODO: Restrict to 2 dimensions. Allowing no scan results
            # in old buffers... can also just null the buffer
            if attack_target == 1 or attack_target == 0: # Scan everything
                self.scan_aoe(player, self.vision_range, exclude_friendly=True,
                    exclude_hostile=False, exclude_creeps=False,
                    exclude_neutrals=False, exclude_towers=False)
            elif attack_target == 2: # Scan only heros and towers
                self.scan_aoe(player, self.vision_range, exclude_friendly=True,
                    exclude_hostile=False, exclude_creeps=True,
                    exclude_neutrals=True, exclude_towers=False)
 
            target = NULL
            if self.scanned_targets[pid][0] != NULL:
                target = self.nearest_scanned_target(player)

            if use_q and player.q_timer <= 0 and self.skills[pid][0](self, player, target) and player.q_uses < MAX_USES:
                player.q_uses += 1
            elif use_w and player.w_timer <= 0 and self.skills[pid][1](self, player, target) and player.w_uses < MAX_USES:
                player.w_uses += 1
            elif use_e and player.e_timer <= 0 and self.skills[pid][2](self, player, target) and player.e_uses < MAX_USES:
                player.e_uses += 1
            elif target != NULL and self.basic_attack(player, target) and player.basic_attack_uses < MAX_USES:
                player.basic_attack_uses += 1

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

        for pid in range(self.num_agents + self.num_towers + self.num_creeps + self.num_neutrals):
            player = self.get_entity(pid)
            player.target_pid = -1
            player.attack_aoe = 0
            player.last_x = player.x
            player.last_y = player.y
            player.is_hit = 0

        self.step_neutrals()
        self.step_creeps()
        self.step_towers()
        self.step_players()

        self.tick += 1

        #if self.tick % 1000 == 0:
        #    print(f'22 health: {self.get_tower(22).health}')
        #    print(f'23 health: {self.get_tower(23).health}')

        '''
        cdef Entity* ancient = self.get_tower(23)
        if ancient.health > 0 and self.grid[int(ancient.y), int(ancient.spawn_x)] == EMPTY:
            print('Ancient disappeared')
            print(f'Ancient health: {ancient.health}')
            print(f'Ancient spawn x: {ancient.spawn_x}')
            print(f'Ancient spawn y: {ancient.spawn_y}')
            print(f'Ancient y: {ancient.y}')
            print(f'Ancient x: {ancient.x}')
            exit(0)
        '''

        if self.get_tower(22).health <= 0:
            self.reset(0)
            self.radiant_victories += 1
            return 1 
        if self.get_tower(23).health <= 0:
            self.reset(0)
            self.dire_victories += 1
            return 2

        self.compute_observations()
        return 0
