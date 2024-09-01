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

cdef extern from "moba.h":
    int EMPTY
    int WALL
    int TOWER
    int RADIANT_CREEP
    int DIRE_CREEP
    int NEUTRAL
    int RADIANT_SUPPORT
    int RADIANT_ASSASSIN
    int RADIANT_BURST
    int RADIANT_TANK
    int RADIANT_CARRY
    int DIRE_SUPPORT
    int DIRE_ASSASSIN
    int DIRE_BURST
    int DIRE_TANK
    int DIRE_CARRY

    int TOWER_VISION
    int CREEP_VISION
    int NEUTRAL_VISION

    int ENTITY_PLAYER
    int ENTITY_CREEP
    int ENTITY_NEUTRAL
    int ENTITY_TOWER

    int XP_RANGE
    int MAX_USES

    ctypedef struct Map:
        unsigned char* grid;
        int* pids;
        int width;
        int height;

    ctypedef struct CachedRNG:
        float* rng;
        int rng_n;
        int rng_idx;

    ctypedef struct Reward:
        float death;
        float xp;
        float distance;
        float tower;

    ctypedef struct MOBA:
        int num_agents;
        int num_creeps;
        int num_neutrals;
        int num_towers;
        int vision_range;
        float agent_speed;
        bint discretize;
        int obs_size;
        int creep_idx;
        int tick;

        Map* map;
        unsigned char* orig_grid;
        unsigned char* ai_paths;
        int atn_map[2][8];
        unsigned char* observations_map;
        unsigned char* observations_extra;
        int xp_for_level[30];
        int* actions;
        Entity* entities;

        float reward_death;
        float reward_xp;
        float reward_distance;
        float reward_tower;
        
        int total_towers_taken;
        int total_levels_gained;
        int radiant_victories;
        int dire_victories;

        # MAX_ENTITIES x MAX_SCANNED_TARGETS
        Entity* scanned_targets[256][121];
        void* skills[10][3];

        Reward* rewards;
        float* sum_rewards;
        float* norm_rewards;
        float waypoints[6][20][2];

        CachedRNG *rng;

    ctypedef struct Entity:
        int pid;
        int entity_type;
        int hero_type;
        int grid_id;
        int team;
        float health;
        float max_health;
        float mana;
        float max_mana;
        float y;
        float x;
        float spawn_y;
        float spawn_x;
        float damage;
        int lane;
        int waypoint;
        float move_speed;
        float move_modifier;
        int stun_timer;
        int move_timer;
        int q_timer;
        int w_timer;
        int e_timer;
        int q_uses;
        int w_uses;
        int e_uses;
        int basic_attack_uses;
        int basic_attack_timer;
        int basic_attack_cd;
        int is_hit;
        int level;
        int xp;
        int xp_on_kill;
        float reward;
        int tier;
        float base_health;
        float base_mana;
        float base_damage;
        int hp_gain_per_level;
        int mana_gain_per_level;
        int damage_gain_per_level;
        float damage_dealt;
        float damage_received;
        float healing_dealt;
        float healing_received;
        int deaths;
        int heros_killed;
        int creeps_killed;
        int neutrals_killed;
        int towers_killed;
        float last_x;
        float last_y;
        int target_pid;
        int attack_aoe;

    ctypedef struct Reward
    MOBA* init_moba()
    int creep_offset(MOBA* moba)
    int neutral_offset(MOBA* moba)
    int tower_offset(MOBA* moba)
    int player_offset(MOBA* moba)

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
  
cdef class Environment:
    cdef MOBA* env

    def __init__(self, cnp.ndarray grid, cnp.ndarray ai_paths,
            cnp.ndarray pids, cnp.ndarray entities, dict entity_data,
            cnp.ndarray player_obs, cnp.ndarray observations_map, cnp.ndarray observations_extra,
            cnp.ndarray rewards, cnp.ndarray sum_rewards, cnp.ndarray norm_rewards, cnp.ndarray actions,
            int num_agents, int num_creeps, int num_neutrals,
            int num_towers, int vision_range, float agent_speed, bint discretize, float reward_death,
            float reward_xp, float reward_distance, float reward_tower):

        cdef MOBA* env = init_moba();
        self.env = env

        # TODO: COPY THE GRID!!!
        cdef cnp.ndarray grid_copy = grid.copy()
        env.orig_grid = <unsigned char*> grid_copy.data
        env.map.grid = <unsigned char*> grid.data
        env.map.pids = <int*> pids.data
        env.map.width = 128
        env.map.height = 128

        env.ai_paths = <unsigned char*> ai_paths.data
        env.entities = <Entity*> entities.data
        env.observations_map = <unsigned char*> observations_map.data
        env.observations_extra = <unsigned char*> observations_extra.data
        env.rewards = <Reward*> rewards.data
        env.sum_rewards = <float*> sum_rewards.data
        env.norm_rewards = <float*> norm_rewards.data
        env.actions = <int*> actions.data

        env.num_agents = num_agents
        env.num_creeps = num_creeps
        env.num_neutrals = num_neutrals
        env.num_towers = num_towers
        env.vision_range = vision_range
        env.agent_speed = agent_speed
        env.discretize = discretize
        env.obs_size = 2*vision_range + 1
        env.creep_idx = 0

        env.reward_death = reward_death
        env.reward_xp = reward_xp
        env.reward_distance = reward_distance
        env.reward_tower = reward_tower
        env.total_towers_taken = 0
        env.total_levels_gained = 0
        env.radiant_victories = 0
        env.dire_victories = 0

        # Hey, change the scanned_targets size to match!
        #assert num_agents + num_creeps + num_neutrals + num_towers <= 256
        #assert self.obs_size * self.obs_size <= 121

        env.tick = 0

        # Preallocate RNG -1 to 1
        env.xp_for_level = np.array([
            0, 240, 640, 1160, 1760, 2440, 3200, 4000, 4900, 4900, 7000, 8200,
            9500, 10900, 12400, 14000, 15700, 17500, 19400, 21400, 23600, 26000,
            28600, 31400, 34400, 38400, 43400, 49400, 56400, 63900], dtype=np.int32)

        env.atn_map = np.array([
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
                player = &env.entities[pid]
                player.pid = pid
                player.entity_type = ENTITY_PLAYER
                player.team = team
                player.spawn_y = spawn_y
                player.spawn_x = spawn_x
                player.move_speed = agent_speed
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
            player = &env.entities[pid]
            player.pid = pid
            player.entity_type = ENTITY_PLAYER
            player.grid_id = RADIANT_SUPPORT + team*5
            player.hero_type = 0
            player.lane = 2 + 3*team
            env.skills[pid][0] = env.skill_support_hook
            env.skills[pid][1] = env.skill_support_aoe_heal
            env.skills[pid][2] = env.skill_support_stun
            player.base_health = 500
            player.base_mana = 250
            player.hp_gain_per_level = 100
            player.mana_gain_per_level = 50
            player.damage_gain_per_level = 10

            pid = 5*team + 1
            player = &env.entities[pid]
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
            player = &env.entities[pid]
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
            player = &env.entities[pid]
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
            player = &env.entities[pid]
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
            pid = tower_offset(env) + idx
            tower = &env.entities[pid]
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
                pid = neutral_offset(env) + idx
                neutral = &env.entities[pid]
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


