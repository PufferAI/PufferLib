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

    ctypedef int (*skill)(MOBA*, Entity*, Entity*) noexcept

    ctypedef struct Map:
        unsigned char* grid;
        int* pids
        int width
        int height

    ctypedef struct CachedRNG:
        float* rng
        int rng_n
        int rng_idx

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
        skill* skills[10][3];

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

    void reset(MOBA* env)
    void step(MOBA* env)

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

        cdef MOBA* env = init_moba(num_agents, num_creeps, num_neutrals,
            num_towers, vision_range, agent_speed, discretize, reward_death,
            reward_xp, reward_distance, reward_tower)
        self.env = env

        # TODO: COPY THE GRID!!!
        cdef cnp.ndarray grid_copy = grid.copy()
        env.orig_grid = <unsigned char*> grid_copy.data
        env.map.grid = <unsigned char*> grid.data
        env.map.pids = <int*> pids.data

        env.ai_paths = <unsigned char*> ai_paths.data
        env.entities = <Entity*> entities.data
        env.observations_map = <unsigned char*> observations_map.data
        env.observations_extra = <unsigned char*> observations_extra.data
        env.rewards = <Reward*> rewards.data
        env.sum_rewards = <float*> sum_rewards.data
        env.norm_rewards = <float*> norm_rewards.data
        env.actions = <int*> actions.data

        # Hey, change the scanned_targets size to match!
        #assert num_agents + num_creeps + num_neutrals + num_towers <= 256
        #assert self.obs_size * self.obs_size <= 121

        env.tick = 0


    def reset(self):
        reset(self.env)

    def step(self):
        step(self.env)

