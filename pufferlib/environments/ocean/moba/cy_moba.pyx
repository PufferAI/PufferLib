from libc.stdlib cimport calloc, free
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

    int LOG_BUFFER_SIZE

    ctypedef struct PlayerLog:
        float episode_return
        float reward_death
        float reward_xp
        float reward_distance
        float reward_tower
        float level
        float kills
        float deaths
        float damage_dealt
        float damage_received
        float healing_dealt
        float healing_received
        float creeps_killed
        float neutrals_killed
        float towers_killed
        float usage_auto
        float usage_q
        float usage_w
        float usage_e

    ctypedef struct Log:
        float episode_return
        float episode_length
        float reward_death
        float reward_xp
        float reward_distance
        float reward_tower
     
        float radiant_victory
        float radiant_level
        float radiant_towers_alive

        float dire_victory
        float dire_level
        float dire_towers_alive
       
        PlayerLog radiant_support
        PlayerLog radiant_assassin
        PlayerLog radiant_burst
        PlayerLog radiant_tank
        PlayerLog radiant_carry

        PlayerLog dire_support
        PlayerLog dire_assassin
        PlayerLog dire_burst
        PlayerLog dire_tank
        PlayerLog dire_carry

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

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
        float last_x;
        float last_y;
        int target_pid;
        int attack_aoe;

    ctypedef struct MOBA:
        int vision_range;
        float agent_speed;
        unsigned char discretize;
        unsigned char script_opponents;
        int obs_size;
        int creep_idx;
        int tick;

        Map* map;
        unsigned char* ai_paths;
        int* ai_path_buffer;
        unsigned char* observations;
        int* actions;
        float* rewards;
        unsigned char* terminals;
        unsigned char* truncations;
        Entity* entities;
        Reward* reward_components;
        LogBuffer* log_buffer;
        PlayerLog log[10];

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
        skill skills[10][3];

        CachedRNG *rng;

    ctypedef struct GameRenderer
    GameRenderer* init_game_renderer(int cell_size, int width, int height)
    int render_game(GameRenderer* renderer, MOBA* env, int frame)
    void close_game_renderer(GameRenderer* renderer)

    ctypedef struct Reward
    void init_moba(MOBA* env, unsigned char* game_map_npy)
    void free_moba(MOBA* env)
 
    unsigned char* read_file(char* filename)

    void reset(MOBA* env)
    void step(MOBA* env)
    void randomize_tower_hp(MOBA* env)

cpdef entity_dtype():
    '''Make a dummy entity to get the dtype'''
    cdef Entity entity
    return np.asarray(<Entity[:1]>&entity).dtype

cpdef reward_dtype():
    '''Make a dummy reward to get the dtype'''
    cdef Reward reward
    return np.asarray(<Reward[:1]>&reward).dtype

cdef class CyMOBA:
    cdef MOBA* envs
    cdef GameRenderer* client
    cdef int num_envs
    cdef LogBuffer* logs

    cdef int* ai_path_buffer
    cdef unsigned char* ai_paths

    def __init__(self, unsigned char[:, :] observations, int[:, :] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,  int vision_range,
            float agent_speed, bint discretize, float reward_death, float reward_xp,
            float reward_distance, float reward_tower, bint script_opponents):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <MOBA*> calloc(num_envs, sizeof(MOBA))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef unsigned char* game_map_npy = read_file("resources/moba/game_map.npy");

        self.ai_path_buffer = <int*> calloc(3*8*128*128, sizeof(int))
        self.ai_paths = <unsigned char*> calloc(128*128*128*128, sizeof(unsigned char))
        cdef int i
        for i in range(128*128*128*128):
            self.ai_paths[i] = 255

        cdef int inc = 5 if script_opponents else 10
        for i in range(num_envs):
            self.envs[i] = MOBA(
                observations=&observations[inc*i, 0],
                actions=&actions[inc*i, 0],
                rewards=&rewards[inc*i],
                terminals=&terminals[inc*i],
                ai_paths = self.ai_paths,
                ai_path_buffer = self.ai_path_buffer,
                log_buffer=self.logs,
                vision_range=vision_range,
                agent_speed=agent_speed,
                discretize=discretize,
                reward_death=reward_death,
                reward_xp=reward_xp,
                reward_distance=reward_distance,
                reward_tower=reward_tower,
                script_opponents=script_opponents,
            )
            init_moba(&self.envs[i], game_map_npy)

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def render(self, int tick):
        if self.client == NULL:
            self.client = init_game_renderer(32, 41, 23)

        render_game(self.client, &self.envs[0], tick)

    def close(self):
        if self.client != NULL:
            close_game_renderer(self.client)
            self.client = NULL

        # TODO: free
        #free_moba(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
