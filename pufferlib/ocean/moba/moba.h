// Incremental port of Puffer Moba to C. Be careful to add semicolons and avoid leftover cython syntax
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h> // xxd -i game_map.npy > game_map.h #include "game_map.h"

#include "raylib.h"

#if defined(PLATFORM_DESKTOP)
    #define GLSL_VERSION 330
#else
    #define GLSL_VERSION 100
#endif


#define NUM_PLAYERS 10
#define NUM_CREEPS 100
#define NUM_NEUTRALS 72
#define NUM_TOWERS 24
#define NUM_ENTITIES (NUM_PLAYERS + NUM_CREEPS + NUM_NEUTRALS + NUM_TOWERS)
#define CREEP_OFFSET NUM_PLAYERS
#define NEUTRAL_OFFSET (NUM_PLAYERS + NUM_CREEPS)
#define TOWER_OFFSET (NUM_PLAYERS + NUM_CREEPS + NUM_NEUTRALS)

#define EMPTY 0
#define WALL 1
#define TOWER 2
#define RADIANT_CREEP 3
#define DIRE_CREEP 4
#define NEUTRAL 5
#define RADIANT_SUPPORT 6
#define RADIANT_ASSASSIN 7
#define RADIANT_BURST 8
#define RADIANT_TANK 9
#define RADIANT_CARRY 10
#define DIRE_SUPPORT 11
#define DIRE_ASSASSIN 12
#define DIRE_BURST 13
#define DIRE_TANK 14
#define DIRE_CARRY 15

#define TOWER_VISION 5
#define CREEP_VISION 7
#define NEUTRAL_VISION 3

#define ENTITY_PLAYER 0
#define ENTITY_CREEP 1
#define ENTITY_NEUTRAL 2
#define ENTITY_TOWER 3

#define XP_RANGE 7

const float XP_FOR_LEVEL[] = {0, 240, 640, 1160, 1760, 2440, 3200, 4000, 4900, 4900, 7000, 8200, 9500, 10900, 12400, 14000, 15700, 17500, 19400, 21400, 23600, 26000, 28600, 31400, 34400, 38400, 43400, 49400, 56400, 63900};

const float ATN_MAP[][8] = {
    {1, -1, 0, 0, 1, -1, -1, 1},
    {0, 0, 1, -1, -1, -1, 1, 1}
};

const float NEUTRAL_CAMP_X[] = {44.753846153846155, 74.41538461538462, 101.67692307692307, 89.92307692307692, 73.95384615384616, 64.38461538461539, 31.657692307692308, 95.67692307692307, 81.1076923076923, 34.99230769230769, 50.784615384615385, 63.646153846153844, 59.49230769230769, 44.69230769230769, 98.67307692307692, 28.642307692307693, 64.87692307692308, 51.46153846153846};
const float NEUTRAL_CAMP_Y[] = {71.99230769230769, 108.15384615384616, 102.16923076923078, 102.78461538461539, 40.753846153846155, 39.92307692307692, 39.96923076923077, 70.78461538461538, 69.18461538461538, 59.52307692307692, 99.95384615384614, 93.97692307692307, 49.86153846153846, 31.353846153846156, 61.06153846153846, 69.92307692307692, 83.83076923076923, 33.98461538461538};
const float TOWER_DAMAGE[] = {175.0, 175.0, 175.0, 175.0, 110.0, 175.0, 175.0, 110.0, 175.0, 175.0, 175.0, 110.0, 175.0, 110.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 110.0, 110.0, 0, 0};
const float TOWER_HEALTH[] = {2000, 2000, 2000, 2000, 1800, 2000, 2000, 1800, 2100, 2100, 2000, 1800, 2000, 1800, 2000, 2000, 2000, 2000, 2100, 2100, 1800, 1800, 4500, 4500};
const int TOWER_TEAM[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0};
const int TOWER_TIER[] = {3, 3, 3, 2, 1, 2, 2, 1, 4, 4, 2, 1, 2, 1, 2, 3, 3, 3, 4, 4, 1, 1, 5, 5};
const float TOWER_X[] = {34.6, 29.307692307692307, 14.292307692307695, 64.2, 102.87692307692308, 38.29230769230769, 17.615384615384613, 16.876923076923077, 21.06153846153846, 103.03076923076924, 65.0, 29.06153846153846, 84.2, 69.03076923076924, 112.75384615384615, 113.73846153846154, 92.32307692307693, 97.86153846153846, 105.61538461538461, 23.52307692307692, 53.12307692307692, 113.22307692307692, 107.52307692307693, 19.46153846153846};
const float TOWER_Y[] = {112.01538461538462, 96.87692307692308, 91.21538461538461, 113.61538461538461, 112.13846153846154, 85.43076923076923, 71.70769230769231, 51.03076923076923, 102.41538461538462, 28.261538461538464, 18.723076923076924, 18.723076923076924, 48.753846153846155, 59.98461538461538, 62.04615384615384, 41.676923076923075, 20.56923076923077, 36.08461538461539, 30.907692307692308, 104.93846153846154, 75.83076923076923, 78.3, 26.53846153846154, 106.16923076923078};
const float WAYPOINTS[][20][2] = {{{96.26153846153846, 14.04615384615385}, {93.61538461538461, 14.292307692307695}, {58.23076923076923, 16.07692307692308}, {43.4, 16.01538461538462}, {32.10769230769231, 17.584615384615383}, {24.93846153846154, 19.123076923076923}, {22.96923076923077, 20.446153846153848}, {21.06153846153846, 25.307692307692307}, {20.200000000000003, 30.47692307692308}, {19.4, 35.892307692307696}, {18.47692307692308, 43.52307692307693}, {18.04615384615385, 50.723076923076924}, {20.50769230769231, 94.66153846153847}, {27.676923076923075, 105.94615384615385}}, {{99.52307692307693, 26.47692307692308}, {98.66153846153847, 27.646153846153844}, {86.04615384615384, 41.12307692307692}, {71.33846153846154, 57.49230769230769}, {66.96923076923076, 62.23076923076923}, {64.07692307692308, 66.35384615384615}, {60.2, 72.01538461538462}, {51.892307692307696, 84.87692307692308}, {34.66153846153846, 99.4}, {28.169230769230772, 105.94615384615385}}, {{112.01538461538462, 36.93846153846154}, {112.01538461538462, 32.50769230769231}, {116.2, 68.6923076923077}, {113.73846153846154, 80.75384615384615}, {113.55384615384615, 90.53846153846155}, {113.21538461538461, 95.98461538461538}, {111.49230769230769, 100.93846153846154}, {109.06153846153846, 108.07692307692308}, {107.18461538461538, 111.36923076923077}, {101.73846153846154, 114.66153846153847}, {90.53846153846155, 111.76923076923077}, {39.4, 113.73846153846154}, {28.169230769230772, 106.43846153846154}}, {{20.446153846153848, 89.36923076923077}, {20.630769230769232, 94.66153846153847}, {18.784615384615385, 50.6}, {22.292307692307695, 24.50769230769231}, {22.815384615384616, 18.815384615384616}, {27.52307692307692, 17.03076923076923}, {35.03076923076923, 15.95384615384615}, {93.61538461538461, 14.415384615384617}, {103.64615384615385, 21.99230769230769}}, {{37.43076923076923, 96.50769230769231}, {34.53846153846154, 99.27692307692308}, {51.707692307692305, 84.38461538461539}, {59.83076923076923, 71.64615384615385}, {63.83076923076923, 66.1076923076923}, {66.72307692307692, 61.98461538461538}, {70.84615384615384, 57.246153846153845}, {98.53846153846155, 27.52307692307692}, {103.64615384615385, 22.48461538461538}}, {{36.93846153846154, 113.24615384615385}, {39.4, 113.61538461538461}, {62.292307692307695, 114.6}, {83.64615384615385, 114.6}, {90.66153846153847, 109.8}, {94.87692307692308, 112.01538461538462}, {104.26153846153846, 112.2923076923077}, {109.3076923076923, 107.83076923076922}, {112.23076923076923, 105.86153846153846}, {114.44615384615385, 72.96923076923076}, {111.8923076923077, 32.50769230769231}, {104.13846153846154, 22.48461538461538}}};
const int WAYPOINTS_N[6] = {14, 10, 13, 9, 9, 12};

#define LOG_BUFFER_SIZE 1024


typedef struct PlayerLog PlayerLog;
struct PlayerLog {
    float episode_return;
    float reward_death;
    float reward_xp;
    float reward_distance;
    float reward_tower;
    float level;
    float kills;
    float deaths;
    float damage_dealt;
    float damage_received;
    float healing_dealt;
    float healing_received;
    float creeps_killed;
    float neutrals_killed;
    float towers_killed;
    float usage_auto;
    float usage_q;
    float usage_w;
    float usage_e;
};

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float reward_death;
    float reward_xp;
    float reward_distance;
    float reward_tower;
 
    float radiant_victory;
    float radiant_level;
    float radiant_towers_alive;

    float dire_victory;
    float dire_level;
    float dire_towers_alive;
   
    PlayerLog radiant_support;
    PlayerLog radiant_assassin;
    PlayerLog radiant_burst;
    PlayerLog radiant_tank;
    PlayerLog radiant_carry;

    PlayerLog dire_support;
    PlayerLog dire_assassin;
    PlayerLog dire_burst;
    PlayerLog dire_tank;
    PlayerLog dire_carry;
};

typedef struct LogBuffer LogBuffer;
struct LogBuffer {
    Log* logs;
    int length;
    int idx;
};

LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
    //printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    PlayerLog* aggregated[10] = {
        &log.radiant_support, &log.radiant_assassin, &log.radiant_burst,
        &log.radiant_tank, &log.radiant_carry, &log.dire_support,
        &log.dire_assassin, &log.dire_burst, &log.dire_tank, &log.dire_carry
    };
    for (int i = 0; i < logs->idx; i++) {
        log.perf += logs->logs[i].radiant_victory / logs->idx;
        log.score += logs->logs[i].radiant_towers_alive / logs->idx;
        log.episode_return += logs->logs[i].episode_return / logs->idx;
        log.episode_length += logs->logs[i].episode_length / logs->idx;
        log.reward_death += logs->logs[i].reward_death / logs->idx;
        log.reward_xp += logs->logs[i].reward_xp / logs->idx;
        log.reward_distance += logs->logs[i].reward_distance / logs->idx;
        log.reward_tower += logs->logs[i].reward_tower / logs->idx;
        log.radiant_victory += logs->logs[i].radiant_victory / logs->idx;
        log.radiant_level += logs->logs[i].radiant_level / logs->idx;
        log.radiant_towers_alive += logs->logs[i].radiant_towers_alive / logs->idx;
        log.dire_victory += logs->logs[i].dire_victory / logs->idx;
        log.dire_level += logs->logs[i].dire_level / logs->idx;
        log.dire_towers_alive += logs->logs[i].dire_towers_alive / logs->idx;

        PlayerLog* individual[10] = {
            &logs->logs[i].radiant_support, &logs->logs[i].radiant_assassin,
            &logs->logs[i].radiant_burst, &logs->logs[i].radiant_tank,
            &logs->logs[i].radiant_carry, &logs->logs[i].dire_support,
            &logs->logs[i].dire_assassin, &logs->logs[i].dire_burst,
            &logs->logs[i].dire_tank, &logs->logs[i].dire_carry
        };

        for (int j = 0; j < 10; j++) {
            aggregated[j]->episode_return += individual[j]->episode_return / logs->idx;
            aggregated[j]->reward_death += individual[j]->reward_death / logs->idx;
            aggregated[j]->reward_xp += individual[j]->reward_xp / logs->idx;
            aggregated[j]->reward_distance += individual[j]->reward_distance / logs->idx;
            aggregated[j]->reward_tower += individual[j]->reward_tower / logs->idx;
            aggregated[j]->level += individual[j]->level / logs->idx;
            aggregated[j]->kills += individual[j]->kills / logs->idx;
            aggregated[j]->deaths += individual[j]->deaths / logs->idx;
            aggregated[j]->damage_dealt += individual[j]->damage_dealt / logs->idx;
            aggregated[j]->damage_received += individual[j]->damage_received / logs->idx;
            aggregated[j]->healing_dealt += individual[j]->healing_dealt / logs->idx;
            aggregated[j]->healing_received += individual[j]->healing_received / logs->idx;
            aggregated[j]->creeps_killed += individual[j]->creeps_killed / logs->idx;
            aggregated[j]->neutrals_killed += individual[j]->neutrals_killed / logs->idx;
            aggregated[j]->towers_killed += individual[j]->towers_killed / logs->idx;
            aggregated[j]->usage_auto += individual[j]->usage_auto / logs->idx;
            aggregated[j]->usage_q += individual[j]->usage_q / logs->idx;
            aggregated[j]->usage_w += individual[j]->usage_w / logs->idx;
            aggregated[j]->usage_e += individual[j]->usage_e / logs->idx;
        }
    }
    logs->idx = 0;
    return log;
}
 
typedef struct MOBA MOBA;
typedef struct Entity Entity;
typedef int (*skill)(MOBA*, Entity*, Entity*);

struct Entity {
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
};

typedef struct {
    unsigned char* grid;
    int* pids;
    int width;
    int height;
} Map;

static inline int map_offset(Map* map, int y, int x) {
    return y*map->width + x;
}

static inline int ai_offset(int y_dst, int x_dst, int y_src, int x_src) {
    return y_dst*128*128*128 + x_dst*128*128 + y_src*128 + x_src;
}

typedef struct {
    float death;
    float xp;
    float distance;
    float tower;
} Reward;

typedef struct {
    float* rng;
    int rng_n;
    int rng_idx;
} CachedRNG;

float fast_rng(CachedRNG* rng) {
    float val = rng->rng[rng->rng_idx];
    rng->rng_idx += 1;
    if (rng->rng_idx >= rng->rng_n - 1)
        rng->rng_idx = 0;
    return val;
}

int bfs(Map *map, unsigned char *flat_paths, int* flat_buffer, int dest_r, int dest_c) {
    int N = map->width;
    unsigned char (*paths)[N] = (unsigned char(*)[N])flat_paths;
    int (*buffer)[3] = (int(*)[3])flat_buffer;

    int start = 0;
    int end = 1;

    int adr = map_offset(map, dest_r, dest_c);
    if (map->grid[adr] == 1) {
        return 1;
    }

    buffer[start][0] = 0;
    buffer[start][1] = dest_r;
    buffer[start][2] = dest_c;
    while (start < end) {
        int atn = buffer[start][0];
        int start_r = buffer[start][1];
        int start_c = buffer[start][2];
        start++;

        if (start_r < 0 || start_r >= N || start_c < 0 || start_c >= N) {
            continue;
        }
        if (paths[start_r][start_c] != 255) {
            continue;
        }
        int adr = map_offset(map, start_r, start_c);
        if (map->grid[adr] == 1) {
            paths[start_r][start_c] = 8;
            continue;
        }

        paths[start_r][start_c] = atn;

        buffer[end][0] = 0;
        buffer[end][1] = start_r - 1;
        buffer[end][2] = start_c;
        end++;

        buffer[end][0] = 1;
        buffer[end][1] = start_r + 1;
        buffer[end][2] = start_c;
        end++;

        buffer[end][0] = 2;
        buffer[end][1] = start_r;
        buffer[end][2] = start_c - 1;
        end++;

        buffer[end][0] = 3;
        buffer[end][1] = start_r;
        buffer[end][2] = start_c + 1;
        end++;

        buffer[end][0] = 4;
        buffer[end][1] = start_r - 1;
        buffer[end][2] = start_c + 1;
        end++;

        buffer[end][0] = 5;
        buffer[end][1] = start_r + 1;
        buffer[end][2] = start_c + 1;
        end++;

        buffer[end][0] = 6;
        buffer[end][1] = start_r + 1;
        buffer[end][2] = start_c - 1;
        end++;

        buffer[end][0] = 7;
        buffer[end][1] = start_r - 1;
        buffer[end][2] = start_c - 1;
        end++;
    }
    paths[dest_r][dest_c] = 8;
    return 0;
}

unsigned char* precompute_pathing(Map* map){
    int N = map->width;
    unsigned char* paths = calloc(N*N*N*N, 1);
    int* buffer = calloc(3*8*N*N, sizeof(int));
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            for (int rr = 0; rr < N; rr++) {
                for (int cc = 0; cc < N; cc++) {
                    int adr = ai_offset(r, c, rr, cc);
                    paths[adr] = 255;
                }
            }
            int adr = ai_offset(r, c, 0, 0);
            bfs(map, &paths[adr], buffer, r, c);
        }
    }
    return paths;
}

struct MOBA {
    int vision_range;
    float agent_speed;
    bool discretize;
    bool script_opponents;
    int obs_size;
    int creep_idx;
    int tick;
    bool should_reset;

    Map* map;
    unsigned char* orig_grid;
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

    // MAX_ENTITIES x MAX_SCANNED_TARGETS
    Entity* scanned_targets[256][121];
    skill skills[10][3];

    CachedRNG *rng;
};

void free_moba(MOBA* env) {
    free(env->reward_components);
    free(env->map->grid);
    free(env->map);
    free(env->orig_grid);
    free(env->rng->rng);
    free(env->rng);
}

void free_allocated_moba(MOBA* env) {
    free_logbuffer(env->log_buffer);
    free(env->rewards);
    free(env->map->pids);
    free(env->ai_path_buffer);
    free(env->ai_paths);
    free(env->observations);
    free(env->actions);
    free(env->terminals);
    free(env->truncations);
    free(env->entities);
    free_moba(env);
}

void compute_observations(MOBA* env) {
    int agents = (env->script_opponents) ? NUM_PLAYERS/2 : NUM_PLAYERS;
    memset(env->observations, 0, agents*(11*11*4 + 26)*sizeof(unsigned char));

    int vis = env->vision_range;
    Map* map = env->map;

    unsigned char (*observations)[11*11*4 + 26] = (unsigned char(*)[11*11*4 + 26])env->observations;
    for (int pid = 0; pid < agents; pid++) {
        // Does this copy?
        unsigned char* obs_map = &observations[pid][0];
        unsigned char* obs_extra = &observations[pid][11*11*4];

        Entity* player = &env->entities[pid];
        Reward* reward = &env->reward_components[pid];

        int y = player->y;
        int x = player->x;

        // TODO: Add bounds debug checks asserts
        obs_extra[0] = 2*x;
        obs_extra[1] = 2*y;
        obs_extra[2] = 255*player->level/30.0;
        obs_extra[3] = 255*player->health/player->max_health;
        obs_extra[4] = 255*player->mana/player->max_mana;
        obs_extra[5] = player->damage / 4.0;
        obs_extra[6] = 100*player->move_speed;
        obs_extra[7] = player->move_modifier*100;
        obs_extra[8] = 2*player->stun_timer;
        obs_extra[9] = 2*player->move_timer;
        obs_extra[10] = 2*player->q_timer;
        obs_extra[11] = 2*player->w_timer;
        obs_extra[12] = 2*player->e_timer;
        obs_extra[13] = 50*player->basic_attack_timer;
        obs_extra[14] = 50*player->basic_attack_cd;
        obs_extra[15] = 255*player->is_hit;
        obs_extra[16] = 255*player->team;
        obs_extra[17 + player->hero_type] = 255;

        // Assumes scaled between -1 and 1, else overflows
        obs_extra[22] = (reward->death == 0) ? 0 : 255;
        obs_extra[23] = (reward->xp == 0) ? 0 : 255;
        obs_extra[24] = (reward->distance == 0) ? 0 : 255;
        obs_extra[25] = (reward->tower == 0) ? 0 : 255;

        for (int dy = -vis; dy <= vis; dy++) {
            for (int dx = -vis; dx <= vis; dx++) {
                int xx = x + dx;
                int yy = y + dy;
                int ob_x = dx + vis;
                int ob_y = dy + vis;

                int map_idx = ob_y*11 + ob_x;

                int adr = map_offset(map, yy, xx);
                int tile = map->grid[adr];
                obs_map[map_idx] = tile;
                if (tile > 15) {
                    printf("Invalid map value: %i at %i, %i\n", map->grid[adr], yy, xx);
                }
                int target_pid = env->map->pids[adr];
                if (target_pid == -1)
                    continue;

                Entity* target = &env->entities[target_pid];
                obs_map[map_idx+1] = 255*target->health/target->max_health;
                if (target->max_mana > 0) { // Towers do not have mana
                    obs_map[map_idx+2] = 255*target->mana/target->max_mana;
                }
                obs_map[map_idx+3] = target->level/30.0;
            }
        }
    }
}
        
static inline int xp_for_player_kill(Entity* entity) {
    return 100 + (int)(entity->xp / 7.69);
}
 
static inline float clip(float x) {
    return fmaxf(-1.0f, fminf(x, 1.0f));
}

static inline float l1_distance(float x1, float y1, float x2, float y2) {
    return fabs(x1 - x2) + fabs(y1 - y2);
}

// TODO: Should not take entire moba. Rename to min_greater_than or similar
int calc_level(MOBA* env, int xp) {
    int i;
    for (i = 0; i < 30; i++) {
        if (xp < XP_FOR_LEVEL[i])
            return i + 1;
    }
    return i;
}

Reward* get_reward(MOBA* env, int pid) {
    return &env->reward_components[pid];
}

int move_to(Map* map, Entity* player, float dest_y, float dest_x) {
    int src = map_offset(map, (int)player->y, (int)player->x);
    int dst = map_offset(map, (int)dest_y, (int)dest_x);

    if (map->grid[dst] != EMPTY && map->pids[dst] != player->pid)
        return 1;

    /*
    int idest_y = dest_y;
    int idest_x = dest_x;
    if (idest_y == 106 && idest_x == 19 && player->pid != 205) {
        printf("Moving player pid %i to %i, %i\n", player->pid, idest_y, idest_x);
        exit(0);
    }
    */

    map->grid[src] = EMPTY;
    map->grid[dst] = player->grid_id;

    map->pids[src] = -1;
    map->pids[dst] = player->pid;

    player->y = dest_y;
    player->x = dest_x;
    return 0;
}

int move_near(Map* map, Entity* entity, Entity* target) {
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (move_to(map, entity, target->y + dy, target->x + dx) == 0)
                return 0;
        }
    }
    return 1;
}

int move_towards(MOBA* env, Entity* entity, int y_dst, int x_dst, float speed) {
    int y_src = entity->y;
    int x_src = entity->x;

    int adr = ai_offset(y_dst, x_dst, y_src, x_src);
    int atn = env->ai_paths[adr];

    // Compute path if not cached
    if (atn == 255) {
        int bfs_adr = ai_offset(y_dst, x_dst, 0, 0);
        bfs(env->map, &env->ai_paths[bfs_adr], env->ai_path_buffer, y_dst, x_dst);
        atn = env->ai_paths[adr];
    }

    if (atn >= 8)
        return 0;

    float modifier = speed * entity->move_modifier;
    y_dst = y_src + modifier*ATN_MAP[0][atn];
    x_dst = x_src + modifier*ATN_MAP[1][atn];

    if (move_to(env->map, entity, y_dst, x_dst) == 0)
        return 0;

    float jitter_x = fast_rng(env->rng);
    float jitter_y = fast_rng(env->rng);
    return move_to(env->map, entity, entity->y + jitter_y, entity->x + jitter_x);
}

void kill_entity(Map* map, Entity* entity) {
    if (entity->pid == -1) {
        return;
    }
    int adr = map_offset(map, (int)entity->y, (int)entity->x);
    map->grid[adr] = EMPTY;
    map->pids[adr] = -1;
    entity->pid = -1;
    entity->target_pid = -1;
    entity->health = 0;
    entity->last_x = 0;
    entity->last_y = 0;
    entity->x = 0;
    entity->y = 0;
}

void spawn_player(Map* map, Entity* entity) {
    int pid = entity->pid;
    kill_entity(map, entity);
    entity->pid = pid;

    entity->max_health = entity->base_health + entity->level*entity->hp_gain_per_level;
    entity->max_mana = entity->base_mana + entity->level*entity->mana_gain_per_level;
    entity->damage = entity->base_damage + entity->level*entity->damage_gain_per_level;
 
    entity->health = entity->max_health;
    entity->mana = entity->max_mana;
    entity->basic_attack_timer = 0;
    entity->move_modifier = 0;
    entity->move_timer = 0;
    entity->stun_timer = 0;
    entity->q_timer = 0;
    entity->w_timer = 0;
    entity->e_timer = 0;
    entity->target_pid = -1;
    entity->waypoint = 1;
    
    // TODO: Cache noise?
    // Also.. technically can infinite loop?
    bool valid_pos = false;
    int y, x;
    while (!valid_pos) {
        y = entity->spawn_y + rand()%15 - 7;
        x = entity->spawn_x + rand()%15 - 7;
        valid_pos = map->grid[map_offset(map, y, x)] == EMPTY;
    }
    entity->last_x = x;
    entity->last_y = y;
    int err = move_to(map, entity, y, x);
    if (err) {
        printf("Failed to move entity %i to %i, %i\n", entity->pid, y, x);
        exit(0);
    }
}

int attack(MOBA* env, Entity* player, Entity* target, float damage) {
    if (target->pid == -1 || target->team == player->team)
        return 1;

    float dist_to_target = l1_distance(player->y, player->x, target->y, target->x);
    if (dist_to_target > 12) {
        return 1;
        //printf("Attacker %i at %f, %f, target %i at %f, %f, dist %f\n", player->pid, player->y, player->x, target->pid, target->y, target->x, dist_to_target);
    }

    // Dummy logs for non-player entities
    // TODO: Improve this design
    PlayerLog empty_log = {0};
    PlayerLog* player_log = &empty_log;
    if (player->entity_type == ENTITY_PLAYER) {
        player_log = &env->log[player->pid];
    }
    PlayerLog* target_log = &empty_log;
    if (target->entity_type == ENTITY_PLAYER) {
        target_log = &env->log[target->pid];
    }

    if (damage < target->health) {
        player_log->damage_dealt += damage;
        target_log->damage_received += damage;
        target->health -= damage;
        player->target_pid = target->pid;
        target->is_hit = 1;
        return 0;
    }

    player_log->damage_dealt += target->health;
    target_log->damage_received += target->health;
    target->health = 0;

    if (target->pid  == 205) {
        //printf("Killed the radiant ancient\n");
        env->should_reset = true;
    }

    int target_type = target->entity_type;
    if (target_type == ENTITY_PLAYER) {
        env->reward_components[target->pid].death = env->reward_death;
        target_log->reward_death = env->reward_death;
        player_log->kills += 1;
        target_log->deaths += 1;
        spawn_player(env->map, target);
    } else if (target_type == ENTITY_CREEP) {
        player_log->creeps_killed += 1;
        kill_entity(env->map, target);
    } else if (target_type == ENTITY_NEUTRAL) {
        player_log->neutrals_killed += 1;
        kill_entity(env->map, target);
    } else if (target_type == ENTITY_TOWER) {
        player_log->towers_killed += 1;
        kill_entity(env->map, target);
    }

    if (player->entity_type != ENTITY_PLAYER)
        return 0;

    int xp = 0;
    if (target_type == ENTITY_PLAYER)
        xp = xp_for_player_kill(target);
    else
        xp = target->xp_on_kill;

    // Share xp with allies in range
    // TODO: You can replace this array with a pointer array
    int first_player_on_team = (player->team == 0) ? 0 : 5;
    bool in_range[5] = {false, false, false, false, false};
    float target_x = target->x;
    float target_y = target->y;
    int num_in_range = 0;
    for (int i = 0; i < 5; i++) {
        Entity* ally = &env->entities[first_player_on_team + i];
        if (ally->pid == player->pid) {
            in_range[i] = true;
            num_in_range += 1;
            continue;
        }

        if (l1_distance(ally->y, ally->x, target_y, target_x) <= XP_RANGE) {
            in_range[i] = true;
            num_in_range += 1;
        }
    }

    xp /= num_in_range;

    for (int i = 0; i < 5; i++) {
        if (!in_range[i])
            continue;

        Entity* ally = &env->entities[first_player_on_team + i];
        if (ally->xp > 10000000)
            continue;

        ally->xp += xp;
        env->reward_components[first_player_on_team + i].xp = xp*env->reward_xp;

        int level = ally->level;
        ally->level = calc_level(env, ally->xp);
        if (ally->level > level) {
            PlayerLog* ally_log = &env->log[ally->pid];
            ally_log->level = ally->level;
            env->total_levels_gained += 1;
        }

        ally->max_health = ally->base_health + ally->level*ally->hp_gain_per_level;
        ally->max_mana = ally->base_mana + ally->level*ally->mana_gain_per_level;
        ally->damage = ally->base_damage + ally->level*ally->damage_gain_per_level;
    }

    if (target->entity_type == ENTITY_TOWER) {
        env->reward_components[player->pid].tower = env->reward_tower;
        env->total_towers_taken += 1;
    }
    return 0;
}

int basic_attack(MOBA* env, Entity* player, Entity* target) {
    if (player->basic_attack_timer > 0)
        return 1;

    player->basic_attack_timer = player->basic_attack_cd;
    return attack(env, player, target, player->damage);
}

int heal(MOBA* env, Entity* player, Entity* target, float amount) {
    if (target->pid == -1 || target->team != player->team)
        return 1;

    // Currently only allowed to heal players
    if (target->entity_type != ENTITY_PLAYER)
        return 1;

    PlayerLog* player_log = &env->log[player->pid];
    PlayerLog* target_log = &env->log[target->pid];
    int missing_health = target->max_health - target->health;
    if (amount <= missing_health) {
        target->health += amount;
        player_log->healing_dealt += amount;
        target_log->healing_received += amount;
        return 0;
    }

    target->health = target->max_health;
    player_log->healing_dealt += missing_health;
    target_log->healing_received += missing_health;
    return 0;
}

int spawn_creep(MOBA* env, int idx, int lane) {
    int pid = CREEP_OFFSET + idx;
    Entity* creep = &env->entities[pid];

    if (lane < 3) {
        creep->team = 0;
        creep->grid_id = RADIANT_CREEP;
    } else {
        creep->team = 1;
        creep->grid_id = DIRE_CREEP;
    }
    creep->pid = pid;
    creep->entity_type = ENTITY_CREEP;
    creep->health = 450;
    creep->max_health = 450;
    creep->lane = lane;
    creep->waypoint = 0;
    creep->xp_on_kill = 60;
    creep->damage = 22;
    creep->basic_attack_cd = 5;

    int spawn_y = WAYPOINTS[lane][0][0];
    int spawn_x = WAYPOINTS[lane][0][1];

    Map* map = env->map;
    int y, x;
    for (int i = 0; i < 10; i++) {
        y = spawn_y + rand() % 7 - 3;
        x = spawn_x + rand() % 7 - 3;
        int adr = map_offset(map, y, x);
        if (map->grid[adr] == EMPTY) {
            break;
        }
    }
    creep->health = creep->max_health;
    creep->target_pid = -1;
    creep->waypoint = 1;
    creep->last_x = x;
    creep->last_y = y;
    return move_to(env->map, creep, y, x);
}

int spawn_neutral(MOBA* env, int idx) {
    int pid = NEUTRAL_OFFSET + idx;
    Entity* neutral = &env->entities[pid];
    neutral->pid = pid;
    neutral->health = neutral->max_health;
    neutral->basic_attack_timer = 0;
    neutral->target_pid = -1;

    // TODO: Clean up spawn regions. Some might be offset and are obscured.
    // Maybe check all valid spawn spots?
    int y, x;
    Map* map = env->map;
    int spawn_y = (int)neutral->spawn_y;
    int spawn_x = (int)neutral->spawn_x;
    for (int i = 0; i < 100; i++) {
        y = spawn_y + rand() % 7 - 3;
        x = spawn_x + rand() % 7 - 3;
        int adr = map_offset(map, y, x);
        if (map->grid[adr] == EMPTY) {
            break;
        }
    }
    neutral->last_x = x;
    neutral->last_y = y;
    return move_to(env->map, neutral, y, x);
}

// TODO: Rework spawn system
int spawn_at(Map* map, Entity* entity, float y, float x) {
    int adr = map_offset(map, (int)y, (int)x);

    if (map->grid[adr] != EMPTY)
        return 1;

    map->grid[adr] = entity->grid_id;
    map->pids[adr] = entity->pid;
    entity->y = y;
    entity->x = x;
    return 0;
}

int scan_aoe(MOBA* env, Entity* player, int radius,
        bool exclude_friendly, bool exclude_hostile, bool exclude_creeps,
        bool exclude_neutrals, bool exclude_towers) {

    Map* map = env->map;
    int player_y = player->y;
    int player_x = player->x;
    int player_team = player->team;
    int pid = player->pid;
    int idx = 0;

    for (int y = player_y-radius; y <= player_y+radius; y++) {
        for (int x = player_x-radius; x <= player_x+radius; x++) {
            if (y < 0 || y >= 128 || x < 0 || x >= 128)
                continue;

            int adr = map_offset(map, y, x);
            int target_pid = map->pids[adr];
            if (target_pid == -1)
                continue;

            Entity* target = &env->entities[target_pid];

            int target_team = target->team;
            if (exclude_friendly && target_team == player_team)
                continue;
            if (exclude_hostile && target_team != player_team)
                continue;

            int target_type = target->entity_type;
            if (exclude_neutrals && target_type == ENTITY_NEUTRAL)
                continue;
            if (exclude_creeps && target_type == ENTITY_CREEP)
                continue;
            if (exclude_towers && target_type == ENTITY_TOWER)
                continue;

            /*if (player->pid >= 5 && player->entity_type == ENTITY_PLAYER && target_type == ENTITY_TOWER && target->tier==5) {
                printf("Player %i at %f, %f scanned tower %i at %f, %f\n", player->pid, player->y, player->x, target->pid, target->y, target->x);
            }
            */

            env->scanned_targets[pid][idx] = target;
            float dist_to_target = l1_distance(player->y, player->x, target->y, target->x);
            if (dist_to_target > 20) {
                printf("Invalid target at %f, %f\n", target->y, target->x);
                printf("player x: %f, y: %f, target x: %f, y: %f, dist: %f\n", player->x, player->y, target->x, target->y, dist_to_target);
                printf("player pid: %i, target pid: %i\n", player->pid, target->pid);
                printf("Tick: %i\n", env->tick);
                exit(0);
            }
            idx += 1;
        }
    }
    env->scanned_targets[pid][idx] = NULL;
    return (idx == 0) ? 1 : 0;
}

Entity* nearest_scanned_target(MOBA* env, Entity* player){
    Entity* nearest_target = NULL;
    float nearest_dist = 9999999;
    float player_y = player->y;
    float player_x = player->x;
    int pid = player->pid;

    // TODO: Clean up
    for (int idx = 0; idx < 121; idx++) {
        Entity* target = env->scanned_targets[pid][idx];
        if (target == NULL)
            break;

        float dist = l1_distance(player_y, player_x, target->y, target->x);
        if (dist < nearest_dist) {
            nearest_target = target;
            nearest_dist = dist;
        }
    }
    return nearest_target;
}

void aoe_scanned(MOBA* env, Entity* player, Entity* target, float damage, int stun) {
    int pid = player->pid;
    for (int idx = 0; idx < 121; idx++) {
        Entity* target = env->scanned_targets[pid][idx];

        if (target == NULL)
            break;

        // Negative damage is healing
        if (damage < 0) {
            heal(env, player, target, -damage);
            continue;
        }

        attack(env, player, target, damage);
        if (stun > 0)
            target->stun_timer = stun;
    }
}

int player_aoe_attack(MOBA* env, Entity* player,
        Entity* target, int radius, float damage, int stun) {
    bool exclude_hostile = damage < 0;
    bool exclude_friendly = !exclude_hostile;

    int err = scan_aoe(env, player, radius, exclude_friendly,
        exclude_hostile, false, false, false);

    if (err != 0)
        return 1;

    aoe_scanned(env, player, target, damage, stun);
    player->target_pid = target->pid;
    player->attack_aoe = radius;
    return 0;
}

int push(MOBA* env, Entity* player, Entity* target, float amount) {
    float dx = target->x - player->x;
    float dy = target->y - player->y;
    float dist = fabs(dx) + fabs(dy);

    if (dist == 0.0)
        return 1;

    // Norm to unit vector
    dx = amount * dx / dist;
    dy = amount * dy / dist;
    return move_to(env->map, target, target->y + dy, target->x + dx);
}

int pull(MOBA* env, Entity* player, Entity* target, float amount) {
    return push(env, player, target, -amount);
}

int aoe_pull(MOBA* env, Entity* player, int radius, float amount) {
    scan_aoe(env, player, radius, true, false, false, false, true);
    int err = 1;
    int pid = player->pid;
    for (int idx = 0; idx < 121; idx++) {
        Entity* target = env->scanned_targets[pid][idx];
        if (target == NULL)
            break;

        pull(env, target, player, amount);
        err = 0;
    }
    return err;
}

void creep_ai(MOBA* env, Entity* creep) {
    int waypoint = creep->waypoint;
    int lane = creep->lane;
    int pid = creep->pid;

    if (env->tick % 5 == 0)
        scan_aoe(env, creep, CREEP_VISION, true, false, false, true, false);

    if (env->scanned_targets[pid][0] != NULL) {
        Entity* target = nearest_scanned_target(env, creep);
        float dest_y = target->y;
        float dest_x = target->x;
        float dist = l1_distance(creep->y, creep->x, dest_y, dest_x);
        if (dist < 2)
            basic_attack(env, creep, target);

        move_towards(env, creep, dest_y, dest_x, env->agent_speed);
    } else {
        float dest_y = WAYPOINTS[lane][waypoint][0];
        float dest_x = WAYPOINTS[lane][waypoint][1];
        move_towards(env, creep, dest_y, dest_x, env->agent_speed);

        float dist = l1_distance(creep->y, creep->x, dest_y, dest_x);
        if (dist < 2 && creep->waypoint < WAYPOINTS_N[lane] - 1)
            creep->waypoint += 1;
        //if (creep->waypoint == WAYPOINTS_N[lane] - 1)
        //    printf("Creep %i at %f, %f, waypoint %i\n", creep->pid, creep->y, creep->x, creep->waypoint);
    }
}

void neutral_ai(MOBA* env, Entity* neutral) {
    if (env->tick % 5 == 0) {
        scan_aoe(env, neutral, NEUTRAL_VISION, true, false, true, true, true);
    }
    
    int pid = neutral->pid;
    if (env->scanned_targets[pid][0] != NULL) {
        Entity* target = nearest_scanned_target(env, neutral);
        if (l1_distance(neutral->y, neutral->x, target->y, target->x) < 2)
            basic_attack(env, neutral, target);
        else
            move_towards(env, neutral, target->y, target->x, env->agent_speed);
        
    } else if (l1_distance(neutral->y, neutral->x,
            neutral->spawn_y, neutral->spawn_x) > 2) {
        move_towards(env, neutral, neutral->spawn_y, neutral->spawn_x, env->agent_speed);
    }
}

void randomize_tower_hp(MOBA* env) {
    for (int i = 0; i < NUM_TOWERS; i++) {
        int pid = TOWER_OFFSET + i;
        Entity* tower = &env->entities[pid];
        tower->health = rand() % (int)tower->max_health + 1;
    }
}

void update_status(Entity* entity) {
    if (entity->stun_timer > 0)
        entity->stun_timer -= 1;
    
    if (entity->move_timer > 0)
        entity->move_timer -= 1;
    
    if (entity->move_timer == 0)
        entity->move_modifier = 1.0;
}
    
void update_cooldowns(Entity* entity) {
    if (entity->q_timer > 0)
        entity->q_timer -= 1;
    
    if (entity->w_timer > 0)
        entity->w_timer -= 1;
    
    if (entity->e_timer > 0)
        entity->e_timer -= 1;
    
    if (entity->basic_attack_timer > 0)
        entity->basic_attack_timer -= 1;
}

// TODO: Fix
int skill_support_hook(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 100;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    pull(env, target, player, 1.5 + 0.1*player->level);
    player->mana -= mana_cost;
    player->q_timer = 15;
    return 0;
}

int skill_support_aoe_heal(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 100;
    if (player->mana < mana_cost)
        return 1;

    if (player_aoe_attack(env, player, player, 5, -350 - 50*player->level, 0) == 0) {
        player->mana -= mana_cost;
        player->w_timer = 50;
        return 0;
    }
    return 1;
}

int skill_support_stun(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 75;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    if (attack(env, player, target, 50 + 20*player->level) == 0) {
        target->stun_timer = 15 + 0.5*player->level;
        player->mana -= mana_cost;
        player->e_timer = 60;
        return 0;
    }
    return 1;
}

int skill_assassin_aoe_minions(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 100;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    int target_type = target->entity_type;
    if (target_type != ENTITY_CREEP && target_type != ENTITY_NEUTRAL)
        return 1;

    if (player_aoe_attack(env, player, target, 3, 100 + 20*player->level, 0) == 0) {
        player->mana -= mana_cost;
        player->q_timer = 40;
        return 0;
    }
    return 1;
}

int skill_assassin_tp_damage(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 150;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    if (move_near(env->map, player, target) != 0) {
        return 1;
    }

    player->mana -= mana_cost;
    if (attack(env, player, target, 250+50*player->level) == 0) {
        player->w_timer = 60;
        return 0;
    }
    return 1;
}

int skill_assassin_move_buff(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 100;
    if (player->mana < mana_cost)
        return 1;
    
    player->move_modifier = 2.0;
    player->move_timer = 25;
    player->mana -= mana_cost;
    player->e_timer = 100;
    return 0;
}

int skill_burst_nuke(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 200;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    if (attack(env, player, target, 250 + 40*player->level) == 0) {
        player->mana -= mana_cost;
        player->q_timer = 70;
        return 0;
    }
    return 1;
}

int skill_burst_aoe(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 100;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    if (player_aoe_attack(env, player, target, 2, 100 + 40*player->level, 0) == 0) {
        player->mana -= mana_cost;
        player->w_timer = 40;
        return 0;
    }
    return 1;
}

int skill_burst_aoe_stun(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 75;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    if (player_aoe_attack(env, player, target, 2, 0, 10 + 0.5*player->level) == 0) {
        player->mana -= mana_cost;
        player->e_timer = 50;
        return 0;
    }
    return 1;
}

int skill_tank_aoe_dot(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 5;
    if (player->mana < mana_cost)
        return 1;

    if (player_aoe_attack(env, player, player, 2, 25 + 2.0*player->level, 0) == 0) {
        player->mana -= mana_cost;
        return 0;
    }
    return 1;
}

// TODO: Fix
int skill_tank_self_heal(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 100;
    if (player->mana < mana_cost)
        return 1;

    if (heal(env, player, player, 400 + 125*player->level) == 0) {
        player->mana -= mana_cost;
        player->w_timer = 70;
        return 0;
    }
    return 1;
}

//Engages but doesnt push
int skill_tank_engage_aoe(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 50;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    if (move_near(env->map, player, target) == 0) {
        player->mana -= mana_cost;
        player->e_timer = 40;
        aoe_pull(env, player, 4, 2.0 + 0.1*player->level);
        return 0;
    }
    return 1;
}

int skill_carry_retreat_slow(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 25;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    int err = 1;
    for (int i = 0; i < 3; i++) {
        if (target == NULL || player->mana < mana_cost)
            return err;

        if (push(env, target, player, 3 + 0.1*player->level) == 0) {
            target->move_timer = 15;
            target->move_modifier = 0.5;
            player->mana -= mana_cost;
            player->q_timer = 40;
            err = 0;
        }
    }
    return err;
}

int skill_carry_slow_damage(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 150;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    if (attack(env, player, target, 50 + 20*player->level) == 0) {
        target->move_timer = 20 + player->level;
        target->move_modifier = 0.5;
        player->mana -= mana_cost;
        player->w_timer = 40;
        return 0;
    }
    return 1;
}

int skill_carry_aoe(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 100;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    if (player_aoe_attack(env, player, target, 2, 100 + 20*player->level, 0) == 0) {
        player->mana -= mana_cost;
        player->e_timer = 40;
        return 0;
    }
    return 1;
}

void step_creeps(MOBA* env) {
    // Spawn wave
    if (env->tick % 150 == 0) {
        for (int lane = 0; lane < 6; lane++) {
            for (int i = 0; i < 5; i++) {
                int creep_pid = CREEP_OFFSET + env->creep_idx;
                kill_entity(env->map, &env->entities[creep_pid]);
                spawn_creep(env, env->creep_idx, lane);
                env->creep_idx = (env->creep_idx + 1) % NUM_CREEPS;
            }
        }
    }
    for (int idx = 0; idx < NUM_CREEPS; idx++) {
        int pid = CREEP_OFFSET + idx;
        Entity* creep = &env->entities[pid];
        if (creep->pid == -1)
            continue;

        update_status(creep);
        update_cooldowns(creep);
        if (creep->stun_timer > 0)
            continue;

        creep_ai(env, creep);
    }
}

void step_neutrals(MOBA* env) {
    if (env->tick % 600 == 0) {
        for (int camp = 0; camp < 18; camp++) {
            for (int neut = 0; neut < 4; neut++) {
                spawn_neutral(env, 4*camp + neut);
            }
        }
    }
    for (int idx = 0; idx < NUM_NEUTRALS; idx++) {
        int pid = NEUTRAL_OFFSET + idx;
        Entity* neutral = &env->entities[pid];
        if (neutral->pid == -1)
            continue;

        update_status(neutral);
        update_cooldowns(neutral);
        if (neutral->stun_timer > 0)
            continue;

        neutral_ai(env, neutral);
    }
}

void step_towers(MOBA* env) {
    for (int idx = 0; idx < NUM_TOWERS; idx++) {
        int pid = TOWER_OFFSET + idx;
        Entity* tower = &env->entities[pid];
        if (tower->pid == -1)
            continue;

        /*
        if (tower->health < tower->max_health)
            tower->health += 10;
        if (tower->health > tower->max_health)
            tower->health = tower->max_health;
        */


        update_cooldowns(tower);
        if (tower->basic_attack_timer > 0)
            continue;

        if (env->tick % 3 == 0) { // Is this fast enough?
            scan_aoe(env, tower, TOWER_VISION, true, false, false, true, true);
            if (env->scanned_targets[tower->pid][0] != NULL) {
                float distance_to_first_scanned = l1_distance(tower->y, tower->x, env->scanned_targets[tower->pid][0]->y, env->scanned_targets[tower->pid][0]->x);
                if (distance_to_first_scanned > 12) {
                    printf("Tower %i at %f, %f, target %i at %f, %f, dist %f\n", tower->pid, tower->y, tower->x, env->scanned_targets[tower->pid][0]->pid, env->scanned_targets[tower->pid][0]->y, env->scanned_targets[tower->pid][0]->x, distance_to_first_scanned);
                }
            }

        }

        Entity* target = nearest_scanned_target(env, tower);
        if (target != NULL) 
            basic_attack(env, tower, target);
    }
}

void step_players(MOBA* env) {
    // Clear rewards
    for (int pid = 0; pid < NUM_PLAYERS; pid++) {
        Reward* reward = &env->reward_components[pid];
        reward->death = 0;
        reward->xp = 0;
        reward->distance = 0;
        reward->tower = 0;
    }

    for (int pid = 0; pid < NUM_PLAYERS; pid++) {
        Entity* player = &env->entities[pid];
        PlayerLog* log = &env->log[pid];
        Reward* reward = &env->reward_components[pid];
        // TODO: Is this needed?
        //if (rand() % 1024 == 0)
        //    spawn_player(env->map, player);

        if (player->mana < player->max_mana)
            player->mana += 2;
        if (player->mana > player->max_mana)
            player->mana = player->max_mana;
        if (player->health < player->max_health)
            player->health += 2;
        if (player->health > player->max_health)
            player->health = player->max_health;

        update_status(player);
        update_cooldowns(player);

        if (player->stun_timer > 0)
            continue;

        if (env->script_opponents && pid >= 5) {
            creep_ai(env, player);
            /*
            if (env->scanned_targets[pid][0] != NULL) {
                Entity* target = nearest_scanned_target(env, player);
                float dest_y = target->y;
                float dest_x = target->x;
                float dist = l1_distance(player->y, player->x, dest_y, dest_x);
                if (dist < 5) {
                    if (player->q_timer <= 0 && env->skills[pid][0](env, player, target) == 0) {
                        log->usage_q += 1;
                    } else if (player->w_timer <= 0 && env->skills[pid][1](env, player, target) == 0) {
                        log->usage_w += 1;
                    } else if (player->e_timer <= 0 && env->skills[pid][2](env, player, target) == 0) {
                        log->usage_e += 1;
                    }
                }
            }
            */
        } else {
            int (*actions)[6] = (int(*)[6])env->actions;
            //float vel_y = (actions[pid][0] > 0) ? 1 : -1;
            //float vel_x = (actions[pid][1] > 0) ? 1 : -1;
            float vel_y = actions[pid][0] / 300.0f;
            float vel_x = actions[pid][1] / 300.0f;
            float mag = sqrtf(vel_y*vel_y + vel_x*vel_x);
            if (mag > 1) {
                vel_y /= mag;
                vel_x /= mag;
            }

            int attack_target = actions[pid][2];
            bool use_q = actions[pid][3];
            bool use_w = actions[pid][4];
            bool use_e = actions[pid][5];

            if (attack_target == 1 || attack_target == 0) {
                // Scan everything
                scan_aoe(env, player, env->vision_range, true, false, false, false, false);
            } else if (attack_target == 2) {
                // Scan only heros and towers
                scan_aoe(env, player, env->vision_range, true, false, true, true, false);
            }

            Entity* target = NULL;
            // TODO: What is this logic here?
            if (env->scanned_targets[pid][0] != NULL)
                target = nearest_scanned_target(env, player);

            // TODO: Clean this mess
            if (use_q && player->q_timer <= 0 && env->skills[pid][0](env, player, target) == 0) {
                log->usage_q += 1;
            } else if (use_w && player->w_timer <= 0 && env->skills[pid][1](env, player, target) == 0) {
                log->usage_w += 1;
            } else if (use_e && player->e_timer <= 0 && env->skills[pid][2](env, player, target) == 0) {
                log->usage_e += 1;
            } else if (target != NULL && basic_attack(env, player, target)==0) {
                log->usage_auto += 1;
            }

            float dest_y = player->y + player->move_modifier*env->agent_speed*vel_y;
            float dest_x = player->x + player->move_modifier*env->agent_speed*vel_x;
            move_to(env->map, player, dest_y, dest_x);
        }

        float sum_reward = (
            reward->death +
            reward->xp +
            reward->distance +
            reward->tower
        );

        env->log[pid].reward_death = reward->death;
        env->log[pid].reward_xp = reward->xp;
        env->log[pid].reward_distance = reward->distance;
        env->log[pid].reward_tower = reward->tower;
        env->log[pid].episode_return += sum_reward;

        if (!env->script_opponents || pid < 5) {
            env->rewards[pid] = sum_reward;
        }
    }
}

unsigned char* read_file(char* filename) {
    FILE* file;
    unsigned char* array;
    file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    size_t file_bytes = ftell(file);
    fseek(file, 0, SEEK_SET);
    array = calloc(file_bytes, sizeof(unsigned char));
    if (array == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        return NULL;
    }
    size_t num_read = fread(array, 1, file_bytes, file);
    if (num_read != file_bytes) {
        perror("Error reading file");
        free(array);
        fclose(file);
        return NULL;
    }
    return array;
}

void init_moba(MOBA* env, unsigned char* game_map_npy) {
    env->obs_size = 2*env->vision_range + 1;
    env->creep_idx = 0;
    env->total_towers_taken = 0;
    env->total_levels_gained = 0;
    env->radiant_victories = 0;
    env->dire_victories = 0;

    env->entities = calloc(NUM_ENTITIES, sizeof(Entity));
    env->reward_components = calloc(NUM_PLAYERS, sizeof(Reward));

    env->map = (Map*)calloc(1, sizeof(Map));
    env->map->grid = calloc(128*128, sizeof(unsigned char));
    env->orig_grid = calloc(128*128, sizeof(unsigned char));
    memcpy(env->map->grid, game_map_npy, 128*128);
    memcpy(env->orig_grid, game_map_npy, 128*128);
    if (env->map->grid == NULL) {
        printf("Failed to load game map\n");
        exit(1);
    }
    env->map->width = 128;
    env->map->height = 128;
    env->map->pids = calloc(128*128, sizeof(int));
    for (int i = 0; i < 128*128; i++)
        env->map->pids[i] = -1;

    // Zero out scanned targets
    for (int i = 0; i < 256; i++) {
        env->scanned_targets[i][0] = NULL;
        env->scanned_targets[i][1] = NULL;
    }

    env->rng = (CachedRNG*)calloc(1, sizeof(CachedRNG));
    env->rng->rng_n = 10000;
    env->rng->rng_idx = 0;
    env->rng->rng = calloc(env->rng->rng_n, sizeof(float));
    for (int i = 0; i < env->rng->rng_n; i++)
        env->rng->rng[i] = -1+2*((float)rand())/(float)RAND_MAX;

    // Initialize Players
    Entity *player;
    for (int team = 0; team < 2; team++) {
        int spawn_y, spawn_x;
        if (team == 0) {
            spawn_y = 128 - 15;
            spawn_x = 12;
        } else {
            spawn_y = 15;
            spawn_x = 128 - 12;
        }

        for (int pid = team*5; pid < team*5 + 5; pid++) {
            player = &env->entities[pid];
            player->pid = pid;
            player->entity_type = ENTITY_PLAYER;
            player->team = team;
            player->spawn_y = spawn_y;
            player->spawn_x = spawn_x;
            player->x = 0;
            player->y = 0;
            player->move_speed = env->agent_speed;
            player->basic_attack_cd = 8;
            player->base_damage = 50;
            player->last_x = 0;
            player->last_y = 0;
        }

        int pid = 5*team;
        player = &env->entities[pid];
        player->pid = pid;
        player->entity_type = ENTITY_PLAYER;
        player->grid_id = RADIANT_SUPPORT + team*5;
        player->hero_type = 0;
        player->lane = 2 + 3*team;
        env->skills[pid][0] = skill_support_hook;
        env->skills[pid][1] = skill_support_aoe_heal;
        env->skills[pid][2] = skill_support_stun;
        player->base_health = 500;
        player->base_mana = 250;
        player->hp_gain_per_level = 100;
        player->mana_gain_per_level = 50;
        player->damage_gain_per_level = 10;
        player->lane = 2 + 3*team;

        pid = 5*team + 1;
        player = &env->entities[pid];
        player->pid = pid;
        player->entity_type = ENTITY_PLAYER;
        player->grid_id = RADIANT_ASSASSIN + team*5;
        player->hero_type = 1;
        player->lane = 2 + 3*team;
        env->skills[pid][0] = skill_assassin_aoe_minions;
        env->skills[pid][1] = skill_assassin_tp_damage;
        env->skills[pid][2] = skill_assassin_move_buff;
        player->base_health = 400;
        player->base_mana = 300;
        player->hp_gain_per_level = 100;
        player->mana_gain_per_level = 65;
        player->damage_gain_per_level = 10;
        player->lane = 1 + 3*team;

        pid = 5*team + 2;
        player = &env->entities[pid];
        player->pid = pid;
        player->entity_type = ENTITY_PLAYER;
        player->grid_id = RADIANT_BURST + team*5;
        player->hero_type = 2;
        player->lane = 1 + 3*team;
        env->skills[pid][0] = skill_burst_nuke;
        env->skills[pid][1] = skill_burst_aoe;
        env->skills[pid][2] = skill_burst_aoe_stun;
        player->base_health = 400;
        player->base_mana = 300;
        player->hp_gain_per_level = 75;
        player->mana_gain_per_level = 90;
        player->damage_gain_per_level = 10;
        player->lane = 1 + 3*team;

        pid = 5*team + 3;
        player = &env->entities[pid];
        player->pid = pid;
        player->entity_type = ENTITY_PLAYER;
        player->grid_id = RADIANT_TANK + team*5;
        player->hero_type = 3;
        player->lane = 3*team;
        env->skills[pid][0] = skill_tank_aoe_dot;
        env->skills[pid][1] = skill_tank_self_heal;
        env->skills[pid][2] = skill_tank_engage_aoe;
        player->base_health = 700;
        player->base_mana = 200;
        player->hp_gain_per_level = 150;
        player->mana_gain_per_level = 50;
        player->damage_gain_per_level = 15;
        player->lane = 3*team;

        pid = 5*team + 4;
        player = &env->entities[pid];
        player->pid = pid;
        player->entity_type = ENTITY_PLAYER;
        player->grid_id = RADIANT_CARRY + team*5;
        player->hero_type = 4;
        player->lane = 2 + 3*team;
        env->skills[pid][0] = skill_carry_retreat_slow;
        env->skills[pid][1] = skill_carry_slow_damage;
        env->skills[pid][2] = skill_carry_aoe;
        player->base_health = 300;
        player->base_mana = 250;
        player->hp_gain_per_level = 50;
        player->mana_gain_per_level = 50;
        player->damage_gain_per_level = 25;
        player->lane = 2 + 3*team;
    }

    Entity *tower;
    for (int idx = 0; idx < NUM_TOWERS; idx++) {
        int pid = TOWER_OFFSET + idx;
        tower = &env->entities[pid];
        tower->pid = pid;
        tower->entity_type = ENTITY_TOWER;
        tower->grid_id = TOWER;
        tower->basic_attack_cd = 5;
        tower->team = TOWER_TEAM[idx];
        tower->spawn_y = TOWER_Y[idx];
        tower->spawn_x = TOWER_X[idx];
        tower->x = 0;
        tower->y = 0;
        tower->max_health = TOWER_HEALTH[idx];
        tower->damage = TOWER_DAMAGE[idx];
        tower->tier = TOWER_TIER[idx];
        tower->xp_on_kill = 0;//800 * tower->tier;
    }

    int idx = 0;
    Entity* neutral;
    for (int camp = 0; camp < NUM_NEUTRALS/4; camp++) {
        // 4 neutrals per camp
        for (int i = 0; i < 4; i++) {
            int pid = NEUTRAL_OFFSET + idx;
            neutral = &env->entities[pid];
            // TODO: Consider initializing pid for start of game
            neutral->entity_type = ENTITY_NEUTRAL;
            neutral->grid_id = NEUTRAL;
            neutral->max_health = 500;
            neutral->team = 2;
            neutral->spawn_y = NEUTRAL_CAMP_Y[camp];
            neutral->spawn_x = NEUTRAL_CAMP_X[camp];
            neutral->x = 0;
            neutral->y = 0;
            neutral->xp_on_kill = 35;
            neutral->basic_attack_cd = 5;
            neutral->damage = 22;
            idx++;
        }
    }

    Entity* creep;
    for (int i = 0; i < NUM_CREEPS; i++) {
        creep = &env->entities[CREEP_OFFSET + i];
        creep->pid = -1;
        creep->x = 0;
        creep->y = 0;
    }
}

MOBA* allocate_moba(MOBA* env) {
    // TODO: Don't hardcode sizes
    int agents = (env->script_opponents) ? NUM_PLAYERS/2 : NUM_PLAYERS;
    env->observations = calloc(agents*(11*11*4 + 26), sizeof(unsigned char));
    env->actions = calloc(agents*6, sizeof(int));
    env->rewards = calloc(agents, sizeof(float));
    env->terminals = calloc(agents, sizeof(unsigned char));
    env->truncations = calloc(agents, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);

    unsigned char* game_map_npy = read_file("resources/moba/game_map.npy");
    env->ai_path_buffer = calloc(3*8*128*128, sizeof(int));
    env->ai_paths = calloc(128*128*128*128, sizeof(unsigned char));
    for (int i = 0; i < 128*128*128*128; i++) {
        env->ai_paths[i] = 255;
    }

    init_moba(env, game_map_npy);
    free(game_map_npy);
    return env;
}
 
void c_reset(MOBA* env) {
    //map->pids[:] = -1
    //randomize_tower_hp(env);
    
    env->tick = 0;

    // Reset scanned targets
    for (int i = 0; i < NUM_ENTITIES; i++) {
        Entity* entity = &env->entities[i];
        entity->target_pid = -1;
        env->scanned_targets[i][0] = NULL;
        kill_entity(env->map, entity);
    }

    Map* map = env->map;
    memcpy(map->grid, env->orig_grid, 128*128);
    for (int i = 0; i < 128*128; i++) {
        map->pids[i] = -1;
    }

    // Respawn towers
    for (int idx = 0; idx < NUM_TOWERS; idx++) {
        int pid = TOWER_OFFSET + idx;
        Entity* tower = &env->entities[pid];
        tower->target_pid = -1;
        tower->pid = pid;
        tower->health = tower->max_health;
        tower->basic_attack_timer = 0;
        tower->x = 0;
        tower->y = 0;
        int adr = map_offset(map, tower->spawn_y, tower->spawn_x);
        map->grid[adr] = EMPTY;
        int success = move_to(env->map, tower, tower->spawn_y, tower->spawn_x);
        if (success == 1) {
            printf("Failed to move tower %i to %f, %f\n", pid, tower->spawn_y, tower->spawn_x);
            printf("Currently occupied by %i\n", map->pids[adr]);
            exit(1);
        }
        tower->last_x = tower->x;
        tower->last_y = tower->y;
    }

    Entity* rad = &env->entities[205];
    int adr = map_offset(env->map, rad->y, rad->x);
    if (rad->health > 0 && env->map->pids[adr] != 205) {
        printf("glitch state\n");
    }

    // Respawn agents
    for (int i = 0; i < NUM_PLAYERS; i++) {
        env->log[i] = (PlayerLog){0};
        Entity* player = &env->entities[i];
        player->pid = i;
        player->target_pid = -1;
        player->xp = 0;
        player->level = 1;
        //player->x = 0;
        //player->y = 0;
        spawn_player(env->map, player);
    }

    rad = &env->entities[205];
    adr = map_offset(env->map, rad->y, rad->x);
    if (rad->health > 0 && env->map->pids[adr] != 205) {
        printf("glitch state\n");
    }

    compute_observations(env);
}

void c_step(MOBA* env) {
    for (int pid = 0; pid < NUM_ENTITIES; pid++) {
        Entity* entity = &env->entities[pid];
        entity->target_pid = -1;
        entity->attack_aoe = 0;
        entity->last_x = entity->x;
        entity->last_y = entity->y;
        entity->is_hit = 0;
    }

    step_neutrals(env);
    step_creeps(env);
    step_towers(env);
    step_players(env);
    env->tick += 1;

    Entity* rad = &env->entities[205];
    int adr = map_offset(env->map, rad->y, rad->x);
    if (rad->health > 0 && env->map->pids[adr] != 205) {
        printf("glitch state\n");
    }

    int radiant_pid = TOWER_OFFSET + 23;
    int dire_pid = TOWER_OFFSET + 22;

    bool do_reset = false;
    float radiant_victory = 0;
    float dire_victory = 0;
    Entity* ancient = &env->entities[dire_pid];
    if (ancient->health <= 0 || ancient->pid == -1) {
        do_reset = true;
        radiant_victory = 1;
        /*
        printf("Radiant victory\n");
        //Print tower states
        for (int i = 0; i < NUM_TOWERS; i++) {
            Entity* tower = &env->entities[TOWER_OFFSET + i];
            printf("Tower %i (%i): %f, %f, %f, %i\n", i, tower->pid, tower->health, tower->y, tower->x, tower->team);
            int y = tower->y;
            int x = tower->x;
            int adr = map_offset(env->map, y, x);
            int tile = env->map->grid[adr];
            int pid = env->map->pids[adr];
            printf("\ttile %i, pid %i\n", tile, pid);
        }
        //Print levels and pos
        for (int i = 0; i < NUM_PLAYERS; i++) {
            Entity* player = &env->entities[i];
            printf("Player %i (%i): %i, %f, %f, %i\n", i, player->pid, player->level, player->y, player->x, player->team);
            int y = player->y;
            int x = player->x;
            int adr = map_offset(env->map, y, x);
            int tile = env->map->grid[adr];
            int pid = env->map->pids[adr];
            printf("\ttile %i, pid %i\n", tile, pid);
        }
        printf("Tick %i", env->tick);
        exit(0);
        */
    }

    ancient = &env->entities[radiant_pid];
    if (ancient->health <= 0 || ancient->pid == -1) {
        do_reset = true;
        dire_victory = 1;
    }

    if (!do_reset && env->should_reset) {
        printf("We should have reset but didn't\n");
        printf("Radiant pid: %i\n", radiant_pid);
        Entity* radiant = &env->entities[radiant_pid];
        printf("Radiant %i: %f, %f, %f\n", radiant->pid, radiant->y, radiant->x, radiant->health);
        exit(0);
    }

    if (do_reset) {
        env->should_reset = false;
        Log log = {0};
        log.episode_length = env->tick;
        log.radiant_victory = radiant_victory;
        log.dire_victory = dire_victory;
        for (int i = 0; i < 5; i++) {
            log.radiant_level += env->entities[i].level / 5.0f;
        }
        for (int i = 5; i < 10; i++) {
            log.dire_level += env->entities[i].level / 5.0f;
        }
        for (int i = 0; i < NUM_TOWERS; i++) {
            Entity* tower = &env->entities[TOWER_OFFSET + i];
            if (TOWER_TEAM[i] == 0) {
                log.radiant_towers_alive += tower->health > 0;
            } else {
                log.dire_towers_alive += tower->health > 0;
            }
        }
        log.radiant_support = env->log[0];
        log.radiant_assassin = env->log[1];
        log.radiant_burst = env->log[2];
        log.radiant_tank = env->log[3];
        log.radiant_carry = env->log[4];
        log.dire_support = env->log[5];
        log.dire_assassin = env->log[6];
        log.dire_burst = env->log[7];
        log.dire_tank = env->log[8];
        log.dire_carry = env->log[9];
        add_log(env->log_buffer, &log);
        if (do_reset) {
            c_reset(env);
        }
    }
    compute_observations(env);
}

// Raylib client
Color COLORS[] = {
    (Color){6, 24, 24, 255},     // Empty
    (Color){0, 178, 178, 255},   // Wall
    (Color){255, 165, 0, 255},   // Tower
    (Color){0, 0, 128, 255},     // Radiant Creep
    (Color){128, 0, 0, 255},     // Dire Creep
    (Color){128, 128, 128, 255}, // Neutral
    (Color){0, 0, 255, 255},     // Radiant Support
    (Color){0, 0, 255, 255},     // Radiant Assassin
    (Color){0, 0, 255, 255},     // Radiant Burst
    (Color){0, 0, 255, 255},     // Radiant Tank
    (Color){0, 0, 255, 255},     // Radiant Carry
    (Color){255, 0, 0, 255},     // Dire Support
    (Color){255, 0, 0, 255},     // Dire Assassin
    (Color){255, 0, 0, 255},     // Dire Burst
    (Color){255, 0, 0, 255},     // Dire Tank
    (Color){255, 0, 0, 255},     // Dire Carry
};
 
// High-level map overview
typedef struct {
    int cell_size;
    int width;
    int height;
} MapRenderer;

MapRenderer* init_map_renderer(int cell_size, int width, int height) {
    MapRenderer* renderer = (MapRenderer*)malloc(sizeof(MapRenderer));
    renderer->cell_size = cell_size;
    renderer->width = width;
    renderer->height = height;
    InitWindow(width*cell_size, height*cell_size, "Puffer MOBA");
    SetTargetFPS(10);
    return renderer;
}

void close_map_renderer(MapRenderer* renderer) {
    CloseWindow();
    free(renderer);
}

void render_map(MapRenderer* renderer, MOBA* env) {
    BeginDrawing();
    ClearBackground(COLORS[0]);
    int sz = renderer->cell_size;
    for (int y = 0; y < renderer->height; y++) {
        for (int x = 0; x < renderer->width; x++){
            int adr = map_offset(env->map, y, x);
            int tile = env->map->grid[adr];
            if (tile != EMPTY)
                DrawRectangle(x*sz, y*sz, sz, sz, COLORS[tile]);
        }
    }
    DrawText("Reinforcement learned MOBA agents running in your browswer!", 10, 10, 20, COLORS[8]);
    DrawText("Written in pure C by @jsuarez5341. Star it on GitHub/pufferai/pufferlib to support my work!", 10, 40, 20, COLORS[8]);
    EndDrawing();
}

// Player client view
typedef struct {
    int cell_size;
    int width;
    int height;
    Camera2D camera;
    Rectangle asset_map[16];
    Rectangle stun_uv;
    Rectangle slow_uv;
    Rectangle speed_uv;
    Texture2D game_map;
    Texture2D puffer;
    Image shader_background;
    Texture2D shader_canvas;
    Shader shader;
    float shader_x;
    float shader_y;
    double shader_start_seconds;
    float shader_seconds;
    int shader_resolution_loc;
    float shader_resolution[3];
    Shader bloom_shader;
    float shader_camera_x;
    float shader_camera_y;
    float shader_time;
    int shader_texture1;
    float last_click_x;	
    float last_click_y;
    int render_entities[128*128];
    int human_player;
} GameRenderer;

GameRenderer* init_game_renderer(int cell_size, int width, int height) {
    GameRenderer* renderer = (GameRenderer*)calloc(1, sizeof(GameRenderer));
    renderer->cell_size = cell_size;
    renderer->width = width;
    renderer->height = height;

    InitWindow(width*cell_size, height*cell_size, "Puffer MOBA");
    SetTargetFPS(60);

    Rectangle asset_map[] = {
        (Rectangle){0, 0, 0, 0},
        (Rectangle){0, 0, 0, 0},
        (Rectangle){384, 384, 128, 128},
        (Rectangle){384, 0, 128, 128},
        (Rectangle){256, 0, 128, 128},
        (Rectangle){384, 128, 128, 128},
        (Rectangle){256, 256, 128, 128},
        (Rectangle){384, 256, 128, 128},
        (Rectangle){128, 256, 128, 128},
        (Rectangle){0, 256, 128, 128},
        (Rectangle){0, 384, 128, 128},
        (Rectangle){256, 256, 128, 128},
        (Rectangle){384, 256, 128, 128},
        (Rectangle){128, 256, 128, 128},
        (Rectangle){0, 256, 128, 128},
        (Rectangle){0, 384, 128, 128},
    };
    memcpy(renderer->asset_map, asset_map, sizeof(asset_map));

    renderer->stun_uv = (Rectangle){0, 128, 128, 128};
    renderer->slow_uv = (Rectangle){128, 128, 128, 128};
    renderer->speed_uv = (Rectangle){256, 128, 128, 128};

    renderer->game_map = LoadTexture("resources/moba/dota_map.png");
    renderer->puffer = LoadTexture("resources/moba/moba_assets.png");
    renderer->shader_background = GenImageColor(2560, 1440, (Color){0, 0, 0, 255});
    renderer->shader_canvas = LoadTextureFromImage(renderer->shader_background);
    renderer->shader = LoadShader(0, TextFormat("resources/moba/map_shader_%i.fs", GLSL_VERSION));
    renderer->bloom_shader = LoadShader(0, TextFormat("resources/moba/bloom_shader_%i.fs", GLSL_VERSION));

    // TODO: These should be int locs?
    renderer->shader_camera_x = GetShaderLocation(renderer->shader, "camera_x");
    renderer->shader_camera_y = GetShaderLocation(renderer->shader, "camera_y");
    renderer->shader_time = GetShaderLocation(renderer->shader, "time");
    renderer->shader_texture1 = GetShaderLocation(renderer->shader, "texture1");
    renderer->shader_resolution_loc = GetShaderLocation(renderer->shader, "resolution");
    struct timespec time_spec;
    clock_gettime(CLOCK_REALTIME, &time_spec);
    renderer->shader_start_seconds = time_spec.tv_sec;
 
    renderer->camera = (Camera2D){0};
    renderer->camera.target = (Vector2){0.0, 0.0};
    // TODO: Init this?
    //renderer->camera.offset = (Vector2){GetScreenWidth()/2.0f, GetScreenHeight()/2.0f};
    renderer->camera.rotation = 0.0f;
    renderer->camera.zoom = 1.0f;

    renderer->human_player = 1;

    // Init last clicks
    renderer->last_click_x = -1;
    renderer->last_click_y = -1;
    return renderer;
}

//def render(self, grid, pids, entities, obs_players, actions, discretize, frames):
#define FRAMES 12

void draw_bars(Entity* entity, int x, int y, int width, int height, bool draw_text) {
    float health_bar = entity->health / entity->max_health;
    float mana_bar = entity->mana / entity->max_mana;
    if (entity->max_health == 0) {
        health_bar = 2;
    }
    if (entity->max_mana == 0) {
        mana_bar = 2;
    }
    DrawRectangle(x, y, width, height, RED);
    DrawRectangle(x, y, width*health_bar, height, GREEN);

    if (entity->entity_type == 0) {
        DrawRectangle(x, y - height - 2, width, height, RED);
        DrawRectangle(x, y - height - 2, width*mana_bar, height, (Color){0, 255, 255, 255});
    }

    Color color = (entity->team == 0) ? (Color){0, 255, 255, 255} : (Color){255, 0, 0, 255};
    if (draw_text) {
        int health = entity->health;
        int mana = entity->mana;
        int max_health = entity->max_health;
        int max_mana = entity->max_mana;
        DrawText(TextFormat("Health: %i/%i", health, max_health), x+8, y+2, 20, (Color){255, 255, 255, 255});
        DrawText(TextFormat("Mana: %i/%i", mana, max_mana), x+8, y+2 - height - 2, 20, (Color){255, 255, 255, 255});
        DrawText(TextFormat("Experience: %i", entity->xp), x+8, y - 2*height - 4, 20, (Color){255, 255, 255, 255});
    } else if (entity->entity_type == 0) {
        DrawText(TextFormat("Level: %i", entity->level), x+4, y -2*height - 12, 12, color);
    }
}

int render_game(GameRenderer* renderer, MOBA* env, int frame) {
    Map* map = env->map;
    Entity* my_player = &env->entities[renderer->human_player];
    int ts = renderer->cell_size;

    renderer->width = GetScreenWidth() / ts;
    renderer->height = GetScreenHeight() / ts;
    renderer->shader_resolution[0] = renderer->width;
    renderer->shader_resolution[1] = renderer->height;

    float tick_frac = (float)frame / (float)FRAMES;

    float fmain_r = my_player->last_y + tick_frac*(my_player->y - my_player->last_y);
    float fmain_c = my_player->last_x + tick_frac*(my_player->x - my_player->last_x);

    renderer->camera.target.x = (int)((fmain_c - renderer->width/2) * ts);
    renderer->camera.target.y = (int)((fmain_r - renderer->height/2) * ts);

    int main_r = fmain_r;
    int main_c = fmain_c;

    int r_min = main_r - renderer->height/2 - 1;
    int r_max = main_r + renderer->height/2 + 1;
    int c_min = main_c - renderer->width/2 - 1;
    int c_max = main_c + renderer->width/2 + 1;

    Vector2 pos = GetMousePosition();
    float raw_mouse_x = pos.x + renderer->camera.target.x;
    float raw_mouse_y = pos.y + renderer->camera.target.y;

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        renderer->last_click_x = raw_mouse_x / ts;
        renderer->last_click_y = raw_mouse_y / ts;
    }

    int human = renderer->human_player;
    bool HUMAN_CONTROL = IsKeyDown(KEY_LEFT_SHIFT);
    int (*actions)[6] = (int(*)[6])env->actions;

    // Clears so as to not let the nn spam actions
    if (HUMAN_CONTROL && frame % 12 == 0) {
        actions[human][0] = 0;
        actions[human][1] = 0;
        actions[human][2] = 0;
        actions[human][3] = 0;
        actions[human][4] = 0;
        actions[human][5] = 0;
    }

    // TODO: better way to null clicks?
    if (renderer->last_click_x != -1 && renderer->last_click_y != -1) {
        float dest_x = renderer->last_click_x;
        float dest_y = renderer->last_click_y;
        float dy = dest_y - my_player->y;
        float dx = dest_x - my_player->x;

        float mag = sqrtf(dy*dy + dx*dx);
        if (mag < 1) {
            renderer->last_click_x = -1;
            renderer->last_click_y = -1;
        }
       
        if (HUMAN_CONTROL) {
            actions[human][0] = 300*dy;
            actions[human][1] = 300*dx;
        }
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (HUMAN_CONTROL) {
        if (IsKeyDown(KEY_Q) || IsKeyPressed(KEY_Q)) {
            actions[human][3] = 1;
        }
        if (IsKeyDown(KEY_W) || IsKeyPressed(KEY_W)) {
            actions[human][4] = 1;
        }
        if (IsKeyDown(KEY_E) || IsKeyPressed(KEY_E)) {
            actions[human][5] = 1;
        }
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            actions[human][2] = 2; // Target heroes
        }
    }
    // Num keys toggle selected player
    int num_pressed = GetKeyPressed();
    if (num_pressed > KEY_ZERO && num_pressed <= KEY_NINE) {
        renderer->human_player = num_pressed - KEY_ZERO - 1;
    } else if (num_pressed == KEY_ZERO) {
        renderer->human_player = 9;
    }

    BeginDrawing();
    ClearBackground(COLORS[0]);

    // Main environment shader
    BeginShaderMode(renderer->shader);
    renderer->shader_y = (fmain_r - renderer->height/2) / 128;
    renderer->shader_x = (fmain_c - renderer->width/2) / 128;
    struct timespec time_spec;
    clock_gettime(CLOCK_REALTIME, &time_spec);
    renderer->shader_seconds = time_spec.tv_sec - renderer->shader_start_seconds + time_spec.tv_nsec / 1e9;
    SetShaderValue(renderer->shader, renderer->shader_camera_x, &renderer->shader_x, SHADER_UNIFORM_FLOAT);
    SetShaderValue(renderer->shader, renderer->shader_camera_y, &renderer->shader_y, SHADER_UNIFORM_FLOAT);
    SetShaderValue(renderer->shader, renderer->shader_time, &renderer->shader_seconds, SHADER_UNIFORM_FLOAT);
    SetShaderValue(renderer->shader, renderer->shader_resolution_loc, renderer->shader_resolution, SHADER_UNIFORM_VEC3);
    SetShaderValueTexture(renderer->shader, renderer->shader_texture1, renderer->game_map);
    DrawTexture(renderer->shader_canvas, 0, 0, WHITE);
    EndShaderMode();

    BeginMode2D(renderer->camera);

    int render_idx = 0;
    for (int y = r_min; y < r_max+1; y++) {
        for (int x = c_min; x < c_max+1; x++) {
            if (y < 0 || y >= 128 || x < 0 || x >= 128) {
                continue;
            }

            int adr = map_offset(map, y, x);
            int pid = map->pids[adr];
            //if (pid != -1) {
            //    DrawRectangle(x*ts, y*ts, ts, ts, RED);
            //}

            unsigned char tile = map->grid[adr];
            if (tile == EMPTY || tile == WALL) {
                continue;
            }
       
            pid = map->pids[adr];
            if (pid == -1) {
                DrawRectangle(x*ts, y*ts, ts, ts, RED);
            }

            renderer->render_entities[render_idx] = pid;
            render_idx++;
        }
    }

    // Targeting overlays
    for (int i = 0; i < render_idx; i++) {
        int pid = renderer->render_entities[i];
        if (pid == -1) {
            continue;
        }

        Entity* entity = &env->entities[pid];
        int target_pid = entity->target_pid;
        if (target_pid == -1) {
            continue;
        }

        Entity* target = &env->entities[target_pid];
        float entity_x = entity->last_x + tick_frac*(entity->x - entity->last_x);
        float entity_y = entity->last_y + tick_frac*(entity->y - entity->last_y);
        float target_x = target->last_x + tick_frac*(target->x - target->last_x);
        float target_y = target->last_y + tick_frac*(target->y - target->last_y);

        Color base;
        Color accent;
        if (entity->team == 0) {
            base = (Color){0, 128, 128, 255};
            accent = (Color){0, 255, 255, 255};
        } else if (entity->team == 1) {
            base = (Color){128, 0, 0, 255};
            accent = (Color){255, 0, 0, 255};
        } else {
            base = (Color){128, 128, 128, 255};
            accent = (Color){255, 255, 255, 255};
        }

        int target_px = target_x*ts + ts/2;
        int target_py = target_y*ts + ts/2;
        int entity_px = entity_x*ts + ts/2;
        int entity_py = entity_y*ts + ts/2;

        if (entity->attack_aoe == 0) {
            Vector2 line_start = (Vector2){entity_px, entity_py};
            Vector2 line_end = (Vector2){target_px, target_py};
            DrawLineEx(line_start, line_end, ts/16, accent);
        } else {
            int radius = entity->attack_aoe*ts;
            DrawRectangle(target_px - radius, target_py - radius,
                2*radius, 2*radius, base);
            Rectangle rec = (Rectangle){target_px - radius,
                target_py - radius, 2*radius, 2*radius};
            DrawRectangleLinesEx(rec, ts/8, accent);
        }
    }

    // Entity renders
    for (int i = 0; i < render_idx; i++) {
        Color tint = (Color){255, 255, 255, 255};

        int pid = renderer->render_entities[i];
        if (pid == -1) {
            continue;
        }
        Entity* entity = &env->entities[pid];
        int y = entity->y;
        int x = entity->x;

        float entity_x = entity->last_x + tick_frac*(entity->x - entity->last_x);
        float entity_y = entity->last_y + tick_frac*(entity->y - entity->last_y);
        int tx = entity_x*ts;
        int ty = entity_y*ts;
        draw_bars(entity, tx, ty-8, ts, 4, false);

        int adr = map_offset(map, y, x);
        int tile = map->grid[adr];

        // TODO: Might need a vector type
        Rectangle source_rect = renderer->asset_map[tile];
        Rectangle dest_rect = (Rectangle){tx, ty, ts, ts};

        if (entity->is_hit) {
            BeginShaderMode(renderer->bloom_shader);
        }
        Vector2 origin = (Vector2){0, 0};
        DrawTexturePro(renderer->puffer, source_rect, dest_rect, origin, 0, tint);
        if (entity->is_hit) {
            EndShaderMode();
        }

        // Draw status icons
        if (entity->stun_timer > 0) {
            DrawTexturePro(renderer->puffer, renderer->stun_uv, dest_rect, origin, 0, tint);
        }
        if (entity->move_timer > 0) {
            if (entity->move_modifier < 0) {
                DrawTexturePro(renderer->puffer, renderer->slow_uv, dest_rect, origin, 0, tint);
            }
            if (entity->move_modifier > 0) {
                DrawTexturePro(renderer->puffer, renderer->speed_uv, dest_rect, origin, 0, tint);
            }
        }
    }

    //DrawCircle(ts*mouse_x + ts/2, ts*mouse_y + ts/8, ts/8, WHITE);
    EndMode2D();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        DrawCircle(pos.x, pos.y, ts/8, RED);
    }

    // Draw HUD
    Entity* player = &env->entities[human];
    DrawFPS(10, 10);

    float hud_y = renderer->height*ts - 2*ts;
    draw_bars(player, 2*ts, hud_y, 10*ts, 24, true);

    Color off_color = (Color){255, 255, 255, 255};
    Color on_color = (player->team == 0) ? (Color){0, 255, 255, 255} : (Color){255, 0, 0, 255};

    Color q_color = (actions[human][3]) ? on_color : off_color;
    Color w_color = (actions[human][4]) ? on_color : off_color;
    Color e_color = (actions[human][5]) ? on_color : off_color;

    int q_cd = player->q_timer;
    int w_cd = player->w_timer;
    int e_cd = player->e_timer;

    DrawText(TextFormat("Q: %i", q_cd), 13*ts, hud_y - 20, 40, q_color);
    DrawText(TextFormat("W: %i", w_cd), 17*ts, hud_y - 20, 40, w_color);
    DrawText(TextFormat("E: %i", e_cd), 21*ts, hud_y - 20, 40, e_color);
    DrawText(TextFormat("Stun: %i", player->stun_timer), 25*ts, hud_y - 20, 20, (player->stun_timer > 0) ? on_color : off_color);
    DrawText(TextFormat("Move: %i", player->move_timer), 25*ts, hud_y, 20, (player->move_timer > 0) ? on_color : off_color);

    EndDrawing();
    return 0;
}

void close_game_renderer(GameRenderer* renderer) {
    UnloadImage(renderer->shader_background);
    UnloadShader(renderer->shader);
    UnloadShader(renderer->bloom_shader);
    CloseWindow();
    free(renderer);
}


