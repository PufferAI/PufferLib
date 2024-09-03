// Incremental port of Puffer Moba to C. Be careful to add semicolons and avoid leftover cython syntax
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include "raylib.h"

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
#define CREEP_VISION 5
#define NEUTRAL_VISION 3

#define ENTITY_PLAYER 0
#define ENTITY_CREEP 1
#define ENTITY_NEUTRAL 2
#define ENTITY_TOWER 3

#define XP_RANGE 7
#define MAX_USES 2000000000

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
const float WAYPOINTS[][20][2] = {{{96.26153846153846, 14.04615384615385}, {14.292307692307695, 93.61538461538461}, {16.07692307692308, 58.23076923076923}, {16.01538461538462, 43.4}, {17.584615384615383, 32.10769230769231}, {19.123076923076923, 24.93846153846154}, {20.446153846153848, 22.96923076923077}, {25.307692307692307, 21.06153846153846}, {30.47692307692308, 20.200000000000003}, {35.892307692307696, 19.4}, {43.52307692307693, 18.47692307692308}, {50.723076923076924, 18.04615384615385}, {94.66153846153847, 20.50769230769231}, {105.94615384615385, 27.676923076923075}}, {{99.52307692307693, 26.47692307692308}, {27.646153846153844, 98.66153846153847}, {41.12307692307692, 86.04615384615384}, {57.49230769230769, 71.33846153846154}, {62.23076923076923, 66.96923076923076}, {66.35384615384615, 64.07692307692308}, {72.01538461538462, 60.2}, {84.87692307692308, 51.892307692307696}, {99.4, 34.66153846153846}, {105.94615384615385, 28.169230769230772}}, {{112.01538461538462, 36.93846153846154}, {32.50769230769231, 112.01538461538462}, {68.6923076923077, 116.2}, {80.75384615384615, 113.73846153846154}, {90.53846153846155, 113.55384615384615}, {95.98461538461538, 113.21538461538461}, {100.93846153846154, 111.49230769230769}, {108.07692307692308, 109.06153846153846}, {111.36923076923077, 107.18461538461538}, {114.66153846153847, 101.73846153846154}, {111.76923076923077, 90.53846153846155}, {113.73846153846154, 39.4}, {106.43846153846154, 28.169230769230772}}, {{20.446153846153848, 89.36923076923077}, {94.66153846153847, 20.630769230769232}, {50.6, 18.784615384615385}, {24.50769230769231, 22.292307692307695}, {18.815384615384616, 22.815384615384616}, {17.03076923076923, 27.52307692307692}, {15.95384615384615, 35.03076923076923}, {14.415384615384617, 93.61538461538461}, {21.99230769230769, 103.64615384615385}}, {{37.43076923076923, 96.50769230769231}, {99.27692307692308, 34.53846153846154}, {84.38461538461539, 51.707692307692305}, {71.64615384615385, 59.83076923076923}, {66.1076923076923, 63.83076923076923}, {61.98461538461538, 66.72307692307692}, {57.246153846153845, 70.84615384615384}, {27.52307692307692, 98.53846153846155}, {22.48461538461538, 103.64615384615385}}, {{36.93846153846154, 113.24615384615385}, {113.61538461538461, 39.4}, {114.6, 62.292307692307695}, {114.6, 83.64615384615385}, {109.8, 90.66153846153847}, {112.01538461538462, 94.87692307692308}, {112.2923076923077, 104.26153846153846}, {107.83076923076922, 109.3076923076923}, {105.86153846153846, 112.23076923076923}, {72.96923076923076, 114.44615384615385}, {32.50769230769231, 111.8923076923077}, {22.48461538461538, 104.13846153846154}}};

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

struct MOBA {
    int num_agents;
    int num_creeps;
    int num_neutrals;
    int num_towers;
    int num_entities;
    int vision_range;
    float agent_speed;
    bool discretize;
    int obs_size;
    int creep_idx;
    int tick;

    Map* map;
    unsigned char* orig_grid;
    unsigned char* ai_paths;
    unsigned char* observations_map;
    unsigned char* observations_extra;
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

    // MAX_ENTITIES x MAX_SCANNED_TARGETS
    Entity* scanned_targets[256][121];
    skill skills[10][3];

    Reward* rewards;
    float* sum_rewards;
    float* norm_rewards;

    CachedRNG *rng;
};

static inline int ai_offset(int y_dst, int x_dst, int y_src, int x_src) {
    return y_dst*128*128*128 + x_dst*128*128 + y_src*128 + x_src;
}

void free_moba(MOBA* env) {
    free(env->rewards);
    free(env->sum_rewards);
    free(env->norm_rewards);
    free(env->ai_paths);
    free(env->map->pids);
    free(env->map->grid);
    free(env->orig_grid);
    free(env->map);
    free(env->observations_map);
    free(env->observations_extra);
    free(env->actions);
    free(env->entities);
    free(env->rng->rng);
    free(env->rng);
    free(env);
}
 
void compute_observations(MOBA* env) {
    // Does this copy?
    unsigned char (*obs_map)[11][11][4] = (unsigned char(*)[11][11][4])env->observations_map;
    unsigned char (*obs_extra)[26] = (unsigned char(*)[26])env->observations_extra;

    // TODO: Zero out
    //self.observations_map[:] = 0

    // Probably safe to not clear this
    //self.observations_extra[:] = 0

    int vis = env->vision_range;
    Map* map = env->map;

    for (int pid = 0; pid < env->num_agents; pid++) {
        Entity* player = &env->entities[pid];
        Reward* reward = &env->rewards[pid];

        int y = player->y;
        int x = player->x;

        // TODO: Add bounds debug checks asserts
        obs_extra[pid][0] = 2*x;
        obs_extra[pid][1] = 2*y;
        obs_extra[pid][2] = 255*player->level/30.0;
        obs_extra[pid][3] = 255*player->health/player->max_health;
        obs_extra[pid][4] = 255*player->mana/player->max_mana;
        obs_extra[pid][5] = player->damage;
        obs_extra[pid][6] = 100*player->move_speed;
        obs_extra[pid][7] = player->move_modifier*100;
        obs_extra[pid][8] = 2*player->stun_timer;
        obs_extra[pid][9] = 2*player->move_timer;
        obs_extra[pid][10] = 2*player->q_timer;
        obs_extra[pid][11] = 2*player->w_timer;
        obs_extra[pid][12] = 2*player->e_timer;
        obs_extra[pid][13] = 50*player->basic_attack_timer;
        obs_extra[pid][14] = 50*player->basic_attack_cd;
        obs_extra[pid][15] = 255*player->is_hit;
        obs_extra[pid][16] = 255*player->team;
        obs_extra[pid][17 + player->hero_type] = 255;

        // Assumes scaled between -1 and 1, else overflows
        obs_extra[pid][22] = 127*reward->death + 128;
        obs_extra[pid][23] = 25*reward->xp;
        obs_extra[pid][24] = 127*reward->distance + 128;
        obs_extra[pid][25] = 70*reward->tower;

        for (int dy = -vis; dy <= vis; dy++) {
            for (int dx = -vis; dx <= vis; dx++) {
                int xx = x + dx;
                int yy = y + dy;
                int ob_x = dx + vis;
                int ob_y = dy + vis;

                int adr = map_offset(map, yy, xx);
                obs_map[pid][ob_y][ob_x][0] = map->grid[adr];
                int target_pid = env->map->pids[adr];
                if (target_pid == -1)
                    continue;

                Entity* target = &env->entities[target_pid];
                obs_map[pid][ob_y][ob_x][1] = 255*target->health/target->max_health;
                obs_map[pid][ob_y][ob_x][2] = 255*target->mana/target->max_mana;
                obs_map[pid][ob_y][ob_x][3] = target->level/30.0;
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
    return i + 1;
}

Reward* get_reward(MOBA* env, int pid) {
    return &env->rewards[pid];
}

static inline int creep_offset(MOBA* moba) {
    return moba->num_agents;
}

static inline int neutral_offset(MOBA* moba) {
    return moba->num_agents + moba->num_creeps;
}

static inline int tower_offset(MOBA* moba) {
    return moba->num_agents + moba->num_creeps + moba->num_neutrals;
}

int move_to(Map* map, Entity* player, float dest_y, float dest_x) {
    int src = map_offset(map, (int)player->y, (int)player->x);
    int dst = map_offset(map, (int)dest_y, (int)dest_x);

    if (map->grid[dst] != EMPTY && map->pids[dst] != player->pid)
        return 1;

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
    int adr = map_offset(map, (int)entity->y, (int)entity->x);
    map->grid[adr] = EMPTY;
    map->pids[adr] = -1;
    entity->pid = -1;
    entity->x = 0;
    entity->y = 0;
}

void respawn_player(Map* map, Entity* entity) {
    int pid = entity->pid;
    kill_entity(map, entity);
    entity->pid = pid;

    entity->max_health = entity->base_health;
    entity->max_mana = entity->base_mana;
    entity->health = entity->max_health;
    entity->mana = entity->max_mana;
    entity->damage = entity->base_damage;
    entity->basic_attack_timer = 0;
    entity->move_modifier = 0;
    entity->move_timer = 0;
    entity->stun_timer = 0;
    
    // TODO: Cache noise?
    // Also.. technically can infinite loop?
    bool valid_pos = false;
    int y, x;
    while (!valid_pos) {
        y = entity->spawn_y + rand()%15 - 7;
        x = entity->spawn_x + rand()%15 - 7;
        valid_pos = map->grid[map_offset(map, y, x)] == EMPTY;
    }
    move_to(map, entity, y, x);
}

int attack(MOBA* env, Entity* player, Entity* target, float damage) {
    if (target->pid == -1 || target->team == player->team)
        return 1;

    player->target_pid = target->pid;
    target->is_hit = 1;

    if (damage < target->health) {
        player->damage_dealt += damage;
        target->damage_received += damage;
        target->health -= damage;
        return 0;
    }

    player->damage_dealt += target->health;
    target->damage_received += target->health;
    target->health = 0;

    int target_type = target->entity_type;
    if (target_type == ENTITY_PLAYER) {
        env->rewards[target->pid].death = env->reward_death;
        player->heros_killed += 1;
        target->deaths += 1;
        respawn_player(env->map, target);
    } else if (target_type == ENTITY_CREEP) {
        player->creeps_killed += 1;
        kill_entity(env->map, target);
    } else if (target_type == ENTITY_NEUTRAL) {
        player->neutrals_killed += 1;
        kill_entity(env->map, target);
    } else if (target_type == ENTITY_TOWER) {
        player->towers_killed += 1;
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
        env->rewards[first_player_on_team + i].xp = xp*env->reward_xp;

        int level = ally->level;
        ally->level = calc_level(env, ally->xp);
        if (ally->level > level)
            env->total_levels_gained += 1;

        ally->max_health = ally->base_health + ally->level*ally->hp_gain_per_level;
        ally->max_mana = ally->base_mana + ally->level*ally->mana_gain_per_level;
        ally->damage = ally->base_damage + ally->level*ally->damage_gain_per_level;
    }

    if (target->entity_type == ENTITY_TOWER) {
        env->rewards[player->pid].tower = env->reward_tower;
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
    if (target->pid == -1 || target->team == player->team)
        return 1;

    // Currently only allowed to heal players
    if (target->entity_type != ENTITY_PLAYER)
        return 1;

    int missing_health = target->max_health - target->health;
    if (amount <= missing_health) {
        target->health += amount;
        player->healing_dealt += amount;
        target->healing_received += amount;
        return 0;
    }

    target->health = target->max_health;
    player->healing_dealt += missing_health;
    target->healing_received += missing_health;
    return 0;
}

int respawn_creep(MOBA* env, Entity* entity, int lane) {
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
    entity->health = entity->max_health;
    entity->waypoint = 1;
    return move_to(env->map, entity, y, x);
}

// TODO: Only 1 spawn needs to exist
void spawn_creep(MOBA* env, int idx, int lane) {
    int pid = creep_offset(env) + idx;
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

    respawn_creep(env, creep, lane);
}

int respawn_neutral(MOBA* env, int idx) {
    int pid = neutral_offset(env) + idx;
    Entity* neutral = &env->entities[pid];
    neutral->pid = pid;
    neutral->health = neutral->max_health;
    neutral->basic_attack_timer = 0;

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
    return move_to(env->map, neutral, y, x);
}

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

            env->scanned_targets[pid][idx] = target;
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
    float dist = l1_distance(target->y, target->x, player->y, player->x);
    float dx = target->x - player->x;
    float dy = target->y - player->y;

    // TODO: Push enable? I think was pushing off map or something
    return 1;

    if (dist == 0.0)
        return 1;

    // Norm to unit vector
    dx = amount * dx / dist;
    dy = amount * dy / dist;

    return move_to(env->map, player, target->y + dy, target->x + dx);
}

int pull(MOBA* env, Entity* player, Entity* target, float amount) {
    return push(env, target, player, -amount);
}

int aoe_push(MOBA* env, Entity* player, int radius, float amount) {
    scan_aoe(env, player, radius, true, false, false, false, true);
    int err = 1;
    int pid = player->pid;
    for (int idx = 0; idx < 121; idx++) {
        Entity* target = env->scanned_targets[pid][idx];
        if (target == NULL)
            break;

        push(env, player, target, amount);
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

        move_towards(env, creep, env->agent_speed, dest_y, dest_x);
    } else {
        float dest_y = WAYPOINTS[lane][waypoint][0];
        float dest_x = WAYPOINTS[lane][waypoint][1];
        move_towards(env, creep, env->agent_speed, dest_y, dest_x);

        // TODO: Last waypoint?
        float dist = l1_distance(creep->y, creep->x, dest_y, dest_x);
        if (dist < 2 && WAYPOINTS[lane][waypoint+1][0] != 0)
            creep->waypoint += 1;
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
            move_towards(env, neutral, env->agent_speed, target->y, target->x);
        
    } else if (l1_distance(neutral->y, neutral->x,
            neutral->spawn_y, neutral->spawn_x) > 2) {
        move_towards(env, neutral, env->agent_speed, neutral->spawn_y, neutral->spawn_x);
    }
}

void randomize_tower_hp(MOBA* env) {
    for (int i = 0; i < env->num_towers; i++) {
        int pid = tower_offset(env) + i;
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

int skill_tank_engage_aoe(MOBA* env, Entity* player, Entity* target) {
    int mana_cost = 50;
    if (target == NULL || player->mana < mana_cost)
        return 1;

    if (move_near(env->map, player, target) == 0) {
        player->mana -= mana_cost;
        player->e_timer = 40;
        aoe_push(env, player, 4, 2.0 + 0.1*player->level);
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

        if (push(env, target, player, 1.0 + 0.05*player->level) == 0) {
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

void step_creeps(MOBA* env) {
    // Spawn wave
    if (env->tick % 150 == 0) {
        for (int lane = 0; lane < 6; lane++) {
            for (int i = 0; i < 5; i++) {
                spawn_creep(env, env->creep_idx, lane);
                env->creep_idx = (env->creep_idx + 1) % env->num_creeps;
            }
        }
    }
    for (int idx = 0; idx < env->num_creeps; idx++) {
        int pid = creep_offset(env) + idx;
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
                respawn_neutral(env, 4*camp + neut);
            }
        }
    }
    for (int idx = 0; idx < env->num_neutrals; idx++) {
        int pid = neutral_offset(env) + idx;
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
    for (int idx = 0; idx < env->num_towers; idx++) {
        int pid = tower_offset(env) + idx;
        Entity* tower = &env->entities[pid];
        if (tower->pid == -1)
            continue;

        update_cooldowns(tower);
        if (tower->basic_attack_timer > 0)
            continue;

        if (env->tick % 3 == 0) // Is this fast enough?
            scan_aoe(env, tower, TOWER_VISION, true, false, false, true, true);

        Entity* target = nearest_scanned_target(env, tower);
        if (target != NULL) 
            basic_attack(env, tower, target);
    }
}

void step_players(MOBA* env) {
    // Clear rewards
    for (int pid = 0; pid < env->num_agents; pid++) {
        Reward* reward = &env->rewards[pid];
        reward->death = 0;
        reward->xp = 0;
        reward->distance = 0;
        reward->tower = 0;

        env->sum_rewards[pid] = 0;
    }

    for (int pid = 0; pid < env->num_agents; pid++) {
        Entity* player = &env->entities[pid];
        if (rand() % 1024 == 0)
            respawn_player(env->map, player);

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

        int (*actions)[6] = (int(*)[6])env->actions;
        float vel_y = actions[pid][0] / 100.0f;
        float vel_x = actions[pid][1] / 100.0f;
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
        if (use_q && player->q_timer <= 0 && skill_support_hook(env, player, target) && player->q_uses < MAX_USES) {
            player->q_uses += 1;
        } else if (use_w && player->w_timer <= 0 && skill_support_aoe_heal(env, player, target) && player->w_uses < MAX_USES) {
            player->w_uses += 1;
        } else if (use_e && player->e_timer <= 0 && skill_support_stun(env, player, target) && player->e_uses < MAX_USES) {
            player->e_uses += 1;
        } else if (target != NULL && basic_attack(env, player, target) && player->basic_attack_uses < MAX_USES) {
            player->basic_attack_uses += 1;
        }

        float dest_y = player->y + player->move_modifier*env->agent_speed*vel_y;
        float dest_x = player->x + player->move_modifier*env->agent_speed*vel_x;
        move_to(env->map, player, dest_y, dest_x);

        Reward* reward = &env->rewards[pid];
        env->sum_rewards[pid] = (
            reward->death +
            reward->xp +
            reward->distance +
            reward->tower
        );
        env->norm_rewards[pid] = (
            reward->death/env->reward_death +
            reward->xp/env->reward_xp +
            reward->distance/env->reward_distance +
            reward->tower/env->reward_tower
        );
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
    long file_bytes = ftell(file);
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

MOBA* init_moba(num_agents, num_creeps, num_neutrals, num_towers, vision_range,
        agent_speed, discretize, reward_death, reward_xp, reward_distance, reward_tower) {
    MOBA* env = (MOBA*)calloc(1, sizeof(MOBA));

    env->num_agents = num_agents;
    env->num_creeps = num_creeps;
    env->num_neutrals = num_neutrals;
    env->num_towers = num_towers;
    env->num_entities = num_agents + num_creeps + num_neutrals + num_towers;
    env->vision_range = vision_range;
    env->agent_speed = agent_speed;
    env->discretize = discretize;
    env->obs_size = 2*vision_range + 1;
    env->creep_idx = 0;

    env->reward_death = reward_death;
    env->reward_xp = reward_xp;
    env->reward_distance = reward_distance;
    env->reward_tower = reward_tower;
    env->total_towers_taken = 0;
    env->total_levels_gained = 0;
    env->radiant_victories = 0;
    env->dire_victories = 0;

    env->rewards = calloc(env->num_agents, sizeof(Reward));
    env->sum_rewards = calloc(env->num_agents, sizeof(float));
    env->norm_rewards = calloc(env->num_agents, sizeof(float));

    env->map = (Map*)calloc(1, sizeof(Map));
    env->map->grid = read_file("game_map.npy");
    env->orig_grid = read_file("game_map.npy"); // TODO: Remove
    env->map->width = 128;
    env->map->height = 128;
    env->map->pids = calloc(env->map->width*env->map->height, sizeof(int));
    for (int i = 0; i < env->map->width*env->map->height; i++)
        env->map->pids[i] = -1;

    env->ai_paths = read_file("ai_paths.npy");

    // Zero out scanned targets
    for (int i = 0; i < 256; i++) {
        env->scanned_targets[i][0] = NULL;
        env->scanned_targets[i][1] = NULL;
    }

    // TODO: Don't hardcode sizes
    env->observations_map = calloc(env->num_agents*121*121*4, sizeof(unsigned char));
    env->observations_extra = calloc(env->num_agents*26, sizeof(unsigned char));
    env->actions = calloc(env->num_agents*6, sizeof(int));
    env->entities = calloc(env->num_entities, sizeof(Entity));

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
            player->move_speed = agent_speed;
            player->basic_attack_cd = 8;
            player->base_damage = 50;
            player->q_uses = 0;
            player->w_uses = 0;
            player->e_uses = 0;
            player->basic_attack_uses = 0;
            player->damage_dealt = 0;
            player->damage_received = 0;
            player->healing_dealt = 0;
            player->healing_received = 0;
            player->deaths = 0;
            player->heros_killed = 0;
            player->creeps_killed = 0;
            player->neutrals_killed = 0;
            player->towers_killed = 0;
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
    }

    Entity *tower;
    for (int idx = 0; idx < env->num_towers; idx++) {
        int pid = tower_offset(env) + idx;
        tower = &env->entities[pid];
        tower->pid = pid;
        tower->entity_type = ENTITY_TOWER;
        tower->grid_id = TOWER;
        tower->basic_attack_cd = 5;
        tower->team = TOWER_TEAM[idx];
        tower->spawn_y = TOWER_Y[idx];
        tower->spawn_x = TOWER_X[idx];
        tower->max_health = TOWER_HEALTH[idx];
        tower->damage = TOWER_DAMAGE[idx];
        tower->tier = TOWER_TIER[idx];
        tower->xp_on_kill = 800 * tower->tier;
    }

    int idx = 0;
    Entity* neutral;
    for (int camp = 0; camp < env->num_neutrals/4; camp++) {
        // 4 neutrals per camp
        for (int i = 0; i < 4; i++) {
            int pid = neutral_offset(env) + idx;
            neutral = &env->entities[pid];
            // TODO: Consider initializing pid for start of game
            neutral->entity_type = ENTITY_NEUTRAL;
            neutral->grid_id = NEUTRAL;
            neutral->max_health = 500;
            neutral->team = 2;
            neutral->spawn_y = NEUTRAL_CAMP_Y[camp];
            neutral->spawn_x = NEUTRAL_CAMP_X[camp];
            neutral->xp_on_kill = 35;
            neutral->basic_attack_cd = 5;
            neutral->damage = 22;
            idx++;
        }
    }
    return env;
}

void reset(MOBA* env) {
    //self.grid[:] = self.orig_grid
    //map->pids[:] = -1
    
    env->tick = 0;
    Map* map = env->map;

    // Respawn towers
    for (int idx = 0; idx < env->num_towers; idx++) {
        int pid = tower_offset(env) + idx;
        Entity* tower = &env->entities[pid];
        tower->target_pid = -1;
        tower->pid = pid;
        tower->health = tower->max_health;
        tower->basic_attack_timer = 0;
        tower->x = 0;
        tower->y = 0;
        int adr = map_offset(map, tower->spawn_y, tower->spawn_x);
        map->grid[adr] = EMPTY;
        move_to(env->map, tower, tower->spawn_y, tower->spawn_x);
        tower->last_x = tower->x;
        tower->last_y = tower->y;
    }

    // Respawn agents
    for (int i = 0; i < env->num_agents; i++) {
        Entity* player = &env->entities[i];
        player->target_pid = -1;
        player->xp = 0;
        player->level = 1;
        player->x = 0;
        player->y = 0;
        respawn_player(env->map, player);
    }

    // Despawn creeps
    for (int i = 0; i < env->num_creeps; i++) {
        int pid = creep_offset(env) + i;
        Entity* creep = &env->entities[pid];
        creep->target_pid = -1;
        creep->pid = -1;
        creep->x = 0;
        creep->y = 0;
    }

    // Despawn neutrals
    for (int i = 0; i < env->num_neutrals; i++) {
        int pid = neutral_offset(env) + i;
        Entity* neutral = &env->entities[pid];
        neutral->target_pid = -1;
        neutral->pid = -1;
        neutral->x = 0;
        neutral->y = 0;
    }
    compute_observations(env);
}

int step(MOBA* env) {
    int num_entities = env->num_agents + env->num_towers + env->num_creeps + env->num_neutrals;
    for (int pid = 0; pid < num_entities; pid++) {
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

    int radiant_pid = tower_offset(env) + 22;
    int dire_pid = tower_offset(env) + 23;

    Entity* ancient = &env->entities[radiant_pid];
    if (ancient->health <= 0) {
        reset(env);
        env->radiant_victories += 1;
        return 1;
    }

    ancient = &env->entities[dire_pid];
    if (ancient->health <= 0) {
        reset(env);
        env->dire_victories += 1;
        return 2;
    }

    compute_observations(env);
    return 0;
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
 
typedef struct {
    int cell_size;
    int width;
    int height;
} Renderer;

Renderer* init_renderer(int cell_size, int width, int height) {
    Renderer* renderer = (Renderer*)malloc(sizeof(Renderer));
    renderer->cell_size = cell_size;
    renderer->width = width;
    renderer->height = height;
    InitWindow(width*cell_size, height*cell_size, "Puffer MOBA");
    SetTargetFPS(10);
    return renderer;
}

void close_renderer(Renderer* renderer) {
    CloseWindow();
    free(renderer);
}

void render_global(Renderer* renderer, MOBA* env) {
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


