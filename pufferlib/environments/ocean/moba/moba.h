// Incremental port of Puffer Moba to C. Be careful to add semicolons and avoid leftover cython syntax
#include <math.h>
#include <stdbool.h>

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

typedef struct {
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
} Entity;

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

float* fast_rng(CachedRNG* rng) {
    float val = rng->rng[rng->rng_idx];
    rng->rng_idx += 1
    if rng->rng_idx >= rng->rng_n - 1:
        rng->rng_idx = 0
    return val
}

typedef struct {
    int num_agents;
    int num_creeps;
    int num_neutrals;
    int num_towers;
    int vision_range;
    float agent_speed;
    bool discretize;
    int obs_size;
    int creep_idx;
    int tick;

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

    float reward_death;
    float reward_xp;
    float reward_distance;
    float reward_tower;
    
    public int total_towers_taken;
    public int total_levels_gained;
    public int radiant_victories;
    public int dire_victories;

    # MAX_ENTITIES x MAX_SCANNED_TARGETS
    Entity* scanned_targets[256][121]
    skill skills[10][3]

    int rng_n;
    int rng_idx;
    float[:] rng;
} MOBA
 
inline int xp_for_player_kill(Entity* entity) {
    return 100 + (int)(entity->xp / 7.69);
}
 
inline float clip(float x) {
    return fmaxf(-1.0f, fminf(x, 1.0f));
}

inline float l1_distance(float x1, float y1, float x2, float y2) {
    return fabs(x1 - x2) + fabs(y1 - y2);
}

// TODO: Should not take entire moba. Rename to min_greater_than or similar
int level(MOBA* self, int xp) {
    for (int i = 0; i < 30; i++) {
        if (xp < self->xp_for_level[i])
            return i + 1;
    }
    return i + 1;
}
     
cdef Reward* get_reward(MOBA* self, int pid) {
    return &self.rewards[pid];
}
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


typedef struct {
    unsigned char* grid;
    int* pid_map;
} Map

inline int map_offset(Map* map, int y, int x) {
    return y*map->width + x;
}

inline int creep_offset(MOBA* moba, int y, int x) {
    return moba->num_agents;
}

inline int neutral_offset(MOBA* moba, int y, int x) {
    return moba->num_agents + moba->num_creeps;
}

inline int tower_offset(MOBA* moba, int y, int x) {
    return moba->num_agents + moba->num_creeps + moba->num_neutrals;
}

int move_to(Map* map, Entity* player, float dest_y, float dest_x):
    int src = map_offset(map, (int)player->y, (int)player->x);
    int dst = map_offset(map, (int)dest_y, (int)dest_x);

    if (map->grid[idx] != EMPTY and map->pid_map[idx] != player->pid)
        return 1;

    map->grid[src] = EMPTY;
    map->grid[dst] = player->grid_id;

    map->pid_map[src] = -1;
    map->pid_map[dst] = player->pid;

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
    int y_src = int(entity.y);
    int x_src = int(entity.x);

    int atn = env->ai_paths[y_dst, x_dst, y_src, x_src];
    if (atn >= 8)
        return 0;

    float modifier = speed * entity->move_modifier;
    y_dst = y_src + modifier*self.atn_map[0, atn];
    x_dst = x_src + modifier*self.atn_map[1, atn];

    if (move_to(env, entity, dst_y, dst_x) == 0)
        return 0;

    float jitter_x = fast_rng(env->rng);
    float jitter_y = fast_rng(env->rng);
    return move_to(entity, entity.y + jitter_y, entity.x + jitter_x)

int attack(MOBA* env, Entity* player, Entity* target, float damage):
    if target->pid == -1 or target->team == player->team:
        return False

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
        self.rewards[target->pid]->death = self.reward_death;
        player->heros_killed += 1;
        target->deaths += 1;
        self.respawn_player(target);
    } else if (target_type == ENTITY_CREEP) {
        player->creeps_killed += 1;
        self.kill(target);
    } else if (target_type == ENTITY_NEUTRAL) {
        player->neutrals_killed += 1;
        self.kill(target);
    } else if (target_type == ENTITY_TOWER) {
        player->towers_killed += 1;
        self.kill(target);
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
        Entity* ally = &self.entities[first_player_on_team + i];
        if (ally->pid == player->pid) {
            in_range[i] = true;
            num_in_range += 1;
            continue;
        }

        if (l1_distance(ally->y, ally->x, target_y, target_x) <= XP_RANGE) {
            in_range[i] = true;
            num_in_range += 1;
        }

    xp /= num_in_range;

    for (int i = 0; i < 5; i++) {
        if (!in_range[i])
            continue;

        Entity* ally = &self.entities[first_player_on_team + i];
        if (ally->xp > 10000000)
            continue;

        ally->xp += xp;
        env->rewards[first_player_on_team + i]->xp = env->reward_xp * xp;

        int level = ally->level;
        ally->level = self.level(ally->xp);
        if (ally->level > level)
            env->total_levels_gained += 1;

        ally->max_health = ally->base_health + ally->level*ally->hp_gain_per_level;
        ally->max_mana = ally->base_mana + ally->level*ally->mana_gain_per_level;
        ally->damage = ally->base_damage + ally->level*ally->damage_gain_per_level;

    if (target->entity_type == ENTITY_TOWER) {
        env->rewards[player->pid]->tower = env->reward_tower;
        env->total_towers_taken += 1;
    }

    return 0;

int basic_attack(MOBA* env, Entity* player, Entity* target):
    if (player->basic_attack_timer > 0)
        return 1;

    player->basic_attack_timer = player->basic_attack_cd;
    return attack(player, target, player->damage);

int heal(MOBA* env, Entity* player, Entity* target, float amount):
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

void kill(Map* map, Entity* entity):
    int adr = map_offset(map, (int)entity->y, (int)entity->x);
    map->grid[adr] = EMPTY;
    map->pids[adr] = -1;
    entity.pid = -1;
    entity.x = 0;
    entity.y = 0;

void respawn_player(Map* map, Entity* entity):
    int pid = entity->pid;
    kill(map, entity);
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
    while (not valid_pos) {
        y = entity.spawn_y + rand()%15 - 7
        x = entity.spawn_x + rand()%15 - 7
        valid_pos = map->grid[map_offset(map, y, x)] == EMPTY;
    }

    self.move_to(entity, y, x)

int spawn_at(Map* map, Entity* entity, float y, float x):
    int adr = map_offset(map, (int)y, (int)x);

    if (map->grid[adr] != EMPTY)
        return 1;

    map->grid[adr] = entity->grid_id;
    map->pid_map[adr] = entity->pid;
    entity->y = y;
    entity->x = x;
    return 0;
}

void update_status(Entity* entity):
    if (entity->stun_timer > 0)
        entity->stun_timer -= 1;
    
    if (entity->move_timer > 0)
        entity->move_timer -= 1;
    
    if (entity->move_timer == 0)
        entity->move_modifier = 1.0;
    
void update_cooldowns(Entity* entity):
    if (entity->q_timer > 0)
        entity->q_timer -= 1;
    
    if (entity->w_timer > 0)
        entity->w_timer -= 1;
    
    if (entity->e_timer > 0)
        entity->e_timer -= 1;
    
    if (entity->basic_attack_timer > 0)
        entity->basic_attack_timer -= 1;


