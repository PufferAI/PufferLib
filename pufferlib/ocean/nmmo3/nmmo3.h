// Neural MMO 3 by Joseph Suarez
// This was the first new environment I started for Puffer Ocean.
// I started it in Cython and then ported it to C. This is why there
// are still some commented sections with features I didn't get to fully
// implement, like the command console. Feel free to add and PR!
// The assets get generated from a separate script. Message me if you need those,
// since they use ManaSeed assets. I've licensed them for the project but
// can't include the source files before they've gone through spritesheet gen.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <limits.h>
#include "simplex.h"
#include "tile_atlas.h"
#include "raylib.h"

#if defined(PLATFORM_DESKTOP)
    #define GLSL_VERSION 330
#else
    #define GLSL_VERSION 100
#endif

// Play modes
#define MODE_PLAY 0
#define MODE_BUY_TIER 1
#define MODE_BUY_ITEM 2
#define MODE_SELL_SELECT 3
#define MODE_SELL_PRICE 4

// Animations
#define ANIM_IDLE 0
#define ANIM_MOVE 1
#define ANIM_ATTACK 2
#define ANIM_SWORD 3
#define ANIM_BOW 4
#define ANIM_DEATH 5
#define ANIM_RUN 6

// Actions - the order of these is used for offset math
#define ATN_DOWN 0
#define ATN_UP 1
#define ATN_RIGHT 2
#define ATN_LEFT 3
#define ATN_NOOP 4
#define ATN_ATTACK 5
#define ATN_UI 7
#define ATN_ONE 8
#define ATN_TWO 9
#define ATN_THREE 10
#define ATN_FOUR 11
#define ATN_FIVE 12
#define ATN_SIX 13
#define ATN_SEVEN 14
#define ATN_EIGHT 15
#define ATN_NINE 16
#define ATN_ZERO 17
#define ATN_MINUS 18
#define ATN_EQUALS 19
#define ATN_BUY 20
#define ATN_SELL 21
#define ATN_DOWN_SHIFT 22
#define ATN_UP_SHIFT 23
#define ATN_RIGHT_SHIFT 24
#define ATN_LEFT_SHIFT 25

// Entity types
#define ENTITY_NULL 0
#define ENTITY_PLAYER 1
#define ENTITY_ENEMY 2

// Elements
#define ELEM_NEUTRAL 0
#define ELEM_FIRE 1
#define ELEM_WATER 2
#define ELEM_EARTH 3
#define ELEM_AIR 4

// Tiles

#define TILE_SPRING_GRASS 0
#define TILE_SUMMER_GRASS 1
#define TILE_AUTUMN_GRASS 2
#define TILE_WINTER_GRASS 3
#define TILE_SPRING_DIRT 4
#define TILE_SUMMER_DIRT 5
#define TILE_AUTUMN_DIRT 6
#define TILE_WINTER_DIRT 7
#define TILE_SPRING_STONE 8
#define TILE_SUMMER_STONE 9
#define TILE_AUTUMN_STONE 10
#define TILE_WINTER_STONE 11
#define TILE_SPRING_WATER 12
#define TILE_SUMMER_WATER 13
#define TILE_AUTUMN_WATER 14
#define TILE_WINTER_WATER 15

// Entity
#define P_N 44
#define P_TYPE 0
#define P_COMB_LVL 1
#define P_ELEMENT 2
#define P_DIR 3
#define P_ANIM 4
#define P_HP 5
#define P_HP_MAX 6
#define P_PROF_LVL 7
#define P_EQUIP_HELM 8
#define P_EQUIP_CHEST 9
#define P_EQUIP_LEGS 10
#define P_EQUIP_WEAPON 11
#define P_EQUIP_GEM 12
#define P_UI_MODE 13
#define P_MARKET_TIER 14
#define P_SELL_IDX 15
#define P_GOLD 16
#define P_IN_COMBAT 17
#define P_INVENTORY 18
#define P_INVENTORY_SIZE 12
#define P_EQUIP_BOOLS 30
#define P_WANDER_RANGE 42
#define P_RANGED 43

// Items
#define I_N 17
#define I_NULL 0
#define I_HELM 1
#define I_CHEST 2
#define I_LEGS 3
#define I_SWORD 4
#define I_BOW 5
#define I_TOOL 6
#define I_EARTH 7
#define I_FIRE 8
#define I_AIR 9
#define I_WATER 10
#define I_HERB 11
#define I_ORE 12
#define I_WOOD 13
#define I_HILT 14
#define I_SILVER 15
#define I_GOLD 16

#define INVENTORY_SIZE 12

// Equipment
#define SLOT_HELM 0
#define SLOT_CHEST 1
#define SLOT_LEGS 2
#define SLOT_HELD 3
#define SLOT_GEM 4

// Map dims
#define D_N 2
#define D_MAP 0
#define D_ITEM 1

// Extra constants
#define IN_COMBAT_TICKS 5
#define LEVEL_MUL 2.0
#define EQUIP_MUL 1.0
#define TIER_EXP_BASE 8
#define MAX_TIERS 5
#define NPC_AGGRO_RANGE 4

void range(int* array, int n) {
    for (int i = 0; i < n; i++) {
        array[i] = i;
    }
}

void shuffle(int* array, int n) {
    for (int i = 0; i < n; i++) {
        int j = rand() % n;
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

double sample_exponential(double halving_rate) {
    double u = (double)rand() / RAND_MAX; // Random number u in [0, 1)
    return 1 + halving_rate*(-log(1 - u) / log(2));
}

// Terrain gen
char ALL_GRASS[4] = {TILE_SPRING_GRASS, TILE_SUMMER_GRASS, TILE_AUTUMN_GRASS, TILE_WINTER_GRASS};
char ALL_DIRT[4] = {TILE_SPRING_DIRT, TILE_SUMMER_DIRT, TILE_AUTUMN_DIRT, TILE_WINTER_DIRT};
char ALL_STONE[4] = {TILE_SPRING_STONE, TILE_SUMMER_STONE, TILE_AUTUMN_STONE, TILE_WINTER_STONE};
char ALL_WATER[4] = {TILE_SPRING_WATER, TILE_SUMMER_WATER, TILE_AUTUMN_WATER, TILE_WINTER_WATER};

unsigned char RENDER_COLORS[16][3] = {
    {60, 220, 75},   // Spring grass
    {20, 180, 40},   // Summer grass
    {210, 180, 40},  // Autumn grass
    {240, 250, 250}, // Winter grass
    {130, 110, 70},  // Spring dirt
    {160, 140, 70},  // Summer dirt
    {140, 120, 90},  // Autumn dirt
    {130, 120, 100}, // Winter dirt
    {120, 120, 130}, // Spring stone
    {110, 110, 120}, // Summer stone
    {100, 100, 110}, // Autumn stone
    {180, 180, 190}, // Winter stone
    {70, 130, 180},  // Spring water
    {0, 120, 200},   // Summer water
    {50, 100, 160},  // Autumn water
    {210, 240, 255}, // Winter water
};

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float return_comb_lvl;
    float return_prof_lvl;
    float return_item_atk_lvl;
    float return_item_def_lvl;
    float return_market_buy;
    float return_market_sell;
    float return_death;
    float min_comb_prof;
    float purchases;
    float sales;
    float equip_attack;
    float equip_defense;
    float r;
    float c;
};
 
// TODO: This is actually simplex and we should probably use the original impl
// ALSO: Not seeded correctly
void perlin_noise(float* map, int width, int height,
        float base_frequency, int octaves, int offset_x, int offset_y) {
    float frequencies[octaves];
    for (int i = 0; i < octaves; i++) {
        frequencies[i] = base_frequency*pow(2, i);
    }

    float min_value = FLT_MAX;
    float max_value = FLT_MIN;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            for (int oct = 0; oct < octaves; oct++) {
                float freq = frequencies[oct];
                map[adr] = noise2(freq*c + offset_x, freq*r + offset_y);
            }
            float val = map[adr];
            if (val < min_value) {
                min_value = val;
            }
            if (val > max_value) {
                max_value = val;
            }
        }
    }

    // TODO: You scale wrong in the original code
    float scale = 1.0/(max_value - min_value);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            map[adr] = scale * (map[adr] - min_value);
        }
    }
}

void flood_fill(unsigned char* input, char* output,
        int width, int height, int n, int max_fill) {

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            output[r*width + c] = -1;
        }
    }

    int* pos = calloc(width*height, sizeof(int));
    range((int*)pos, width*height);
    shuffle((int*)pos, width*height);

    short queue[2*max_fill];
    for (int i = 0; i < 2*max_fill; i++) {
        queue[i] = 0;
    }

    for (int idx = 0; idx < width*height; idx++) {
        int r = pos[idx] / width;
        int c = pos[idx] % width;
        int adr = r*width + c;

        if (input[adr] != 0 || output[adr] != -1) {
            continue;
        }

        int color = rand() % n;
        output[adr] = color;
        queue[0] = r;
        queue[1] = c;

        int queue_idx = 0;
        int queue_n = 1;
        while (queue_idx < max_fill && queue_idx < queue_n) {
            r = queue[2*queue_idx];
            c = queue[2*queue_idx + 1];

            // These checks are done before adding, even though that is
            // more verbose, to preserve the max q length. There are
            // also some perf benefits with the bounds check
            int dd = r-1;
            adr = dd*width + c;
            if (dd >= 0 && input[adr] == 0 && output[adr] == -1) {
                output[adr] = color;
                queue[2*queue_n] = dd;
                queue[2*queue_n + 1] = c;
                queue_n += 1;
                if (queue_n == max_fill) {
                    break;
                }
            }
            dd = c-1;
            adr = r*width + dd;
            if (dd >= 0 && input[adr] == 0 && output[adr] == -1) {
                output[adr] = color;
                queue[2*queue_n] = r;
                queue[2*queue_n + 1] = dd;
                queue_n += 1;
                if (queue_n == max_fill) {
                    break;
                }
            }
            dd = r+1;
            adr = dd*width + c;
            if (dd < width && input[adr] == 0 && output[adr] == -1) {
                output[adr] = color;
                queue[2*queue_n] = dd;
                queue[2*queue_n + 1] = c;
                queue_n += 1;
                if (queue_n == max_fill) {
                    break;
                }
            }
            dd = c+1;
            adr = r*width + dd;
            if (dd < width && input[adr] == 0 && output[adr] == -1) {
                output[adr] = color;
                queue[2*queue_n] = r;
                queue[2*queue_n + 1] = dd;
                queue_n += 1;
                if (queue_n == max_fill) {
                    break;
                }
            }
            queue_idx += 1;
        }
    }
    free(pos);
}

void cellular_automata(char* grid,
        int width, int height, int colors, int max_fill) {

    int* pos = calloc(2*width*height, sizeof(int));
    int pos_sz = 0;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int inp_adr = r*width + c;
            if (grid[inp_adr] != -1) {
                continue;
            }
            pos[pos_sz] = r;
            pos_sz++;
            pos[pos_sz] = c;
            pos_sz++;
        }
    }

    bool done = false;
    while (!done) {
        // In place shuffle on active buffer only
        for (int i = 0; i < pos_sz; i+=2) {
            int r = pos[i];
            int c = pos[i + 1];
            int adr = rand() % pos_sz;
            if (adr % 2 == 1) {
                adr--;
            }
            pos[i] = pos[adr];
            pos[i + 1] = pos[adr + 1];
            pos[adr] = r;
            pos[adr + 1] = c;
        }

        done = true;
        int pos_adr = 0;
        for (int i = 0; i < pos_sz; i+=2) {
            int r = pos[i];
            int c = pos[i + 1];

            int counts[colors];
            for (int i = 0; i < colors; i++) {
                counts[i] = 0;
            }

            bool no_neighbors = true;
            for (int rr = r-1; rr <= r+1; rr++) {
                for (int cc = c-1; cc <= c+1; cc++) {
                    if (rr < 0 || rr >= height || cc < 0 || cc >= width) {
                        continue;
                    }
                    int adr = rr*width + cc;
                    int val = grid[adr];
                    if (val != -1) {
                        counts[val] += 1;
                        no_neighbors = false;
                    }
                }
            }

            if (no_neighbors) {
                done = false;
                pos[pos_adr] = r;
                pos_adr++;
                pos[pos_adr] = c;
                pos_adr++;
                continue;
            }

            // Find maximum count and ties
            int max_count = 0;
            for (int i = 0; i < colors; i++) {
                int val_count = counts[i];
                if (val_count > max_count) {
                    max_count = val_count;
                }
            }
            int num_ties = 0;
            for (int i = 0; i < colors; i++) {
                if (counts[i] == max_count) {
                    num_ties += 1;
                }
            }

            int idx = 0;
            int winner = rand() % num_ties;
            for (int j = 0; j < colors; j++) {
                if (counts[j] == max_count) {
                    if (idx == winner) {
                        int adr = r*width + c;
                        grid[adr] = j;
                        break;
                    }
                    idx += 1;
                }
            }
        }
        pos_sz = pos_adr;
        pos_adr = 0;
    }
    free(pos);
}

void generate_terrain(char* terrain, unsigned char* rendered,
        int R, int C, int x_border, int y_border) {
    // Perlin noise for the base terrain
    // TODO: Not handling octaves correctly
    float* perlin_map = calloc(R*C, sizeof(float));
    int offset_x = rand() % 100000;
    int offset_y = rand() % 100000;
    perlin_noise(perlin_map, C, R, 1.0/64.0, 2, offset_x, offset_y);
 
    // Flood fill connected components to determine biomes
    unsigned char* ridges = calloc(R*C, sizeof(unsigned char));
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            int adr = r*C + c;
            ridges[adr] = (perlin_map[adr]>0.35) & (perlin_map[adr]<0.65);
        }
    }
    char *biomes = calloc(R*C, sizeof(char));
    flood_fill(ridges, biomes, R, C, 4, 4000);

    // Cellular automata to cover unfilled ridges
    cellular_automata(biomes, R, C, 4, 4000);

    unsigned char (*rendered_ary)[C][3] = (unsigned char(*)[C][3])rendered;

    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            int tile;
            int adr = r*C + c;
            if (r < y_border || r >= R-y_border || c < x_border || c >= C-x_border) {
                tile = TILE_SPRING_WATER;
            } else {
                int season = biomes[adr];
                float val = perlin_map[adr];
                if (val > 0.75) {
                    tile = ALL_STONE[season];
                } else if (val < 0.25) {
                    tile = ALL_WATER[season];
                } else {
                    tile = ALL_GRASS[season];
                }
            }
            terrain[adr] = tile;
            rendered_ary[r][c][0] = RENDER_COLORS[tile][0];
            rendered_ary[r][c][1] = RENDER_COLORS[tile][1];
            rendered_ary[r][c][2] = RENDER_COLORS[tile][2];
        }
    }
    free(perlin_map);
    free(ridges);
    free(biomes);
}

typedef struct Entity Entity;
struct Entity {
    int type;
    int comb_lvl;
    int element;
    int dir;
    int anim;
    int hp;
    int hp_max;
    int prof_lvl;
    int ui_mode;
    int market_tier;
    int sell_idx;
    int gold;
    int in_combat;
    int equipment[5];
    int inventory[12];
    int is_equipped[12];
    int wander_range;
    int ranged;
    int goal;
    int equipment_attack;
    int equipment_defense;
    int r;
    int c;
    int spawn_r;
    int spawn_c;
    int min_comb_prof[500];
    int min_comb_prof_idx;
    int time_alive;
    int purchases;
    int sales;
};

typedef struct Item Item;
struct Item {
    int id;
    int type;
    int tier;
};

Item ITEMS[(MAX_TIERS+1)*I_N + 1];

int item_index(int i, int tier) {
    return (tier-1)*I_N + i;
}

void init_items() {
    Item* item_ptr;
    for (int tier = 1; tier <= MAX_TIERS+1; tier++) {
        for (int i = 1; i <= I_N; i++) {
            int id = item_index(i, tier);
            item_ptr = &ITEMS[id];
            item_ptr->id = id;
            item_ptr->type = i;
            item_ptr->tier = tier;
        }
    }
}

#define MAX_MARKET_OFFERS 32

typedef struct MarketOffer MarketOffer;
struct MarketOffer {
    int id;
    int seller;
    int price;
};

typedef struct ItemMarket ItemMarket;
struct ItemMarket {
    MarketOffer offers[MAX_MARKET_OFFERS];
    int next_offer_id;
    int item_id;
    int stock;
};

// TODO: Redundant?
int peek_price(ItemMarket* market) {
    int stock = market->stock;
    if (stock == 0) {
        return 0;
    }
    return market->offers[stock-1].price;
}

typedef struct Reward Reward;
struct Reward{
    float death;
    float pioneer;
    float comb_lvl;
    float prof_lvl;
    float item_atk_lvl;
    float item_def_lvl;
    float item_tool_lvl;
    float market_buy;
    float market_sell;
};

typedef struct Respawnable Respawnable;
struct Respawnable {
    int id;
    int r;
    int c;
};

typedef struct RespawnBuffer RespawnBuffer;
struct RespawnBuffer {
    Respawnable* data;
    int* lengths;
    int ticks;
    int size;
};

RespawnBuffer* make_respawn_buffer(int size, int ticks) {
    RespawnBuffer* buffer = calloc(1, sizeof(RespawnBuffer));
    buffer->data = calloc(ticks*size, sizeof(Respawnable));
    buffer->lengths = calloc(ticks, sizeof(int));
    buffer->ticks = ticks;
    buffer->size = size;
    return buffer;
}

bool has_elements(RespawnBuffer* buffer, int tick) {
    return buffer->lengths[tick % buffer->ticks] > 0;
}

void free_respawn_buffer(RespawnBuffer* buffer) {
    free(buffer->data);
    free(buffer->lengths);
    free(buffer);
}

void clear_respawn_buffer(RespawnBuffer* buffer) {
    for (int i = 0; i < buffer->ticks; i++) {
        buffer->lengths[i] = 0;
    }
}

void add_to_buffer(RespawnBuffer* buffer, Respawnable elem, int tick) {
    tick = tick % buffer->ticks;
    assert(buffer->lengths[tick] < buffer->size);
    int offset = tick*buffer->size + buffer->lengths[tick];
    buffer->data[offset] = elem; buffer->lengths[tick] += 1; }

Respawnable pop_from_buffer(RespawnBuffer* buffer, int tick) {
    tick = tick % buffer->ticks;
    assert(buffer->lengths[tick] > 0);
    buffer->lengths[tick] -= 1;
    int offset = tick*buffer->size + buffer->lengths[tick];
    return buffer->data[offset];
}

typedef struct Client Client;
typedef struct MMO MMO;
struct MMO {
    Client* client;
    int width;
    int height;
    int num_players;
    int num_enemies;
    int num_resources;
    int num_weapons;
    int num_gems;
    char* terrain; // TODO: Unsigned?
    unsigned char* rendered;
    Entity* players;
    Entity* enemies;
    short* pids;
    unsigned char* items;
    unsigned char* counts;
    unsigned char* observations;
    float* rewards;
    float* terminals;
    Reward* reward_struct;
    Reward* returns;
    int* actions;
    int tick;
    int tiers;
    int levels;
    float teleportitis_prob;
    int x_window;
    int y_window;
    int obs_size;
    int enemy_respawn_ticks;
    int item_respawn_ticks;
    ItemMarket* market;
    int market_buys;
    int market_sells;
    RespawnBuffer* resource_respawn_buffer;
    RespawnBuffer* enemy_respawn_buffer;
    RespawnBuffer* drop_respawn_buffer;
    Log log;
    float reward_combat_level;
    float reward_prof_level;
    float reward_item_level;
    float reward_market;
    float reward_death;
};

Entity* get_entity(MMO* env, int pid) {
    if (pid < env->num_players) {
        return &env->players[pid];
    } else {
        return &env->enemies[pid - env->num_players];
    }
}

void add_player_log(MMO* env, int pid) {
    Reward * ret = &env->returns[pid];
    Entity* player = get_entity(env, pid);
    Log* log = &env->log;
    log->return_comb_lvl += ret->comb_lvl;
    log->return_prof_lvl += ret->prof_lvl;
    log->return_item_atk_lvl += ret->item_atk_lvl;
    log->return_item_def_lvl += ret->item_def_lvl;
    log->return_market_buy += ret->market_buy;
    log->return_market_sell += ret->market_sell;
    log->return_death += ret->death;
    log->min_comb_prof += (player->prof_lvl > player->comb_lvl) ? player->comb_lvl : player->prof_lvl;
    log->purchases += player->purchases;
    log->sales += player->sales;
    log->equip_attack += player->equipment_attack;
    log->equip_defense += player->equipment_defense;
    log->r += player->r;
    log->c += player->c;
    log->episode_return += (
        ret->comb_lvl + ret->prof_lvl
        + ret->item_atk_lvl + ret->item_def_lvl
        + ret->market_buy + ret->market_sell
        + ret->death
    );
    log->episode_length += player->time_alive;
    log->score = log->min_comb_prof;
    log->perf = log->min_comb_prof / (float)env->levels;
    log->n++;
    *ret = (Reward){0};
}

void init(MMO* env) {
    init_items();

    int sz = env->width*env->height;
    env->counts = calloc(sz, sizeof(unsigned char));
    env->terrain = calloc(sz, sizeof(char));
    env->rendered = calloc(sz*3, sizeof(unsigned char));

    env->pids = calloc(sz, sizeof(short));
    env->items = calloc(sz, sizeof(unsigned char));

    // Circular buffers for respawning resources and enemies
    env->resource_respawn_buffer = make_respawn_buffer(2*env->num_resources
        + 2*env->num_weapons + 4*env->num_gems, env->item_respawn_ticks);
    env->enemy_respawn_buffer = make_respawn_buffer(
        env->num_enemies, env->enemy_respawn_ticks);
    env->drop_respawn_buffer = make_respawn_buffer(2*env->num_enemies, 20);

    env->returns = calloc(env->num_players, sizeof(Reward));
    env->reward_struct = calloc(env->num_players, sizeof(Reward));
    env->players = calloc(env->num_players, sizeof(Entity));
    env->enemies = calloc(env->num_enemies, sizeof(Entity));

    // TODO: Figure out how to cast to array. Size is static
    int num_market = (MAX_TIERS+1)*(I_N+1);
    env->market = (ItemMarket*)calloc(num_market, sizeof(ItemMarket));
}

void allocate_mmo(MMO* env) {
    // TODO: Not hardcode
    env->observations = calloc(env->num_players*(11*15*10+47+10), sizeof(unsigned char));
    env->rewards = calloc(env->num_players, sizeof(float));
    env->terminals = calloc(env->num_players, sizeof(float));
    env->actions = calloc(env->num_players, sizeof(int));
    init(env);
}

void free_mmo(MMO* env) {
    free(env->counts);
    free(env->terrain);
    free(env->rendered);
    free(env->pids);
    free(env->items);
    free_respawn_buffer(env->resource_respawn_buffer);
    free_respawn_buffer(env->enemy_respawn_buffer);
    free_respawn_buffer(env->drop_respawn_buffer);
    free(env->market);
}

void free_allocated_mmo(MMO* env) {
    free(env->observations);
    free(env->rewards);
    free(env->terminals);
    free(env->returns);
    free(env->reward_struct);
    free(env->players);
    free(env->enemies);
    free(env->actions);
    free_mmo(env);
}

bool is_buy(int mode) {
    return mode == MODE_BUY_TIER || mode == MODE_BUY_ITEM;
}

bool is_sell(int mode) {
    return mode == MODE_SELL_SELECT || mode == MODE_SELL_PRICE;
}

bool is_move(int action) {
    return action >= ATN_DOWN && action <= ATN_LEFT;
}

bool is_run(int action) {
    return action >= ATN_DOWN_SHIFT && action <= ATN_LEFT_SHIFT;
}

bool is_num(int action) {
    return action >= ATN_ONE && action <= ATN_NINE;
}

int EFFECT_MATRIX[5][5] = {
    {1, 1, 1, 1, 1},
    {1, 1, 0, 1, 2},
    {1, 2, 1, 0, 1},
    {1, 1, 2, 1, 0},
    {1, 0, 1, 2, 1},
};

int DELTAS[4][2] = {
    {1, 0},
    {-1, 0},
    {0, 1},
    {0, -1},
};

int ATTACK_BASIC[4][1][2] = {
    {{1, 0}},
    {{-1, 0}},
    {{0, 1}},
    {{0, -1}},
};
    
int ATTACK_SWORD[4][3][2] = {
    {{1, -1}, {1, 0}, {1, 1}},
    {{-1, -1}, {-1, 0}, {-1, 1}},
    {{-1, 1}, {0, 1}, {1, 1}},
    {{-1, -1}, {0, -1}, {1, -1}},
};

int ATTACK_BOW[4][12][2] = {
    {{1, 0}, {2, 0}, {3, 0}, {4, 0}},
    {{-1, 0}, {-2, 0}, {-3, 0}, {-4, 0}},
    {{0, 1}, {0, 2}, {0, 3}, {0, 4},},
    {{0, -1}, {0, -2}, {0, -3}, {0, -4}},
};

float tier_level(float tier) {
    return TIER_EXP_BASE*pow(2, tier-1);
}

float level_tier(int level) {
    if (level < TIER_EXP_BASE) {
        return 1;
    }
    return 1 + ceil(log2(level/(float)TIER_EXP_BASE));
}

bool PASSABLE[16] = {
    true, true, true, true,     // Grass tiles
    true, true, true, true,     // Dirt tiles
    false, false, false, false, // Stone tiles
    false, false, false, false,  // Water tiles
};

bool is_grass(int tile) {
    return (tile >= TILE_SPRING_GRASS && tile <= TILE_WINTER_GRASS);
} 

bool is_dirt(int tile) {
    return (tile >= TILE_SPRING_DIRT && tile <= TILE_WINTER_DIRT);
}

bool is_stone(int tile) {
    return (tile >= TILE_SPRING_STONE && tile <= TILE_WINTER_STONE);
}

bool is_water(int tile) {
    return (tile >= TILE_SPRING_WATER && tile <= TILE_WINTER_WATER);
}

int map_offset(MMO* env, int r, int c) {
    return r*env->width + c;
}

float sell_price(int idx) {
    return 0.5 + 0.1*idx;
}

void compute_all_obs(MMO* env) {
    for (int pid = 0; pid < env->num_players; pid++) {
        Entity* player = get_entity(env, pid);
        int r = player->r;
        int c = player->c;

        int start_row = r - env->y_window;
        int end_row = r + env->y_window + 1;
        int start_col = c - env->x_window;
        int end_col = c + env->x_window + 1;

        assert(start_row >= 0);
        assert(end_row <= env->height);
        assert(start_col >= 0);
        assert(end_col <= env->width);

        int comb_lvl = player->comb_lvl;
        int obs_adr = pid*(11*15*10+47+10);
        for (int obs_r = start_row; obs_r < end_row; obs_r++) {
            for (int obs_c = start_col; obs_c < end_col; obs_c++) {
                int map_adr = map_offset(env, obs_r, obs_c);

                // Split by terrain type and season
                unsigned char terrain = env->terrain[map_adr];
                env->observations[obs_adr] = terrain % 4;
                env->observations[obs_adr+1] = terrain / 4;

                // Split by item type and tier
                unsigned char item = env->items[map_adr];
                env->observations[obs_adr+2] = item % 17;
                env->observations[obs_adr+3] = item / 17;

                int pid = env->pids[map_adr];
                if (pid != -1) {
                    Entity* seen = get_entity(env, pid);
                    env->observations[obs_adr+4] = seen->type;
                    env->observations[obs_adr+5] = seen->element;
                    int delta_comb_obs = (seen->comb_lvl - comb_lvl) / 2;
                    if (delta_comb_obs < 0) {
                        delta_comb_obs = 0;
                    }
                    if (delta_comb_obs > 4) {
                        delta_comb_obs = 4;
                    }
                    env->observations[obs_adr+6] = delta_comb_obs;
                    env->observations[obs_adr+7] = seen->hp / 20; // Bucketed for discrete
                    env->observations[obs_adr+8] = seen->anim;
                    env->observations[obs_adr+9] = seen->dir;
                }
                obs_adr += 10;
            }
        }

        // Player observation
        env->observations[obs_adr] = player->type;
        env->observations[obs_adr+1] = player->comb_lvl;
        env->observations[obs_adr+2] = player->element;
        env->observations[obs_adr+3] = player->dir;
        env->observations[obs_adr+4] = player->anim;
        env->observations[obs_adr+5] = player->hp;
        env->observations[obs_adr+6] = player->hp_max;
        env->observations[obs_adr+7] = player->prof_lvl;
        env->observations[obs_adr+8] = player->ui_mode;
        env->observations[obs_adr+9] = player->market_tier;
        env->observations[obs_adr+10] = player->sell_idx;
        env->observations[obs_adr+11] = player->gold;
        env->observations[obs_adr+12] = player->in_combat;
        for (int j = 0; j < 5; j++) {
            env->observations[obs_adr+13+j] = player->equipment[j];
        }
        for (int j = 0; j < 12; j++) {
            env->observations[obs_adr+18+j] = player->inventory[j];
        }
        for (int j = 0; j < 12; j++) {
            env->observations[obs_adr+30+j] = player->is_equipped[j];
        }
        env->observations[obs_adr+42] = player->wander_range;
        env->observations[obs_adr+43] = player->ranged;
        env->observations[obs_adr+44] = player->goal;
        env->observations[obs_adr+45] = player->equipment_attack;
        env->observations[obs_adr+46] = player->equipment_defense;

        // Reward observation
        Reward* reward = &env->reward_struct[pid];
        env->observations[obs_adr+47] = (reward->death == 0) ? 0 : 1;
        env->observations[obs_adr+48] = (reward->pioneer == 0) ? 0 : 1;
        env->observations[obs_adr+49] = reward->comb_lvl / 20;
        env->observations[obs_adr+50] = reward->prof_lvl / 20;
        env->observations[obs_adr+51] = reward->item_atk_lvl / 20;
        env->observations[obs_adr+52] = reward->item_def_lvl / 20;
        env->observations[obs_adr+53] = reward->item_tool_lvl / 20;
        env->observations[obs_adr+54] = reward->market_buy / 20;
        env->observations[obs_adr+55] = reward->market_sell / 20;
    }
}

int safe_tile(MMO* env, int delta) {
    bool valid = false;
    int idx;
    while (!valid) {
        valid = true;
        idx = rand() % (env->width * env->height);
        char tile = env->terrain[idx];
        if (!is_grass(tile)) {
            valid = false;
            continue;
        }
        int r = idx / env->width;
        int c = idx % env->width;
 
        for (int dr = -delta; dr <= delta; dr++) {
            for (int dc = -delta; dc <= delta; dc++) {
                int adr = map_offset(env, r+dr, c+dc);
                if (env->pids[adr] != -1) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                break;
            }
        }
    }
    return idx;
}

// Spawns a player at the specified position.
// Can spawn on top of another player, but this will not corrupt the state.
// They will just move off of each other.
void spawn(MMO* env, Entity* entity) {
    entity->hp = 99;
    entity->time_alive = 0;
    entity->purchases = 0;
    entity->sales = 0;

    int idx = safe_tile(env, 5);
    int r = idx / env->width;
    int c = idx % env->width;

    //entity->r = entity->spawn_r;
    //entity->c = entity->spawn_c;
    entity->spawn_r = r;
    entity->spawn_c = c;
    entity->r = r;
    entity->c = c;

    entity->anim = ANIM_IDLE;
    entity->dir = ATN_DOWN;
    entity->ui_mode = MODE_PLAY;
    entity->gold = 0;
    entity->in_combat = 0;
    entity->equipment_attack = 0;
    entity->equipment_defense = 0;

    // Try zeroing levels too
    entity->prof_lvl = 1;
    entity->comb_lvl = 1;

    entity->equipment[SLOT_HELM] = 0;
    entity->equipment[SLOT_CHEST] = 0;
    entity->equipment[SLOT_LEGS] = 0;
    entity->equipment[SLOT_HELD] = 0;
    entity->equipment[SLOT_GEM] = 0;

    int num_slots = sizeof(entity->inventory) / sizeof(entity->inventory[0]);
    for (int idx = 0; idx < num_slots; idx++) {
        entity->inventory[idx] = 0;
        entity->is_equipped[idx] = 0;
    }

    entity->goal = (rand() % 2) == 0;
    memset(entity->min_comb_prof, 0, sizeof(entity->min_comb_prof));
    entity->min_comb_prof_idx = 0;
}

void give_starter_gear(MMO* env, int pid, int tier) {
    assert(tier >= 1);
    assert(tier <= env->tiers);

    Entity* player = &env->players[pid];
    int idx = (rand() % 6) + 1;
    tier = (rand() % tier) + 1;
    player->inventory[0] = item_index(idx, tier);
    player->gold += 50;
}

int get_free_inventory_idx(MMO* env, int pid) {
    Entity* player = &env->players[pid];
    // TODO: #define this
    int num_slots = sizeof(player->inventory) / sizeof(player->inventory[0]);
    for (int idx = 0; idx < num_slots; idx++) {
        int item_type = player->inventory[idx];
        if (item_type == 0) {
            return idx;
        }
    }
    return -1;
}

void pickup_item(MMO* env, int pid) {
    Entity* player = &env->players[pid];
    if (player->type != ENTITY_PLAYER) {
        return;
    }

    int r = player->r;
    int c = player->c;
    int adr = map_offset(env, r, c);
    int ground_id = env->items[adr];
    if (ground_id == 0) {
        return;
    }

    int inventory_idx = get_free_inventory_idx(env, pid);
    if (inventory_idx == -1) {
        return;
    }

    Item* ground_item = &ITEMS[ground_id];
    int ground_type = ground_item->type;

    // This is the only item that can be picked up without a tool
    if (ground_type == I_TOOL) {
        player->inventory[inventory_idx] = ground_id;
        env->items[adr] = 0;
        return;
    }

    int ground_tier = ground_item->tier;
    int held_id = player->equipment[SLOT_HELD];
    Item* held_item = &ITEMS[held_id];
    int held_type = held_item->type;
    int held_tier = held_item->tier;
    if (held_type != I_TOOL) {
        return;
    }
    if (held_tier < ground_tier) {
        return;
    }

    // Harvest resource
    Respawnable respawnable = {.id = ground_id, .r = r, .c = c};
    add_to_buffer(env->resource_respawn_buffer, respawnable, env->tick);
    Reward* reward = &env->reward_struct[pid];
    Reward* ret = &env->returns[pid];

    // Level up for a worthy harvest
    if (player->prof_lvl < env->levels && player->prof_lvl < tier_level(ground_tier)) {
        player->prof_lvl += 1;
        reward->prof_lvl = env->reward_prof_level;
        ret->prof_lvl += env->reward_prof_level;
    }

    // Some items are different on the ground and in inventory
    if (ground_type == I_ORE) {
        int armor_id = I_HELM + rand() % 3;
        ground_id = item_index(armor_id, ground_tier);
    } else if (ground_type == I_HILT) {
        ground_id = item_index(I_SWORD, ground_tier);
    } else if (ground_type == I_WOOD) {
        ground_id = item_index(I_BOW, ground_tier);
    } else {
        ground_id = item_index(ground_type, ground_tier);
    }
    player->inventory[inventory_idx] = ground_id;
    env->items[adr] = 0;
}

bool dest_check(MMO* env, int r, int c);
inline bool dest_check(MMO* env, int r, int c) {
    int adr = map_offset(env, r, c);
    return PASSABLE[(int)env->terrain[adr]] & (env->pids[adr] == -1);
}

void move(MMO* env, int pid, int direction, bool run) {
    Entity* entity = get_entity(env, pid);
    int r = entity->r;
    int c = entity->c;
    int dr = DELTAS[direction][0];
    int dc = DELTAS[direction][1];
    int rr = r + dr;
    int cc = c + dc;

    entity->dir = direction;

    if (!dest_check(env, rr, cc)) {
        return;
    }

    if (run) {
        rr += dr;
        cc += dc;
        if (!dest_check(env, rr, cc)) {
            return;
        }
    }

    // Move to new pos.
    entity->r = rr;
    entity->c = cc;
    entity->anim = (run ? ANIM_RUN : ANIM_MOVE);
    env->pids[map_offset(env, rr, cc)] = pid;

    int old_adr = map_offset(env, r, c);
    env->pids[old_adr] = -1;

    // Update visitation map. Skips run tiles
    if (entity->type == ENTITY_PLAYER) {
        if (env->counts[map_offset(env, rr, cc)] == 0) {
            env->reward_struct[pid].pioneer = 1.0;
        }
        if (env->counts[map_offset(env, rr, cc)] < 255) {
            env->counts[map_offset(env, rr, cc)] += 1;
        }
        pickup_item(env, pid);
    }
}

void wander(MMO* env, int pid) {
    Entity* entity = get_entity(env, pid);
    int wander_range = entity->wander_range;
    int spawn_r = entity->spawn_r;
    int spawn_c = entity->spawn_c;
    int end_r = spawn_r;
    int end_c = spawn_c;

    // Return entity to wander area
    if (end_r - spawn_r > wander_range) {
        move(env, pid, ATN_UP, false);
        return;
    }
    if (end_r - spawn_r < -wander_range) {
        move(env, pid, ATN_DOWN, false);
        return;
    }
    if (end_c - spawn_c > wander_range) {
        move(env, pid, ATN_LEFT, false);
        return;
    }
    if (end_c - spawn_c < -wander_range) {
        move(env, pid, ATN_RIGHT, false);
        return;
    }

    // Move randomly
    int direction = rand() % 4;
    if (direction == ATN_UP) {
        end_r -= 1;
    } else if (direction == ATN_DOWN) {
        end_r += 1;
    } else if (direction == ATN_LEFT) {
        end_c -= 1;
    } else if (direction == ATN_RIGHT) {
        end_c += 1;
    }

    move(env, pid, direction, false);
}

// Agents gain 2 damage per level and 1 per equip level. With 3
// pieces of equipment, that is a total of 5 per level. Base damage is
// 40 and enemies start with a decrease of 25. So with a 5 level difference,
// players and enemies are equally matched. 
int calc_damage(MMO* env, int pid, int target_id) {
    Entity* attacker = get_entity(env, pid);
    Entity* defender = get_entity(env, target_id);

    int attack = 40 + LEVEL_MUL*attacker->comb_lvl + attacker->equipment_attack;
    int defense = LEVEL_MUL*defender->comb_lvl + defender->equipment_defense;

    // These buffs compensate for enemies not having equipment
    if (attacker->type == ENTITY_ENEMY) {
        attack += 3*EQUIP_MUL*attacker->comb_lvl - 25;
    }
    if (defender->type == ENTITY_ENEMY) {
        defense += 3*EQUIP_MUL*defender->comb_lvl;
    }

    int damage = fmax(attack - defense, 0);

    // Not very / normal / super effective
    return damage * EFFECT_MATRIX[attacker->element][defender->element];
}

int find_target(MMO* env, int pid, int entity_type) {
    Entity* entity = get_entity(env, pid);
    int r = entity->r;
    int c = entity->c;
    int weapon_id = entity->equipment[SLOT_HELD];
    int anim;
    int* flat_deltas;
    int num_deltas = 0;
    if (weapon_id == 0 || ITEMS[weapon_id].type == I_TOOL) {
        flat_deltas = (int*)ATTACK_BASIC;
        anim = ANIM_ATTACK;
        num_deltas = 1;
    } else if (ITEMS[weapon_id].type == I_BOW) {
        flat_deltas = (int*)ATTACK_BOW;
        anim = ANIM_BOW;
        num_deltas = 12;
    } else if (ITEMS[weapon_id].type == I_SWORD) {
        flat_deltas = (int*)ATTACK_SWORD;
        anim = ANIM_SWORD;
        num_deltas = 3;
    } else {
        assert(false);
        exit(1);
    }

    entity->anim = anim;
    int (*deltas)[num_deltas][2] = (int(*)[num_deltas][2])flat_deltas;
    for (int direction = 0; direction < 4; direction++) {
        for (int idx = 0; idx < num_deltas; idx++) {
            int dr = deltas[direction][idx][0];
            int dc = deltas[direction][idx][1];
            int rr = r + dr;
            int cc = c + dc;

            int adr = map_offset(env, rr, cc);
            int target_id = env->pids[adr];
            if (target_id == -1) {
                continue;
            }

            Entity* target = get_entity(env, target_id);
            if (target->type != entity_type) {
                continue;
            }

            entity->dir = direction;
            return target_id;
        }
    }
    return -1;
}

void drop_loot(MMO* env, int pid) {
    Entity* entity = get_entity(env, pid);
    int loot_tier = level_tier(entity->comb_lvl);
    if (loot_tier > env->tiers) {
        loot_tier = env->tiers;
    }

    int drop = item_index(I_TOOL, loot_tier);
    int r = entity->r;
    int c = entity->c;

    // Drop loot on a free tile
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            int adr = map_offset(env, r+dr, c+dc);
            if (env->items[adr] != 0) {
                continue;
            }
            env->items[adr] = drop;
            Respawnable elem = {.id = drop, .r = r+dr, .c = c+dc};
            add_to_buffer(env->drop_respawn_buffer, elem, env->tick);
            return;
        }
    }
}

void attack(MMO* env, int pid, int target_id) {
    Entity* attacker = get_entity(env, pid);
    Entity* defender = get_entity(env, target_id);

    // Extra check avoids multiple xp/loot drops
    // if two players attack the same target at the same time
    if (defender->hp == 0) {
        return;
    }

    attacker->in_combat = IN_COMBAT_TICKS;
    defender->in_combat = IN_COMBAT_TICKS;
    int dmg = calc_damage(env, pid, target_id);

    // Simple case: target survives
    if (dmg < defender->hp) {
        defender->hp -= dmg;
        return;
    }

    // Defender dies
    defender->hp = 0;
    if (defender->type == ENTITY_PLAYER) {
        Reward* reward = &env->reward_struct[target_id];
        Reward* ret = &env->returns[target_id];
        reward->death = env->reward_death;
        ret->death += env->reward_death;
        env->reward_struct[target_id].death = -1;
        add_player_log(env, target_id);
    } else {
        // Add to respawn buffer
        Respawnable respawnable = {.id = target_id,
            .r = defender->spawn_r, .c = defender->spawn_c};
        add_to_buffer(env->enemy_respawn_buffer, respawnable, env->tick);
    }

    if (attacker->type == ENTITY_PLAYER) {
        Reward* reward = &env->reward_struct[pid];
        Reward* ret = &env->returns[pid];
        int attacker_lvl = attacker->comb_lvl;
        int defender_lvl = defender->comb_lvl;

        // Level up for defeating worthy foe
        if (defender_lvl >= attacker_lvl && attacker_lvl < env->levels) {
            attacker->comb_lvl += 1;
            reward->comb_lvl = env->reward_combat_level;
            ret->comb_lvl += env->reward_combat_level;
        }
        if (defender->type == ENTITY_ENEMY) {
            drop_loot(env, target_id);
            attacker->gold += 1 + defender_lvl / 10;
            // Overflow
            if (attacker->gold > 99) {
                attacker->gold = 99;
            }
        }
    }
}

void use_item(MMO* env, int pid, int inventory_idx) {
    Entity* player = &env->players[pid];
    Reward* reward = &env->reward_struct[pid];
    Reward* ret = &env->returns[pid];
    int item_id = player->inventory[inventory_idx];

    if (item_id == 0) {
        return;
    }

    Item* item = &ITEMS[item_id];
    int item_type = item->type;
    int tier = item->tier;

    // Consumable
    if (item_type == I_HERB) {
        int hp_restore = 50 + 10*tier;
        if (player->hp > player->hp_max - hp_restore) {
            player->hp = player->hp_max;
        } else {
            player->hp += hp_restore;
        }
        player->inventory[inventory_idx] = 0;
        return;
    }

    // Cannot equip in combat
    if (player->in_combat > 0) {
        return;
    }

    int element = -1;
    int attack = 0;
    int defense = 0;
    int equip_slot = 0;

    if (item_type == I_HELM) {
        equip_slot = SLOT_HELM;
        defense = EQUIP_MUL*tier_level(tier);
    } else if (item_type == I_CHEST) {
        equip_slot = SLOT_CHEST;
        defense = EQUIP_MUL*tier_level(tier);
    } else if (item_type == I_LEGS) {
        equip_slot = SLOT_LEGS;
        defense = EQUIP_MUL*tier_level(tier);
    } else if (item_type == I_SWORD) {
        equip_slot = SLOT_HELD;
        attack = 3*EQUIP_MUL*tier_level(tier);
    } else if (item_type == I_BOW) {
        equip_slot = SLOT_HELD;
        attack = 3*EQUIP_MUL*tier_level(tier - 0.5);
    } else if (item_type == I_TOOL) {
        equip_slot = SLOT_HELD;
    } else if (item_type == I_EARTH) {
        equip_slot = SLOT_GEM;
        element = ELEM_EARTH;
    } else if (item_type == I_FIRE) {
        equip_slot = SLOT_GEM;
        element = ELEM_FIRE;
    } else if (item_type == I_AIR) {
        equip_slot = SLOT_GEM;
        element = ELEM_AIR;
    } else if (item_type == I_WATER) {
        equip_slot = SLOT_GEM;
        element = ELEM_WATER;
    } else {
        exit(1);
    }

    float item_reward = env->reward_item_level * (float)tier / env->tiers;

    // Unequip item if already equipped
    if (player->is_equipped[inventory_idx]) {
        player->is_equipped[inventory_idx] = 0;
        player->equipment[equip_slot] = 0;
        player->equipment_attack -= attack;
        player->equipment_defense -= defense;
        if (item_type == I_TOOL) {
            reward->item_tool_lvl = -item_reward;
        } else {
            if (attack > 0) {
                reward->item_atk_lvl = -item_reward;
                ret->item_atk_lvl -= item_reward;
            }
            if (defense > 0) {
                reward->item_def_lvl = -item_reward;
                ret->item_def_lvl -= item_reward;
            }
        }
        if (equip_slot == SLOT_GEM) {
            player->element = ELEM_NEUTRAL;
        }
        return;
    }

    // Another item is already equipped. We don't support switching
    // gear without unequipping because it adds complexity to the item repr
    if (player->equipment[equip_slot] != 0) {
        return;
    }
    
    // Equip the current item
    player->is_equipped[inventory_idx] = 1;
    player->equipment[equip_slot] = item_id;
    player->equipment_attack += attack;
    player->equipment_defense += defense;
    if (item_type == I_TOOL) {
        reward->item_tool_lvl = item_reward;
    } else {
        if (attack > 0) {
            reward->item_atk_lvl = item_reward;
            ret->item_atk_lvl += item_reward;
        }
        if (defense > 0) {
            reward->item_def_lvl = item_reward;
            ret->item_def_lvl += item_reward;
        }
    }

    // Update element for gems
    if (element != -1) {
        player->element = element;
    }
}

void enemy_ai(MMO* env, int pid) {
    Entity* enemy = get_entity(env, pid);
    int r = enemy->r;
    int c = enemy->c;

    for (int rr = r-NPC_AGGRO_RANGE; rr <= r+NPC_AGGRO_RANGE; rr++) {
        for (int cc = c-NPC_AGGRO_RANGE; cc <= c+NPC_AGGRO_RANGE; cc++) {
            int adr = map_offset(env, rr, cc);
            int target_id = env->pids[adr];
            if (target_id == -1 || target_id >= env->num_players) {
                continue;
            }

            int dr = rr - r;
            int dc = cc - c;
            int abs_dr = abs(dr);
            int abs_dc = abs(dc);

            int direction;
            if (enemy->ranged) {
                if (abs_dr == 0 && abs_dc <= NPC_AGGRO_RANGE) {
                    direction = (dc > 0) ? ATN_RIGHT : ATN_LEFT;
                    enemy->anim = ANIM_BOW;
                    attack(env, pid, target_id);
                } else if (abs_dc == 0 && abs_dr <= NPC_AGGRO_RANGE) {
                    direction = (dr > 0) ? ATN_DOWN : ATN_UP;
                    enemy->anim = ANIM_BOW;
                    attack(env, pid, target_id);
                } else {
                    if (abs_dr > abs_dc) {
                        direction = (dc > 0) ? ATN_RIGHT : ATN_LEFT;
                    } else {
                        direction = (dr > 0) ? ATN_DOWN : ATN_UP;
                    }
                    // Move along shortest axis
                    move(env, pid, direction, false);
                }
            } else {
                if (abs_dr + abs_dc == 1) {
                    if (dr > 0) {
                        direction = ATN_DOWN;
                    } else if (dr < 0) {
                        direction = ATN_UP;
                    } else if (dc > 0) {
                        direction = ATN_RIGHT;
                    } else {
                        direction = ATN_LEFT;
                    }
                    enemy->anim = ANIM_SWORD;
                    attack(env, pid, target_id);
                } else {
                    // Move along longest axis
                    if (abs_dr > abs_dc) {
                        direction = (dr > 0) ? ATN_DOWN : ATN_UP;
                    } else {
                        direction = (dc > 0) ? ATN_RIGHT : ATN_LEFT;
                    }
                    move(env, pid, direction, false);
                }
            }
            enemy->dir = direction;
            return;
        }
    }
    wander(env, pid);
}

void c_reset(MMO* env) {
    env->tick = 0;

    env->market_sells = 0;
    env->market_buys = 0;

    clear_respawn_buffer(env->resource_respawn_buffer);
    clear_respawn_buffer(env->enemy_respawn_buffer);

    // TODO: Check width/height args!
    generate_terrain(env->terrain, env->rendered, env->width, env->height,
        env->x_window, env->y_window);

    for (int i = 0; i < env->width*env->height; i++) {
        env->pids[i] = -1;
        env->items[i] = 0;
        //env->counts[i] = 0;
    }
    
    // Pid crops?
    int ore_count = 0;
    int herb_count = 0;
    int wood_count = 0;
    int hilt_count = 0;
    int earth_gem_count = 0;
    int fire_gem_count = 0;
    int air_gem_count = 0;
    int water_gem_count = 0;
    int player_count = 0;
    int enemy_count = 0;

    // Randomly generate spawn candidates
    int *spawn_cands = calloc(env->width*env->height, sizeof(int));
    range((int*)spawn_cands, env->width*env->height);
    shuffle((int*)spawn_cands, env->width*env->height);

    for (int cand_idx = 0; cand_idx < env->width*env->height; cand_idx++) {
        int cand = spawn_cands[cand_idx];
        int r = cand / env->width;
        int c = cand % env->width;
        int tile = env->terrain[cand];

        if (!is_grass(tile)) {
            continue;
        }

        // Materials only spawn south
        //if (r < env->height/2) {
        //    continue;
        //}

        int spawned = false;
        int i_type;
        for (int d = 0; d < 4; d++) {
            int adr = map_offset(env, r+DELTAS[d][0], c+DELTAS[d][1]);
            int tile = env->terrain[adr];
            if (is_stone(tile)) {
                if (ore_count < env->num_resources) {
                    i_type = I_ORE;
                    ore_count += 1;
                    spawned = true;
                    break;
                }
                if (hilt_count < env->num_weapons) {
                    i_type = I_HILT;
                    hilt_count += 1;
                    spawned = true;
                    break;
                }
            } else if (is_water(tile)) {
                if (herb_count < env->num_resources) {
                    i_type = I_HERB;
                    herb_count += 1;
                    spawned = true;
                    break;
                }
                if (wood_count < env->num_weapons) {
                    i_type = I_WOOD;
                    wood_count += 1;
                    spawned = true;
                    break;
                }
            }
        }

        int adr = map_offset(env, r, c);
        //int tier = 1 + env->tiers*level/env->levels;
        int tier = 0;
        while (tier < 1 || tier > env->tiers) {
            tier = sample_exponential(1);
        }

        if (spawned) {
            env->items[adr] = item_index(i_type, tier);
            continue;
        }

        // Spawn gems
        i_type = 0;
        if (tile == TILE_SPRING_GRASS && earth_gem_count < env->num_gems) {
            earth_gem_count += 1;
            i_type = I_EARTH;
        } else if (tile == TILE_SUMMER_GRASS && fire_gem_count < env->num_gems) {
            fire_gem_count += 1;
            i_type = I_FIRE;
        } else if (tile == TILE_AUTUMN_GRASS && air_gem_count < env->num_gems) {
            air_gem_count += 1;
            i_type = I_AIR;
        } else if (tile == TILE_WINTER_GRASS && water_gem_count < env->num_gems) {
            water_gem_count += 1;
            i_type = I_WATER;
        }

        if (i_type > 0) {
            env->items[adr] = item_index(i_type, tier);
        }

        if (
            player_count == env->num_players && 
            enemy_count == env->num_enemies && 
            ore_count == env->num_resources && 
            herb_count == env->num_resources && 
            wood_count == env->num_weapons && 
            hilt_count == env->num_weapons && 
            earth_gem_count == env->num_gems && 
            fire_gem_count == env->num_gems && 
            air_gem_count == env->num_gems && 
            water_gem_count == env->num_gems
        ) {
            break;
        }
    }

    assert(ore_count == env->num_resources);
    assert(herb_count == env->num_resources);
    assert(wood_count == env->num_weapons);
    assert(hilt_count == env->num_weapons);
    assert(earth_gem_count == env->num_gems);
    assert(fire_gem_count == env->num_gems);
    assert(air_gem_count == env->num_gems);
    assert(water_gem_count == env->num_gems);
    free(spawn_cands);

    //int distance = abs(r - env->height/2);
    for (int player_count = 0; player_count < env->num_players; player_count++) {
        int pid = player_count;
        Entity* player = &env->players[pid];
        player->type = ENTITY_PLAYER;
        player->element = ELEM_NEUTRAL;
        player->comb_lvl = 1;
        player->prof_lvl = 1;
        player->hp_max = 99;
        spawn(env, player);
        int adr = map_offset(env, player->r, player->c);
        env->pids[adr] = pid;
        // Debug starter gear
        //give_starter_gear(env, pid, env->tiers);
    }

    // Spawn enemies off of middle Y
    //int level = fmax(1, env->levels * (distance-12) / (0.9*env->height/2 - 24));
    //level = fmin(level, env->levels);
    for (int enemy_count = 0; enemy_count < env->num_enemies; enemy_count++) {
        int level = 0;
        while (level < 1 || level > env->levels) {
            level = sample_exponential(8);
        }
        if (rand() % 8 == 0) {
            level = 1;
        }
        //if (distance > 8 && r < env->height/2 && enemy_count < env->num_enemies) {
        Entity* enemy = &env->enemies[enemy_count];
        enemy->type = ENTITY_ENEMY;
        enemy->hp_max = 99;
        enemy->wander_range = 3;

        spawn(env, enemy);
        int adr = map_offset(env, enemy->r, enemy->c);
        char tile = env->terrain[adr];

        int element = ELEM_NEUTRAL;
        int ranged = true;
        if (level < 15) {
            ranged = false;
        } else if (tile == TILE_SPRING_GRASS) {
            element = ELEM_EARTH;
        } else if (tile == TILE_SUMMER_GRASS) {
            element = ELEM_FIRE;
        } else if (tile == TILE_AUTUMN_GRASS) {
            element = ELEM_AIR;
        } else if (tile == TILE_WINTER_GRASS) {
            element = ELEM_WATER;
        }
        enemy->element = element;
        enemy->ranged = ranged;

        env->pids[adr] = env->num_players + enemy_count;
        enemy->comb_lvl = level;
    }

    compute_all_obs(env);
}

void c_step(MMO* env) {
    env->tick += 1;
    int tick = env->tick;

    // Respawn resources
    RespawnBuffer* buffer = env->resource_respawn_buffer;
    while (has_elements(buffer, tick)) {
        Respawnable item = pop_from_buffer(buffer, tick);
        int item_id = item.id;
        assert(item_id > 0);
        int adr = map_offset(env, item.r, item.c);
        env->items[adr] = item_id;
    }

    // Respawn enemies
    buffer = env->enemy_respawn_buffer;
    while (has_elements(buffer, tick)) {
        int pid = pop_from_buffer(buffer, tick).id;
        assert(pid >= 0);
        Entity* entity = get_entity(env, pid);
        int lvl = entity->comb_lvl;
        spawn(env, entity);
        int adr = map_offset(env, entity->r, entity->c);
        env->pids[adr] = pid;
        entity->comb_lvl = lvl;
    }

    // Despawn dropped items
    buffer = env->drop_respawn_buffer;
    while (has_elements(buffer, tick)) {
        Respawnable item = pop_from_buffer(buffer, tick);
        int id = item.id;
        int r = item.r;
        int c = item.c;
        int adr = map_offset(env, r, c);
        if (env->items[adr] == id) {
            env->items[adr] = 0;
        }
    }

    for (int pid = 0; pid < env->num_players + env->num_enemies; pid++) {
        Entity* entity = get_entity(env, pid);
        entity->time_alive += 1;
        int entity_type = entity->type;
        int r = entity->r;
        int c = entity->c;
        int adr = map_offset(env, r, c);

        // Respawn dead entity
        if (entity->hp == 0) {
            if (entity->anim != ANIM_DEATH) {
                entity->anim = ANIM_DEATH;
            } else if (env->pids[adr] == pid) {
                env->pids[adr] = -1;
            } else if (entity_type == ENTITY_PLAYER) {
                spawn(env, entity);
                adr = map_offset(env, entity->r, entity->c);
                env->pids[adr] = pid;
                //give_starter_gear(env, pid, env->tiers);
            }
            continue;
        }

        // Teleportitis: Randomly teleport players and enemies
        // to a safe tile. This prevents players from clumping
        // and messing up training dynamics
        double prob = (double)rand() / RAND_MAX;
        if (prob < env->teleportitis_prob) {
            r = entity->r;
            c = entity->c;
            adr = map_offset(env, r, c);
            env->pids[adr] = -1;

            int idx = safe_tile(env, 5);
            r = idx / env->width;
            c = idx % env->width;

            adr = map_offset(env, r, c);
            env->pids[adr] = pid;

            entity->r = r;
            entity->c = c;
        }

        if (entity_type == ENTITY_PLAYER) {
            int min_comb_prof = entity->prof_lvl;
            if (min_comb_prof > entity->comb_lvl) {
                min_comb_prof = entity->comb_lvl;
            }
            entity->min_comb_prof[entity->min_comb_prof_idx] = min_comb_prof;
            entity->min_comb_prof_idx += 1;
            if (entity->min_comb_prof_idx == 500) {
                entity->min_comb_prof_idx = 0;
                if (min_comb_prof <= entity->min_comb_prof[0]) {
                    add_player_log(env, pid);

                    // Has not improved in 500 ticks
                    r = entity->r;
                    c = entity->c;
                    adr = map_offset(env, r, c);
                    env->pids[adr] = -1;
                    int lvl = entity->comb_lvl;
                    spawn(env, entity);
                    r = entity->r;
                    c = entity->c;
                    adr = map_offset(env, r, c);
                    env->pids[adr] = pid;
                    if (entity->type == ENTITY_PLAYER) {
                        //give_starter_gear(env, pid, env->tiers);
                    } else {
                        entity->comb_lvl = lvl;
                    }
                    continue;
                }
            }
        }

        entity->anim = ANIM_IDLE;

        // Restore 1 HP each tick
        if (entity->hp < entity->hp_max) {
            entity->hp += 1;
        }
        
        // Decrement combat counter
        if (entity->in_combat > 0) {
            entity->in_combat -= 1;
        }

        // Enemy AI
        if (entity_type == ENTITY_ENEMY) {
            enemy_ai(env, pid);
            continue;
        }

        Reward* reward = &env->reward_struct[pid];
        *reward = (Reward){0};

        // Update entity heading
        int action = env->actions[pid];
        if (is_move(action)) {
            entity->dir = action - ATN_DOWN;
        } else if (is_run(action)) {
            entity->dir = action - ATN_DOWN_SHIFT;
        }

        // Market mode
        int ui_mode = entity->ui_mode;
        if (is_buy(ui_mode)) {
            if (action != ATN_NOOP) {
                entity->ui_mode = MODE_PLAY;
            }
            if (!is_num(action)) {
                continue;
            }
            if (entity->in_combat > 0) {
                continue;
            }
            int action_idx = action - ATN_ONE;
            if (ui_mode == MODE_BUY_TIER) {
                if (action_idx >= env->tiers) {
                    continue;
                }
                entity->market_tier = action_idx + 1;
                entity->ui_mode = MODE_BUY_ITEM;
                continue;
            }
            if (action_idx >= 11) {
                continue;
            }
            int market_tier = entity->market_tier;
            int item_id = I_N*(market_tier - 1) + action_idx + 1;
            ItemMarket* market = &env->market[item_id];

            int stock = market->stock;
            if (stock == 0) {
                continue;
            }

            MarketOffer* offer = &market->offers[stock-1];

            int price = offer->price;
            Entity* buyer = get_entity(env, pid);
            if (buyer->gold < price) {
                continue;
            }

            int inventory_idx = get_free_inventory_idx(env, pid);
            if (inventory_idx == -1) {
                continue;
            }
         
            buyer->gold -= price;
            buyer->inventory[inventory_idx] = market->item_id;

            Entity* seller = &env->players[offer->seller];
            seller->gold += price;
            if (seller->gold > 99) {
                seller->gold = 99;
            }

            market->stock -= 1;
            env->market_buys += 1;
            reward->market_buy = env->reward_market;
            Reward* ret = &env->returns[pid];
            ret->market_buy += env->reward_market;
            // env->rewards[buyer_id].gold += price;
            // if (env->rewards[buyer_id].gold > 99) {
            //     env->rewards[buyer_id].gold = 99;
            // }
            
            entity->inventory[inventory_idx] = item_id;
            entity->purchases += 1;
        } else if (is_sell(ui_mode)) {
            if (action != ATN_NOOP) {
                entity->ui_mode = MODE_PLAY;
            }
            if (!is_num(action)) {
                continue;
            }
            if (entity->in_combat > 0) {
                continue;
            }
            int action_idx = action - ATN_ONE;
            if (ui_mode == MODE_SELL_SELECT) {
                int item_type = entity->inventory[action_idx];
                if (item_type == 0) {
                    continue;
                }
                entity->sell_idx = action_idx;
                entity->ui_mode = MODE_SELL_PRICE;
                continue;
            }
            int price = action_idx + 1; //sell_price(action_idx);
            int inventory_idx = entity->sell_idx;
            int item_type = entity->inventory[inventory_idx];
            if (item_type == 0) {
                continue;
            }
            if (entity->is_equipped[inventory_idx]) {
                use_item(env, pid, inventory_idx);
            }

            ItemMarket* market = &env->market[item_type];
            int stock = market->stock;

            // TODO: Will have to update once prices become dynamic
            if (stock == MAX_MARKET_OFFERS) {
                continue;
            }

            MarketOffer* offer = &market->offers[stock];
            offer->id = market->next_offer_id;
            offer->seller = pid;
            offer->price = price;
            market->next_offer_id += 1;
            market->stock += 1;

            entity->inventory[inventory_idx] = 0;
            entity->sales += 1;
            env->market_sells += 1;
            reward->market_sell = env->reward_market;
            Reward* ret = &env->returns[pid];
            ret->market_sell += env->reward_market;
        } else if (action == ATN_ATTACK) {
            int target_id = find_target(env, pid, ENTITY_ENEMY);
            if (target_id != -1) {
                attack(env, pid, target_id);
            }
        } else if (is_move(action)) {
            move(env, pid, action, false);
        } else if (is_run(action)) {
            move(env, pid, action - ATN_DOWN_SHIFT, true);
        } else if (is_num(action)) {
            use_item(env, pid, action - ATN_ONE);
        } else if (action == ATN_BUY) {
            entity->ui_mode = MODE_BUY_TIER;
        } else if (action == ATN_SELL) {
            entity->ui_mode = MODE_SELL_SELECT;
        }
    }
    compute_all_obs(env);
    for (int pid = 0; pid < env->num_players; pid++) {
        Reward* reward = &env->reward_struct[pid];
        env->rewards[pid] = (
            reward->death + reward->comb_lvl
            + reward->prof_lvl + reward->item_atk_lvl + reward->item_def_lvl
            + reward->market_buy + reward->market_sell
        );
    }
}

#define FRAME_RATE 60
#define TICK_FRAMES 36
#define DELAY_FRAMES 24
#define SPRITE_SIZE 128
#define TILE_SIZE 64
#define X_WINDOW 7
#define Y_WINDOW 5

#define SCREEN_WIDTH TILE_SIZE * (2*X_WINDOW + 1)
#define SCREEN_HEIGHT TILE_SIZE * (2*Y_WINDOW + 1)

#define NUM_PLAYER_TEXTURES 10

// Health bars
#define HEALTH_BAR_WIDTH 48
#define HEALTH_BAR_HEIGHT 6

#define WATER_ANIMS 3
#define WATER_ANIM_FRAMES 4
#define WATER_TICKS_PER_FRAME TICK_FRAMES / WATER_ANIM_FRAMES

#define COMMAND_CHARS 16

#define ANIM_IDLE 0
#define ANIM_MOVE 1
#define ANIM_ATTACK 2
#define ANIM_SWORD 3
#define ANIM_BOW 4
#define ANIM_DEATH 5
#define ANIM_RUN 6

#define MAX_ANIM_FRAMES 16
#define SPRITE_SIZE 128

#define OVERLAY_NONE 0
#define OVERLAY_COUNTS 1
#define OVERLAY_VALUE 2

#define ITEM_TYPES 17

int ITEM_TEXTURES[ITEM_TYPES*MAX_TIERS];

struct Client {
    Texture2D tiles;
    Texture2D players[5][NUM_PLAYER_TEXTURES];
    Shader shader;
    int shader_camera_x_loc;
    float shader_camera_x;
    int shader_camera_y_loc;
    float shader_camera_y;
    int shader_time_loc;
    float shader_time;
    int shader_terrain_loc;
    int shader_map_width_loc;
    int shader_map_height_loc;
    unsigned char *shader_terrain_data;
    Texture2D shader_terrain;
    int shader_texture_tiles_loc;
    Texture2D shader_texture_tiles;
    int shader_resolution_loc;
    float shader_resolution[3];

    //Texture2D players;
    Texture2D items;
    Texture2D inventory;
    Texture2D inventory_equip;
    Texture2D inventory_selected;
    Font font;
    int* terrain;
    int command_mode;
    char command[COMMAND_CHARS];
    int command_len;
    Camera2D camera;
    RenderTexture2D map_buffer;
    RenderTexture2D ui_buffer;
    int render_mode;
    Texture2D overlay_texture;
    int active_overlay;
    int my_player;
    int start_time;
    float render_delta;
};

#define TILE_SPRING_GRASS 0
#define TILE_SUMMER_GRASS 1
#define TILE_AUTUMN_GRASS 2
#define TILE_WINTER_GRASS 3
#define TILE_SPRING_DIRT 4
#define TILE_SUMMER_DIRT 5
#define TILE_AUTUMN_DIRT 6
#define TILE_WINTER_DIRT 7
#define TILE_SPRING_STONE 8
#define TILE_SUMMER_STONE 9
#define TILE_AUTUMN_STONE 10
#define TILE_WINTER_STONE 11
#define TILE_SPRING_WATER 12
#define TILE_SUMMER_WATER 13
#define TILE_AUTUMN_WATER 14
#define TILE_WINTER_WATER 15

#define RENDER_MODE_FIXED 0
#define RENDER_MODE_CENTERED 1

char* KEYS[12] = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="
};

int TILE_UV[16][2] = {
    {0, 8*TILE_SIZE},
    {0, 13*TILE_SIZE},
    {4*TILE_SIZE, 8*TILE_SIZE},
    {2*TILE_SIZE, 8*TILE_SIZE},
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 0},
    {8*TILE_SIZE, 3*TILE_SIZE},
    {8*TILE_SIZE, 3*TILE_SIZE},
    {8*TILE_SIZE, 3*TILE_SIZE},
    {8*TILE_SIZE, 3*TILE_SIZE},
    {0, 4*TILE_SIZE},
    {0, 4*TILE_SIZE},
    {0, 4*TILE_SIZE},
    {0, 4*TILE_SIZE},
};

typedef struct Animation Animation;
struct Animation {
    int num_frames;
    int tiles_traveled;
    int offset; // Number of tiles from the top of the sheet
    int frames[10]; // Order of frames in sheet, left to right
};

Animation ANIMATIONS[7] = {
    (Animation){ // ANIM_IDLE
        .num_frames = 1,
        .tiles_traveled = 0,
        .offset = 0,
        .frames = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    },
    (Animation){ // ANIM_MOVE
        .num_frames = 6,
        .tiles_traveled = 1,
        .offset = 4,
        .frames = {0, 1, 2, 3, 4, 5, 0, 0, 0, 0}
    },
    (Animation){ // ANIM_ATTACK
        .num_frames = 2,
        .tiles_traveled = 0,
        .offset = 0,
        .frames = {1, 2, 0, 0, 0, 0, 0, 0, 0, 0}
    },
    (Animation){ // ANIM_SWORD
        .num_frames = 8,
        .tiles_traveled = 0,
        .offset = 8,
        .frames = {0, 1, 6, 3, 4, 7, 0, 0, 0, 0}
    },
    (Animation){ // ANIM_BOW
        .num_frames = 8,
        .tiles_traveled = 0,
        .offset = 0,
        .frames = {8, 9, 10, 11, 12, 13, 14, 15, 0, 0}
    },
    (Animation){ // ANIM_DEATH
        .num_frames = 3,
        .tiles_traveled = 0,
        .offset = 0,
        .frames = {5, 6, 7, 0, 0, 0, 0, 0, 0, 0}
    },
    (Animation){ // ANIM_RUN
        .num_frames = 6,
        .tiles_traveled = 2,
        .offset = 4,
        .frames = {0, 1, 6, 3, 4, 7, 0, 0, 0, 0}
    },
};

#define TEX_FULL -2
#define TEX_EMPTY -1
#define TEX_TL_CORNER 0
#define TEX_T_FLAT 1
#define TEX_TR_CORNER 2
#define TEX_L_FLAT 3
#define TEX_CENTER 4
#define TEX_R_FLAT 5
#define TEX_BL_CORNER 6
#define TEX_B_FLAT 7
#define TEX_BR_CORNER 8
#define TEX_TL_DIAG 9
#define TEX_TR_DIAG 10
#define TEX_BL_DIAG 11
#define TEX_BR_DIAG 12
#define TEX_TRR_DIAG 13
#define TEX_BRR_DIAG 14

#define OFF 20
#define GRASS_OFFSET 512+32
#define WATER_OFFSET OFF * 2
#define STONE_OFFSET OFF * 1
#define DIRT_OFFSET  0

void render_conversion(char* flat_tiles, int* flat_converted, int R, int C) {
    char* tex_codes = tile_atlas;
    char (*tiles)[C] = (char(*)[C])flat_tiles;
    int (*converted)[C] = (int(*)[C])flat_converted;

    for (int r = 1; r < R-1; r++) {
        for (int c = 1; c < C-1; c++) {
            int tile = tiles[r][c];
            assert(flat_tiles[r*C + c] == tile);
            int byte_code = 0;
            if (is_grass(tile)) {
                byte_code = 255;
            } else {
                if (tiles[r-1][c-1] != tile) {
                    byte_code += 128;
                }
                if (tiles[r-1][c] != tile) {
                    byte_code += 64;
                }
                if (tiles[r-1][c+1] != tile) {
                    byte_code += 32;
                }
                if (tiles[r][c-1] != tile) {
                    byte_code += 16;
                }
                if (tiles[r][c+1] != tile) {
                    byte_code += 8;
                }
                if (tiles[r+1][c-1] != tile) {
                    byte_code += 4;
                }
                if (tiles[r+1][c] != tile) {
                    byte_code += 2;
                }
                if (tiles[r+1][c+1] != tile) {
                    byte_code += 1;
                }
            }

            // Code maps local tile regions to a snapping tile index
            int code = tex_codes[byte_code];
            int idx = code;
            if (code == TEX_FULL) {
                if (is_dirt(tile)) {
                    idx = DIRT_OFFSET + rand() % 5;
                } else if (is_stone(tile)) {
                    idx = STONE_OFFSET + rand() % 5;
                } else if (is_water(tile)) {
                    idx = WATER_OFFSET + rand() % 5;
                }
            } else if (is_dirt(tile)) {
                idx += DIRT_OFFSET + 5;
            } else if (is_stone(tile)) {
                idx += STONE_OFFSET + 5;
            } else if (is_water(tile)) {
                idx += WATER_OFFSET + 5;
            }

            if (!is_grass(tile)) {
                if (tile == TILE_SUMMER_DIRT || tile == TILE_SUMMER_STONE
                        || tile == TILE_SUMMER_WATER) {
                    idx += 3*OFF;
                } else if (tile == TILE_AUTUMN_DIRT || tile == TILE_AUTUMN_STONE
                        || tile == TILE_AUTUMN_WATER) {
                    idx += 6*OFF;
                } else if (tile == TILE_WINTER_DIRT || tile == TILE_WINTER_STONE
                        || tile == TILE_WINTER_WATER) {
                    idx += 9*OFF;
                }
            }
            if (is_grass(tile) || code == TEX_EMPTY) {
                int num_spring = 0;
                int num_summer = 0;
                int num_autumn = 0;
                int num_winter = 0;
                for (int rr = r-1; rr <= r+1; rr++) {
                    for (int cc = c-1; cc <= c+1; cc++) {
                        int tile = tiles[rr][cc];
                        if (tile == TILE_SPRING_GRASS) {
                            num_spring += 1;
                        } else if (tile == TILE_SUMMER_GRASS) {
                            num_summer += 1;
                        } else if (tile == TILE_AUTUMN_GRASS) {
                            num_autumn += 1;
                        } else if (tile == TILE_WINTER_GRASS) {
                            num_winter += 1;
                        }
                    }
                }

                if (num_spring == 0 && num_summer == 0
                        && num_autumn == 0 && num_winter == 0) {
                    idx = 240;
                } else {
                    int lookup = (1000*num_spring + 100*num_summer
                        + 10*num_autumn + num_winter);
                    int offset = (rand() % 4) * 714; // num_lerps;
                    idx = lerps[lookup] + offset + 240 + 5*4*3*4;
                }
            }
            if (code == TEX_FULL && is_water(tile)) {
                int variant = (rand() % 5);
                int anim = rand() % 3;
                idx = 240 + 3*4*4*variant + 4*4*anim;
                if (tile == TILE_SPRING_WATER) {
                    idx += 0;
                } else if (tile == TILE_SUMMER_WATER) {
                    idx += 4;
                } else if (tile == TILE_AUTUMN_WATER) {
                    idx += 8;
                } else if (tile == TILE_WINTER_WATER) {
                    idx += 12;
                }
            }
            converted[r][c] = idx;
        }
    }
}

Client* make_client(MMO* env) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "NMMO3");
    SetTargetFPS(FRAME_RATE);

    Client* client = calloc(1, sizeof(Client));
    client->start_time = time(NULL);
    client->render_delta = 1.0/TICK_FRAMES;
    client->command_len = 0;

    client->terrain = calloc(env->height*env->width, sizeof(int));
    render_conversion(env->terrain, client->terrain, env->height, env->width);

    client->shader = LoadShader("", TextFormat("resources/nmmo3/map_shader_%i.fs", GLSL_VERSION));

    // TODO: These should be int locs?
    client->shader_map_width_loc = GetShaderLocation(client->shader, "map_width");
    client->shader_map_height_loc = GetShaderLocation(client->shader, "map_height");
    client->shader_camera_x_loc = GetShaderLocation(client->shader, "camera_x");
    client->shader_camera_y_loc = GetShaderLocation(client->shader, "camera_y");
    client->shader_time_loc = GetShaderLocation(client->shader, "time");
    client->shader_resolution_loc = GetShaderLocation(client->shader, "resolution");
    client->shader_texture_tiles_loc = GetShaderLocation(client->shader, "texture_tiles");
    client->shader_terrain_loc = GetShaderLocation(client->shader, "terrain");
    Image img = GenImageColor(env->width, env->height, WHITE);
    ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    client->shader_terrain = LoadTextureFromImage(img);
    UnloadImage(img);
    client->shader_terrain_data = malloc(env->width*env->height*4);
    //SetShaderValue(client->shader, client->shader_terrain_loc, &client->terrain, SHADER_UNIFORM_INT);
   
    for (int i = 0; i < env->width*env->height; i++) {
        int tile = client->terrain[i];
        //if (tile >= 240 && tile < 240+4*4*4*4) {
        //    tile += 3.9*delta;
        //}
 
        client->shader_terrain_data[4*i] = tile/64;
        client->shader_terrain_data[4*i+1] = tile%64;
        //client->shader_terrain_data[2*i] = 0;
        //client->shader_terrain_data[2*i+1] = 0;
        client->shader_terrain_data[4*i+2] = 0;
        client->shader_terrain_data[4*i+3] = 255;
    }

    client->render_mode = RENDER_MODE_CENTERED;
    client->tiles = LoadTexture("resources/nmmo3/merged_sheet.png");

    //client->players = LoadTexture("../resource/neutral_0.png");
    client->items = LoadTexture("resources/nmmo3/items_condensed.png");
    client->inventory = LoadTexture("resources/nmmo3/inventory_64.png");
    client->inventory_equip = LoadTexture("resources/nmmo3/inventory_64_selected.png");
    client->inventory_selected = LoadTexture("resources/nmmo3/inventory_64_press.png");
    client->font = LoadFont("resources/nmmo3/ManaSeedBody.ttf");
    for (int i = 0; i < NUM_PLAYER_TEXTURES; i++) {
        client->players[0][i] = LoadTexture(TextFormat("resources/nmmo3/neutral_%d.png", i));
        client->players[1][i] = LoadTexture(TextFormat("resources/nmmo3/fire_%d.png", i));
        client->players[2][i] = LoadTexture(TextFormat("resources/nmmo3/water_%d.png", i));
        client->players[3][i] = LoadTexture(TextFormat("resources/nmmo3/earth_%d.png", i));
        client->players[4][i] = LoadTexture(TextFormat("resources/nmmo3/air_%d.png", i));
    }

    // TODO: Why do I need to cast here?
    client->camera = (Camera2D){
        .target = {.x = env->width/2*TILE_SIZE, .y = env->height/2*TILE_SIZE},
        .offset = {.x = 0.0, .y = 0.0},
        .rotation = 0.0,
        .zoom = 1.0,
    };

    int buffer_width = SCREEN_WIDTH + 4*TILE_SIZE;
    int buffer_height = SCREEN_HEIGHT + 4*TILE_SIZE;
    client->map_buffer = LoadRenderTexture(buffer_width, buffer_height);
    client->ui_buffer = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT);

    return client;
}

void close_client(Client* client) {
    UnloadRenderTexture(client->map_buffer);
    UnloadRenderTexture(client->ui_buffer);
    for (int i = 0; i < NUM_PLAYER_TEXTURES; i++) {
        for (int element = 0; element < 5; element++) {
            UnloadTexture(client->players[element][i]);
        }
    }
    UnloadFont(client->font);
    UnloadTexture(client->tiles);
    UnloadTexture(client->items);
    UnloadTexture(client->inventory);
    UnloadTexture(client->inventory_equip);
    UnloadTexture(client->inventory_selected);
    UnloadTexture(client->shader_terrain);
    free(client->shader_terrain_data);
    CloseWindow();
}
        
void draw_health_bar(int bar_x, int bar_y, int health, int max_health) {
    DrawRectangle(bar_x, bar_y, HEALTH_BAR_WIDTH,
        HEALTH_BAR_HEIGHT, RED);
    DrawRectangle(bar_x, bar_y,
        HEALTH_BAR_WIDTH * health / max_health,
        HEALTH_BAR_HEIGHT, GREEN);
    DrawRectangleLines(bar_x, bar_y, HEALTH_BAR_WIDTH,
        HEALTH_BAR_HEIGHT, BLACK);
}

void draw_inventory_item(Client* client, int idx, int item_type) {
    if (item_type == 0) {
        return;
    }
    Vector2 pos = {
        .x = TILE_SIZE*idx + TILE_SIZE/4,
        .y = SCREEN_HEIGHT - 5*TILE_SIZE/4,
    };
    Rectangle source_rect = {
        .x = TILE_SIZE*(ITEMS[item_type].tier - 1),
        .y = TILE_SIZE*(ITEMS[item_type].type - 1),
        .width = TILE_SIZE,
        .height = TILE_SIZE,
    };
    DrawTextureRec(client->items, source_rect, pos, WHITE);
}

void draw_inventory_slot(Client* client, int idx, Texture2D* tex) {
    Vector2 pos = {
        .x = TILE_SIZE*idx + TILE_SIZE/4,
        .y = SCREEN_HEIGHT - 5*TILE_SIZE/4,
    };
    Rectangle source_rect = {
        .x = 0,
        .y = 0,
        .width = TILE_SIZE,
        .height = TILE_SIZE,
    };
    DrawTextureRec(*tex, source_rect, pos, WHITE);
}

void draw_inventory_label(Client* client, int idx, const char* label) {
    Vector2 pos = {
        .x = TILE_SIZE*idx + TILE_SIZE/2,
        .y = SCREEN_HEIGHT - 5*TILE_SIZE/4 - 20,
    };
    DrawTextEx(client->font, label, pos, 20, 4, YELLOW);
}

void draw_all_slots(Client* client, Entity* player, int action) {
    for (int i = 0; i < 12; i++) {
        Texture2D* tex;
        int mode = player->ui_mode;
        if (i == action - ATN_ONE) {
            tex = &client->inventory_selected;
        } else if ((mode==MODE_PLAY || mode==MODE_SELL_SELECT) &&
                player->is_equipped[i] == 1) {
            tex = &client->inventory_equip;
        } else {
            tex = &client->inventory;
        }
        //TODO: Draw inventory slot
        draw_inventory_slot(client, i, tex);
    }
}

void draw_ui(Client* client, MMO* env, Entity* player, int action) {
    draw_all_slots(client, player, action);

    int mode = player->ui_mode;
    if (mode == MODE_PLAY || mode == MODE_SELL_SELECT) {
        for (int idx = 0; idx < INVENTORY_SIZE; idx++) {
            int item_type = player->inventory[idx];
            draw_inventory_item(client, idx, item_type);
        }
    } else if (mode == MODE_SELL_PRICE) {
        for (int tier = 0; tier < 5; tier++) {
            int item_type = item_index(I_SILVER, tier+1);
            draw_inventory_item(client, tier, item_type);
        }
        for (int tier = 0; tier < 5; tier++) {
            int item_type = item_index(I_GOLD, tier+1);
            draw_inventory_item(client, tier+5, item_type);
        }
        for (int idx = 0; idx < 10; idx++) {
            int price = idx + 1;
            draw_inventory_label(client, idx, TextFormat("$%d", price));
        }
    } else if (mode == MODE_BUY_TIER) {
        for (int tier = 0; tier < MAX_TIERS; tier++) {
            int item_type = item_index(I_SWORD, tier+1);
            draw_inventory_item(client, tier, item_type);
            draw_inventory_label(client, tier, TextFormat("T%d", tier+1));
        }
    } else if (mode == MODE_BUY_ITEM) {
        int tier = player->market_tier;
        for (int idx = 0; idx < 11; idx++) {
            int item_id = I_N*(tier-1) + idx + 1;
            draw_inventory_item(client, idx, item_id);
            // TODO: add prices to obs
            int price = peek_price(&env->market[item_id]);
            //price = extra_player_ob[idx + P.INVENTORY];
            const char* label = (price == 0) ? "Out!" : TextFormat("$%d", price);
            draw_inventory_label(client, idx, label);
        }
    }

    // Draw number keys
    for (int i = 0; i < 12; i++) {
        Vector2 pos = {
            .x = TILE_SIZE*i + TILE_SIZE/2 - 4,
            .y = SCREEN_HEIGHT - TILE_SIZE/2 - 12,
        };
        DrawTextEx(client->font, KEYS[i], pos, 20, 0, YELLOW);
    }

    if (mode != MODE_PLAY) {
        char* label;
        if (mode == MODE_BUY_TIER || mode == MODE_BUY_ITEM) {
            label = (char*) TextFormat("Buy Mode (b=cancel)\n\nYour gold: $%d", player->gold);
        } else {
            label = (char*) TextFormat("Sell Mode (v=cancel)");
        }

        Vector2 pos = {
            .x = TILE_SIZE/2,
            .y = SCREEN_HEIGHT - 2.5*TILE_SIZE,
        };
        DrawTextEx(client->font, label, pos, 20, 4, YELLOW);
    }

    if (player->in_combat > 0) {
        Vector2 pos = {
            .x = SCREEN_WIDTH - 500,
            .y = TILE_SIZE/2,
        };
        DrawTextEx(client->font, TextFormat("In combat. Cannot equip items."),
            pos, 20, 4, RED);
    }
}

int simple_hash(int n) {
    return ((n * 2654435761) & 0xFFFFFFFF) % INT_MAX;
}

void draw_entity(Client* client, MMO* env, int pid, float delta) {
    Entity* entity = get_entity(env, pid);
    Animation* animation = &ANIMATIONS[entity->anim];

    // Player texture
    int element = entity->element;
    int hashed = simple_hash(pid + client->start_time % 100);
    Texture2D* tex = &client->players[element][hashed % NUM_PLAYER_TEXTURES];
 
    int frame = delta * animation->num_frames;
    Rectangle source_rect = {
        .x = SPRITE_SIZE*animation->frames[frame],
        .y = SPRITE_SIZE*(animation->offset + entity->dir),
        .width = SPRITE_SIZE,
        .height = SPRITE_SIZE,
    };

    float dx = 0;
    float dy = 0;
    if (entity->dir == 0) {
        dy = -animation->tiles_traveled;
    } else if (entity->dir == 1) {
        dy = animation->tiles_traveled;
    } else if (entity->dir == 2) {
        dx = -animation->tiles_traveled;
    } else if (entity->dir == 3) {
        dx = animation->tiles_traveled;
    }
    dx = (1.0 - delta) * dx;
    dy = (1.0 - delta) * dy;

    int x_pos = (dx + entity->c - 0.5f)*TILE_SIZE;
    int y_pos = (dy + entity->r - 0.5f)*TILE_SIZE;
    Vector2 pos = {.x = x_pos, .y = y_pos};

    DrawTextureRec(*tex, source_rect, pos, WHITE);

    // Health bar
    int bar_x = x_pos + TILE_SIZE - HEALTH_BAR_WIDTH/2;
    int bar_y = y_pos;
    draw_health_bar(bar_x, bar_y, entity->hp, entity->hp_max);

    // Overhead text
    int comb_lvl = entity->comb_lvl;
    int prof_lvl = entity->prof_lvl;
    char* txt;
    Color color;
    if (entity->type == ENTITY_PLAYER) {
        txt = (char*) TextFormat("%d: Lv %d/%d", pid, comb_lvl, prof_lvl);
        color = GREEN;
    } else {
        txt = (char*) TextFormat("%d: Lv %d", pid, comb_lvl);
        color = RED;
    }

    Vector2 text_pos = {.x = bar_x, .y = bar_y - 20};
    DrawTextEx(client->font, txt, text_pos, 14, 1, color);
}

void draw_min(Client* client, MMO* env, int x, int y,
        int width, int height, int C, int R, float scale, float delta) {
    client->shader_resolution[0] = GetRenderWidth();
    client->shader_resolution[1] = GetRenderHeight();
    client->shader_resolution[2] = client->camera.zoom;
    client->shader_camera_x = client->camera.target.x;
    client->shader_camera_y = client->camera.target.y;
    client->shader_time = delta;

    BeginShaderMode(client->shader);
    float map_width = env->width;
    float map_height = env->height;
    SetShaderValue(client->shader, client->shader_map_width_loc, &map_width, SHADER_UNIFORM_FLOAT);
    SetShaderValue(client->shader, client->shader_map_height_loc, &map_height, SHADER_UNIFORM_FLOAT);
    SetShaderValue(client->shader, client->shader_camera_x_loc, &client->shader_camera_x, SHADER_UNIFORM_FLOAT);
	SetShaderValue(client->shader, client->shader_camera_y_loc, &client->shader_camera_y, SHADER_UNIFORM_FLOAT);
	SetShaderValue(client->shader, client->shader_time_loc, &client->shader_time, SHADER_UNIFORM_FLOAT);
    SetShaderValue(client->shader, client->shader_resolution_loc, client->shader_resolution, SHADER_UNIFORM_VEC3);

    SetShaderValueTexture(client->shader, client->shader_texture_tiles_loc, client->tiles);

    UpdateTexture(client->shader_terrain, client->shader_terrain_data);
    SetShaderValueTexture(client->shader, client->shader_terrain_loc, client->shader_terrain);

    DrawRectangle(
        client->camera.target.x - GetRenderWidth()/2/client->camera.zoom,
        client->camera.target.y - GetRenderHeight()/2/client->camera.zoom,
        GetRenderWidth()/client->camera.zoom,
        GetRenderHeight()/client->camera.zoom,
        WHITE
    );

    EndShaderMode();

    for (int r = y; r < y+height; r++) {
        for (int c = x; c < x+width; c++) {
            int adr = r*C + c;
            int tile = client->terrain[adr];
            if (tile >= 240 && tile < 240+4*4*4*4) {
                tile += 3.9*delta;
            }
            //int u = TILE_SIZE*(tile % 64);
            //int v = TILE_SIZE*(tile / 64);
            Vector2 pos = {
                .x = c*TILE_SIZE,
                .y = r*TILE_SIZE,
            };
            if (IsKeyDown(KEY_H) && env->pids[adr] != -1) {
                DrawRectangle(pos.x, pos.y, TILE_SIZE, TILE_SIZE, (Color){0, 255, 255, 128});
            }
            /*
            Rectangle source_rect = (Rectangle){
                .x = u,
                .y = v,
                .width = TILE_SIZE,
                .height = TILE_SIZE,
            };

            DrawTextureRec(client->tiles, source_rect, pos, (Color){255, 255, 255, 128});
            */

            // Draw item
            if (env->items[adr] != 0) {
                int item_id = env->items[adr];
                int item_tier = ITEMS[item_id].tier;
                int item_type = ITEMS[item_id].type;
                Rectangle source_rect = {
                    .x = (item_tier - 1)*TILE_SIZE,
                    .y = (item_type - 1)*TILE_SIZE,
                    .width = TILE_SIZE,
                    .height = TILE_SIZE,
                };
                DrawTextureRec(client->items, source_rect, pos, WHITE);
            }
        }
    }

}

void render_centered(Client* client, MMO* env, int pid, int action, float delta) {
    Entity* player = get_entity(env, pid);
    int r = player->r;
    int c = player->c;

    //Animation* animation = ANIM_SPRITE[player->anim];
    //float travel_x, travel_y;
    //travel_x, travel_y = animation.get_travel(player.dir)
    float travel_x = 0;
    float travel_y = 0;
    Animation* animation = &ANIMATIONS[player->anim];
    if (player->dir == 0) {
        travel_y = -animation->tiles_traveled;
    } else if (player->dir == 1) {
        travel_y = animation->tiles_traveled;
    } else if (player->dir == 2) {
        travel_x = animation->tiles_traveled;
    } else if (player->dir == 3) {
        travel_x = -animation->tiles_traveled;
    }
    travel_x *= TILE_SIZE;
    travel_y *= TILE_SIZE;
 
    client->camera.offset.x = SCREEN_WIDTH/2;
    client->camera.offset.y = SCREEN_HEIGHT/2;
    client->camera.target.x = (c + 0.5)*TILE_SIZE + (delta - 1)*travel_x;
    client->camera.target.y = (r + 0.5)*TILE_SIZE + (1 - delta)*travel_y;
    client->camera.zoom = 1.0;

    int start_c = c - X_WINDOW - 2;
    if (start_c < 0) {
        start_c = 0;
    }
    
    int start_r = r - Y_WINDOW - 2;
    if (start_r < 0) {
        start_r = 0;
    }

    int end_r = r + Y_WINDOW + 3;
    if (end_r > env->height) {
        end_r = env->height;
    }

    int end_c = c + X_WINDOW + 3;
    if (end_c > env->width) {
        end_c = env->width;
    }

    //BeginMode2D(client.camera);
    BeginMode2D(client->camera);
    draw_min(client, env, start_c,
        start_r, end_c-start_c, end_r-start_r,
        env->width, env->height, 1, delta);

    for (int pid = 0; pid < env->num_players+env->num_enemies; pid++) {
        draw_entity(client, env, pid, delta);
    }

    EndMode2D();
    draw_ui(client, env, player, action);
}

bool up_key() {
    return IsKeyDown(KEY_UP) || IsKeyDown(KEY_W);
}

bool down_key() {
    return IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S);
}

bool left_key() {
    return IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A);
}

bool right_key() {
    return IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D);
}

bool shift_key() {
    return IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
}

int process_centered_input() {
    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
    }

    if (shift_key()) {
        if (down_key()) {
            return ATN_DOWN_SHIFT;
        } else if (up_key()) {
            return ATN_UP_SHIFT;
        } else if (left_key()) {
            return ATN_LEFT_SHIFT;
        } else if (right_key()) {
            return ATN_RIGHT_SHIFT;
        }
    } else if (up_key()) {
        return ATN_UP;
    } else if (down_key()) {
        return ATN_DOWN;
    } else if (left_key()) {
        return ATN_LEFT;
    } else if (right_key()) {
        return ATN_RIGHT;
    } else if (IsKeyDown(KEY_SPACE)) {
        return ATN_ATTACK;
    } else if (IsKeyDown(KEY_ONE)) {
        return ATN_ONE;
    } else if (IsKeyDown(KEY_TWO)) {
        return ATN_TWO;
    } else if (IsKeyDown(KEY_THREE)) {
        return ATN_THREE;
    } else if (IsKeyDown(KEY_FOUR)) {
        return ATN_FOUR;
    } else if (IsKeyDown(KEY_FIVE)) {
        return ATN_FIVE;
    } else if (IsKeyDown(KEY_SIX)) {
        return ATN_SIX;
    } else if (IsKeyDown(KEY_SEVEN)) {
        return ATN_SEVEN;
    } else if (IsKeyDown(KEY_EIGHT)) {
        return ATN_EIGHT;
    } else if (IsKeyDown(KEY_NINE)) {
        return ATN_NINE;
    } else if (IsKeyDown(KEY_ZERO)) {
        return ATN_ZERO;
    } else if (IsKeyDown(KEY_MINUS)) {
        return ATN_MINUS;
    } else if (IsKeyDown(KEY_EQUAL)) {
        return ATN_EQUALS;
    } else if (IsKeyDown(KEY_V)) {
        return ATN_SELL;
    } else if (IsKeyDown(KEY_B)) {
        return ATN_BUY;
    }
    return ATN_NOOP;
}

void process_fixed_input(Client* client) {
    float move_speed = 20 / client->camera.zoom;
    float zoom_delta = 0.05;
    float zoom = client->camera.zoom;
    if (shift_key()) {
        move_speed *= 2;
        zoom_delta *= 2;
    }
    if (down_key()) {
        client->camera.target.y += move_speed;
    }
    if (up_key()) {
        client->camera.target.y -= move_speed;
    }
    if (left_key()) {
        client->camera.target.x -= move_speed;
    }
    if (right_key()) {
        client->camera.target.x += move_speed;
    }
    if ((IsKeyDown(KEY_EQUAL) || IsKeyDown(KEY_E)) && zoom < 8.0) {
        client->camera.zoom *= (1 + zoom_delta);
    }
    if ((IsKeyDown(KEY_MINUS) || IsKeyDown(KEY_Q)) && zoom > 1.0/32.0) {
        client->camera.zoom *= (1 - zoom_delta);
    }
}

void render_fixed(Client* client, MMO* env, float delta) {
    // Draw tilemap
    float y = client->camera.target.y;
    float x = client->camera.target.x;
    float zoom = client->camera.zoom;

    BeginMode2D(client->camera);

    int X = GetRenderWidth();
    int Y = GetRenderHeight();
    client->camera.offset.x = X/2;
    client->camera.offset.y = Y/2;

    int start_r = (y - Y/2/zoom) / TILE_SIZE;
    if (start_r < 0) {
        start_r = 0;
    }

    int start_c = (x - X/2/zoom) / TILE_SIZE;
    if (start_c < 0) {
        start_c = 0;
    }

    int end_r = (y + Y/2/zoom) / TILE_SIZE + 1;
    if (end_r > env->height) {
        end_r = env->height;
    }

    int end_c = (x + X/2/zoom) / TILE_SIZE + 1;
    if (end_c > env->width) {
        end_c = env->width;
    }

    /*
    if client.active_overlay is None:
        overlay = None
    elif client.active_overlay == 'counts':
        overlay = client.overlays.counts
        overlay = smooth_cyan(overlay)
        overlay = overlay[start_r:end_r, start_c:end_c]
    elif client.active_overlay == 'value':
        overlay = client.overlays.value_function
        overlay = clip_rgb(overlay)
        overlay = overlay[start_r:end_r, start_c:end_c]
    */

    draw_min(client, env, start_c, start_r,
        end_c-start_c, end_r-start_r, env->width, env->height, 1, delta);

    for (int pid = 0; pid < env->num_players+env->num_enemies; pid++) {
        draw_entity(client, env, pid, delta);
    }

    EndMode2D();
}

// Did not finish porting console from Cython
void process_command_input(Client* client, MMO* env) {
    int key = GetCharPressed();
    while (key > 0) {
        if (key >= 32 && key <= 125 && client->command_len < COMMAND_CHARS) {
            client->command[client->command_len] = key;
            client->command_len += 1;
        }
        key = GetCharPressed();
    }
    if (IsKeyPressed(KEY_BACKSPACE)) {
        client->command_len = client->command_len - 1;
    }
    if (IsKeyPressed(KEY_ENTER)) {
        char* command = client->command;
        client->command_len = 0;

        if (client->command_len == 5 && strncmp(command, "help", 5) == 0) {
            //client->command = COMMAND_HELP;
        } else {
            client->command_mode = false;
        }

        if (client->command_len == 11 && strncmp(command, "overlay env", 11) == 0) {
            client->active_overlay = OVERLAY_NONE;
        } else if (client->command_len == 14 && strncmp(command, "overlay counts", 14) == 0) {
            client->active_overlay = OVERLAY_COUNTS;
            //arr = smooth_cyan(client->overlays.counts);
            //Image.fromarray(arr).save('overlays/counts.png');
            //client->overlay_texture = rl.LoadTexture('overlays/counts.png'.encode());
        } else if (client->command_len == 13 && strncmp(command, "overlay value", 13) == 0) {
            client->active_overlay = OVERLAY_VALUE;
            //arr = clip_rgb(client->overlays.value_function);
            //Image.fromarray(arr).save('overlays/values.png');
            //client->overlay_texture = rl.LoadTexture('overlays/values.png'.encode());
        } else if (client->command_len == 4 && strncmp(command, "play", 4) == 0) {
            client->my_player = 0;
            client->render_mode = RENDER_MODE_CENTERED;
        } else if (client->command_len >= 9 && strncmp(command, "follow ", 7) == 0) {
            /*
            char* pid = command + 7;
            pid = pid;
            int pid = atoi(pid);
            if (pid < 0 || pid > env->num_players) {
                client->command = "Invalid player id";
            }
            client->my_player = pid;
            client->render_mode = RENDER_MODE_CENTERED;
            */
        }
    }

    Color term_color = {255, 255, 255, 200};
    DrawRectangle(0, 0, SCREEN_WIDTH, 32, term_color);
    client->command[client->command_len] = '\0';
    const char* text = TextFormat("> %s", client->command);
    DrawText(text, 10, 10, 20, BLACK);
}

int c_render(MMO* env) {
    if (env->client == NULL) {
        // Must reset before making client
        env->client = make_client(env);
    }
    Client* client = env->client;
    float delta = client->render_delta;

    BeginDrawing();
    ClearBackground(BLANK);
    int action = 0;

    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleBorderlessWindowed();
        if (client->render_mode == RENDER_MODE_CENTERED) {
            client->render_mode = RENDER_MODE_FIXED;
        } else {
            client->render_mode = RENDER_MODE_CENTERED;
        }
    }
    if (IsKeyPressed(KEY_GRAVE)) { // tilde
        client->command_mode = !client->command_mode;
        GetCharPressed(); // clear tilde key
    }
    if (client->render_mode == RENDER_MODE_FIXED) {
        if (!client->command_mode) {
            process_fixed_input(client);
        }
        render_fixed(client, env, delta);
    } else {
        if (!client->command_mode) {
            action = process_centered_input();
        }
        render_centered(client, env, client->my_player, action, delta);
    }
    if (client->command_mode) {
        process_command_input(client, env);
    }

    if (IsKeyDown(KEY_H)) {
        DrawTextEx(client->font, TextFormat("FPS: %d", GetFPS()),
            (Vector2){16, 16}, 24, 4, YELLOW);
    }

    EndDrawing();
    return action;
}


