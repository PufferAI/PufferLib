// Craftax-Classic environment for PufferLib Ocean.
//
// Single-header per-env implementation. PufferLib's vec layer owns the
// observation/action/reward/terminal buffers and parallelizes c_step
// across env instances via OpenMP; this file never allocates its own
// threads or batches.
//
// Game rules follow Matthews et al. 2024 "Craftax-Classic" (ICML 2024).
// This port is derived from the CPU port at github.com/Infatoshi/craftax.c
// (47.8M SPS standalone), restructured to match the Ocean conventions
// used by breakout/drmario/etc.
//
// Observation: 1345 float32:
//   - 63 tiles (7x9 local view) x 21 channels (17 block one-hot + 4 mob) = 1323
//   - 12 inventory (0..9) / 10
//   -  4 intrinsics (health, food, drink, energy / 10)
//   -  4 direction one-hot
//   -  1 light level [0, 1]
//   -  1 is_sleeping {0, 1}
// Matches the JAX/CUDA Craftax-Classic-Symbolic-v1 layout exactly.
//
// Action: 1 discrete in 0..16 (NOOP, 4 moves, DO, SLEEP,
//         4 place, 3 make-pick, 3 make-sword).

#pragma once
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>
#include "raylib.h"

// ============================================================
// Constants
// ============================================================
#define MAP_SIZE 64
#define MAP_PACKED_ROW 32
#define MAP_PACKED_SIZE (MAP_SIZE * MAP_PACKED_ROW)

#define MAX_ZOMBIES 3
#define MAX_COWS 3
#define MAX_SKELETONS 2
#define MAX_ARROWS 3
#define MAX_PLANTS 10
#define NUM_ACHIEVEMENTS 22
#define NUM_ACTIONS 17
#define NUM_BLOCK_TYPES 17
#define OBS_DIM 1345
#define NUM_INVENTORY 12
#define MAX_TIMESTEPS 10000
#define DAY_LENGTH 300
#define MOB_DESPAWN_DIST 14

// Block types
#define BLK_INVALID       0
#define BLK_OUT_OF_BOUNDS 1
#define BLK_GRASS         2
#define BLK_WATER         3
#define BLK_STONE         4
#define BLK_TREE          5
#define BLK_WOOD          6
#define BLK_PATH          7
#define BLK_COAL          8
#define BLK_IRON          9
#define BLK_DIAMOND      10
#define BLK_TABLE        11
#define BLK_FURNACE      12
#define BLK_SAND         13
#define BLK_LAVA         14
#define BLK_PLANT        15
#define BLK_RIPE_PLANT   16

// Actions
#define ACT_NOOP          0
#define ACT_LEFT          1
#define ACT_RIGHT         2
#define ACT_UP            3
#define ACT_DOWN          4
#define ACT_DO            5
#define ACT_SLEEP         6
#define ACT_PLACE_STONE   7
#define ACT_PLACE_TABLE   8
#define ACT_PLACE_FURNACE 9
#define ACT_PLACE_PLANT  10
#define ACT_MAKE_WOOD_PICK   11
#define ACT_MAKE_STONE_PICK  12
#define ACT_MAKE_IRON_PICK   13
#define ACT_MAKE_WOOD_SWORD  14
#define ACT_MAKE_STONE_SWORD 15
#define ACT_MAKE_IRON_SWORD  16

// Achievements (index in env->log.achievements[])
#define ACH_COLLECT_WOOD     0
#define ACH_PLACE_TABLE      1
#define ACH_EAT_COW          2
#define ACH_COLLECT_SAPLING  3
#define ACH_COLLECT_DRINK    4
#define ACH_MAKE_WOOD_PICK   5
#define ACH_MAKE_WOOD_SWORD  6
#define ACH_PLACE_PLANT      7
#define ACH_DEFEAT_ZOMBIE    8
#define ACH_COLLECT_STONE    9
#define ACH_PLACE_STONE     10
#define ACH_EAT_PLANT       11
#define ACH_DEFEAT_SKELETON 12
#define ACH_MAKE_STONE_PICK 13
#define ACH_MAKE_STONE_SWORD 14
#define ACH_WAKE_UP         15
#define ACH_PLACE_FURNACE   16
#define ACH_COLLECT_COAL    17
#define ACH_COLLECT_IRON    18
#define ACH_COLLECT_DIAMOND 19
#define ACH_MAKE_IRON_PICK  20
#define ACH_MAKE_IRON_SWORD 21

static const int DIR_DR[5] = {0, 0, 0, -1, 1};
static const int DIR_DC[5] = {0, -1, 1, 0, 0};

// ============================================================
// Tiny PCG-style RNG (single 64-bit state)
// ============================================================
static inline uint32_t cr_pcg(uint64_t* s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x = (uint32_t)(((*s >> 18u) ^ *s) >> 27u);
    uint32_t rot = (uint32_t)(*s >> 59u);
    return (x >> rot) | (x << ((-(int32_t)rot) & 31));
}
static inline float    cr_rf(uint64_t* s)      { return (cr_pcg(s) >> 8) * (1.0f / 16777216.0f); }
static inline int      cr_ri(uint64_t* s, int n) { return (int)(cr_pcg(s) % (uint32_t)n); }

// ============================================================
// PufferLib-required structs
// ============================================================
typedef struct Log {
    float perf;                         // 0-1 normalized progress (achievements / 22)
    float score;                        // sum of episode returns seen so far
    float episode_return;               // last episode return
    float episode_length;               // last episode length
    float achievements[NUM_ACHIEVEMENTS];
    float n;                            // required counter (last field)
} Log;

typedef struct Client {
    int dummy;                          // handled by raylib globally; no per-env handle needed
} Client;

// ============================================================
// Env struct
// ============================================================
typedef struct CraftaxClassic {
    Client* client;
    Log log;

    float* observations;                // (OBS_DIM,) fp32, PufferLib-owned
    float* actions;                     // (1,) fp32
    float* rewards;                     // (1,)
    float* terminals;                   // (1,)

    int num_agents;                     // = 1

    unsigned int rng;                   // populated by default my_vec_init (env index)
    uint64_t pcg;                       // actual RNG state (seeded from rng in my_init)

    // Packed map (2 blocks/byte)
    uint8_t map_packed[MAP_PACKED_SIZE];

    // Per-type occupancy bitmaps: bit c of bits[r] = "mob-type at (r,c)"
    uint64_t mob_bits[MAP_SIZE];        // zombie | cow | skel (used by has_mob_at / can_move_mob)
    uint64_t zombie_bits[MAP_SIZE];
    uint64_t cow_bits[MAP_SIZE];
    uint64_t skel_bits[MAP_SIZE];
    uint64_t arrow_bits[MAP_SIZE];

    // Player
    int16_t player_r, player_c;
    int8_t  player_dir;

    // Intrinsics
    int8_t health, food, drink, energy;
    bool   is_sleeping;
    float  recover, hunger, thirst, fatigue;

    // Inventory (wood, stone, coal, iron, diamond, sapling,
    //            wpick, spick, ipick, wsword, ssword, isword)
    int8_t inv[NUM_INVENTORY];

    // Mobs
    int16_t zombie_r[MAX_ZOMBIES], zombie_c[MAX_ZOMBIES];
    int8_t  zombie_hp[MAX_ZOMBIES], zombie_cd[MAX_ZOMBIES];
    bool    zombie_mask[MAX_ZOMBIES];

    int16_t cow_r[MAX_COWS], cow_c[MAX_COWS];
    int8_t  cow_hp[MAX_COWS];
    bool    cow_mask[MAX_COWS];

    int16_t skel_r[MAX_SKELETONS], skel_c[MAX_SKELETONS];
    int8_t  skel_hp[MAX_SKELETONS], skel_cd[MAX_SKELETONS];
    bool    skel_mask[MAX_SKELETONS];

    int16_t arrow_r[MAX_ARROWS], arrow_c[MAX_ARROWS];
    int8_t  arrow_dr[MAX_ARROWS], arrow_dc[MAX_ARROWS];
    bool    arrow_mask[MAX_ARROWS];

    int16_t plant_r[MAX_PLANTS], plant_c[MAX_PLANTS];
    int16_t plant_age[MAX_PLANTS];
    bool    plant_mask[MAX_PLANTS];

    float   light_level;
    bool    achievements[NUM_ACHIEVEMENTS];
    int32_t timestep;

    // Episode stats (accumulated; flushed into env->log on terminal)
    float episode_return_accum;
    int32_t episode_length_accum;

    // Scratch for per-step reward computation
    int8_t old_health;
    bool   old_achievements[NUM_ACHIEVEMENTS];
} CraftaxClassic;

// ============================================================
// Map accessors + small helpers
// ============================================================
static inline int8_t map_get(const CraftaxClassic* s, int r, int c) {
    int idx = r * MAP_PACKED_ROW + (c >> 1);
    uint8_t b = s->map_packed[idx];
    return (c & 1) ? (int8_t)(b >> 4) : (int8_t)(b & 0x0F);
}
static inline void map_set(CraftaxClassic* s, int r, int c, int8_t v) {
    int idx = r * MAP_PACKED_ROW + (c >> 1);
    uint8_t b = s->map_packed[idx];
    if (c & 1) s->map_packed[idx] = (b & 0x0F) | ((v & 0x0F) << 4);
    else       s->map_packed[idx] = (b & 0xF0) | (v & 0x0F);
}
static inline bool in_bounds(int r, int c) { return (unsigned)r < MAP_SIZE && (unsigned)c < MAP_SIZE; }
static inline bool is_solid(int8_t b) {
    return b == BLK_WATER || b == BLK_STONE || b == BLK_TREE ||
           b == BLK_COAL  || b == BLK_IRON  || b == BLK_DIAMOND ||
           b == BLK_TABLE || b == BLK_FURNACE ||
           b == BLK_PLANT || b == BLK_RIPE_PLANT;
}
static inline int  l1_dist(int r1, int c1, int r2, int c2) {
    int dr = r1 - r2; if (dr < 0) dr = -dr;
    int dc = c1 - c2; if (dc < 0) dc = -dc;
    return dr + dc;
}
static inline int   cr_clamp_i(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
static inline int   cr_min_i(int a,int b){return a<b?a:b;}
static inline int   cr_max_i(int a,int b){return a>b?a:b;}
static inline float cr_min_f(float a,float b){return a<b?a:b;}
static inline int   cr_sign_i(int v){return (v>0)-(v<0);}

// Bitmap maintenance
static inline void mb_set(uint64_t* bits, int r, int c)   { bits[r] |=  (1ULL << c); }
static inline void mb_clear(uint64_t* bits, int r, int c) { bits[r] &= ~(1ULL << c); }
static inline bool mb_get(const uint64_t* bits, int r, int c) { return (bits[r] >> c) & 1ULL; }

static inline bool has_mob_at(const CraftaxClassic* s, int r, int c) {
    if ((unsigned)r >= MAP_SIZE || (unsigned)c >= MAP_SIZE) return false;
    return ((s->mob_bits[r] >> c) & 1ULL) != 0;
}

static bool is_near_block(const CraftaxClassic* s, int8_t blk) {
    int pr = s->player_r, pc = s->player_c;
    static const int dr8[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    static const int dc8[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    for (int i = 0; i < 8; i++) {
        int nr = pr + dr8[i], nc = pc + dc8[i];
        if (in_bounds(nr, nc) && map_get(s, nr, nc) == blk) return true;
    }
    return false;
}

static inline int get_damage(const CraftaxClassic* s) {
    if (s->inv[11] > 0) return 5;
    if (s->inv[10] > 0) return 3;
    if (s->inv[9]  > 0) return 2;
    return 1;
}

// ============================================================
// Perlin worldgen (AVX-512, per-env)
// ============================================================
static inline float perlin_interp(float t) { return t*t*t*(t*(t*6.0f-15.0f)+10.0f); }

#if defined(__clang__) || defined(__GNUC__)
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
#endif
static void generate_world(CraftaxClassic* s) {
    // Reset maps and bitmaps
    for (int i = 0; i < MAP_PACKED_SIZE; i++)
        s->map_packed[i] = (uint8_t)(BLK_GRASS | (BLK_GRASS << 4));
    memset(s->mob_bits,    0, sizeof(s->mob_bits));
    memset(s->zombie_bits, 0, sizeof(s->zombie_bits));
    memset(s->cow_bits,    0, sizeof(s->cow_bits));
    memset(s->skel_bits,   0, sizeof(s->skel_bits));
    memset(s->arrow_bits,  0, sizeof(s->arrow_bits));

    // Perlin gradient tables (precompute cos/sin of the per-grid random angles).
    // Padded by +16 floats so AVX-512 permute-load at the last grid row doesn't
    // read out of bounds.
    enum { GRID = 10, GRID_PAD = GRID * GRID + 16 };
    _Alignas(64) float cos_a[4][GRID_PAD];
    _Alignas(64) float sin_a[4][GRID_PAD];
    for (int layer = 0; layer < 4; layer++) {
        for (int i = 0; i < GRID * GRID; i++) {
            float a = cr_rf(&s->pcg) * 2.0f * 3.14159265f;
            cos_a[layer][i] = cosf(a);
            sin_a[layer][i] = sinf(a);
        }
        for (int i = GRID * GRID; i < GRID_PAD; i++) { cos_a[layer][i] = 0; sin_a[layer][i] = 0; }
    }

    float scale = (float)MAP_SIZE / (float)(GRID - 1);
    float inv_scale = 1.0f / scale;
    int center = MAP_SIZE / 2;

    _Alignas(64) float noise[4][MAP_SIZE][MAP_SIZE];
    {
        const __m512 c_lane = _mm512_setr_ps(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        const __m512 one    = _mm512_set1_ps(1.0f);
        const __m512 half   = _mm512_set1_ps(0.5f);
        const __m512 c6     = _mm512_set1_ps(6.0f);
        const __m512 c15    = _mm512_set1_ps(15.0f);
        const __m512 c10    = _mm512_set1_ps(10.0f);
        const __m512 invs   = _mm512_set1_ps(inv_scale);
        const __m512i i_one = _mm512_set1_epi32(1);

        for (int r = 0; r < MAP_SIZE; r++) {
            float nr = (float)r * inv_scale;
            int x0 = (int)nr;
            float fx = nr - x0;
            float fx1 = fx - 1.0f;
            float u = perlin_interp(fx);
            int row0 = x0 * GRID, row1 = row0 + GRID;
            __m512 fx_v  = _mm512_set1_ps(fx);
            __m512 fx1_v = _mm512_set1_ps(fx1);
            __m512 u_v   = _mm512_set1_ps(u);

            for (int c_base = 0; c_base < MAP_SIZE; c_base += 16) {
                __m512 c_v  = _mm512_add_ps(_mm512_set1_ps((float)c_base), c_lane);
                __m512 nc_v = _mm512_mul_ps(c_v, invs);
                __m512i y0_v = _mm512_cvttps_epi32(nc_v);
                __m512 y0_f = _mm512_cvtepi32_ps(y0_v);
                __m512 fy_v  = _mm512_sub_ps(nc_v, y0_f);
                __m512 fy1_v = _mm512_sub_ps(fy_v, one);
                __m512 t = _mm512_fmsub_ps(fy_v, c6, c15);
                t = _mm512_fmadd_ps(fy_v, t, c10);
                __m512 fy2 = _mm512_mul_ps(fy_v, fy_v);
                __m512 fy3 = _mm512_mul_ps(fy2, fy_v);
                __m512 v_v = _mm512_mul_ps(fy3, t);
                __m512i y1_v = _mm512_add_epi32(y0_v, i_one);

                for (int k = 0; k < 4; k++) {
                    __m512 cos_r0 = _mm512_loadu_ps(&cos_a[k][row0]);
                    __m512 cos_r1 = _mm512_loadu_ps(&cos_a[k][row1]);
                    __m512 sin_r0 = _mm512_loadu_ps(&sin_a[k][row0]);
                    __m512 sin_r1 = _mm512_loadu_ps(&sin_a[k][row1]);

                    __m512 c00 = _mm512_permutexvar_ps(y0_v, cos_r0);
                    __m512 c10v= _mm512_permutexvar_ps(y0_v, cos_r1);
                    __m512 c01 = _mm512_permutexvar_ps(y1_v, cos_r0);
                    __m512 c11 = _mm512_permutexvar_ps(y1_v, cos_r1);
                    __m512 s00 = _mm512_permutexvar_ps(y0_v, sin_r0);
                    __m512 s10 = _mm512_permutexvar_ps(y0_v, sin_r1);
                    __m512 s01 = _mm512_permutexvar_ps(y1_v, sin_r0);
                    __m512 s11 = _mm512_permutexvar_ps(y1_v, sin_r1);

                    __m512 n00 = _mm512_fmadd_ps(c00,  fx_v,  _mm512_mul_ps(s00, fy_v));
                    __m512 n10 = _mm512_fmadd_ps(c10v, fx1_v, _mm512_mul_ps(s10, fy_v));
                    __m512 n01 = _mm512_fmadd_ps(c01,  fx_v,  _mm512_mul_ps(s01, fy1_v));
                    __m512 n11 = _mm512_fmadd_ps(c11,  fx1_v, _mm512_mul_ps(s11, fy1_v));

                    __m512 nx0 = _mm512_fmadd_ps(u_v, _mm512_sub_ps(n10, n00), n00);
                    __m512 nx1 = _mm512_fmadd_ps(u_v, _mm512_sub_ps(n11, n01), n01);
                    __m512 n = _mm512_fmadd_ps(v_v, _mm512_sub_ps(nx1, nx0), nx0);
                    n = _mm512_mul_ps(_mm512_add_ps(n, one), half);

                    _mm512_storeu_ps(&noise[k][r][c_base], n);
                }
            }
        }
    }

    // Tile-logic sweep -- reads precomputed noise, writes blocks
    for (int r = 0; r < MAP_SIZE; r++) {
        for (int c = 0; c < MAP_SIZE; c++) {
            float water_noise    = noise[0][r][c];
            float mountain_noise = noise[1][r][c];
            float tree_noise     = noise[2][r][c];
            float path_noise     = noise[3][r][c];

            float dist = sqrtf((float)((r-center)*(r-center) + (c-center)*(c-center)));
            float prox = 1.0f - cr_min_f(dist / 20.0f, 1.0f);

            float water_val = water_noise - prox * 0.3f;
            float mountain_val = mountain_noise - prox * 0.3f;

            int8_t blk = BLK_GRASS;
            if (water_val > 0.7f) blk = BLK_WATER;
            else if (water_val > 0.6f && water_val <= 0.75f) blk = BLK_SAND;
            else if (mountain_val > 0.7f) {
                blk = BLK_STONE;
                if (path_noise > 0.8f) blk = BLK_PATH;
                if (mountain_val > 0.85f && water_noise > 0.4f) blk = BLK_PATH;
                if (mountain_val > 0.85f && tree_noise > 0.7f)  blk = BLK_LAVA;
            }
            if (blk == BLK_STONE) {
                float ore = cr_rf(&s->pcg);
                if (ore < 0.005f && mountain_val > 0.8f) blk = BLK_DIAMOND;
                else if (ore < 0.035f) blk = BLK_IRON;
                else if (ore < 0.075f) blk = BLK_COAL;
            }
            if (blk == BLK_GRASS && tree_noise > 0.5f && cr_rf(&s->pcg) > 0.8f)
                blk = BLK_TREE;
            map_set(s, r, c, blk);
        }
    }

    map_set(s, center, center, BLK_GRASS);  // player spawn always grass

    bool has_diamond = false;
    for (int r = 0; r < MAP_SIZE && !has_diamond; r++)
        for (int c = 0; c < MAP_SIZE && !has_diamond; c++)
            if (map_get(s, r, c) == BLK_DIAMOND) has_diamond = true;
    if (!has_diamond) {
        for (int att = 0; att < 1000; att++) {
            int r = cr_ri(&s->pcg, MAP_SIZE), c = cr_ri(&s->pcg, MAP_SIZE);
            if (map_get(s, r, c) == BLK_STONE) { map_set(s, r, c, BLK_DIAMOND); break; }
        }
    }

    // Initial intrinsics + inventory + mobs
    s->player_r = center; s->player_c = center; s->player_dir = 4;
    s->health = 9; s->food = 9; s->drink = 9; s->energy = 9;
    s->is_sleeping = false;
    s->recover = s->hunger = s->thirst = s->fatigue = 0;
    memset(s->inv, 0, sizeof(s->inv));
    memset(s->zombie_mask, 0, sizeof(s->zombie_mask));
    memset(s->zombie_hp,   0, sizeof(s->zombie_hp));
    memset(s->zombie_cd,   0, sizeof(s->zombie_cd));
    memset(s->cow_mask, 0, sizeof(s->cow_mask));
    memset(s->cow_hp,   0, sizeof(s->cow_hp));
    memset(s->skel_mask, 0, sizeof(s->skel_mask));
    memset(s->skel_hp,   0, sizeof(s->skel_hp));
    memset(s->skel_cd,   0, sizeof(s->skel_cd));
    memset(s->arrow_mask, 0, sizeof(s->arrow_mask));
    memset(s->plant_mask, 0, sizeof(s->plant_mask));
    memset(s->plant_age,  0, sizeof(s->plant_age));
    memset(s->achievements, 0, sizeof(s->achievements));
    s->timestep = 0;
    s->light_level = 1.0f;
}

// ============================================================
// Step sub-actions
// ============================================================
static void do_crafting(CraftaxClassic* s, int action) {
    bool t = is_near_block(s, BLK_TABLE);
    bool f = is_near_block(s, BLK_FURNACE);
    if (action == ACT_MAKE_WOOD_PICK  && t && s->inv[0] >= 1) { s->inv[0]--; s->inv[6]++; s->achievements[ACH_MAKE_WOOD_PICK] = true; }
    if (action == ACT_MAKE_STONE_PICK && t && s->inv[0] >= 1 && s->inv[1] >= 1) { s->inv[0]--; s->inv[1]--; s->inv[7]++; s->achievements[ACH_MAKE_STONE_PICK] = true; }
    if (action == ACT_MAKE_IRON_PICK  && t && f && s->inv[0] >= 1 && s->inv[1] >= 1 && s->inv[3] >= 1 && s->inv[2] >= 1) {
        s->inv[0]--; s->inv[1]--; s->inv[3]--; s->inv[2]--; s->inv[8]++; s->achievements[ACH_MAKE_IRON_PICK] = true;
    }
    if (action == ACT_MAKE_WOOD_SWORD  && t && s->inv[0] >= 1) { s->inv[0]--; s->inv[9]++;  s->achievements[ACH_MAKE_WOOD_SWORD] = true; }
    if (action == ACT_MAKE_STONE_SWORD && t && s->inv[0] >= 1 && s->inv[1] >= 1) { s->inv[0]--; s->inv[1]--; s->inv[10]++; s->achievements[ACH_MAKE_STONE_SWORD] = true; }
    if (action == ACT_MAKE_IRON_SWORD  && t && f && s->inv[0] >= 1 && s->inv[1] >= 1 && s->inv[3] >= 1 && s->inv[2] >= 1) {
        s->inv[0]--; s->inv[1]--; s->inv[3]--; s->inv[2]--; s->inv[11]++; s->achievements[ACH_MAKE_IRON_SWORD] = true;
    }
}

static void do_action(CraftaxClassic* s) {
    int tr = s->player_r + DIR_DR[s->player_dir];
    int tc = s->player_c + DIR_DC[s->player_dir];
    if (!in_bounds(tr, tc)) return;
    int dmg = get_damage(s);
    bool attacked = false;

    for (int i = 0; i < MAX_ZOMBIES && !attacked; i++)
        if (s->zombie_mask[i] && s->zombie_r[i] == tr && s->zombie_c[i] == tc) {
            s->zombie_hp[i] -= dmg;
            if (s->zombie_hp[i] <= 0) {
                s->zombie_mask[i] = false;
                mb_clear(s->mob_bits, tr, tc); mb_clear(s->zombie_bits, tr, tc);
                s->achievements[ACH_DEFEAT_ZOMBIE] = true;
            }
            attacked = true;
        }
    for (int i = 0; i < MAX_COWS && !attacked; i++)
        if (s->cow_mask[i] && s->cow_r[i] == tr && s->cow_c[i] == tc) {
            s->cow_hp[i] -= dmg;
            if (s->cow_hp[i] <= 0) {
                s->cow_mask[i] = false;
                mb_clear(s->mob_bits, tr, tc); mb_clear(s->cow_bits, tr, tc);
                s->achievements[ACH_EAT_COW] = true;
                s->food = (int8_t)cr_min_i(9, s->food + 6); s->hunger = 0;
            }
            attacked = true;
        }
    for (int i = 0; i < MAX_SKELETONS && !attacked; i++)
        if (s->skel_mask[i] && s->skel_r[i] == tr && s->skel_c[i] == tc) {
            s->skel_hp[i] -= dmg;
            if (s->skel_hp[i] <= 0) {
                s->skel_mask[i] = false;
                mb_clear(s->mob_bits, tr, tc); mb_clear(s->skel_bits, tr, tc);
                s->achievements[ACH_DEFEAT_SKELETON] = true;
            }
            attacked = true;
        }
    if (attacked) return;

    int8_t blk = map_get(s, tr, tc);
    switch (blk) {
        case BLK_TREE:
            map_set(s, tr, tc, BLK_GRASS);
            s->inv[0] = (int8_t)cr_min_i(9, s->inv[0] + 1);
            s->achievements[ACH_COLLECT_WOOD] = true; break;
        case BLK_STONE:
            if (s->inv[6] > 0 || s->inv[7] > 0 || s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[1] = (int8_t)cr_min_i(9, s->inv[1] + 1);
                s->achievements[ACH_COLLECT_STONE] = true;
            } break;
        case BLK_COAL:
            if (s->inv[6] > 0 || s->inv[7] > 0 || s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[2] = (int8_t)cr_min_i(9, s->inv[2] + 1);
                s->achievements[ACH_COLLECT_COAL] = true;
            } break;
        case BLK_IRON:
            if (s->inv[7] > 0 || s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[3] = (int8_t)cr_min_i(9, s->inv[3] + 1);
                s->achievements[ACH_COLLECT_IRON] = true;
            } break;
        case BLK_DIAMOND:
            if (s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[4] = (int8_t)cr_min_i(9, s->inv[4] + 1);
                s->achievements[ACH_COLLECT_DIAMOND] = true;
            } break;
        case BLK_GRASS:
            if (cr_rf(&s->pcg) < 0.1f) {
                s->inv[5] = (int8_t)cr_min_i(9, s->inv[5] + 1);
                s->achievements[ACH_COLLECT_SAPLING] = true;
            } break;
        case BLK_WATER:
            s->drink = (int8_t)cr_min_i(9, s->drink + 1); s->thirst = 0;
            s->achievements[ACH_COLLECT_DRINK] = true; break;
        case BLK_RIPE_PLANT:
            map_set(s, tr, tc, BLK_PLANT);
            s->food = (int8_t)cr_min_i(9, s->food + 4); s->hunger = 0;
            s->achievements[ACH_EAT_PLANT] = true;
            for (int i = 0; i < MAX_PLANTS; i++)
                if (s->plant_mask[i] && s->plant_r[i] == tr && s->plant_c[i] == tc) {
                    s->plant_age[i] = 0; break;
                }
            break;
    }
}

static void place_block(CraftaxClassic* s, int action) {
    int tr = s->player_r + DIR_DR[s->player_dir];
    int tc = s->player_c + DIR_DC[s->player_dir];
    if (!in_bounds(tr, tc)) return;
    if (has_mob_at(s, tr, tc)) return;
    int8_t blk = map_get(s, tr, tc);
    if (action == ACT_PLACE_TABLE && s->inv[0] >= 2 && !is_solid(blk)) {
        map_set(s, tr, tc, BLK_TABLE); s->inv[0] -= 2;
        s->achievements[ACH_PLACE_TABLE] = true;
    } else if (action == ACT_PLACE_FURNACE && s->inv[1] >= 1 && !is_solid(blk)) {
        map_set(s, tr, tc, BLK_FURNACE); s->inv[1] -= 1;
        s->achievements[ACH_PLACE_FURNACE] = true;
    } else if (action == ACT_PLACE_STONE && s->inv[1] >= 1 && (!is_solid(blk) || blk == BLK_WATER)) {
        map_set(s, tr, tc, BLK_STONE); s->inv[1] -= 1;
        s->achievements[ACH_PLACE_STONE] = true;
    } else if (action == ACT_PLACE_PLANT && s->inv[5] >= 1 && blk == BLK_GRASS) {
        map_set(s, tr, tc, BLK_PLANT); s->inv[5] -= 1;
        s->achievements[ACH_PLACE_PLANT] = true;
        for (int i = 0; i < MAX_PLANTS; i++) {
            if (!s->plant_mask[i]) {
                s->plant_r[i] = tr; s->plant_c[i] = tc;
                s->plant_age[i] = 0; s->plant_mask[i] = true; break;
            }
        }
    }
}

static void move_player(CraftaxClassic* s, int action) {
    if (action < 1 || action > 4) return;
    int nr = s->player_r + DIR_DR[action];
    int nc = s->player_c + DIR_DC[action];
    s->player_dir = (int8_t)action;
    if (!in_bounds(nr, nc)) return;
    if (is_solid(map_get(s, nr, nc))) return;
    if (has_mob_at(s, nr, nc)) return;
    s->player_r = (int16_t)nr; s->player_c = (int16_t)nc;
}

static bool can_move_mob(const CraftaxClassic* s, int r, int c) {
    if (!in_bounds(r, c)) return false;
    int8_t blk = map_get(s, r, c);
    if (is_solid(blk)) return false;
    if (blk == BLK_LAVA) return false;
    if (has_mob_at(s, r, c)) return false;
    if (r == s->player_r && c == s->player_c) return false;
    return true;
}

static void update_mobs(CraftaxClassic* s) {
    int pr = s->player_r, pc = s->player_c;

    for (int i = 0; i < MAX_ZOMBIES; i++) {
        if (!s->zombie_mask[i]) continue;
        int zr = s->zombie_r[i], zc = s->zombie_c[i];
        int dist = l1_dist(zr, zc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) {
            s->zombie_mask[i] = false;
            mb_clear(s->mob_bits, zr, zc); mb_clear(s->zombie_bits, zr, zc);
            continue;
        }
        if (dist <= 1 && s->zombie_cd[i] <= 0) {
            int dmg = s->is_sleeping ? 7 : 2;
            s->health -= dmg;
            s->zombie_cd[i] = 5;
            s->is_sleeping = false;
        }
        s->zombie_cd[i] = (int8_t)cr_max_i(0, s->zombie_cd[i] - 1);

        int dr = 0, dc = 0;
        if (dist < 10 && cr_rf(&s->pcg) < 0.75f) {
            int adr = abs(pr - zr), adc = abs(pc - zc);
            if (adr > adc || (adr == adc && cr_rf(&s->pcg) < 0.5f)) dr = cr_sign_i(pr - zr);
            else                                                    dc = cr_sign_i(pc - zc);
        } else {
            int d = cr_ri(&s->pcg, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = zr + dr, nc = zc + dc;
        if (can_move_mob(s, nr, nc)) {
            mb_clear(s->mob_bits, zr, zc); mb_clear(s->zombie_bits, zr, zc);
            s->zombie_r[i] = (int16_t)nr; s->zombie_c[i] = (int16_t)nc;
            mb_set(s->mob_bits, nr, nc);   mb_set(s->zombie_bits, nr, nc);
        }
    }

    for (int i = 0; i < MAX_COWS; i++) {
        if (!s->cow_mask[i]) continue;
        int cr = s->cow_r[i], cc = s->cow_c[i];
        int dist = l1_dist(cr, cc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) {
            s->cow_mask[i] = false;
            mb_clear(s->mob_bits, cr, cc); mb_clear(s->cow_bits, cr, cc);
            continue;
        }
        int d = cr_ri(&s->pcg, 8);
        if (d < 4) {
            int dr = DIR_DR[d+1], dc2 = DIR_DC[d+1];
            int nr = cr + dr, nc = cc + dc2;
            if (can_move_mob(s, nr, nc)) {
                mb_clear(s->mob_bits, cr, cc); mb_clear(s->cow_bits, cr, cc);
                s->cow_r[i] = (int16_t)nr; s->cow_c[i] = (int16_t)nc;
                mb_set(s->mob_bits, nr, nc);   mb_set(s->cow_bits, nr, nc);
            }
        }
    }

    for (int i = 0; i < MAX_SKELETONS; i++) {
        if (!s->skel_mask[i]) continue;
        int sr = s->skel_r[i], sc = s->skel_c[i];
        int dist = l1_dist(sr, sc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) {
            s->skel_mask[i] = false;
            mb_clear(s->mob_bits, sr, sc); mb_clear(s->skel_bits, sr, sc);
            continue;
        }
        if (dist >= 4 && dist <= 5 && s->skel_cd[i] <= 0) {
            for (int a = 0; a < MAX_ARROWS; a++) {
                if (!s->arrow_mask[a]) {
                    s->arrow_mask[a] = true;
                    s->arrow_r[a] = (int16_t)sr; s->arrow_c[a] = (int16_t)sc;
                    mb_set(s->arrow_bits, sr, sc);
                    int adr = abs(pr - sr), adc = abs(pc - sc);
                    s->arrow_dr[a] = (int8_t)((adr > 0) ? cr_sign_i(pr - sr) : 0);
                    s->arrow_dc[a] = (int8_t)((adc > 0) ? cr_sign_i(pc - sc) : 0);
                    break;
                }
            }
            s->skel_cd[i] = 4;
        }
        s->skel_cd[i] = (int8_t)cr_max_i(0, s->skel_cd[i] - 1);

        int dr = 0, dc = 0;
        bool random_move = cr_rf(&s->pcg) < 0.15f;
        if (!random_move) {
            if (dist >= 10) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && cr_rf(&s->pcg) < 0.5f)) dr = cr_sign_i(pr - sr);
                else                                                    dc = cr_sign_i(pc - sc);
            } else if (dist <= 3) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && cr_rf(&s->pcg) < 0.5f)) dr = -cr_sign_i(pr - sr);
                else                                                    dc = -cr_sign_i(pc - sc);
            } else {
                random_move = true;
            }
        }
        if (random_move) {
            int d = cr_ri(&s->pcg, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = sr + dr, nc = sc + dc;
        if (can_move_mob(s, nr, nc)) {
            mb_clear(s->mob_bits, sr, sc); mb_clear(s->skel_bits, sr, sc);
            s->skel_r[i] = (int16_t)nr; s->skel_c[i] = (int16_t)nc;
            mb_set(s->mob_bits, nr, nc);   mb_set(s->skel_bits, nr, nc);
        }
    }

    for (int i = 0; i < MAX_ARROWS; i++) {
        if (!s->arrow_mask[i]) continue;
        int ar = s->arrow_r[i], ac = s->arrow_c[i];
        int nr = ar + s->arrow_dr[i], nc = ac + s->arrow_dc[i];
        if (!in_bounds(nr, nc)) { s->arrow_mask[i] = false; mb_clear(s->arrow_bits, ar, ac); continue; }
        int8_t blk = map_get(s, nr, nc);
        if (is_solid(blk) && blk != BLK_WATER) {
            if (blk == BLK_FURNACE || blk == BLK_TABLE) map_set(s, nr, nc, BLK_PATH);
            s->arrow_mask[i] = false; mb_clear(s->arrow_bits, ar, ac); continue;
        }
        if (nr == pr && nc == pc) {
            s->health -= 2; s->is_sleeping = false;
            s->arrow_mask[i] = false; mb_clear(s->arrow_bits, ar, ac); continue;
        }
        mb_clear(s->arrow_bits, ar, ac);
        s->arrow_r[i] = (int16_t)nr; s->arrow_c[i] = (int16_t)nc;
        mb_set(s->arrow_bits, nr, nc);
    }
}

static bool try_spawn(CraftaxClassic* s, int min_d, int max_d, bool need_grass, bool need_path,
                      int* or_, int* oc_) {
    int pr = s->player_r, pc = s->player_c;
    for (int att = 0; att < 20; att++) {
        int r = cr_ri(&s->pcg, MAP_SIZE), c = cr_ri(&s->pcg, MAP_SIZE);
        int dist = l1_dist(r, c, pr, pc);
        if (dist < min_d || dist >= max_d) continue;
        if (has_mob_at(s, r, c)) continue;
        if (r == pr && c == pc) continue;
        int8_t blk = map_get(s, r, c);
        if (need_grass && blk != BLK_GRASS) continue;
        if (need_path  && blk != BLK_PATH ) continue;
        if (!need_grass && !need_path && blk != BLK_GRASS && blk != BLK_PATH) continue;
        *or_ = r; *oc_ = c; return true;
    }
    return false;
}

static void spawn_mobs(CraftaxClassic* s) {
    int n_cows = 0, n_z = 0, n_sk = 0;
    for (int i = 0; i < MAX_COWS;      i++) n_cows += s->cow_mask[i];
    for (int i = 0; i < MAX_ZOMBIES;   i++) n_z    += s->zombie_mask[i];
    for (int i = 0; i < MAX_SKELETONS; i++) n_sk   += s->skel_mask[i];

    if (n_cows < MAX_COWS && cr_rf(&s->pcg) < 0.1f) {
        int r, c;
        if (try_spawn(s, 3, MOB_DESPAWN_DIST, true, false, &r, &c)) {
            for (int i = 0; i < MAX_COWS; i++) if (!s->cow_mask[i]) {
                s->cow_mask[i] = true; s->cow_r[i] = (int16_t)r; s->cow_c[i] = (int16_t)c; s->cow_hp[i] = 3;
                mb_set(s->mob_bits, r, c); mb_set(s->cow_bits, r, c);
                break;
            }
        }
    }
    float zombie_chance = 0.02f + 0.1f * (1.0f - s->light_level) * (1.0f - s->light_level);
    if (n_z < MAX_ZOMBIES && cr_rf(&s->pcg) < zombie_chance) {
        int r, c;
        if (try_spawn(s, 9, MOB_DESPAWN_DIST, false, false, &r, &c)) {
            for (int i = 0; i < MAX_ZOMBIES; i++) if (!s->zombie_mask[i]) {
                s->zombie_mask[i] = true; s->zombie_r[i] = (int16_t)r; s->zombie_c[i] = (int16_t)c;
                s->zombie_hp[i] = 5; s->zombie_cd[i] = 0;
                mb_set(s->mob_bits, r, c); mb_set(s->zombie_bits, r, c);
                break;
            }
        }
    }
    if (n_sk < MAX_SKELETONS && cr_rf(&s->pcg) < 0.05f) {
        int r, c;
        if (try_spawn(s, 9, MOB_DESPAWN_DIST, false, true, &r, &c)) {
            for (int i = 0; i < MAX_SKELETONS; i++) if (!s->skel_mask[i]) {
                s->skel_mask[i] = true; s->skel_r[i] = (int16_t)r; s->skel_c[i] = (int16_t)c;
                s->skel_hp[i] = 3; s->skel_cd[i] = 0;
                mb_set(s->mob_bits, r, c); mb_set(s->skel_bits, r, c);
                break;
            }
        }
    }
}

static void update_plants(CraftaxClassic* s) {
    for (int i = 0; i < MAX_PLANTS; i++) {
        if (!s->plant_mask[i]) continue;
        s->plant_age[i]++;
        if (s->plant_age[i] >= 600) {
            int r = s->plant_r[i], c = s->plant_c[i];
            if (in_bounds(r, c) && map_get(s, r, c) == BLK_PLANT)
                map_set(s, r, c, BLK_RIPE_PLANT);
        }
    }
}

static void update_intrinsics(CraftaxClassic* s, int action) {
    if (action == ACT_SLEEP && s->energy < 9) s->is_sleeping = true;
    if (s->energy >= 9 && s->is_sleeping) {
        s->is_sleeping = false;
        s->achievements[ACH_WAKE_UP] = true;
    }
    float mul = s->is_sleeping ? 0.5f : 1.0f;
    s->hunger += mul; if (s->hunger > 25.0f) { s->food--; s->hunger = 0; }
    s->thirst += mul; if (s->thirst > 20.0f) { s->drink--; s->thirst = 0; }
    if (s->is_sleeping) s->fatigue -= 1.0f; else s->fatigue += 1.0f;
    if (s->fatigue >  30.0f) { s->energy--;                                     s->fatigue = 0; }
    if (s->fatigue < -10.0f) { s->energy = (int8_t)cr_min_i(s->energy + 1, 9); s->fatigue = 0; }
    bool ok = (s->food > 0) && (s->drink > 0) && (s->energy > 0 || s->is_sleeping);
    if (ok) s->recover += s->is_sleeping ? 2.0f : 1.0f;
    else    s->recover += s->is_sleeping ? -0.5f : -1.0f;
    if (s->recover >  25.0f) { s->health = (int8_t)cr_min_i(s->health + 1, 9); s->recover = 0; }
    if (s->recover < -15.0f) { s->health--;                                    s->recover = 0; }
}

// ============================================================
// Observation builder (writes OBS_DIM floats into env->observations)
// ============================================================
static void compute_observations(CraftaxClassic* s) {
    float* obs = s->observations;
    int pr = s->player_r, pc = s->player_c;
    int idx = 0;
    for (int dr = -3; dr <= 3; dr++) {
        int r = pr + dr;
        bool row_ok = (unsigned)r < MAP_SIZE;
        uint64_t zb = row_ok ? s->zombie_bits[r] : 0;
        uint64_t cb = row_ok ? s->cow_bits[r]    : 0;
        uint64_t sb = row_ok ? s->skel_bits[r]   : 0;
        uint64_t ab = row_ok ? s->arrow_bits[r]  : 0;
        for (int dc = -4; dc <= 4; dc++) {
            int c = pc + dc;
            int8_t blk = (row_ok && (unsigned)c < MAP_SIZE) ? map_get(s, r, c) : BLK_OUT_OF_BOUNDS;
            float* dst = obs + idx;
            for (int b = 0; b < NUM_BLOCK_TYPES; b++) dst[b] = 0.0f;
            if ((unsigned)blk < NUM_BLOCK_TYPES) dst[blk] = 1.0f;
            idx += NUM_BLOCK_TYPES;
            float mz = 0, mc = 0, ms = 0, ma = 0;
            if (row_ok && (unsigned)c < MAP_SIZE) {
                uint64_t bit = 1ULL << c;
                mz = (zb & bit) ? 1.0f : 0.0f;
                mc = (cb & bit) ? 1.0f : 0.0f;
                ms = (sb & bit) ? 1.0f : 0.0f;
                ma = (ab & bit) ? 1.0f : 0.0f;
            }
            obs[idx++] = mz; obs[idx++] = mc; obs[idx++] = ms; obs[idx++] = ma;
        }
    }
    for (int i = 0; i < NUM_INVENTORY; i++) obs[idx++] = (float)s->inv[i] * 0.1f;
    obs[idx++] = (float)s->health * 0.1f;
    obs[idx++] = (float)s->food   * 0.1f;
    obs[idx++] = (float)s->drink  * 0.1f;
    obs[idx++] = (float)s->energy * 0.1f;
    for (int d = 1; d <= 4; d++) obs[idx++] = (s->player_dir == d) ? 1.0f : 0.0f;
    obs[idx++] = s->light_level;
    obs[idx++] = s->is_sleeping ? 1.0f : 0.0f;
}

// ============================================================
// Logging (stats accumulated into env->log; flushed at vec-level by PufferLib)
// ============================================================
static void add_log(CraftaxClassic* env) {
    int unlocked = 0;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
        if (env->achievements[i]) {
            unlocked++;
            env->log.achievements[i] += 1.0f;
        }
    }
    env->log.perf           += (float)unlocked / (float)NUM_ACHIEVEMENTS;
    env->log.score          += env->episode_return_accum;
    env->log.episode_return += env->episode_return_accum;
    env->log.episode_length += (float)env->episode_length_accum;
    env->log.n              += 1.0f;
}

// ============================================================
// Public API: c_init / c_reset / c_step / c_close / c_render
// ============================================================
static void c_init(CraftaxClassic* env) {
    env->num_agents = 1;
    env->client = NULL;
    // env->rng was seeded by default my_vec_init to the env index; use it to
    // initialize a proper 64-bit PCG state.
    uint64_t seed = (uint64_t)env->rng;
    env->pcg = seed * 0x9E3779B97F4A7C15ULL + 0x87C37B91114253D5ULL;
    // Warm the RNG a bit so small seeds don't produce correlated worlds.
    for (int i = 0; i < 8; i++) (void)cr_pcg(&env->pcg);
    memset(&env->log, 0, sizeof(env->log));
}

static void c_reset(CraftaxClassic* env) {
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    generate_world(env);
    compute_observations(env);
}

static void c_step(CraftaxClassic* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) action = 0;
    if (action >= NUM_ACTIONS) action = NUM_ACTIONS - 1;

    // Snapshot for reward computation
    env->old_health = env->health;
    memcpy(env->old_achievements, env->achievements, sizeof(env->achievements));

    int eff_action = env->is_sleeping ? ACT_NOOP : action;
    do_crafting(env, eff_action);
    if (eff_action == ACT_DO) do_action(env);
    if (eff_action >= ACT_PLACE_STONE && eff_action <= ACT_PLACE_PLANT) place_block(env, eff_action);
    move_player(env, eff_action);
    update_mobs(env);
    spawn_mobs(env);
    update_plants(env);
    update_intrinsics(env, action);

    for (int i = 0; i < NUM_INVENTORY; i++)
        env->inv[i] = (int8_t)cr_clamp_i(env->inv[i], 0, 9);

    env->timestep++;
    float t_frac = fmodf((float)env->timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
    float cv = cosf(3.14159265f * t_frac);
    env->light_level = 1.0f - fabsf(cv * cv * cv);

    // Reward: new achievements + health change * 0.1
    float ach_r = 0.0f;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++)
        ach_r += (float)(env->achievements[i] && !env->old_achievements[i]);
    float hp_r = (float)(env->health - env->old_health) * 0.1f;
    float r = ach_r + hp_r;
    env->rewards[0] = r;
    env->episode_return_accum += r;
    env->episode_length_accum += 1;

    // Terminal conditions
    bool done = (env->timestep >= MAX_TIMESTEPS) || (env->health <= 0);
    if (in_bounds(env->player_r, env->player_c)
        && map_get(env, env->player_r, env->player_c) == BLK_LAVA) done = true;

    if (done) {
        env->terminals[0] = 1.0f;
        add_log(env);
        c_reset(env);   // auto-reset (observation written inside)
    } else {
        compute_observations(env);
    }
}

static void c_close(CraftaxClassic* env) {
    (void)env;
}

// ============================================================
// Tile-based renderer sharing the full-Craftax textures.bin
// ============================================================
// Shared layout (see ocean/craftax/pack_textures.py):
//   [0..36]  block textures (first 17 used by classic, indexed by BLK_*)
//   [37..41] player: down, up, left, right, sleep
//   [42..46] items (unused by classic)
//   [47..49] mobs: zombie, skeleton, cow
//   [50..53] arrows: down, up, left, right

#include <stdio.h>

#define CC_TEX_TILE_PX 16
#define CC_TEX_SCALE 4
#define CC_TEX_DRAW_PX (CC_TEX_TILE_PX * CC_TEX_SCALE)
#define CC_TEX_NUM (37 + 5 + 5 + 3 + 4)

#define CC_TEX_PLAYER_DOWN 37
#define CC_TEX_PLAYER_UP 38
#define CC_TEX_PLAYER_LEFT 39
#define CC_TEX_PLAYER_RIGHT 40
#define CC_TEX_PLAYER_SLEEP 41
#define CC_TEX_MOB_ZOMBIE 47
#define CC_TEX_MOB_SKELETON 48
#define CC_TEX_MOB_COW 49
#define CC_TEX_ARROW_DOWN 50
#define CC_TEX_ARROW_UP 51
#define CC_TEX_ARROW_LEFT 52
#define CC_TEX_ARROW_RIGHT 53

#define CC_RENDER_ROWS 16
#define CC_RENDER_COLS 16

static Texture2D cc_textures[CC_TEX_NUM];
static bool cc_textures_loaded = false;

static void cc_load_textures(void) {
    if (cc_textures_loaded) return;
    const char* candidates[] = {
        "resources/craftax/textures.bin",
        "../resources/craftax/textures.bin",
        "../../resources/craftax/textures.bin",
    };
    FILE* f = NULL;
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        f = fopen(candidates[i], "rb");
        if (f) break;
    }
    if (!f) {
        fprintf(stderr, "craftax_classic: textures.bin not found in resources/craftax -- run ocean/craftax/pack_textures.py\n");
        exit(1);
    }
    const size_t tile_bytes = CC_TEX_TILE_PX * CC_TEX_TILE_PX * 4;
    uint8_t* buf = (uint8_t*)malloc(tile_bytes);
    for (int i = 0; i < CC_TEX_NUM; i++) {
        if (fread(buf, 1, tile_bytes, f) != tile_bytes) {
            fprintf(stderr, "craftax_classic: short read on textures.bin at tile %d\n", i);
            exit(1);
        }
        Image img = {
            .data = buf,
            .width = CC_TEX_TILE_PX,
            .height = CC_TEX_TILE_PX,
            .mipmaps = 1,
            .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
        };
        cc_textures[i] = LoadTextureFromImage(img);
        SetTextureFilter(cc_textures[i], TEXTURE_FILTER_POINT);
    }
    free(buf);
    fclose(f);
    cc_textures_loaded = true;
}

static int cc_player_tex_id(int8_t dir, bool sleeping) {
    if (sleeping) return CC_TEX_PLAYER_SLEEP;
    switch (dir) {
        case 1: return CC_TEX_PLAYER_LEFT;
        case 2: return CC_TEX_PLAYER_RIGHT;
        case 3: return CC_TEX_PLAYER_UP;
        case 4: return CC_TEX_PLAYER_DOWN;
        default: return CC_TEX_PLAYER_DOWN;
    }
}

static int cc_arrow_tex_id(int8_t dr, int8_t dc) {
    if (dr < 0) return CC_TEX_ARROW_UP;
    if (dr > 0) return CC_TEX_ARROW_DOWN;
    if (dc < 0) return CC_TEX_ARROW_LEFT;
    return CC_TEX_ARROW_RIGHT;
}

static void cc_draw_tile(int tex_id, int dst_x, int dst_y) {
    if (tex_id < 0 || tex_id >= CC_TEX_NUM) return;
    Rectangle src = {0, 0, CC_TEX_TILE_PX, CC_TEX_TILE_PX};
    Rectangle dst = {(float)dst_x, (float)dst_y, CC_TEX_DRAW_PX, CC_TEX_DRAW_PX};
    DrawTexturePro(cc_textures[tex_id], src, dst, (Vector2){0, 0}, 0.0f, WHITE);
}

static void c_render(CraftaxClassic* env) {
    const int view_w = CC_RENDER_COLS * CC_TEX_DRAW_PX;
    const int view_h = CC_RENDER_ROWS * CC_TEX_DRAW_PX;
    const int hud_h = 60;

    if (!IsWindowReady()) {
        InitWindow(view_w, view_h + hud_h, "PufferLib Craftax-Classic");
        SetTargetFPS(30);
    }
    if (!cc_textures_loaded) cc_load_textures();
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    int pr = env->player_r;
    int pc = env->player_c;
    int half_r = CC_RENDER_ROWS / 2;
    int half_c = CC_RENDER_COLS / 2;

    BeginDrawing();
    ClearBackground(BLACK);

    for (int vr = 0; vr < CC_RENDER_ROWS; vr++) {
        for (int vc = 0; vc < CC_RENDER_COLS; vc++) {
            int wr = pr - half_r + vr;
            int wc = pc - half_c + vc;
            int dst_x = vc * CC_TEX_DRAW_PX;
            int dst_y = vr * CC_TEX_DRAW_PX;

            int blk = BLK_OUT_OF_BOUNDS;
            if (in_bounds(wr, wc)) blk = map_get(env, wr, wc);
            if (blk < 0 || blk >= 17) blk = 0;
            cc_draw_tile(blk, dst_x, dst_y);
        }
    }

    // Mobs
    for (int i = 0; i < MAX_ZOMBIES; i++) {
        if (!env->zombie_mask[i]) continue;
        int vr = env->zombie_r[i] - pr + half_r;
        int vc = env->zombie_c[i] - pc + half_c;
        if (vr < 0 || vr >= CC_RENDER_ROWS || vc < 0 || vc >= CC_RENDER_COLS) continue;
        cc_draw_tile(CC_TEX_MOB_ZOMBIE, vc * CC_TEX_DRAW_PX, vr * CC_TEX_DRAW_PX);
    }
    for (int i = 0; i < MAX_SKELETONS; i++) {
        if (!env->skel_mask[i]) continue;
        int vr = env->skel_r[i] - pr + half_r;
        int vc = env->skel_c[i] - pc + half_c;
        if (vr < 0 || vr >= CC_RENDER_ROWS || vc < 0 || vc >= CC_RENDER_COLS) continue;
        cc_draw_tile(CC_TEX_MOB_SKELETON, vc * CC_TEX_DRAW_PX, vr * CC_TEX_DRAW_PX);
    }
    for (int i = 0; i < MAX_COWS; i++) {
        if (!env->cow_mask[i]) continue;
        int vr = env->cow_r[i] - pr + half_r;
        int vc = env->cow_c[i] - pc + half_c;
        if (vr < 0 || vr >= CC_RENDER_ROWS || vc < 0 || vc >= CC_RENDER_COLS) continue;
        cc_draw_tile(CC_TEX_MOB_COW, vc * CC_TEX_DRAW_PX, vr * CC_TEX_DRAW_PX);
    }
    for (int i = 0; i < MAX_ARROWS; i++) {
        if (!env->arrow_mask[i]) continue;
        int vr = env->arrow_r[i] - pr + half_r;
        int vc = env->arrow_c[i] - pc + half_c;
        if (vr < 0 || vr >= CC_RENDER_ROWS || vc < 0 || vc >= CC_RENDER_COLS) continue;
        cc_draw_tile(cc_arrow_tex_id(env->arrow_dr[i], env->arrow_dc[i]),
                     vc * CC_TEX_DRAW_PX, vr * CC_TEX_DRAW_PX);
    }

    // Player in center
    cc_draw_tile(cc_player_tex_id(env->player_dir, env->is_sleeping),
                 half_c * CC_TEX_DRAW_PX, half_r * CC_TEX_DRAW_PX);

    // Night dim
    if (env->light_level < 1.0f) {
        unsigned char a = (unsigned char)((1.0f - env->light_level) * 140.0f);
        DrawRectangle(0, 0, view_w, view_h, (Color){0, 0, 40, a});
    }

    // HUD
    int hud_y = view_h;
    DrawRectangle(0, hud_y, view_w, hud_h, (Color){20, 20, 20, 255});
    DrawText(TextFormat("HP:%d  F:%d  D:%d  E:%d  t:%d  light:%.2f",
             env->health, env->food, env->drink, env->energy,
             env->timestep, env->light_level),
             4, hud_y + 4, 14, WHITE);
    int ach_count = 0;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) ach_count += env->achievements[i] ? 1 : 0;
    DrawText(TextFormat("ach:%d/%d  ret:%.2f  len:%d", ach_count, NUM_ACHIEVEMENTS,
             env->episode_return_accum, env->episode_length_accum),
             4, hud_y + 22, 14, (Color){180, 220, 180, 255});
    DrawText(TextFormat("inv: w=%d s=%d c=%d i=%d d=%d sap=%d  pick w/s/i:%d/%d/%d  sword w/s/i:%d/%d/%d",
             env->inv[0], env->inv[1], env->inv[2], env->inv[3], env->inv[4], env->inv[5],
             env->inv[6], env->inv[7], env->inv[8], env->inv[9], env->inv[10], env->inv[11]),
             4, hud_y + 40, 12, (Color){180, 180, 180, 255});
    EndDrawing();
}
