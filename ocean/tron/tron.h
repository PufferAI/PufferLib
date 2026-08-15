#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
#include "pufferenv.h"

#define WIDTH 24
#define HEIGHT 17
#define CELLS (WIDTH * HEIGHT)
#define PLAYERS 2
#define MAX_TICKS (CELLS / PLAYERS)
#define CYAN_START_X 18
#define CYAN_START_Y 8
#define RED_START_X 5
#define RED_START_Y 8

#define LOCAL_RADIUS 5
#define LOCAL_SIZE (2 * LOCAL_RADIUS + 1)
#define LOCAL_CELLS (LOCAL_SIZE * LOCAL_SIZE)

#define ACT_SIZES {3}
#define NUM_ATNS 1

#define BOARD_X 44
#define BOARD_Y 44
#define CELL_SIZE 36
#define BOARD_WIDTH (WIDTH * CELL_SIZE)
#define BOARD_HEIGHT (HEIGHT * CELL_SIZE)
#define WINDOW_WIDTH (BOARD_WIDTH + 2 * BOARD_X)
#define WINDOW_HEIGHT (BOARD_HEIGHT + 2 * BOARD_Y)
#define TRAIL_WIDTH 8
#define CYCLE_LENGTH 44
#define CYCLE_WIDTH 24
#define PUFFER_SIZE 48
#define RENDER_FPS 60
#define RENDER_TICKS_PER_SECOND 6
#define RENDER_FRAMES_PER_TICK (RENDER_FPS / RENDER_TICKS_PER_SECOND)

enum {
    PLAYER_CYAN,
    PLAYER_RED
};

enum {
    NORTH,
    EAST,
    SOUTH,
    WEST
};

typedef enum {
    LEFT,
    STRAIGHT,
    RIGHT
} TronAction;

enum {
    CELL_OPEN,
    CYAN_TRAIL,
    RED_TRAIL
};

enum {
    PLAYING,
    CYAN_WIN,
    RED_WIN,
    DRAW
};

typedef uint8_t obs_t;

typedef struct {
    uint8_t own_trail[CELLS];
    uint8_t opponent_trail[CELLS];
    uint8_t own_head_x[WIDTH];
    uint8_t own_head_y[HEIGHT];
    uint8_t opponent_head_x[WIDTH];
    uint8_t opponent_head_y[HEIGHT];
    uint8_t own_heading[4];
    uint8_t opponent_heading[4];
    uint8_t territory_advantage[17];
    uint8_t own_safe_actions[4];
    uint8_t opponent_safe_actions[4];
    uint8_t connected;
    uint8_t relative_direction[9];
    uint8_t head_distance[4];
    uint8_t local_own_trail[LOCAL_CELLS];
    uint8_t local_opponent_trail[LOCAL_CELLS];
    uint8_t local_opponent_head[LOCAL_CELLS];
    uint8_t local_wall[LOCAL_CELLS];
} TronObs;

#define OBS_SIZE sizeof(TronObs)

typedef struct {
    TronAction player[PLAYERS];
} TronActions;

typedef struct {
    uint8_t trails[CELLS];
    uint8_t x[PLAYERS];
    uint8_t y[PLAYERS];
    uint8_t heading[PLAYERS];
    uint16_t tick;
    uint8_t outcome;
} TronGame;

typedef struct {
    uint16_t cell[CELLS];
    uint16_t count;
} TronTrail;

typedef struct {
    Texture2D cycle;
    Texture2D puffer;
    Texture2D crash;
    TronGame previous;
    TronTrail trail[PLAYERS];
} TronRenderer;

typedef struct {
    uint16_t territory[PLAYERS];
    uint8_t safe_actions[PLAYERS];
    bool connected;
} TronFeat;

typedef struct {
    uint8_t seen[CELLS];
    uint8_t mark;
} BotCache;

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float draw_rate;
    float slot_0_score;
    float policy_0_score;
    float hist_score_bank_0;
    float hist_n_bank_0;
    float n;
};

struct Env {
    Log log;
    Agent agents[PLAYERS];
    int num_agents;

    int tag;
    int boundary_reached;
    unsigned int rng;

    int opening_steps;
    int opening_ticks;
    int bot_level;
    float reward_territory;
    float territory_gamma;
    float episode_return;

    // Derived features require flood fills and feed both shaping and observations,
    // so cache them once per state instead of recomputing them for each consumer.
    TronGame state;
    TronFeat feat;

    // map logical colors to policy slots so each policy plays both sides
    uint8_t slot_for_player[PLAYERS];

    // Simulation resets immediately for throughput. These fields preserve the
    // terminal board and ordered random opening long enough to render smoothly.
    TronGame render_game;
    TronRenderer renderer;
    TronTrail opening_trail[PLAYERS];

    int rendering;
    int render_pending_reset;

    BotCache bot_cache;
};
typedef Env Tron;

static const int8_t DX[4] = {0, 1, 0, -1};
static const int8_t DY[4] = {-1, 0, 1, 0};

void reset(TronGame *game) {
    memset(game->trails, CELL_OPEN, sizeof(game->trails));
    game->x[PLAYER_CYAN] = CYAN_START_X;
    game->y[PLAYER_CYAN] = CYAN_START_Y;
    game->x[PLAYER_RED] = RED_START_X;
    game->y[PLAYER_RED] = RED_START_Y;
    game->heading[PLAYER_CYAN] = WEST;
    game->heading[PLAYER_RED] = EAST;
    game->tick = 0;
    game->outcome = PLAYING;
}

static int trail_index(int x, int y) {
    return y * WIDTH + x;
}

static bool on_board(int x, int y) {
    return x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT;
}

static int other_player(int player) {
    if (player == PLAYER_CYAN) return PLAYER_RED;
    return PLAYER_CYAN;
}

static int player_trail(int player) {
    if (player == PLAYER_CYAN) return CYAN_TRAIL;
    return RED_TRAIL;
}

static int player_win(int player) {
    if (player == PLAYER_CYAN) return CYAN_WIN;
    return RED_WIN;
}

static int next_dir(int dir, TronAction atn) {
    if (atn == LEFT) return (dir + 3) % 4;
    if (atn == RIGHT) return (dir + 1) % 4;
    return dir;
}

// trail history is only used for smooth rendering
static void trails_reset(TronTrail trail[PLAYERS], const TronGame *game) {
    for (int p = 0; p < PLAYERS; p++) {
        trail[p].cell[0] = trail_index(game->x[p], game->y[p]);
        trail[p].count = 1;
    }
}

static void trails_record(TronTrail trail[PLAYERS], const TronGame *game) {
    for (int p = 0; p < PLAYERS; p++) {
        int cell = trail_index(game->x[p], game->y[p]);
        if (trail[p].cell[trail[p].count - 1] != cell)
            trail[p].cell[trail[p].count++] = cell;
    }
}

void step(TronGame *game, TronActions actions) {
    int x[PLAYERS];
    int y[PLAYERS];
    bool crash[PLAYERS] = {false};

    for (int p = 0; p < PLAYERS; p++) {
        game->heading[p] = next_dir(game->heading[p], actions.player[p]);
        game->trails[trail_index(game->x[p], game->y[p])] = player_trail(p);
        x[p] = game->x[p] + DX[game->heading[p]];
        y[p] = game->y[p] + DY[game->heading[p]];
    }
    for (int p = 0; p < PLAYERS; p++) {
        if (!on_board(x[p], y[p]) ||
            game->trails[trail_index(x[p], y[p])] != CELL_OPEN) {
            crash[p] = true;
        }
    }
    if (x[PLAYER_CYAN] == x[PLAYER_RED] && y[PLAYER_CYAN] == y[PLAYER_RED]) {
        crash[PLAYER_CYAN] = true;
        crash[PLAYER_RED] = true;
    }
    for (int p = 0; p < PLAYERS; p++) {
        if (!crash[p]) {
            game->x[p] = x[p];
            game->y[p] = y[p];
        }
    }
    game->tick++;
    if (crash[0] && crash[1]) game->outcome = DRAW;
    else if (crash[0]) game->outcome = RED_WIN;
    else if (crash[1]) game->outcome = CYAN_WIN;
}

// Bot flood fill uses a queue because it evaluates hypothetical boards. Heads are
// still open cells after step(), so block the rival head explicitly.
static int flood(const TronGame *game, int start, int blocked, uint8_t seen[CELLS],
                 uint16_t queue[CELLS], uint8_t mark) {
    int front = 0;
    int back = 0;
    seen[start] = mark;
    queue[back++] = start;

    while (front < back) {
        int cell = queue[front++];
        int x = cell % WIDTH;
        int y = cell / WIDTH;
        for (int dir = 0; dir < 4; dir++) {
            int nx = x + DX[dir];
            int ny = y + DY[dir];
            if (!on_board(nx, ny)) continue;

            int next = trail_index(nx, ny);
            if (next == blocked || seen[next] == mark ||
                game->trails[next] != CELL_OPEN) {
                continue;
            }
            seen[next] = mark;
            queue[back++] = next;
        }
    }
    return back;
}

#include "bots.h"

static int safe_count(const TronGame *game, int player) {
    int opp = other_player(player);
    int count = 0;
    for (int i = 0; i < 3; i++) {
        TronAction atn = BOT_ACTIONS[i];
        int dir = next_dir(game->heading[player], atn);
        int x = game->x[player] + DX[dir];
        int y = game->y[player] + DY[dir];
        if (!on_board(x, y)) continue;
        if (game->trails[trail_index(x, y)] != CELL_OPEN) continue;
        if (x == game->x[opp] && y == game->y[opp]) continue;

        bool safe = true;
        for (int j = 0; j < 3; j++) {
            TronAction opp_atn = BOT_ACTIONS[j];
            int opp_dir = next_dir(game->heading[opp], opp_atn);
            int ox = game->x[opp] + DX[opp_dir];
            int oy = game->y[opp] + DY[opp_dir];
            if (x == ox && y == oy) {
                safe = false;
                break;
            }
        }
        if (safe) count++;
    }
    return count;
}

// Hot-path flood fill: WIDTH fits in one uint32_t row, so bitsets avoid queue
// traffic while features are recomputed on every live environment tick.
static int component_size(const TronGame *game, int start,
                          int target, bool *found) {
    uint32_t open[HEIGHT] = {0};
    uint32_t visited[HEIGHT] = {0};
    uint32_t frontier[HEIGHT] = {0};
    for (int row = 0; row < HEIGHT; row++) {
        for (int x = 0; x < WIDTH; x++) {
            if (game->trails[trail_index(x, row)] == CELL_OPEN) {
                // 1u is an unsigned 1; shifting it left selects cell x
                open[row] |= 1u << x;
            }
        }
    }

    int row = start / WIDTH;
    frontier[row] = 1u << (start % WIDTH);
    visited[row] = frontier[row];

    while (true) {
        uint32_t next[HEIGHT];
        bool any = false;
        for (int row = 0; row < HEIGHT; row++) {
            // Shifting moves every frontier cell one place left or right
            uint32_t near = (frontier[row] << 1) | (frontier[row] >> 1);
            if (row > 0) near |= frontier[row - 1];
            if (row + 1 < HEIGHT) near |= frontier[row + 1];
            // Keep only cells that are open and have not already been visited
            next[row] = near & open[row] & ~visited[row];
            any = any || next[row] != 0;
        }
        if (!any) break;
        for (int row = 0; row < HEIGHT; row++) {
            visited[row] |= next[row];
            frontier[row] = next[row];
        }
    }

    int size = 0;
    for (int row = 0; row < HEIGHT; row++) {
        // This compiler intrinsic counts the 1 bits, which are visited cells
        size += __builtin_popcount(visited[row]);
    }
    if (found != NULL) {
        *found = (visited[target / WIDTH] & (1u << (target % WIDTH))) != 0;
    }
    return size;
}

static TronFeat compute_feat(const TronGame *game) {
    TronFeat feat = {
        .safe_actions = {
            safe_count(game, PLAYER_CYAN),
            safe_count(game, PLAYER_RED),
        },
    };
    int head[PLAYERS] = {
        trail_index(game->x[PLAYER_CYAN], game->y[PLAYER_CYAN]),
        trail_index(game->x[PLAYER_RED], game->y[PLAYER_RED]),
    };
    feat.territory[PLAYER_CYAN] = component_size(
        game, head[PLAYER_CYAN], head[PLAYER_RED], &feat.connected);
    // Connected heads share the same open component. A second fill only carries
    // new information after trails divide the board into private regions
    if (feat.connected) {
        feat.territory[PLAYER_RED] = feat.territory[PLAYER_CYAN];
    } else {
        feat.territory[PLAYER_RED] = component_size(
            game, head[PLAYER_RED], head[PLAYER_CYAN], NULL);
    }
    return feat;
}

static float phi(const TronFeat *feat, int player) {
    // Territory shaping starts once the players can no longer reach one another
    if (feat->connected) return 0.0f;
    int opp = other_player(player);
    return (float)(feat->territory[player] - feat->territory[opp]) /
           CELLS;
}

static int obs_cell(int cell, int p) {
    if (p == PLAYER_CYAN) return cell;
    return CELLS - 1 - cell;
}

static void observe_local(const TronGame *game, int player,
                          TronObs *obs) {
    int opp = other_player(player);
    int dir = game->heading[player];
    int right = (dir + 1) % 4;
    for (int forward = -LOCAL_RADIUS; forward <= LOCAL_RADIUS; forward++) {
        int row = (LOCAL_RADIUS - forward) * LOCAL_SIZE;
        // start at the left edge, then walk across the row one cell at a time
        int x = game->x[player] + forward * DX[dir] - LOCAL_RADIUS * DX[right];
        int y = game->y[player] + forward * DY[dir] - LOCAL_RADIUS * DY[right];
        for (int side = 0; side < LOCAL_SIZE; side++) {
            int local = row + side;
            if (!on_board(x, y)) {
                obs->local_wall[local] = 1;
            } else {
                int world = trail_index(x, y);
                if (game->trails[world] == player_trail(player))
                    obs->local_own_trail[local] = 1;
                if (game->trails[world] == player_trail(opp))
                    obs->local_opponent_trail[local] = 1;
                if (x == game->x[opp] && y == game->y[opp])
                    obs->local_opponent_head[local] = 1;
            }
            x += DX[right];
            y += DY[right];
        }
    }
}

// Global features use the color-canonical frame; the local planes above use the
// separate heading-relative frame needed for immediate steering decisions.
static void write_feat(const TronGame *game, const TronFeat *feat,
                       int player, TronObs *obs) {
    int opp = other_player(player);
    int own = obs_cell(
        trail_index(game->x[player], game->y[player]), player);
    int opponent_head = obs_cell(
        trail_index(game->x[opp], game->y[opp]), player);
    obs->own_head_x[own % WIDTH] = 1;
    obs->own_head_y[own / WIDTH] = 1;
    obs->opponent_head_x[opponent_head % WIDTH] = 1;
    obs->opponent_head_y[opponent_head / WIDTH] = 1;

    int turn = 0;
    if (player == PLAYER_RED) turn = 2;
    obs->own_heading[(game->heading[player] + turn) % 4] = 1;
    obs->opponent_heading[(game->heading[opp] + turn) % 4] = 1;

    int diff = feat->territory[player] - feat->territory[opp];
    int bucket = (diff + CELLS) * 16 / (2 * CELLS);
    obs->territory_advantage[bucket] = 1;
    obs->own_safe_actions[feat->safe_actions[player]] = 1;
    obs->opponent_safe_actions[feat->safe_actions[opp]] = 1;
    obs->connected = feat->connected;

    int view = 1;
    if (player == PLAYER_RED) view = -1;
    int dx = (game->x[opp] - game->x[player]) * view;
    int dy = (game->y[opp] - game->y[player]) * view;
    int xdir = (dx > 0) - (dx < 0);
    int ydir = (dy > 0) - (dy < 0);
    int dir = (ydir + 1) * 3 + xdir + 1;
    obs->relative_direction[dir] = 1;

    int dist = abs(dx) + abs(dy);
    if (dist <= 4) bucket = 0;
    else if (dist <= 8) bucket = 1;
    else if (dist <= 16) bucket = 2;
    else bucket = 3;
    obs->head_distance[bucket] = 1;
}

static void write_obs(const TronGame *game, const TronFeat *feat,
                      int player, TronObs *obs) {
    memset(obs, 0, sizeof(*obs));
    int opp = other_player(player);
    for (int cell = 0; cell < CELLS; cell++) {
        int seen = obs_cell(cell, player);
        if (game->trails[cell] == player_trail(player))
            obs->own_trail[seen] = 1;
        if (game->trails[cell] == player_trail(opp))
            obs->opponent_trail[seen] = 1;
    }
    observe_local(game, player, obs);
    write_feat(game, feat, player, obs);
}

static TronObs *obs_ptr(Tron *env, int player) {
    int slot = env->slot_for_player[player];
    return (TronObs *)env->agents[slot].observations;
}

static void observe_env(Tron *env) {
    if (env->num_agents != PLAYERS) {
        write_obs(&env->state, &env->feat, PLAYER_CYAN,
                  obs_ptr(env, PLAYER_CYAN));
        return;
    }

    // Intentional duplication of write_obs: emitting both global trail
    // planes in one board pass is about 12% faster than calling it twice.
    TronObs *cyan = obs_ptr(env, PLAYER_CYAN);
    TronObs *red = obs_ptr(env, PLAYER_RED);
    memset(cyan, 0, sizeof(*cyan));
    memset(red, 0, sizeof(*red));
    for (int cell = 0; cell < CELLS; cell++) {
        int rotated_cell = CELLS - 1 - cell;
        if (env->state.trails[cell] == CYAN_TRAIL) {
            cyan->own_trail[cell] = 1;
            red->opponent_trail[rotated_cell] = 1;
        } else if (env->state.trails[cell] == RED_TRAIL) {
            cyan->opponent_trail[cell] = 1;
            red->own_trail[rotated_cell] = 1;
        }
    }

    observe_local(&env->state, PLAYER_CYAN, cyan);
    observe_local(&env->state, PLAYER_RED, red);
    write_feat(&env->state, &env->feat, PLAYER_CYAN, cyan);
    write_feat(&env->state, &env->feat, PLAYER_RED, red);
}

void observe_player(const TronGame *game, int player,
                    TronObs *obs) {
    TronFeat feat = compute_feat(game);
    write_obs(game, &feat, player, obs);
}

static TronAction read_action(const Tron *env, int player) {
    int slot = env->slot_for_player[player];
    int value = (int)env->agents[slot].actions[0];
    if (value == LEFT) return LEFT;
    if (value == RIGHT) return RIGHT;
    return STRAIGHT;
}

void puf_reset(Tron *env) {
    int steps = env->opening_steps
                    ? rand_r(&env->rng) % (env->opening_steps + 1)
                    : 0;
    do {
        reset(&env->state);
        trails_reset(env->opening_trail, &env->state);
        for (int i = 0; i < steps && env->state.outcome == PLAYING; i++) {
            TronActions actions = {.player = {
                                       [PLAYER_CYAN] = (TronAction)(rand_r(&env->rng) % 3),
                                       [PLAYER_RED] = (TronAction)(rand_r(&env->rng) % 3),
                                   }};
            step(&env->state, actions);
            trails_record(env->opening_trail, &env->state);
        }
    } while (env->state.outcome != PLAYING);
    env->opening_ticks = env->state.tick;
    env->episode_return = 0;
    env->feat = compute_feat(&env->state);
    observe_env(env);
    if (env->rendering && !env->render_pending_reset) {
        env->renderer.previous = env->state;
        memcpy(env->renderer.trail, env->opening_trail, sizeof(env->opening_trail));
    }
}

void puf_step(Tron *env) {
    float *reward_ptr[PLAYERS];
    float *terminal_ptr[PLAYERS];
    for (int p = 0; p < env->num_agents; p++) {
        int slot = env->slot_for_player[p];
        reward_ptr[p] = env->agents[slot].rewards;
        terminal_ptr[p] = env->agents[slot].terminals;
        *reward_ptr[p] = 0.0f;
        *terminal_ptr[p] = 0.0f;
    }
    if (env->rendering && env->render_pending_reset) {
        env->render_pending_reset = 0;
        env->renderer.previous = env->state;
        memcpy(env->renderer.trail, env->opening_trail, sizeof(env->opening_trail));
    }
    float prev_phi = phi(&env->feat, PLAYER_CYAN);
    TronAction red_atn = env->num_agents == PLAYERS
                             ? read_action(env, PLAYER_RED)
                             : bot_action(&env->state, PLAYER_RED, env->bot_level,
                                          &env->rng, &env->bot_cache);
    TronActions atn = {.player = {
                           [PLAYER_CYAN] = read_action(env, PLAYER_CYAN),
                           [PLAYER_RED] = red_atn,
                       }};
    if (env->rendering) env->renderer.previous = env->state;
    step(&env->state, atn);
    if (env->rendering) trails_record(env->renderer.trail, &env->state);
    float reward = env->reward_territory * -prev_phi;
    if (env->state.outcome == PLAYING) {
        env->feat = compute_feat(&env->state);
        reward += env->reward_territory * env->territory_gamma *
                  phi(&env->feat, PLAYER_CYAN);
        for (int p = 0; p < env->num_agents; p++)
            *reward_ptr[p] = p == PLAYER_CYAN ? reward : -reward;
        env->episode_return += reward;
        observe_env(env);
        return;
    }
    float outcome = 0.0f;
    if (env->state.outcome == CYAN_WIN) outcome = 1.0f;
    if (env->state.outcome == RED_WIN) outcome = -1.0f;
    reward += (1.0f - env->reward_territory) * outcome;
    for (int p = 0; p < env->num_agents; p++) {
        *reward_ptr[p] = p == PLAYER_CYAN ? reward : -reward;
        *terminal_ptr[p] = 1.0f;
    }
    env->episode_return += reward;
    float score = (outcome + 1.0f) / 2.0f;
    env->log.perf += score;
    env->log.score += outcome;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->state.tick - env->opening_ticks;
    env->log.draw_rate += env->state.outcome == DRAW;
    float slot_score = env->slot_for_player[PLAYER_CYAN] == 0
                            ? score
                            : 1.0f - score;
    env->log.slot_0_score += slot_score;
    env->log.policy_0_score += slot_score;
    if (env->tag > 0) {
        env->log.hist_score_bank_0 += slot_score;
        env->log.hist_n_bank_0 += 1.0f;
        env->boundary_reached = 1;
    }
    env->log.n += 1;
    if (env->rendering) {
        env->render_game = env->state;
        env->render_pending_reset = 1;
    }
    puf_reset(env);
}

// Rendering
static Vector2 cell_center(int cell) {
    return (Vector2){
        .x = (float)(BOARD_X + (cell % WIDTH) * CELL_SIZE + CELL_SIZE / 2),
        .y = (float)(BOARD_Y + (cell / WIDTH) * CELL_SIZE + CELL_SIZE / 2),
    };
}

void renderer_init(TronRenderer *renderer, const TronGame *game) {
    renderer->cycle = LoadTexture("resources/tron/light_cycle.png");
    renderer->puffer = LoadTexture("resources/shared/puffers_128.png");
    renderer->crash = LoadTexture("resources/tron/crash_burst.png");
    SetTextureFilter(renderer->cycle, TEXTURE_FILTER_POINT);
    SetTextureFilter(renderer->puffer, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(renderer->crash, TEXTURE_FILTER_POINT);
    renderer->previous = *game;
    trails_reset(renderer->trail, game);
}

void renderer_draw(TronRenderer *renderer, const TronGame *game,
                   float interpolation, float crash_age) {
    Color colors[PLAYERS] = {
        [PLAYER_CYAN] = {.r = 10, .g = 202, .b = 222, .a = 255},
        [PLAYER_RED] = {.r = 224, .g = 35, .b = 28, .a = 255},
    };

    BeginDrawing();
    ClearBackground((Color){.r = 0, .g = 4, .b = 9, .a = 255});
    DrawRectangle(BOARD_X - 4, BOARD_Y - 4, BOARD_WIDTH + 8, BOARD_HEIGHT + 8,
                  (Color){.r = 72, .g = 132, .b = 143, .a = 255});
    DrawRectangle(BOARD_X, BOARD_Y, BOARD_WIDTH, BOARD_HEIGHT,
                  (Color){.r = 1, .g = 11, .b = 24, .a = 255});
    for (int x = 1; x < WIDTH; x++) {
        Color color = x % 4
                          ? (Color){.r = 32, .g = 80, .b = 99, .a = 38}
                          : (Color){.r = 56, .g = 116, .b = 132, .a = 64};
        DrawLine(BOARD_X + x * CELL_SIZE, BOARD_Y,
                 BOARD_X + x * CELL_SIZE, BOARD_Y + BOARD_HEIGHT, color);
    }

    for (int y = 1; y < HEIGHT; y++) {
        Color color = y % 4
                          ? (Color){.r = 32, .g = 80, .b = 99, .a = 38}
                          : (Color){.r = 56, .g = 116, .b = 132, .a = 64};
        DrawLine(BOARD_X, BOARD_Y + y * CELL_SIZE,
                 BOARD_X + BOARD_WIDTH, BOARD_Y + y * CELL_SIZE, color);
    }

    for (int p = 0; p < PLAYERS; p++) {
        int count = renderer->trail[p].count;
        int head = trail_index(game->x[p], game->y[p]);
        int prior = trail_index(renderer->previous.x[p], renderer->previous.y[p]);
        if (interpolation < 1.0f && prior != head && count > 1 &&
            renderer->trail[p].cell[count - 1] == head) {
            count--;
        }
        for (int i = 1; i < count; i++) {
            Vector2 a = cell_center(renderer->trail[p].cell[i - 1]);
            Vector2 b = cell_center(renderer->trail[p].cell[i]);
            DrawLineEx(a, b, TRAIL_WIDTH, colors[p]);
            DrawRectangle(b.x - TRAIL_WIDTH / 2, b.y - TRAIL_WIDTH / 2,
                          TRAIL_WIDTH, TRAIL_WIDTH, colors[p]);
        }
    }

    for (int p = 0; p < PLAYERS; p++) {
        Vector2 prior = cell_center(
            trail_index(renderer->previous.x[p], renderer->previous.y[p]));
        Vector2 pos = cell_center(trail_index(game->x[p], game->y[p]));
        pos.x = prior.x + (pos.x - prior.x) * interpolation;
        pos.y = prior.y + (pos.y - prior.y) * interpolation;
        DrawLineEx(prior, pos, TRAIL_WIDTH, colors[p]);

        bool live = interpolation < 1.0f || game->outcome == PLAYING ||
                    game->outcome == player_win(p);
        if (!live) continue;
        bool draw_puffer = p == PLAYER_CYAN;
        Texture2D sprite = draw_puffer ? renderer->puffer : renderer->cycle;
        Rectangle source = {
            .x = 0,
            .y = 0,
            .width = draw_puffer ? sprite.width / 2.0f : (float)sprite.width,
            .height = (float)sprite.height,
        };
        float rotation = game->heading[p] * 90.0f - 90.0f;
        if (draw_puffer && game->heading[p] == WEST) {
            source.x = source.width;
            rotation = 0.0f;
        }
        float width = draw_puffer ? PUFFER_SIZE : CYCLE_LENGTH;
        float height = draw_puffer ? PUFFER_SIZE : CYCLE_WIDTH;
        Rectangle dest = {
            .x = pos.x,
            .y = pos.y,
            .width = width,
            .height = height,
        };
        Vector2 origin = {
            .x = width / 2.0f,
            .y = height / 2.0f,
        };
        Color tint = draw_puffer
                         ? WHITE
                         : (Color){.r = 255, .g = 111, .b = 78, .a = 255};
        DrawTexturePro(sprite, source, dest, origin, rotation, tint);
    }
    if (game->outcome != PLAYING && crash_age < 0.45f) {
        float progress = crash_age / 0.45f;
        int frame = progress * 6;
        float width = renderer->crash.width / 3.0f;
        float height = renderer->crash.height / 2.0f;
        unsigned char alpha = progress > 0.75f
                                  ? 255.0f * (1.0f - progress) / 0.25f
                                  : 255;
        for (int p = 0; p < PLAYERS; p++) {
            if (game->outcome == player_win(p)) continue;
            Vector2 pos = cell_center(trail_index(game->x[p], game->y[p]));
            pos.x += DX[game->heading[p]] * CELL_SIZE * 0.4f;
            pos.y += DY[game->heading[p]] * CELL_SIZE * 0.4f;
            Rectangle source = {
                .x = (frame % 3) * width,
                .y = (frame / 3) * height,
                .width = width,
                .height = height,
            };
            Rectangle dest = {.x = pos.x, .y = pos.y, .width = 64, .height = 64};
            Vector2 origin = {.x = 32, .y = 32};
            Color tint = {.r = 255, .g = 255, .b = 255, .a = alpha};
            DrawTexturePro(renderer->crash, source, dest, origin, 0, tint);
        }
    }
    EndDrawing();
}

void puf_close(Tron *env) {
    if (env->rendering) {
        UnloadTexture(env->renderer.cycle);
        UnloadTexture(env->renderer.puffer);
        UnloadTexture(env->renderer.crash);
        CloseWindow();
    }
    env->rendering = 0;
}

void puf_render(Tron *env) {
    if (!env->rendering) {
        InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "PufferLib // Tron");
        SetExitKey(KEY_NULL);
        SetTargetFPS(RENDER_FPS);
        renderer_init(&env->renderer, &env->state);
        memcpy(env->renderer.trail, env->opening_trail, sizeof(env->opening_trail));
        env->rendering = 1;
    }
    TronGame *game = env->render_pending_reset ? &env->render_game : &env->state;
    for (int i = 1; i <= RENDER_FRAMES_PER_TICK; i++) {
        if (WindowShouldClose()) {
            puf_close(env);
            exit(0);
        }
        float interpolation = (float)i / RENDER_FRAMES_PER_TICK;
        renderer_draw(&env->renderer, game, interpolation, 1.0f);
    }
    if (env->render_pending_reset) {
        for (int i = 0; i < 27; i++)
            renderer_draw(&env->renderer, game, 1.0f,
                          (float)i / RENDER_FPS);
        memcpy(env->renderer.trail, env->opening_trail, sizeof(env->opening_trail));
    }
    env->render_pending_reset = 0;
    env->renderer.previous = env->state;
}

void puf_init(Env *base, Dict *kwargs) {
    Tron *env = (Tron *)base;
    env->num_agents = (int)dict_get(kwargs, "num_agents");
    env->bot_level = (int)dict_get(kwargs, "bot_difficulty");
    env->opening_steps = (int)dict_get(kwargs, "opening_steps");
    env->reward_territory = (float)dict_get(kwargs, "reward_territory");
    env->territory_gamma = (float)dict_get(kwargs, "territory_gamma");

    bool swap_sides = env->num_agents == PLAYERS && (env->rng & 1u);
    env->slot_for_player[PLAYER_CYAN] = swap_sides ? 1 : 0;
    env->slot_for_player[PLAYER_RED] = swap_sides ? 0 : 1;
    for (int slot = 0; slot < env->num_agents; slot++) {
        env->agents[slot].policy = slot;
        env->agents[slot].action_mask = NULL;
    }
}

void puf_log(Log *log, Dict *out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "slot_0_score", log->slot_0_score);
    dict_set(out, "policy_0_score", log->policy_0_score);
    dict_set(out, "hist_score_bank_0", log->hist_score_bank_0);
    dict_set(out, "hist_n_bank_0", log->hist_n_bank_0);
}
