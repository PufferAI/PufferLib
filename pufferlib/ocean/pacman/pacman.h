#include "helpers.h"
#include "raylib.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PX_PADDING_TOP 40 // 40px padding on top of the window

#define LOG_BUFFER_SIZE 1024

#define MAX_STEPS 10000 // max steps before truncation

#define GHOST_OBSERVATIONS_COUNT 9
#define PLAYER_OBSERVATIONS_COUNT 11
#define NUM_GHOSTS 4
#define FRAMES 7  // Fixed number of frames per step for interpolation

#define NUM_DOTS 240
#define NUM_POWERUPS 4
#define OBSERVATIONS_COUNT                                                                         \
    (PLAYER_OBSERVATIONS_COUNT + GHOST_OBSERVATIONS_COUNT * NUM_GHOSTS + NUM_DOTS + NUM_POWERUPS)

#define PINKY_TARGET_LEAD 4
#define INKY_TARGET_LEAD 2
#define CLYDE_TARGET_RADIUS 8

typedef struct Log Log;
struct Log {
        float episode_return;
        float episode_length;
        float score;
        float n;
};

typedef enum Tile {
    WALL_TILE = '#',
    DOT_TILE = '.',
    POWER_TILE = 'x',
    PLAYER_TILE = 'p',
    EMPTY_TILE = ' ',

    INKY_TILE = '1',
    BLINKY_TILE = '2',
    PINKY_TILE = '3',
    CLYDE_TILE = '4',
} Tile;

#define MAP_HEIGHT 31
#define MAP_WIDTH 28

static const char original_map[MAP_HEIGHT][MAP_WIDTH] = {
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#x####.#####.##.#####.####x#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "######.##### ## #####.######",
    "######.##   1234   ##.######",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "      .   ########   .      ",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "######.##          ##.######",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#x..##.......p .......##..x#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################",
};

static const Position GHOST_CORNERS[NUM_GHOSTS] = {
    {3, -3},                     // PINKY
    {MAP_WIDTH - 4, -3},         // BLINKY
    {MAP_WIDTH - 1, MAP_HEIGHT}, // INKY
    {0, MAP_HEIGHT}              // CLYDE
};

static const char GHOST_TILES[NUM_GHOSTS] = {PINKY_TILE, BLINKY_TILE, INKY_TILE, CLYDE_TILE};

typedef struct Ghost {
        Position spawn_pos;
        Position pos;
        Position last_pos;    // use for interpolation
        Position target;
        int direction;
        int start_timeout;    // randomized delay before moving
        bool frightened;      // whether the ghost is frightened
        bool return_to_spawn; // whether to return to spawn position
        bool half_move;
} Ghost;

typedef struct Client Client;
typedef struct PacmanEnv {
        Client *client;
        bool randomize_starting_position; // randomize player starting position
        int min_start_timeout;            // randomized ghost delay range
        int max_start_timeout;
        int frightened_time; // ghost frighten time
        int max_mode_changes;
        int scatter_mode_length;
        int chase_mode_length;

        float *observations;
        int *actions;
        float *rewards;
        char *terminals;
        Log log;

        int step_count;
        int score;

        Tile *game_map;
        float *pickup_obs;
        float **pickup_obs_map;

        int remaining_pickups;

        Position *possible_spawn_pos;

        Position player_spawn_pos; // for when it's not randomized
        Position player_pos;
        Position last_player_pos;
        int player_direction;

        bool reverse_directions;
        bool scatter_mode;
        int mode_time_left;
        int mode_changes;

        int frightened_time_left;

        bool player_caught;

        Ghost ghosts[NUM_GHOSTS];
} PacmanEnv;

void add_log(PacmanEnv *env) {
    env->log.score += env->score;
    env->log.episode_return += env->score;
    env->log.episode_length = env->step_count;
    env->log.n++;
}

static inline Position pos_move_wrapped(Position pos, int direction, int distance) {
    pos = pos_move(pos, direction, distance);
    if (pos.x < 0)
        pos.x = MAP_WIDTH - 1;
    else if (pos.x >= MAP_WIDTH)
        pos.x = 0;
    return pos;
}

static inline Tile *tile_at(PacmanEnv *env, Position pos) {
    return &env->game_map[pos.y * MAP_WIDTH + pos.x];
}

static inline bool can_move_in_direction(PacmanEnv *env, Position pos, int direction) {
    return *tile_at(env, pos_move_wrapped(pos, direction, 1)) != WALL_TILE;
}

void init(PacmanEnv *env) {
    int dot_count = 0;
    Position pos;
    Tile source_tile;
    Tile *target_tile;

    env->game_map = (Tile *)calloc(MAP_WIDTH * MAP_HEIGHT, sizeof(Tile));
    env->possible_spawn_pos = (Position *)calloc(NUM_DOTS, sizeof(Position));

    env->pickup_obs = (float *)calloc(NUM_DOTS + NUM_POWERUPS, sizeof(float));
    env->pickup_obs_map = (float **)calloc(MAP_WIDTH * MAP_HEIGHT, sizeof(float *));

    int obs_map_n = 0;

    // one time map setup
    for (int y = 0; y < MAP_HEIGHT; y++) {
        for (int x = 0; x < MAP_WIDTH; x++) {
            source_tile = original_map[y][x];

            pos = (Position){x, y};
            target_tile = tile_at(env, pos);
            *target_tile = source_tile;

            float **p = &env->pickup_obs_map[y * MAP_WIDTH + x];
            *p = NULL;

            switch (source_tile) {
            case DOT_TILE:
                env->possible_spawn_pos[dot_count] = pos;
                dot_count++;
            case POWER_TILE:
                *p = &env->pickup_obs[obs_map_n++];
                break;
            case PLAYER_TILE:
                env->player_spawn_pos = pos;
                break;
            default:
                break;
            }

            for (int i = 0; i < NUM_GHOSTS; i++) {
                if (source_tile == GHOST_TILES[i]) {
                    env->ghosts[i].spawn_pos = pos;
                }
            }
        }
    }
}

void allocate(PacmanEnv *env) {
    init(env);
    env->observations = (float *)calloc(OBSERVATIONS_COUNT, sizeof(float));
    env->actions = (int *)calloc(1, sizeof(int));
    env->rewards = (float *)calloc(1, sizeof(float));
    env->terminals = (char *)calloc(1, sizeof(char));
}

void c_close(PacmanEnv *env) {
    free(env->game_map);
    free(env->possible_spawn_pos);

    free(env->pickup_obs);
    free(env->pickup_obs_map);
}

void free_allocated(PacmanEnv *env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

#define INV_MAP_WIDTH (1.0f / MAP_WIDTH)
#define INV_MAP_HEIGHT (1.0f / MAP_HEIGHT)

void compute_observations(PacmanEnv *env) {
    float *obs = env->observations;
    // player observations
    obs[0] = env->player_pos.x * INV_MAP_WIDTH;
    obs[1] = env->player_pos.y * INV_MAP_HEIGHT;
    obs[2] = env->player_direction == UP;
    obs[3] = env->player_direction == DOWN;
    obs[4] = env->player_direction == LEFT;
    obs[5] = env->player_direction == RIGHT;
    obs[6] = can_move_in_direction(env, env->player_pos, UP);
    obs[7] = can_move_in_direction(env, env->player_pos, DOWN);
    obs[8] = can_move_in_direction(env, env->player_pos, LEFT);
    obs[9] = can_move_in_direction(env, env->player_pos, RIGHT);
    obs[10] = env->frightened_time_left / (float)env->frightened_time;

    // ghost obs
    for (int i = 0; i < NUM_GHOSTS; i++) {
        Ghost *ghost = &env->ghosts[i];
        int p = PLAYER_OBSERVATIONS_COUNT + (i * GHOST_OBSERVATIONS_COUNT);

        obs[p] = ghost->pos.x * INV_MAP_WIDTH;
        obs[p + 1] = ghost->pos.y * INV_MAP_HEIGHT;
        obs[p + 2] = ghost->direction == UP;
        obs[p + 3] = ghost->direction == DOWN;
        obs[p + 4] = ghost->direction == LEFT;
        obs[p + 5] = ghost->direction == RIGHT;
        obs[p + 6] = !ghost->frightened && !ghost->return_to_spawn;
        obs[p + 7] = ghost->frightened;
        obs[p + 8] = ghost->return_to_spawn;
    }

    memcpy(obs + PLAYER_OBSERVATIONS_COUNT + (NUM_GHOSTS * GHOST_OBSERVATIONS_COUNT),
           env->pickup_obs, sizeof(float) * (NUM_DOTS + NUM_POWERUPS));
}

void update_interpolation(PacmanEnv *env) {
    env->last_player_pos = env->player_pos;

    for (int i = 0; i < NUM_GHOSTS; i++) {
        Ghost *ghost = &env->ghosts[i];
        if (!ghost->frightened || !ghost->half_move) {
            ghost->last_pos = ghost->pos;
        }
    }
}

static inline void reset_round(PacmanEnv *env) {
    env->scatter_mode = false;
    env->mode_time_left = 0;
    env->mode_changes = 0;
    env->frightened_time_left = 0;

    env->step_count = 0;
    env->remaining_pickups = NUM_DOTS + NUM_POWERUPS;

    for (int i = 0; i < NUM_DOTS + NUM_POWERUPS; i++) {
        env->pickup_obs[i] = 1.0f;
    }

    for (int i = 0; i < NUM_GHOSTS; i++) {
        Ghost *ghost = &env->ghosts[i];
        ghost->pos = ghost->spawn_pos;
        ghost->direction = UP;
        ghost->start_timeout = rand_range(env->min_start_timeout, env->max_start_timeout);
        ghost->frightened = false;
        ghost->return_to_spawn = false;
        ghost->half_move = false;
    }

    for (int y = 0; y < MAP_HEIGHT; y++) {
        for (int x = 0; x < MAP_WIDTH; x++) {
            env->game_map[y * MAP_WIDTH + x] = (Tile)original_map[y][x];
        }
    }

    if (env->randomize_starting_position) {
        int player_randomizer = rand() % NUM_DOTS;
        env->player_pos = env->possible_spawn_pos[player_randomizer];
    } else {
        env->player_pos = env->player_spawn_pos;
    }
    env->player_direction = RIGHT;
}

void c_reset(PacmanEnv *env) {
    env->score = 0;
    reset_round(env);
    compute_observations(env);
    update_interpolation(env);
}

static inline void set_frightened(PacmanEnv *env) {
    env->frightened_time_left = env->frightened_time;
    env->reverse_directions = true;

    for (int i = 0; i < NUM_GHOSTS; i++) {
        env->ghosts[i].frightened = !env->ghosts[i].return_to_spawn;
    }
}

static inline void unset_pickup_obs(PacmanEnv *env, Position pos) {
    float *obs = env->pickup_obs_map[pos.y * MAP_WIDTH + pos.x];
    if (obs != NULL) {
        *obs = 0.0f;
    }
}

static inline void player_move(PacmanEnv *env, int action) {
    Position new_pos = pos_move_wrapped(env->player_pos, action, 1);
    Tile *new_tile = tile_at(env, new_pos);

    // if the player action is into a wall, move in the current direction
    if (*new_tile == WALL_TILE) {
        new_pos = pos_move_wrapped(env->player_pos, env->player_direction, 1);
        new_tile = tile_at(env, new_pos);
    } else {
        env->player_direction = action;
    }

    if (*new_tile != WALL_TILE) {
        env->player_pos = new_pos;

        if (*new_tile == POWER_TILE) {
            set_frightened(env);
        }
        if (*new_tile == DOT_TILE || *new_tile == POWER_TILE) {
            env->score += 1.0f;
            env->rewards[0] += 1.0f;

            env->remaining_pickups--;
            *new_tile = EMPTY_TILE;

            unset_pickup_obs(env, new_pos);
        }
    }
}

static inline int ghost_movement_options(PacmanEnv *env, Ghost *ghost, int *directions) {
    int n = 0;
    int rev = reverse_direction(ghost->direction);

    for (int i = 0; i < 4; i++) {
        if (i != rev && can_move_in_direction(env, ghost->pos, i)) {
            directions[n++] = i;
        }
    }
    return n;
}

static inline int min_index(int *array, int length) {
    int min_index = 0;
    for (int i = 1; i < length; i++) {
        if (array[i] < array[min_index]) {
            min_index = i;
        }
    }
    return min_index;
}

static inline int ghost_direction(PacmanEnv *env, Ghost *ghost) {
    int directions[4];
    int distances[4];
    if (env->reverse_directions && !ghost->return_to_spawn) {
        return reverse_direction(ghost->direction);
    }

    int option_count = ghost_movement_options(env, ghost, directions);

    if (option_count == 1) {
        return directions[0];
    }

    if (ghost->frightened) {
        int random_index = rand() % option_count;
        return directions[random_index];
    }

    for (int i = 0; i < option_count; i++) {
        distances[i] = pos_distance_squared(pos_move(ghost->pos, directions[i], 1), ghost->target);
    }

    return directions[min_index(distances, option_count)];
}

#define PINKY 0
#define BLINKY 1
#define INKY 2
#define CLYDE 3

static inline void set_chase_targets(PacmanEnv *env) {
    env->ghosts[PINKY].target = pos_move(env->player_pos, env->player_direction, PINKY_TARGET_LEAD);
    env->ghosts[BLINKY].target = env->player_pos;

    Position inky_intermediate = pos_move(env->player_pos, env->player_direction, INKY_TARGET_LEAD);
    env->ghosts[INKY].target.x = 2 * inky_intermediate.x - env->ghosts[1].pos.x;
    env->ghosts[INKY].target.y = 2 * inky_intermediate.y - env->ghosts[1].pos.y;

    int clyde_distance = pos_distance_squared(env->player_pos, env->ghosts[3].pos);
    if (clyde_distance > CLYDE_TARGET_RADIUS * CLYDE_TARGET_RADIUS) {
        env->ghosts[CLYDE].target = env->player_pos;
    } else {
        env->ghosts[CLYDE].target = GHOST_CORNERS[3];
    }
}

static inline bool check_collision(Position a, Position old_a, Position b, Position old_b) {
    return (a.x >= b.x - 1 && a.x <= b.x + 1 && a.y == b.y) ||
           (a.y >= b.y - 1 && a.y <= b.y + 1 && a.x == b.x);
}

static inline void ghost_move(PacmanEnv *env, Ghost *ghost, Position old_player_pos) {
    Position old_ghost_position = ghost->pos;
    --ghost->start_timeout;

    if (ghost->frightened && ghost->half_move) {
        ghost->half_move = false;
    } else {
        ghost->half_move = true;

        if (ghost->return_to_spawn) {
            ghost->target = ghost->spawn_pos;
        }

        ghost->direction = ghost_direction(env, ghost);

        if (ghost->start_timeout < 0) {
            Position new_pos = pos_move_wrapped(ghost->pos, ghost->direction, 1);
            if (*tile_at(env, new_pos) != WALL_TILE) {
                ghost->pos = new_pos;
            }
        }
    }

    if (ghost->return_to_spawn) {
        if (pos_equal(ghost->pos, ghost->spawn_pos)) {
            ghost->return_to_spawn = false;
        }
    } else if (check_collision(ghost->pos, old_ghost_position, env->player_pos, old_player_pos)) {
        if (ghost->frightened) {
            ghost->frightened = false;
            ghost->half_move = false;
            ghost->return_to_spawn = true;

            env->rewards[0] += 1.0f;
        } else {
            env->player_caught = true;
        }
    }
}

static inline void check_mode_change(PacmanEnv *env) {
    if (env->mode_changes > env->max_mode_changes) {
        return;
    }

    if (--env->mode_time_left <= 0) {
        env->scatter_mode = !env->scatter_mode;
        env->reverse_directions = true;
        env->mode_changes++;

        if (env->scatter_mode) {
            env->mode_time_left = env->scatter_mode_length;
        } else {
            env->mode_time_left = env->chase_mode_length;
        }
    }
}

void c_step(PacmanEnv *env) {
    update_interpolation(env);

    Position old_player_pos = env->player_pos;
    int action = env->actions[0];

    env->step_count += 1;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;

    env->reverse_directions = false;
    env->player_caught = false;

    if (env->frightened_time_left > 0) {
        env->frightened_time_left--;
    } else {
        for (int i = 0; i < NUM_GHOSTS; i++) {
            env->ghosts[i].frightened = false;
            env->ghosts[i].half_move = false;
        }
    }

    check_mode_change(env);
    if (env->scatter_mode) {
        for (int i = 0; i < NUM_GHOSTS; i++) {
            env->ghosts[i].target = GHOST_CORNERS[i];
        }
    } else {
        set_chase_targets(env);
    }

    player_move(env, action);

    for (int i = 0; i < NUM_GHOSTS; i++) {
        ghost_move(env, &env->ghosts[i], old_player_pos);
    }

    compute_observations(env);

    if (env->player_caught || env->step_count >= MAX_STEPS || env->remaining_pickups <= 0) {
        add_log(env);

        env->terminals[0] = 1;
        c_reset(env);
    }
}

typedef struct DirectionSprites {
        Texture2D up;
        Texture2D down;
        Texture2D left;
        Texture2D right;
} DirectionSprites;

typedef struct Client Client;
struct Client {
        int tile_size;
        int frame;

        Texture2D tileset;
        Texture2D pacman;
        Texture2D frightened;
        DirectionSprites ghost_sprites[4];
        DirectionSprites eyes;
};

Vector2 lerp_position(Position a, Position b, float progress) {
    if (abs(a.x - b.x) > 1) {
        b.x = a.x;
    }

    float a_x = (float)a.x;
    float a_y = (float)a.y;
    float b_x = (float)b.x;
    float b_y = (float)b.y;

    return (Vector2){a_x + (b_x - a_x) * progress, a_y + (b_y - a_y) * progress};
}

void draw_tiled(Client *client, Texture2D texture, Vector2 position, float rotation, bool flip_x,
                float source_width, float source_height) {
    Rectangle source = (Rectangle){0, 0, flip_x ? -source_width : source_width, source_height};

    DrawTexturePro(texture, source,
                   (Rectangle){(position.x + 0.50f) * client->tile_size,
                               (position.y + 0.50f) * client->tile_size + PX_PADDING_TOP,
                               client->tile_size * 1.5f, client->tile_size * 1.5f},
                   (Vector2){client->tile_size * 0.75f, client->tile_size * 0.75f}, rotation,
                   WHITE);
}

void draw_entity(Client *client, Texture2D texture, Position previous_pos, Position pos,
                 float progress, float rotation, bool flip_x, float source_width,
                 float source_height) {
    Vector2 position;

    if (pos.x == 0 && previous_pos.x == MAP_WIDTH - 1) {
        position = lerp_position((Position){-1, previous_pos.y}, pos, progress);
        draw_tiled(client, texture, position, rotation, flip_x, source_width, source_height);

        position.x += (float)MAP_WIDTH;
        draw_tiled(client, texture, position, rotation, flip_x, source_width, source_height);
    } else if (previous_pos.x == 0 && pos.x == MAP_WIDTH - 1) {
        position = lerp_position((Position){MAP_WIDTH, previous_pos.y}, pos, progress);
        draw_tiled(client, texture, position, rotation, flip_x, source_width, source_height);

        position.x -= (float)MAP_WIDTH;
        draw_tiled(client, texture, position, rotation, flip_x, source_width, source_height);
    } else {
        position = lerp_position(previous_pos, pos, progress);
        draw_tiled(client, texture, position, rotation, flip_x, source_width, source_height);
    }
}


Client *make_client(PacmanEnv *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));
    env->client = client;
    client->tile_size = 20;

    update_interpolation(env);

    srand(time(NULL));

    InitWindow(client->tile_size * MAP_WIDTH, client->tile_size * MAP_HEIGHT + PX_PADDING_TOP,
               "PufferLib Pacman");
    SetTargetFPS(60);

    client->tileset = LoadTexture("resources/pacman/tileset.png");
    client->pacman = LoadTexture("resources/shared/puffers_128.png");
    client->frightened = LoadTexture("resources/pacman/scared.png");

    client->ghost_sprites[0].up = LoadTexture("resources/pacman/pinky_up.png");
    client->ghost_sprites[0].down = LoadTexture("resources/pacman/pinky_down.png");
    client->ghost_sprites[0].left = LoadTexture("resources/pacman/pinky_left.png");
    client->ghost_sprites[0].right = LoadTexture("resources/pacman/pinky_right.png");

    client->ghost_sprites[1].up = LoadTexture("resources/pacman/blinky_up.png");
    client->ghost_sprites[1].down = LoadTexture("resources/pacman/blinky_down.png");
    client->ghost_sprites[1].left = LoadTexture("resources/pacman/blinky_left.png");
    client->ghost_sprites[1].right = LoadTexture("resources/pacman/blinky_right.png");

    client->ghost_sprites[2].up = LoadTexture("resources/pacman/inky_up.png");
    client->ghost_sprites[2].down = LoadTexture("resources/pacman/inky_down.png");
    client->ghost_sprites[2].left = LoadTexture("resources/pacman/inky_left.png");
    client->ghost_sprites[2].right = LoadTexture("resources/pacman/inky_right.png");

    client->ghost_sprites[3].up = LoadTexture("resources/pacman/clyde_up.png");
    client->ghost_sprites[3].down = LoadTexture("resources/pacman/clyde_down.png");
    client->ghost_sprites[3].left = LoadTexture("resources/pacman/clyde_left.png");
    client->ghost_sprites[3].right = LoadTexture("resources/pacman/clyde_right.png");

    client->eyes.up = LoadTexture("resources/pacman/eyes_up.png");
    client->eyes.down = LoadTexture("resources/pacman/eyes_down.png");
    client->eyes.left = LoadTexture("resources/pacman/eyes_left.png");
    client->eyes.right = LoadTexture("resources/pacman/eyes_right.png");

    return client;
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void render_ghost(Client *client, Ghost *ghost, DirectionSprites *sprites, float progress) {
    Texture2D texture;

    if (ghost->frightened) {
        texture = client->frightened;

        progress *= 0.5f;
        if (!ghost->half_move) {
            progress += 0.5f;
        }
    } else {
        if (ghost->return_to_spawn) {
            sprites = &client->eyes;
        }

        switch (ghost->direction) {
        case UP:
            texture = sprites->up;
            break;
        case DOWN:
            texture = sprites->down;
            break;
        case LEFT:
            texture = sprites->left;
            break;
        case RIGHT:
            texture = sprites->right;
            break;
        }
    }

    draw_entity(client, texture, ghost->last_pos, ghost->pos, progress, 0.0f, false, 16, 16);
}

void render_player(Client *client, PacmanEnv *env, float progress) {
    float rotation = 0.0f;

    if (env->player_direction == UP) {
        rotation = 270.0f;
    } else if (env->player_direction == DOWN) {
        rotation = 90.0f;
    }

    draw_entity(client, client->pacman, env->last_player_pos, env->player_pos, progress,
                rotation, env->player_direction == LEFT, 128, 128);
}

bool is_wall(PacmanEnv *env, int x, int y) {
    Position pos = {x, y};
    if (pos.x < 0 || pos.x >= MAP_WIDTH || pos.y < 0 || pos.y >= MAP_HEIGHT) {
        return true;
    }

    return *tile_at(env, pos) == WALL_TILE;
}

#define TILE_N (1 << 0)
#define TILE_NE (1 << 1)
#define TILE_E (1 << 2)
#define TILE_SE (1 << 3)
#define TILE_S (1 << 4)
#define TILE_SW (1 << 5)
#define TILE_W (1 << 6)
#define TILE_NW (1 << 7)

int get_tile_index(PacmanEnv *env, Position pos) {
    int tile_bits = 0;

    if (is_wall(env, pos.x, pos.y - 1))
        tile_bits |= TILE_N;
    if (is_wall(env, pos.x + 1, pos.y - 1))
        tile_bits |= TILE_NE;
    if (is_wall(env, pos.x + 1, pos.y))
        tile_bits |= TILE_E;
    if (is_wall(env, pos.x + 1, pos.y + 1))
        tile_bits |= TILE_SE;
    if (is_wall(env, pos.x, pos.y + 1))
        tile_bits |= TILE_S;
    if (is_wall(env, pos.x - 1, pos.y + 1))
        tile_bits |= TILE_SW;
    if (is_wall(env, pos.x - 1, pos.y))
        tile_bits |= TILE_W;
    if (is_wall(env, pos.x - 1, pos.y - 1))
        tile_bits |= TILE_NW;

    switch (tile_bits) {
    case TILE_NW | TILE_N | TILE_NE | TILE_W | TILE_E | TILE_SE:
    case TILE_NW | TILE_N | TILE_NE | TILE_W | TILE_E | TILE_SW:
    case TILE_NW | TILE_N | TILE_NE | TILE_W | TILE_E:
        return 13;

    case TILE_NE | TILE_E | TILE_SE | TILE_N | TILE_S | TILE_NW:
    case TILE_NE | TILE_E | TILE_SE | TILE_N | TILE_S | TILE_SW:
    case TILE_NE | TILE_E | TILE_SE | TILE_N | TILE_S:
        return 6;

    case TILE_SE | TILE_S | TILE_SW | TILE_E | TILE_W | TILE_NW:
    case TILE_SE | TILE_S | TILE_SW | TILE_E | TILE_W | TILE_NE:
    case TILE_SE | TILE_S | TILE_SW | TILE_E | TILE_W:
        return 1;

    case TILE_SW | TILE_W | TILE_NW | TILE_S | TILE_N | TILE_NE:
    case TILE_SW | TILE_W | TILE_NW | TILE_S | TILE_N | TILE_SE:
    case TILE_SW | TILE_W | TILE_NW | TILE_S | TILE_N:
        return 8;
    case TILE_S | TILE_E | TILE_SE:
        return 0;
    case TILE_S | TILE_W | TILE_SW:
        return 2;
    case TILE_N | TILE_E | TILE_NE:
        return 12;
    case TILE_N | TILE_W | TILE_NW:
        return 14;

    case 255 & ~TILE_NW:
        return 25;
    case 255 & ~TILE_NE:
        return 26;
    case 255 & ~TILE_SW:
        return 31;
    case 255 & ~TILE_SE:
        return 32;
    }

    return 7;
}

void render_tile(Client *client, PacmanEnv *env, Position pos) {
    int tile_index = get_tile_index(env, pos);
    int tile_x = tile_index % 6;
    int tile_y = tile_index / 6;
    Rectangle source = (Rectangle){tile_x * 9, tile_y * 9, 8, 8};

    DrawTexturePro(client->tileset, source,
                   (Rectangle){pos.x * client->tile_size,
                               pos.y * client->tile_size + PX_PADDING_TOP, client->tile_size,
                               client->tile_size},
                   (Vector2){0, 0}, 0.0f, WHITE);
}

void render_map(Client *client, PacmanEnv *env) {
    for (int y = 0; y < MAP_HEIGHT; y++) {
        for (int x = 0; x < MAP_WIDTH; x++) {
            char tile = env->game_map[y * MAP_WIDTH + x];
            if (tile == WALL_TILE) {
                render_tile(client, env, (Position){x, y});
            } else if (tile == DOT_TILE) {
                float width = client->tile_size / 4.0f;
                float height = client->tile_size / 4.0f;
                DrawRectangle(x * client->tile_size + client->tile_size / 2.0f - width / 2.0f,
                              y * client->tile_size + client->tile_size / 2.0f - height / 2.0f +
                                  PX_PADDING_TOP,
                              width, height, PUFF_WHITE);
            } else if (tile == POWER_TILE) {
                DrawCircle(x * client->tile_size + client->tile_size / 2.0f,
                           y * client->tile_size + client->tile_size / 2.0f + PX_PADDING_TOP,
                           client->tile_size / 3.0f, PUFF_RED);
            }
        }
    }
}

void handle_input(PacmanEnv *env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }
}

void c_render(PacmanEnv *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client *client = env->client;

    float progress = client->frame / (float)FRAMES;
    client->frame = (client->frame + 1) % (FRAMES);

    handle_input(env);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    render_map(client, env);

    render_player(client, env, progress);
    for (int i = 0; i < 4; i++) {
        render_ghost(client, &env->ghosts[i], &client->ghost_sprites[i], progress);
    }

    DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
    DrawText(TextFormat("Mode: %s", env->scatter_mode ? "Scatter" : "Chase"), 150, 10, 20, WHITE);

    EndDrawing();
}

void unload_direction_sprites(DirectionSprites *sprites) {
    UnloadTexture(sprites->up);
    UnloadTexture(sprites->down);
    UnloadTexture(sprites->left);
    UnloadTexture(sprites->right);
}

void close_client(Client *client) {
    CloseWindow();

    UnloadTexture(client->tileset);
    UnloadTexture(client->pacman);
    UnloadTexture(client->frightened);
    for (int i = 0; i < 4; i++) {
        unload_direction_sprites(&client->ghost_sprites[i]);
    }
    unload_direction_sprites(&client->eyes);
    free(client);
}
