/* Boxoban: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */


//SWITCH CLEAR AND OBS BOXES CONDITIONS TO CHECK WE ARENT CHECKING OUT OF BOUNDS

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "raylib.h"

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char AGENT = 0;
const unsigned char WALLS = 1;
const unsigned char BOXES = 2;
const unsigned char TARGET = 3;

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

typedef struct {
    Texture2D wall;
    Texture2D box;
    Texture2D target;
    Texture2D floor;
    Texture2D agent;
} Client;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    unsigned char* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int size;
    int tick;
    uint8_t* puzzles;
    int puzzle_count;
    uint8_t* v_puzzles;
    int v_puzzle_count;
    int agent_x;
    int agent_y;
    Client* client;
} Boxoban;



//100 bytes agent
//100 bytes walls
//100 bytes boxes
//100 bytes targets
#define PUZZLE_SIZE 400

static inline void load_bin(uint8_t **buffer, int *count, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("error opening %s\n", path);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    *count = size / PUZZLE_SIZE;
    *buffer = malloc(size);
    if (!*buffer) {
        printf("malloc failed\n");
        exit(1);
    }

    fread(*buffer, 1, size, f);
    fclose(f);
}

static inline const uint8_t* get_puzzle(const Boxoban *env, int index) {
    return env->puzzles + index * PUZZLE_SIZE;
}

static inline const uint8_t* get_v_puzzle(const Boxoban *env, int index) {
    return env->v_puzzles + index * PUZZLE_SIZE;
}

static inline const uint8_t* get_random_puzzle(const Boxoban *env) {
    int idx = rand() % env->puzzle_count;
    return get_puzzle(env, idx);
}

static inline const uint8_t* get_random_v_puzzle(const Boxoban *env) {
    int idx = rand() % env->v_puzzle_count;
    return get_v_puzzle(env, idx);
}

void init (Boxoban* env) {
    load_bin(&env->puzzles, &env->puzzle_count, "boxoban_maps.bin");
    load_bin(&env->v_puzzles, &env->v_puzzle_count, "boxoban_maps_valid.bin");
}

//Entity,x,y  convention y moves top to bottom
#define OBS(e,x,y) (env->observations[(e)*env->size*env->size + (y)*env->size + (x)])

void add_log(Boxoban* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

void get_agent_pos(Boxoban* env){
    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            if (OBS(AGENT, x, y) == 1) {
                env->agent_x = x;
                env->agent_y = y;
            }
        }
    }
}

bool clear(Boxoban* env, int x, int y) {
    if (x < 0 || y < 0 || x >= env->size || y >= env->size) {
        return false;
    }
    return (OBS(WALLS, x, y) == 0) && (OBS(BOXES, x, y) == 0);
}

// Required function
void c_reset(Boxoban* env) {
    const uint8_t *p = get_random_puzzle(env);
    memcpy(env->observations, p, PUZZLE_SIZE);
    env->tick = 0;
    get_agent_pos(env);
}

void move_entity(Boxoban* env,unsigned char entity,int x, int y, int dx, int dy) {
    OBS(entity, x, y) = 0;
    OBS(entity, x + dx, y + dy) = 1;
}


void take_action(Boxoban* env, int action) {
    int dx = 0;
    int dy = 0;
    if (action == DOWN) {
        dy = 1;
        if (clear(env, env->agent_x, env->agent_y + dy)) {
            move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
            env->agent_y += dy;
            return;
        }
        else if (OBS(BOXES, env->agent_x, env->agent_y + dy) == 1 
                && clear(env, env->agent_x, env->agent_y + 2*dy)) 
        {
            move_entity(env, BOXES, env->agent_x, env->agent_y + dy, dx, dy);
            move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
            env->agent_y += dy;
            return;
        }
    }
    else if (action == UP) {
        dy = -1;
        if (clear(env, env->agent_x, env->agent_y + dy)) {
            move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
            env->agent_y += dy;
            return;
        }
        else if (OBS(BOXES, env->agent_x, env->agent_y + dy) == 1 
                && clear(env, env->agent_x, env->agent_y + 2*dy)) 
        {
            move_entity(env, BOXES, env->agent_x, env->agent_y+dy, dx, dy);
            move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
            env->agent_y += dy;
            return;
        }
    }
    else if (action == LEFT) {
        dx = -1;
        if (clear(env, env->agent_x + dx, env->agent_y)) {
            move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
            env->agent_x += dx;
            return;
        }
        else if (OBS(BOXES, env->agent_x + dx, env->agent_y) == 1 
                && clear(env, env->agent_x + 2*dx, env->agent_y)) 
        {
            move_entity(env, BOXES, env->agent_x+dx, env->agent_y, dx, dy);
            move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
            env->agent_x += dx;
            return;
        }
    }
    else if (action == RIGHT) {
        dx = 1;
        if (clear(env, env->agent_x + dx, env->agent_y)) {
            move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
            env->agent_x += dx;
            return;
        }
        else if (OBS(BOXES, env->agent_x + dx, env->agent_y) == 1 
                && clear(env, env->agent_x + 2*dx, env->agent_y)) 
        {
            move_entity(env, BOXES, env->agent_x+dx, env->agent_y, dx, dy);
            move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
            env->agent_x += dx;
            return;
        }
        
    }
}

        
bool goal(Boxoban* env) {
    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            if (OBS(BOXES, x, y) == 1 && OBS(TARGET, x, y) == 0) {
                return false;
            }
        }
    }
    return true;
}

// Required function
void c_step(Boxoban* env) {
    env->tick += 1;

    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    //take action
    take_action(env, action);
    
    //Terminals
    if (goal(env)) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    //new obs is modified in place
    env->rewards[0] -= 0.1; //length penalty
}

Client* c_create(Boxoban* env) {
    Client* client = calloc(1,sizeof(Client));
    const char *sprite_search_paths[] = {
        "sprites_pack/PNG",
        "pufferlib/ocean/boxoban/sprites_pack/PNG",
        "../pufferlib/ocean/boxoban/sprites_pack/PNG",
    };
    const char *sprite_base = NULL;
    for (unsigned i = 0; i < sizeof(sprite_search_paths)/sizeof(sprite_search_paths[0]); i++) {
        if (DirectoryExists(sprite_search_paths[i])) {
            sprite_base = sprite_search_paths[i];
            break;
        }
    }
    if (sprite_base == NULL) {
        TraceLog(LOG_WARNING, "Boxoban sprites not found next to executable, using default relative path");
        sprite_base = "sprites_pack/PNG";
    }

    char resource_path[256] = {0};

    snprintf(resource_path, sizeof(resource_path), "%s/Wall_black.png", sprite_base);
    client->wall = LoadTexture(resource_path);
    snprintf(resource_path, sizeof(resource_path), "%s/Crate_black.png", sprite_base);
    client->box = LoadTexture(resource_path);
    snprintf(resource_path, sizeof(resource_path), "%s/EndPoint_black.png", sprite_base);
    client->target = LoadTexture(resource_path);
    snprintf(resource_path, sizeof(resource_path), "%s/GroundGravel_Concrete.png", sprite_base);
    client->floor = LoadTexture(resource_path);
    client->agent = LoadTexture("resources/shared/puffers_128.png");

    env-> client = client;
    return client;
}

#define TILE 32

Texture2D choose_sprite(Client *c, Boxoban *env, int x, int y) {
    int a = OBS(AGENT,  x, y);
    int w = OBS(WALLS,  x, y);
    int b = OBS(BOXES,  x, y);
    int t = OBS(TARGET, x, y);

    if (w) return c->wall;
    //if (b && t) return c->box_on_target;
    if (b) return c->box;
    //if (a && t) return c->agent_on_target;
    if (a) return c->agent;
    if (t) return c->target;

    return c->floor;
}


// Required function. Should handle creating the client on first call
void c_render(Boxoban* env) {
    if (!IsWindowReady()) {
        InitWindow(TILE*env->size, TILE*env->size, "PufferLib Boxoban");
        SetTargetFPS(5);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (env->client == NULL) {
        env->client = c_create(env);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    
    for (int y = 0; y < env->size; y++) {
      for (int x = 0; x < env->size; x++) {
          Texture2D tex = choose_sprite(env->client, env, x, y);
          Rectangle dest = {x * TILE, y * TILE, TILE, TILE};

          if (tex.id == env->client->agent.id) {
              Rectangle src = {0, 0, tex.width / 2.0f, (float)tex.height};
              DrawTexturePro(tex, src, dest, (Vector2){0, 0}, 0.0f, WHITE);
          } else {
              DrawTexturePro(
                  tex,
                  (Rectangle){0, 0, (float)tex.width, (float)tex.height},
                  dest,
                  (Vector2){0, 0},
                  0.0f,
                  WHITE
              );
          }
      }
  }


    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Boxoban* env) {
    if (IsWindowReady()) {

        UnloadTexture(env->client->wall);
        UnloadTexture(env->client->box);
        UnloadTexture(env->client->target);
        UnloadTexture(env->client->floor);
        UnloadTexture(env->client->agent);
        free(env->client);
        free(env->puzzles);
        free(env->v_puzzles);
        CloseWindow();
    }
}

