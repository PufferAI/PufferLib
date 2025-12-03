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

extern uint8_t *MAP_BASE;
extern size_t MAP_FILESIZE;
extern size_t PUZZLE_COUNT;
extern size_t PUZZLE_SIZE;

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
    Texture2D box_on_target;
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
    int max_steps;
    int agent_x;
    int agent_y;
    Client* client;
} Boxoban;

void ensure_map_loaded(void); //declare from binding.c

static inline const uint8_t get_random_puzzle_idx(const Boxoban *env) {
    int idx = rand() % PUZZLE_COUNT;
    return idx;
}

void init (Boxoban* env) {
    ensure_map_loaded();
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
    const uint8_t i = get_random_puzzle_idx(env);
    memcpy(env->observations, 
            MAP_BASE + (size_t)i * PUZZLE_SIZE, PUZZLE_SIZE);
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

    if (env->tick >= env->max_steps) {
        env->terminals[0] = 1;
        env->rewards[0] = -1;
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
    snprintf(resource_path, sizeof(resource_path), "%s/EndPoint_Blue.png", sprite_base);
    client->box_on_target = LoadTexture(resource_path);
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
    if (b && t) return c->box_on_target;
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
        CloseWindow();
    }
}

