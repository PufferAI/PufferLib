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


/*Maps are stored in the binary files with the name indicating the difficulty.
If the bin doesn't exist it is created on the fly.
Once the bin exists and MMAP is created and shared between envs (see binding.c)
*/

extern uint8_t *MAP_BASE;
extern size_t MAP_FILESIZE;
extern size_t PUZZLE_COUNT;
extern size_t PUZZLE_SIZE;

#ifdef BOXOBAN_MAPS_IMPLEMENTATION
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

uint8_t *MAP_BASE = NULL;
size_t MAP_FILESIZE = 0;
size_t PUZZLE_COUNT = 0;
size_t PUZZLE_SIZE = 400;
static char* BOXOBAN_MAP_PATH = NULL;

static void reset_map_cache(void) {
    if (MAP_BASE != NULL && MAP_BASE != MAP_FAILED && MAP_FILESIZE > 0) {
        munmap(MAP_BASE, MAP_FILESIZE);
    }
    MAP_BASE = NULL;
    MAP_FILESIZE = 0;
    PUZZLE_COUNT = 0;
}

int boxoban_set_map_path(const char *path) {
    if (path == NULL) {
        return -1;
    }
    if (BOXOBAN_MAP_PATH != NULL && strcmp(BOXOBAN_MAP_PATH, path) == 0) {
        return 0;
    }

    char* copied = malloc(strlen(path) + 1);
    if (copied == NULL) {
        return -1;
    }
    strcpy(copied, path);

    reset_map_cache();
    free(BOXOBAN_MAP_PATH);
    BOXOBAN_MAP_PATH = copied;
    return 0;
}

static const char* get_default_map_path(void) {
    const char* env_path = getenv("BOXOBAN_MAP_BIN");
    if (env_path != NULL) {
        return env_path;
    }
    return "pufferlib/ocean/boxoban/boxoban_maps_basic.bin";
}

void ensure_map_loaded(void) {
    if (MAP_BASE != NULL)
        return;

    if (BOXOBAN_MAP_PATH == NULL) {
        const char* default_path = get_default_map_path();
        if (boxoban_set_map_path(default_path) != 0) {
            fprintf(stderr, "Failed to set default Boxoban map path\n");
            abort();
        }
    }

    int fd = open(BOXOBAN_MAP_PATH, O_RDONLY);
    if (fd < 0) {
        perror("open");
        abort();
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        abort();
    }

    MAP_FILESIZE = st.st_size;
    PUZZLE_COUNT = MAP_FILESIZE/PUZZLE_SIZE;

    MAP_BASE = mmap(NULL, MAP_FILESIZE, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (MAP_BASE == MAP_FAILED) {
        perror("mmap");
        abort();
    }
}
#endif

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n_targets; // Number of targets currently boxed
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
    unsigned char* intermediate_rewards;
    float int_r_coeff;
    float target_loss_pen_coeff;
    int n_targets; //num targets currently boxed
    int n_boxes; //boxes in map
    Client* client;
    int win;
} Boxoban;

void ensure_map_loaded(void); //declare from binding.c
int boxoban_set_map_path(const char *path);

//Entity,x,y  convention y moves top to bottom

static inline void set_entity(Boxoban *env, int entity, int x, int y, unsigned char value) {
    env->observations[(entity)*env->size*env->size + (y)*env->size + (x)] = value;
}

static inline unsigned char get_entity(Boxoban *env, int entity, int x, int y) {
    return env->observations[(entity)*env->size*env->size + (y)*env->size + (x)];
}

static inline void set_intermediate_reward(Boxoban *env, int x, int y, unsigned char value) {
    env->intermediate_rewards[(y)*env->size + (x)] = value;
}

static inline unsigned char get_intermediate_reward_status(Boxoban *env, int x, int y) {
    return env->intermediate_rewards[(y)*env->size + (x)];
}

static inline const uint32_t get_random_puzzle_idx(const Boxoban *env) {
    int idx = rand() % PUZZLE_COUNT;
    return idx;
}

static inline int count_boxes(Boxoban *env){
    int total = 0;
    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            total += get_entity(env, BOXES, x, y);
        }
    }
    return total;
}

static inline int boxes_on_targets(Boxoban *env); //decare to allow add_log

void init (Boxoban* env) {
    ensure_map_loaded();
    env->intermediate_rewards = calloc(env->size*env->size, sizeof(int));
    env->win = 0;
  }


void add_log(Boxoban* env) {
    float denom = (float)env->n_boxes;
    float num = (float)boxes_on_targets(env);
    env->log.perf += (env->win== 1) ? 1.0 : num/denom;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n_targets += env->n_targets;
    env->log.n++;
}

void get_agent_pos(Boxoban* env){
    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            if (get_entity(env, AGENT, x, y) == 1) {
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
    return (get_entity(env, WALLS, x, y) == 0) && (get_entity(env, BOXES, x, y) == 0);
}

// Required function
void c_reset(Boxoban* env) {
    const uint32_t i = get_random_puzzle_idx(env);
    memcpy(env->observations, 
            MAP_BASE + (size_t)i * PUZZLE_SIZE, PUZZLE_SIZE);

    memset(env->intermediate_rewards, 0, env->size*env->size*sizeof(int));
    memcpy(env->intermediate_rewards,
            env->observations + TARGET * env->size * env->size,env->size * env->size);
    env->n_boxes = count_boxes(env);
    env->tick = 0;
    env->n_targets = 0;
    get_agent_pos(env);
    env->win = 0;
}

//Updates OBS for moved entity
void move_entity(Boxoban* env,unsigned char entity,int x, int y, int dx, int dy) {
    set_entity(env, entity, x, y, 0);
    set_entity(env, entity, x + dx, y + dy, 1);
}

//NB THIS IS DESTRUCTIVE AND SHOULD BE RUN ONCE PER STEP
//INTERMEDIATE_REWARD(x, y) is a grid and = 1  means there is reward left to claim
// for that target
float get_intermediate_rewards(Boxoban* env) {
    float int_r = 0;
    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            if (get_entity(env, BOXES, x, y) == 1
                    && get_entity(env, TARGET, x, y) == 1
                    && get_intermediate_reward_status(env, x, y) == 1) {
                int_r += 1.0;
                set_intermediate_reward(env, x, y, 0);
            }
                
        }
    }
    return int_r;
 }

static inline int boxes_on_targets(Boxoban *env) {
    int total = 0;
    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            total += get_entity(env, BOXES, x, y) && get_entity(env, TARGET, x, y);
        }
    }
     return total;
}

//If clear is true, move the agent to the new position
//If clear is false, but its a box and box is clear move both
//If not clear, or not clear beyond box, do nothing
//Updates agent position and calls move_entity to update OBS
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
        else if (clear(env, env->agent_x, env->agent_y + 2*dy)
                && get_entity(env, BOXES, env->agent_x, env->agent_y + dy) == 1)

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
        else if (clear(env, env->agent_x, env->agent_y + 2*dy) 
                && get_entity(env, BOXES, env->agent_x, env->agent_y + dy) == 1)
                
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
        else if (clear(env, env->agent_x + 2*dx, env->agent_y)
                && get_entity(env, BOXES, env->agent_x + dx, env->agent_y) == 1)
                
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
        else if (clear(env, env->agent_x + 2*dx, env->agent_y)
                && get_entity(env, BOXES, env->agent_x + dx, env->agent_y) == 1)
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
            if (get_entity(env, BOXES, x, y) == 1 && get_entity(env, TARGET, x, y) == 0)
            {
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
    env->rewards[0] = 0.0;

    float on_target = boxes_on_targets(env);

    take_action(env, action); //modifies observations in place

    float on_target_after = boxes_on_targets(env);


    if (on_target_after < on_target) {
        env->rewards[0] -= env->target_loss_pen_coeff * (on_target - on_target_after);
    }

    float num_int_rewards = get_intermediate_rewards(env); //get available rewards for first time box targets - destructive

    if (num_int_rewards > 0) {
        env->n_targets += (int)num_int_rewards;
        if (env->int_r_coeff > 0) {
            env->rewards[0] += num_int_rewards * env->int_r_coeff;
        }
    }
    
    //Terminals
    if (goal(env)) {
        env->terminals[0] = 1;
        env->rewards[0] += 1.0;
        env->win = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    if (env->tick >= env->max_steps) {
        env->terminals[0] = 1;
        env->rewards[0] -= 1.0;
        add_log(env);
        c_reset(env);
        return;
    }

}

/*Rendering stuff*/

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
    int a = get_entity(env, AGENT, x, y);
    int w = get_entity(env, WALLS, x, y);
    int b = get_entity(env, BOXES, x, y);
    int t = get_entity(env, TARGET, x, y);

    if (w) return c->wall;
    if (b && t) return c->box_on_target;
    if (b) return c->box;
    if (a) return c->agent;
    if (t) return c->target;

    return c->floor;
}

void draw_tile(Boxoban *env, int x, int y) {
      Client *c = env->client;
      Rectangle dest = {x * TILE, y * TILE, TILE, TILE};

      // Always lay down the base tile
      DrawTexturePro(
          c->floor,
          (Rectangle){0, 0, (float)c->floor.width, (float)c->floor.height},
          dest,
          (Vector2){0, 0},
          0.0f,
          WHITE);

      if (get_entity(env, TARGET, x, y)) {
          DrawTexturePro(
              c->target,
              (Rectangle){0, 0, (float)c->target.width, (float)c->target.height},
              dest,
              (Vector2){0, 0},
              0.0f,
              WHITE);
      }
      if (get_entity(env, BOXES, x, y)) {
          Texture2D tex = get_entity(env, TARGET, x, y) ? c->box_on_target : c->box;
          DrawTexturePro(
              tex,
              (Rectangle){0, 0, (float)tex.width, (float)tex.height},
              dest,
              (Vector2){0, 0},
              0.0f,
              WHITE);
      }
      if (get_entity(env, WALLS, x, y)) {
          DrawTexturePro(
              c->wall,
              (Rectangle){0, 0, (float)c->wall.width, (float)c->wall.height},
              dest,
              (Vector2){0, 0},
              0.0f,
              WHITE);
      }
      if (get_entity(env, AGENT, x, y)) {
          Rectangle src = {0, 0, c->agent.width / 2.0f, (float)c->agent.height};
          DrawTexturePro(c->agent, src, dest, (Vector2){0, 0}, 0.0f, WHITE);
      }
  }


// Required function. Should handle creating the client on first call
void c_render(Boxoban* env) {
    if (!IsWindowReady()) {
        InitWindow(TILE*env->size, TILE*env->size, "PufferLib Boxoban");
        SetTargetFPS(10);
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
            draw_tile(env, x, y);
        }
    }


    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Boxoban* env) {
    if (env->intermediate_rewards) {
          free(env->intermediate_rewards);
          env->intermediate_rewards = NULL;
      }
    if (IsWindowReady()) {
        if (env->client) {
            UnloadTexture(env->client->wall);
            UnloadTexture(env->client->box);
            UnloadTexture(env->client->target);
            UnloadTexture(env->client->floor);
            UnloadTexture(env->client->agent);
            free(env->client);
            env->client = NULL;
        }
        CloseWindow();
    }
}
