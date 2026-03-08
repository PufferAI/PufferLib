#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include "raylib.h"
#include "generate_maps.h"
#include "parse_maps.h"

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
extern size_t PUZZLE_OBS_BYTES;

int boxoban_prepare_maps_for_difficulty(const char* difficulty, char* out_path, size_t out_cap);
int boxoban_set_map_path(const char *path);
int boxoban_difficulty_id_from_name(const char* difficulty_name);
const char* boxoban_difficulty_name_from_id(int difficulty_id);

#ifdef BOXOBAN_MAPS_IMPLEMENTATION
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

uint8_t *MAP_BASE = NULL;
size_t MAP_FILESIZE = 0;
size_t PUZZLE_COUNT = 0;
size_t PUZZLE_SIZE = BOXOBAN_PUZZLE_BYTES;
size_t PUZZLE_OBS_BYTES = BOXOBAN_PUZZLE_OBS_BYTES;
static char* BOXOBAN_MAP_PATH = NULL;
static const char* BOXOBAN_LEVEL_ROOT = "pufferlib/ocean/boxoban/boxoban-levels";

typedef struct {
    const char* difficulty;
    const char* rel_paths[2];
    size_t rel_path_count;
} BoxobanDifficultySpec;

static const BoxobanDifficultySpec BOXOBAN_DIFFICULTIES[] = {
    {"basic", {"basic/train", NULL}, 1},
    {"easy", {"easy/train", NULL}, 1},
    {"medium", {"medium/train", NULL}, 1},
    {"hard", {"hard", NULL}, 1},
    {"unfiltered", {"unfiltered/train", NULL}, 1},
};

typedef struct {
    char** items;
    size_t count;
    size_t cap;
} BoxobanPathList;

static int boxoban_cmp_strings(const void* a, const void* b) {
    const char* const* sa = (const char* const*)a;
    const char* const* sb = (const char* const*)b;
    return strcmp(*sa, *sb);
}

static void boxoban_path_list_free(BoxobanPathList* list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->items[i]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->cap = 0;
}

static int boxoban_path_list_append(BoxobanPathList* list, const char* path) {
    if (list->count == list->cap) {
        size_t next_cap = list->cap == 0 ? 64 : list->cap * 2;
        char** next = (char**)realloc(list->items, next_cap * sizeof(char*));
        if (next == NULL) {
            return -1;
        }
        list->items = next;
        list->cap = next_cap;
    }
    char* copied = (char*)malloc(strlen(path) + 1);
    if (copied == NULL) {
        return -1;
    }
    strcpy(copied, path);
    list->items[list->count++] = copied;
    return 0;
}

static int boxoban_has_txt_suffix(const char* name) {
    size_t len = strlen(name);
    return len >= 4 && strcmp(name + len - 4, ".txt") == 0;
}

static const BoxobanDifficultySpec* boxoban_get_difficulty_spec(const char* difficulty) {
    for (size_t i = 0; i < sizeof(BOXOBAN_DIFFICULTIES) / sizeof(BOXOBAN_DIFFICULTIES[0]); i++) {
        if (strcmp(difficulty, BOXOBAN_DIFFICULTIES[i].difficulty) == 0) {
            return &BOXOBAN_DIFFICULTIES[i];
        }
    }
    return NULL;
}

int boxoban_difficulty_id_from_name(const char* difficulty_name) {
    if (difficulty_name == NULL) {
        return -1;
    }
    for (size_t i = 0; i < sizeof(BOXOBAN_DIFFICULTIES) / sizeof(BOXOBAN_DIFFICULTIES[0]); i++) {
        if (strcmp(difficulty_name, BOXOBAN_DIFFICULTIES[i].difficulty) == 0) {
            return (int)i;
        }
    }
    return -1;
}

const char* boxoban_difficulty_name_from_id(int difficulty_id) {
    if (difficulty_id < 0 || difficulty_id >= (int)(sizeof(BOXOBAN_DIFFICULTIES) / sizeof(BOXOBAN_DIFFICULTIES[0]))) {
        return NULL;
    }
    return BOXOBAN_DIFFICULTIES[difficulty_id].difficulty;
}

static int boxoban_dir_has_txt(const char* dir_path) {
    DIR* dir = opendir(dir_path);
    if (dir == NULL) {
        return 0;
    }
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (boxoban_has_txt_suffix(ent->d_name)) {
            closedir(dir);
            return 1;
        }
    }
    closedir(dir);
    return 0;
}

static int boxoban_collect_sorted_txt_paths_in_dir(const char* dir_path, BoxobanPathList* out_paths) {
    DIR* dir = opendir(dir_path);
    if (dir == NULL) {
        fprintf(stderr, "Missing level directory %s\n", dir_path);
        return -1;
    }

    BoxobanPathList names = {0};
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (!boxoban_has_txt_suffix(ent->d_name)) {
            continue;
        }
        if (boxoban_path_list_append(&names, ent->d_name) != 0) {
            boxoban_path_list_free(&names);
            closedir(dir);
            return -1;
        }
    }
    closedir(dir);

    qsort(names.items, names.count, sizeof(char*), boxoban_cmp_strings);
    for (size_t i = 0; i < names.count; i++) {
        char full_path[1400];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, names.items[i]);
        if (boxoban_path_list_append(out_paths, full_path) != 0) {
            boxoban_path_list_free(&names);
            return -1;
        }
    }
    boxoban_path_list_free(&names);
    return 0;
}

static int boxoban_collect_maps(const char* difficulty, BoxobanPathList* out_paths) {
    const BoxobanDifficultySpec* spec = boxoban_get_difficulty_spec(difficulty);
    if (spec == NULL) {
        fprintf(stderr, "Invalid difficulty '%s'\n", difficulty);
        return -1;
    }

    for (size_t i = 0; i < spec->rel_path_count; i++) {
        char level_dir[1400];
        struct stat st;
        snprintf(level_dir, sizeof(level_dir), "%s/%s", BOXOBAN_LEVEL_ROOT, spec->rel_paths[i]);
        if (stat(level_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
            fprintf(stderr, "Missing level directory %s\n", level_dir);
            return -1;
        }
        if (boxoban_collect_sorted_txt_paths_in_dir(level_dir, out_paths) != 0) {
            return -1;
        }
    }

    if (out_paths->count == 0) {
        fprintf(stderr, "No map files found for difficulty '%s'\n", difficulty);
        return -1;
    }
    return 0;
}

static int boxoban_download_text_maps(const char* difficulty) {
    char zip_url[512];
    snprintf(zip_url, sizeof(zip_url),
        "https://raw.githubusercontent.com/TBBristol/pufferlib_boxoban_levels/main/%s.zip",
        difficulty);
    fprintf(stdout, "[Boxoban] Downloading %s maps from %s\n", difficulty, zip_url);

    char tmp_template[] = "/tmp/boxoban_maps_XXXXXX";
    char* tmp_dir = mkdtemp(tmp_template);
    if (tmp_dir == NULL) {
        return -1;
    }

    char zip_path[1400];
    snprintf(zip_path, sizeof(zip_path), "%s/%s.zip", tmp_dir, difficulty);

    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "curl -L --fail -o '%s' '%s' > /dev/null 2>&1", zip_path, zip_url);
    if (system(cmd) != 0) {
        fprintf(stderr, "Failed to download Boxoban maps with curl\n");
        return -1;
    }

    snprintf(cmd, sizeof(cmd), "unzip -q '%s' -d '%s'", zip_path, tmp_dir);
    if (system(cmd) != 0) {
        fprintf(stderr, "Failed to unzip Boxoban maps archive\n");
        return -1;
    }

    char extracted_root[1400] = {0};
    char find_cmd[4096];
    snprintf(find_cmd, sizeof(find_cmd), "find '%s' -type d -name '%s' | head -n 1", tmp_dir, difficulty);
    FILE* find_pipe = popen(find_cmd, "r");
    if (find_pipe == NULL) {
        return -1;
    }
    if (fgets(extracted_root, sizeof(extracted_root), find_pipe) == NULL) {
        pclose(find_pipe);
        fprintf(stderr, "Downloaded zip missing '%s' directory\n", difficulty);
        return -1;
    }
    pclose(find_pipe);
    extracted_root[strcspn(extracted_root, "\r\n")] = '\0';

    char dest_root[1400];
    snprintf(dest_root, sizeof(dest_root), "%s/%s", BOXOBAN_LEVEL_ROOT, difficulty);
    if (boxoban_mkdir_p(dest_root) != 0) {
        return -1;
    }

    snprintf(cmd, sizeof(cmd), "cp -R '%s/.' '%s/'", extracted_root, dest_root);
    if (system(cmd) != 0) {
        fprintf(stderr, "Failed to copy downloaded maps into %s\n", dest_root);
        return -1;
    }
    return 0;
}

static int boxoban_ensure_text_maps(const char* difficulty) {
    const BoxobanDifficultySpec* spec = boxoban_get_difficulty_spec(difficulty);
    if (spec == NULL) {
        return -1;
    }

    for (size_t i = 0; i < spec->rel_path_count; i++) {
        char level_dir[1400];
        snprintf(level_dir, sizeof(level_dir), "%s/%s", BOXOBAN_LEVEL_ROOT, spec->rel_paths[i]);
        if (boxoban_dir_has_txt(level_dir)) {
            return 0;
        }
    }

    char output_dir[1400];
    if (strcmp(difficulty, "basic") == 0) {
        snprintf(output_dir, sizeof(output_dir), "%s/basic/train", BOXOBAN_LEVEL_ROOT);
        fprintf(stdout, "[Boxoban] Generating basic maps at %s\n", output_dir);
        return boxoban_generate_basic_maps(output_dir, 0);
    }
    if (strcmp(difficulty, "easy") == 0) {
        snprintf(output_dir, sizeof(output_dir), "%s/easy/train", BOXOBAN_LEVEL_ROOT);
        fprintf(stdout, "[Boxoban] Generating easy maps at %s\n", output_dir);
        return boxoban_generate_easy_maps(output_dir, 0);
    }
    return boxoban_download_text_maps(difficulty);
}

static int boxoban_bin_path(const char* difficulty, char* out_path, size_t out_cap) {
    int written = snprintf(out_path, out_cap, "pufferlib/ocean/boxoban/boxoban_maps_%s.bin", difficulty);
    if (written <= 0 || (size_t)written >= out_cap) {
        return -1;
    }
    return 0;
}

int boxoban_prepare_maps_for_difficulty(const char* difficulty, char* out_path, size_t out_cap) {
    if (difficulty == NULL || out_path == NULL) {
        return -1;
    }
    if (boxoban_get_difficulty_spec(difficulty) == NULL) {
        return -1;
    }
    if (boxoban_bin_path(difficulty, out_path, out_cap) != 0) {
        return -1;
    }

    if (access(out_path, F_OK) != 0) {
        if (boxoban_ensure_text_maps(difficulty) != 0) {
            return -1;
        }

        BoxobanPathList maps = {0};
        size_t puzzle_count = 0;
        if (boxoban_collect_maps(difficulty, &maps) != 0) {
            boxoban_path_list_free(&maps);
            return -1;
        }

        if (boxoban_write_bin_from_files((const char* const*)maps.items, maps.count, out_path, 0, &puzzle_count) != 0) {
            boxoban_path_list_free(&maps);
            return -1;
        }
        boxoban_path_list_free(&maps);
        fprintf(stdout, "[Boxoban] Generated %zu puzzles for '%s' at %s\n", puzzle_count, difficulty, out_path);
    }

    if (boxoban_set_map_path(out_path) != 0) {
        return -1;
    }
    return 0;
}

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
    return NULL;
}

void ensure_map_loaded(void) {
    if (MAP_BASE != NULL)
        return;

    if (BOXOBAN_MAP_PATH == NULL) {
        const char* default_path = get_default_map_path();
        if (default_path != NULL) {
            if (boxoban_set_map_path(default_path) != 0) {
                fprintf(stderr, "Failed to set default Boxoban map path\n");
                abort();
            }
        } else {
            char prepared_path[512];
            if (boxoban_prepare_maps_for_difficulty("basic", prepared_path, sizeof(prepared_path)) != 0) {
                fprintf(stderr, "Failed to prepare default Boxoban maps\n");
                abort();
            }
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
    if (MAP_FILESIZE % PUZZLE_SIZE != 0) {
        fprintf(stderr, "Invalid Boxoban map file size %zu (expected multiple of %zu)\n",
            MAP_FILESIZE, PUZZLE_SIZE);
        abort();
    }
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
    float on_targets; // Number of targets currently boxed
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
    int on_target; //num targets currently boxed
    int n_boxes; //boxes in map
    int n_targets; //targets in map
    int difficulty_id; // 0=basic,1=easy,2=medium,3=hard,4=unfiltered
    Client* client;
    int win;
} Boxoban;

void ensure_map_loaded(void);

static int boxoban_configure_maps_from_env(Boxoban* env) {
    if (env->difficulty_id >= 0) {
        const char* difficulty_name = boxoban_difficulty_name_from_id(env->difficulty_id);
        if (difficulty_name == NULL) {
            fprintf(stderr, "Invalid Boxoban difficulty id %d\n", env->difficulty_id);
            return -1;
        }
        char prepared_path[512];
        if (boxoban_prepare_maps_for_difficulty(difficulty_name, prepared_path, sizeof(prepared_path)) != 0) {
            return -1;
        }
    }

    return 0;
}

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


void init (Boxoban* env) {
    if (boxoban_configure_maps_from_env(env) != 0) {
        fprintf(stderr, "Failed to configure Boxoban maps\n");
        abort();
    }
    ensure_map_loaded();
    env->intermediate_rewards = calloc(env->size*env->size, sizeof(unsigned char));
    env->win = 0;
  }


void add_log(Boxoban* env) {
    float denom = (float)env->n_boxes;
    float num = (float)env->on_target;
    env->log.perf += (env->win== 1) ? 1.0 : num/denom;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.on_targets += env->on_target;
    env->log.n++;
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
    const uint8_t* puzzle = MAP_BASE + (size_t)i * PUZZLE_SIZE;
    memcpy(env->observations, puzzle, PUZZLE_OBS_BYTES);

    const uint8_t* meta = puzzle + PUZZLE_OBS_BYTES;
    env->agent_x = (int)meta[0];
    env->agent_y = (int)meta[1];
    env->n_boxes = (int)meta[2];
    env->n_targets = (int)meta[3];
    env->on_target = (int)meta[4];

    memcpy(env->intermediate_rewards,
            env->observations + TARGET * env->size * env->size,env->size * env->size);

    env->tick = 0;
    env->win = 0;

}

//Updates OBS for moved entity
void move_entity(Boxoban* env,unsigned char entity,int x, int y, int dx, int dy) {
    set_entity(env, entity, x, y, 0);
    set_entity(env, entity, x + dx, y + dy, 1);
}

//Updates state and intermediate reward array in place
int take_action(Boxoban* env, int action) {

    int dx = 0;
    int dy = 0;
    int int_r = 0;

    if (action == NOOP) {
        return 0;
    }
    else if (action == DOWN) {
        dy = 1;
    }
    else if (action == UP) {
        dy = -1;
    }
    else if (action == LEFT) {
        dx = -1;
    }
    else if (action == RIGHT) {
        dx = 1;
    }

    //if move space is clear, move agent
    if (clear(env, env->agent_x + dx, env->agent_y + dy)) {
        
        move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
        env->agent_y += dy;
        env->agent_x += dx;
        return 0;
    }
    //if its not clear, but its a box and box is clear to move, move both
    else if (clear(env, env->agent_x+ 2*dx, env->agent_y + 2*dy)
            && get_entity(env, BOXES, env->agent_x + dx, env->agent_y + dy) == 1) {

            //if box is on target currently, remove from on_target count
            if (get_entity(env, TARGET, env->agent_x + dx, env->agent_y + dy) == 1) {

                env->on_target -= 1;
            }
            //move both entities
            move_entity(env, BOXES, env->agent_x + dx, env->agent_y + dy, dx, dy);
            move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
            env->agent_y += dy;
            env->agent_x += dx;
        
            //if box is now on target, add to on_target count
            //if its a new target recieve intermediate reward and zero out intermediate reward
            if (get_entity(env, TARGET, env->agent_x + dx, env->agent_y + dy) == 1) {
                
                env->on_target += 1;
                int_r = get_intermediate_reward_status(env, env->agent_x + dx, env->agent_y + dy);
                set_intermediate_reward(env, env->agent_x + dx, env->agent_y + dy, 0);
            }
            return int_r;
    }
    return 0;
}

// Required function
void c_step(Boxoban* env) {
    env->tick += 1;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
       
    int action = env->actions[0];

    float on_target = env->on_target;
    int int_r = take_action(env, action); //int_r _new_ tgts covered, modifies observations in place
    float on_target_after = env->on_target;
                                          
    env->rewards[0] += (float)int_r * env->int_r_coeff;
 
    if (on_target_after < on_target) {
        env->rewards[0] -= env->target_loss_pen_coeff;
    }

    //Terminals
    if (env->on_target == env->n_targets) {
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
