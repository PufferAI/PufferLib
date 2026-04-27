/* Pure C demo file for Boxoban. Usage:
 *   bash scripts/build_ocean.sh boxoban
 *   ./boxoban [difficulty|path_to_bin]
 *
 * If you pass one of the known difficulty names (basic, easy, medium,
 * hard, unfiltered) the demo looks for pufferlib/ocean/boxoban/boxoban_maps_<difficulty>.bin
 * Otherwise the argument is treated as an explicit path to a bin file.
 */

#define BOXOBAN_MAPS_IMPLEMENTATION
#include <time.h>
#include "boxoban.h"

static int is_named_difficulty(const char* arg) {
    return strcmp(arg, "basic") == 0 ||
        strcmp(arg, "easy") == 0 ||
        strcmp(arg, "medium") == 0 ||
        strcmp(arg, "hard") == 0 ||
        strcmp(arg, "unfiltered") == 0;
}

static const char* resolve_map_path(int argc, char** argv, char* buffer, size_t buf_sz) {
    const char* arg = argc > 1 ? argv[1] : NULL;
    if (arg == NULL) {
        if (boxoban_prepare_maps_for_difficulty("easy", buffer, buf_sz) != 0) {
            return NULL;
        }
        return buffer;
    }
    if (strchr(arg, '/')) {
        return arg;
    }
    if (is_named_difficulty(arg)) {
        if (boxoban_prepare_maps_for_difficulty(arg, buffer, buf_sz) != 0) {
            return NULL;
        }
        return buffer;
    }
    snprintf(buffer, buf_sz, "pufferlib/ocean/boxoban/boxoban_maps_%s.bin", arg);
    return buffer;
}


int demo(int argc, char** argv) {
    char path_buffer[512];
    const char* chosen_path = resolve_map_path(argc, argv, path_buffer, sizeof(path_buffer));
    if (chosen_path == NULL) {
        fprintf(stderr, "Failed to prepare map path\n");
        return 1;
    }
    if (boxoban_set_map_path(chosen_path) != 0) {
        fprintf(stderr, "Failed to set map path: %s\n", chosen_path);
        return 1;
    }

    Boxoban env = {
        .size = 10,
        .observations = NULL,
        .actions = NULL,
        .rewards = NULL,
        .terminals = NULL,
        .max_steps = 500,
        .int_r_coeff = 0.1f,
        .target_loss_pen_coeff = 0.5f,
        .tick = 0,
        .agent_x = 0,
        .agent_y = 0,
        .intermediate_rewards = NULL,
        .on_target = 0,
        .n_boxes = 0,
        .win = 0,
        .difficulty_id = -1,
        .client = NULL,
        .n_targets = 0,

    };

    size_t obs_count = 4u * (size_t)env.size * (size_t)env.size;
    env.observations = calloc(obs_count, sizeof(unsigned char));
    env.actions = calloc(1, sizeof(int));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(unsigned char));

    init(&env);
    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_LEFT_SHIFT) || IsKeyPressed(KEY_RIGHT_SHIFT)) {
            TraceLog(LOG_INFO, "Shift key pressed");
        }
        bool manual = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
        bool stepped = false;
        if (manual) {
            int new_action = -1;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) new_action = UP;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) new_action = DOWN;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) new_action = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) new_action = RIGHT;

            if (new_action >= 0) {
                env.actions[0] = new_action;
                c_step(&env);
                stepped = true;
            }
        } else {
            env.actions[0] = rand() % 5;
            c_step(&env);
            stepped = true;
        }

        if (!stepped) {
            // Manual mode with no direction: stay paused
        }
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}

void test_performance(int argc, char** argv, int timeout) {
    char path_buffer[512];
    const char* chosen_path = resolve_map_path(argc, argv, path_buffer, sizeof(path_buffer));
    if (chosen_path == NULL) {
        fprintf(stderr, "Failed to prepare map path\n");
        return;
    }
    if (boxoban_set_map_path(chosen_path) != 0) {
        fprintf(stderr, "Failed to set map path: %s\n", chosen_path);
        return;
    }
    printf("Loaded map: %s\n", chosen_path);

    Boxoban env = {
        .size = 10,
        .observations = NULL,
        .actions = NULL,
        .rewards = NULL,
        .terminals = NULL,
        .max_steps = 500,
        .int_r_coeff = 0.1f,
        .target_loss_pen_coeff = 0.5f,
        .tick = 0,
        .agent_x = 0,
        .agent_y = 0,
        .intermediate_rewards = NULL,
        .on_target = 0,
        .n_boxes = 0,
        .win = 0,
        .difficulty_id = -1,
        .client = NULL,
        .n_targets = 0,
    };

    size_t obs_count = 4u * (size_t)env.size * (size_t)env.size;
    env.observations = calloc(obs_count, sizeof(unsigned char));
    env.actions = calloc(1, sizeof(int));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(unsigned char));

    printf("Initializing...\n");
    init(&env);
    printf("Resetting...\n");
    c_reset(&env);
    printf("Starting test...\n");

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        env.actions[0] = rand() % 5;
        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

int main(int argc, char** argv) {
    demo(argc, argv);
    setbuf(stdout, NULL);
    fprintf(stderr, "Entered main\n");
    fflush(stderr);
    //test_performance(argc, argv,10);
    return 0;
}
