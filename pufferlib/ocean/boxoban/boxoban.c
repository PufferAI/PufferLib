/* Pure C demo file for Boxoban. Usage:
 *   bash scripts/build_ocean.sh boxoban
 *   ./boxoban [difficulty|path_to_bin]
 *
 * If you pass one of the known difficulty names (basic, easy, medium,
 * hard, unfiltered) the demo looks for pufferlib/ocean/boxoban/boxoban_maps_<difficulty>.bin
 * Otherwise the argument is treated as an explicit path to a bin file.
 */

#define BOXOBAN_MAPS_IMPLEMENTATION
#include "boxoban.h"

static const char* resolve_map_path(int argc, char** argv, char* buffer, size_t buf_sz) {
    const char* arg = argc > 1 ? argv[1] : NULL;
    if (arg == NULL) {
        return "pufferlib/ocean/boxoban/boxoban_maps_basic.bin";
    }
    if (strchr(arg, '/')) {
        return arg;
    }
    snprintf(buffer, buf_sz, "pufferlib/ocean/boxoban/boxoban_maps_%s.bin", arg);
    return buffer;
}

int main(int argc, char** argv) {
    char path_buffer[512];
    const char* chosen_path = resolve_map_path(argc, argv, path_buffer, sizeof(path_buffer));
    if (boxoban_set_map_path(chosen_path) != 0) {
        fprintf(stderr, "Failed to set map path: %s\n", chosen_path);
        return 1;
    }

    Boxoban env = {.size = 10};
    env.observations = (unsigned char*)calloc(4*env.size*env.size, sizeof(unsigned char));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.max_steps = 500;
    env.int_r_coeff = 0.1f;
    env.target_loss_pen_coeff = 0.5f;
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
}
