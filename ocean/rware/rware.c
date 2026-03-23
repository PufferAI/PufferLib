#include <time.h>
#include <unistd.h>
#include "rware.h"
#include "puffernet.h"
#define MAP_TINY_WIDTH 640
#define MAP_TINY_HEIGHT 704
#define MAP_SMALL_WIDTH 1280
#define MAP_SMALL_HEIGHT 640
#define MAP_MEDIUM_WIDTH 1280
#define MAP_MEDIUM_HEIGHT 640

void demo(int map_choice) {
    int width;
    int height;
    if (map_choice == 1) {
        width = MAP_TINY_WIDTH;
        height = MAP_TINY_HEIGHT;
    } else if (map_choice == 2) {
        width = MAP_SMALL_WIDTH;
        height = MAP_SMALL_HEIGHT;
    } else {
        width = MAP_MEDIUM_WIDTH;
        height = MAP_MEDIUM_HEIGHT;
    }
    CRware env = {
        .width = width,
        .height = height,
        .map_choice = map_choice,
        .num_agents = 8,
        .num_requested_shelves = 8,
        .grid_square_size = 64,
        .human_agent_idx = 0,
	.reward_type = 2
    };
    Weights* weights = load_weights("resources/rware/rware_weights.bin", 136454);
    int logit_sizes[1] = {5};
    LinearLSTM* net = make_linearlstm(weights, env.num_agents, 27, logit_sizes, 1);

    allocate(&env);
    c_reset(&env);
    c_render(&env);

    int tick = 1;
    while (!WindowShouldClose()) {
        if (tick % 12 == 0) {
            tick = 0;

            int human_action = env.actions[env.human_agent_idx];
            for (int i = 0; i < env.num_agents * 27; i++) {
                net->obs[i] = (float)env.observations[i];
            }
            forward_linearlstm(net, net->obs, env.actions);

            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                env.actions[env.human_agent_idx] = human_action;
            }

            c_step(&env);

            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                env.actions[env.human_agent_idx] = NOOP;
            }
        }
        tick++;

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Handle keyboard input only for selected agent
            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
                env.actions[env.human_agent_idx] = FORWARD;
            }
            if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
                env.actions[env.human_agent_idx] = LEFT;
            }
            if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
                env.actions[env.human_agent_idx] = RIGHT;
            }
            if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER)) {
                env.actions[env.human_agent_idx] = TOGGLE_LOAD;
            }
            // Add agent switching with TAB key
            if (IsKeyPressed(KEY_TAB)) {
                env.human_agent_idx = (env.human_agent_idx + 1) % env.num_agents;
            }
        }

        c_render(&env);
    }
    //close_client(client);
    free_allocated(&env);
}

void performance_test() {
    long test_time = 10;
    CRware env = {
        .width = 1280,
        .height = 704,
        .map_choice = 2,
        .num_agents = 4,
        .num_requested_shelves = 4,
	.reward_type = 2
    };
    allocate(&env);
    c_reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % 5;
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(&env);
}

int main() {
    demo(2);
    //performance_test();
    return 0;
}
