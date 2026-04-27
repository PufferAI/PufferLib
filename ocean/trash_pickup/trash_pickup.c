#include <time.h>
#include "trash_pickup.h"
#include "puffernet.h"

void demo() {
    CTrashPickupEnv env = {
        .grid_size = 20,
        .num_agents = 8,
        .num_trash = 40,
        .num_bins = 2,
        .max_steps = 300,
        .agent_sight_range = 5,
        .do_human_control = true
    };

    Weights* weights = load_weights("resources/trash_pickup/trash_pickup_weights.bin");
    int logit_sizes[1] = {4};
    PufferNet* net = make_puffernet(weights, env.num_agents, 605, 128, 2, logit_sizes, 1);

    allocate(&env);
    c_reset(&env);
    c_render(&env);

    int tick = 0;
    while (!WindowShouldClose()) {
        if (tick % 2 == 0) {
            for (int e = 0; e < env.num_agents * 605; e++) {
                net->obs[e] = env.observations[e];
            }
            forward_puffernet(net, net->obs, env.actions);

            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) env.actions[0] = ACTION_UP;
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[0] = ACTION_LEFT;
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = ACTION_RIGHT;
                if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) env.actions[0] = ACTION_DOWN;
            }

            c_step(&env);
        }
        tick++;
        c_render(&env);
    }

    free_puffernet(net);
    free(weights);
    free_allocated(&env);
}

int main() {
    demo();
    return 0;
}
