#include "moba.h"
#include "puffernet.h"

void demo() {
    // encoder(64x510=32640) + decoder(24x64=1536) + 5x mingru(192x64=12288) = 95616
    Weights* weights = load_weights("resources/moba/moba_weights.bin");

    int logit_sizes[6] = {7, 7, 3, 2, 2, 2};
    PufferNet* net = make_puffernet(weights, 5, 510, 64, 5, logit_sizes, 6);

    MOBA env = {
        .vision_range = 5,
        .agent_speed = 1.0,
        .reward_death = -0.163764,
        .reward_xp = 0.00665677,
        .reward_distance = 0,
        .reward_tower = 0.642119,
        .script_opponents = true,
    };
    allocate_moba(&env);
    c_reset(&env);

    float obs_f[5 * 510];
    c_render(&env);
    int frame = 1;
    while (!WindowShouldClose()) {
        if (frame % 12 == 0) {
            for (int i = 0; i < 5 * 510; i++)
                obs_f[i] = (float)env.observations[i];
            forward_puffernet(net, obs_f, env.actions);
            c_step(&env);
        }
        c_render(&env);
        frame = (frame + 1) % 12;
    }
    free_puffernet(net);
    free(weights);
    free_allocated_moba(&env);
}

int main() {
    demo();
}
