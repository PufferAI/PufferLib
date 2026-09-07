/*
 * fight_caves.c — Standalone entry point for testing without PufferLib.
 *
 * Compiled with: ./build.sh --local (debug) or ./build.sh --fast (optimized)
 * Runs N episodes with random actions and prints stats.
 */

#include "fight_caves.h"
#include <stdio.h>
#include <time.h>

int main(void) {
    FightCaves env = {0};
    env.num_agents = 1;
    env.observations = (float*)calloc(FC_PUFFER_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(FC_PUFFER_NUM_ATNS, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    {
        env.reward_params = fc_reward_default_params();
        env.initial_sharks = 0;
        env.initial_prayer_doses = 0;

        /* Obs ablation flags default to 0 (no ablation) for the standalone harness. */
        env.obs_ablate_npc_distance = 0;
        env.obs_ablate_incoming_aggregates = 0;
        env.obs_ablate_npc_valid = 0;
    }

    fc_init(&env.state);

    srand((unsigned)time(NULL));
    int episodes = 100;
    int total_ticks = 0;
    float total_reward = 0;
    int max_wave = 0;

    printf("Running %d episodes with random actions...\n", episodes);
    clock_t start = clock();

    for (int ep = 0; ep < episodes; ep++) {
        c_reset(&env);
        env.terminals[0] = 0.0f;
        int ep_ticks = 0;
        while (!env.terminals[0] && ep_ticks < 30000) {
            if (env.state.current_wave > max_wave) {
                max_wave = env.state.current_wave;
            }
            for (int h = 0; h < FC_PUFFER_NUM_ATNS; h++)
                env.actions[h] = (float)(rand() % 17);
            env.actions[0] = (rand() % 3 == 0) ? (float)(rand() % 17) : 0.0f;
            env.actions[1] = (rand() % 5 == 0) ? (float)(rand() % 9) : 0.0f;
            env.actions[2] = (rand() % 10 == 0)
                ? (float)(rand() % FC_PRAYER_DIM) : 0.0f;
            c_step(&env);
            total_reward += env.rewards[0];
            ep_ticks++;
        }
        total_ticks += ep_ticks;
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Results:\n");
    printf("  Episodes:    %d\n", episodes);
    printf("  Total ticks: %d\n", total_ticks);
    printf("  SPS:         %.0f steps/sec\n", total_ticks / elapsed);
    printf("  Avg reward:  %.2f\n", total_reward / episodes);
    printf("  Max wave:    %d\n", max_wave);
    printf("  Time:        %.2fs\n", elapsed);
    printf("  Log: ep_len=%.1f wave=%.1f n=%.0f\n",
           env.log.episode_length, env.log.wave_reached, env.log.n);

    c_close(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
