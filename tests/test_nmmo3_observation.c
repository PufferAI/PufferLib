/*
 * Native regression test; pytest and the install-only CI job do not discover it.
 * Run after ./build.sh nmmo3 --fast has downloaded the platform Raylib archive.
 *
 * Linux:
 *   mkdir -p build
 *   ${CC:-clang} -std=gnu11 -O0 -g -DPLATFORM_DESKTOP \
 *     -I raylib-5.5_linux_amd64/include -I ocean/nmmo3 \
 *     tests/test_nmmo3_observation.c \
 *     raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread \
 *     -o build/test_nmmo3_observation
 *   ./build/test_nmmo3_observation
 *
 * macOS:
 *   mkdir -p build
 *   ${CC:-clang} -std=gnu11 -O0 -g -DPLATFORM_DESKTOP \
 *     -I raylib-5.5_macos/include -I ocean/nmmo3 \
 *     tests/test_nmmo3_observation.c raylib-5.5_macos/lib/libraylib.a \
 *     -framework Cocoa -framework IOKit -framework CoreVideo \
 *     -framework OpenGL -lm -lpthread -o build/test_nmmo3_observation
 *   ./build/test_nmmo3_observation
 *
 * Adding one of these commands to CI remains an upstream integration decision.
 */

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
static int rand_r(unsigned int* seed) {
    *seed = *seed * 1103515245U + 12345U;
    return (int)((*seed / 65536U) % 32768U);
}
#endif

#include "nmmo3.h"

/* Exercise the production allocation, data layout, and encoder directly. */
_Static_assert(11 * 15 * 10 + 47 + 10 == 1707, "unexpected observation layout");

static int entity_bytes_are_zero(const unsigned char* observation, int offset) {
    for (int index = 4; index <= 9; index++) {
        if (observation[offset + index] != 0) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    MMO env = {
        .width = 15,
        .height = 11,
        .num_agents = 1,
        .num_enemies = 1,
        .x_window = 7,
        .y_window = 5,
        .enemy_respawn_ticks = 1,
        .item_respawn_ticks = 1,
    };

    allocate_mmo(&env);
    const int map_size = env.width * env.height;
    for (int index = 0; index < map_size; index++) {
        env.pids[index] = -1;
    }

    Entity* observer = &env.players[0];
    observer->type = ENTITY_PLAYER;
    observer->r = env.y_window;
    observer->c = env.x_window;
    observer->comb_lvl = 2;

    Entity* enemy = &env.enemies[0];
    enemy->type = ENTITY_ENEMY;
    enemy->element = ELEM_EARTH;
    enemy->comb_lvl = 10;
    enemy->hp = 80;
    enemy->anim = ANIM_MOVE;
    enemy->dir = ATN_RIGHT;

    const int enemy_pid = env.num_agents;
    const int old_row = observer->r;
    const int old_col = observer->c + 1;
    const int old_map_offset = map_offset(&env, old_row, old_col);
    const int old_observation_offset = (
        (old_row - (observer->r - env.y_window)) * 15
        + old_col - (observer->c - env.x_window)
    ) * 10;

    enemy->r = old_row;
    enemy->c = old_col;
    env.pids[old_map_offset] = (short)enemy_pid;
    compute_all_obs(&env);
    if (env.observations[old_observation_offset + 4] != ENTITY_ENEMY) {
        fputs("setup failed: occupied cell was not encoded\n", stderr);
        return EXIT_FAILURE;
    }

    const int new_row = old_row + 1;
    const int new_col = old_col;
    const int new_map_offset = map_offset(&env, new_row, new_col);
    const int new_observation_offset = (
        (new_row - (observer->r - env.y_window)) * 15
        + new_col - (observer->c - env.x_window)
    ) * 10;

    env.pids[old_map_offset] = -1;
    env.pids[new_map_offset] = (short)enemy_pid;
    enemy->r = new_row;
    enemy->c = new_col;
    compute_all_obs(&env);

    if (env.observations[new_observation_offset + 4] != ENTITY_ENEMY) {
        fputs("moved entity was not encoded at its new cell\n", stderr);
        return EXIT_FAILURE;
    }
    if (!entity_bytes_are_zero(env.observations, old_observation_offset)) {
        fprintf(
            stderr,
            "empty cell retained entity bytes: [%u, %u, %u, %u, %u, %u]\n",
            env.observations[old_observation_offset + 4],
            env.observations[old_observation_offset + 5],
            env.observations[old_observation_offset + 6],
            env.observations[old_observation_offset + 7],
            env.observations[old_observation_offset + 8],
            env.observations[old_observation_offset + 9]
        );
        return EXIT_FAILURE;
    }

    free_allocated_mmo(&env);
    return EXIT_SUCCESS;
}
