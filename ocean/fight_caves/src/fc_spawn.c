#include "fc_spawn_internal.h"

#include "fc_npc.h"
#include "fc_pathfinding.h"

int fc_spawn_find_available_footprint(const FcState* state,
                                      int preferred_x, int preferred_y,
                                      int size, int max_radius,
                                      int* out_x, int* out_y) {
    uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    fc_build_occupancy(state, occupied, -1, 0);

    *out_x = preferred_x;
    *out_y = preferred_y;
    if (fc_footprint_available_dynamic(preferred_x, preferred_y, size,
                                       state->walkable, occupied)) {
        return 1;
    }

    for (int radius = 1; radius <= max_radius; radius++) {
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                if (dx != -radius && dx != radius &&
                    dy != -radius && dy != radius) {
                    continue;
                }
                int x = preferred_x + dx;
                int y = preferred_y + dy;
                if (fc_footprint_available_dynamic(x, y, size,
                                                   state->walkable, occupied)) {
                    *out_x = x;
                    *out_y = y;
                    return 1;
                }
            }
        }
    }

    return 0;
}

int fc_spawn_npc_first_free(FcState* state, int npc_type, int x, int y) {
    for (int slot = 0; slot < FC_MAX_NPCS; slot++) {
        if (state->npcs[slot].active) continue;
        fc_npc_spawn(&state->npcs[slot], npc_type, x, y,
                     state->next_spawn_index++);
        return slot;
    }
    return -1;
}
