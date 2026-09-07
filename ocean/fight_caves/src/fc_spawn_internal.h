#ifndef FC_SPAWN_INTERNAL_H
#define FC_SPAWN_INTERNAL_H

#include "fc_types.h"

int fc_spawn_find_available_footprint(const FcState* state,
                                      int preferred_x, int preferred_y,
                                      int size, int max_radius,
                                      int* out_x, int* out_y);

int fc_spawn_npc_first_free(FcState* state, int npc_type, int x, int y);

#endif /* FC_SPAWN_INTERNAL_H */
