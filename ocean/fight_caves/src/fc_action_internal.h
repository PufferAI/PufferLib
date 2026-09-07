#ifndef FC_ACTION_INTERNAL_H
#define FC_ACTION_INTERNAL_H

#include "fc_types.h"

/* An already-active run may consume its remaining energy below 1%. Starting
 * run mode follows the OSRS client/server minimum of one displayed percent. */
static inline int fc_player_can_run(const FcPlayer* player) {
    return player->run_energy > 0 &&
           (player->is_running ||
            player->run_energy >= FC_RUN_ENERGY_MIN_START);
}

int fc_eat_action_valid(const FcState* state, int action);
int fc_drink_action_valid(const FcState* state, int action);

#endif
