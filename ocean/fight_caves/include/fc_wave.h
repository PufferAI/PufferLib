#ifndef FC_WAVE_H
#define FC_WAVE_H

#include "fc_types.h"

/*
 * fc_wave.h — Wave system for Fight Caves.
 *
 * 63 waves, 15 spawn rotations per wave.
 * Wave table sourced from OSRS wiki + Kotlin archive TOML data.
 *
 * Spawn directions map to arena coordinates:
 *   SPAWN_SOUTH      → (32, 5)   bottom center
 *   SPAWN_SOUTH_WEST → (8, 8)    bottom left
 *   SPAWN_NORTH_WEST → (8, 55)   top left
 *   SPAWN_SOUTH_EAST → (55, 8)   bottom right
 *   SPAWN_CENTER     → (32, 32)  arena center
 */

/* Spawn direction → arena coordinate mapping */
void fc_spawn_position(int spawn_dir, int* x, int* y);

/* Spawn all NPCs for the given wave using the state's rotation_id */
void fc_wave_spawn(FcState* state, int wave_num);

/* Check if wave is cleared and advance to next wave.
 * Called from check_terminal in fc_tick.c.
 * Returns 1 if wave was advanced, 0 otherwise. */
int fc_wave_check_advance(FcState* state);

/* Jad healer spawn threshold: spawn 4 Yt-HurKot when Jad HP drops below this.
 * Respawns are re-armed only if the previous healers restored Jad to full HP. */
#define FC_JAD_HEALER_THRESHOLD_HP_TENTHS  1500  /* 150 HP */
#define FC_JAD_NUM_HEALERS            4

#endif /* FC_WAVE_H */
