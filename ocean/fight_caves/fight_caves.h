/* Native PufferLib 5.0 interface. Game, adapter and viewer implementations
 * are compiled as C in binding.c; the native trainer includes declarations only. */
#ifndef FIGHT_CAVES_ENV_H
#define FIGHT_CAVES_ENV_H

typedef float obs_t;

#ifdef __cplusplus
extern "C" {
#endif
#include "pufferenv.h"
#include "core.h"

#define OBS_SIZE FC_PUFFER_OBS_SIZE
#define NUM_ATNS FC_PUFFER_NUM_ATNS
#define ACT_SIZES FC_PUFFER_ACT_SIZES

struct Log {
    float zero_progress_ticks;
    float wave_reached;
    float wrong_prayer_hits;
    float reached_wave_63;
    float jad_kill_rate;
    float prayer_uptime_range;
    float prayer_uptime_melee;
    float prayer_uptime_magic;
    float npc_healing_total;
    float jad_healing_total;
    float episode_length;
    float n;  /* PufferLib episode count; must be last. */
};

typedef struct ViewerState ViewerState;

struct Env {
    Log log;                    /* required by PufferLib */
    Agent agents[1];            /* buffers are wired by Puffer after puf_init */
    int num_agents;             /* always 1 for Fight Caves */
    unsigned int rng;           /* per-env seed supplied by Puffer */
    int tag;                    /* Puffer policy grouping */
    int boundary_reached;       /* Puffer episode-boundary bookkeeping */

    /* Game state */
    FcState state;
    ViewerState* viewer;         /* NULL throughout headless training */

    /* Reward weights and shaping configuration, initialized once per env. */
    FcRewardParams reward_params;
    int initial_sharks;
    int initial_prayer_doses;
    FcRewardRuntime reward_runtime;

    /* RNG seed counter (increments each episode for variety) */
    uint32_t seed_counter;
};
typedef Env FightCaves;

/* Manual play: implemented in viewer.c, run by fight_caves.c's main. */
int fc_viewer_main(int argc, char** argv);

#ifdef __cplusplus
}
#endif
#endif /* FIGHT_CAVES_ENV_H */
