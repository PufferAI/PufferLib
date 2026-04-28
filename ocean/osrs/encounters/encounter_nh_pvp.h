/**
 * @file encounter_nh_pvp.h
 * @brief NH (No Honor) PvP encounter — the original 1v1 LMS-style fight.
 *
 * Wraps the existing osrs_pvp_api.h (pvp_init/pvp_reset/pvp_step) as an
 * EncounterDef implementation. This is the first encounter and serves as
 * the reference for how to add new encounters.
 *
 * Entity layout: 2 players (agent + opponent).
 * Obs: 334 features. Actions: 7 heads [9,13,6,2,5,2,2]. Mask: 39.
 */

#ifndef ENCOUNTER_NH_PVP_H
#define ENCOUNTER_NH_PVP_H

#include "../osrs_encounter.h"
#include "../osrs_env.h"

/* obs/action dimensions from osrs_types.h */
static const int NH_PVP_ACTION_DIMS[] = {
    LOADOUT_DIM, COMBAT_DIM, OVERHEAD_DIM,
    FOOD_DIM, POTION_DIM, KARAMBWAN_DIM, VENG_DIM
};


typedef struct {
    OsrsEnv env;
} NhPvpState;


static EncounterState* nh_pvp_create(void) {
    NhPvpState* s = (NhPvpState*)calloc(1, sizeof(NhPvpState));
    pvp_init(&s->env);
    /* pvp_init sets internal buf pointers for game logic (observations, actions, etc.).
       also wire the ocean pointers to internal buffers so pvp_step can write obs/rewards
       without needing the PufferLib binding. */
    s->env.ocean_io.agent_obs = s->env._obs_buf;
    s->env.ocean_io.agent_actions = s->env._acts_buf;
    s->env.ocean_io.agent_rewards = s->env._rews_buf;
    s->env.ocean_io.agent_terminals = s->env._terms_buf;
    return (EncounterState*)s;
}

static void nh_pvp_destroy(EncounterState* state) {
    NhPvpState* s = (NhPvpState*)state;
    pvp_close(&s->env);
    free(s);
}

static void nh_pvp_reset(EncounterState* state, uint32_t seed) {
    NhPvpState* s = (NhPvpState*)state;
    if (seed != 0) {
        s->env.has_rng_seed = 1;
        s->env.rng_seed = seed;
    }
    pvp_reset(&s->env);
}

static void nh_pvp_step(EncounterState* state, const int* actions) {
    NhPvpState* s = (NhPvpState*)state;
    /* pvp_step reads agent 0 actions from ocean_io.agent_actions. */
    memcpy(s->env.ocean_io.agent_actions, actions, NUM_ACTION_HEADS * sizeof(int));
    pvp_step(&s->env);
}


static void nh_pvp_write_obs(EncounterState* state, float* obs_out) {
    NhPvpState* s = (NhPvpState*)state;
    /* observations are already computed by pvp_step into _obs_buf.
       copy agent 0's observations (SLOT_NUM_OBSERVATIONS floats). */
    memcpy(obs_out, s->env._obs_buf, SLOT_NUM_OBSERVATIONS * sizeof(float));
}

static void nh_pvp_write_mask(EncounterState* state, float* mask_out) {
    NhPvpState* s = (NhPvpState*)state;
    /* masks are in _masks_buf, ACTION_MASK_SIZE bytes for agent 0.
       convert to float for the encounter interface. */
    for (int i = 0; i < ACTION_MASK_SIZE; i++) {
        mask_out[i] = (float)s->env._masks_buf[i];
    }
}

static float nh_pvp_get_reward(EncounterState* state) {
    NhPvpState* s = (NhPvpState*)state;
    return s->env._rews_buf[0];
}

static int nh_pvp_is_terminal(EncounterState* state) {
    NhPvpState* s = (NhPvpState*)state;
    return s->env.episode_over;
}


static int nh_pvp_get_entity_count(EncounterState* state) {
    (void)state;
    return NUM_AGENTS;  /* always 2 for NH PvP */
}

static void* nh_pvp_get_entity(EncounterState* state, int index) {
    NhPvpState* s = (NhPvpState*)state;
    return &s->env.players[index];
}


static void nh_pvp_fill_render_entities(EncounterState* state, RenderEntity* out, int max_entities, int* count) {
    NhPvpState* s = (NhPvpState*)state;
    int n = NUM_AGENTS < max_entities ? NUM_AGENTS : max_entities;
    for (int i = 0; i < n; i++) {
        render_entity_from_player(&s->env.players[i], &out[i]);
    }
    *count = n;
}


static void nh_pvp_put_int(EncounterState* state, const char* key, int value) {
    NhPvpState* s = (NhPvpState*)state;
    if (strcmp(key, "opponent_type") == 0) {
        s->env.pvp_runtime.opponent.type = (OpponentType)value;
    } else if (strcmp(key, "is_lms") == 0) {
        s->env.is_lms = value;
    } else if (strcmp(key, "use_c_opponent") == 0) {
        s->env.pvp_runtime.use_c_opponent = value;
    } else if (strcmp(key, "auto_reset") == 0) {
        s->env.auto_reset = value;
    } else if (strcmp(key, "seed") == 0) {
        s->env.has_rng_seed = 1;
        s->env.rng_seed = (uint32_t)value;
    }
}

static void nh_pvp_put_float(EncounterState* state, const char* key, float value) {
    NhPvpState* s = (NhPvpState*)state;
    if (strcmp(key, "shaping_scale") == 0) {
        s->env.shaping.shaping_scale = value;
    }
}

static void nh_pvp_put_ptr(EncounterState* state, const char* key, void* value) {
    NhPvpState* s = (NhPvpState*)state;
    if (strcmp(key, "collision_map") == 0) {
        s->env.collision_map = value;
    }
}


static void* nh_pvp_get_log(EncounterState* state) {
    NhPvpState* s = (NhPvpState*)state;
    return &s->env.log;
}

static int nh_pvp_get_tick(EncounterState* state) {
    NhPvpState* s = (NhPvpState*)state;
    return s->env.tick;
}

static int nh_pvp_get_winner(EncounterState* state) {
    NhPvpState* s = (NhPvpState*)state;
    return s->env.winner;
}


static const EncounterDef ENCOUNTER_NH_PVP = {
    .name = "nh_pvp",
    .obs_size = SLOT_NUM_OBSERVATIONS,
    .num_action_heads = NUM_ACTION_HEADS,
    .action_head_dims = NH_PVP_ACTION_DIMS,
    .mask_size = ACTION_MASK_SIZE,

    .create = nh_pvp_create,
    .destroy = nh_pvp_destroy,
    .reset = nh_pvp_reset,
    .step = nh_pvp_step,

    .write_obs = nh_pvp_write_obs,
    .write_mask = nh_pvp_write_mask,
    .get_reward = nh_pvp_get_reward,
    .is_terminal = nh_pvp_is_terminal,

    .get_entity_count = nh_pvp_get_entity_count,
    .get_entity = nh_pvp_get_entity,
    .fill_render_entities = nh_pvp_fill_render_entities,

    .put_int = nh_pvp_put_int,
    .put_float = nh_pvp_put_float,
    .put_ptr = nh_pvp_put_ptr,

    .render_post_tick = NULL,
    .get_log = nh_pvp_get_log,
    .get_tick = nh_pvp_get_tick,
    .get_winner = nh_pvp_get_winner,

    .translate_human_input = NULL,
    .head_move = -1,
    .head_prayer = -1,
    .head_target = -1,
};

/* auto-register on include */
__attribute__((constructor))
static void nh_pvp_register(void) {
    encounter_register(&ENCOUNTER_NH_PVP);
}

#endif /* ENCOUNTER_NH_PVP_H */
