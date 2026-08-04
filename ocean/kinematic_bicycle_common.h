#ifndef KINEMATIC_BICYCLE_COMMON_H
#define KINEMATIC_BICYCLE_COMMON_H

#include <math.h>
#include <stdlib.h>

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float failures;
    float n;
} Log;

typedef struct {
    Log log;

    float* observations;
    float* actions;
    float* rewards;
    float* terminals;

    int num_agents;
    unsigned int rng;

    float x;
    float y;
    float theta;
    float z;
    float previous_y;

    int step_count;
    float current_episode_return;

    float dt;
    float v;
    float wheelbase;
    float psi_max;
    float y_max;
    int max_steps;

    float d_y;
    int regime_switch_step;
    float d_y_after_switch;

    float k_psi;
    int action_delay_steps;
    float* action_buffer;
    int action_buffer_index;

    float sensor_noise_y_std;
    float sensor_noise_theta_std;

    float y0_min;
    float y0_max;
    float theta0_min;
    float theta0_max;

    float w_y;
    float w_theta;
    float w_psi;
    float alive_bonus;
    float failure_penalty;
    float w_center4;

    float z_clip;
    float reference;
    float lambda_value;

} KinematicBicycle;

static inline float kb_clamp(float value, float lower, float upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static inline float kb_uniform(KinematicBicycle* env, float lower, float upper) {
    float unit = (float)rand_r(&env->rng) / (float)RAND_MAX;
    return lower + unit * (upper - lower);
}

static inline float kb_standard_normal(KinematicBicycle* env) {
    // Box-Muller transform. Clamp u1 away from zero before logf().
    float u1 = kb_uniform(env, 1.0e-7f, 1.0f);
    float u2 = kb_uniform(env, 0.0f, 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

static inline void kb_reset_action_buffer(KinematicBicycle* env) {
    env->action_buffer_index = 0;

    for (int i = 0; i < env->action_delay_steps; i++) {
        env->action_buffer[i] = 0.0f;
    }
}

static inline float kb_apply_action_delay(
    KinematicBicycle* env,
    float psi_cmd
) {
    if (env->action_delay_steps <= 0) {
        return psi_cmd;
    }

    float delayed = env->action_buffer[env->action_buffer_index];
    env->action_buffer[env->action_buffer_index] = psi_cmd;

    env->action_buffer_index += 1;
    if (env->action_buffer_index >= env->action_delay_steps) {
        env->action_buffer_index = 0;
    }

    return delayed;
}

static inline void kb_write_observation(
    KinematicBicycle* env,
    int include_integral
) {
    float y_observed = env->y;
    float theta_observed = env->theta;

    if (env->sensor_noise_y_std > 0.0f) {
        y_observed += (
            env->sensor_noise_y_std
            * kb_standard_normal(env)
        );
    }

    if (env->sensor_noise_theta_std > 0.0f) {
        theta_observed += (
            env->sensor_noise_theta_std
            * kb_standard_normal(env)
        );
    }

    env->observations[0] = y_observed;
    env->observations[1] = theta_observed;

    if (include_integral) {
        env->observations[2] = env->z;
    }

    // The integral state uses the previously observed lateral position,
    // matching the Python wrapper behavior.
    env->previous_y = y_observed;
}

static inline void kb_reset_state(
    KinematicBicycle* env,
    int include_integral
) {
    env->x = 0.0f;
    env->y = kb_uniform(env, env->y0_min, env->y0_max);
    env->theta = kb_uniform(
        env,
        env->theta0_min,
        env->theta0_max
    );

    env->z = 0.0f;
    env->previous_y = env->y;
    env->step_count = 0;
    env->current_episode_return = 0.0f;

    kb_reset_action_buffer(env);
    kb_write_observation(env, include_integral);
}

static inline float kb_active_disturbance(const KinematicBicycle* env) {
    if (
        env->regime_switch_step >= 0
        && env->step_count >= env->regime_switch_step
    ) {
        return env->d_y_after_switch;
    }

    return env->d_y;
}

static inline void kb_finish_episode(
    KinematicBicycle* env,
    int failed
) {
    env->log.episode_return += env->current_episode_return;
    env->log.episode_length += (float)env->step_count;
    env->log.score += env->current_episode_return;
    env->log.failures += failed ? 1.0f : 0.0f;
    env->log.perf += failed ? 0.0f : 1.0f;
    env->log.n += 1.0f;
}

static inline void kb_step(
    KinematicBicycle* env,
    int include_integral
) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    if (include_integral) {
        env->z = kb_clamp(
            env->z + env->dt * (env->reference - env->previous_y),
            -env->z_clip,
            env->z_clip
        );
    }

    float normalized_action = kb_clamp(
        env->actions[0],
        -1.0f,
        1.0f
    );
    float psi_cmd = normalized_action * env->psi_max;
    float psi_cmd_applied = kb_apply_action_delay(env, psi_cmd);
    float psi = env->k_psi * psi_cmd_applied;
    float disturbance = kb_active_disturbance(env);

    float x_next = (
        env->x
        + env->v * cosf(env->theta) * env->dt
    );
    float y_next = (
        env->y
        + env->v * sinf(env->theta) * env->dt
        + disturbance * env->dt
    );
    float theta_next = (
        env->theta
        + (env->v / env->wheelbase) * tanf(psi) * env->dt
    );

    env->x = x_next;
    env->y = y_next;
    env->theta = theta_next;
    env->step_count += 1;

    float tracking_cost = (
        env->w_y * env->y * env->y
        + env->w_theta * env->theta * env->theta
        + env->w_psi * psi * psi
    );
    float centerline_penalty = (
        env->w_center4
        * env->y * env->y * env->y * env->y
    );

    float reward = (
        env->alive_bonus
        - tracking_cost
        - centerline_penalty
    );

    if (include_integral) {
        reward -= env->lambda_value * env->z * env->z;
    }

    int failed = fabsf(env->y) > env->y_max;
    int truncated = env->step_count >= env->max_steps;

    if (failed) {
        reward -= env->failure_penalty;
    }

    env->rewards[0] = reward;
    env->current_episode_return += reward;

    if (failed || truncated) {
        env->terminals[0] = 1.0f;
        kb_finish_episode(env, failed);
        kb_reset_state(env, include_integral);
        return;
    }

    kb_write_observation(env, include_integral);
}

#endif
