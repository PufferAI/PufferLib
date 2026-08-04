#include "kinematic_bicycle_integral.h"

#define OBS_SIZE 3
#define NUM_ATNS 1
#define ACT_SIZES {1}
#define OBS_TENSOR_T FloatTensor

#define Env KinematicBicycle
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;

    env->dt = (float)dict_get(kwargs, "dt")->value;
    env->v = (float)dict_get(kwargs, "v")->value;
    env->wheelbase = (float)dict_get(kwargs, "wheelbase")->value;
    env->psi_max = (float)dict_get(kwargs, "psi_max")->value;
    env->y_max = (float)dict_get(kwargs, "y_max")->value;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;

    env->d_y = (float)dict_get(kwargs, "d_y")->value;
    env->regime_switch_step =
        (int)dict_get(kwargs, "regime_switch_step")->value;
    env->d_y_after_switch =
        (float)dict_get(kwargs, "d_y_after_switch")->value;

    env->k_psi = (float)dict_get(kwargs, "k_psi")->value;
    env->action_delay_steps =
        (int)dict_get(kwargs, "action_delay_steps")->value;

    env->sensor_noise_y_std =
        (float)dict_get(kwargs, "sensor_noise_y_std")->value;
    env->sensor_noise_theta_std =
        (float)dict_get(kwargs, "sensor_noise_theta_std")->value;

    env->y0_min = (float)dict_get(kwargs, "y0_min")->value;
    env->y0_max = (float)dict_get(kwargs, "y0_max")->value;
    env->theta0_min = (float)dict_get(kwargs, "theta0_min")->value;
    env->theta0_max = (float)dict_get(kwargs, "theta0_max")->value;

    env->w_y = (float)dict_get(kwargs, "w_y")->value;
    env->w_theta = (float)dict_get(kwargs, "w_theta")->value;
    env->w_psi = (float)dict_get(kwargs, "w_psi")->value;
    env->alive_bonus = (float)dict_get(kwargs, "alive_bonus")->value;
    env->failure_penalty =
        (float)dict_get(kwargs, "failure_penalty")->value;
    env->w_center4 = (float)dict_get(kwargs, "w_center4")->value;

    env->z_clip = (float)dict_get(kwargs, "z_clip")->value;
    env->reference = (float)dict_get(kwargs, "reference")->value;
    env->lambda_value = (float)dict_get(kwargs, "lambda")->value;

    if (env->action_delay_steps < 0) {
        env->action_delay_steps = 0;
    }

    if (env->action_delay_steps > 0) {
        env->action_buffer = (float*)calloc(
            (size_t)env->action_delay_steps,
            sizeof(float)
        );
        assert(env->action_buffer != NULL);
    }
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "failures", log->failures);
}
