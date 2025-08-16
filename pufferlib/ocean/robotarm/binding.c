#include "robot_arm.h"

#define Env RobotArm
#include "../env_binding.h"
#include "robot_arm.c"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    if (!env) return -1;
    env->max_steps = (int)unpack(kwargs, "max_steps");
    env->pick_and_place_mode = (int)unpack(kwargs, "pick_and_place_mode");
    // reach_only removed - pick-and-place mode only
    env->frame_skip = (int)unpack(kwargs, "frame_skip");
    if (PyDict_Contains(kwargs, PyUnicode_FromString("physics_substeps"))) {
        env->physics_substeps = (int)unpack(kwargs, "physics_substeps");
    }
    env->success_distance = (float)unpack(kwargs, "success_distance");
    env->domain_randomization = (int)unpack(kwargs, "domain_randomization");

    env->obs_noise_std = (float)unpack(kwargs, "obs_noise_std");
    env->actuation_noise_std = (float)unpack(kwargs, "actuation_noise_std");
    env->action_smoothing_alpha = (float)unpack(kwargs, "action_smoothing_alpha");
    env->accel_limit = (float)unpack(kwargs, "accel_limit");
    env->damping = (float)unpack(kwargs, "damping");
    if (PyDict_Contains(kwargs, PyUnicode_FromString("action_penalty_coef"))) {
        env->action_penalty_coef = (float)unpack(kwargs, "action_penalty_coef");
    } else {
        env->action_penalty_coef = 0.0f;
    }
    if (PyDict_Contains(kwargs, PyUnicode_FromString("reward_scale"))) {
        env->reward_scale = (float)unpack(kwargs, "reward_scale");
    } else {
        env->reward_scale = 1.0f;
    }

    env->curriculum_episodes = (int)unpack(kwargs, "curriculum_episodes");
    env->success_distance_start = (float)unpack(kwargs, "success_distance_start");
    env->success_distance_min = (float)unpack(kwargs, "success_distance_min");
    env->on_gripper_spawn_start = (float)unpack(kwargs, "on_gripper_spawn_start");
    env->on_gripper_spawn_min = (float)unpack(kwargs, "on_gripper_spawn_min");
    env->on_gripper_spawn_prob = (float)unpack(kwargs, "on_gripper_spawn_prob");
    env->start_grasp_prob = (float)unpack(kwargs, "start_grasp_prob");
    env->touch_bonus_max = (float)unpack(kwargs, "touch_bonus_max");
    env->touch_decay_steps = (int)unpack(kwargs, "touch_decay_steps");

    env->assist_enabled = (int)unpack(kwargs, "assist_enabled");
    env->assist_episodes = (int)unpack(kwargs, "assist_episodes");

    // Rendering controls (optional)
    env->headless = (int)unpack(kwargs, "headless");
    int rd = (int)unpack(kwargs, "render_decimation");
    env->render_decimation = rd > 0 ? rd : 1;
    env->render_target_fps = (int)unpack(kwargs, "render_target_fps");
    env->vsync = (int)unpack(kwargs, "vsync");
    if (PyDict_Contains(kwargs, PyUnicode_FromString("cube_model_visual_mul"))) {
        env->cube_model_visual_mul = (float)unpack(kwargs, "cube_model_visual_mul");
    }

    env->task = 0;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    if (!dict || !log) return -1;
    float n = log->n;
    float inv = (n > 0.0f) ? (1.0f / n) : 0.0f;
    assign_to_dict(dict, "n", n);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return * inv);
    assign_to_dict(dict, "episode_length", log->episode_length * inv);
    assign_to_dict(dict, "pick_success_rate", log->pick_success_rate * inv);
    assign_to_dict(dict, "place_success_rate", log->place_success_rate * inv);
    log->perf = 0.0f;
    log->score = 0.0f;
    log->episode_return = 0.0f;
    log->episode_length = 0.0f;
    log->pick_success_rate = 0.0f;
    log->place_success_rate = 0.0f;
    log->n = 0.0f;
    return 0;
}