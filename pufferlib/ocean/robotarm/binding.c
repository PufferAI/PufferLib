#include "robot_arm.h"

#define Env RobotArm
#include "../env_binding.h"

// Pull in implementation to satisfy c_* symbols in a single TU build
#include "robot_arm.c"

// Environment initialization - pattern from squared.h
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // Null pointer safety check
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "Environment pointer is null");
        return -1;
    }
    
    // Set default max steps if provided
    if (kwargs) {
        PyObject* max_steps_obj = PyDict_GetItemString(kwargs, "max_steps");
        if (max_steps_obj && PyLong_Check(max_steps_obj)) {
            long max_steps_value = PyLong_AsLong(max_steps_obj);
            // Validate max_steps is within reasonable bounds
            if (max_steps_value > 0 && max_steps_value <= 10000) {
                env->max_steps = (int)max_steps_value;
            } else {
                env->max_steps = 500; // Default for invalid values
            }
        } else {
            env->max_steps = 500; // Default
        }

        // Keep compatibility with previous kwargs (accept but no-op where unused)
        PyObject* pick_and_place_obj = PyDict_GetItemString(kwargs, "pick_and_place_mode");
        env->pick_and_place_mode = (pick_and_place_obj && PyBool_Check(pick_and_place_obj) && PyObject_IsTrue(pick_and_place_obj)) ? 1 : 0;
        env->task = 0; // Reach by default

        // Optional production flag (no-op here but kept for forward compatibility)
        PyObject* production_obj = PyDict_GetItemString(kwargs, "production_mode");
        (void)production_obj;

        // Optional curriculum: reach_only
        PyObject* reach_only_obj = PyDict_GetItemString(kwargs, "reach_only");
        if (reach_only_obj && PyBool_Check(reach_only_obj) && PyObject_IsTrue(reach_only_obj)) {
            env->reach_only = 1;
            env->task = 0; // enforce REACH
        } else {
            env->reach_only = 0;
        }

        // Optional: training knobs
        PyObject* fs = PyDict_GetItemString(kwargs, "frame_skip");
        if (fs && PyLong_Check(fs)) env->frame_skip = (int)PyLong_AsLong(fs);
        PyObject* on = PyDict_GetItemString(kwargs, "obs_noise_std");
        if (on && (PyFloat_Check(on) || PyLong_Check(on))) env->obs_noise_std = (float)PyFloat_AsDouble(on);
        PyObject* an = PyDict_GetItemString(kwargs, "actuation_noise_std");
        if (an && (PyFloat_Check(an) || PyLong_Check(an))) env->actuation_noise_std = (float)PyFloat_AsDouble(an);
        PyObject* sd = PyDict_GetItemString(kwargs, "success_distance");
        if (sd && (PyFloat_Check(sd) || PyLong_Check(sd))) env->success_distance = (float)PyFloat_AsDouble(sd);
        PyObject* dr = PyDict_GetItemString(kwargs, "domain_randomization");
        if (dr && PyBool_Check(dr)) env->domain_randomization = PyObject_IsTrue(dr);

        // Motion smoothing knobs
        PyObject* sm = PyDict_GetItemString(kwargs, "action_smoothing_alpha");
        if (sm && (PyFloat_Check(sm) || PyLong_Check(sm))) env->action_smoothing_alpha = (float)PyFloat_AsDouble(sm);
        PyObject* al = PyDict_GetItemString(kwargs, "accel_limit");
        if (al && (PyFloat_Check(al) || PyLong_Check(al))) env->accel_limit = (float)PyFloat_AsDouble(al);
        PyObject* dp = PyDict_GetItemString(kwargs, "damping");
        if (dp && (PyFloat_Check(dp) || PyLong_Check(dp))) env->damping = (float)PyFloat_AsDouble(dp);

        // Curriculum knobs (optional)
        PyObject* cur_ep = PyDict_GetItemString(kwargs, "curriculum_episodes");
        if (cur_ep && PyLong_Check(cur_ep)) env->curriculum_episodes = (int)PyLong_AsLong(cur_ep);
        PyObject* sd_start = PyDict_GetItemString(kwargs, "success_distance_start");
        if (sd_start && (PyFloat_Check(sd_start) || PyLong_Check(sd_start))) env->success_distance_start = (float)PyFloat_AsDouble(sd_start);
        PyObject* sd_min = PyDict_GetItemString(kwargs, "success_distance_min");
        if (sd_min && (PyFloat_Check(sd_min) || PyLong_Check(sd_min))) env->success_distance_min = (float)PyFloat_AsDouble(sd_min);
        PyObject* og_start = PyDict_GetItemString(kwargs, "on_gripper_spawn_start");
        if (og_start && (PyFloat_Check(og_start) || PyLong_Check(og_start))) env->on_gripper_spawn_start = (float)PyFloat_AsDouble(og_start);
        PyObject* og_min = PyDict_GetItemString(kwargs, "on_gripper_spawn_min");
        if (og_min && (PyFloat_Check(og_min) || PyLong_Check(og_min))) env->on_gripper_spawn_min = (float)PyFloat_AsDouble(og_min);

        // Touch bonus scheduling (optional)
        PyObject* tbmax = PyDict_GetItemString(kwargs, "touch_bonus_max");
        if (tbmax && (PyFloat_Check(tbmax) || PyLong_Check(tbmax))) env->touch_bonus_max = (float)PyFloat_AsDouble(tbmax);
        PyObject* tbd = PyDict_GetItemString(kwargs, "touch_decay_steps");
        if (tbd && PyLong_Check(tbd)) env->touch_decay_steps = (int)PyLong_AsLong(tbd);

        // Pick-and-place episode start probability of pre-grasp
        PyObject* sgp = PyDict_GetItemString(kwargs, "start_grasp_prob");
        if (sgp && (PyFloat_Check(sgp) || PyLong_Check(sgp))) env->start_grasp_prob = (float)PyFloat_AsDouble(sgp);

        // Assistive curriculum switches
        PyObject* ae = PyDict_GetItemString(kwargs, "assist_enabled");
        if (ae && PyBool_Check(ae)) env->assist_enabled = PyObject_IsTrue(ae);
        PyObject* aeps = PyDict_GetItemString(kwargs, "assist_episodes");
        if (aeps && PyLong_Check(aeps)) env->assist_episodes = (int)PyLong_AsLong(aeps);
    } else {
        env->max_steps = 500;
        env->task = 0;
        env->pick_and_place_mode = 0;
    }
    
    return 0;
}

// Log aggregation - exact pattern from squared.h
static int my_log(PyObject* dict, Log* log) {
    // Null pointer safety checks
    if (!dict || !log) {
        return -1;
    }
    
    // Safe assignment with error checking
    if (assign_to_dict(dict, "perf", log->perf) != 0) return -1;
    if (assign_to_dict(dict, "score", log->score) != 0) return -1;
    if (assign_to_dict(dict, "episode_return", log->episode_return) != 0) return -1;
    if (assign_to_dict(dict, "episode_length", log->episode_length) != 0) return -1;
    // Derived metric: average reward per step (guards against divide-by-zero)
    float avg_steps = log->episode_length;
    float avg_return = log->episode_return;
    float avg_rps = (avg_steps > 0.0f) ? (avg_return / avg_steps) : 0.0f;
    if (assign_to_dict(dict, "avg_reward_per_step", avg_rps) != 0) return -1;
    
    return 0;
}