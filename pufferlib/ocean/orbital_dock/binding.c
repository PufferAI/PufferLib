#include "orbital_dock.h"
#include "render.h"

#define Env OrbitalDock
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // Initialize render client to NULL
    env->client = NULL;
    // Physics parameters
    env->mu = unpack(kwargs, "mu");
    env->station_radius = unpack(kwargs, "station_radius");
    env->dt = unpack(kwargs, "dt");
    env->max_thrust = unpack(kwargs, "max_thrust");
    env->mass = unpack(kwargs, "mass");
    env->fuel_budget = unpack(kwargs, "fuel_budget");
    env->max_steps = (int)unpack(kwargs, "max_steps");

    // Docking conditions
    env->dock_dist = unpack(kwargs, "dock_dist");
    env->dock_speed = unpack(kwargs, "dock_speed");

    // Termination thresholds (fixed values)
    env->earth_radius = 6.371e6;   // Earth radius in meters
    env->deorbit_alt = 100000.0;   // 100 km
    env->escape_alt = 50000000.0;  // 50,000 km

    // Difficulty/randomization
    env->difficulty = unpack(kwargs, "difficulty");
    env->alt_offset_max = 50000.0;   // 50 km
    env->phase_offset_max = 0.524;   // 30 degrees in radians
    env->incl_offset_max = 0.262;    // 15 degrees in radians
    env->vel_perturb_max = 5.0;      // 5 m/s

    // Reward weights
    env->rw_dock = unpack(kwargs, "reward_dock");
    env->rw_dist_shaping = unpack(kwargs, "reward_dist_shaping");
    env->rw_closing = unpack(kwargs, "reward_closing");
    env->rw_vel_match = unpack(kwargs, "reward_vel_match");
    env->rw_fuel = unpack(kwargs, "reward_fuel_penalty");
    env->rw_crash = unpack(kwargs, "reward_crash");
    env->rw_deorbit = unpack(kwargs, "reward_deorbit");
    env->rw_escape = unpack(kwargs, "reward_escape");
    env->rw_plane_align = unpack(kwargs, "reward_plane_align");
    env->rw_node_timing = unpack(kwargs, "reward_node_timing");

    // Curriculum: start at stage 0 (free docks)
    env->curriculum_stage = 0;
    env->curriculum_docks = 0;
    env->curriculum_episodes = 0;
    env->curriculum_window = 500;  // Check every 500 episodes

    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "dock_success", log->dock_success);
    assign_to_dict(dict, "crash_rate", log->crash_rate);
    assign_to_dict(dict, "deorbit_rate", log->deorbit_rate);
    assign_to_dict(dict, "escape_rate", log->escape_rate);
    assign_to_dict(dict, "timeout_rate", log->timeout_rate);
    assign_to_dict(dict, "fuel_used", log->fuel_used);
    assign_to_dict(dict, "final_distance", log->final_distance);
    assign_to_dict(dict, "final_rel_speed", log->final_rel_speed);
    assign_to_dict(dict, "curriculum_stage", log->curriculum_stage);
    return 0;
}
