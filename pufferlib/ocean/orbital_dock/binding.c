#include "orbital_dock.h"
#include "render.h"

#define Env OrbitalDock
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->client = NULL;

    // Physics parameters
    env->mu = unpack(kwargs, "mu");
    env->station_radius = unpack(kwargs, "station_radius");
    env->dt = unpack(kwargs, "dt");
    env->max_thrust = unpack(kwargs, "max_thrust");
    env->mass = unpack(kwargs, "mass");
    env->fuel_budget = unpack(kwargs, "fuel_budget");
    env->max_steps = (int)unpack(kwargs, "max_steps");

    // Docking point (STELLAR: [0, 60, 0])
    env->dock_x = unpack(kwargs, "dock_x");
    env->dock_y = unpack(kwargs, "dock_y");
    env->dock_z = unpack(kwargs, "dock_z");
    env->dock_dist = unpack(kwargs, "dock_dist");
    env->dock_speed = unpack(kwargs, "dock_speed");
    env->dock_speed_start = unpack(kwargs, "dock_speed_start");
    env->anneal_steps = (int)unpack(kwargs, "anneal_steps");
    env->global_step = 0;

    // LOS cone
    double los_angle_deg = unpack(kwargs, "los_angle");
    env->los_half_angle = (los_angle_deg / 2.0) * M_PI / 180.0;
    env->los_extent = unpack(kwargs, "los_extent");

    // Initial condition ranges (STELLAR V-bar approach)
    env->init_x_center = unpack(kwargs, "init_x_center");
    env->init_y_center = unpack(kwargs, "init_y_center");
    env->init_z_center = unpack(kwargs, "init_z_center");
    env->init_x_range = unpack(kwargs, "init_x_range");
    env->init_y_range = unpack(kwargs, "init_y_range");
    env->init_z_range = unpack(kwargs, "init_z_range");

    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "dock_success", log->dock_success);
    assign_to_dict(dict, "crash_rate", log->crash_rate);
    assign_to_dict(dict, "timeout_rate", log->timeout_rate);
    assign_to_dict(dict, "fuel_used", log->fuel_used);
    assign_to_dict(dict, "final_distance", log->final_distance);
    assign_to_dict(dict, "final_rel_speed", log->final_rel_speed);
    return 0;
}
