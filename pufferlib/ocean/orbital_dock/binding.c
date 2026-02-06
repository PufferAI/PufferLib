#include "orbital_dock.h"
#include "render.h"

#define Env OrbitalDock
#include "../env_binding.h"

// Global curriculum state (shared across all envs in process)
static int g_curriculum_stage = 0;
static int g_curriculum_docks = 0;
static int g_curriculum_episodes = 0;
static int g_consecutive_above = 0;   // consecutive windows above target
static int g_consecutive_below = 0;   // consecutive windows below demotion threshold
#define G_CURRICULUM_WINDOW 10000     // Check every 10K global episodes
#define G_ADVANCE_STREAK 3            // Require 3 consecutive windows above target
#define G_DEMOTE_STREAK 2             // Require 2 consecutive windows below threshold

// Called from orbital_dock.h when an episode ends
void global_curriculum_update(OrbitalDock* env, int docked) {
    g_curriculum_episodes++;
    if (docked) {
        g_curriculum_docks++;
    }

    if (g_curriculum_episodes >= G_CURRICULUM_WINDOW) {
        double dock_rate = (double)g_curriculum_docks / (double)g_curriculum_episodes;
        double target_rate = CURRICULUM_PARAMS[g_curriculum_stage][5];

        if (dock_rate >= target_rate && g_curriculum_stage < 4) {
            g_consecutive_above++;
            g_consecutive_below = 0;
            printf("CURRICULUM: Stage %d streak %d/%d (dock_rate=%.1f%% >= target=%.1f%%)\n",
                   g_curriculum_stage, g_consecutive_above, G_ADVANCE_STREAK,
                   dock_rate * 100.0, target_rate * 100.0);
            if (g_consecutive_above >= G_ADVANCE_STREAK) {
                g_curriculum_stage++;
                g_consecutive_above = 0;
                printf("CURRICULUM: ADVANCED to stage %d\n", g_curriculum_stage);
            }
            fflush(stdout);
        } else if (g_curriculum_stage > 0 && dock_rate < target_rate * 0.5) {
            g_consecutive_below++;
            g_consecutive_above = 0;
            printf("CURRICULUM: Demote streak %d/%d (dock_rate=%.1f%% < %.1f%%)\n",
                   g_consecutive_below, G_DEMOTE_STREAK,
                   dock_rate * 100.0, target_rate * 50.0);
            if (g_consecutive_below >= G_DEMOTE_STREAK) {
                g_curriculum_stage--;
                g_consecutive_below = 0;
                printf("CURRICULUM: DEMOTED to stage %d\n", g_curriculum_stage);
            }
            fflush(stdout);
        } else {
            // In between — reset both streaks
            g_consecutive_above = 0;
            g_consecutive_below = 0;
        }

        g_curriculum_docks = 0;
        g_curriculum_episodes = 0;
    }

    // Sync this env's stage with global
    env->curriculum_stage = g_curriculum_stage;
}

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

    // Hierarchical velocity control parameters
    env->kp = unpack(kwargs, "kp");
    env->max_cmd_vel = unpack(kwargs, "max_cmd_vel");

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

    // Curriculum: initialize global stage from difficulty param (for eval)
    int init_stage = (int)env->difficulty;
    if (init_stage < 0) init_stage = 0;
    if (init_stage > 4) init_stage = 4;
    if (init_stage > g_curriculum_stage) {
        g_curriculum_stage = init_stage;
    }
    env->curriculum_stage = g_curriculum_stage;
    env->curriculum_docks = 0;
    env->curriculum_episodes = 0;
    env->curriculum_window = G_CURRICULUM_WINDOW;

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
