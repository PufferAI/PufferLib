// Standalone C demo for orbital_dock environment
// Compile using: ./scripts/build_ocean.sh orbital_dock [local|fast]
// Run with: ./orbital_dock

#include "orbital_dock.h"
#include "render.h"
#include <time.h>

void generate_random_actions(OrbitalDock *env) {
    env->actions[0] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[1] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[2] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

int main() {
    srand(time(NULL));

    OrbitalDock *env = calloc(1, sizeof(OrbitalDock));

    // Physics parameters (GEO orbit)
    env->mu = 3.986e14;
    env->station_radius = 42164e3;
    env->dt = 1.0;
    env->max_thrust = 10.0;
    env->mass = 500.0;
    env->fuel_budget = 100.0;
    env->max_steps = 2500;

    // Docking point
    env->dock_x = 0.0;
    env->dock_y = 60.0;
    env->dock_z = 0.0;
    env->dock_dist = 10.0;
    env->dock_speed = 2.0;
    env->dock_speed_start = 10.0;
    env->anneal_steps = 50000;
    env->global_step = 0;

    // LOS cone (60 deg total)
    env->los_half_angle = 30.0 * M_PI / 180.0;
    env->los_extent = 800.0;

    // Initial condition ranges
    env->init_x_center = 0.0;
    env->init_y_center = 800.0;
    env->init_z_center = 0.0;
    env->init_x_range = 400.0;
    env->init_y_range = 300.0;
    env->init_z_range = 400.0;

    // Allocate buffers
    env->observations = (float *)calloc(10, sizeof(float));
    env->actions = (float *)calloc(3, sizeof(float));
    env->rewards = (float *)calloc(1, sizeof(float));
    env->terminals = (unsigned char *)calloc(1, sizeof(unsigned char));
    env->client = NULL;

    c_reset(env);
    c_render(env);

    while (!WindowShouldClose()) {
        generate_random_actions(env);
        c_step(env);
        c_render(env);
    }

    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);

    return 0;
}
