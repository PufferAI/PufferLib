#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

const unsigned char SEARCH = 0;
const unsigned char TRACK1 = 1;
const unsigned char TRACK2 = 2;
const unsigned char TRACK3 = 3;
const unsigned char TRACK4 = 4;
const unsigned char TRACK5 = 5;
const unsigned char NOOP = 6;

const int MAX_AZ_SLICES = 30;
const int MAX_EL_SLICES = 10;

const int MAX_SEARCHERS = 1;
const int MAX_TRACKERS = 5;
const int FEATURES_PER_TRACKER = 3; // t_desired, t_deadline, t_dwell_estimate

const int PLACEHOLDER_FOR_SENSOR_ID = 1;
const int OBSERVATION_SIZE = MAX_AZ_SLICES * MAX_EL_SLICES + MAX_TRACKERS * FEATURES_PER_TRACKER + PLACEHOLDER_FOR_SENSOR_ID;

const int MAX_TARGET_XY_RANGE = 184000000; // 184 km in millimeters
const int MAX_TARGET_Z_RANGE = 20000000; // 20 km in millimeters
const int MAX_TARGET_XY_VELOCITY = 1000; // 1000 m/s
const int MIN_TARGET_SINGER_SIGMA = 0;
const int MAX_TARGET_SINGER_SIGMA = 35;
const int MIN_TARGET_SINGER_THETA = 1; //TODO: should this be 1000 milliseconds?
const int MAX_TARGET_SINGER_THETA = 50; //TODO: should this be 50000 milliseconds?
// Table III "Singer Manoeuvre Parameters for Three Target Types"
// From A. Charlish, K. Woodbridge, and H. Griffiths,
// ‘Phased array radar resource management using continuous double auction’,
// IEEE Transactions on Aerospace and Electronic Systems,
// vol. 51, no. 3, pp. 2212–2224, 2015.
// DOI. No. 10.1109/TAES.2015.130558.
// Type 1: BigSigma in [20,35], BigTheta in [10,20]
// Type 2: BigSigma in [0,5], BigTheta in [1,4]
// Type 3: BigSigma in [5,20], BigTheta in [30,50]
// For now I'm just doing (min, max) over all types.

const int MIN_DWELL_TIME = 10; // 10 milliseconds
const int SEARCH_DWELL_TIME = 10; // 10 milliseconds (from Sunilas slides)

// For reset
const int ZERO_COST_SEARCH_TIME = MAX_AZ_SLICES * MAX_EL_SLICES * SEARCH_DWELL_TIME;
const int NO_TARGET = -1;

const unsigned int S_BAND_SENSOR = 0;
const unsigned int X_BAND_SENSOR = 1;

typedef struct Target Target;
struct Target
{
    float x;
    float x_velocity;
    float x_acceleration;
    float y;
    float y_velocity;
    float y_acceleration;
    float z;
    float z_velocity;
    float z_acceleration;
    float singer_sigma; // maneuver standard deviation
    float singer_theta; // maneuver time constant
    float priority;
};

typedef struct Radars Radars;
struct Radars
{
    int16_t *observations;
    int *actions;
    float *rewards;
    unsigned char *terminals;
    int tick;
    int s_band_t_until_free;
    int x_band_t_until_free;
    Target* targets; // should I allow more targets than trackers?
    int initial_targets;
};

void allocate(Radars *env)
{       
    env->observations = (int16_t *)calloc(OBSERVATION_SIZE, sizeof(int16_t));
    env->actions = (int *)calloc(1, sizeof(int));
    env->rewards = (float *)calloc(1, sizeof(float));
    env->terminals = (unsigned char *)calloc(1, sizeof(unsigned char));
}

void free_allocated(Radars *env)
{
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

void c_reset(Radars *env)
{
    // Set all search times to ZERO_COST_SEARCH_TIME
    memset(env->observations, ZERO_COST_SEARCH_TIME, OBSERVATION_SIZE * sizeof(int16_t));

    // Set all trackers to NO_TARGET
    for (int i = 0; i < MAX_TRACKERS * FEATURES_PER_TRACKER; i++)
    {
        env->observations[
            MAX_AZ_SLICES * MAX_EL_SLICES + i
        ] = NO_TARGET;
    }

    // Set the sensor type to S_BAND_SENSOR
    env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + MAX_TRACKERS * FEATURES_PER_TRACKER] = S_BAND_SENSOR;
   
    // Reset the clock
    env->tick = 0;

    // Set the initial targets
    for (int i = 0; i < MAX_TRACKERS; i++)
    {
        env->targets[i].x = rand() % MAX_TARGET_XY_RANGE;
        env->targets[i].y = rand() % MAX_TARGET_XY_RANGE;
        env->targets[i].z = rand() % MAX_TARGET_Z_RANGE;
        env->targets[i].x_velocity = rand() % MAX_TARGET_XY_VELOCITY;
        env->targets[i].y_velocity = rand() % MAX_TARGET_XY_VELOCITY;
        env->targets[i].z_velocity = 0; // targets spawn doing level flight
        env->targets[i].x_acceleration = 0;
        env->targets[i].y_acceleration = 0;
        env->targets[i].z_acceleration = 0;
        env->targets[i].singer_sigma = rand() % (MAX_TARGET_SINGER_SIGMA - MIN_TARGET_SINGER_SIGMA) + MIN_TARGET_SINGER_SIGMA;
        env->targets[i].singer_theta = rand() % (MAX_TARGET_SINGER_THETA - MIN_TARGET_SINGER_THETA) + MIN_TARGET_SINGER_THETA;
        env->targets[i].priority = 0;
    }
}

void c_step(Radars *env)
{
    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    // env->observations[OBSERVATION_SIZE] = TODO: update ;

    if (action == SEARCH)
    {
        // TODO: Implement search
    }
    else if (action == TRACK1 || action == TRACK2 || action == TRACK3 || action == TRACK4 || action == TRACK5)
    {
        // TODO: Implement tracking 
    }

    if (env->tick > 60000) // 1 minute ?
        {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        c_reset(env);
        return;
    }

    env->tick += 1;
}

typedef struct Client Client;
struct Client
{
    unsigned int px;
};

Client *make_client(Radars *env)
{
    Client *client = (Client *)calloc(1, sizeof(Client));
    int px = 64 * MAX_AZ_SLICES;
    InitWindow(px, px, "PufferLib Radars");
    SetTargetFPS(5);

    return client;
}

void close_client(Client *client)
{
    CloseWindow();
    free(client);
}

void c_render(Client *client, Radars *env)
{
    if (IsKeyDown(KEY_ESCAPE))
    {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    //int px = 64;
    //for (int i = 0; i < env->size; i++)
    //{
    //    for (int j = 0; j < env->size; j++)
    //    {
    //        int tex = env->observations[i * env->size + j];
    //        if (tex == EMPTY)
    //        {
    //            continue;
    //        }
    //        Color color = (tex == AGENT) ? (Color){0, 255, 255, 255} : (Color){255, 0, 0, 255};
    //        DrawRectangle(j * px, i * px, px, px, color);
    //    }
    //}
    EndDrawing();
}
