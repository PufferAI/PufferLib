#include <math.h>
#include <stdlib.h>
#include <string.h>

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
const float AZ_DEGREES_PER_SLICE = 90.0f / MAX_AZ_SLICES;
const float EL_DEGREES_PER_SLICE = 30.0f / MAX_EL_SLICES;

const int MAX_SEARCHERS = 1;
const int FEATURES_PER_TRACKER = 3; // t_desired, t_deadline, t_dwell_estimate

const int PLACEHOLDER_FOR_SENSOR_ID = 1;

const float S_BAND_MAX_RANGE = 184000000.0f; // 184 km in millimeters
const float X_BAND_MAX_RANGE = 100000000.0f; // 100 km in millimeters
const float S_BAND_MIN_RANGE = 10000000.0f;  // 10 km in millimeters
const float X_BAND_MIN_RANGE = 1000000.0f;   // 1 km in millimeters

const float MAX_TARGET_XY_RANGE = 184000000.0f; // 184 km in millimeters
const float MAX_TARGET_Z_RANGE = 20000000.0f;   // 20 km in millimeters
const float MAX_TARGET_XY_VELOCITY = 1000.0f;   // 1000 m/s
const float MIN_TARGET_SINGER_SIGMA = 0.0f;
const float MAX_TARGET_SINGER_SIGMA = 35.0f;
const float MIN_TARGET_SINGER_THETA =
    1.0f; // TODO: should this be 1000 milliseconds?
const float MAX_TARGET_SINGER_THETA =
    50.0f; // TODO: should this be 50000 milliseconds?
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

const int PRIORITY_LEVELS = 3;

const int MIN_DWELL_TIME = 10;    // 10 milliseconds
const int MAX_DWELL_TIME = 100;   // 100 milliseconds
const int SEARCH_DWELL_TIME = 10; // 10 milliseconds (from Sunilas slides)

// For reset
const int ZERO_COST_SEARCH_TIME =
    MAX_AZ_SLICES * MAX_EL_SLICES * SEARCH_DWELL_TIME;
const int NO_TARGET = -1;

const unsigned int S_BAND_SENSOR = 0;
const unsigned int X_BAND_SENSOR = 1;

const float REFERENCE_DWELL_TIME = 0.01f;
const float REFERENCE_RANGE = 184000000.0f;
const float REFERENCE_CROSS_SECTION = 1.0f;
const float REFERENCE_SNR = 40.0f;

const float TRACK_UPDATE_REWARD = 1.0f;
const float TRACK_DELAY_PENALTY = 0.1f; // Penalty per ms
const float TRACK_LOSS_PENALTY = 10.0f;

const float VERTICAL_MOTION_FACTOR = 0.1f;

const int WINDOW_X_PX = 480;
const int WINDOW_Y_PX = 270;

typedef struct Target Target;
struct Target {
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
  bool is_active;
  bool is_tracked;
};

typedef struct Radars Radars;
struct Radars {
  int16_t *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int tick;
  int s_band_t_until_free;
  int x_band_t_until_free;
  Target *targets; // should I allow more targets than trackers?
  int initial_targets;
  int max_trackers;
};

void allocate(Radars *env) {
  env->observations = (int16_t *)calloc(
      (MAX_AZ_SLICES * MAX_EL_SLICES +
       env->max_trackers * FEATURES_PER_TRACKER + PLACEHOLDER_FOR_SENSOR_ID),
      sizeof(int16_t));
  env->actions = (int *)calloc(1, sizeof(int));
  env->rewards = (float *)calloc(1, sizeof(float));
  env->terminals = (unsigned char *)calloc(1, sizeof(unsigned char));
}

void free_allocated(Radars *env) {
  free(env->observations);
  free(env->actions);
  free(env->rewards);
  free(env->terminals);
}

float normal(float mean, float stddev) {
  float u1 = (float)rand() / RAND_MAX;
  float u2 = (float)rand() / RAND_MAX;
  // Apply Box-Muller transform
  float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
  return mean + stddev * z0;
}

void initialize_target(Radars *env, int target_index) {
  env->targets[target_index].x =
      (float)rand() / (float)(RAND_MAX / MAX_TARGET_XY_RANGE);
  env->targets[target_index].y =
      (float)rand() / (float)(RAND_MAX / MAX_TARGET_XY_RANGE);
  env->targets[target_index].z =
      (float)rand() / (float)(RAND_MAX / MAX_TARGET_Z_RANGE);
  env->targets[target_index].x_velocity =
      (float)rand() / (float)(RAND_MAX / MAX_TARGET_XY_VELOCITY);
  env->targets[target_index].y_velocity =
      (float)rand() / (float)(RAND_MAX / MAX_TARGET_XY_VELOCITY);
  env->targets[target_index].z_velocity = 0; // targets spawn doing level flight
  env->targets[target_index].x_acceleration = 0;
  env->targets[target_index].y_acceleration = 0;
  env->targets[target_index].z_acceleration = 0;
  env->targets[target_index].singer_sigma =
      (float)rand() / (float)(RAND_MAX / (MAX_TARGET_SINGER_SIGMA -
                                          MIN_TARGET_SINGER_SIGMA)) +
      MIN_TARGET_SINGER_SIGMA;
  env->targets[target_index].singer_theta =
      (float)rand() / (float)(RAND_MAX / (MAX_TARGET_SINGER_THETA -
                                          MIN_TARGET_SINGER_THETA)) +
      MIN_TARGET_SINGER_THETA;
  env->targets[target_index].priority = rand() % PRIORITY_LEVELS;
  env->targets[target_index].is_active = false;
  env->targets[target_index].is_tracked = false;
}

void c_reset(Radars *env) {
  // Set all sectors to zero cost search time
  for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
    env->observations[i] = ZERO_COST_SEARCH_TIME;
  }

  // Set all trackers to NO_TARGET
  for (int i = 0; i < env->max_trackers * FEATURES_PER_TRACKER; i++) {
    env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i] = NO_TARGET;
  }

  // Set the sensor type to S_BAND_SENSOR
  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                    env->max_trackers * FEATURES_PER_TRACKER] = S_BAND_SENSOR;

  env->tick = 0;
  env->s_band_t_until_free = 0;
  env->x_band_t_until_free = 0;

  for (int i = 0; i < env->max_trackers; i++) {
    initialize_target(env, i);
  }
  for (int i = 0; i < env->initial_targets; i++) {
    env->targets[i].is_active = true;
  }
}

void update_tracker(Radars *env, int tracker_id) {
  float target_range =
      sqrt(env->targets[tracker_id].x * env->targets[tracker_id].x +
           env->targets[tracker_id].y * env->targets[tracker_id].y +
           env->targets[tracker_id].z * env->targets[tracker_id].z);
  // TODO: replace 42 with a per-target rcs
  float target_cross_section = 42;

  // @Sunila todo fix sigma_theta as well
  float sigma_theta = 1; // I'm not sure how this translates to 3d, I also
                         // think this should be sensor dependent
  // This should also be sensor dependent
  float u = 0.3; // Magic number from the slides.
  float new_t_desired = (0.4 *
                         pow((target_range * sigma_theta *
                              sqrt(env->targets[tracker_id].singer_theta) /
                              env->targets[tracker_id].singer_sigma),
                             0.4) *
                         pow(u, 2.4) / (1 + 0.5 * pow(u, 2)));
  float new_t_deadline = new_t_desired * 2 * env->targets[tracker_id].priority;
  float new_t_dwell_estimate =
      (REFERENCE_DWELL_TIME * pow((target_range / REFERENCE_RANGE), 4) *
       (REFERENCE_CROSS_SECTION / target_cross_section) * REFERENCE_SNR);
  if (new_t_dwell_estimate < MIN_DWELL_TIME) {
    new_t_dwell_estimate = MIN_DWELL_TIME;
  }
  if (new_t_dwell_estimate > MAX_DWELL_TIME) {
    new_t_dwell_estimate = MAX_DWELL_TIME;
  }

  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                    tracker_id * FEATURES_PER_TRACKER] = new_t_desired;
  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                    tracker_id * FEATURES_PER_TRACKER + 1] = new_t_deadline;
  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                    tracker_id * FEATURES_PER_TRACKER + 2] =
      new_t_dwell_estimate;
}

void search_sector(Radars *env, int sector) {
  float az_min =
      (sector % MAX_AZ_SLICES) * (AZ_DEGREES_PER_SLICE)*M_PI / 180.0f;
  float az_max =
      ((sector + 1) % MAX_AZ_SLICES) * (AZ_DEGREES_PER_SLICE)*M_PI / 180.0f;
  float el_min =
      (sector / MAX_AZ_SLICES) * (EL_DEGREES_PER_SLICE)*M_PI / 180.0f;
  float el_max = ((sector + MAX_AZ_SLICES) / MAX_AZ_SLICES) *
                 (EL_DEGREES_PER_SLICE)*M_PI / 180.0f;

  for (int i = 0; i < env->max_trackers; i++) {
    if (env->targets[i].is_active && !env->targets[i].is_tracked) {
      float az = atan2(env->targets[i].y, env->targets[i].x);
      float el =
          asin(env->targets[i].z / sqrt(env->targets[i].x * env->targets[i].x +
                                        env->targets[i].y * env->targets[i].y +
                                        env->targets[i].z * env->targets[i].z));
      if (az >= az_min && az < az_max && el >= el_min && el < el_max) {
        float target_range = sqrt(env->targets[i].x * env->targets[i].x +
                                  env->targets[i].y * env->targets[i].y +
                                  env->targets[i].z * env->targets[i].z);
        float probability_of_detection = 0.0f;
        if (env->s_band_t_until_free == 0) {
          // Probability of detection is approximated as
          // 1-exp^(-BigConstant/range^4) for s-band, 10ms dwell on 1m
          // square target at 184km has 0.75 probability of detection
          // 1-exp(-(10e32/184_000_000^4)) = 0.53,
          if (target_range >= S_BAND_MIN_RANGE &&
              target_range <= S_BAND_MAX_RANGE) {
            probability_of_detection =
                1 -
                exp((((-10e32 / target_range) / target_range) / target_range) /
                    target_range);
          }
        } else {
          // 1-exp(-(10e31/100_000_000^4)) = 0.63
          if (target_range >= X_BAND_MIN_RANGE &&
              target_range <= X_BAND_MAX_RANGE) {
            probability_of_detection =
                1 -
                exp((((-10e31 / target_range) / target_range) / target_range) /
                    target_range);
          }
        }
        if ((float)rand() / (float)(RAND_MAX) < probability_of_detection) {
          env->targets[i].is_tracked = true;
          update_tracker(env, i);
        }
      }
    }
  }
}

void c_step(Radars *env) {
  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = 0.0f;

  if (action == SEARCH) {
    // Find the least recently used sector
    int least_recently_used_sector = 0;
    for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
      if (env->observations[i] <
          env->observations[least_recently_used_sector]) {
        least_recently_used_sector = i;
      }
    }

    if (env->s_band_t_until_free == 0) {
      // S-band searches four sectors, right and below the least recently
      // used sector
      search_sector(env, least_recently_used_sector);
      env->observations[least_recently_used_sector] = ZERO_COST_SEARCH_TIME;

      search_sector(env, (least_recently_used_sector + 1) %
                             (MAX_AZ_SLICES * MAX_EL_SLICES));
      env->observations[(least_recently_used_sector + 1) %
                        (MAX_AZ_SLICES * MAX_EL_SLICES)] =
          ZERO_COST_SEARCH_TIME;

      search_sector(env, (least_recently_used_sector + MAX_AZ_SLICES) %
                             (MAX_AZ_SLICES * MAX_EL_SLICES));
      env->observations[(least_recently_used_sector + MAX_AZ_SLICES) %
                        (MAX_AZ_SLICES * MAX_EL_SLICES)] =
          ZERO_COST_SEARCH_TIME;

      search_sector(env, (least_recently_used_sector + MAX_AZ_SLICES + 1) %
                             (MAX_AZ_SLICES * MAX_EL_SLICES));
      env->observations[(least_recently_used_sector + MAX_AZ_SLICES + 1) %
                        (MAX_AZ_SLICES * MAX_EL_SLICES)] =
          ZERO_COST_SEARCH_TIME;

      env->s_band_t_until_free = SEARCH_DWELL_TIME;
    } else {
      // X-band searches the least recently used sector
      search_sector(env, least_recently_used_sector);
      env->observations[least_recently_used_sector] = ZERO_COST_SEARCH_TIME;

      env->x_band_t_until_free = SEARCH_DWELL_TIME;
    }
  }
  // else if (action == TRACK1 || action == TRACK2 || action == TRACK3 || action
  // == TRACK4 || action == TRACK5)
  else if (action <= env->max_trackers) {
    action -= 1; // easier than -1 all over the place for indexing.

    if (!env->targets[action].is_tracked) {
      env->rewards[0] = -1.0f;
    } else {
      env->rewards[0] = TRACK_UPDATE_REWARD;
      // Penalize the delay in updating the tracker
      if (env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                            action * FEATURES_PER_TRACKER] < 0) {
        env->rewards[0] -=
            (float)(env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                                      action * FEATURES_PER_TRACKER] *
                    TRACK_DELAY_PENALTY / (1 + env->targets[action].priority));
      }

      float target_range =
          sqrt(env->targets[action].x * env->targets[action].x +
               env->targets[action].y * env->targets[action].y +
               env->targets[action].z * env->targets[action].z);
      if (env->s_band_t_until_free == 0 && target_range < S_BAND_MAX_RANGE &&
          target_range > S_BAND_MIN_RANGE) {
        update_tracker(env, action);
      } else if (env->x_band_t_until_free == 0 &&
                 target_range < X_BAND_MAX_RANGE &&
                 target_range > X_BAND_MIN_RANGE) {
        update_tracker(env, action);
      }
      if (env->s_band_t_until_free == 0) {
        env->s_band_t_until_free =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                              action * FEATURES_PER_TRACKER + 2];
      } else {
        // TODO: actually calculate t_dwell for x-band, for now, just
        // make it faster than s-band
        env->x_band_t_until_free =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                              action * FEATURES_PER_TRACKER + 2] /
            2;
      }
    }
  } else {
    // TODO: Throw error?
  }

  // Move simulation forward
  int delta_t = env->s_band_t_until_free;
  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                    env->max_trackers * FEATURES_PER_TRACKER + 1] =
      S_BAND_SENSOR;
  if (env->x_band_t_until_free < delta_t) {
    delta_t = env->x_band_t_until_free;
    env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                      env->max_trackers * FEATURES_PER_TRACKER + 1] =
        X_BAND_SENSOR;
  }

  if (delta_t > 0) {
    env->tick += delta_t;
    env->s_band_t_until_free -= delta_t;
    env->x_band_t_until_free -= delta_t;
    for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
      env->observations[i] -= delta_t;
    }
    for (int i = 0; i < env->max_trackers; i++) {
      if (env->targets[i].is_tracked) {
        env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                          i * FEATURES_PER_TRACKER] -= delta_t;
        env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                          i * FEATURES_PER_TRACKER + 1] -= delta_t;
        // if the tracker has expired, lose the track and apply the penalty
        if (env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                              i * FEATURES_PER_TRACKER + 1] < 0) {
          env->targets[i].is_tracked = false;
          env->rewards[0] -= TRACK_LOSS_PENALTY;
        }
        // t_dwell_estimate does not change
      }
    }

    // Update locations

    for (int i = 0; i < env->max_trackers; i++) {
      env->targets[i].x += env->targets[i].x_velocity * delta_t / 1000.0f +
                           env->targets[i].x_acceleration *
                               (delta_t / 1000.0f) * (delta_t / 1000.0f);
      env->targets[i].y += env->targets[i].y_velocity * delta_t / 1000.0f +
                           env->targets[i].y_acceleration *
                               (delta_t / 1000.0f) * (delta_t / 1000.0f);
      env->targets[i].z += env->targets[i].z_velocity * delta_t / 1000.0f +
                           env->targets[i].z_acceleration *
                               (delta_t / 1000.0f) * (delta_t / 1000.0f);
      env->targets[i].x_velocity +=
          env->targets[i].x_acceleration * delta_t / 1000.0f;
      env->targets[i].y_velocity +=
          env->targets[i].y_acceleration * delta_t / 1000.0f;
      env->targets[i].z_velocity +=
          env->targets[i].z_acceleration * delta_t / 1000.0f;
      float rho = exp(-(delta_t / 1000.0f) / env->targets[i].singer_theta);
      env->targets[i].x_acceleration +=
          sqrt(1 - rho * rho) * normal(0, env->targets[i].singer_sigma);
      env->targets[i].y_acceleration +=
          sqrt(1 - rho * rho) * normal(0, env->targets[i].singer_sigma);
      env->targets[i].z_acceleration +=
          sqrt(1 - rho * rho) * normal(0, env->targets[i].singer_sigma) *
          VERTICAL_MOTION_FACTOR;
    }

    // Reset targets that have gone out of bounds
    for (int i = 0; i < env->max_trackers; i++) {
      if (env->targets[i].x < 0 || env->targets[i].x > MAX_TARGET_XY_RANGE ||
          env->targets[i].y < 0 || env->targets[i].y > MAX_TARGET_XY_RANGE ||
          env->targets[i].z < 0 || env->targets[i].z > MAX_TARGET_Z_RANGE ||
          env->targets[i].x * env->targets[i].x +
                  env->targets[i].y * env->targets[i].y +
                  env->targets[i].z * env->targets[i].z >
              S_BAND_MAX_RANGE * S_BAND_MAX_RANGE * 1.05) {
        initialize_target(env, i);
      }
    }

    if (env->tick > 60000) // 1 minute ?
    {
      env->terminals[0] = 1;
      env->rewards[0] = -1.0;
      c_reset(env);
      return;
    }
  }
}

typedef struct {
  int scale;
} Client;

Client *make_client(int scale) {
  Client *client = (Client *)malloc(sizeof(Client));
  if (!(scale == 1 || scale == 2 || scale == 4 || scale == 8)) {
    fprintf(stderr, "Error: scale is one of 1,2,4,8. (4 is 1080p, 8 is 4k).\n");
    exit(1);
  }

  // The plan position indicator will be square on left of screen
  // Origin will be at bottom-left
  // X axis positive to right
  // Y axis positive to top
  // I'm going to initially just try to do a WINDOW_Y_PX x WINDOW_Y_PX ppi with
  // a 240 x 80 search indicator on the top right and just hope they don't
  // overlap. int ppi_dimension = WINDOW_Y_PX * scale; int
  // search_indicator_width = 8 * scale;

  client->scale = scale;
  InitWindow(WINDOW_X_PX * scale, WINDOW_Y_PX * scale, "PufferLib Radars");
  SetTargetFPS(30);
  return client;
}

void close_client(Client *client) {
  CloseWindow();
  free(client);
}

Color COLORS[] = {
    (Color){6, 24, 24, 255},     // 0: Darkteal
    (Color){0, 0, 255, 255},     // 1: Blue
    (Color){0, 128, 255, 255},   // 2: Sky
    (Color){128, 128, 128, 255}, // 3: Gray
    (Color){255, 0, 0, 255},     // 4: Red
    (Color){255, 255, 255, 255}, // 5: White
    (Color){255, 85, 85, 255},   // 6: Salmon
    (Color){170, 170, 170, 255}, // 7: Silver
    (Color){0, 255, 255, 255},   // 8: Cyan
    (Color){255, 255, 0, 255},   // 9: Yellow
};

void c_render(Client *client, Radars *env) {
  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(COLORS[0]);

  // Draw the search indicator
  int cell_width = 9 * client->scale;
  int cell_height = 9 * client->scale;
  int grid_x = WINDOW_X_PX * client->scale - MAX_AZ_SLICES * cell_width;

  for (int i = 0; i < MAX_EL_SLICES; i++) {
    for (int j = 0; j < MAX_AZ_SLICES; j++) {
      int sector = i * MAX_AZ_SLICES + j;
      int zero_cost_time_remaining = env->observations[sector];
      Color color;
      if (zero_cost_time_remaining > 0) {
        // Map zero_cost_time_remaining to a cyan - to - grey scale
        float good_intensity =
            (float)zero_cost_time_remaining / ZERO_COST_SEARCH_TIME;
        if (good_intensity < 0) {
          good_intensity = 0;
        }
        if (good_intensity > 1) {
          good_intensity = 1;
        }
        color = Fade(COLORS[8], good_intensity);

      } else if (zero_cost_time_remaining < 0) {
        // Map zero_cost_time_remaining to a yellow - to - red scale
        float red_intensity =
            (float)abs(zero_cost_time_remaining) / ZERO_COST_SEARCH_TIME;
        if (red_intensity < 0) {
          red_intensity = 0;
        }
        if (red_intensity > 1) {
          red_intensity = 1;
        }
        color = (Color){(unsigned char)(255),
                        (unsigned char)(255 - red_intensity * 255),
                        (unsigned char)(0), 255};
      } else {
        color = GRAY;
      }
      DrawRectangle(grid_x + j * cell_width, i * cell_height, cell_width,
                    cell_height, color);
    }
  }

  // Draw the sensor ranges

  float ppmm = WINDOW_Y_PX * client->scale / S_BAND_MAX_RANGE;
  Vector2 center = {0, WINDOW_Y_PX * client->scale};

  // DrawRingLines(center, innerRadius, outerRadius, startAngle, endAngle,
  // (int)segments, color)
  DrawRingLines(center, S_BAND_MIN_RANGE * ppmm, S_BAND_MAX_RANGE * ppmm,
                -90.0f, 0.0f, 0.0f, COLORS[3]);
  DrawRingLines(center, X_BAND_MIN_RANGE * ppmm, X_BAND_MAX_RANGE * ppmm,
                -90.0f, 0.0f, 0.0f, COLORS[3]);

  int action = env->actions[0];
  if (action == SEARCH) {
    DrawText("SEARCH", 300, 10, 20, WHITE);
  } else if (action <= env->max_trackers) {
    DrawText(TextFormat("TRACK %d", action), 300, 10, 20, WHITE);
    action -= 1;
    if (env->targets[action].is_tracked) {
      DrawCircle(env->targets[action].x * ppmm,
                 WINDOW_Y_PX * client->scale - env->targets[action].y * ppmm,
                 10 * client->scale, COLORS[6]);
    }
  }

  // Draw the targets
  for (int i = 0; i < env->max_trackers; i++) {
    if (env->targets[i].is_active) {
      float x = env->targets[i].x * ppmm;
      float y = env->targets[i].y * ppmm;
      Vector2 position = {x, WINDOW_Y_PX * client->scale - y};
      float heading =
          atan2f(env->targets[i].y_velocity, env->targets[i].x_velocity);
      float arrow_size = 2 * client->scale;
      Vector2 points[3] = {
          (Vector2){position.x + arrow_size * cosf(heading + PI / 2),
                    position.y + arrow_size * sinf(heading + PI / 2)},
          (Vector2){position.x + 3 * arrow_size * cosf(heading),
                    position.y + 3 * arrow_size * sinf(heading)},
          (Vector2){position.x + arrow_size * cosf(heading - PI / 2),
                    position.y + arrow_size * sinf(heading - PI / 2)},
      };
      if (env->targets[i].is_tracked) {
        if (env->targets[i].priority >= 2) {
          DrawTriangle(points[0], points[1], points[2], COLORS[1]);
        } else if (env->targets[i].priority >= 1) {
          DrawTriangle(points[0], points[1], points[2], COLORS[2]);
        } else {
          DrawTriangle(points[0], points[1], points[2], COLORS[8]);
        }
      } else {
        // This draws untracked targets
        // DrawTriangle(points[0], points[1], points[2], COLORS[7]);
      }
    }
  }

  EndDrawing();
}
