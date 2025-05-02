#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include <float.h>
#include "raylib.h"

// Constant defs
#define MAX_ENEMIES 10
#define OBSERVATIONS_MAX_SIZE (8 + (5 * MAX_ENEMIES) + 9 + 1)
#define TARGET_FPS 60 // Used to calculate wiggle spawn frequency
#define LOG_BUFFER_SIZE 4096
#define SCREEN_WIDTH 152
#define SCREEN_HEIGHT 210
#define PLAYABLE_AREA_TOP 0
#define PLAYABLE_AREA_BOTTOM 154
#define PLAYABLE_AREA_LEFT 0
#define PLAYABLE_AREA_RIGHT 152
#define ACTION_HEIGHT (PLAYABLE_AREA_BOTTOM - PLAYABLE_AREA_TOP)
#define CAR_WIDTH 16
#define CAR_HEIGHT 11
#define CRASH_NOOP_DURATION_CAR_VS_CAR 90 // How long controls are disabled after car v car collision
#define CRASH_NOOP_DURATION_CAR_VS_ROAD 20 // How long controls are disabled after car v road edge collision
#define INITIAL_CARS_TO_PASS 200
#define VANISHING_POINT_Y 52
#define VANISHING_POINT_TRANSITION_SPEED 1.0f
#define CURVE_TRANSITION_SPEED 0.05f
#define LOGICAL_VANISHING_Y (VANISHING_POINT_Y + 12)  // Separate logical vanishing point for cars disappearing
#define PLAYER_MAX_Y (ACTION_HEIGHT - CAR_HEIGHT) // Max y is car length from bottom
#define PLAYER_MIN_Y (ACTION_HEIGHT - CAR_HEIGHT - 9) // Min y is 2 car lengths from bottom
#define ACCELERATION_RATE 0.2f
#define DECELERATION_RATE 0.01f
#define MIN_SPEED -2.5f
#define MAX_SPEED 7.5f
#define ENEMY_CAR_SPEED 0.1f 

const float MAX_SPAWN_INTERVALS[] = {0.5f, 0.25f, 0.4f};

// Times of day logic
#define NUM_BACKGROUND_TRANSITIONS 16
// Seconds spent in each time of day
static const float BACKGROUND_TRANSITION_TIMES[] = {
    20.0f, 40.0f, 60.0f, 100.0f, 108.0f, 114.0f, 116.0f, 120.0f,
    124.0f, 130.0f, 134.0f, 138.0f, 170.0f, 198.0f, 214.0f, 232.0f
};

// Curve constants
#define CURVE_STRAIGHT 0
#define CURVE_LEFT -1
#define CURVE_RIGHT 1
#define NUM_LANES 3
#define CURVE_VANISHING_POINT_SHIFT 55.0f
#define CURVE_PLAYER_SHIFT_FACTOR 0.025f // Moves player car towards outside edge of curves

// Curve wiggle effect timing and amplitude
#define WIGGLE_AMPLITUDE 10.0f // Maximum 'bump-in' offset in pixels
#define WIGGLE_SPEED 10.1f // Speed at which the wiggle moves down the screen
#define WIGGLE_LENGTH 26.0f // Vertical length of the wiggle effect


// Rendering constants
#define SCORE_DIGITS 5
#define CARS_DIGITS  4
#define DIGIT_WIDTH 8
#define DIGIT_HEIGHT 9

// Magic numbers - don't change
// The below block is specific to resolution 152x210px
#define INITIAL_PLAYER_X 69.0f // Adjusted from 77.0f
#define PLAYER_MIN_X 57.5f     // Adjusted from 65.5f
#define PLAYER_MAX_X 83.5f     // Adjusted from 91.5f
#define VANISHING_POINT_X 78.0f // Adjusted from 86
#define VANISHING_POINT_X_LEFT 102.0f // Adjusted from 110.0f
#define VANISHING_POINT_X_RIGHT 54.0f // Adjusted from 62.0f
#define ROAD_LEFT_OFFSET 46.0f  // Adjusted from 50.0f
#define ROAD_RIGHT_OFFSET 47.0f // Adjusted from 51.0f

#define CONTINUOUS_SCALE (1) // Scale enemy cars continuously with y?

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float reward;
    float step_rew_car_passed_no_crash;
    float crashed_penalty;
    float passed_cars;
    float passed_by_enemy;
    float cars_to_pass;
    float days_completed;
    float days_failed;
    float collisions_player_vs_car;
    float collisions_player_vs_road;
    float n;
};

// Car struct for enemy cars
typedef struct Car {
    int lane;   // Lane index: 0=left lane, 1=mid, 2=right lane
    float x;    // Current x position
    float y;    // Current y position
    float last_x; // x post last step
    float last_y; // y post last step
    int passed; // Flag to indicate if car has been passed by player
    int colorIndex; // Car color idx (0-5)
} Car;

// Rendering struct
typedef struct GameState {
    Texture2D spritesheet;
    RenderTexture2D renderTarget; // for scaling up render
    // Indices into asset_map[] for various assets
    int backgroundIndices[16];
    int mountainIndices[16];
    int digitIndices[11];      // 0-9 and "CAR"
    int greenDigitIndices[10]; // Green digits 0-9
    int yellowDigitIndices[10]; // Yellow digits 0-9
    // Enemy car indices: [color][tread]
    int enemyCarIndices[6][2];
    int enemyCarNightTailLightsIndex;
    int enemyCarNightFogTailLightsIndex;
    // Player car indices
    int playerCarLeftTreadIndex;
    int playerCarRightTreadIndex;
    // Flag indices
    int levelCompleteFlagLeftIndex;
    int levelCompleteFlagRightIndex;
    // For car animation
    float carAnimationTimer;
    float carAnimationInterval;
    unsigned char showLeftTread;
    float mountainPosition; // Position of the mountain texture
    // Variables for alternating flags
    unsigned char victoryAchieved;
    int flagTimer;
    unsigned char showLeftFlag; // true shows left flag, false shows right flag
    int victoryDisplayTimer;    // Timer for how long victory effects have been displayed
    // Variables for scrolling yellow digits
    float yellowDigitOffset; // Offset for scrolling effect
    int yellowDigitCurrent;  // Current yellow digit being displayed
    int yellowDigitNext;     // Next yellow digit to scroll in
    // Variables for scrolling digits
    float scoreDigitOffsets[SCORE_DIGITS];   // Offset for scrolling effect for each digit
    int scoreDigitCurrents[SCORE_DIGITS];    // Current digit being displayed for each position
    int scoreDigitNexts[SCORE_DIGITS];       // Next digit to scroll in for each position
    unsigned char scoreDigitScrolling[SCORE_DIGITS];  // Scrolling state for each digit
    int scoreTimer; // Timer to control score increment
    int day;
    int carsLeftGameState;
    int score; // Score for scoreboard rendering
    // Background state vars
    int currentBackgroundIndex;
    int previousBackgroundIndex;
    float elapsedTime;
    // Variable needed from Enduro to maintain separation
    float speed;
    float min_speed;
    float max_speed;
    int current_curve_direction;
    float current_curve_factor;
    float player_x;
    float player_y;
    float initial_player_x;
    float vanishing_point_x;
    float t_p;
    unsigned char dayCompleted;
} GameState;

typedef struct Client Client;
typedef struct Enduro Enduro;
// Game environment struct
typedef struct Enduro {
    Client* client;
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    size_t obs_size;
    int num_envs;
    float width;
    float height;
    float car_width;
    float car_height;
    int max_enemies;
    float elapsedTimeEnv;
    int initial_cars_to_pass;
    float min_speed;
    float max_speed;
    float player_x;
    float player_y;
    float speed;
    int score;
    int day;
    int lane;
    int tick;
    int numEnemies;
    int carsToPass;
    float collision_cooldown_car_vs_car; // Timer for car vs car collisions
    float collision_cooldown_car_vs_road; // Timer for car vs road edge collisions
    int drift_direction; // Which way player car drifts whilst noops after crash w/ other car
    float action_height;
    Car enemyCars[MAX_ENEMIES];
    float initial_player_x;
    float road_scroll_offset;
    // Road curve variables
    int current_curve_stage;
    int steps_in_current_stage;
    int current_curve_direction; // 1: Right, -1: Left, 0: Straight
    float current_curve_factor;
    float target_curve_factor;
    float current_step_threshold;
    float target_vanishing_point_x;     // Next curve direction vanishing point
    float current_vanishing_point_x;    // Current interpolated vanishing point
    float base_target_vanishing_point_x; // Target for the base vanishing point
    float vanishing_point_x;
    float base_vanishing_point_x;
    float t_p;
    // Roadside wiggle effect
    float wiggle_y;            // Current y position of the wiggle
    float wiggle_speed;        // Speed at which the wiggle moves down the screen
    float wiggle_length;       // Vertical length of the wiggle effect
    float wiggle_amplitude;    // How far into road wiggle extends
    unsigned char wiggle_active;        // Whether the wiggle is active
    // Player car acceleration
    int currentGear;
    float gearSpeedThresholds[4]; // Speeds at which gear changes occur
    float gearAccelerationRates[4]; // Acceleration rates per gear
    float gearTimings[4]; // Time durations per gear
    float gearElapsedTime; // Time spent in current gear
    // Enemy spawning
    float enemySpawnTimer;
    float enemySpawnInterval; // Spawn interval based on current stage
    // Enemy movement speed
    float enemySpeed;
    // Day completed (victory) variables
    unsigned char dayCompleted;
    // Logging
    float last_road_left;
    float last_road_right;
    int closest_edge_lane;
    int last_spawned_lane;
    // Vars that increment log struct
    float tracking_episode_return;
    float tracking_episode_length;
    float tracking_score;
    float tracking_reward;
    float tracking_step_rew_car_passed_no_crash;
    float tracking_crashed_penalty;
    float tracking_passed_cars;
    float tracking_passed_by_enemy;
    float tracking_cars_to_pass;
    float tracking_days_completed;
    float tracking_days_failed;
    float tracking_collisions_player_vs_car;
    float tracking_collisions_player_vs_road;

    // Rendering
    float parallaxFactor;
    // Variables for time of day
    float dayTransitionTimes[NUM_BACKGROUND_TRANSITIONS];
    int dayTimeIndex;
    int currentDayTimeIndex;
    int previousDayTimeIndex;
    // RNG
    unsigned int rng_state;
    int reset_count;
    // Rewards
    // Reward flag for stepwise rewards if car passed && !crashed
    // Effectively spreads out reward for passing cars
    unsigned char car_passed_no_crash_active;
    float step_rew_car_passed_no_crash;
    float crashed_penalty;
    int continuous;

    int seed;
} Enduro;

// Action enumeration
typedef enum {
    ACTION_NOOP = 0,
    ACTION_FIRE = 1,
    ACTION_RIGHT = 2,
    ACTION_LEFT = 3,
    ACTION_DOWN = 4,
    ACTION_DOWNRIGHT = 5,
    ACTION_DOWNLEFT = 6,
    ACTION_RIGHTFIRE = 7,
    ACTION_LEFTFIRE = 8,
} Action;

Rectangle asset_map[] = {
    (Rectangle){ 328, 15, 152, 210 },  // 0_bg
    (Rectangle){ 480, 15, 152, 210 },  // 1_bg
    (Rectangle){ 632, 15, 152, 210 },  // 2_bg
    (Rectangle){ 784, 15, 152, 210 },  // 3_bg
    (Rectangle){ 0, 225, 152, 210 },  // 4_bg
    (Rectangle){ 152, 225, 152, 210 },  // 5_bg
    (Rectangle){ 304, 225, 152, 210 },  // 6_bg
    (Rectangle){ 456, 225, 152, 210 },  // 7_bg
    (Rectangle){ 608, 225, 152, 210 },  // 8_bg
    (Rectangle){ 760, 225, 152, 210 },  // 9_bg
    (Rectangle){ 0, 435, 152, 210 },  // 10_bg
    (Rectangle){ 152, 435, 152, 210 },  // 11_bg
    (Rectangle){ 304, 435, 152, 210 },  // 12_bg
    (Rectangle){ 456, 435, 152, 210 },  // 13_bg
    (Rectangle){ 608, 435, 152, 210 },  // 14_bg
    (Rectangle){ 760, 435, 152, 210 },  // 15_bg
    (Rectangle){ 0, 0, 100, 6 },  // 0_mtns
    (Rectangle){ 100, 0, 100, 6 },  // 1_mtns
    (Rectangle){ 200, 0, 100, 6 },  // 2_mtns
    (Rectangle){ 300, 0, 100, 6 },  // 3_mtns
    (Rectangle){ 400, 0, 100, 6 },  // 4_mtns
    (Rectangle){ 500, 0, 100, 6 },  // 5_mtns
    (Rectangle){ 600, 0, 100, 6 },  // 6_mtns
    (Rectangle){ 700, 0, 100, 6 },  // 7_mtns
    (Rectangle){ 800, 0, 100, 6 },  // 8_mtns
    (Rectangle){ 0, 6, 100, 6 },  // 9_mtns
    (Rectangle){ 100, 6, 100, 6 },  // 10_mtns
    (Rectangle){ 200, 6, 100, 6 },  // 11_mtns
    (Rectangle){ 300, 6, 100, 6 },  // 12_mtns
    (Rectangle){ 400, 6, 100, 6 },  // 13_mtns
    (Rectangle){ 500, 6, 100, 6 },  // 14_mtns
    (Rectangle){ 600, 6, 100, 6 },  // 15_mtns
    (Rectangle){ 700, 6, 8, 9 },  // digits_0
    (Rectangle){ 708, 6, 8, 9 },  // digits_1
    (Rectangle){ 716, 6, 8, 9 },  // digits_2
    (Rectangle){ 724, 6, 8, 9 },  // digits_3
    (Rectangle){ 732, 6, 8, 9 },  // digits_4
    (Rectangle){ 740, 6, 8, 9 },  // digits_5
    (Rectangle){ 748, 6, 8, 9 },  // digits_6
    (Rectangle){ 756, 6, 8, 9 },  // digits_7
    (Rectangle){ 764, 6, 8, 9 },  // digits_8
    (Rectangle){ 772, 6, 8, 9 },  // digits_9
    (Rectangle){ 780, 6, 8, 9 },  // digits_car
    (Rectangle){ 788, 6, 8, 9 },  // green_digits_0
    (Rectangle){ 796, 6, 8, 9 },  // green_digits_1
    (Rectangle){ 804, 6, 8, 9 },  // green_digits_2
    (Rectangle){ 812, 6, 8, 9 },  // green_digits_3
    (Rectangle){ 820, 6, 8, 9 },  // green_digits_4
    (Rectangle){ 828, 6, 8, 9 },  // green_digits_5
    (Rectangle){ 836, 6, 8, 9 },  // green_digits_6
    (Rectangle){ 844, 6, 8, 9 },  // green_digits_7
    (Rectangle){ 852, 6, 8, 9 },  // green_digits_8
    (Rectangle){ 860, 6, 8, 9 },  // green_digits_9
    (Rectangle){ 932, 6, 8, 9 },  // yellow_digits_0
    (Rectangle){ 0, 15, 8, 9 },  // yellow_digits_1
    (Rectangle){ 8, 15, 8, 9 },  // yellow_digits_2
    (Rectangle){ 16, 15, 8, 9 },  // yellow_digits_3
    (Rectangle){ 24, 15, 8, 9 },  // yellow_digits_4
    (Rectangle){ 32, 15, 8, 9 },  // yellow_digits_5
    (Rectangle){ 40, 15, 8, 9 },  // yellow_digits_6
    (Rectangle){ 48, 15, 8, 9 },  // yellow_digits_7
    (Rectangle){ 56, 15, 8, 9 },  // yellow_digits_8
    (Rectangle){ 64, 15, 8, 9 },  // yellow_digits_9
    (Rectangle){ 72, 15, 16, 11 },  // enemy_car_blue_left_tread
    (Rectangle){ 88, 15, 16, 11 },  // enemy_car_blue_right_tread
    (Rectangle){ 104, 15, 16, 11 },  // enemy_car_gold_left_tread
    (Rectangle){ 120, 15, 16, 11 },  // enemy_car_gold_right_tread
    (Rectangle){ 168, 15, 16, 11 },  // enemy_car_pink_left_tread
    (Rectangle){ 184, 15, 16, 11 },  // enemy_car_pink_right_tread
    (Rectangle){ 200, 15, 16, 11 },  // enemy_car_salmon_left_tread
    (Rectangle){ 216, 15, 16, 11 },  // enemy_car_salmon_right_tread
    (Rectangle){ 232, 15, 16, 11 },  // enemy_car_teal_left_tread
    (Rectangle){ 248, 15, 16, 11 },  // enemy_car_teal_right_tread
    (Rectangle){ 264, 15, 16, 11 },  // enemy_car_yellow_left_tread
    (Rectangle){ 280, 15, 16, 11 },  // enemy_car_yellow_right_tread
    (Rectangle){ 136, 15, 16, 11 },  // enemy_car_night_fog_tail_lights
    (Rectangle){ 152, 15, 16, 11 },  // enemy_car_night_tail_lights
    (Rectangle){ 296, 15, 16, 11 },  // player_car_left_tread
    (Rectangle){ 312, 15, 16, 11 },  // player_car_right_tread
    (Rectangle){ 900, 6, 32, 9 },  // level_complete_flag_right
    (Rectangle){ 868, 6, 32, 9 },  // level_complete_flag_left
};

enum AssetIndices {
    ASSET_BG_0 = 0,
    ASSET_BG_1 = 1,
    ASSET_BG_2 = 2,
    ASSET_BG_3 = 3,
    ASSET_BG_4 = 4,
    ASSET_BG_5 = 5,
    ASSET_BG_6 = 6,
    ASSET_BG_7 = 7,
    ASSET_BG_8 = 8,
    ASSET_BG_9 = 9,
    ASSET_BG_10 = 10,
    ASSET_BG_11 = 11,
    ASSET_BG_12 = 12,
    ASSET_BG_13 = 13,
    ASSET_BG_14 = 14,
    ASSET_BG_15 = 15,

    ASSET_MOUNTAIN_0 = 16,
    ASSET_MOUNTAIN_1,
    ASSET_MOUNTAIN_2,
    ASSET_MOUNTAIN_3,
    ASSET_MOUNTAIN_4,
    ASSET_MOUNTAIN_5,
    ASSET_MOUNTAIN_6,
    ASSET_MOUNTAIN_7,
    ASSET_MOUNTAIN_8,
    ASSET_MOUNTAIN_9,
    ASSET_MOUNTAIN_10,
    ASSET_MOUNTAIN_11,
    ASSET_MOUNTAIN_12,
    ASSET_MOUNTAIN_13,
    ASSET_MOUNTAIN_14,
    ASSET_MOUNTAIN_15,
    ASSET_DIGITS_0 = 32,
    ASSET_DIGITS_1 = 33,
    ASSET_DIGITS_2 = 34,
    ASSET_DIGITS_3 = 35,
    ASSET_DIGITS_4 = 36,
    ASSET_DIGITS_5 = 37,
    ASSET_DIGITS_6 = 38,
    ASSET_DIGITS_7 = 39,
    ASSET_DIGITS_8 = 40,
    ASSET_DIGITS_9 = 41,
    ASSET_DIGITS_CAR = 42,
    ASSET_GREEN_DIGITS_0 = 43,
    ASSET_GREEN_DIGITS_1 = 44,
    ASSET_GREEN_DIGITS_2 = 45,
    ASSET_GREEN_DIGITS_3 = 46,
    ASSET_GREEN_DIGITS_4 = 47,
    ASSET_GREEN_DIGITS_5 = 48,
    ASSET_GREEN_DIGITS_6 = 49,
    ASSET_GREEN_DIGITS_7 = 50,
    ASSET_GREEN_DIGITS_8 = 51,
    ASSET_GREEN_DIGITS_9 = 52,
    ASSET_YELLOW_DIGITS_0 = 53,
    ASSET_YELLOW_DIGITS_1 = 54,
    ASSET_YELLOW_DIGITS_2 = 55,
    ASSET_YELLOW_DIGITS_3 = 56,
    ASSET_YELLOW_DIGITS_4 = 57,
    ASSET_YELLOW_DIGITS_5 = 58,
    ASSET_YELLOW_DIGITS_6 = 59,
    ASSET_YELLOW_DIGITS_7 = 60,
    ASSET_YELLOW_DIGITS_8 = 61,
    ASSET_YELLOW_DIGITS_9 = 62,
    ASSET_ENEMY_CAR_BLUE_LEFT_TREAD = 63,
    ASSET_ENEMY_CAR_BLUE_RIGHT_TREAD = 64,
    ASSET_ENEMY_CAR_GOLD_LEFT_TREAD = 65,
    ASSET_ENEMY_CAR_GOLD_RIGHT_TREAD = 66,
    ASSET_ENEMY_CAR_PINK_LEFT_TREAD = 67,
    ASSET_ENEMY_CAR_PINK_RIGHT_TREAD = 68,
    ASSET_ENEMY_CAR_SALMON_LEFT_TREAD = 69,
    ASSET_ENEMY_CAR_SALMON_RIGHT_TREAD = 70,
    ASSET_ENEMY_CAR_TEAL_LEFT_TREAD = 71,
    ASSET_ENEMY_CAR_TEAL_RIGHT_TREAD = 72,
    ASSET_ENEMY_CAR_YELLOW_LEFT_TREAD = 73,
    ASSET_ENEMY_CAR_YELLOW_RIGHT_TREAD = 74,
    ASSET_ENEMY_CAR_NIGHT_FOG_TAIL_LIGHTS = 75,
    ASSET_ENEMY_CAR_NIGHT_TAIL_LIGHTS = 76,
    ASSET_PLAYER_CAR_LEFT_TREAD = 77,
    ASSET_PLAYER_CAR_RIGHT_TREAD = 78,
    ASSET_LEVEL_COMPLETE_FLAG_RIGHT = 79,
    ASSET_LEVEL_COMPLETE_FLAG_LEFT = 80,
};

// Client struct
struct Client {
    float width;
    float height;
    GameState gameState;
};

// Prototypes //
// RNG
unsigned int xorshift32(unsigned int *state);

// Logging
void add_log(Enduro* env);

// Environment functions
void allocate(Enduro* env);
void init(Enduro* env);
void free_allocated(Enduro* env);
void reset_round(Enduro* env);
void c_reset(Enduro* env);
unsigned char check_collision(Enduro* env, Car* car);
int get_player_lane(Enduro* env);
float get_car_scale(float y);
void add_enemy_car(Enduro* env);
void update_time_of_day(Enduro* env);
void accelerate(Enduro* env);
void c_step(Enduro* env);
void update_road_curve(Enduro* env);
float quadratic_bezier(float bottom_x, float control_x, float top_x, float t);
float road_edge_x(Enduro* env, float y, float offset, unsigned char left);
float car_x_in_lane(Enduro* env, int lane, float y);
void compute_observations(Enduro* env);

// Client functions
Client* make_client(Enduro* env);
void close_client(Client* client);
void render_car(GameState* gameState);

// GameState rendering functions
void initRaylib(GameState* gameState); 
void loadTextures(GameState* gameState);
void updateCarAnimation(GameState* gameState);
void updateScoreboard(GameState* gameState);
void renderBackground(GameState* gameState);
void renderScoreboard(GameState* gameState);
void updateMountains(GameState* gameState);
void renderMountains(GameState* gameState);
void updateVictoryEffects(GameState* gameState);
void c_render(Enduro* env);
void cleanup(GameState* gameState);

// Function definitions //
// RNG
unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

void add_log(Enduro* env) {
    env->log.episode_return += env->tracking_episode_return;
    env->log.episode_length += env->tracking_episode_length;
    env->log.score += env->tracking_score;
    env->log.perf += fminf(env->tracking_days_completed/10.0f, 1.0f);
    env->log.reward += env->tracking_reward;
    env->log.step_rew_car_passed_no_crash += env->tracking_step_rew_car_passed_no_crash;
    env->log.crashed_penalty += env->tracking_crashed_penalty;
    env->log.passed_cars += env->tracking_passed_cars;
    env->log.passed_by_enemy += env->tracking_passed_by_enemy;
    env->log.cars_to_pass += env->tracking_cars_to_pass;
    env->log.days_completed += env->tracking_days_completed;
    env->log.days_failed += env->tracking_days_failed;
    env->log.collisions_player_vs_car += env->tracking_collisions_player_vs_car;
    env->log.collisions_player_vs_road += env->tracking_collisions_player_vs_road;
    
    // Episode counter
    env->log.n += 1.0f;
}

void init(Enduro* env) {
    env->rng_state = env->seed;
    env->reset_count = 0;
    env->last_road_left = -1.0f;
    env->last_road_right = -1.0f;

    if (env->rng_state == 0) { // Activate with seed==0
        // Start the environment at the beginning of the day
        env->rng_state = 0;
        env->elapsedTimeEnv = 0.0f;
        env->currentDayTimeIndex = 0;
        env->previousDayTimeIndex = NUM_BACKGROUND_TRANSITIONS;
    } else {
        // Randomize elapsed time within the day's total duration
        float total_day_duration = BACKGROUND_TRANSITION_TIMES[NUM_BACKGROUND_TRANSITIONS - 1];
        env->elapsedTimeEnv = ((float)xorshift32(&env->rng_state) / (float)UINT32_MAX) * total_day_duration;

        // Determine the current time index
        env->currentDayTimeIndex = 0;
        for (int i = 0; i < NUM_BACKGROUND_TRANSITIONS - 1; i++) {
            if (env->elapsedTimeEnv >= env->dayTransitionTimes[i] &&
                env->elapsedTimeEnv < env->dayTransitionTimes[i + 1]) {
                env->currentDayTimeIndex = i;
                break;
            }
        }

        // Handle the last interval
        if (env->elapsedTimeEnv >= BACKGROUND_TRANSITION_TIMES[NUM_BACKGROUND_TRANSITIONS - 1]) {
            env->currentDayTimeIndex = NUM_BACKGROUND_TRANSITIONS - 1;
        }
    }

    env->numEnemies = 0;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->enemyCars[i].lane = -1; // Default invalid lane
        env->enemyCars[i].y = 0.0f;
        env->enemyCars[i].passed = 0;
    }

    env->obs_size = OBSERVATIONS_MAX_SIZE;
    env->max_enemies = MAX_ENEMIES;
    env->score = 0;
    env->numEnemies = 0;
    env->player_x = INITIAL_PLAYER_X;
    env->player_y = PLAYER_MAX_Y;
    env->speed = MIN_SPEED;
    env->carsToPass = INITIAL_CARS_TO_PASS;
    env->width = SCREEN_WIDTH;
    env->height = SCREEN_HEIGHT;
    env->car_width = CAR_WIDTH;
    env->car_height = CAR_HEIGHT;

    memcpy(env->dayTransitionTimes, BACKGROUND_TRANSITION_TIMES, sizeof(BACKGROUND_TRANSITION_TIMES));
    
    env->tick = 0;
    env->collision_cooldown_car_vs_car = 0.0f;
    env->collision_cooldown_car_vs_road = 0.0f;
    env->action_height = ACTION_HEIGHT;
    env->elapsedTimeEnv = 0.0f;
    env->enemySpawnTimer = 0.0f;
    env->enemySpawnInterval = 0.8777f;
    env->last_spawned_lane = -1;
    env->closest_edge_lane = -1;
    env->base_vanishing_point_x = 86.0f;
    env->current_vanishing_point_x = 86.0f;
    env->target_vanishing_point_x = 86.0f;
    env->vanishing_point_x = 86.0f;
    env->initial_player_x = INITIAL_PLAYER_X;
    env->player_y = PLAYER_MAX_Y;
    env->min_speed = MIN_SPEED;
    env->enemySpeed = ENEMY_CAR_SPEED;
    env->max_speed = MAX_SPEED;
    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->day = 1;
    env->drift_direction = 0; // Means in noop, but only if crashed state
    env->crashed_penalty = 0.0f;
    env->car_passed_no_crash_active = 1;
    env->step_rew_car_passed_no_crash = 0.0f;
    env->current_curve_stage = 0; // 0: straight
    env->steps_in_current_stage = 0;
    env->current_curve_direction = CURVE_STRAIGHT;
    env->current_curve_factor = 0.0f;
    env->target_curve_factor = 0.0f;
    env->current_step_threshold = 0.0f;
    env->wiggle_y = VANISHING_POINT_Y;
    env->wiggle_speed = WIGGLE_SPEED;
    env->wiggle_length = WIGGLE_LENGTH;
    env->wiggle_amplitude = WIGGLE_AMPLITUDE;
    env->wiggle_active = true;
    env->currentGear = 0;
    env->gearElapsedTime = 0.0f;
    env->gearTimings[0] = 4.0f;
    env->gearTimings[1] = 2.5f;
    env->gearTimings[2] = 3.25f;
    env->gearTimings[3] = 1.5f;
    float totalSpeedRange = env->max_speed - env->min_speed;
    float totalTime = 0.0f;
    for (int i = 0; i < 4; i++) {
        totalTime += env->gearTimings[i];
    }
    float cumulativeSpeed = env->min_speed;
    for (int i = 0; i < 4; i++) {
        float gearTime = env->gearTimings[i];
        float gearSpeedIncrement = totalSpeedRange * (gearTime / totalTime);
        env->gearSpeedThresholds[i] = cumulativeSpeed + gearSpeedIncrement;
        env->gearAccelerationRates[i] = gearSpeedIncrement / (gearTime * TARGET_FPS);
        cumulativeSpeed = env->gearSpeedThresholds[i];
    }

    // Randomize the initial time of day for each environment
    if (env->rng_state == 0) {
        env->elapsedTimeEnv = 0;
        env->currentDayTimeIndex = 0;
        env->dayTimeIndex = 0;
        env->previousDayTimeIndex = 0;
    } else {
        float total_day_duration = BACKGROUND_TRANSITION_TIMES[15];
        env->elapsedTimeEnv = ((float)rand_r(&env->rng_state) / (float)RAND_MAX) * total_day_duration;
        env->currentDayTimeIndex = 0;
        env->dayTimeIndex = 0;
        env->previousDayTimeIndex = 0;

        // Advance currentDayTimeIndex to match randomized elapsedTimeEnv
        for (int i = 0; i < NUM_BACKGROUND_TRANSITIONS; i++) {
            if (env->elapsedTimeEnv >= env->dayTransitionTimes[i]) {
                env->currentDayTimeIndex = i;
            } else {
                break;
            }
        }

        env->previousDayTimeIndex = (env->currentDayTimeIndex > 0) ? env->currentDayTimeIndex - 1 : NUM_BACKGROUND_TRANSITIONS - 1;
    }

    env->road_scroll_offset = 1.0f;
    env->base_target_vanishing_point_x = 86.0f;
    env->t_p = 86.0f;
    env->parallaxFactor = 1.0f;
    env->dayCompleted = 0;
    env->lane = 1;
    env->terminals[0] = 0;

    // Reset rewards and logs
    env->rewards[0] = 0.0f;
    // Initialize tracking variables
    env->tracking_episode_return = 0.0f;
    env->tracking_episode_length = 0.0f;
    env->tracking_score = 0.0f;
    env->tracking_reward = 0.0f;
    env->tracking_step_rew_car_passed_no_crash = 0.0f;
    env->tracking_crashed_penalty = 0.0f;
    env->tracking_passed_cars = 0.0f;
    env->tracking_passed_by_enemy = 0.0f;
    env->tracking_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->tracking_days_completed = 0.0f;
    env->tracking_days_failed = 0.0f;
    env->tracking_collisions_player_vs_car = 0.0f;
    env->tracking_collisions_player_vs_road = 0.0f;

    env->log.episode_return = 0.0f;
    env->log.episode_length = 0.0f;
    env->log.score = 0.0f;
    env->log.reward = 0.0f;
    env->log.step_rew_car_passed_no_crash = 0.0f;
    env->log.crashed_penalty = 0.0f;
    env->log.passed_cars = 0.0f;
    env->log.passed_by_enemy = 0.0f;
    env->log.cars_to_pass = INITIAL_CARS_TO_PASS;
    env->log.days_completed = 0;
    env->log.days_failed = 0;
    env->log.collisions_player_vs_car = 0.0f;
    env->log.collisions_player_vs_road = 0.0f;
    env->log.n = 0.0f;
}

void allocate(Enduro* env) {
    env->observations = (float*)calloc(env->obs_size, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_allocated(Enduro* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

// Called when a day is failed by player
// Restarts the game at Day 1
void reset_round(Enduro* env) {
    // Preserve RNG state
    unsigned int preserved_rng_state = env->rng_state;

    // Reset most environment variables
    env->score = 0;
    env->carsToPass = INITIAL_CARS_TO_PASS;
    env->day = 1;
    env->tick = 0;
    env->numEnemies = 0;
    env->speed = env->min_speed;
    env->player_x = env->initial_player_x;
    env->player_y = PLAYER_MAX_Y;
    env->car_passed_no_crash_active = 0;
    env->step_rew_car_passed_no_crash = 0.0f;
    env->crashed_penalty = 0.0f;
    env->collision_cooldown_car_vs_car = 0.0f;
    env->collision_cooldown_car_vs_road = 0.0f;
    env->enemySpawnTimer = 0.0f;
    env->enemySpawnInterval = 0.8777f;
    env->last_spawned_lane = -1;
    env->closest_edge_lane = -1;
    env->base_vanishing_point_x = 86.0f;
    env->current_vanishing_point_x = 86.0f;
    env->target_vanishing_point_x = 86.0f;
    env->vanishing_point_x = 86.0f;
    env->initial_player_x = INITIAL_PLAYER_X;
    env->player_y = PLAYER_MAX_Y;
    env->min_speed = MIN_SPEED;
    env->enemySpeed = ENEMY_CAR_SPEED;
    env->max_speed = MAX_SPEED;
    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->day = 1;
    env->drift_direction = 0; // Means in noop, but only if crashed state
    env->crashed_penalty = 0.0f;
    env->car_passed_no_crash_active = 1;
    env->step_rew_car_passed_no_crash = 0.0f;
    env->current_curve_stage = 0; // 0: straight
    env->steps_in_current_stage = 0;
    env->current_curve_direction = CURVE_STRAIGHT;
    env->current_curve_factor = 0.0f;
    env->target_curve_factor = 0.0f;
    env->current_step_threshold = 0.0f;
    env->wiggle_y = VANISHING_POINT_Y;
    env->wiggle_speed = WIGGLE_SPEED;
    env->wiggle_length = WIGGLE_LENGTH;
    env->wiggle_amplitude = WIGGLE_AMPLITUDE;
    env->wiggle_active = true;
    env->currentGear = 0;
    env->gearElapsedTime = 0.0f;
    env->gearTimings[0] = 4.0f;
    env->gearTimings[1] = 2.5f;
    env->gearTimings[2] = 3.25f;
    env->gearTimings[3] = 1.5f;
    float totalSpeedRange = env->max_speed - env->min_speed;
    float totalTime = 0.0f;
    for (int i = 0; i < 4; i++) {
        totalTime += env->gearTimings[i];
    }
    float cumulativeSpeed = env->min_speed;
    for (int i = 0; i < 4; i++) {
        float gearTime = env->gearTimings[i];
        float gearSpeedIncrement = totalSpeedRange * (gearTime / totalTime);
        env->gearSpeedThresholds[i] = cumulativeSpeed + gearSpeedIncrement;
        env->gearAccelerationRates[i] = gearSpeedIncrement / (gearTime * TARGET_FPS);
        cumulativeSpeed = env->gearSpeedThresholds[i];
    }

    // Reset enemy cars
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->enemyCars[i].lane = -1;
        env->enemyCars[i].y = 0.0f;
        env->enemyCars[i].passed = 0;
    }

    // Reset rewards and logs
    env->rewards[0] = 0.0f;
    env->tracking_episode_return = 0.0f;
    env->tracking_episode_length = 0.0f;
    env->tracking_score = 0.0f;
    env->tracking_reward = 0.0f;
    env->tracking_step_rew_car_passed_no_crash = 0.0f;
    env->tracking_crashed_penalty = 0.0f;
    env->tracking_passed_cars = 0.0f;
    env->tracking_passed_by_enemy = 0.0f;
    env->tracking_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->tracking_collisions_player_vs_car = 0.0f;
    env->tracking_collisions_player_vs_road = 0.0f;   

    // Restore preserved RNG state to maintain reproducibility
    env->rng_state = preserved_rng_state;

    // Restart the environment at the beginning of the day
    env->elapsedTimeEnv = 0.0f;
    env->currentDayTimeIndex = 0;
    env->previousDayTimeIndex = NUM_BACKGROUND_TRANSITIONS - 1;

    // Reset flags and transient states
    env->dayCompleted = 0;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
}

// Reset all init vars; only called once after init
void c_reset(Enduro* env) {
    // No random after first reset
    int reset_seed = (env->reset_count == 0) ? xorshift32(&env->rng_state) : 0;
    env->rng_state = reset_seed;
    init(env);
    env->reset_count += 1;
}

unsigned char check_collision(Enduro* env, Car* car) {
    // Compute the scale factor based on vanishing point reference
    float depth = (car->y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float scale = fmax(0.1f, 0.9f * depth);
    float car_width = CAR_WIDTH * scale;
    float car_height = CAR_HEIGHT * scale;
    float car_center_x = car_x_in_lane(env, car->lane, car->y);
    float car_x = car_center_x - car_width / 2.0f;
    return !(env->player_x > car_x + car_width
            || env->player_x + CAR_WIDTH < car_x
            || env->player_y > car->y + car_height
            || env->player_y + CAR_HEIGHT < car->y);
}

// Determines which of the 3 lanes the player's car is in
int get_player_lane(Enduro* env) {
    float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
    float offset = (env->player_x - env->initial_player_x) * 0.5f;
    float left_edge = road_edge_x(env, env->player_y, offset, true);
    float right_edge = road_edge_x(env, env->player_y, offset, false);
    float lane_width = (right_edge - left_edge) / 3.0f;
    env->lane = (int)((player_center_x - left_edge) / lane_width);
    if (env->lane < 0) env->lane = 0;
    if (env->lane > 2) env->lane = 2;
    return env->lane;
}

float get_car_scale(float y) {
    float depth = (y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    return fmaxf(0.1f, 0.9f * depth);
}

void add_enemy_car(Enduro* env) {
    if (env->numEnemies >= MAX_ENEMIES) {
        return;
    }

    int player_lane = get_player_lane(env);
    int possible_lanes[NUM_LANES];
    int num_possible_lanes = 0;

    // Determine the furthest lane from the player
    int furthest_lane;
    if (player_lane == 0) {
        furthest_lane = 2;
    } else if (player_lane == 2) {
        furthest_lane = 0;
    } else {
        // Player is in the middle lane
        // Decide based on player's position relative to the road center
        float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
        float road_center_x = (road_edge_x(env, env->player_y, 0, true) +
                            road_edge_x(env, env->player_y, 0, false)) / 2.0f;
        if (player_center_x < road_center_x) {
            furthest_lane = 2; // Player is on the left side, choose rightmost lane
        } else {
            furthest_lane = 0; // Player is on the right side, choose leftmost lane
        }
    }

    if (env->speed <= 0.0f) {
        // Only spawn in the lane furthest from the player
        possible_lanes[num_possible_lanes++] = furthest_lane;
    } else {
        for (int i = 0; i < NUM_LANES; i++) {
            possible_lanes[num_possible_lanes++] = i;
        }
    }

    if (num_possible_lanes == 0) {
        return; // Rare
    }

    // Randomly select a lane
    int lane = possible_lanes[rand() % num_possible_lanes];
    // Preferentially spawn in the last_spawned_lane 30% of the time
    if (rand() % 100 < 60 && env->last_spawned_lane != -1) {
        lane = env->last_spawned_lane;
    }
    env->last_spawned_lane = lane;
    // Init car
    Car car = {
        .lane = lane,
        .x = car_x_in_lane(env, lane, VANISHING_POINT_Y),
        .y = (env->speed > 0.0f) ? VANISHING_POINT_Y + 10.0f : PLAYABLE_AREA_BOTTOM + CAR_HEIGHT,
        .last_x = car_x_in_lane(env, lane, VANISHING_POINT_Y),
        .last_y = VANISHING_POINT_Y,
        .passed = false,
        .colorIndex = rand() % 6
    };
    // Ensure minimum spacing between cars in the same lane
    float depth = (car.y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float scale = fmax(0.1f, 0.9f * depth + 0.1f);
    float scaled_car_length = CAR_HEIGHT * scale;
    // Randomize min spacing between 1.0f and 6.0f car lengths
    float dynamic_spacing_factor = (rand() / (float)RAND_MAX) * 6.0f + 0.5f;
    float min_spacing = dynamic_spacing_factor * scaled_car_length;
    for (int i = 0; i < env->numEnemies; i++) {
        Car* existing_car = &env->enemyCars[i];
        if (existing_car->lane != car.lane) {
            continue;
        }
        float y_distance = fabs(existing_car->y - car.y);
        if (y_distance < min_spacing) {
            return; // Too close, do not spawn this car
        }
    }
    // Ensure not occupying all lanes within vertical range of 6 car lengths
    float min_vertical_range = 6.0f * CAR_HEIGHT;
    int lanes_occupied = 0;
    unsigned char lane_occupied[NUM_LANES] = { false };
    for (int i = 0; i < env->numEnemies; i++) {
        Car* existing_car = &env->enemyCars[i];
        float y_distance = fabs(existing_car->y - car.y);
        if (y_distance < min_vertical_range) {
            lane_occupied[existing_car->lane] = true;
        }
    }
    for (int i = 0; i < NUM_LANES; i++) {
        if (lane_occupied[i]) lanes_occupied++;
    }
    if (lanes_occupied >= NUM_LANES - 1 && !lane_occupied[lane]) {
        return;
    }
    env->enemyCars[env->numEnemies++] = car;
}

void update_time_of_day(Enduro* env) {
    float elapsedTime = env->elapsedTimeEnv;
    float totalDuration = env->dayTransitionTimes[15];

    if (elapsedTime >= totalDuration) {
        elapsedTime -= totalDuration;
        env->elapsedTimeEnv = elapsedTime; // Reset elapsed time
        env->dayTimeIndex = 0;
    }

    env->previousDayTimeIndex = env->currentDayTimeIndex;

    while (env->dayTimeIndex < 15 &&
           elapsedTime >= env->dayTransitionTimes[env->dayTimeIndex]) {
        env->dayTimeIndex++;
    }
    env->currentDayTimeIndex = env->dayTimeIndex % 16;
}

void clamp_speed(Enduro* env) {
    if (env->speed < env->min_speed || env->speed > env->max_speed) {
        env->speed = fmaxf(env->min_speed, fminf(env->speed, env->max_speed));
    }
}

void clamp_gear(Enduro* env) {
    if (env->currentGear < 0 || env->currentGear > 3) {
        env->currentGear = 0;
    }
}

void accelerate(Enduro* env) {
    clamp_speed(env);
    clamp_gear(env);

    if (env->speed < env->max_speed) {
        // Gear transition
        if (env->speed >= env->gearSpeedThresholds[env->currentGear] && env->currentGear < 3) {
            env->currentGear++;
            env->gearElapsedTime = 0.0f;
        }

        // Calculate new speed
        float accel = env->gearAccelerationRates[env->currentGear];
        float multiplier = (env->currentGear == 0) ? 4.0f : 2.0f;
        env->speed += accel * multiplier;

        clamp_speed(env);

        // Cap speed to gear threshold
        if (env->speed > env->gearSpeedThresholds[env->currentGear]) {
            env->speed = env->gearSpeedThresholds[env->currentGear];
        }
    }
    clamp_speed(env);
}

void c_step(Enduro* env) {  
    env->rewards[0] = 0.0f;
    env->elapsedTimeEnv += (1.0f / TARGET_FPS);
    update_time_of_day(env);
    update_road_curve(env);
    env->tracking_episode_length += 1;
    env->terminals[0] = 0;
    env->road_scroll_offset += env->speed;

    // Update enemy car positions
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        // Compute movement speed adjusted for scaling
        float scale = get_car_scale(car->y);
        float movement_speed = env->speed * scale * 0.75f;
        // Update car position
        car->y += movement_speed;
    }

    // Calculate road edges
    float road_left = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;

    env->last_road_left = road_left;
    env->last_road_right = road_right;

    // Reduced handling on snow
    unsigned char isSnowStage = (env->currentDayTimeIndex == 3);
    float movement_amount = 0.5f; // Default
    if (isSnowStage) {
        movement_amount = 0.3f; // Snow
    }

    // Player movement logic == action space (Discrete[9])
    if (env->collision_cooldown_car_vs_car <= 0 && env->collision_cooldown_car_vs_road <= 0) {
        int act = env->actions[0];
        movement_amount = (((env->speed - env->min_speed) / (env->max_speed - env->min_speed)) + 0.3f); // Magic number
        switch (act) {
            case ACTION_NOOP:
                break;
            case ACTION_FIRE:
                accelerate(env);
                break;
            case ACTION_RIGHT:
                env->player_x += movement_amount;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_LEFT:
                env->player_x -= movement_amount;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
            case ACTION_DOWN:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                break;
            case ACTION_DOWNRIGHT:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                env->player_x += movement_amount;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_DOWNLEFT:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                env->player_x -= movement_amount;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
            case ACTION_RIGHTFIRE:
                accelerate(env);
                env->player_x += movement_amount;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_LEFTFIRE:
                accelerate(env);
                env->player_x -= movement_amount;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
        
        // Reset crashed_penalty
        env->crashed_penalty = 0.0f;
        }

    } else {

        if (env->collision_cooldown_car_vs_car > 0) {
            env->collision_cooldown_car_vs_car -= 1;
            env->crashed_penalty = -0.01f;
        }
        if (env->collision_cooldown_car_vs_road > 0) {
            env->collision_cooldown_car_vs_road -= 1;
            env->crashed_penalty = -0.01f;
        }

        // Drift towards furthest road edge
        if (env->drift_direction == 0) { // drift_direction is 0 when noop starts
            env->drift_direction = (env->player_x > (road_left + road_right) / 2) ? -1 : 1;
            // Remove enemy cars in middle lane and lane player is drifting towards
            // only if they are behind the player (y > player_y) to avoid crashes
            for (int i = 0; i < env->numEnemies; i++) {
                Car* car = &env->enemyCars[i];

                if ((car->lane == 1 || car->lane == env->lane + env->drift_direction) && (car->y > env->player_y)) {
                    for (int j = i; j < env->numEnemies - 1; j++) {
                        env->enemyCars[j] = env->enemyCars[j + 1];
                    }
                    env->numEnemies--;
                    i--;
                }
            }
        }
        // Drift distance per step
        if (env->collision_cooldown_car_vs_road > 0) {
            env->player_x += env->drift_direction * 0.12f;
        } else {
        env->player_x += env->drift_direction * 0.25f;
        }
    }

    // Update player's lane
    env->lane = get_player_lane(env);

    // Compute is_lane_empty and nearest_car_distance
    float nearest_car_distance[NUM_LANES];
    bool is_lane_empty[NUM_LANES];

    float MAX_DISTANCE = PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y; // Maximum possible distance

    for (int l = 0; l < NUM_LANES; l++) {
        nearest_car_distance[l] = MAX_DISTANCE;
        is_lane_empty[l] = true;
    }

    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        if (car->lane >= 0 && car->lane < NUM_LANES && car->y < env->player_y) {
            float distance = env->player_y - car->y;
            if (distance < nearest_car_distance[car->lane]) {
                nearest_car_distance[car->lane] = distance;
                is_lane_empty[car->lane] = false;
            }
        }
    }

    // Give a reward if the player's lane is empty in front
    float reward_amount = 0.05f; // Empty lane reward
    if (is_lane_empty[env->lane]) {
        env->rewards[0] += reward_amount;
    }

    // Road curve/vanishing point movement logic
    // Adjust player's x position based on the current curve
    float curve_shift = -env->current_curve_factor * CURVE_PLAYER_SHIFT_FACTOR * fabs(env->speed);
    env->player_x += curve_shift;
    // Clamp player x position to within road edges
    if (env->player_x < road_left) env->player_x = road_left;
    if (env->player_x > road_right) env->player_x = road_right;
    // Update player's horizontal position ratio, t_p
    float t_p = (env->player_x - PLAYER_MIN_X) / (PLAYER_MAX_X - PLAYER_MIN_X);
    t_p = fmaxf(0.0f, fminf(1.0f, t_p));
    env->t_p = t_p;
    // Base vanishing point based on player's horizontal movement (without curve)
    env->base_vanishing_point_x = VANISHING_POINT_X_LEFT - t_p * (VANISHING_POINT_X_LEFT - VANISHING_POINT_X_RIGHT);
    // Adjust vanishing point based on current curve
    float curve_vanishing_point_shift = env->current_curve_direction * CURVE_VANISHING_POINT_SHIFT;
    env->vanishing_point_x = env->base_vanishing_point_x + curve_vanishing_point_shift;

    // After curve shift        
    // Wiggle logic
    if (env->wiggle_active) {
        float min_wiggle_period = 5.8f; // Slow wiggle period at min speed
        float max_wiggle_period = 0.3f; // Fast wiggle period at max speed
        // Apply non-linear scaling for wiggle period using square root
        float speed_normalized = (env->speed - env->min_speed) / (env->max_speed - env->min_speed);
        speed_normalized = fmaxf(0.0f, fminf(1.0f, speed_normalized));  // Clamp between 0 and 1
        // Adjust wiggle period with non-linear scale
        float current_wiggle_period = min_wiggle_period - powf(speed_normalized, 0.25) * (min_wiggle_period - max_wiggle_period);
        // Calculate wiggle speed based on adjusted wiggle period
        env->wiggle_speed = (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y) / (current_wiggle_period * TARGET_FPS);
        // Update wiggle position
        env->wiggle_y += env->wiggle_speed;
        // Reset wiggle when it reaches the bottom
        if (env->wiggle_y > PLAYABLE_AREA_BOTTOM) {
            env->wiggle_y = VANISHING_POINT_Y;
        }
    }

    // Player car moves forward slightly according to speed
    // Update player y position based on speed
    env->player_y = PLAYER_MAX_Y - (env->speed - env->min_speed) / (env->max_speed - env->min_speed) * (PLAYER_MAX_Y - PLAYER_MIN_Y);
    // Clamp player_y to measured range
    if (env->player_y < PLAYER_MIN_Y) env->player_y = PLAYER_MIN_Y;
    if (env->player_y > PLAYER_MAX_Y) env->player_y = PLAYER_MAX_Y;

    // Check for and handle collisions between player and road edges
    if (env->player_x <= road_left || env->player_x >= road_right) {
        env->tracking_collisions_player_vs_road++;
        env->rewards[0] -= 0.5f;
        env->speed = fmaxf((env->speed - 1.25f), MIN_SPEED);
        env->collision_cooldown_car_vs_road = CRASH_NOOP_DURATION_CAR_VS_ROAD;
        env->drift_direction = 0; // Reset drift direction, has priority over car collisions
        env->player_x = fmaxf(road_left + 1, fminf(road_right - 1, env->player_x));        
    }

    // Enemy car logic
    for (int i = 0; i < env->numEnemies; i++) {    
        Car* car = &env->enemyCars[i];

        // Remove off-screen cars that move below the screen
        if (car->y > PLAYABLE_AREA_BOTTOM + CAR_HEIGHT * 5) {
            // Remove car from array if it moves below the screen
            for (int j = i; j < env->numEnemies - 1; j++) {
                env->enemyCars[j] = env->enemyCars[j + 1];
            }
            env->numEnemies--;
            i--;
            continue;
        }

        // Remove cars that reach or surpass the logical vanishing point if moving up (player speed negative)
        if (env->speed < 0 && car->y <= LOGICAL_VANISHING_Y) {
            // Remove car from array if it reaches the logical vanishing point if moving down (player speed positive)
            for (int j = i; j < env->numEnemies - 1; j++) {
                env->enemyCars[j] = env->enemyCars[j + 1];
            }
            env->numEnemies--;
            i--;
            continue;
        }
    
        // If the car is behind the player and speed <= 0, move it to the furthest lane
        if (env->speed <= 0 && car->y >= env->player_y + CAR_HEIGHT) {
            // Determine the furthest lane
            int furthest_lane;
            int player_lane = get_player_lane(env);
            if (player_lane == 0) {
                furthest_lane = 2;
            } else if (player_lane == 2) {
                furthest_lane = 0;
            } else {
                // Player is in the middle lane
                // Decide based on player's position relative to the road center
                float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
                float road_center_x = (road_edge_x(env, env->player_y, 0, true) +
                                    road_edge_x(env, env->player_y, 0, false)) / 2.0f;
                if (player_center_x < road_center_x) {
                    furthest_lane = 2; // Player is on the left side
                } else {
                    furthest_lane = 0; // Player is on the right side
                }
            }
            car->lane = furthest_lane;
            continue;
        }

        // Check for passing logic **only if not on collision cooldown**
        if (env->speed > 0 && car->last_y < env->player_y + CAR_HEIGHT
                && car->y >= env->player_y + CAR_HEIGHT
                && env->collision_cooldown_car_vs_car <= 0
                && env->collision_cooldown_car_vs_road <= 0) {
            if (env->carsToPass > 0) {
                env->carsToPass -= 1;
            }
            if (!car->passed) {
                env->tracking_passed_cars += 1;
                env->rewards[0] += 1.0f; // Car passed reward
                env->car_passed_no_crash_active = 1; // Stepwise rewards activated
                env->step_rew_car_passed_no_crash += 0.001f; // Stepwise reward
            }
            car->passed = true;
        } else if (env->speed < 0 && car->last_y > env->player_y && car->y <= env->player_y) {
            int maxCarsToPass = (env->day == 1) ? 200 : 300; // Day 1: 200 cars, Day 2+: 300 cars
            if (env->carsToPass == maxCarsToPass) {
                // Do nothing; log the event
                env->tracking_passed_by_enemy += 1.0f;
            } else {
                env->carsToPass += 1;
                env->tracking_passed_by_enemy += 1.0f;
                env->rewards[0] -= 0.1f;
            }
        }

        // Preserve last x and y for passing, obs
        car->last_y = car->y;
        car->last_x = car->x;

        // Check for and handle collisions between player and enemy cars
        if (env->collision_cooldown_car_vs_car <= 0 && check_collision(env, car)) {
            env->tracking_collisions_player_vs_car++;
            env->rewards[0] -= 0.5f;
            env->speed = 1 + MIN_SPEED;
            env->collision_cooldown_car_vs_car = CRASH_NOOP_DURATION_CAR_VS_CAR;
            env->drift_direction = 0; // Reset drift direction
            env->car_passed_no_crash_active = 0; // Stepwise rewards deactivated until next car passed
            env->step_rew_car_passed_no_crash = 0.0f; // Reset stepwise reward
        }
    }

    // Calculate enemy spawn interval based on player speed and day
    // Values measured from original gameplay
    float min_spawn_interval = 0.5f; // 0.8777f; // Minimum spawn interval
    float max_spawn_interval;
    int dayIndex = env->day - 1;
    int numMaxSpawnIntervals = sizeof(MAX_SPAWN_INTERVALS) / sizeof(MAX_SPAWN_INTERVALS[0]);

    if (dayIndex < numMaxSpawnIntervals) {
        max_spawn_interval = MAX_SPAWN_INTERVALS[dayIndex];
    } else {
        // For days beyond first, decrease max_spawn_interval further
        max_spawn_interval = MAX_SPAWN_INTERVALS[numMaxSpawnIntervals - 1] - (dayIndex - numMaxSpawnIntervals + 1) * 0.1f;
        if (max_spawn_interval < 0.1f) {
            max_spawn_interval = 0.1f; 
        }
    }

    // Ensure min_spawn_interval is greater than or equal to max_spawn_interval
    if (min_spawn_interval < max_spawn_interval) {
        min_spawn_interval = max_spawn_interval;
    }

    // Calculate speed factor for enemy spawning
    float speed_factor = (env->speed - env->min_speed) / (env->max_speed - env->min_speed);
    if (speed_factor < 0.0f) speed_factor = 0.0f;
    if (speed_factor > 1.0f) speed_factor = 1.0f;

    // Interpolate between min and max spawn intervals to scale to player speed
    env->enemySpawnInterval = min_spawn_interval - speed_factor * (min_spawn_interval - max_spawn_interval) * 1.5f;

    // Update enemy spawn timer
    env->enemySpawnTimer += (1.0f / TARGET_FPS);
    if (env->enemySpawnTimer >= env->enemySpawnInterval) {
        env->enemySpawnTimer -= env->enemySpawnInterval;

        if (env->numEnemies < MAX_ENEMIES) {
            // Add a clumping factor based on speed
            float clump_probability = fminf((env->speed - env->min_speed) / (env->max_speed - env->min_speed), 1.0f); // Scales with speed
            int num_to_spawn = 1;

            // Randomly decide to spawn more cars in a clump
            if ((rand() / (float)RAND_MAX) < clump_probability) {
                num_to_spawn = 1 + rand() % 2; // Spawn 1 to 3 cars
            }

            // Track occupied lanes to prevent over-blocking
            int occupied_lanes[NUM_LANES] = {0};

            for (int i = 0; i < num_to_spawn && env->numEnemies < MAX_ENEMIES; i++) {
                // Find an unoccupied lane
                int lane;
                do {
                    lane = rand() % NUM_LANES;
                } while (occupied_lanes[lane]);

                // Mark the lane as occupied
                occupied_lanes[lane] = 1;

                // Add the enemy car
                int previous_num_enemies = env->numEnemies;
                add_enemy_car(env);
                if (env->numEnemies > previous_num_enemies) {
                    Car* new_car = &env->enemyCars[env->numEnemies - 1];
                    new_car->lane = lane;
                    new_car->y -= i * (CAR_HEIGHT * 3); // Vertical spacing for clump
                }
            }
        }
    }

    // Day completed logic
    if (env->carsToPass <= 0 && !env->dayCompleted) {
        env->dayCompleted = true;
    }

    // Handle day transition when background cycles back to 0
    if (env->currentDayTimeIndex == 0 && env->previousDayTimeIndex == 15) {
        // Background cycled back to 0
        if (env->dayCompleted) {
            env->tracking_days_completed += 1;
            env->day += 1;
            env->rewards[0] += 1.0f;
            env->carsToPass = 300; // Always 300 after the first day
            env->dayCompleted = false;
            add_log(env);
                    
        } else {
            // Player failed to pass required cars, soft-reset environment
            env->tracking_days_failed += 1.0f;
            env->terminals[0] = true;
            add_log(env);
            compute_observations(env); // Call compute_observations before reset to log
            reset_round(env); // Reset round == soft reset
            return;
        }
    }

    // Reward each step after a car is passed until a collision occurs. 
    // Then, no rewards per step until next car is passed.
    if (env->car_passed_no_crash_active) {
        env->rewards[0] += env->step_rew_car_passed_no_crash;
    }

    env->rewards[0] += env->crashed_penalty;
    env->tracking_crashed_penalty = env->crashed_penalty;
    env->tracking_step_rew_car_passed_no_crash = env->step_rew_car_passed_no_crash;
    env->tracking_reward = env->rewards[0];
    env->tracking_episode_return = env->rewards[0];
    env->tick++;

    float normalizedSpeed = fminf(fmaxf(env->speed, 1.0f), 2.0f);
    env->score += (int)normalizedSpeed;
    
    env->tracking_score = env->score;
    int local_cars_to_pass = env->carsToPass;
    env->tracking_cars_to_pass = (int)local_cars_to_pass;

    compute_observations(env);
}

void compute_observations(Enduro* env) {
    float* obs = env->observations;
    int obs_index = 0;

    // Most obs normalized to [0, 1]
    // Bounding box around player
    float player_x_norm = (env->player_x - env->last_road_left) / (env->last_road_right  - env->last_road_left);
    float player_y_norm = (PLAYER_MAX_Y - env->player_y) / (PLAYER_MAX_Y - PLAYER_MIN_Y);
    // float player_width_norm = CAR_WIDTH / (env->last_road_right - env->last_road_left);
    // float player_height_norm = CAR_HEIGHT / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);

    // Player position and speed
    // idx 1-3
    obs[obs_index++] = player_x_norm;
    obs[obs_index++] = player_y_norm;
    obs[obs_index++] = (env->speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED);

    // Road edges (separate lines for clarity)
    float road_left = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;

    // Road edges and last road edges
    // idx 4-7
    obs[obs_index++] = (road_left - PLAYABLE_AREA_LEFT) /
                       (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    obs[obs_index++] = (road_right - PLAYABLE_AREA_LEFT) /
                       (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    obs[obs_index++] = (env->last_road_left - PLAYABLE_AREA_LEFT) /
                        (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    obs[obs_index++] = (env->last_road_right - PLAYABLE_AREA_LEFT) /
                        (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);

    // Player lane number (0, 1, 2)
    // idx 8
    obs[obs_index++] = (float)get_player_lane(env) / (NUM_LANES - 1);

    // Enemy cars (numEnemies * 5 values (x, y, delta y, same lane as player car)) = 10 * 4 = 50 values
    // idx 9-58
    for (int i = 0; i < env->max_enemies; i++) {
        Car* car = &env->enemyCars[i];

        if (car->y > VANISHING_POINT_Y && car->y < PLAYABLE_AREA_BOTTOM) {
            // Enemy car buffer zone
            float buffer_x = CAR_WIDTH * 0.5f;
            float buffer_y = CAR_HEIGHT * 0.5f;

            // Normalize car x position relative to road edges
            float car_x_norm = ((car->x - buffer_x) - env->last_road_left) / (env->last_road_right - env->last_road_left);
            car_x_norm = fmaxf(0.0f, fminf(1.0f, car_x_norm)); // Clamp between 0 and 1
            // Normalize car y position relative to the full road height
            float car_y_norm = (PLAYABLE_AREA_BOTTOM - (car->y - buffer_y)) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
            car_y_norm = fmaxf(0.0f, fminf(1.0f, car_y_norm)); // Clamp between 0 and 1
            // Calculate delta_x for lateral movement
            float delta_x_norm = (car->last_x - car->x) / (env->last_road_right - env->last_road_left);
            // Calculate delta_y for relative speed
            float delta_y_norm = (car->last_y - car->y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
            // Determine if the car is in the same lane as the player
            int is_same_lane = (car->lane == env->lane);
            // Add normalized car x position
            obs[obs_index++] = car_x_norm;
            // Add normalized car y position
            obs[obs_index++] = car_y_norm;
            // Add normalized delta x (lateral movement)
            obs[obs_index++] = delta_x_norm;
            // Add normalized delta y (relative speed)
            obs[obs_index++] = delta_y_norm;
            // Add lane information (binary flag for lane match)
            obs[obs_index++] = (float)is_same_lane;
        } else {
            // Default values for cars that don't exist
            obs[obs_index++] = 0.5f; // Neutral x position
            obs[obs_index++] = 0.5f; // Neutral y position
            obs[obs_index++] = 0.0f; // No movement (delta_x = 0)
            obs[obs_index++] = 0.0f; // No movement (delta_y = 0)
            obs[obs_index++] = 0.0f; // Not in the same lane
        }
    }

    // Curve direction
    // idx 59
    obs[obs_index++] = (float)(env->current_curve_direction + 1) / 2.0f;

    // Observation for player's drift due to road curvature
    // idx 60-62
    // Drift direction and magnitude
    float drift_magnitude = env->current_curve_factor * CURVE_PLAYER_SHIFT_FACTOR * fabs(env->speed);
    float drift_direction = (env->current_curve_factor > 0) ? 1.0f : -1.0f; // 1 for right drift, -1 for left drift

    // Normalize drift magnitude (assume max absolute curve factor is 1.0 for normalization)
    float max_drift_magnitude = CURVE_PLAYER_SHIFT_FACTOR * env->max_speed;
    float normalized_drift_magnitude = fabs(drift_magnitude) / max_drift_magnitude;

    // Add drift direction (-1.0 to 1.0), normalized magnitude (0.0 to 1.0), and curve factor (-1.0 to 1.0)
    obs[obs_index++] = drift_direction;
    obs[obs_index++] = normalized_drift_magnitude;
    obs[obs_index++] = env->current_curve_factor; 
    
    // Time of day
    // idx 63
    float total_day_length = env->dayTransitionTimes[15];
    obs[obs_index++] = fmodf(env->elapsedTimeEnv, total_day_length) / total_day_length;

    // Cars to pass
    // idx 64
    obs[obs_index++] = (float)env->carsToPass / env->initial_cars_to_pass;

    // Compute per-lane observations: nearest enemy car distances in each lane
    // idx 65-67
    float nearest_car_distance[NUM_LANES];
    bool is_lane_empty[NUM_LANES];

    float MAX_DISTANCE = PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y; // Maximum possible distance

    for (int l = 0; l < NUM_LANES; l++) {
        nearest_car_distance[l] = MAX_DISTANCE;
        is_lane_empty[l] = true;
    }

    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        if (car->lane >= 0 && car->lane < NUM_LANES && car->y < env->player_y) {
            float distance = env->player_y - car->y;
            if (distance < nearest_car_distance[car->lane]) {
                nearest_car_distance[car->lane] = distance;
                is_lane_empty[car->lane] = false;
            }
        }
    }

    // Add per-lane normalized distances to observations
    for (int l = 0; l < NUM_LANES; l++) {
        float normalized_distance;
        if (is_lane_empty[l]) {
            normalized_distance = 1.0f; // No enemy car in front in this lane
        } else {
            normalized_distance = nearest_car_distance[l] / MAX_DISTANCE;
        }
        obs[obs_index++] = normalized_distance;
    }
}

// When to curve road and how to curve it, including dense smooth transitions
// An ugly, dense function, but it is necessary
void update_road_curve(Enduro* env) {
    int* current_curve_stage = &env->current_curve_stage;
    int* steps_in_current_stage = &env->steps_in_current_stage;
    
    // Map speed to the scale between 0.5 and 3.5
    float speed_scale = 0.5f + ((fabs(env->speed) / env->max_speed) * (MAX_SPEED - MIN_SPEED));
    float vanishing_point_transition_speed = VANISHING_POINT_TRANSITION_SPEED + speed_scale; 

    // Randomize step thresholds and curve directions
    int step_thresholds[3];
    int curve_directions[3];
    int last_direction = 0; // Tracks the last curve direction, initialized to straight (0)

    for (int i = 0; i < 3; i++) {
        // Generate random step thresholds
        step_thresholds[i] = 1500 + rand() % 3801; // Random value between 1500 and 3800

        // Generate a random curve direction (-1, 0, 1) with rules
        int direction_choices[] = {-1, 0, 1};
        int next_direction;

        do {
            next_direction = direction_choices[rand() % 3];
        } while ((last_direction == -1 && next_direction == 1) || (last_direction == 1 && next_direction == -1));

        curve_directions[i] = next_direction;
        last_direction = next_direction;
    }

    // Use step thresholds and curve directions dynamically
    env->current_step_threshold = step_thresholds[*current_curve_stage % 3];
    (*steps_in_current_stage)++;

    if (*steps_in_current_stage >= step_thresholds[*current_curve_stage % 3]) {
        env->target_curve_factor = (float)curve_directions[*current_curve_stage % 3];
        *steps_in_current_stage = 0;
        *current_curve_stage = (*current_curve_stage + 1) % 3;
    }

    // Determine sizes of step_thresholds and curve_directions
    size_t step_thresholds_size = sizeof(step_thresholds) / sizeof(step_thresholds[0]);
    size_t curve_directions_size = sizeof(curve_directions) / sizeof(curve_directions[0]);

    // Find the maximum size
    size_t max_size = (step_thresholds_size > curve_directions_size) ? step_thresholds_size : curve_directions_size;

    // Adjust arrays dynamically
    int adjusted_step_thresholds[max_size];
    int adjusted_curve_directions[max_size];

    for (size_t i = 0; i < max_size; i++) {
        adjusted_step_thresholds[i] = step_thresholds[i % step_thresholds_size];
        adjusted_curve_directions[i] = curve_directions[i % curve_directions_size];
    }

    // Use adjusted arrays for current calculations
    env->current_step_threshold = adjusted_step_thresholds[*current_curve_stage % max_size];
    (*steps_in_current_stage)++;

    if (*steps_in_current_stage >= adjusted_step_thresholds[*current_curve_stage]) {
        env->target_curve_factor = (float)adjusted_curve_directions[*current_curve_stage % max_size];
        *steps_in_current_stage = 0;
        *current_curve_stage = (*current_curve_stage + 1) % max_size;
    }

    if (env->current_curve_factor < env->target_curve_factor) {
        env->current_curve_factor = fminf(env->current_curve_factor + CURVE_TRANSITION_SPEED, env->target_curve_factor);
    } else if (env->current_curve_factor > env->target_curve_factor) {
        env->current_curve_factor = fmaxf(env->current_curve_factor - CURVE_TRANSITION_SPEED, env->target_curve_factor);
    }
    env->current_curve_direction = fabsf(env->current_curve_factor) < 0.1f ? CURVE_STRAIGHT 
                             : (env->current_curve_factor > 0) ? CURVE_RIGHT : CURVE_LEFT;

    // Move the vanishing point gradually
    env->base_target_vanishing_point_x = VANISHING_POINT_X_LEFT - env->t_p * (VANISHING_POINT_X_LEFT - VANISHING_POINT_X_RIGHT);
    float target_shift = env->current_curve_direction * CURVE_VANISHING_POINT_SHIFT;
    env->target_vanishing_point_x = env->base_target_vanishing_point_x + target_shift;

    if (env->current_vanishing_point_x < env->target_vanishing_point_x) {
        env->current_vanishing_point_x = fminf(env->current_vanishing_point_x + vanishing_point_transition_speed, env->target_vanishing_point_x);
    } else if (env->current_vanishing_point_x > env->target_vanishing_point_x) {
        env->current_vanishing_point_x = fmaxf(env->current_vanishing_point_x - vanishing_point_transition_speed, env->target_vanishing_point_x);
    }
    env->vanishing_point_x = env->current_vanishing_point_x;
}

// B(t) = (1−t)^2 * P0​+2(1−t) * t * P1​+t^2 * P2​, t∈[0,1]
// Quadratic bezier curve helper function
float quadratic_bezier(float bottom_x, float control_x, float top_x, float t) {
    float one_minus_t = 1.0f - t;
    return one_minus_t * one_minus_t * bottom_x + 
           2.0f * one_minus_t * t * control_x + 
           t * t * top_x;
}

// Computes the edges of the road. Use for both L and R. 
// Lots of magic numbers to replicate as exactly as possible
// original Atari 2600 Enduro road rendering.
float road_edge_x(Enduro* env, float y, float offset, unsigned char left) {
    float t = (PLAYABLE_AREA_BOTTOM - y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float base_offset = left ? -ROAD_LEFT_OFFSET : ROAD_RIGHT_OFFSET;
    float bottom_x = env->base_vanishing_point_x + base_offset + offset;
    float top_x = env->current_vanishing_point_x + offset;    
    float edge_x;
    if (fabsf(env->current_curve_factor) < 0.01f) {
        // Straight road
        edge_x = bottom_x + t * (top_x - bottom_x);
    } else {
        // Adjust curve offset based on curve direction
        float curve_offset = (env->current_curve_factor > 0 ? -30.0f : 30.0f) * fabsf(env->current_curve_factor);
        float control_x = bottom_x + (top_x - bottom_x) * 0.5f + curve_offset;
        // Calculate edge using Bézier curve for proper curvature
        edge_x = quadratic_bezier(bottom_x, control_x, top_x, t);
    }

    // Wiggle effect
    float wiggle_offset = 0.0f;
    if (env->wiggle_active && y >= env->wiggle_y && y <= env->wiggle_y + env->wiggle_length) {
        float t_wiggle = (y - env->wiggle_y) / env->wiggle_length; // Ranges from 0 to 1
        // Trapezoidal wave calculation
        if (t_wiggle < 0.15f) {
            // Connection to road edge
            wiggle_offset = env->wiggle_amplitude * (t_wiggle / 0.15f);
        } else if (t_wiggle < 0.87f) {
            // Flat top of wiggle
            wiggle_offset = env->wiggle_amplitude;
        } else {
            // Reconnection to road edge
            wiggle_offset = env->wiggle_amplitude * ((1.0f - t_wiggle) / 0.13f);
        }
        // Wiggle towards road center
        wiggle_offset *= (left ? 1.0f : -1.0f);
        // Scale wiggle offset based on y position, starting at 0.03f at the vanishing point
        float depth = (y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
        float scale = 0.03f + (depth * depth); 
        if (scale > 0.3f) {
            scale = 0.3f;
        }
        wiggle_offset *= scale;
    }
    // Apply the wiggle offset
    edge_x += wiggle_offset;
    return edge_x;
}

// Computes x position of car in a given lane
float car_x_in_lane(Enduro* env, int lane, float y) {
    // Set offset to 0 to ensure enemy cars align with the road rendering
    float offset = 0.0f;
    float left_edge = road_edge_x(env, y, offset, true);
    float right_edge = road_edge_x(env, y, offset, false);
    float lane_width = (right_edge - left_edge) / NUM_LANES;
    return left_edge + lane_width * (lane + 0.5f);
}

// Handles rendering logic
Client* make_client(Enduro* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    // State data from env (Enduro*)
    client->width = env->width;
    client->height = env->height;
    
    initRaylib(&client->gameState); // Pass gameState here
    loadTextures(&client->gameState);

    return client;
}

void close_client(Client* client) {
    cleanup(&client->gameState);
    CloseWindow();
    free(client);
}

void render_car(GameState* gameState) {
    int carAssetIndex = gameState->showLeftTread ? gameState->playerCarLeftTreadIndex : gameState->playerCarRightTreadIndex;
    Rectangle srcRect = asset_map[carAssetIndex];
    Vector2 position = { gameState->player_x, gameState->player_y };
    DrawTextureRec(gameState->spritesheet, srcRect, position, WHITE);
}

void initRaylib(GameState* gameState) {
    if (!IsWindowReady()) {
        InitWindow(SCREEN_WIDTH * 2, SCREEN_HEIGHT * 2, "puffer_enduro");
        SetTargetFPS(60);
        gameState->renderTarget = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT);
    }
}

void loadTextures(GameState* gameState) { 
    // Load background and mountain textures for different times of day per original env
    gameState->spritesheet = LoadTexture("resources/enduro/enduro_spritesheet.png");
    
    // Initialize animation variables
    gameState->carAnimationTimer = 0.0f;
    gameState->carAnimationInterval = 0.05f; // Init; updated based on speed
    gameState->showLeftTread = true;
    gameState->mountainPosition = 0.0f;

    // Initialize victory effect variables
    gameState->showLeftFlag = true;
    gameState->flagTimer = 0;
    gameState->victoryDisplayTimer = 0;
    gameState->victoryAchieved = false;

    // Initialize scoreboard variables
    gameState->score = 0;
    gameState->scoreTimer = 0;
    gameState->carsLeftGameState = 0;
    gameState->day = 1;

    // Initialize score digits arrays
    for (int i = 0; i < SCORE_DIGITS; i++) {
        gameState->scoreDigitCurrents[i] = 0;
        gameState->scoreDigitNexts[i] = 0;
        gameState->scoreDigitOffsets[i] = 0.0f;
        gameState->scoreDigitScrolling[i] = false;
    }

    // Initialize time-of-day variables
    gameState->elapsedTime = 0.0f;
    gameState->currentBackgroundIndex = 0;
    gameState->previousBackgroundIndex = 0;

    // Initialize background and mountain indices
    for (int i = 0; i < 16; ++i) {
        gameState->backgroundIndices[i] = ASSET_BG_0 + i;
        gameState->mountainIndices[i] = ASSET_MOUNTAIN_0 + i;
    }

    // Initialize digit indices
    for (int i = 0; i < 10; ++i) {
        gameState->digitIndices[i] = ASSET_DIGITS_0 + i;
        gameState->greenDigitIndices[i] = ASSET_GREEN_DIGITS_0 + i;
        gameState->yellowDigitIndices[i] = ASSET_YELLOW_DIGITS_0 + i;
    }
    gameState->digitIndices[10] = ASSET_DIGITS_CAR; // Index for "CAR"

    // Initialize enemy car indices
    int baseEnemyCarIndex = ASSET_ENEMY_CAR_BLUE_LEFT_TREAD;
    for (int color = 0; color < 6; ++color) {
        for (int tread = 0; tread < 2; ++tread) {
            gameState->enemyCarIndices[color][tread] = baseEnemyCarIndex + color * 2 + tread;
        }
    }

    // Load other asset indices
    gameState->enemyCarNightTailLightsIndex = ASSET_ENEMY_CAR_NIGHT_TAIL_LIGHTS;
    gameState->enemyCarNightFogTailLightsIndex = ASSET_ENEMY_CAR_NIGHT_FOG_TAIL_LIGHTS;
    gameState->playerCarLeftTreadIndex = ASSET_PLAYER_CAR_LEFT_TREAD;
    gameState->playerCarRightTreadIndex = ASSET_PLAYER_CAR_RIGHT_TREAD;
    gameState->levelCompleteFlagLeftIndex = ASSET_LEVEL_COMPLETE_FLAG_LEFT;
    gameState->levelCompleteFlagRightIndex = ASSET_LEVEL_COMPLETE_FLAG_RIGHT;
    
    // Initialize animation variables
    gameState->carAnimationTimer = 0.0f;
    gameState->carAnimationInterval = 0.05f; // Initial interval, will be updated based on speed
    gameState->showLeftTread = true;
    gameState->mountainPosition = 0.0f;
}

void cleanup(GameState* gameState) {
    UnloadRenderTexture(gameState->renderTarget);
    UnloadTexture(gameState->spritesheet);
}

void updateCarAnimation(GameState* gameState) {
    // Update the animation interval based on the player's speed
    // Faster speed means faster alternation
    float minInterval = 0.005f;  // Minimum interval at max speed
    float maxInterval = 0.075f;  // Maximum interval at min speed

    float speedRatio = (gameState->speed - gameState->min_speed) / (gameState->max_speed - gameState->min_speed);
    gameState->carAnimationInterval = maxInterval - (maxInterval - minInterval) * speedRatio;

    // Update the animation timer
    gameState->carAnimationTimer += GetFrameTime(); // Time since last frame

    if (gameState->carAnimationTimer >= gameState->carAnimationInterval) {
        gameState->carAnimationTimer = 0.0f;
        gameState->showLeftTread = !gameState->showLeftTread; // Switch texture
    }
}

void updateScoreboard(GameState* gameState) {
    float normalizedSpeed = fminf(fmaxf(gameState->speed, 1.0f), 2.0f);
    // Determine the frame interval for score increment based on speed
    int frameInterval = (int)(30 / normalizedSpeed);
    gameState->scoreTimer++;

    if (gameState->scoreTimer >= frameInterval) {
        gameState->scoreTimer = 0;
        // Increment the score based on normalized speed
        gameState->score += (int)normalizedSpeed;
        if (gameState->score > 99999) {
            gameState->score = 0;
        }
        // Determine which digits have changed and start scrolling them
        int tempScore = gameState->score;
        for (int i = SCORE_DIGITS - 1; i >= 0; i--) {
            int newDigit = tempScore % 10;
            tempScore /= 10;
            if (newDigit != gameState->scoreDigitCurrents[i]) {
                gameState->scoreDigitNexts[i] = newDigit;
                gameState->scoreDigitOffsets[i] = 0.0f;
                gameState->scoreDigitScrolling[i] = true;
            }
        }
    }
    // Update scrolling digits
    float scrollSpeed = 0.55f * normalizedSpeed;
    for (int i = 0; i < SCORE_DIGITS; i++) {
        if (gameState->scoreDigitScrolling[i]) {
            gameState->scoreDigitOffsets[i] += scrollSpeed; // Scroll speed
            if (gameState->scoreDigitOffsets[i] >= DIGIT_HEIGHT) {
                gameState->scoreDigitOffsets[i] = 0.0f;
                gameState->scoreDigitCurrents[i] = gameState->scoreDigitNexts[i];
                gameState->scoreDigitScrolling[i] = false; // Stop scrolling
            }
        }
    }
}

void renderBackground(GameState* gameState) {
    int bgIndex = gameState->backgroundIndices[gameState->currentBackgroundIndex];
    Rectangle srcRect = asset_map[bgIndex];
    DrawTextureRec(gameState->spritesheet, srcRect, (Vector2){0, 0}, WHITE);
}

void renderScoreboard(GameState* gameState) {
    int digitWidth = DIGIT_WIDTH;
    int digitHeight = DIGIT_HEIGHT;
    // Convert bottom-left coordinates to top-left origin
    // -8 for x resolution change from 160 to 152
    int scoreStartX = 56 + digitWidth - 8;
    int scoreStartY = 173 - digitHeight;
    int dayX = 56 - 8;
    int dayY = 188 - digitHeight;
    int carsX = 72 - 8;
    int carsY = 188 - digitHeight;

    // Render score with scrolling effect
    for (int i = 0; i < SCORE_DIGITS; ++i) {
        int digitX = scoreStartX + i * digitWidth;
        int currentDigitIndex = gameState->scoreDigitCurrents[i];
        int nextDigitIndex = gameState->scoreDigitNexts[i];

        int currentAssetIndex, nextAssetIndex;
        if (i == SCORE_DIGITS - 1) {
            // Use yellow digits for the last digit
            currentAssetIndex = gameState->yellowDigitIndices[currentDigitIndex];
            nextAssetIndex = gameState->yellowDigitIndices[nextDigitIndex];
        } else {
            // Use regular digits
            currentAssetIndex = gameState->digitIndices[currentDigitIndex];
            nextAssetIndex = gameState->digitIndices[nextDigitIndex];
        }
        Rectangle srcRectCurrentFull = asset_map[currentAssetIndex];
        Rectangle srcRectNextFull = asset_map[nextAssetIndex];

        if (gameState->scoreDigitScrolling[i]) {
            // Scrolling effect for this digit
            float offset = gameState->scoreDigitOffsets[i];
            // Render current digit moving up
            Rectangle srcRectCurrent = srcRectCurrentFull;
            srcRectCurrent.height = digitHeight - (int)offset;
            Rectangle destRectCurrent = { digitX, scoreStartY + (int)offset, digitWidth, digitHeight - (int)offset };
            DrawTexturePro(
                gameState->spritesheet,
                srcRectCurrent,
                destRectCurrent,
                (Vector2){ 0, 0 },
                0.0f,
                WHITE
            );
            // Render next digit coming up from below
            Rectangle srcRectNext = srcRectNextFull;
            srcRectNext.y += digitHeight - (int)offset;
            srcRectNext.height = (int)offset;
            Rectangle destRectNext = { digitX, scoreStartY, digitWidth, (int)offset };
            DrawTexturePro(
                gameState->spritesheet,
                srcRectNext,
                destRectNext,
                (Vector2){ 0, 0 },
                0.0f,
                WHITE
            );
        } else {
            // No scrolling, render the current digit normally
            Rectangle srcRect = asset_map[currentAssetIndex];
            Vector2 position = { digitX, scoreStartY };
            DrawTextureRec(gameState->spritesheet, srcRect, position, WHITE);
        }
    }

    // Render day number
    int day = gameState->day % 10;
    int dayTextureIndex = day;
    // Pass dayCompleted condition from Enduro to GameState
    if (gameState->dayCompleted) {
        gameState->victoryAchieved = true;
    }
    Rectangle daySrcRect;
    if (gameState->victoryAchieved) {
        // Green day digits during victory
        int assetIndex = gameState->greenDigitIndices[dayTextureIndex];
        daySrcRect = asset_map[assetIndex];
    } else {
        // Use normal digits
        int assetIndex = gameState->digitIndices[dayTextureIndex];
        daySrcRect = asset_map[assetIndex];
    }
    Vector2 dayPosition = { dayX, dayY };
    DrawTextureRec(gameState->spritesheet, daySrcRect, dayPosition, WHITE);

    // Render "CAR" digit or flags for cars to pass
    if (gameState->victoryAchieved) {
        // Alternate between level_complete_flag_left and level_complete_flag_right
        int flagAssetIndex = gameState->showLeftFlag ? gameState->levelCompleteFlagLeftIndex : gameState->levelCompleteFlagRightIndex;
        Rectangle flagSrcRect = asset_map[flagAssetIndex];
        Rectangle destRect = { carsX, carsY, flagSrcRect.width, flagSrcRect.height };
        DrawTexturePro(
            gameState->spritesheet,
            flagSrcRect,
            destRect,
            (Vector2){ 0, 0 },
            0.0f,
            WHITE
        );
    } else {
        // Render "CAR" label
        int carAssetIndex = gameState->digitIndices[10]; // Index for "CAR"
        Rectangle carSrcRect = asset_map[carAssetIndex];
        Vector2 carPosition = { carsX, carsY };
        DrawTextureRec(gameState->spritesheet, carSrcRect, carPosition, WHITE);

        // Render the remaining digits for cars to pass
        int cars = gameState->carsLeftGameState;
        if (cars < 0) cars = 0; // Ensure cars is not negative
        for (int i = 1; i < CARS_DIGITS; ++i) {
            int divisor = (int)pow(10, CARS_DIGITS - i - 1);
            int digit = (cars / divisor) % 10;
            if (digit < 0 || digit > 9) digit = 0; // Clamp digit to valid range
            int digitX = carsX + i * (digitWidth + 1); // Add spacing between digits
            int assetIndex = gameState->digitIndices[digit];
            Rectangle srcRect = asset_map[assetIndex];
            Vector2 position = { digitX, carsY };
            DrawTextureRec(gameState->spritesheet, srcRect, position, WHITE);
        }
    }
}

// Triggers the day completed 'victory' display
// Solely for flapping flag visual effect
void updateVictoryEffects(GameState* gameState) {
    if (!gameState->victoryAchieved) {
        return;
    }
    gameState->flagTimer++;
    // Modulo triggers flag direction change
    // Flag renders in that direction until next change
    if (gameState->flagTimer % 50 == 0) {
        gameState->showLeftFlag = !gameState->showLeftFlag;
    }
    gameState->victoryDisplayTimer++;
    if (gameState->victoryDisplayTimer >= 10) {
        gameState->victoryDisplayTimer = 0;
    }
}

void updateMountains(GameState* gameState) {
    // Mountain scrolling effect when road is curving
    float baseSpeed = 0.0f;
    float curveStrength = fabsf(gameState->current_curve_factor);
    float speedMultiplier = 1.0f; // Scroll speed
    float scrollSpeed = baseSpeed + curveStrength * speedMultiplier;
    int mountainIndex = gameState->mountainIndices[0]; // Use any mountain index since width is consistent
    int mountainWidth = asset_map[mountainIndex].width;
    if (gameState->current_curve_direction == 1) { // Turning left
        gameState->mountainPosition += scrollSpeed;
        if (gameState->mountainPosition >= mountainWidth) {
            gameState->mountainPosition -= mountainWidth;
        }
    } else if (gameState->current_curve_direction == -1) { // Turning right
        gameState->mountainPosition -= scrollSpeed;
        if (gameState->mountainPosition <= -mountainWidth) {
            gameState->mountainPosition += mountainWidth;
        }
    }
}

void renderMountains(GameState* gameState) {
    int mountainIndex = gameState->mountainIndices[gameState->currentBackgroundIndex];
    Rectangle srcRect = asset_map[mountainIndex];
    int mountainWidth = srcRect.width;
    int mountainY = 45; // Y position per original environment

    float playerCenterX = (PLAYER_MIN_X + PLAYER_MAX_X) / 2.0f;
    float playerOffset = gameState->player_x - playerCenterX;
    float parallaxFactor = 0.5f;
    float adjustedOffset = -playerOffset * parallaxFactor;
    float mountainX = -gameState->mountainPosition + adjustedOffset;

    BeginScissorMode(PLAYABLE_AREA_LEFT, 0, SCREEN_WIDTH - PLAYABLE_AREA_LEFT, SCREEN_HEIGHT);
    for (int x = (int)mountainX; x < SCREEN_WIDTH; x += mountainWidth) {
        DrawTextureRec(gameState->spritesheet, srcRect, (Vector2){x, mountainY}, WHITE);
    }
    for (int x = (int)mountainX - mountainWidth; x > -mountainWidth; x -= mountainWidth) {
        DrawTextureRec(gameState->spritesheet, srcRect, (Vector2){x, mountainY}, WHITE);
    }
    EndScissorMode();
}

void c_render(Enduro* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    // Client* client = env->client;
    GameState* gameState = &env->client->gameState;

    // Copy env state to gameState
    gameState->speed = env->speed;
    gameState->min_speed = env->min_speed;
    gameState->max_speed = env->max_speed;
    gameState->current_curve_direction = env->current_curve_direction;
    gameState->current_curve_factor = env->current_curve_factor;
    gameState->player_x = env->player_x;
    gameState->player_y = env->player_y;
    gameState->initial_player_x = env->initial_player_x;
    gameState->vanishing_point_x = env->vanishing_point_x;
    gameState->t_p = env->t_p;
    gameState->dayCompleted = env->dayCompleted;
    gameState->currentBackgroundIndex = env->currentDayTimeIndex;
    gameState->carsLeftGameState = env->carsToPass;
    gameState->day = env->day;
    gameState->elapsedTime = env->elapsedTimeEnv;
    // -1 represents reset of env
    if (env->score == 0) {
        gameState->score = 0;
    }

    // Render to a texture for scaling up
    BeginTextureMode(gameState->renderTarget);

    // Do not call BeginDrawing() here
    ClearBackground(BLACK);
    BeginBlendMode(BLEND_ALPHA);

    renderBackground(gameState);
    updateCarAnimation(gameState);
    updateMountains(gameState);
    renderMountains(gameState);
    
    int bgIndex = gameState->currentBackgroundIndex;
    unsigned char isNightFogStage = (bgIndex == 13);
    unsigned char isNightStage = (bgIndex == 12 || bgIndex == 13 || bgIndex == 14);

    // During night fog stage, clip rendering to y >= 92
    float clipStartY = isNightFogStage ? 92.0f : VANISHING_POINT_Y;
    float clipHeight = PLAYABLE_AREA_BOTTOM - clipStartY;
    Rectangle clipRect = { PLAYABLE_AREA_LEFT, clipStartY, PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT, clipHeight };
    BeginScissorMode(clipRect.x, clipRect.y, clipRect.width, clipRect.height);

    // Render road edges
    float roadStartY = isNightFogStage ? 92.0f : VANISHING_POINT_Y;
    Vector2 previousLeftPoint = {0}, previousRightPoint = {0};
    unsigned char firstPoint = true;

    for (float y = roadStartY; y <= PLAYABLE_AREA_BOTTOM; y += 0.75f) {
        float adjusted_y = (env->speed < 0) ? y : y + fmod(env->road_scroll_offset, 0.75f);
        if (adjusted_y > PLAYABLE_AREA_BOTTOM) continue;

        float left_edge = road_edge_x(env, adjusted_y, 0, true);
        float right_edge = road_edge_x(env, adjusted_y, 0, false);

        // Road color based on y position
        Color roadColor;
        if (adjusted_y >= 52 && adjusted_y < 91) {
            roadColor = (Color){74, 74, 74, 255};
        } else if (adjusted_y >= 91 && adjusted_y < 106) {
            roadColor = (Color){111, 111, 111, 255};
        } else if (adjusted_y >= 106 && adjusted_y <= 154) {
            roadColor = (Color){170, 170, 170, 255};
        } else {
            roadColor = WHITE;
        }

        Vector2 currentLeftPoint = {left_edge, adjusted_y};
        Vector2 currentRightPoint = {right_edge, adjusted_y};

        if (!firstPoint) {
            DrawLineV(previousLeftPoint, currentLeftPoint, roadColor);
            DrawLineV(previousRightPoint, currentRightPoint, roadColor);
        }

        previousLeftPoint = currentLeftPoint;
        previousRightPoint = currentRightPoint;
        firstPoint = false;
    }

    // Render enemy cars scaled stages for distance/closeness effect
    unsigned char skipFogCars = isNightFogStage;
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        
        // Don't render cars in fog
        if (skipFogCars && car->y < 92.0f) {
            continue;
        }

        // Determine the car scale based on distance
        float car_scale = get_car_scale(car->y);
        // Select the correct texture based on the car's color and current tread
        int carAssetIndex;
        if (isNightStage) {
            carAssetIndex = (bgIndex == 13) ? gameState->enemyCarNightFogTailLightsIndex : gameState->enemyCarNightTailLightsIndex;
        } else {
            int colorIndex = car->colorIndex;
            int treadIndex = gameState->showLeftTread ? 0 : 1;
            carAssetIndex = gameState->enemyCarIndices[colorIndex][treadIndex];
        }
        Rectangle srcRect = asset_map[carAssetIndex];

        // Compute car coordinates
        float car_center_x = car_x_in_lane(env, car->lane, car->y);
        float car_x = car_center_x - (srcRect.width * car_scale) / 2.0f;
        float car_y = car->y - (srcRect.height * car_scale) / 2.0f;

        DrawTexturePro(
            gameState->spritesheet,
            srcRect,
            (Rectangle){ car_x, car_y, srcRect.width * car_scale, srcRect.height * car_scale },
            (Vector2){ 0, 0 },
            0.0f,
            WHITE
        );
    }

    render_car(gameState); // Unscaled player car

    EndScissorMode();
    EndBlendMode();
    updateVictoryEffects(gameState);
    updateScoreboard(gameState);
    renderScoreboard(gameState);

    // Update GameState data from environment data;
    env->client->gameState.victoryAchieved = env->dayCompleted;

    // Finish rendering to the texture
    EndTextureMode();

    // Now render the scaled-up texture to the screen
    BeginDrawing();
    ClearBackground(BLACK);

    // Draw the render texture scaled up
    DrawTexturePro(
        gameState->renderTarget.texture,
        (Rectangle){ 0, 0, (float)gameState->renderTarget.texture.width, -(float)gameState->renderTarget.texture.height },
        (Rectangle){ 0, 0, (float)SCREEN_WIDTH * 2, (float)SCREEN_HEIGHT * 2 },
        (Vector2){ 0, 0 },
        0.0f,
        WHITE
    );

    EndDrawing();
}
