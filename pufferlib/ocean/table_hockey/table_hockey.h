#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "raylib.h"
#ifdef __AVX2__
#include <immintrin.h>
#define SIMD_AVAILABLE 1
#elif __SSE2__
#include <emmintrin.h>
#define SIMD_AVAILABLE 1
#else
#define SIMD_AVAILABLE 0
#endif
#define TABLE_WIDTH 8.0f
#define TABLE_HEIGHT 4.0f
#define GOAL_WIDTH 1.0f
#define PUCK_RADIUS 0.1f
#define PADDLE_RADIUS 0.15f
#define MAX_SPEED 3.0f
#define PUCK_MAX_SPEED 6.0f
#define COLLISION_BOOST 1.1f
#define WALL_RESTITUTION 0.95f
#define MAX_STEPS 2000
#define PADDLE_SMOOTHING 0.3f
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define SCALE 80.0f
typedef struct PhysicsCache {
    float inv_table_width_half;
    float inv_table_height_half;
    float inv_puck_max_speed;
    float inv_paddle_max_speed;
    float collision_dist_squared;
    __attribute__((aligned(32))) float wall_bounds[8];
} PhysicsCache;

typedef enum {
    RENDER_HEADLESS = 0,
    RENDER_NORMAL = 1,
    RENDER_VR = 2
} RenderMode;

typedef enum {
    ACTION_CONTINUOUS = 0,
    ACTION_DISCRETE = 1
} ActionMode;

typedef struct Puck {
    float x, y;
    float vx, vy;
} Puck;

typedef struct Paddle {
    float x, y;
    float vx, vy;
    float target_x, target_y;
} Paddle;

typedef struct GameState {
    int player_score;
    int opponent_score;
    int episode_length;
    int puck_hits;
    float last_hit_time;
    bool goal_scored;
    bool episode_done;
    int wall_bounces_this_rally;
    int current_rally_length;
    int max_rally_this_episode;
    float total_puck_speed;
    int total_shots;
    int defensive_hits;
    int total_blocks;
    int opponent_shots;
    float total_reaction_time;
    int reaction_count;  
    float player_distance_traveled;
    float opponent_distance_traveled;
    float player_min_x, player_max_x;
    float player_min_y, player_max_y;
    int wall_bounce_goals;
    int direct_goals;
    bool puck_hit_wall_this_rally;
    float last_puck_vx, last_puck_vy;
    float prev_puck_x, prev_puck_y;
    float player_vel_x, player_vel_y;
    float opponent_vel_x, opponent_vel_y;
    float time_since_last_hit;
    float last_shot_power;
    int consecutive_saves;
    float total_movement_this_episode;
    bool in_defensive_zone;
    float distance_to_puck;
} GameState;

typedef struct Log {
    float win_rate;
    float player_goals;
    float opponent_goals;
    float episode_length;
    float puck_hits_per_episode;
    float avg_reward;
    float perf;
    float n;
} Log;

typedef struct Client {
    RenderMode render_mode;
    bool window_initialized;
    bool vr_initialized;
    Camera3D camera;
    Vector2 mouse_position;
    bool show_debug_info;
} Client;

typedef struct TableHockey {
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;
    Puck puck;
    Paddle player_paddle;
    Paddle opponent_paddle;
    GameState game_state;
    Log log;
    RenderMode render_mode;
    ActionMode action_mode;
    bool self_play;
    float dt;
    int tick;
    float max_paddle_speed;
    float puck_hit_reward;
    float goal_reward;
    float idle_penalty;
    float positioning_reward;
    float power_shot_bonus;
    float anticipation_reward;
    float save_zone_reward;
    float efficiency_penalty;
    float rally_bonus_base;
    Client* client;
    float episode_return;
    PhysicsCache perf_cache;
} TableHockey;

void init(TableHockey* env);
void allocate(TableHockey* env);
void free_allocated(TableHockey* env);
void c_close(TableHockey* env);

void c_reset(TableHockey* env);
void c_step(TableHockey* env);
void c_render(TableHockey* env);

void update_physics(TableHockey* env);
void update_puck(TableHockey* env);
void update_paddles(TableHockey* env);
void handle_collisions(TableHockey* env);
bool check_puck_paddle_collision(Puck* puck, Paddle* paddle, float* collision_normal_x, float* collision_normal_y);
void handle_wall_collisions(TableHockey* env);
void handle_goal_scoring(TableHockey* env);
void reset_puck_position(TableHockey* env);
void reset_rally_tracking(TableHockey* env);

void reset_game(TableHockey* env);
void apply_actions(TableHockey* env);
void calculate_rewards(TableHockey* env);
void update_observations(TableHockey* env);
void add_log(TableHockey* env);
void process_human_input(TableHockey* env);
void process_vr_input(TableHockey* env);

Client* make_client(TableHockey* env);
void render_frame(TableHockey* env);
void render_2d(TableHockey* env);
void render_3d(TableHockey* env);
void render_vr(TableHockey* env);
void draw_game_elements_2d(TableHockey* env);
void draw_game_elements_3d(TableHockey* env);
void draw_ui(TableHockey* env);

float distance(float x1, float y1, float x2, float y2);
float clamp_value(float value, float min_val, float max_val);
void normalize_vector(float* x, float* y);
float vector_length(float x, float y);

void init_physics_cache(TableHockey* env);

extern const Color PUFF_RED;
extern const Color PUFF_CYAN;
extern const Color PUFF_WHITE;
extern const Color PUFF_BACKGROUND;
extern const Color PUFF_GREEN;
extern const Color PUFF_BLUE;