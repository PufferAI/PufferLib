#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include "freeway_levels.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#define NOOP 0
#define UP 1
#define DOWN 2

// Gameplay related
#define TICK_RATE 1.0f/60.0f
#define GAME_LENGTH 136.0f // Game length in seconds
#define RANDOMIZE_SPEED_FREQ 360 // How many ticks before randomize the speed of the enemies
#define TICKS_STUNT 40
#define PENALTY_HIT -0.01f // Penalty for hitting an enemy
// Rendering related
#define HALF_LINEWIDTH 2
#define DASH_SPACING 32
#define DASH_SIZE 32

// Based on https://ale.farama.org/environments/freeway/
typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float up_action_frac;
    float hits;
    float n;
};

typedef struct FreewayPlayer FreewayPlayer;
struct FreewayPlayer {
    float player_x;
    float player_y;
    int best_lane_idx;// furthest lane achieved so far
    int ticks_stunts_left; 
    int score;
    int hits;
    int is_human;
};

typedef struct FreewayEnemy FreewayEnemy;
struct FreewayEnemy {
    float enemy_x;
    float enemy_y;
    float enemy_initial_x;
    float enemy_vx; // velocity in pixels per second
    int speed_randomization; // 0 for no randomization, 1 for randomization
    int initial_speed_idx; // index of the initial speed in the speed array
    int current_speed_idx; // index of the current speed in the speed array
    int is_enabled;
    int lane_idx; // lane index
    int type;
    int enemy_width;
    int enemy_height;
};

typedef struct Client Client;
typedef struct Freeway Freeway;
struct Freeway {
    Client* client;
    Log log;
    float* observations;
    int* actions;
    int* human_actions;
    float* rewards;
    unsigned char* terminals;

    FreewayPlayer ai_player; // Player-Related
    FreewayPlayer human_player; 
    int player_width;
    int player_height; 
    float ep_return;
    int up_count;

    FreewayEnemy* enemies; // Enemy-Related
    int car_width;
    int car_height;
    int truck_width;
    int truck_height;
    
    int difficulty; // Global
    int level; 
    int lane_size;
    float road_start;
    float road_end;
    int width;
    int height;
    int tick;
    float time_left;
    int frameskip;
    int use_dense_rewards;
    int env_randomization;
    int enable_human_player;
};

void load_level(Freeway* env, int level) {
    FreewayEnemy* enemy;
    for (int lane = 0; lane < NUM_LANES; lane++) {
        for (int i = 0; i < MAX_ENEMIES_PER_LANE; i++){
            enemy = &env->enemies[lane * MAX_ENEMIES_PER_LANE + i];
            enemy->is_enabled = (i < ENEMIES_PER_LANE[level][lane]);
            enemy->enemy_x = 0.0f;
            enemy->enemy_initial_x = ENEMIES_INITIAL_X[level][lane][i] * env->width;
            enemy->speed_randomization = SPEED_RANDOMIZATION[level];
            enemy->initial_speed_idx = ENEMIES_INITIAL_SPEED_IDX[level][lane];
            enemy->current_speed_idx = enemy->initial_speed_idx;
            enemy->lane_idx = lane;
            enemy->enemy_y = (env->road_start + (env->road_end - env->road_start) * lane / (float) NUM_LANES) - env->lane_size  / 2;
            enemy->type = ENEMIES_TYPES[level][lane];
            enemy->enemy_width = enemy->type == 0 ? env->car_width : env->truck_width;
            enemy->enemy_height = enemy->type == 0 ? env->car_height : env->truck_height;

            enemy->enemy_vx = enemy->lane_idx < NUM_LANES/2 ? SPEED_VALUES[enemy->current_speed_idx] * TICK_RATE * env->width: -SPEED_VALUES[enemy->current_speed_idx] * TICK_RATE * env->width;

        }
    }
}

void init(Freeway* env) {
    env->ai_player.player_x = env->width / 4;
    env->ai_player.player_y = env->height / 2;
    env->ai_player.best_lane_idx = 0;
    env->ai_player.ticks_stunts_left = 0;
    env->ai_player.score = 0;
    env->ai_player.is_human = 0;
    env->ai_player.hits = 0;

    env->human_player.player_x = 3 * env->width / 4;
    env->human_player.player_y = env->height / 2;
    env->human_player.best_lane_idx = 0;
    env->human_player.ticks_stunts_left = 0;
    env->human_player.score = 0;
    env->human_player.is_human = 1;
    env->human_player.hits = 0;
    

    env->truck_height = env->car_height;
    env->truck_width = 2*env->car_width;
    env->road_start = env->height / 2 + (NUM_LANES * env->lane_size) / 2;
    env->road_end = env->road_start - (NUM_LANES * env->lane_size);
    //enemies 
    env->enemies = (FreewayEnemy*)calloc(NUM_LANES*MAX_ENEMIES_PER_LANE, sizeof(FreewayEnemy));
    env->human_actions = (int*)calloc(1, sizeof(int));
    if ((env->level < 0) || (env->level >= NUM_LEVELS)) {
        env->level = rand() % NUM_LEVELS;
    }
    load_level(env, env->level);
}

void allocate(Freeway* env) {
    init(env);
    env->observations = (float*)calloc(4 + NUM_LANES*MAX_ENEMIES_PER_LANE, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void c_close(Freeway* env) {
    free(env->human_actions);
    free(env->enemies);
}

void free_allocated(Freeway* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(Freeway* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->ep_return;
    env->log.score += env->ai_player.score;
    env->log.perf += env->ai_player.score / ((float) HUMAN_HIGH_SCORE[env->level] * (GAME_LENGTH / 136.0f ));
    env->log.up_action_frac += env->up_count / (float) env->tick;
    env->log.hits += env->ai_player.hits;
    env->log.n += 1;
}

void compute_observations(Freeway* env) {
    env->observations[0] = env->ai_player.player_y / env->height;
    env->observations[1] = env->ai_player.best_lane_idx /(float) NUM_LANES;
    env->observations[2] = env->ai_player.score / (float) HUMAN_HIGH_SCORE[env->level];
    env->observations[3] = (env->ai_player.ticks_stunts_left  > 0);

    FreewayEnemy* enemy;
    for (int lane = 0; lane < NUM_LANES; lane++) {
        for (int i = 0; i < MAX_ENEMIES_PER_LANE; i++){
            enemy = &env->enemies[lane*MAX_ENEMIES_PER_LANE + i];
            if (enemy->is_enabled){
                env->observations[4 + lane * MAX_ENEMIES_PER_LANE + i] = enemy->enemy_x / env->width;
                env->observations[4 + lane * MAX_ENEMIES_PER_LANE + i] += (lane < NUM_LANES/2 ? enemy->enemy_height/(2 * env->width): -enemy->enemy_height/(2 * env->width));
            }
            else {
                env->observations[4 + lane * MAX_ENEMIES_PER_LANE + i] = 0.0f;
            }
        }
    }   
}

void spawn_enemies(Freeway* env) {
    float lane_offset_x;
    FreewayEnemy* enemy;
    for (int lane = 0; lane < NUM_LANES; lane++) {
        lane_offset_x =  env->width * (rand() / (float) RAND_MAX);
        for (int i = 0; i < MAX_ENEMIES_PER_LANE; i++){
            enemy = &env->enemies[lane * MAX_ENEMIES_PER_LANE + i];
            if (enemy->is_enabled){
                enemy->enemy_x = enemy->enemy_initial_x;
                if (lane>=NUM_LANES/2){
                    enemy->enemy_x = env->width - enemy->enemy_x;
                }
                if (env->env_randomization){
                    enemy->enemy_x += lane_offset_x;
                }
            }
        }
    }
}

void reset_player(Freeway* env, FreewayPlayer* player) {
    player->player_y = env->height - env->player_height / 2;
    player->ticks_stunts_left = 0;
}

bool check_collision(float player_min_x, float player_max_x,
                      float player_miny, float player_maxy,
                      float enemy_minx, float enemy_maxx,
                      float enemy_miny, float enemy_maxy) {
    return (player_min_x < enemy_maxx && player_max_x > enemy_minx &&
            player_miny < enemy_maxy && player_maxy > enemy_miny);
}

bool check_enemy_collisions(Freeway* env, FreewayPlayer* player){
    FreewayEnemy* enemy;
    for (int lane = 0; lane < NUM_LANES; lane++) {
        for (int i = 0; i < MAX_ENEMIES_PER_LANE; i++) {
            enemy = &env->enemies[lane*MAX_ENEMIES_PER_LANE + i];
            if (enemy->is_enabled) {
                float player_min_x = player->player_x - env->player_width / 2;
                float player_max_x = player->player_x + env->player_width / 2;
                float player_miny = player->player_y - env->player_height / 2;
                float player_maxy = player->player_y + env->player_height / 2;

                float enemy_minx = enemy->enemy_x - enemy->enemy_width / 2;
                float enemy_maxx = enemy->enemy_x + enemy->enemy_width / 2;
                float enemy_miny = enemy->enemy_y - enemy->enemy_height / 2;
                float enemy_maxy = enemy->enemy_y + enemy->enemy_height / 2;

                if (check_collision(player_min_x, player_max_x, player_miny, player_maxy,
                                    enemy_minx, enemy_maxx, enemy_miny, enemy_maxy)) {
                    return true;
                }
            }
        }
    }
    return false;
}

void reached_end(Freeway* env, FreewayPlayer* player){
    reset_player(env, player);
    player->best_lane_idx = 0;
    player->score += 1;
}

void clip_player_position(Freeway* env, FreewayPlayer* player){
    if (player->player_y <= env->player_height/2){
        player->player_y = fmaxf(env->player_height/2, player->player_y);
    } else {
        player->player_y = fminf(env->height - env->player_height/2, player->player_y);
    }
}

void clip_enemy_position(Freeway* env, FreewayEnemy* enemy){
    if (enemy->enemy_x > env->width + enemy->enemy_width / 2) {
        enemy->enemy_x -= env->width;
    }
    else if (enemy->enemy_x < -enemy->enemy_width / 2){
        enemy->enemy_x += env->width;
    }
}

void randomize_enemy_speed(Freeway* env) {
    FreewayEnemy* enemy;
    for (int lane = 0; lane < NUM_LANES; lane++) {
        int delta_speed = (rand() % 3) - 1; // Randomly increase or decrease speed
        for (int i = 0; i < MAX_ENEMIES_PER_LANE; i++) {
            if (enemy->speed_randomization) {
                enemy = &env->enemies[lane*MAX_ENEMIES_PER_LANE + i];
                enemy->current_speed_idx = min(max(enemy->initial_speed_idx-2, enemy->current_speed_idx), enemy->initial_speed_idx+2);
                enemy->current_speed_idx = min(max(0, enemy->current_speed_idx + delta_speed), 5);
                enemy->enemy_vx = enemy->lane_idx < NUM_LANES/2 ? SPEED_VALUES[enemy->current_speed_idx] * TICK_RATE * env->width: -SPEED_VALUES[enemy->current_speed_idx] * TICK_RATE * env->width;
            }
        }
    }
}

void move_enemies(Freeway* env) {
    FreewayEnemy* enemy;
    for (int lane = 0; lane < NUM_LANES; lane++) {
        for (int i = 0; i < MAX_ENEMIES_PER_LANE; i++) {
            enemy = &env->enemies[lane*MAX_ENEMIES_PER_LANE + i];
            if (enemy->is_enabled) {
                enemy->enemy_x += enemy->enemy_vx;
            }
            clip_enemy_position(env, enemy);
        }
    }
}

void step_player(Freeway* env, FreewayPlayer* player, int action) {
    float player_dy = 0.0;

    if (action == DOWN) {
        player_dy = -1.0;
    } 
    else if (action == UP) {
        player_dy = 1.0;
        env->up_count += 1;
    }

    if (player->ticks_stunts_left == 0){
        player->player_y -= player_dy * BASE_PLAYER_SPEED * env->height * TICK_RATE;
    }
    else {
        player->ticks_stunts_left -= 1;
        if (env->difficulty == 0){
            player->player_y += 1.5f * env->lane_size / (float) TICKS_STUNT;
        } 
    }
    clip_player_position(env, player);
    
    if (player->ticks_stunts_left == 0) {
        if (check_enemy_collisions(env, player) && player->ticks_stunts_left < TICKS_STUNT/4){
            player->hits+=1;
            player->ticks_stunts_left = TICKS_STUNT;
            if (env->use_dense_rewards){
                env->rewards[0] += PENALTY_HIT;
                env->ep_return += PENALTY_HIT;
            }
            if (env->difficulty == 1){
                reset_player(env, player);
            }  
        }
    }

    if (player->player_y <= env->road_start - (player->best_lane_idx+1) * env->lane_size){
        player->best_lane_idx += 1; 
        if (env->use_dense_rewards){
            env->rewards[0] += 1.0 / (float) NUM_LANES;
            env->ep_return += 1.0 / (float) NUM_LANES;
        }
        else{
            if (player->best_lane_idx == NUM_LANES){
                env->rewards[0] = 1.0;
                env->ep_return += 1.0;
            }
        }
    }

    if (player->best_lane_idx == NUM_LANES) {
        reached_end(env, player);
        env->rewards[0] += 1.0;
        env->ep_return += 1.0;
    }
}
void c_reset(Freeway* env) {
    env->ai_player.player_y = env->height / 2;
    env->ai_player.best_lane_idx = 0;
    env->ai_player.ticks_stunts_left = 0;
    env->ai_player.score = 0;
    env->ai_player.hits = 0;

    env->human_player.player_y = env->height / 2;
    env->human_player.best_lane_idx = 0;
    env->human_player.ticks_stunts_left = 0;
    env->human_player.score = 0;
    env->human_player.hits = 0;

    env->ep_return = 0.0;
    env->tick = 0;
    env->up_count = 0;
    env->time_left = GAME_LENGTH;
    reset_player(env, &env->ai_player);
    reset_player(env, &env->human_player);
    spawn_enemies(env);
    compute_observations(env);
}

void c_step(Freeway* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
    int ai_action = env->actions[0];
    int human_action = env->human_actions[0];
    env->time_left = GAME_LENGTH - env->tick*TICK_RATE;

    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_player(env, &env->ai_player, ai_action);
        if (env->enable_human_player){
            step_player(env, &env->human_player, human_action);
        }
        move_enemies(env);
    }
    if (env->tick * TICK_RATE >= GAME_LENGTH) {
        env->terminals[0] = 1.0;
        add_log(env);
        c_reset(env);
    }
    if (env->tick % RANDOMIZE_SPEED_FREQ == 0) {
        randomize_enemy_speed(env);
    }
    compute_observations(env);
}



typedef struct Client Client;
struct Client {
    Texture2D chicken;
    Texture2D puffer;
    Texture2D car_body;
    Texture2D car_wheels;
    Texture2D truck_body;
    Texture2D truck_wheels;
};

static inline bool file_exists(const char* path) {
    return access(path, F_OK) != -1;
}

Client* make_client(Freeway* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    
    InitWindow(env->width, env->height, "PufferLib Freeway");
    SetTargetFPS(60/env->frameskip);
    client->car_body = LoadTexture("resources/freeway/tex_car_body.png");
    client->car_wheels = LoadTexture("resources/freeway/tex_car_wheels.png");
    client->chicken = LoadTexture("resources/freeway/tex_chicken0.png");
    client->puffer = LoadTexture("resources/shared/puffers.png");
    client->truck_body = LoadTexture("resources/freeway/tex_truck_body.png");
    client->truck_wheels = LoadTexture("resources/freeway/tex_truck_wheels.png");
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

Color CAR_COLORS[10] = {
    (Color){ 139, 0, 0, 255 },      // Dark Red
    (Color){ 255, 140, 0, 255 },    // Dark Orange
    (Color){ 204, 204, 0, 255 },    // Dark Yellow
    (Color){ 0, 100, 0, 255 },      // Dark Green
    (Color){ 0, 0, 139, 255 },      // Dark Blue
    (Color){ 139, 0, 0, 255 },      // Dark Red
    (Color){ 255, 140, 0, 255 },    // Dark Orange
    (Color){ 204, 204, 0, 255 },    // Dark Yellow
    (Color){ 0, 100, 0, 255 },      // Dark Green
    (Color){ 0, 0, 139, 255 }       // Dark Blue
};
void c_render(Freeway* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    Client* client = env->client;

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    BeginDrawing();
    ClearBackground((Color){170, 170, 170, 255});
    
    // Draw the road
    DrawRectangle(
        0, 
        env->road_end,
        env->width, env->road_start - env->road_end, (Color){150, 150, 150, 255}
    );
    DrawRectangle(
        0, 
        env->road_end- HALF_LINEWIDTH,
        env->width, 2*HALF_LINEWIDTH, (Color){0, 255}
    );
    DrawRectangle(
        0, 
        env->road_start - HALF_LINEWIDTH,
        env->width, 2*HALF_LINEWIDTH, (Color){0, 255}
    );

    for (int lane = 1; lane < NUM_LANES; lane++) {
        if (lane != NUM_LANES/2){
            for (int dash = 0; dash < env->width / (DASH_SPACING + DASH_SIZE) ; dash++){
                int dash_start = DASH_SPACING / 2 + (DASH_SPACING + DASH_SIZE) * dash;            
                DrawRectangle(
                    dash_start, 
                    env->road_start + (env->road_end - env->road_start) * lane/NUM_LANES - HALF_LINEWIDTH,
                    DASH_SIZE, 2*HALF_LINEWIDTH, (Color){235, 235, 235, 255}
                );
            }
        }
    }

    for (int dash = 0; dash < env->width / (DASH_SPACING + DASH_SIZE) ; dash++){
        int dash_start = DASH_SPACING / 2 + (DASH_SPACING + DASH_SIZE) * dash;            
        DrawRectangle(
            dash_start, 
            env->road_start + (env->road_end - env->road_start) / 2 - 3*HALF_LINEWIDTH,
            DASH_SIZE, 2*HALF_LINEWIDTH, (Color){235, 235, 100, 255}
        );
        DrawRectangle(
            dash_start, 
            env->road_start + (env->road_end - env->road_start) / 2 + HALF_LINEWIDTH,
            DASH_SIZE, 2*HALF_LINEWIDTH, (Color){235, 235, 100, 255}
        );
    }
    
    // Draw ai player
    DrawTexturePro(
        client->puffer,
        (Rectangle){
            0, 0, 128, 128,
        },
        (Rectangle){
            env->ai_player.player_x - env->player_width / 2,
            env->ai_player.player_y - env->player_height / 2,
            env->player_width,
            env->player_height,
        },
        (Vector2){0, 0},
        0,
        WHITE
    );

    DrawTexturePro(
        client->puffer,
        (Rectangle){
            128, 128, 128, 128,
        },
        (Rectangle){
            env->human_player.player_x - env->player_width / 2,
            env->human_player.player_y - env->player_height / 2,
            env->player_width,
            env->player_height,
        },
        (Vector2){0, 0},
        0,
        WHITE
    );

    // Draw enemies
    Rectangle src_rec;
    FreewayEnemy* enemy;
    for (int lane = 0; lane < NUM_LANES; lane++) {
        for (int i = 0; i < MAX_ENEMIES_PER_LANE; i++) {
            enemy = &env->enemies[lane*MAX_ENEMIES_PER_LANE + i];
            if (enemy->is_enabled) {
                Texture2D body = enemy->type == 0 ? client->car_body : client->truck_body;
                Texture2D wheels = enemy->type == 0 ? client->car_wheels : client->truck_wheels;
                if (lane < NUM_LANES/2) {
                    src_rec= enemy->type == 0 ? (Rectangle){16,0,16,10} : (Rectangle){32,10,32,10};
                }
                else {
                    src_rec = enemy->type == 0 ? (Rectangle){16 + 16, 0, -16, 10} : (Rectangle){32 + 32, 10, -32, 10};
                }
                DrawTexturePro(
                    body,
                    src_rec,
                    (Rectangle){
                        enemy->enemy_x - enemy->enemy_width / 2, 
                        enemy->enemy_y - enemy->enemy_height/ 2,
                        enemy->enemy_width, 
                        enemy->enemy_height,
                    },
                    (Vector2){0, 0},
                    0,
                    CAR_COLORS[lane]
                );
                DrawTexturePro(
                    wheels,
                    src_rec,
                    (Rectangle){
                        enemy->enemy_x - enemy->enemy_width / 2, 
                        enemy->enemy_y - enemy->enemy_height/ 2,
                        enemy->enemy_width, 
                        enemy->enemy_height,
                    },
                    (Vector2){0, 0},
                    0,
                    CAR_COLORS[lane]
                );
            }
        }
    }

    // Draw UI
    int rounded_time_left = round(env->time_left);
    DrawText(TextFormat("P1 Score: %i", env->ai_player.score), 10, 3, 40, (Color) {255, 160, 160, 255});
    DrawText(TextFormat("P2 Score: %i", env->human_player.score), round(0.77*env->width), 3, 40, (Color) {255, 160, 160, 255});
    DrawText(TextFormat("Time: %i", rounded_time_left), round(0.45*env->width) - 40, 3, 40, (Color) {255, 160, 160, 255});

    EndDrawing();

    //PlaySound(client->sound);
}
