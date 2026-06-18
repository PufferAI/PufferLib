#include <limits.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include <stdio.h>

// Action Space
const unsigned char NOOP = 0;
const unsigned char UP = 1;

#define SCALE_X 1.0f
#define SCALE_Y 1.0f
#define SCREEN_WIDTH (288.0f * SCALE_X)
#define SCREEN_HEIGHT (512.0f * SCALE_Y)
#define BIRD_WIDTH (34.0f * SCALE_X)
#define BIRD_HEIGHT (24.0f * SCALE_Y)
#define DIGIT_WIDTH (24.0f * SCALE_X)
#define DIGIT_HEIGHT (36.0f * SCALE_Y)
#define BASE_WIDTH SCREEN_WIDTH
#define BASE_HEIGHT (SCREEN_HEIGHT / 4.0f)
#define PIPE_GAP_Y (SCREEN_HEIGHT * 0.2f * SCALE_Y)
#define PIPE_WIDTH (52.0f * SCALE_X)
#define PIPE_HEIGHT (((SCREEN_HEIGHT - BASE_HEIGHT) / 2) * SCALE_Y)
#define PIPE_GAP_X ( (145 + PIPE_WIDTH) * SCALE_X)
#define PIPE_SPEED 2
#define MAX_PIPES 3

#define BIRD_START_POS_X (SCREEN_WIDTH * 0.3f)
#define BIRD_START_POS_Y (SCREEN_HEIGHT * 0.5f)
#define PIPE_START_POS_X SCREEN_WIDTH
#define PIPE_START_POS_Y (SCREEN_HEIGHT * 0.5f)
#define SCORE_POS_X (SCREEN_WIDTH * 0.5f)
#define SCORE_POS_Y (SCREEN_HEIGHT * 0.1f)
#define BASE_POS_X 0.0f
#define BASE_POS_Y (SCREEN_HEIGHT / 4.0f * 3.0f)	

// For Game Mechanics
#define JUMP_VELOCITY 5.0f
#define GRAVITY 0.5f
#define MAX_FALL_SPEED 7.0f
#define BIRD_TILT_UP_ANGLE -30.0f
#define BIRD_TILT_DOWN_ANGLE 70.0f
#define BIRD_TILT_LERP 0.15f
#define BIRD_HITBOX_INSET_X 4.0f
#define BIRD_HITBOX_INSET_TOP 0.0f
#define BIRD_HITBOX_INSET_BOTTOM 10.0f
#define TILT_UP_HITBOX_SCALE 0.45f
#define TOP_HIT_LEAD 4.0f
#define BOTTOM_GAP_FORGIVENESS 7.0f
#define SCORE_MAX_DIGITS 10

#define OBS_SIZE (2 + 4 * MAX_PIPES)

typedef struct {
    float x;
    float y;
    float h;
    float w;
} Pipe;

typedef struct {
    Pipe bottom_pipe;
    Pipe top_pipe;
    float gap_top;
    float gap_bottom;
    bool active;
    bool scored;
} PairOfPipe;

typedef struct {
    float x;
    float y;
    float h;
    float w;
} Base;

typedef struct {
    float x;
    float y;
    float h;
    float w;
    int score;
} ScoreBoard;

typedef enum {
    STATE_DOWNWARD,
    STATE_STATIONARY,
    STATE_UPWARD
} BirdState;

typedef struct {
    float x;
    float y;
    float h;
    float w;
    float y_velocity;
    float angle;
    BirdState state;
} Bird;

typedef struct {
    Texture2D bird_up;
    Texture2D bird_down;
    Texture2D bird_mid;
    Texture2D pipe;
    Texture2D base;
    Texture2D bg;
    Texture2D digits[10];
} Renderer;

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported in binding.c
    float n; // Required as the last field
} Log;

// Required that you have some struct for your env
typedef struct {
    // Required Attributes
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int tick;
    unsigned int rng;
    // Required Attributes for FlappyBird Env.
    Renderer* renderer;
    Bird* bird;
    PairOfPipe* pipes;
    Base* base;
    ScoreBoard* sb;
    int idx_to_spawn;
    int action;
    int distance_to_spawn;
    int pipe_speed;
    float gravity;
    float reward_pipe;
    float reward_death;
    float episode_return_acc;

} FlappyBird;

void add_logs(FlappyBird* env) {
    env->log.episode_return += env->tick / 90.0f + env->sb->score * 1.0f - 2.0f;
    env->log.episode_length += env->tick;
    env->log.score += env->tick / 90.0f + env->sb->score * 1.0f - 2.0f;
    env->log.perf += env->tick / 90.0f + env->sb->score * 1.0f - 2.0f;
    env->log.n++;
}

Renderer* init_renderer_load_textures(){
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Testing FlappyBird");
    SetTargetFPS(60);

    Renderer* renderer = (Renderer*) calloc(1, sizeof(Renderer));
    
    renderer->bg = LoadTexture("resources/flappy_bird/flappy-bird-assets-master/sprites/background-day.png");
    renderer->base = LoadTexture("resources/flappy_bird/flappy-bird-assets-master/sprites/base.png");
    renderer->pipe = LoadTexture("resources/flappy_bird/flappy-bird-assets-master/sprites/pipe-green.png");
    renderer->bird_mid = LoadTexture("resources/flappy_bird/flappy-bird-assets-master/sprites/yellowbird-midflap.png");
    renderer->bird_down = LoadTexture("resources/flappy_bird/flappy-bird-assets-master/sprites/yellowbird-downflap.png");
    renderer->bird_up = LoadTexture("resources/flappy_bird/flappy-bird-assets-master/sprites/yellowbird-upflap.png");
    for (int i = 0 ; i < 10 ; i++){
        renderer->digits[i] = LoadTexture(TextFormat("resources/flappy_bird/flappy-bird-assets-master/sprites/%d.png", i));
    }
    return renderer;
}

void reset_pipes(FlappyBird* env) {
    for (int i = 0; i < MAX_PIPES; i++){
        env->pipes[i].top_pipe.x = 0;
        env->pipes[i].top_pipe.y = 0;
        env->pipes[i].top_pipe.w = 0;
        env->pipes[i].top_pipe.h = 0;
        
	    env->pipes[i].bottom_pipe.x = 0;
        env->pipes[i].bottom_pipe.y = 0;
        env->pipes[i].bottom_pipe.w = 0;
        env->pipes[i].bottom_pipe.h = 0;
        env->pipes[i].active = 0;
        env->pipes[i].scored = 0;
    }
}

void spawn_pipe(FlappyBird* env) {
    if (env->pipes[env->idx_to_spawn].active == 0){
    	float floor_y = SCREEN_HEIGHT - BASE_HEIGHT;
    	float pipe_h = SCREEN_HEIGHT;

    	float margin = 30.0f * SCALE_Y;
    	float min_gap_top = margin;
    	float max_gap_top = floor_y - PIPE_GAP_Y - margin;

    	float gap_top = min_gap_top +(rand_r(&env->rng) % ((int)(max_gap_top - min_gap_top + 1)));
	    float gap_bottom = gap_top + PIPE_GAP_Y;
	    float pipe_x = PIPE_START_POS_X;
     
        // Top pipe   
        env->pipes[env->idx_to_spawn].top_pipe.x = pipe_x;
        env->pipes[env->idx_to_spawn].top_pipe.y = gap_top - pipe_h;
        env->pipes[env->idx_to_spawn].top_pipe.w = PIPE_WIDTH;
        env->pipes[env->idx_to_spawn].top_pipe.h = pipe_h;
        // Bottom pipe   
        env->pipes[env->idx_to_spawn].bottom_pipe.x = pipe_x;
        env->pipes[env->idx_to_spawn].bottom_pipe.y = gap_bottom;
        env->pipes[env->idx_to_spawn].bottom_pipe.w = PIPE_WIDTH;
    	env->pipes[env->idx_to_spawn].bottom_pipe.h = pipe_h;
    	env->pipes[env->idx_to_spawn].gap_top = gap_top;
    	env->pipes[env->idx_to_spawn].gap_bottom = gap_bottom;
    	env->pipes[env->idx_to_spawn].scored = 0;
    	env->pipes[env->idx_to_spawn].active = 1;       
    }

}

void compute_observations(FlappyBird* env){
    int obs_idx = 0;
    memset(env->observations, 0.0f, OBS_SIZE * sizeof(float));
    // Bird y and y_velocity
    env->observations[obs_idx++] = env->bird->y / SCREEN_HEIGHT;
    env->observations[obs_idx++] = env->bird->y_velocity/fmaxf(fabsf(JUMP_VELOCITY), MAX_FALL_SPEED);
    // PairOfPipe provide 
    // 	- is_active_and_not_scored
    //	- pipe_x, which is the same for both top and bottom pipes
    //	- gap_top
    //	- gap_bottom 
    for(int i = 0; i < MAX_PIPES ; i++){
        PairOfPipe* pair = &env->pipes[i];
        if (pair->active && !pair->scored) {
            env->observations[obs_idx++] = 1.0f;
            env->observations[obs_idx++] = (pair->top_pipe.x) / SCREEN_WIDTH;
            env->observations[obs_idx++] = (pair->gap_top + TOP_HIT_LEAD) / SCREEN_HEIGHT;
            env->observations[obs_idx++] = (pair->gap_bottom + BOTTOM_GAP_FORGIVENESS) / SCREEN_HEIGHT;
        }else{
            env->observations[obs_idx++] = -1.0f; 
            env->observations[obs_idx++] = 0.0f;
            env->observations[obs_idx++] = 0.0f;
            env->observations[obs_idx++] = 0.0f;
        } 
    }
    return;		
}


void c_reset(FlappyBird* env){
    // Move the bird to the initial height
    env->bird->x = BIRD_START_POS_X;
    env->bird->y = BIRD_START_POS_Y;
    env->bird->w = BIRD_WIDTH;
    env->bird->h = BIRD_HEIGHT;
    env->bird->y_velocity = 0.0f;
    env->bird->angle = 0.0f;
    env->bird->state = STATE_STATIONARY;

    // Reset all the Pipes
    reset_pipes(env);

    // Reset the ScoreBoard
    env->sb->x = SCORE_POS_X - DIGIT_WIDTH / 2;
    env->sb->y = SCORE_POS_Y - DIGIT_HEIGHT;
    env->sb->score = 0;

    // Reset game
    env->distance_to_spawn = 0;
    env->idx_to_spawn = 0;
    env->tick = 0;

    compute_observations(env);
}

// Initialise the parameters of the env
void c_init(FlappyBird* env){
    env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
    env->bird = (Bird*)calloc(1, sizeof(Bird));
    env->gravity = GRAVITY;
    env->pipe_speed = PIPE_SPEED;
    env->renderer = NULL;

    env->base = (Base*)calloc(1, sizeof(Base));
    env->base->x = BASE_POS_X;
    env->base->y = BASE_POS_Y;
    env->base->w = BASE_WIDTH;
    env->base->h = BASE_HEIGHT;

    env->pipes = (PairOfPipe*)calloc(MAX_PIPES, sizeof(PairOfPipe));
    env->distance_to_spawn = 0;
    env->rng = (unsigned int)time(NULL);
    env->sb = (ScoreBoard*)calloc(1, sizeof(ScoreBoard));

    env->sb->x = SCORE_POS_X - DIGIT_WIDTH / 2;
    env->sb->y = SCORE_POS_Y - DIGIT_HEIGHT;
    env->sb->w = DIGIT_WIDTH; 
    env->sb->h = DIGIT_HEIGHT;
}

void process_pipe_despawns(FlappyBird* env){
    float out_of_bounds = 0.0f - PIPE_WIDTH;
    for (int j = 0; j < MAX_PIPES; j++){
        if (env->pipes[j].top_pipe.x <= out_of_bounds){
	        env->pipes[j].active = 0;	
	    } 
    }
}

void process_pipe_spawns(FlappyBird* env){
    float pipe_speed = (float)env->pipe_speed;
    env->distance_to_spawn -= pipe_speed;
    if (env->distance_to_spawn <= 0){
        spawn_pipe(env);
    	env->idx_to_spawn = (env->idx_to_spawn + 1) % MAX_PIPES;
	    env->distance_to_spawn = PIPE_GAP_X; 
    }
    for (int i = 0; i < MAX_PIPES; i++){
        if (env->pipes[i].active){
            env->pipes[i].top_pipe.x -= pipe_speed;
            env->pipes[i].bottom_pipe.x -= pipe_speed;
	    }
    } 
}

void process_bird_movements(FlappyBird* env){
    // Given Action of the model change the state
    // We should always be updating the y_velocity;
    // Just like in real life you accelerate up to a constant velocity, in the game sense, we directly go to constant velocity of 10 pixels
    if (env->action == UP){
        env->bird->y_velocity = -1.0f * JUMP_VELOCITY; 
    }
    // Apply Gravity
    env->bird->y_velocity += env->gravity;
    // Just like in reallife you will reach terminal velocity, at some point
    if (env->bird->y_velocity > MAX_FALL_SPEED){
        env->bird->y_velocity = MAX_FALL_SPEED;
    }
    // Update bird state for pixel animation
    if (env->bird->y_velocity > 1.0f){
        env->bird->state = STATE_DOWNWARD;
    } else if (env->bird->y_velocity < -1.0f){
        env->bird->state = STATE_UPWARD;
    } else {
        env->bird->state = STATE_STATIONARY;
    }
    float target_angle = 0.0f;
    if (env->bird->y_velocity < 0.0f) {
        float t = env->bird->y_velocity / (-JUMP_VELOCITY);
        if (t > 1.0f) t = 1.0f;
        target_angle = t * BIRD_TILT_UP_ANGLE;
    } else if (env->bird->y_velocity > 0.0f) {
        float t = env->bird->y_velocity / MAX_FALL_SPEED;
        if (t > 1.0f) t = 1.0f;
        target_angle = t * BIRD_TILT_DOWN_ANGLE;
    }
    env->bird->angle += (target_angle - env->bird->angle) * BIRD_TILT_LERP;
    env->bird->y += env->bird->y_velocity;
}

void process_collision(FlappyBird* env){
    Bird* bird = env->bird;
    float left_x = bird->x + BIRD_HITBOX_INSET_X;
    float right_x = bird->x + bird->w - BIRD_HITBOX_INSET_X;
    // Top: small inset + extend upward when nose is up (sprite leads the upright box)
    float top_y = bird->y + BIRD_HITBOX_INSET_TOP;
    if (bird->angle < 0.0f) {
        top_y += bird->angle * TILT_UP_HITBOX_SCALE;
    }
    // Bottom: larger inset + gap forgiveness (sprite lags the upright box when diving)
    float bottom_y = bird->y + bird->h - BIRD_HITBOX_INSET_BOTTOM;
    if (bottom_y > BASE_POS_Y) {
        *env->terminals = 1.0f;
        *env->rewards -= 2.0f;
	    add_logs(env);
        c_reset(env);
        return;
    }

    for (int i = 0; i < MAX_PIPES; i++) {
        PairOfPipe* pair = &env->pipes[i];
        if (!pair->active) {
            continue;
        }
        float pipe_left = pair->top_pipe.x;
        float pipe_right = pipe_left + pair->top_pipe.w;
        if (right_x <= pipe_left || left_x >= pipe_right) {
            continue;
        }
        if (top_y < pair->gap_top + TOP_HIT_LEAD) {
            *env->terminals = 1.0f;
            *env->rewards -= 2.0f;
            add_logs(env);
            c_reset(env);
            return;
        }
        if (bottom_y > pair->gap_bottom + BOTTOM_GAP_FORGIVENESS) {
            *env->terminals = 1.0f;
            *env->rewards -= 2.0f;
            add_logs(env);
            c_reset(env);
            return;
	    }
    }
}

void process_score(FlappyBird* env){
    float bird_mid_x = env->bird->x + env->bird->w * 0.5f;
    for (int i = 0; i < MAX_PIPES; i++) {
        PairOfPipe* pair = &env->pipes[i];
        if (!pair->active || pair->scored) {
            continue;
        }
        float pipe_mid_x = pair->top_pipe.x + pair->top_pipe.w * 0.5f;
        if (bird_mid_x >= pipe_mid_x) {
            pair->scored = 1;
            if (env->sb->score < INT_MAX) {
                env->sb->score += 1;
            }
            *env->rewards += 1.0f;
        }
    }
}

void c_step(FlappyBird* env){
    env->tick += 1;
    *env->rewards += 0.0111f; 
    *env->terminals = 0.0f;

    int action = (int)env->actions[0];
    if (action){
         env->action = UP;
    }else{
	    env->action = NOOP;
    }
    process_pipe_despawns(env);
    process_pipe_spawns(env);
    process_bird_movements(env);
    process_score(env);
    process_collision(env);
    process_score(env);
    compute_observations(env);
}

static void draw_scoreboard(FlappyBird* env) {
    int score = env->sb->score;
    if (score < 0) {
        score = 0;
    }
    int digit_values[SCORE_MAX_DIGITS];
    int num_digits = 0;
    if (score == 0) {
        digit_values[num_digits++] = 0;
    } else {
        int tmp = score;
        while (tmp > 0 && num_digits < SCORE_MAX_DIGITS) {
            digit_values[num_digits++] = tmp % 10;
            tmp /= 10;
        }
    }
    float total_w = num_digits * DIGIT_WIDTH;
    float draw_x = SCORE_POS_X - total_w * 0.5f;
    for (int d = num_digits - 1; d >= 0; d--) {
        int digit = digit_values[d];
        Texture2D digit_tex = env->renderer->digits[digit];
        float x = draw_x + (num_digits - 1 - d) * DIGIT_WIDTH;
        DrawTexturePro(
            digit_tex,
            (Rectangle){0.0f, 0.0f, digit_tex.width, digit_tex.height},
            (Rectangle){x, env->sb->y, DIGIT_WIDTH, DIGIT_HEIGHT},
            (Vector2){0, 0},
            0.0f,
            WHITE
        );
    }
}

void c_render(FlappyBird* env){
    if (env->renderer == NULL){
        env->renderer = init_renderer_load_textures();   
    }

    if(IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    BeginDrawing();
    ClearBackground(RAYWHITE);

    // Draw Background
    Texture2D bg_tex = env->renderer->bg;
    DrawTexturePro(
		   bg_tex,
		   (Rectangle){0.0f, 0.0f, bg_tex.width, bg_tex.height },
		   (Rectangle){0.0f, 0.0f, SCREEN_WIDTH, SCREEN_HEIGHT}, 
		   (Vector2){0, 0}, 
		   0.0f, 
		   WHITE
    );
    // Draw Pipe
    Texture2D pipe = env->renderer->pipe;
    for (int i = 0 ; i < MAX_PIPES; i++){
        if (env->pipes[i].active){
                DrawTexturePro(
                    pipe,
                    (Rectangle){ 0.0f, 0.0f, pipe.width, pipe.height },
                    (Rectangle){ env->pipes[i].bottom_pipe.x, env->pipes[i].bottom_pipe.y, env->pipes[i].bottom_pipe.w, env->pipes[i].bottom_pipe.h}, // POS_Y is relative to the base
                    (Vector2){0, 0}, 
                    0.0f, 
                    WHITE
            );
            DrawTexturePro(
                    pipe,
                    (Rectangle){ 0.0f, 0.0f, pipe.width, -pipe.height },
                    (Rectangle){ env->pipes[i].top_pipe.x, env->pipes[i].top_pipe.y, env->pipes[i].top_pipe.w, env->pipes[i].top_pipe.h}, 
                    (Vector2){0, 0}, 
                    0.0f, 
                    WHITE
                );   
        }
    }
    // Draw Base		
    Texture2D base_tex = env->renderer->base;
    DrawTexturePro(
		   base_tex,
		   (Rectangle){ 0.0f, 0.0f, base_tex.width, base_tex.height },
		   (Rectangle){ env->base->x, env->base->y, env->base->w, env->base->h}, 
		   (Vector2){0, 0}, 
		   0.0f, 
		   WHITE
    );   
    // Draw Bird — sprite picks wing pose, angle adds trajectory tilt
    Texture2D bird_tex;
    switch (env->bird->state) {
        case STATE_DOWNWARD: bird_tex = env->renderer->bird_down; break;
        case STATE_UPWARD:   bird_tex = env->renderer->bird_up;   break;
        default:             bird_tex = env->renderer->bird_mid;  break;
    }
    Vector2 bird_origin = {env->bird->w * 0.5f, env->bird->h * 0.5f};
    DrawTexturePro(
        bird_tex,
        (Rectangle){0.0f, 0.0f, bird_tex.width, bird_tex.height},
        (Rectangle){env->bird->x, env->bird->y, env->bird->w, env->bird->h},
        bird_origin,
        env->bird->angle,
        WHITE
    );
	
    draw_scoreboard(env);
    EndDrawing();
}


void c_close(FlappyBird* env){
	free(env->bird);
	free(env->pipes);
	free(env->renderer);
	free(env->base);
	free(env->sb);
    if(env->renderer != NULL){
    	UnloadTexture(env->renderer->bg);        
    	UnloadTexture(env->renderer->bird_mid);        
    	UnloadTexture(env->renderer->bird_up);        
    	UnloadTexture(env->renderer->bird_down);        
    	UnloadTexture(env->renderer->base);        
    	UnloadTexture(env->renderer->pipe);       
    	for (int i = 0 ; i < 10 ; i++){
	   	UnloadTexture(env->renderer->digits[i]);  
    	}
	    CloseWindow();
	    free(env->renderer);
	    env->renderer = NULL;
    }    
}