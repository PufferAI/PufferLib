#include "raylib.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// CONFIG
#define REF_W 48
#define REF_H REF_W
#define REF_U 1.5 // ground height
#define REF_WALL_WIDTH 1.0
#define REF_WALL_HEIGHT 3.5
#define PLAYER_SPEED_X (10*1.75)
#define PLAYER_SPEED_Y (10*1.35)
#define MAX_BALL_SPEED (15*1.5)
#define TIMESTEP (1.0/30.0)
#define NUDGE 0.1
#define FRICTION 1.0 // 1 means no FRICTION, less means FRICTION. (should be called elasticity imo)
#define INIT_DELAY_FRAMES 30
#define GRAVITY (-9.8*2*1.5)
#define MAXLIVES 5 // game ends when one agent loses this many games
#define WINDOW_WIDTH 1200
#define WINDOW_HEIGHT 500
#define FACTOR (WINDOW_WIDTH / REF_W)
#define PIXEL_MODE false
#define PIXEL_SCALE 4
#define PIXEL_WIDTH (84*2)
#define PIXEL_HEIGHT 84
#define MAX_TICKS 3000

// Day colors
const Color BALL_COLOR = {255, 200, 20, 255};
const Color AGENT_LEFT_COLOR = {240, 75, 0, 255};
const Color AGENT_RIGHT_COLOR = {0, 150, 255, 255};
const Color PIXEL_AGENT_LEFT_COLOR = {240, 75, 0, 255};
const Color PIXEL_AGENT_RIGHT_COLOR = {0, 150, 255, 255};
const Color BACKGROUND_COLOR = {255, 255, 255, 255};
const Color FENCE_COLOR = {240, 210, 130, 255};
const Color COIN_COLOR = {240, 210, 130, 255};
const Color GROUND_COLOR = {128, 227, 153, 255};

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// UTILS
typedef struct {
    float x;
    float y;
    float r;
    float vx;
    float vy;
} SphericalObject;


// convert from space to pixel coordinates
float to_x_pixel(float x){
    return (x + REF_W/2) * FACTOR;
}

float to_p(float x) {
    return x * FACTOR;
}

float to_y_pixel(float y){
    return WINDOW_HEIGHT - y * FACTOR;
}

float randf() {
    return (float)rand() / (float)RAND_MAX;
}

// OBJECTS
typedef struct {
    float x;
    float y;
    float r;
    float vx;
    float vy;
    float prev_x;
    float prev_y;
    Color c;
} Ball;

void ball_move(Ball* ball){
    ball->prev_x = ball->x;
    ball->prev_y = ball->y;
    ball->x += ball->vx * TIMESTEP;
    ball->y += ball->vy * TIMESTEP;
}

void ball_accelerate(Ball* ball, float ax, float ay){
    ball->vx += ax * TIMESTEP;
    ball->vy += ay * TIMESTEP;
}

int ball_check_edges(Ball* ball){
    if (ball->x <= (ball->r-REF_W/2)){
        ball->vx *= -FRICTION;
        ball->x = ball->r-REF_W/2+NUDGE*TIMESTEP;
    }
    if (ball->x >= (REF_W/2-ball->r)){
        ball->vx *= -FRICTION;
        ball->x = REF_W/2-ball->r-NUDGE*TIMESTEP;
    }
    if (ball->y <= (ball->r+REF_U)){
        ball->vy *= -FRICTION;
        ball->y = ball->r+REF_U+NUDGE*TIMESTEP;
        if (ball->x <= 0){
            return -1;
        }
        else{
            return 1;
        }
    }
    if (ball->y >= (REF_H-ball->r)){
        ball->vy *= -FRICTION;
        ball->y = REF_H-ball->r-NUDGE*TIMESTEP;
    }
    // fence:
    if ((ball->x <= (REF_WALL_WIDTH/2+ball->r)) && (ball->prev_x > (REF_WALL_WIDTH/2+ball->r)) && (ball->y <= REF_WALL_HEIGHT)){
        ball->vx *= -FRICTION;
        ball->x = REF_WALL_WIDTH/2+ball->r+NUDGE*TIMESTEP;
    }
    if ((ball->x >= (-REF_WALL_WIDTH/2-ball->r)) && (ball->prev_x < (-REF_WALL_WIDTH/2-ball->r)) && (ball->y <= REF_WALL_HEIGHT)){
        ball->vx *= -FRICTION;
        ball->x = -REF_WALL_WIDTH/2-ball->r-NUDGE*TIMESTEP;
    }
    return 0;
}

float ball_get_dist_squared(Ball* ball, SphericalObject* p){
    float dx = ball->x - p->x;
    float dy = ball->y - p->y;
    return dx*dx + dy*dy;
}

bool ball_is_colliding(Ball* ball, SphericalObject* p){
    float r = ball->r+p->r;
    return r*r > ball_get_dist_squared(ball, p);
}

void ball_bounce(Ball* ball, SphericalObject* p){
    float dx = ball->x - p->x;
    float dy = ball->y - p->y;
    float dist = sqrt(dx*dx + dy*dy);
    dx /= dist; // normalize. unit vector pointing from ball to p.
    dy /= dist;
    float nx = dx; // reuse calculation
    float ny = dy;

    dx *= NUDGE; // separate overlapping objects
    dy *= NUDGE;
    while(ball_is_colliding(ball, p)){
        ball->x += dx;
        ball->y += dy;
    }
    float ux = ball->vx - p->vx; // relative velocity of ball in relation to p
    float uy = ball->vy - p->vy;
    float un = ux*nx + uy*ny;
    float unx = nx*(un*2.); // added factor of 2 for conservation of momentum (elastic collision)
    float uny = ny*(un*2.); // added factor of 2 for conservation of momentum (elastic collision)
    ux -= unx;
    uy -= uny;
    ball->vx = ux + p->vx;
    ball->vy = uy + p->vy;
}

void ball_limit_speed(Ball* ball, float minSpeed, float maxSpeed){
    float mag2 = ball->vx*ball->vx+ball->vy*ball->vy;
    if (mag2 > (maxSpeed*maxSpeed)){
        float mag = sqrt(mag2);
        ball->vx /= mag;
        ball->vy /= mag;
        ball->vx *= maxSpeed;
        ball->vy *= maxSpeed;
    }
}

// Relative State
typedef struct {
    //agent
    float x;
    float y;
    float vx;
    float vy;
    //ball
    float bx;
    float by;
    float bvx;
    float bvy;
    //opponent
    float ox;
    float oy;
    float ovx;
    float ovy;
} RelativeState;

// WALL
typedef struct {
    float x;
    float y;
    float w;
    float h;
    Color c;
} Wall;

void wall_display(Wall* wall){
    Rectangle rec = {to_x_pixel(wall->x - wall->w/2), to_y_pixel(wall->y + wall->h/2),
         to_p(wall->w), to_p(wall->h)};
    DrawRectangleRec(rec, wall->c);
}

// AGENT
typedef struct {
    float x;
    float y;
    float r;
    float vx;
    float vy;
    int dir; // -1 means left, 1 means right player for symmetry
    Color c;
    float desired_vx;
    float desired_vy;
    float* observations;
    RelativeState *state;
    int lives;
} Agent;

void agent_display(Agent *agent, float bx, float by) {
    // fprintf(stderr, "agent_display: x=%f, y=%f, r=%f, lives=%d, dir=%d\n", agent->x, agent->y, agent->r, agent->lives, agent->dir);
    float x = agent->x;
    float y = agent->y;
    float r = agent->r;

    // Draw the agent's body as a half circle
    // Raylib: DrawCircleSector(center, radius, startAngle, endAngle, segments, color)
    DrawCircleSector(
        (Vector2){to_x_pixel(x), to_y_pixel(y)},
        to_p(r),
        180, 360,
        32, // segments
        agent->c
    );

    float angle = agent->dir == -1 ? PI * 60.0f / 180.0f : PI * 120.0f / 180.0f;

    // track ball with eyes
    float c = cosf(angle);
    float s = sinf(angle);
    float eye_base_x = x + 0.6f * r * c;
    float eye_base_y = y + 0.6f * r * s;
    float ballX = bx - eye_base_x;
    float ballY = by - eye_base_y;

    // If agent is sad (no lives), look down and away
    if (agent->lives == 0) {
        ballX = -agent->dir;
        ballY = -3;
    }

    float dist = sqrtf(ballX * ballX + ballY * ballY);
    float eyeX = 0, eyeY = 0;
    if (dist > 0) {
        eyeX = ballX / dist;
        eyeY = ballY / dist;
    }

    // Draw white of the eye
    DrawCircle(
        to_x_pixel(eye_base_x),
        to_y_pixel(eye_base_y),
        to_p(r) * 0.3f,
        WHITE
    );

    // Draw pupil
    DrawCircle(
        to_x_pixel(eye_base_x + eyeX * 0.15f * r),
        to_y_pixel(eye_base_y + eyeY * 0.15f * r),
        to_p(r) * 0.1f,
        BLACK
    );

    // Draw coins (lives) left
    for (int i = 1; i < agent->lives; i++) {
        DrawCircle(
            to_x_pixel(agent->dir * (REF_W / 2 + 0.5f - i * 2.0f)),
            WINDOW_HEIGHT - to_y_pixel(1.5f),
            to_p(0.5f),
            COIN_COLOR
        );
    }
}

void agent_set_action(Agent* agent, float* action){
    bool forward = false;
    bool backward = false;
    bool jump = false;
    if (action[0] > 0){
        forward = true;
    }
    if (action[1] > 0){
        backward = true;
    }
    if (action[2] > 0){
        jump = true;
    }
    agent->desired_vx = 0;
    agent->desired_vy = 0;
    if (forward && !backward){
        agent->desired_vx = -PLAYER_SPEED_X;
    }
    if (backward && !forward){
        agent->desired_vx = PLAYER_SPEED_X;
    }
    if (jump){
        agent->desired_vy = PLAYER_SPEED_Y;
    }
}

void agent_move(Agent* agent){
    agent->x += agent->vx * TIMESTEP;
    agent->y += agent->vy * TIMESTEP;
}

void agent_update(Agent* agent){
    agent->vy += GRAVITY * TIMESTEP;
    if (agent->y <= REF_U + NUDGE*TIMESTEP){ // if grounded
        agent->vy = agent->desired_vy;
    }
    agent->vx = agent->desired_vx*agent->dir;
    agent_move(agent);
    if (agent->y <= REF_U){
        agent->y = REF_U;
        agent->vy = 0;
    }
    // stay in their own half:
    if (agent->x*agent->dir <= (REF_WALL_WIDTH/2+agent->r)){
        agent->vx = 0;
        agent->x = agent->dir*(REF_WALL_WIDTH/2+agent->r);
    }
    if (agent->x*agent->dir >= (REF_W/2-agent->r)){
        agent->vx = 0;
        agent->x = agent->dir*(REF_W/2-agent->r);
    }
}

void agent_update_state(Agent* agent, Ball* ball, Agent* opponent){
    int obs_idx = 0;
    float* observations = agent->observations;
    
    // self
    observations[obs_idx++] = agent->x*agent->dir / 10.0f;
    observations[obs_idx++] = agent->y / 10.0f;  
    observations[obs_idx++] = agent->vx*agent->dir / 10.0f;
    observations[obs_idx++] = agent->vy / 10.0f;
    // ball
    observations[obs_idx++] = ball->x*agent->dir / 10.0f;
    observations[obs_idx++] = ball->y / 10.0f;
    observations[obs_idx++] = ball->vx*agent->dir / 10.0f;
    observations[obs_idx++] = ball->vy / 10.0f;
    // opponent
    observations[obs_idx++] = opponent->x*(-agent->dir) / 10.0f; // negate direction for opponent
    observations[obs_idx++] = opponent->y / 10.0f;
    observations[obs_idx++] = opponent->vx*(-agent->dir) / 10.0f ; // negate direction for opponent
    observations[obs_idx++] = opponent->vy / 10.0f;
}

// ENV

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Agent* agents;
    Wall* ground;
    Wall* fence;
    Ball* fence_stub;
    Ball* ball;
    int delay_frames; // frames to wait before starting
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    float* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int num_agents; // Number of agents being trained. Either 1 or 2. If 1, the first agent is trained and the second is a bot.
    float* bot_observations; // Optional, for bot control
    float* bot_actions; // Optional, for bot control
    int tick;
    Texture2D puffers;
} SlimeVolley;


/* Recommended to have an init function of some kind if you allocate 
* extra memory. This should be freed by c_close. Don't forget to call
* this in binding.c!
*/
void init(SlimeVolley* env) {
    env->ground = malloc(sizeof(Wall));
    *env->ground = (Wall){ .x = 0, .y = REF_U / 2.0, .w = REF_W, .h = REF_U, .c = GROUND_COLOR };
    env->fence = malloc(sizeof(Wall));
    *env->fence = (Wall){ .x = 0, .y = ( REF_U + REF_WALL_HEIGHT)/2.0, .w = REF_WALL_WIDTH, .h = REF_WALL_HEIGHT - 1.5, .c = FENCE_COLOR };
    env->fence_stub = malloc(sizeof(Ball));
    *env->fence_stub = (Ball){ .x = 0, .y = REF_WALL_HEIGHT, .vx=0, .vy=0, .r=REF_WALL_WIDTH/2.0, .c=FENCE_COLOR};
    env->agents = calloc(2, sizeof(Agent));
    env->ball = malloc(sizeof(Ball));
    if (env->num_agents == 1) {
        env->bot_observations = calloc(12, sizeof(float));
        env->bot_actions = calloc(3, sizeof(float));
    }
}

// Required function
void c_reset(SlimeVolley* env) {
    env->tick = 0;
    env->delay_frames = INIT_DELAY_FRAMES;
    float ball_vx = 40.0f*randf() - 20.0f;
    float ball_vy = 15.0f*randf() + 10.0f;
    *env->ball = (Ball){
        .x = 0,
        .y = REF_W/4,
        .vx = ball_vx,
        .vy = ball_vy,
        .r = 0.5,
        .c = BALL_COLOR
    };
    for (int i=0; i < 2; i++) {
        float* observations;
        if (i == 0) {
            observations = env->observations;
        }
        else {
            if (env->num_agents == 1) {
                observations = env->bot_observations;
            }
            else {
                observations = &env->observations[12]; // second agent in two-agent mode
            }
        }
        env->agents[i] = (Agent){
            .dir = i == 0 ? -1 : 1,
            .x = i == 0 ? -REF_W/4 : REF_W/4,
            .y = REF_U,
            .c = i == 0 ? PUFF_RED : PUFF_CYAN,
            .r = 1.5,
            .lives = MAXLIVES,
            .observations = observations
        };
    }
    agent_update_state(&env->agents[0], env->ball, &env->agents[1]);
    agent_update_state(&env->agents[1], env->ball, &env->agents[0]);
}

float clip(float val, float min, float max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    }
    return val;
}

void new_match(SlimeVolley* env) {
    float ball_vx = 40.0f*randf() - 20.0f;
    float ball_vy = 15.0f*randf() + 10.0f;
    *env->ball = (Ball){
        .x = 0,
        .y = REF_W/4,
        .vx = ball_vx,
        .vy = ball_vy,
        .r = 0.5,
        .c = BALL_COLOR
    };
    env->delay_frames = INIT_DELAY_FRAMES;
}

void abranti_simple_bot(float* obs, float* action) {
    // the bot policy. just 7 params but hard to beat.
    float x_agent = obs[0];
    float x_ball = obs[4];
    float vx_ball = obs[6];
    float backward = (-23.757145f * x_agent + 23.206863f * x_ball + 0.7943352f * vx_ball) + 1.4617119f;
    float forward = -64.6463748f * backward + 22.4668393f;
    action[0] = forward;
    action[1] = backward;
    action[2] = 1.0f; // always jump
}

// Required function
void c_step(SlimeVolley* env) {
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    if (env->num_agents == 2){
        env->rewards[1] = 0;
        env->terminals[1] = 0;
    }
    
    Agent* left = &env->agents[0];
    Agent* right = &env->agents[1];
    Ball* ball = env->ball;

    env->tick++;
    agent_set_action(left, &env->actions[0]);
    if (env->num_agents == 1){
        abranti_simple_bot(right->observations, env->bot_actions);
        agent_set_action(right, env->bot_actions);
    }
    else {
        agent_set_action(right, &env->actions[3]);
    }

    // Update
    agent_update(left);
    agent_update(right);

    if (env->delay_frames == 0) {
        ball_accelerate(ball, 0, GRAVITY);
        ball_limit_speed(ball, 0, MAX_BALL_SPEED);
        ball_move(ball);
    }
    else {
        env->delay_frames--;
    }

    if (ball_is_colliding(ball, (SphericalObject*)left)){
        ball_bounce(ball, (SphericalObject*)left);
    }
    if (ball_is_colliding(ball, (SphericalObject*)right)){
        ball_bounce(ball, (SphericalObject*)right);
    }
    if (ball_is_colliding(ball, (SphericalObject*)env->fence_stub)){
        ball_bounce(ball, (SphericalObject*)env->fence_stub);
    }

    int right_reward = -ball_check_edges(ball);

    if (right_reward != 0){
        new_match(env);
        if (right_reward == -1){
            right->lives--;
            env->rewards[0] = 1.0f;
            if (env->num_agents == 2){
                env->rewards[1] = -1.0f;
            }
        }
        else{
            left->lives--;
            env->rewards[0] = -1.0f;
            if (env->num_agents == 2){
                env->rewards[1] = 1.0f;
            }
        }
    }
    agent_update_state(left, ball, right);
    agent_update_state(right, ball, left);

    if (env->tick > MAX_TICKS || left->lives <= 0 || right->lives <= 0){
        env->terminals[0] = 1;
        if (env->num_agents == 2){
            env->terminals[1] = 1;
        }
        env->log.perf = (left->lives - right->lives + 5.0f)  / 10.0f; // normalize to 0-1
        env->log.score = (float)(left->lives - right->lives);
        env->log.episode_return = (5.0f - right->lives);
        env->log.episode_length = (float)env->tick;
        env->log.n++;
        c_reset(env);
    }
    
}

// Required function. Should handle creating the client on first call
void c_render(SlimeVolley* env) {
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "PufferLib SlimeVolley");
        SetTargetFPS(50); // From original
        env->puffers = LoadTexture("resources/shared/puffers.png");
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    wall_display(env->ground);
    wall_display(env->fence);

    // Fence
    Ball* stub = env->fence_stub;
    DrawCircleV(
        (Vector2){to_x_pixel(stub->x), to_y_pixel(stub->y)},
        to_p(stub->r), stub->c
    );

    Ball* puff = env->ball;
    DrawTexturePro(
        env->puffers,
        (Rectangle){
            0,
            (puff->vx > 0 ? 576 : 608),
            32, 32,
        },
        (Rectangle){
            to_x_pixel(puff->x) - 16,
            to_y_pixel(puff->y) - 16,
            32,
           32 
        },
        (Vector2){0, 0},
        0,
        WHITE
    );

    for (int i=0; i<2; i++) {
        agent_display(&env->agents[i], env->ball->x, env->ball->y);
    }

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(SlimeVolley* env) {
    free(env->agents);
    free(env->ground);
    free(env->fence);
    free(env->fence_stub);
    free(env->ball);
    if (env->num_agents == 1) {
        free(env->bot_observations);
        free(env->bot_actions);
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
}
