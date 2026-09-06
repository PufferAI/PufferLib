#include "raylib.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
typedef float obs_t;
#define PUF_HAS_BOT_POLICY
#include "pufferenv.h"

// CONFIG
#define ACT_SIZES {2, 2, 2}
#define OBS_SIZE 12
#define NUM_ATNS 3

typedef Env SlimeVolley;

// Scripted opponents, weakest first. These are the [env] bot_policy ids and the
// rungs of [selfplay] eval_bots.
enum { BOT_ABRANTI = 0 };

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

const Color COIN_COLOR = {240, 210, 130, 255};
const Color FENCE_COLOR = {240, 210, 130, 255};
const Color GROUND_COLOR = {128, 227, 153, 255};
const Color PUFF_RED = {187, 0, 0, 255};
const Color PUFF_CYAN = {0, 187, 187, 255};
const Color PUFF_BACKGROUND = {6, 24, 24, 255};

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

// OBJECTS
typedef struct {
    float x;
    float y;
    float r;
    float vx;
    float vy;
    float prev_x;
} Ball;

void ball_move(Ball* ball){
    ball->prev_x = ball->x;
    ball->x += ball->vx * TIMESTEP;
    ball->y += ball->vy * TIMESTEP;
}

void ball_accelerate(Ball* ball, float ax, float ay){
    ball->vx += ax * TIMESTEP;
    ball->vy += ay * TIMESTEP;
}

// Returns 0 while the ball is live, else the side that conceded: -1 left, 1 right.
int ball_check_edges(Ball* ball){
    // Solid fence. The old test only fired on the tick the ball crossed a face
    // (x inside the band, prev_x outside), so a ball that got inside the band
    // any other way - a player bounce shoving it off the wall, a stub bounce
    // dropping it in - drifted out the far side and scored for free. Resolve
    // against the face the ball came from instead. That also catches a full
    // tunnel in one step, and runs after the player bounces so a hit into the
    // wall cannot push the ball through it.
    float fence = REF_WALL_WIDTH/2 + ball->r;
    if (ball->y <= REF_WALL_HEIGHT){
        float side = (ball->prev_x >= fence) - (ball->prev_x <= -fence);
        if (side == 0){
            side = (ball->x >= 0) ? 1.0f : -1.0f; // already inside: nearest face
        }
        if (side*ball->x < fence){
            ball->x = side*(fence + NUDGE*TIMESTEP);
            if (side*ball->vx < 0){
                ball->vx *= -FRICTION;
            }
        }
    }
    if (ball->x <= (ball->r-REF_W/2)){
        ball->vx *= -FRICTION;
        ball->x = ball->r-REF_W/2+NUDGE*TIMESTEP;
    }
    if (ball->x >= (REF_W/2-ball->r)){
        ball->vx *= -FRICTION;
        ball->x = REF_W/2-ball->r-NUDGE*TIMESTEP;
    }
    if (ball->y >= (REF_H-ball->r)){
        ball->vy *= -FRICTION;
        ball->y = REF_H-ball->r-NUDGE*TIMESTEP;
    }
    if (ball->y <= (ball->r+REF_U)){
        ball->vy *= -FRICTION;
        ball->y = ball->r+REF_U+NUDGE*TIMESTEP;
        return (ball->x <= 0) ? -1 : 1;
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

void ball_limit_speed(Ball* ball, float max_speed){
    float mag2 = ball->vx*ball->vx+ball->vy*ball->vy;
    if (mag2 > (max_speed*max_speed)){
        float mag = sqrt(mag2);
        ball->vx *= max_speed/mag;
        ball->vy *= max_speed/mag;
    }
}

// PLAYER (game entity; RL Agent is from pufferenv)
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
    int lives;
} Player;

void agent_display(Player *agent, float bx, float by) {
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

void agent_set_action(Player* agent, float* action){
    agent->desired_vx = 0;
    agent->desired_vy = 0;
    bool forward = action[0] > 0;
    bool backward = action[1] > 0;
    if (forward && !backward){
        agent->desired_vx = -PLAYER_SPEED_X;
    }
    if (backward && !forward){
        agent->desired_vx = PLAYER_SPEED_X;
    }
    if (action[2] > 0){
        agent->desired_vy = PLAYER_SPEED_Y;
    }
}

void agent_update(Player* agent){
    agent->vy += GRAVITY * TIMESTEP;
    if (agent->y <= REF_U + NUDGE*TIMESTEP){ // if grounded
        agent->vy = agent->desired_vy;
    }
    agent->vx = agent->desired_vx*agent->dir;
    agent->x += agent->vx * TIMESTEP;
    agent->y += agent->vy * TIMESTEP;
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

// Ego-centric and mirrored by dir, so both sides see the same game.
void agent_update_state(Player* agent, Ball* ball, Player* opponent){
    float* observations = agent->observations;
    observations[0] = agent->x*agent->dir / 10.0f;
    observations[1] = agent->y / 10.0f;
    observations[2] = agent->vx*agent->dir / 10.0f;
    observations[3] = agent->vy / 10.0f;
    observations[4] = ball->x*agent->dir / 10.0f;
    observations[5] = ball->y / 10.0f;
    observations[6] = ball->vx*agent->dir / 10.0f;
    observations[7] = ball->vy / 10.0f;
    observations[8] = opponent->x*(-agent->dir) / 10.0f;
    observations[9] = opponent->y / 10.0f;
    observations[10] = opponent->vx*(-agent->dir) / 10.0f;
    observations[11] = opponent->vy / 10.0f;
}

// ENV

// Required struct. Only use floats!
typedef struct Log Log;
struct Log {
    // Lives margin from each slot's own view, mapped to [0, 1]. 0.5 is a draw,
    // 1.0 a 5-0 sweep. Denser than win credit, so it is the sweep metric.
    float perf;
    float score;            // lives margin from each slot's own view
    float episode_return;
    float episode_length;
    // Per-slot win credit for match() scoring + a selfplay sanity check. In
    // selfplay both average to ~0.5; in match A=policy 0, B=policy 1, so
    // policy_0_score is A's win rate. Each game hands out 1.0 of credit total
    // (win=1.0, draw=0.5 each). Scaled by num_agents on accumulation so the
    // eval_log mean (sum / n, n incremented per slot per episode) is the rate.
    float policy_0_score;
    float policy_1_score;
    float draw_rate;
    float n;
};

struct Env {
    Log log;
    Agent agents[2];   // pufferenv RL agents (learning slots)
    Player players[2]; // game entities: 0 = left, 1 = right
    Ball ball;
    float episode_return[2];
    float bot_observations[OBS_SIZE]; // right side when it is a scripted bot
    float bot_actions[NUM_ATNS];
    int num_agents;    // 1 (right side is a bot) or 2 (selfplay)
    int num_bots;
    int bot_policy;    // BOT_* id the scripted side runs
    int max_ticks;     // episode timeout; configured per-run via [env].max_ticks
    int delay_frames;
    int tick;
    // Selfplay-pool tagging. tag = 0 means pure selfplay (both slots = primary
    // policy). tag > 0 means historical: slot 0 = primary, slot 1 = frozen
    // historical opponent. boundary_reached is set on game-end so the trainer
    // can swap frozen banks only between games.
    int tag;
    int boundary_reached;
    Texture2D puffers;
    unsigned int rng;
};

static inline void puf_set_bot_policy(Env* env, int bot_policy) {
    env->bot_policy = bot_policy;
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    env->num_bots = dict_get(kwargs, "num_bots");
    env->bot_policy = dict_get(kwargs, "bot_policy");
    env->max_ticks = dict_get(kwargs, "max_ticks");
    assert(env->num_agents + env->num_bots == 2
        && "slimevolley is 1v1: env.num_agents + env.num_bots must be 2");
    assert(env->bot_policy == BOT_ABRANTI
        && "slimevolley ships one bot: env.bot_policy must be 0");
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = i;
        env->agents[i].action_mask = NULL;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "policy_0_score", log->policy_0_score);
    dict_set(out, "policy_1_score", log->policy_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "n", log->n);
}

float randf(SlimeVolley* env) {
    return (float)rand_r(&env->rng) / (float)RAND_MAX;
}

void new_match(SlimeVolley* env) {
    env->ball = (Ball){
        .x = 0,
        .y = REF_W/4,
        .r = 0.5,
        .vx = 40.0f*randf(env) - 20.0f,
        .vy = 15.0f*randf(env) + 10.0f,
    };
    env->delay_frames = INIT_DELAY_FRAMES;
}

// Required function
void puf_reset(SlimeVolley* env) {
    env->tick = 0;
    for (int i = 0; i < 2; i++) {
        env->players[i] = (Player){
            .x = (i == 0) ? -REF_W/4 : REF_W/4,
            .y = REF_U,
            .r = 1.5,
            .dir = (i == 0) ? -1 : 1,
            .c = (i == 0) ? PUFF_RED : PUFF_CYAN,
            .lives = MAXLIVES,
        };
        env->players[i].observations = (i < env->num_agents)
            ? env->agents[i].observations : env->bot_observations;
        env->episode_return[i] = 0.0f;
    }
    new_match(env);
    agent_update_state(&env->players[0], &env->ball, &env->players[1]);
    agent_update_state(&env->players[1], &env->ball, &env->players[0]);
}

void add_reward(SlimeVolley* env, int slot, float reward) {
    env->agents[slot].rewards[0] += reward;
    env->episode_return[slot] += reward;
}

// Every episode-end path. outcome: 1 slot 0 won, -1 slot 0 lost, 0 draw.
void end_episode(SlimeVolley* env, int outcome) {
    float s0_score = (outcome > 0) ? 1.0f : (outcome < 0) ? 0.0f : 0.5f;
    env->log.policy_0_score += s0_score * env->num_agents;
    env->log.policy_1_score += (1.0f - s0_score) * env->num_agents;
    if (outcome == 0) {
        env->log.draw_rate += env->num_agents;
    }
    if (env->tag > 0) {
        env->boundary_reached = 1;
    }
    int margin = env->players[0].lives - env->players[1].lives;
    for (int i = 0; i < env->num_agents; i++) {
        int slot_margin = (i == 0) ? margin : -margin;
        env->log.perf += (slot_margin + MAXLIVES) / (2.0f*MAXLIVES);
        env->log.score += slot_margin;
        env->log.episode_return += env->episode_return[i];
        env->log.episode_length += env->tick;
        env->log.n += 1.0f;
        env->agents[i].terminals[0] = 1.0f;
    }
    puf_reset(env);
}

void puf_bot(SlimeVolley* env, int bot_idx) {
    // BOT_ABRANTI. just 7 params but hard to beat.
    float* obs = env->players[bot_idx].observations;
    float x_agent = obs[0];
    float x_ball = obs[4];
    float vx_ball = obs[6];
    float backward = -23.757145f*x_agent + 23.206863f*x_ball
        + 0.7943352f*vx_ball + 1.4617119f;
    env->bot_actions[0] = -64.6463748f * backward + 22.4668393f;
    env->bot_actions[1] = backward;
    env->bot_actions[2] = 1.0f; // always jump
}

// Hold Left Shift + A/D/W or arrows/space.
static void slimevolley_human_controls(SlimeVolley *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    env->agents[0].actions[0] = 0.0f;
    env->agents[0].actions[1] = 0.0f;
    env->agents[0].actions[2] = 0.0f;
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->agents[0].actions[0] = 1.0f;
    }
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->agents[0].actions[1] = 1.0f;
    }
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W) || IsKeyDown(KEY_SPACE)) {
        env->agents[0].actions[2] = 1.0f;
    }
}

// Required function
void puf_step(SlimeVolley* env) {
    env->tick++;
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].rewards[0] = 0;
        env->agents[i].terminals[0] = 0;
    }

    Player* left = &env->players[0];
    Player* right = &env->players[1];
    Ball* ball = &env->ball;

    agent_set_action(left, env->agents[0].actions);
    if (env->num_bots == 1){
        puf_bot(env, 1);
        agent_set_action(right, env->bot_actions);
    } else {
        agent_set_action(right, env->agents[1].actions);
    }
    agent_update(left);
    agent_update(right);

    if (env->delay_frames == 0) {
        ball_accelerate(ball, 0, GRAVITY);
        ball_limit_speed(ball, MAX_BALL_SPEED);
        ball_move(ball);
    } else {
        env->delay_frames--;
    }

    if (ball_is_colliding(ball, (SphericalObject*)left)){
        ball_bounce(ball, (SphericalObject*)left);
    }
    if (ball_is_colliding(ball, (SphericalObject*)right)){
        ball_bounce(ball, (SphericalObject*)right);
    }
    SphericalObject fence_stub = {.y = REF_WALL_HEIGHT, .r = REF_WALL_WIDTH/2};
    if (ball_is_colliding(ball, &fence_stub)){
        ball_bounce(ball, &fence_stub);
    }

    int conceded = ball_check_edges(ball);
    if (conceded != 0){
        env->players[(conceded < 0) ? 0 : 1].lives--;
        float reward = (conceded < 0) ? -1.0f : 1.0f; // slot 0's view
        add_reward(env, 0, reward);
        if (env->num_agents == 2){
            add_reward(env, 1, -reward);
        }
        new_match(env);
    }
    agent_update_state(left, ball, right);
    agent_update_state(right, ball, left);

    if (env->tick > env->max_ticks || left->lives <= 0 || right->lives <= 0){
        int outcome = (left->lives > right->lives) - (left->lives < right->lives);
        end_episode(env, outcome);
    }
}

// Required function. Should handle creating the client on first call
void puf_render(SlimeVolley* env) {
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
    slimevolley_human_controls(env);
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawRectangleRec((Rectangle){to_x_pixel(-REF_W/2.0f), to_y_pixel(REF_U),
        to_p(REF_W), to_p(REF_U)}, GROUND_COLOR);
    float fence_h = REF_WALL_HEIGHT - REF_U;
    DrawRectangleRec((Rectangle){to_x_pixel(-REF_WALL_WIDTH/2.0f),
        to_y_pixel(REF_U + fence_h), to_p(REF_WALL_WIDTH), to_p(fence_h)},
        FENCE_COLOR);
    DrawCircleV((Vector2){to_x_pixel(0), to_y_pixel(REF_WALL_HEIGHT)},
        to_p(REF_WALL_WIDTH/2.0f), FENCE_COLOR);

    Ball* puff = &env->ball;
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
        agent_display(&env->players[i], env->ball.x, env->ball.y);
    }

    EndDrawing();
    puf_web_vsync();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void puf_close(SlimeVolley* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
