#include "flappy_puffer.h"
#ifndef HEADLESS
#include "raylib.h"
#include "raymath.h"
#endif

#include <assert.h>

#if defined(PLATFORM_DESKTOP)
    #define GLSL_VERSION 330
#else
    #define GLSL_VERSION 100
#endif

#define COL_SKY_TOP       CLITERAL(Color){ 72, 160, 255, 255}
#define COL_SKY_BOT       CLITERAL(Color){185, 225, 255, 255}
#define COL_CLOUD         CLITERAL(Color){255, 255, 255, 210}
#define COL_CLOUD_DARK    CLITERAL(Color){230, 240, 255, 210}
#define COL_GROUND        CLITERAL(Color){ 97,  62,  28, 255}
#define COL_GRASS         CLITERAL(Color){ 72, 175,  40, 255}
#define COL_GRASS2        CLITERAL(Color){ 55, 145,  30, 255}
#define COL_PIPE          CLITERAL(Color){ 42, 200,  58, 255}
#define COL_PIPE_LIGHT    CLITERAL(Color){ 90, 230,  90, 255}
#define COL_PIPE_DARK     CLITERAL(Color){ 25, 135,  35, 255}
#define COL_PIPE_CAP      CLITERAL(Color){ 34, 170,  48, 255}
#define COL_PIPE_CAP_LT   CLITERAL(Color){ 75, 210,  75, 255}
#define COL_SCORE_GOLD    CLITERAL(Color){255, 215,   0, 255}
#define COL_SCORE_SHADOW  CLITERAL(Color){ 80,  40,   0, 180}
#define COL_DEAD_OVERLAY  CLITERAL(Color){  0,   0,   0, 160}
#define COL_HUD_BG        CLITERAL(Color){  0,   0,   0, 120}
#define COL_WHITE         WHITE
#define COL_BLACK         BLACK

#define FP_FPS          60
#define NUM_CLOUDS       6
#define CLOUD_SPEED      0.45f   /* pixels per frame  */

typedef struct Cloud Cloud;
struct Cloud {
    float x, y;
    float w, h;
    float speed;
};

struct FPClient {
    Texture2D puffer_tex;
    bool      has_puffer;
    Font      font;
    bool      has_font;
    Cloud     clouds[NUM_CLOUDS];
    float     ground_offset;
    int       frame;
};

static inline float randf01(unsigned int* rng) {
    return (float)rand_r(rng) / (float)RAND_MAX;
}

static void reset_pipe(FPPipe* p, float x, unsigned int* rng) {
    p->x = x;
    float margin = (float)FP_PIPE_GAP * 0.5f + 30.0f;
    float min_cy = margin;
    float max_cy = (float)FP_PLAYABLE_H - margin;
    p->gap_cy = min_cy + randf01(rng) * (max_cy - min_cy);
    p->passed = false;
}

static void compute_obs(FlappyPuffer* env) {
    float* obs = env->observations;   

    obs[0] = env->bird_y / (float)FP_PLAYABLE_H;
    obs[1] = env->bird_vel / 20.0f;   

    FPPipe* p1 = NULL, *p2 = NULL;
    for (int i = 0; i < FP_MAX_PIPES; i++) {
        float right = env->pipes[i].x + FP_PIPE_W;
        if (right > FP_BIRD_X - FP_BIRD_R) {
            if (!p1 || env->pipes[i].x < p1->x) { p2 = p1; p1 = &env->pipes[i]; }
            else if (!p2 || env->pipes[i].x < p2->x) { p2 = &env->pipes[i]; }
        }
    }
 
    if (p1) {
        obs[2] = (p1->x - FP_BIRD_X) / (float)FP_SCREEN_W;
        obs[3] =  p1->gap_cy / (float)FP_PLAYABLE_H;
        obs[4] = (p1->gap_cy - FP_PIPE_GAP * 0.5f) / (float)FP_PLAYABLE_H;
        obs[5] = (p1->gap_cy + FP_PIPE_GAP * 0.5f) / (float)FP_PLAYABLE_H;
    } else {
        obs[2] = 1.0f; obs[3] = 0.5f; obs[4] = 0.35f; obs[5] = 0.65f;
    }
    if (p2) {
        obs[6] = (p2->x - FP_BIRD_X) / (float)FP_SCREEN_W;
        obs[7] =  p2->gap_cy / (float)FP_PLAYABLE_H;
    } else {
        obs[6] = 2.0f; obs[7] = 0.5f;
    }
}

void allocate_flappy(FlappyPuffer* env) {
    assert(env->num_agents == 1);
    env->observations = calloc(env->num_agents * FP_OBS_SIZE, sizeof(float));
    env->rewards      = calloc(env->num_agents,               sizeof(float));
    env->terminals    = calloc(env->num_agents,               sizeof(float));
    env->actions      = calloc(env->num_agents,               sizeof(float));
}

void free_flappy(FlappyPuffer* env) {
    free(env->observations);
    free(env->rewards);
    free(env->terminals);
    free(env->actions);
    if (env->client) {
        if (env->client->has_puffer) UnloadTexture(env->client->puffer_tex);
        if (env->client->has_font)   UnloadFont(env->client->font);
        if (IsWindowReady()) CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

void c_reset_fp(FlappyPuffer* env) {
    env->bird_y   = FP_PLAYABLE_H * 0.45f;
    env->bird_vel = 0.0f;
    env->alive    = true;
    env->score    = 0;
    env->tick     = 0;

    for (int i = 0; i < FP_MAX_PIPES; i++) {
        reset_pipe(&env->pipes[i],
                   (float)FP_SCREEN_W + 120.0f + i * FP_PIPE_SPACING,
                   &env->rng);
    }

    for (int a = 0; a < env->num_agents; a++) {
        env->rewards[a]   = 0.0f;
        env->terminals[a] = 0.0f;
    }

    compute_obs(env);
}

void c_step_fp(FlappyPuffer* env) {
    if (!env->alive) {
        c_reset_fp(env);
    }

    env->tick++;

    if ((int)env->actions[0] == ACT_FLAP) {
        env->bird_vel = FP_FLAP_VEL;
    }

    env->bird_vel += FP_GRAVITY;
    if (env->bird_vel >  FP_MAX_VEL) env->bird_vel =  FP_MAX_VEL;
    if (env->bird_vel < -FP_MAX_VEL) env->bird_vel = -FP_MAX_VEL;
    env->bird_y += env->bird_vel;

    float reward = FP_REWARD_ALIVE;

    if (env->bird_y + FP_BIRD_R >= (float)FP_PLAYABLE_H ||
        env->bird_y - FP_BIRD_R <= 0.0f) {
        env->alive = false;
    }

    for (int i = 0; i < FP_MAX_PIPES; i++) {
        env->pipes[i].x -= FP_PIPE_SPEED;

        if (env->pipes[i].x + FP_PIPE_W < -20.0f) {
            float max_x = 0.0f;
            for (int j = 0; j < FP_MAX_PIPES; j++) {
                if (env->pipes[j].x > max_x) max_x = env->pipes[j].x;
            }
            reset_pipe(&env->pipes[i], max_x + FP_PIPE_SPACING, &env->rng);
        }

        if (!env->pipes[i].passed &&
            env->pipes[i].x + FP_PIPE_W * 0.5f < FP_BIRD_X) {
            env->pipes[i].passed = true;
            env->score++;
            reward += FP_REWARD_PIPE;
        }

        if (env->alive) {
            float gap_top = env->pipes[i].gap_cy - FP_PIPE_GAP * 0.5f;
            float gap_bot = env->pipes[i].gap_cy + FP_PIPE_GAP * 0.5f;
            bool  x_hit   = (FP_BIRD_X + FP_BIRD_R > env->pipes[i].x) &&
                            (FP_BIRD_X - FP_BIRD_R < env->pipes[i].x + FP_PIPE_W);
            bool  y_hit   = (env->bird_y - FP_BIRD_R < gap_top) ||
                            (env->bird_y + FP_BIRD_R > gap_bot);
            if (x_hit && y_hit) env->alive = false;
        }
    }


    if (!env->alive) {
        reward = FP_REWARD_DEATH;
        env->terminals[0] = 1.0f;
        env->log.score          += (float)env->score;
        env->log.episode_length += (float)env->tick;
        env->log.n++;
        env->log.perf = env->log.score / env->log.n;
    } else {
        env->terminals[0] = 0.0f;
    }

    env->rewards[0] = reward;
    compute_obs(env);
}


static void draw_background(FPClient* cl) {
    DrawRectangleGradientV(0, 0,
        FP_SCREEN_W, FP_PLAYABLE_H,
        COL_SKY_TOP, COL_SKY_BOT);

    for (int i = 0; i < NUM_CLOUDS; i++) {
        Cloud* c = &cl->clouds[i];
        DrawEllipse((int)c->x,           (int)c->y,           (int)(c->w*0.55f), (int)(c->h*0.65f), COL_CLOUD_DARK);
        DrawEllipse((int)(c->x + c->w*0.30f), (int)(c->y - c->h*0.18f), (int)(c->w*0.50f), (int)(c->h*0.75f), COL_CLOUD);
        DrawEllipse((int)(c->x + c->w*0.65f), (int)(c->y + c->h*0.05f), (int)(c->w*0.42f), (int)(c->h*0.58f), COL_CLOUD_DARK);
        DrawEllipse((int)(c->x + c->w*0.25f), (int)c->y,                 (int)(c->w*0.80f), (int)(c->h*0.50f), COL_CLOUD);
    }
}

static void draw_ground(FPClient* cl) {
    int gy = FP_PLAYABLE_H;

    DrawRectangle(0, gy, FP_SCREEN_W, 16, COL_GRASS);
    for (int x = ((int)cl->ground_offset % 32); x < FP_SCREEN_W; x += 32) {
        DrawRectangle(x, gy, 16, 4, COL_GRASS2);
    }

    DrawRectangle(0, gy + 16, FP_SCREEN_W, FP_GROUND_H - 16, COL_GROUND);

    for (int x = ((int)(cl->ground_offset * 0.5f) % 48); x < FP_SCREEN_W; x += 48) {
        DrawRectangle(x, gy + 20, 20, FP_GROUND_H - 24,
            CLITERAL(Color){ 115, 75, 35, 80});
    }
}

static void draw_pipe(FPPipe* p) {
    int px   = (int)p->x;
    int pw   = FP_PIPE_W;
    int capw = FP_PIPE_CAP_W;
    int caph = FP_PIPE_CAP_H;
    int capx = px - (capw - pw) / 2;

    float gap_top = p->gap_cy - FP_PIPE_GAP * 0.5f;
    float gap_bot = p->gap_cy + FP_PIPE_GAP * 0.5f;

    int top_h = (int)gap_top;
    if (top_h > 0) {
        DrawRectangle(px,     0,    pw,   top_h, COL_PIPE);
        DrawRectangle(px + 4, 0,    10,   top_h, COL_PIPE_LIGHT);
        DrawRectangle(px + pw - 5, 0, 5, top_h, COL_PIPE_DARK);
    }
    DrawRectangle(capx,     top_h - caph, capw, caph, COL_PIPE_CAP);
    DrawRectangle(capx + 5, top_h - caph, 10,   caph, COL_PIPE_CAP_LT);
    DrawRectangleLines(capx, top_h - caph, capw, caph, COL_PIPE_DARK);

    int bot_y = (int)gap_bot;
    int bot_h = FP_PLAYABLE_H - bot_y;
    if (bot_h > 0) {
        DrawRectangle(px,     bot_y, pw,   bot_h, COL_PIPE);
        DrawRectangle(px + 4, bot_y, 10,   bot_h, COL_PIPE_LIGHT);
        DrawRectangle(px + pw - 5, bot_y, 5, bot_h, COL_PIPE_DARK);
    }
    DrawRectangle(capx,     bot_y,        capw, caph, COL_PIPE_CAP);
    DrawRectangle(capx + 5, bot_y,        10,   caph, COL_PIPE_CAP_LT);
    DrawRectangleLines(capx, bot_y, capw, caph, COL_PIPE_DARK);

    if (top_h > 0) DrawRectangleLines(px, 0, pw, top_h, COL_PIPE_DARK);
    if (bot_h > 0) DrawRectangleLines(px, bot_y, pw, bot_h, COL_PIPE_DARK);
}

static void draw_puffer(FPClient* cl, float y, float vel) {
    float cx = FP_BIRD_X;
    float cy = y;
    float sz = FP_BIRD_DRAW_SZ;

    float angle = vel * 4.0f;
    if (angle >  90.0f) angle =  90.0f;
    if (angle < -35.0f) angle = -35.0f;

    if (cl->has_puffer) {
        Rectangle src = {0, 0,
            (float)cl->puffer_tex.width,
            (float)cl->puffer_tex.height};
        Rectangle dst = {cx, cy, sz, sz};
        Vector2 origin = {sz * 0.5f, sz * 0.5f};
        DrawTexturePro(cl->puffer_tex, src, dst, origin, angle, WHITE);
    } else {
        float rad   = sz * 0.45f;
        float cxf   = cx;
        float cyf   = cy;

        DrawEllipse((int)(cxf + 2), (int)(cyf + 3),
            (int)(rad * 1.05f), (int)(rad * 0.92f),
            CLITERAL(Color){0, 0, 0, 60});

        DrawCircle((int)cxf, (int)cyf, (int)rad,
            CLITERAL(Color){255, 190, 40, 255});

        DrawEllipse((int)(cxf - 3), (int)(cyf - 2),
            (int)(rad * 0.55f), (int)(rad * 0.45f),
            CLITERAL(Color){255, 230, 130, 200});

        float ang_rad = angle * DEG2RAD;
        for (int s = 0; s < 8; s++) {
            float a   = s * (2.0f * PI / 8.0f) + ang_rad * 0.5f;
            float cos_a = cosf(a), sin_a = sinf(a);
            int x1 = (int)(cxf + rad * 0.72f * cos_a);
            int y1 = (int)(cyf + rad * 0.72f * sin_a);
            int x2 = (int)(cxf + (rad + 11.0f) * cos_a);
            int y2 = (int)(cyf + (rad + 11.0f) * sin_a);
            DrawLineEx((Vector2){(float)x1,(float)y1},
                       (Vector2){(float)x2,(float)y2},
                       3.0f,
                       CLITERAL(Color){200, 130, 20, 255});
            DrawCircle(x2, y2, 3,
                CLITERAL(Color){240, 160, 30, 255});
        }

        DrawCircle((int)(cxf + 9), (int)(cyf - 6), 7,
            CLITERAL(Color){255, 255, 255, 255});
        DrawCircle((int)(cxf + 11), (int)(cyf - 5), 4,
            CLITERAL(Color){15, 15, 50, 255});
        DrawCircle((int)(cxf + 12), (int)(cyf - 7), 2,
            CLITERAL(Color){255, 255, 255, 255});

        DrawLineEx(
            (Vector2){cxf + 12.0f, cyf + 4.0f},
            (Vector2){cxf + 16.0f, cyf + 6.0f},
            2.0f, CLITERAL(Color){180, 80, 20, 255});
    }
}

static void draw_score(FPClient* cl, int score) {
    const char* txt = TextFormat("%d", score);
    int   fs  = 52;
    int   tw  = MeasureText(txt, fs);
    int   sx  = FP_SCREEN_W / 2 - tw / 2;
    int   sy  = 32;

    DrawText(txt, sx + 3, sy + 3, fs, COL_SCORE_SHADOW);
    DrawText(txt, sx, sy, fs, COL_SCORE_GOLD);
}

/* ── game-over overlay ── */
static void draw_game_over(int score, float best) {
    DrawRectangle(0, 0, FP_SCREEN_W, FP_SCREEN_H, COL_DEAD_OVERLAY);

    const char* go  = "GAME OVER";
    int fs_go = 58;
    int tw_go = MeasureText(go, fs_go);
    DrawText(go, FP_SCREEN_W/2 - tw_go/2 + 3,
             FP_SCREEN_H/2 - 60 + 3, fs_go,
             CLITERAL(Color){120, 0, 0, 255});
    DrawText(go, FP_SCREEN_W/2 - tw_go/2,
             FP_SCREEN_H/2 - 60, fs_go,
             CLITERAL(Color){255, 60, 60, 255});

    const char* sc_txt = TextFormat("Score: %d", score);
    int tw_sc = MeasureText(sc_txt, 34);
    DrawText(sc_txt, FP_SCREEN_W/2 - tw_sc/2, FP_SCREEN_H/2 + 10, 34,
             COL_SCORE_GOLD);

    if (best > 0) {
        const char* best_txt = TextFormat("Best: %.0f", best);
        int tw_b = MeasureText(best_txt, 26);
        DrawText(best_txt, FP_SCREEN_W/2 - tw_b/2, FP_SCREEN_H/2 + 55, 26,
                 CLITERAL(Color){200, 200, 255, 255});
    }

    const char* hint = "SPACE / W / UP to flap";
    int tw_h = MeasureText(hint, 20);
    DrawText(hint, FP_SCREEN_W/2 - tw_h/2, FP_SCREEN_H/2 + 105, 20,
             CLITERAL(Color){200, 200, 200, 200});
}

static void draw_stats(FPLog* log) {
    if (log->n < 1.0f) return;
    const char* txt = TextFormat("Avg: %.1f  Eps: %.0f", log->perf, log->n);
    int tw = MeasureText(txt, 18);
    DrawRectangle(FP_SCREEN_W - tw - 16, 6, tw + 12, 26, COL_HUD_BG);
    DrawText(txt, FP_SCREEN_W - tw - 10, 10, 18,
             CLITERAL(Color){200, 240, 255, 220});
}

static FPClient* make_fp_client(void) {
#ifndef HEADLESS
    InitWindow(FP_SCREEN_W, FP_SCREEN_H, "Flappy Puffer");
    SetTargetFPS(FP_FPS);

    FPClient* cl = calloc(1, sizeof(FPClient));
    cl->frame = 0;
    cl->ground_offset = 0.0f;

    cl->puffer_tex = LoadTexture("../resources/flappy_puffer/inflated_puff.png");
    cl->has_puffer = true;
    

    unsigned int rng = (unsigned int)time(NULL);
    for (int i = 0; i < NUM_CLOUDS; i++) {
        cl->clouds[i].x     = (float)(rand_r(&rng) % FP_SCREEN_W);
        cl->clouds[i].y     = 40.0f + (float)(rand_r(&rng) % (FP_PLAYABLE_H / 2));
        cl->clouds[i].w     = 80.0f + (float)(rand_r(&rng) % 80);
        cl->clouds[i].h     = 28.0f + (float)(rand_r(&rng) % 20);
        cl->clouds[i].speed = 0.25f + (float)(rand_r(&rng) % 100) / 200.0f;
    }

    return cl;
#else
    return NULL;
#endif
}

static void update_client(FPClient* cl, float pipe_speed) {
    cl->frame++;
    cl->ground_offset += pipe_speed;

    for (int i = 0; i < NUM_CLOUDS; i++) {
        cl->clouds[i].x -= cl->clouds[i].speed;
        if (cl->clouds[i].x + cl->clouds[i].w < 0) {
            unsigned int rng = (unsigned int)(cl->frame * 7 + i * 31);
            cl->clouds[i].x  = (float)FP_SCREEN_W + 20.0f;
            cl->clouds[i].y  = 40.0f + (float)(rand_r(&rng) % (FP_PLAYABLE_H / 2));
            cl->clouds[i].w  = 80.0f + (float)(rand_r(&rng) % 80);
            cl->clouds[i].h  = 28.0f + (float)(rand_r(&rng) % 20);
        }
    }
}


int c_render_fp(FlappyPuffer* env) {
#ifndef HEADLESS
    if (!env->client) {
        env->client = make_fp_client();
    }
    FPClient* cl = env->client;

    update_client(cl, env->alive ? FP_PIPE_SPEED : 0.0f);

    BeginDrawing();
    ClearBackground(COL_SKY_TOP);

    draw_background(cl);

    for (int i = 0; i < FP_MAX_PIPES; i++) {
        draw_pipe(&env->pipes[i]);
    }

    draw_ground(cl);
    draw_puffer(cl, env->bird_y, env->bird_vel);


    draw_score(cl, env->score);
    draw_stats(&env->log);

    if (!env->alive) {
        draw_game_over(env->score, env->log.perf);
    }

    EndDrawing();

    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_W) || IsKeyPressed(KEY_UP)) {
        return ACT_FLAP;
    }
#endif
    return ACT_NOOP;
}

void c_close(FlappyPuffer* env) {
    free_flappy(env);
}

void c_reset(FlappyPuffer* env) {
    c_reset_fp(env);
}

void c_step(FlappyPuffer* env) {
    c_step_fp(env);
}

void c_render(FlappyPuffer* env) {
    c_render_fp(env);
}

void demo(void) {
    FlappyPuffer env;
    memset(&env, 0, sizeof(FlappyPuffer));
    env.num_agents = 1;
    env.rng = (unsigned int)time(NULL);

    allocate_flappy(&env);
    c_reset_fp(&env);

#ifndef HEADLESS
    // Initialise the window by calling render once
    c_render_fp(&env);

    while (!WindowShouldClose()) {
        int action = c_render_fp(&env);
        env.actions[0] = (float)action;
        c_step_fp(&env);
    }
#else
    printf("Demo built in HEADLESS mode. No visualization available.\n");
#endif

    free_flappy(&env);
}

int main() {
    printf("Starting flappy_puffer_demo...\n");
    demo();
    return 0;
}