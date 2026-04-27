#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#include "raylib.h"

#if defined(__APPLE__)
    #define GL_SILENCE_DEPRECATION
    #include <OpenGL/gl3.h>
    #include <OpenGL/gl3ext.h>
#else
    #include "glad.h"
#endif
#include "rlgl.h"
#include "raymath.h"

#define GLSL_VERSION 330

#define SCREEN_W 1920
#define SCREEN_H 1080

#define FONT_TITLE 72
#define FONT_LABEL 42

static const Color BG     = {4, 8, 20, 255};
static const Color C_WHITE = {220, 230, 255, 255};
static const Color C_CYAN  = {0, 187, 187, 255};

// ─── Vertex layout (x, y, size_scale,  r, g, b, a) ───────────────────────────
typedef struct {
    float x, y, size_scale;
    float r, g, b, a;
} StarVertex;

// ─── GL draw ──────────────────────────────────────────────────────────────────
static void draw_stars(StarVertex *verts, int n, Shader *sh) {
    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, n * sizeof(StarVertex), verts, GL_STREAM_DRAW);
        glVertexAttribPointer(sh->locs[SHADER_LOC_VERTEX_POSITION],
            3, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)0);
        glEnableVertexAttribArray(sh->locs[SHADER_LOC_VERTEX_POSITION]);
        glVertexAttribPointer(sh->locs[SHADER_LOC_VERTEX_COLOR],
            4, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(sh->locs[SHADER_LOC_VERTEX_COLOR]);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    rlDrawRenderBatchActive();
    rlSetBlendMode(RL_BLEND_ADDITIVE);
    int timeLoc = GetShaderLocation(*sh, "currentTime");
    glUseProgram(sh->id);
        glUniform1f(timeLoc, (float)GetTime());
        Matrix mvp = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection());
        glUniformMatrix4fv(sh->locs[SHADER_LOC_MATRIX_MVP], 1, GL_FALSE, MatrixToFloat(mvp));
        glBindVertexArray(vao);
            glDrawArrays(GL_POINTS, 0, n);
        glBindVertexArray(0);
    glUseProgram(0);
    rlSetBlendMode(RL_BLEND_ALPHA);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

// ─── Shared background star field ─────────────────────────────────────────────
#define NUM_BG_STARS   32768
#define NUM_VISIBLE_STARS 16384 // stars with normal distribution; rest start at center
#define BG_DENSE_R      1200.0f   // full density inside this radius
#define BG_FADE_R      2400.0f   // density lerps to BG_SPARSE_FRAC by here
#define BG_SPARSE_FRAC   0.02f   // density fraction beyond BG_FADE_R
#define BG_MAX_R       4000.0f

typedef struct { float r, angle, vr, brightness, size_scale; } BgStar;
static BgStar bg_stars[NUM_BG_STARS];
static StarVertex bg_verts[NUM_BG_STARS];

// Text layout anchors
#define HAIKU_Y          60.0f    // top of first haiku line
#define FOOTER_Y         860.0f   // top of "PufferLib 4.0" line

// Spiral/nova epicenter — midpoint between bottom of haiku line 3 and top of footer
// line3_bottom = HAIKU_Y + (FONT_TITLE+16)*2 + FONT_TITLE = 60+176+72 = 308
// footer_top   = FOOTER_Y = 860  →  mid = (308+860)/2 = 584
#define SPIRAL_CX        (SCREEN_W * 0.5f)
#define SPIRAL_CY        584.0f

#define SPIRAL_GM        16000000.0f   // gravitational constant — tune for feel
#define SPIRAL_OMEGA     0.1f         // base angular velocity (rad/s)
#define SPIRAL_OMEGA_R   1000.0f       // differential rotation scale (inner faster)

// Area of annulus [r0, r1] proportional to r1^2 - r0^2
// We split visible stars across three zones by desired density weight:
//   dense:  uniform area in [0, BG_DENSE_R]        weight = 1.0
//   fade:   uniform area in [BG_DENSE_R, BG_FADE_R] weight = avg ~0.5*(1+SPARSE_FRAC)
//   outer:  uniform area in [BG_FADE_R, BG_MAX_R]  weight = BG_SPARSE_FRAC
// Stars per zone proportional to area * weight
static void init_bg_stars(void) {
    float a_dense = BG_DENSE_R * BG_DENSE_R;
    float a_fade  = BG_FADE_R  * BG_FADE_R  - a_dense;
    float a_outer = BG_MAX_R   * BG_MAX_R   - BG_FADE_R * BG_FADE_R;
    float w_dense = 1.0f;
    float w_fade  = 0.5f * (1.0f + BG_SPARSE_FRAC);
    float w_outer = BG_SPARSE_FRAC;
    float total   = a_dense * w_dense + a_fade * w_fade + a_outer * w_outer;
    int n_dense   = (int)(NUM_VISIBLE_STARS * a_dense * w_dense / total);
    int n_fade    = (int)(NUM_VISIBLE_STARS * a_fade  * w_fade  / total);
    // remainder goes to outer

    for (int i = 0; i < NUM_BG_STARS; i++) {
        float r;
        if (i >= NUM_VISIBLE_STARS) {
            // Hidden nova stars
            r = 0.5f + ((float)rand() / (float)RAND_MAX) * 4.5f;
        } else if (i < n_dense) {
            float rnd = (float)rand() / (float)RAND_MAX;
            r = BG_DENSE_R * sqrtf(rnd);
        } else if (i < n_dense + n_fade) {
            float rnd = (float)rand() / (float)RAND_MAX;
            r = sqrtf(BG_DENSE_R*BG_DENSE_R + rnd * (BG_FADE_R*BG_FADE_R - BG_DENSE_R*BG_DENSE_R));
        } else {
            float rnd = (float)rand() / (float)RAND_MAX;
            r = sqrtf(BG_FADE_R*BG_FADE_R + rnd * (BG_MAX_R*BG_MAX_R - BG_FADE_R*BG_FADE_R));
        }
        bg_stars[i].r          = r;
        bg_stars[i].angle      = ((float)rand() / (float)RAND_MAX) * 2.0f * 3.14159265f;
        bg_stars[i].vr         = 0.0f;
        bg_stars[i].brightness = 0.2f + (float)(rand() % 80) / 100.0f;
        float s = (float)rand() / (float)RAND_MAX;
        bg_stars[i].size_scale = (s < 0.85f)
            ? (0.2f + s * 0.4f) * 0.7f
            : (0.8f + (s - 0.85f) * 6.0f) * 0.7f;
    }
}

#define NOVA_GM   5000000.0f  // outward repulsion strength for nova

// Integrate one timestep of gravitational attraction toward center
static void update_bg_spiral(float dt) {
    for (int i = 0; i < NUM_BG_STARS; i++) {
        float r = bg_stars[i].r;
        bg_stars[i].vr    -= SPIRAL_GM / (r * r) * dt;
        bg_stars[i].r     += bg_stars[i].vr * dt;
        bg_stars[i].angle += SPIRAL_OMEGA * SPIRAL_OMEGA_R / (r + SPIRAL_OMEGA_R) * dt;
        if (bg_stars[i].r < 1.0f) bg_stars[i].r = 1.0f;
    }
}

// Integrate one timestep of outward nova explosion (1/r^2 repulsion)
static void update_bg_nova(float dt) {
    for (int i = 0; i < NUM_BG_STARS; i++) {
        float r = fmaxf(bg_stars[i].r, 1.0f);
        bg_stars[i].vr += NOVA_GM / (r * r) * dt;
        bg_stars[i].r  += bg_stars[i].vr * dt;
        // No angular update — stars fly straight outward
    }
}

static int bg_draw_count;

static void build_bg_verts_spiral(float alpha, int nova) {
    float cx = SPIRAL_CX, cy = SPIRAL_CY;
    int n = nova ? NUM_BG_STARS : NUM_VISIBLE_STARS;
    for (int i = 0; i < n; i++) {
        float br = bg_stars[i].brightness * alpha;
        bg_verts[i] = (StarVertex){
            .x = cx + bg_stars[i].r * cosf(bg_stars[i].angle),
            .y = cy + bg_stars[i].r * sinf(bg_stars[i].angle),
            .size_scale = bg_stars[i].size_scale,
            .r = C_WHITE.r / 255.0f * br,
            .g = C_WHITE.g / 255.0f * br,
            .b = C_WHITE.b / 255.0f * br,
            .a = (float)i,
        };
    }
    bg_draw_count = n;
}

static void build_bg_verts(void) {
    bg_draw_count = NUM_VISIBLE_STARS;
    for (int i = 0; i < NUM_VISIBLE_STARS; i++) {
        float br = bg_stars[i].brightness;
        float cx = SPIRAL_CX, cy = SPIRAL_CY;
        bg_verts[i] = (StarVertex){
            .x = cx + bg_stars[i].r * cosf(bg_stars[i].angle),
            .y = cy + bg_stars[i].r * sinf(bg_stars[i].angle),
            .size_scale = bg_stars[i].size_scale,
            .r = C_WHITE.r / 255.0f * br,
            .g = C_WHITE.g / 255.0f * br,
            .b = C_WHITE.b / 255.0f * br,
            .a = (float)i,  // stable seed for twinkle
        };
    }
}

/* ============================================================
   ANIMATION 1: Speed Comparison (0–3s)
   Four rows, each a star bouncing at a rate proportional to
   the environment's simulation throughput (20M/4M/1M/30k sps).
   The 20M ball completes one full traverse in 1/1.333 s.
   ============================================================ */

#define NUM_BALLS    4
#define LANE_H       195         // 780px / 4 lanes
#define A1_MARGIN_L  280
#define A1_MARGIN_R  160
#define TRACK_W      (SCREEN_W - A1_MARGIN_L - A1_MARGIN_R)
#define TRACK_Y0     298         // 200 top + 780/8 = first lane center

#define SPEED_20M (2.0f * TRACK_W / 0.75f)
static const float SPEEDS[NUM_BALLS] = {
    SPEED_20M,
    SPEED_20M * (4.0f  / 20.0f),
    SPEED_20M * (1.0f  / 20.0f),
    SPEED_20M * (0.03f / 20.0f),
};
static const char *BALL_LABELS[NUM_BALLS]   = {"20M", "4M", "1M", "30k"};
static const char *BALL_VERSIONS[NUM_BALLS] = {"v4",  "v3", "v2", "v1"};

#define BALL_STAR_SCALE 5.28f

static float ball_x[NUM_BALLS];
static float ball_dir[NUM_BALLS];
static StarVertex ball_verts[NUM_BALLS];

static void init_anim1(void) {
    for (int i = 0; i < NUM_BALLS; i++) {
        ball_x[i]   = A1_MARGIN_L;
        ball_dir[i] = 1.0f;
    }
}

static void update_anim1(float dt, Font roboto, Shader *sh) {
    for (int i = 0; i < NUM_BALLS; i++) {
        ball_x[i] += SPEEDS[i] * ball_dir[i] * dt;
        if (ball_x[i] >= A1_MARGIN_L + TRACK_W) {
            ball_x[i] = A1_MARGIN_L + TRACK_W; ball_dir[i] = -1.0f;
        } else if (ball_x[i] <= A1_MARGIN_L) {
            ball_x[i] = A1_MARGIN_L; ball_dir[i] = 1.0f;
        }
        float lane_y = TRACK_Y0 + i * LANE_H;
        ball_verts[i] = (StarVertex){
            .x = ball_x[i], .y = lane_y,
            .size_scale = BALL_STAR_SCALE,
            .r = C_CYAN.r / 255.0f, .g = C_CYAN.g / 255.0f,
            .b = C_CYAN.b / 255.0f, .a = (float)(NUM_BG_STARS + i),  // stable seed
        };
    }

    build_bg_verts();
    draw_stars(bg_verts, bg_draw_count, sh);
    draw_stars(ball_verts, NUM_BALLS, sh);

    for (int i = 0; i < NUM_BALLS; i++) {
        int lane_y = TRACK_Y0 + i * LANE_H;
        DrawLine(A1_MARGIN_L, lane_y, A1_MARGIN_L + TRACK_W, lane_y,
            (Color){0, 187, 187, 20});
        DrawTextEx(roboto, BALL_VERSIONS[i], (Vector2){40,  lane_y - FONT_LABEL/2}, FONT_LABEL, 1, C_CYAN);
        DrawTextEx(roboto, BALL_LABELS[i],   (Vector2){140, lane_y - FONT_LABEL/2}, FONT_LABEL, 1, C_WHITE);
    }
    const char *title    = "Steps Per Second";
    const char *subtitle = "(1/1,000,000 scale)";
    Vector2 tsz = MeasureTextEx(roboto, title,    FONT_TITLE, 1);
    Vector2 ssz = MeasureTextEx(roboto, subtitle, FONT_LABEL, 1);
    DrawTextEx(roboto, title,
        (Vector2){SCREEN_W/2 - tsz.x/2, 100}, FONT_TITLE, 1, C_WHITE);
    DrawTextEx(roboto, subtitle,
        (Vector2){SCREEN_W/2 - ssz.x/2, 100 + FONT_TITLE + 4}, FONT_LABEL, 1, C_WHITE);
}

/* ============================================================
   ANIMATION 2: Horizontal Bar Chart – Solve Times (3–6s)
   Bars are starry regions. Each bar lerps from a large start
   width down to a smaller end width (showing speedup).
   Stars are pre-scattered across the start width; the right
   edge shrinks by culling stars beyond the current width.
   Phase:  0–1s static at start widths
           1–2s lerp start → end
           2–3s static at end widths
   ============================================================ */

#define NUM_ENVS      5
#define CHART_LEFT    440
#define CHART_MAX_W   (SCREEN_W/2 - CHART_LEFT - 40)
#define CHART_TOP     238        // 200 top + 156/2 - BAR_H/2; first bar center at 278
#define BAR_H         80
#define BAR_GAP       76        // 780px / 5 bars = 156 per slot; 156 - 80 = 76

#define BAR_STAR_SCALE  0.20f
#define STAR_DENSITY    0.075f   // stars per square pixel
#define MAX_BAR_STARS   30000   // upper bound for static allocation

typedef struct {
    const char *name;
    float start_val;
    float end_val;
    float start_px;   // set by init
    float end_ratio;  // end_val / start_val, set by init
    int   star_off;   // offset into flat arrays, set by init
    int   star_count; // set by init
} EnvBar;

static EnvBar envs[NUM_ENVS] = {
    {"Environment A", 27.0f,  3.0f,  0, 0, 0, 0},
    {"Environment B",  3.0f,  0.24f, 0, 0, 0, 0},
    {"Environment C",  1.0f,  0.14f, 0, 0, 0, 0},
    {"Environment D",  0.5f,  0.08f, 0, 0, 0, 0},
    {"Environment E",  0.2f,  0.04f, 0, 0, 0, 0},
};

static float bar_nx[MAX_BAR_STARS];
static float bar_sy[MAX_BAR_STARS];
static StarVertex bar_verts[MAX_BAR_STARS];

static void init_anim2(void) {
    float max_val = 0.0f;
    for (int e = 0; e < NUM_ENVS; e++)
        if (envs[e].start_val > max_val) max_val = envs[e].start_val;

    int offset = 0;
    for (int e = 0; e < NUM_ENVS; e++) {
        envs[e].start_px  = (envs[e].start_val / max_val) * CHART_MAX_W;
        envs[e].end_ratio = envs[e].end_val / envs[e].start_val;
        envs[e].star_count = (int)(STAR_DENSITY * envs[e].start_px * BAR_H);
        envs[e].star_off   = offset;
        float cy = CHART_TOP + e * (BAR_H + BAR_GAP) + BAR_H * 0.5f;
        for (int i = 0; i < envs[e].star_count; i++) {
            bar_nx[offset + i] = (float)rand() / (float)RAND_MAX;
            float fy = ((float)rand() / (float)RAND_MAX) * BAR_H - BAR_H * 0.5f;
            bar_sy[offset + i] = cy + fy;
        }
        offset += envs[e].star_count;
    }
}

static void draw_anim2(float anim_t, Font roboto, Shader *sh) {
    float lerp_t;
    if      (anim_t < 1.0f) lerp_t = 0.0f;
    else if (anim_t < 2.0f) lerp_t = anim_t - 1.0f;
    else                    lerp_t = 1.0f;

    int n = 0;
    for (int e = 0; e < NUM_ENVS; e++) {
        float cull = 1.0f - lerp_t * (1.0f - envs[e].end_ratio);
        int off = envs[e].star_off;
        for (int i = 0; i < envs[e].star_count; i++) {
            if (bar_nx[off + i] > cull) continue;
            bar_verts[n++] = (StarVertex){
                .x = CHART_LEFT + bar_nx[off + i] * envs[e].start_px,
                .y = bar_sy[off + i],
                .size_scale = BAR_STAR_SCALE,
                .r = C_CYAN.r / 255.0f, .g = C_CYAN.g / 255.0f,
                .b = C_CYAN.b / 255.0f, .a = (float)(off + i),  // stable seed
            };
        }
    }

    build_bg_verts();
    draw_stars(bg_verts, bg_draw_count, sh);
    draw_stars(bar_verts, n, sh);

    for (int e = 0; e < NUM_ENVS; e++) {
        float cy = CHART_TOP + e * (BAR_H + BAR_GAP) + BAR_H * 0.5f;
        DrawTextEx(roboto, envs[e].name, (Vector2){100, cy - FONT_LABEL/2}, FONT_LABEL, 1, C_WHITE);
    }
    // Anim2 title: centered in left half
    Vector2 t2sz = MeasureTextEx(roboto, "Solve Time", FONT_TITLE, 1);
    DrawTextEx(roboto, "Solve Time",
        (Vector2){SCREEN_W/4 - t2sz.x/2, 100}, FONT_TITLE, 1, C_WHITE);
}

/* ============================================================
   ANIMATION 3: PufferNet logo with pulsing glow (6–9s)
   Image is wide black outlines on transparent BG. A Sobel
   edge + blurred halo is driven by cos(time) for the pulse.
   ============================================================ */

static void draw_anim3(float anim_t, Shader *glow_sh, Texture2D tex, Shader *star_sh, Font roboto) {
    float glow = 0.65f + 0.15f * cosf(anim_t * 2.5f);  // 0..1 drives the oscillating portion

    float fade = fminf(anim_t / 0.5f, 1.0f);

    int glow_loc     = GetShaderLocation(*glow_sh, "glowStrength");
    int texel_loc    = GetShaderLocation(*glow_sh, "texelSize");
    int fade_loc     = GetShaderLocation(*glow_sh, "fadeAlpha");
    float texel[2]   = {1.0f / tex.width, 1.0f / tex.height};

    SetShaderValue(*glow_sh, glow_loc,  &glow,  SHADER_UNIFORM_FLOAT);
    SetShaderValue(*glow_sh, texel_loc, texel,  SHADER_UNIFORM_VEC2);
    SetShaderValue(*glow_sh, fade_loc,  &fade,  SHADER_UNIFORM_FLOAT);

    // Center image in the right half of the screen
    float x = SCREEN_W/2 + (SCREEN_W/2 - tex.width)  * 0.5f;
    float y = (SCREEN_H - tex.height) * 0.5f + 70;

    BeginShaderMode(*glow_sh);
        DrawTexture(tex, (int)x, (int)y, WHITE);
    EndShaderMode();

    // Title centered in right half, same y=100 as anim2 title
    Color title_col = (Color){C_WHITE.r, C_WHITE.g, C_WHITE.b, (unsigned char)(fade * 255)};
    Vector2 t3sz = MeasureTextEx(roboto, "PufferNet", FONT_TITLE, 1);
    DrawTextEx(roboto, "PufferNet",
        (Vector2){SCREEN_W*3/4 - t3sz.x/2, 100}, FONT_TITLE, 1, title_col);
}

/* ============================================================
   ANIMATION 4: Spiral in (9–12s)
   Anim2+3 fade out over 0.5s. Stars spiral in.
   ============================================================ */

static void draw_anim4(float anim_t, Font roboto, Shader *star_sh,
                       Shader *glow_sh, Texture2D tex) {
    float fade_out = 1.0f - fminf(anim_t / 0.5f, 1.0f);

    if (fade_out > 0.0f) {
        draw_anim3(5.5f, glow_sh, tex, star_sh, roboto);
        DrawRectangle(0, 0, SCREEN_W, SCREEN_H,
            (Color){BG.r, BG.g, BG.b, (unsigned char)((1.0f - fade_out) * 255)});
    }

    build_bg_verts_spiral(1.0f, 0);
    draw_stars(bg_verts, bg_draw_count, star_sh);
}

// ─── Text overlay helpers ─────────────────────────────────────────────────────

// Returns 0..1 alpha: starts fading in at `start`, fully opaque after `fade` seconds
static float fade_in(float t, float start, float fade) {
    return fmaxf(0.0f, fminf((t - start) / fade, 1.0f));
}

// Draw centered text at (SCREEN_W/2, y) with given color and 0..1 alpha
static void draw_centered(Font f, const char *text, float y, Color col, float alpha) {
    if (alpha <= 0.0f) return;
    col.a = (unsigned char)(alpha * 255);
    Vector2 sz = MeasureTextEx(f, text, FONT_TITLE, 1);
    DrawTextEx(f, text, (Vector2){SCREEN_W * 0.5f - sz.x * 0.5f, y}, FONT_TITLE, 1, col);
}

// Draw two strings side-by-side, centered together, at y
static void draw_centered2(Font f,
                           const char *s1, Color c1,
                           const char *s2, Color c2,
                           float y, float alpha) {
    if (alpha <= 0.0f) return;
    c1.a = c2.a = (unsigned char)(alpha * 255);
    Vector2 sz1 = MeasureTextEx(f, s1, FONT_TITLE, 1);
    Vector2 sz2 = MeasureTextEx(f, s2, FONT_TITLE, 1);
    float x = SCREEN_W * 0.5f - (sz1.x + sz2.x) * 0.5f;
    DrawTextEx(f, s1, (Vector2){x,          y}, FONT_TITLE, 1, c1);
    DrawTextEx(f, s2, (Vector2){x + sz1.x,  y}, FONT_TITLE, 1, c2);
}

// ─── Haiku + title overlay (called every frame from anim5 onward) ─────────────
#define T_NOVA  31.5f

static void draw_haiku_overlay(float t, Font roboto) {
    float lh = FONT_TITLE + 16;  // line height
    draw_centered(roboto, "Each thought is a star",   HAIKU_Y,        C_WHITE, fade_in(t, 26.5f, 0.5f));
    draw_centered(roboto, "Few form a constellation",  HAIKU_Y + lh,   C_WHITE, fade_in(t, 28.5f, 0.5f));
    draw_centered(roboto, "But they shine brightest",  HAIKU_Y + lh*2, C_WHITE, fade_in(t, T_NOVA, 0.5f));
    draw_centered2(roboto, "PufferLib ", C_WHITE, "4.0", C_CYAN,
                   FOOTER_Y, fade_in(t, T_NOVA + 2.0f, 0.5f));
    draw_centered2(roboto, "Download now at ", C_WHITE, "puffer.ai", C_CYAN,
                   FOOTER_Y + lh, fade_in(t, T_NOVA + 4.0f, 0.5f));
}

/* ============================================================
   ANIMATION 5: Thumbnails fade in, then spiral in (22–end)
   Each thumbnail placed at a fixed position around the screen.
   After .25s stagger fade-in, positions included in spiral.
   ============================================================ */

#define NUM_THUMBS 32

static const char *thumb_paths[NUM_THUMBS] = {
    "../puffer.ai/docs/assets/2048_thumbnail.png",
    "../puffer.ai/docs/assets/blastar_thumbnail.png",
    "../puffer.ai/docs/assets/breakout_thumbnail.png",
    "../puffer.ai/docs/assets/cartpole_thumbnail.png",
    "../puffer.ai/docs/assets/connect4_thumbnail.png",
    "../puffer.ai/docs/assets/convert_thumbnail.png",
    "../puffer.ai/docs/assets/cpr_thumbnail.png",
    "../puffer.ai/docs/assets/drone_thumbnail.png",
    "../puffer.ai/docs/assets/enduro_thumbnail.png",
    "../puffer.ai/docs/assets/freeway_thumbnail.png",
    "../puffer.ai/docs/assets/go_thumbnail.png",
    "../puffer.ai/docs/assets/gpudrive_thumbnail.png",
    "../puffer.ai/docs/assets/impulse_wars_thumbnail.png",
    "../puffer.ai/docs/assets/moba_thumbnail.png",
    "../puffer.ai/docs/assets/nmmo3_thumbnail.png",
    "../puffer.ai/docs/assets/pacman_thumbnail.png",
    "../puffer.ai/docs/assets/pong_thumbnail.png",
    "../puffer.ai/docs/assets/robocode_thumbnail.png",
    "../puffer.ai/docs/assets/rware_thumbnail.png",
    "../puffer.ai/docs/assets/slimevolley_thumbnail.png",
    "../puffer.ai/docs/assets/snake_thumbnail.png",
    "../puffer.ai/docs/assets/squared_thumbnail.png",
    "../puffer.ai/docs/assets/tactical_thumbnail.png",
    "../puffer.ai/docs/assets/target_thumbnail.png",
    "../puffer.ai/docs/assets/tcg_thumbnail.png",
    "../puffer.ai/docs/assets/template_thumbnail.png",
    "../puffer.ai/docs/assets/terraform_thumbnail.png",
    "../puffer.ai/docs/assets/tetris_thumbnail.png",
    "../puffer.ai/docs/assets/tower_climb_thumbnail.png",
    "../puffer.ai/docs/assets/trash_pickup_thumbnail.png",
    "../puffer.ai/docs/assets/tripletriad_thumbnail.png",
    "../puffer.ai/docs/assets/whisker_racer_thumbnail.png",
};

static Texture2D thumbs[NUM_THUMBS];
static float thumb_r[NUM_THUMBS];
static float thumb_angle[NUM_THUMBS];
static float thumb_vr[NUM_THUMBS];

static void init_anim5(void) {
    for (int i = 0; i < NUM_THUMBS; i++)
        thumbs[i] = LoadTexture(thumb_paths[i]);

    // Spawn in a slight outward spiral: radius grows by THUMB_SPIRAL_DR per step
    float cx = SPIRAL_CX, cy = SPIRAL_CY;
    float ring_r     = 400.0f;
    float spiral_dr  = 12.0f;   // px added to radius per thumbnail
    for (int i = 0; i < NUM_THUMBS; i++) {
        thumb_r[i]     = ring_r + i * spiral_dr;
        thumb_angle[i] = (float)i / (float)NUM_THUMBS * 2.0f * 3.14159265f;
        thumb_vr[i]    = 0.0f;
    }
    (void)cx; (void)cy;
}

static void unload_anim5(void) {
    for (int i = 0; i < NUM_THUMBS; i++)
        UnloadTexture(thumbs[i]);
}

static void draw_anim5(float anim_t, float dt, Shader *star_sh) {
    float cx = SPIRAL_CX, cy = SPIRAL_CY;

    // Stars already integrated via update_bg_spiral each frame
    build_bg_verts_spiral(1.0f, 0);
    draw_stars(bg_verts, bg_draw_count, star_sh);

    for (int i = 0; i < NUM_THUMBS; i++) {
        // Stagger in reverse angular order: highest angle appears first,
        // so each new thumbnail materialises behind the already-spiraling ones
        int rev = NUM_THUMBS - 1 - i;
        float fade_start = rev * 0.07f;
        float alpha = fmaxf(0.0f, fminf((anim_t - fade_start) / 0.25f, 1.0f));

        // Only integrate once visible — invisible ones stay at their start position
        if (alpha > 0.0f) {
            thumb_vr[i]    -= SPIRAL_GM / (thumb_r[i] * thumb_r[i]) * dt;
            thumb_r[i]     += thumb_vr[i] * dt;
            thumb_angle[i] += SPIRAL_OMEGA * SPIRAL_OMEGA_R / (thumb_r[i] + SPIRAL_OMEGA_R) * dt;
            if (thumb_r[i] < 1.0f) thumb_r[i] = 1.0f;
        }

        if (alpha <= 0.0f) continue;

        float r   = thumb_r[i];
        float ang = thumb_angle[i];
        float px  = cx + r * cosf(ang);
        float py  = cy + r * sinf(ang);

        // Scale proportional to r so size reaches 1px at center
        float tw = thumbs[i].width, th = thumbs[i].height;
        float base_scale = fminf(300.0f / tw, 300.0f / th);
        float r_frac = thumb_r[i] / (400.0f + (NUM_THUMBS - 1) * 12.0f);
        float size_scale = fmaxf(1.0f / fmaxf(tw, th), base_scale * r_frac);
        float dw = tw * size_scale, dh = th * size_scale;

        Rectangle src  = {0, 0, tw, th};
        Rectangle dest = {px - dw*0.5f, py - dh*0.5f, dw, dh};
        Color tint = {255, 255, 255, (unsigned char)(alpha * 255)};
        DrawTexturePro(thumbs[i], src, dest, (Vector2){0,0}, 0.0f, tint);
    }
}

// ─── Video recording ──────────────────────────────────────────────────────────
#define RECORD_FPS   30

typedef struct { int pipefd[2]; pid_t pid; } VideoRecorder;

static bool OpenVideo(VideoRecorder *rec, const char *filename, int w, int h) {
    if (pipe(rec->pipefd) == -1) { fprintf(stderr, "pipe failed\n"); return false; }
    rec->pid = fork();
    if (rec->pid == -1) { fprintf(stderr, "fork failed\n"); return false; }
    if (rec->pid == 0) {
        close(rec->pipefd[1]);
        dup2(rec->pipefd[0], STDIN_FILENO);
        close(rec->pipefd[0]);
        for (int fd = 3; fd < 256; fd++) close(fd);
        char sz[32]; snprintf(sz, sizeof(sz), "%dx%d", w, h);
        char fps[8]; snprintf(fps, sizeof(fps), "%d", RECORD_FPS);
        execlp("ffmpeg", "ffmpeg", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", sz, "-r", fps, "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "medium", "-crf", "23",
               "-loglevel", "error",
               filename, NULL);
        fprintf(stderr, "exec ffmpeg failed\n");
        _exit(1);
    }
    close(rec->pipefd[0]);
    return true;
}

static void WriteFrame(VideoRecorder *rec, int w, int h) {
    rlDrawRenderBatchActive();   // flush pending RayLib draws before reading pixels
    unsigned char *data = rlReadScreenPixels(w, h);
    write(rec->pipefd[1], data, w * h * 4);
    RL_FREE(data);
}

static void CloseVideo(VideoRecorder *rec) {
    close(rec->pipefd[1]);
    waitpid(rec->pid, NULL, 0);
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(void) {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(SCREEN_W, SCREEN_H, "PufferLib Trailer");
    SetTargetFPS(60);
    glEnable(GL_PROGRAM_POINT_SIZE);

    Shader star_shader = LoadShader(
        TextFormat("resources/trailer/star_%i.vs", GLSL_VERSION),
        TextFormat("resources/trailer/star_%i.fs", GLSL_VERSION)
    );
    Shader glow_shader = LoadShader(
        TextFormat("resources/trailer/glow_%i.vs", GLSL_VERSION),
        TextFormat("resources/trailer/glow_%i.fs", GLSL_VERSION)
    );
    Texture2D puffernet = LoadTexture("resources/trailer/PufferNet.png");

    Font roboto = LoadFontEx("resources/shared/Roboto-Regular.ttf", FONT_TITLE, NULL, 0);
    SetTextureFilter(roboto.texture, TEXTURE_FILTER_BILINEAR);

    init_bg_stars();
    init_anim1();
    init_anim2();
    init_anim5();

    VideoRecorder recorder;
    int phase = 0;
    float phase_time = 0;
    bool recording = OpenVideo(&recorder, "trailer/v1.mp4", SCREEN_W, SCREEN_H);
    if (!recording) fprintf(stderr, "Warning: video recording disabled\n");

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        phase_time += dt;

        if (phase == 0 && phase_time > 10.0f) {
            if (recording) CloseVideo(&recorder);
            phase = 1;
            phase_time = 0;
            recording = OpenVideo(&recorder, "trailer/v2.mp4", SCREEN_W, SCREEN_H);
            init_bg_stars(); // Reset stars for the second video
        }

        if (phase == 1 && phase_time > 42.5f) break;

        BeginDrawing();
        ClearBackground(BG);

        if (phase == 0) {
            update_anim1(dt, roboto, &star_shader);
        } else {
            float t = phase_time;
            if (t < 5.0f) {
                build_bg_verts();
                draw_stars(bg_verts, bg_draw_count, &star_shader);
            }
            else if (t < 10.5f) {
                build_bg_verts();
                draw_stars(bg_verts, bg_draw_count, &star_shader);
                draw_anim3(t - 5.0f, &glow_shader, puffernet, &star_shader, roboto);
            }
            else if (t < 12.5f) {
                update_bg_spiral(dt);
                draw_anim4(t - 10.5f, roboto, &star_shader, &glow_shader, puffernet);
            }
            else if (t < 22.5f) {
                draw_anim4(2.0f, roboto, &star_shader, &glow_shader, puffernet);
            }
            else if (t < 31.5f) {
                update_bg_spiral(dt);
                draw_anim5(t - 22.5f, dt, &star_shader);
                draw_haiku_overlay(t, roboto);
            }
            else {
                update_bg_nova(dt);
                build_bg_verts_spiral(1.0f, 1);
                draw_stars(bg_verts, bg_draw_count, &star_shader);
                draw_haiku_overlay(t, roboto);
            }
        }

        static int frame_counter = 0;
        if (recording && (frame_counter++ % 2 == 0))
            WriteFrame(&recorder, SCREEN_W, SCREEN_H);
        EndDrawing();
    }

    if (recording) CloseVideo(&recorder);
    unload_anim5();
    UnloadFont(roboto);
    UnloadShader(star_shader);
    UnloadShader(glow_shader);
    UnloadTexture(puffernet);
    CloseWindow();
    return 0;
}
