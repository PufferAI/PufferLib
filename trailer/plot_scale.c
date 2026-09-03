#include "trailer/plot_scale.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "raymath.h"
#include "rlgl.h"

#if defined(__APPLE__)
    #define GL_SILENCE_DEPRECATION
    #include <OpenGL/gl3.h>
#else
    #include "glad.h"
#endif

#define W 1920
#define H 1080

#define PUFF_CYAN  ((Color){0, 187, 187, 255})
#define PUFF_GREEN ((Color){40, 230, 100, 255})
#define PUFF_WHITE ((Color){241, 241, 241, 255})
#define PUFF_GRID  ((Color){18, 72, 72, 255})

#define N_PTS 4
#define N_CURVE 256
static float LEFT = 192.0f;
static float RIGHT = (1920.0f - 192.0f);
static float TOP = 168.0f;
static float BOTTOM = (1080.0f - 156.0f);
static Rectangle PLOT_BOX = {0, 0, 1920, 1080};

typedef struct Glyph {
    float x, y, i;
    float r, g, b, a;
} Glyph;

static const float HIDDEN[N_PTS] = {128, 256, 512, 1024};
static const int LAYERS[N_PTS] = {2, 3, 4, 5};
static const char *PARAM_LABELS[N_PTS] = {"113.9K", "621.1K", "3.2M", "15.9M"};
static const float SPS[N_PTS] = {
    45337600.0f, 16516200.0f, 4727100.0f, 1167300.0f
};
static const float VRAM[N_PTS] = {
    1.1240234375f, 1.5458984375f, 2.5869140625f, 5.1571044921875f
};

static const float X_MIN = 96.0f;
static const float X_MAX = 1280.0f;
static const float SPS_MIN = 8.0e5f;
static const float SPS_MAX = 7.0e7f;
static const float VRAM_MIN = 0.0f;
static const float VRAM_MAX = 6.0f;

static float sps_a, sps_b, vram_a, vram_b;
static Vector2 sps_curve[N_CURVE];
static Vector2 vram_curve[N_CURVE];
static Glyph glyphs[N_PTS * 2];
static int ready;

static float unlerp(float a, float b, float x) { return (x - a) / (b - a); }

static float map_x(float h) {
    return LEFT + unlerp(logf(X_MIN), logf(X_MAX), logf(h)) * (RIGHT - LEFT);
}
static float map_sps(float v) {
    return BOTTOM - unlerp(logf(SPS_MIN), logf(SPS_MAX), logf(v)) * (BOTTOM - TOP);
}
static float map_vram(float v) {
    return BOTTOM - unlerp(VRAM_MIN, VRAM_MAX, v) * (BOTTOM - TOP);
}

static void fit_loglog(const float *x, const float *y, int n, float *a, float *b) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        double lx = log((double)x[i]);
        double ly = log((double)y[i]);
        sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly;
    }
    double d = n * sxx - sx * sx;
    *b = (float)((n * sxy - sx * sy) / d);
    *a = (float)((sy - (*b) * sx) / n);
}

static void fit_linear(const float *x, const float *y, int n, float *a, float *b) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        sx += x[i]; sy += y[i]; sxx += (double)x[i] * x[i]; sxy += (double)x[i] * y[i];
    }
    double d = n * sxx - sx * sx;
    *b = (float)((n * sxy - sx * sy) / d);
    *a = (float)((sy - (*b) * sx) / n);
}

static float pred_power(float a, float b, float x) {
    return expf(a + b * logf(x));
}

static Color ca(Color c, float a) {
    if (a < 0) a = 0;
    if (a > 1) a = 1;
    return (Color){c.r, c.g, c.b, (unsigned char)(c.a * a)};
}

static void text_c(Font font, const char *s, float x, float y, float size, Color c) {
    Vector2 sz = MeasureTextEx(font, s, size, 0);
    DrawTextEx(font, s, (Vector2){x - sz.x * 0.5f, y}, size, 0, c);
}

static void format_sps(char *buf, size_t n, float v) {
    snprintf(buf, n, "%.0fM", v / 1.0e6f);
}

static void draw_glow_spline(Vector2 *pts, int n, Color color) {
    if (n < 2) return;
    DrawSplineLinear(pts, n, 12.0f, Fade(color, 0.10f));
    DrawSplineLinear(pts, n, 6.0f, Fade(color, 0.22f));
    DrawSplineLinear(pts, n, 2.6f, color);
}

static void plot_gl(Glyph *g, int n, Shader *shader, float time) {
    if (!shader || shader->id == 0 || n <= 0) return;
    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, n * (int)sizeof(Glyph), g, GL_STATIC_DRAW);
        glVertexAttribPointer(shader->locs[SHADER_LOC_VERTEX_POSITION],
            3, GL_FLOAT, GL_FALSE, sizeof(Glyph), 0);
        glEnableVertexAttribArray(shader->locs[SHADER_LOC_VERTEX_POSITION]);
        int vertex_color = shader->locs[SHADER_LOC_VERTEX_COLOR];
        glVertexAttribPointer(vertex_color, 4, GL_FLOAT, GL_FALSE,
            sizeof(Glyph), (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(vertex_color);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    rlDrawRenderBatchActive();
    rlSetBlendFactors(GL_ONE, GL_ONE, GL_MAX);
    rlSetBlendMode(RL_BLEND_CUSTOM);
    int time_loc = GetShaderLocation(*shader, "currentTime");
    glUseProgram(shader->id);
        glUniform1f(time_loc, time);
        Matrix mvp = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection());
        glUniformMatrix4fv(shader->locs[SHADER_LOC_MATRIX_MVP], 1, false, MatrixToFloat(mvp));
        glBindVertexArray(vao);
            glDrawArrays(GL_POINTS, 0, n);
        glBindVertexArray(0);
    glUseProgram(0);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    rlSetBlendMode(RL_BLEND_ALPHA);
}

static void rebuild_curves(void) {
    for (int i = 0; i < N_CURVE; i++) {
        float t = i / (float)(N_CURVE - 1);
        float h = expf(logf(X_MIN) + t * (logf(X_MAX) - logf(X_MIN)));
        sps_curve[i] = (Vector2){map_x(h), map_sps(pred_power(sps_a, sps_b, h))};
        vram_curve[i] = (Vector2){map_x(h), map_vram(vram_a + vram_b * h)};
    }
    for (int i = 0; i < N_PTS; i++) {
        glyphs[i] = (Glyph){
            map_x(HIDDEN[i]), map_sps(SPS[i]), 0.0f,
            0.0f, 0.28f, 0.28f, 1.0f
        };
        glyphs[N_PTS + i] = (Glyph){
            map_x(HIDDEN[i]), map_vram(VRAM[i]), 0.0f,
            0.05f, 0.32f, 0.12f, 1.0f
        };
    }
}

static void layout_plot(Rectangle box) {
    PLOT_BOX = box;
    float pad_l = 78.0f, pad_r = 70.0f, pad_t = 118.0f, pad_b = 72.0f;
    if (box.width < 700) {
        pad_l = 58.0f;
        pad_r = 52.0f;
        pad_t = 96.0f;
        pad_b = 64.0f;
    }
    LEFT = box.x + pad_l;
    RIGHT = box.x + box.width - pad_r;
    TOP = box.y + pad_t;
    BOTTOM = box.y + box.height - pad_b;
    if (RIGHT - LEFT < 120) {
        LEFT = box.x + 40;
        RIGHT = box.x + box.width - 40;
    }
    if (BOTTOM - TOP < 80) {
        TOP = box.y + 48;
        BOTTOM = box.y + box.height - 40;
    }
    rebuild_curves();
}

void plot_scale_init(void) {
    fit_loglog(HIDDEN, SPS, N_PTS, &sps_a, &sps_b);
    fit_linear(HIDDEN, VRAM, N_PTS, &vram_a, &vram_b);
    layout_plot((Rectangle){0, 0, W, H});
    ready = 1;
}

void plot_scale_draw(Font title, Font body, Shader *star, float reveal, float alpha) {
    plot_scale_draw_box(title, body, star, reveal, alpha, (Rectangle){0, 0, W, H});
}

void plot_scale_draw_box(Font title, Font body, Shader *star, float reveal, float alpha,
                         Rectangle box) {
    if (!ready) plot_scale_init();
    if (alpha <= 0.001f) return;
    if (reveal < 0) reveal = 0;
    if (reveal > 1) reveal = 1;
    layout_plot(box);

    int n_show = 2 + (int)(reveal * (N_CURVE - 2));
    if (n_show < 2) n_show = 2;
    if (n_show > N_CURVE) n_show = N_CURVE;
    float xa = LEFT + reveal * (RIGHT - LEFT);
    float cx = box.x + box.width * 0.5f;
    int compact = box.width < W * 0.85f;

    if (!compact) {
        text_c(title, "Breakout  ·  scale", cx, box.y + 28.0f, 40.0f, ca(PUFF_WHITE, alpha));
        text_c(body, "8192 envs   32k minibatch   replay 1   RTX 5090",
               cx, box.y + 80.0f, 22.0f, ca(PUFF_WHITE, alpha * 0.70f));
    } else {
        text_c(body, "8192 envs  ·  32k minibatch  ·  RTX 5090",
               cx, box.y + 10.0f, 18.0f, ca(PUFF_WHITE, alpha * 0.72f));
    }

    char eq_sps[64], eq_vram[64];
    snprintf(eq_sps, sizeof(eq_sps), "SPS  ~  H^%.2f", sps_b);
    snprintf(eq_vram, sizeof(eq_vram), "VRAM  ~  %.2f + %.4f H", vram_a, vram_b);
    float lab = compact ? 18.0f : 22.0f;
    float sps_w = MeasureTextEx(body, eq_sps, lab, 0).x;
    float vram_w = MeasureTextEx(body, eq_vram, lab, 0).x;
    float legend_w = 48.0f + sps_w + 40.0f + 48.0f + vram_w;
    float legend_x = cx - legend_w * 0.5f;
    float legend_y = compact ? box.y + 36.0f : box.y + 118.0f;
    DrawLineEx((Vector2){legend_x, legend_y + 10},
        (Vector2){legend_x + 36, legend_y + 10}, 3.5f, ca(PUFF_CYAN, alpha));
    DrawTextEx(body, eq_sps, (Vector2){legend_x + 48, legend_y}, lab, 0, ca(PUFF_CYAN, alpha));
    float g0 = legend_x + 48.0f + sps_w + 40.0f;
    DrawLineEx((Vector2){g0, legend_y + 10},
        (Vector2){g0 + 36, legend_y + 10}, 3.5f, ca(PUFF_GREEN, alpha));
    DrawTextEx(body, eq_vram, (Vector2){g0 + 48, legend_y}, lab, 0, ca(PUFF_GREEN, alpha));

    const float sps_ticks[] = {1e6f, 2e6f, 5e6f, 1e7f, 2e7f, 5e7f};
    const int n_sps_ticks = (int)(sizeof(sps_ticks) / sizeof(sps_ticks[0]));
    const float vram_ticks[] = {0, 1, 2, 3, 4, 5, 6};
    const int n_vram_ticks = (int)(sizeof(vram_ticks) / sizeof(vram_ticks[0]));

    for (int i = 0; i < n_sps_ticks; i++) {
        float y = map_sps(sps_ticks[i]);
        if (y < TOP || y > BOTTOM) continue;
        DrawLineEx((Vector2){LEFT, y}, (Vector2){RIGHT, y}, 1.0f, ca(PUFF_GRID, alpha * 0.55f));
    }
    for (int i = 0; i < N_PTS; i++) {
        float x = map_x(HIDDEN[i]);
        DrawLineEx((Vector2){x, TOP}, (Vector2){x, BOTTOM}, 1.0f, ca(PUFF_GRID, alpha * 0.45f));
    }

    BeginScissorMode((int)LEFT, (int)(TOP - 8), (int)(xa - LEFT + 4), (int)(BOTTOM - TOP + 16));
    draw_glow_spline(sps_curve, n_show, ca(PUFF_CYAN, alpha));
    draw_glow_spline(vram_curve, n_show, ca(PUFF_GREEN, alpha));
    EndScissorMode();

    DrawLineEx((Vector2){LEFT, TOP}, (Vector2){LEFT, BOTTOM}, 2.0f, ca(PUFF_CYAN, alpha));
    DrawLineEx((Vector2){RIGHT, TOP}, (Vector2){RIGHT, BOTTOM}, 2.0f, ca(PUFF_GREEN, alpha));
    DrawLineEx((Vector2){LEFT, BOTTOM}, (Vector2){RIGHT, BOTTOM}, 2.0f, ca(PUFF_WHITE, alpha));

    for (int i = 0; i < n_sps_ticks; i++) {
        float y = map_sps(sps_ticks[i]);
        if (y < TOP || y > BOTTOM) continue;
        DrawLineEx((Vector2){LEFT - 8, y}, (Vector2){LEFT + 8, y}, 2.0f, ca(PUFF_CYAN, alpha));
        char tick[16];
        format_sps(tick, sizeof(tick), sps_ticks[i]);
        float ts = compact ? 16.0f : 20.0f;
        Vector2 sz = MeasureTextEx(body, tick, ts, 0);
        DrawTextEx(body, tick, (Vector2){LEFT - 14.0f - sz.x, y - sz.y * 0.5f},
            ts, 0, ca(PUFF_CYAN, alpha));
    }
    for (int i = 0; i < n_vram_ticks; i++) {
        float y = map_vram(vram_ticks[i]);
        DrawLineEx((Vector2){RIGHT - 8, y}, (Vector2){RIGHT + 8, y}, 2.0f, ca(PUFF_GREEN, alpha));
        char tick[16];
        snprintf(tick, sizeof(tick), "%.0f", vram_ticks[i]);
        float ts = compact ? 16.0f : 20.0f;
        Vector2 sz = MeasureTextEx(body, tick, ts, 0);
        DrawTextEx(body, tick, (Vector2){RIGHT + 14.0f, y - sz.y * 0.5f},
            ts, 0, ca(PUFF_GREEN, alpha));
    }
    for (int i = 0; i < N_PTS; i++) {
        float x = map_x(HIDDEN[i]);
        DrawLineEx((Vector2){x, BOTTOM - 8}, (Vector2){x, BOTTOM + 8}, 2.0f, ca(PUFF_WHITE, alpha));
        char dims[32];
        snprintf(dims, sizeof(dims), "(%.0fx%d)", HIDDEN[i], LAYERS[i]);
        float ts = compact ? 16.0f : 20.0f;
        text_c(body, PARAM_LABELS[i], x, BOTTOM + 12.0f, ts, ca(PUFF_WHITE, alpha));
        text_c(body, dims, x, BOTTOM + 12.0f + ts + 2.0f, ts, ca(PUFF_WHITE, alpha * 0.72f));
    }

    float axis = compact ? 18.0f : 24.0f;
    DrawTextPro(body, "SPS", (Vector2){box.x + 18.0f, (TOP + BOTTOM) * 0.5f},
        (Vector2){MeasureTextEx(body, "SPS", axis, 0).x * 0.5f, 10.0f},
        -90.0f, axis, 0, ca(PUFF_CYAN, alpha));
    DrawTextPro(body, "VRAM (GB)", (Vector2){box.x + box.width - 16.0f, (TOP + BOTTOM) * 0.5f},
        (Vector2){MeasureTextEx(body, "VRAM (GB)", axis, 0).x * 0.5f, 10.0f},
        90.0f, axis, 0, ca(PUFF_GREEN, alpha));
    text_c(body, "Parameters", cx, box.y + box.height - 28.0f, compact ? 18.0f : 22.0f,
           ca(PUFF_WHITE, alpha));

    int n_marks = 0;
    Glyph shown[N_PTS * 2];
    for (int i = 0; i < N_PTS; i++) {
        if (glyphs[i].x <= xa + 8) shown[n_marks++] = glyphs[i];
    }
    for (int i = 0; i < N_PTS; i++) {
        if (glyphs[N_PTS + i].x <= xa + 8) shown[n_marks++] = glyphs[N_PTS + i];
    }
    if (n_marks > 0) plot_gl(shown, n_marks, star, 1.570796f);
}
