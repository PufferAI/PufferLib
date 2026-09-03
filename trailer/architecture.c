// PufferLib 5.0 architecture diagram.
// Build:  ./build.sh trailer   (also writes ./trailer/architecture)
// Run:    ./trailer/architecture
// Stills: ./trailer/architecture --shot
// Record: ./trailer/architecture --record trailer/architecture.mp4
//
// Keys:  space pause   s shot   r restart   q quit

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "trailer/architecture.h"

#if defined(__APPLE__)
    #define GL_SILENCE_DEPRECATION
    #include <OpenGL/gl3.h>
#else
    #include "glad.h"
#endif

#define GLSL_VERSION 330

#define SCREEN_W 1920
#define SCREEN_H 1080
#define HEADER_H 96.0f
#define MARGIN_X 48.0f
#define MARGIN_BOT 48.0f
#define PANEL_GAP 16.0f
#define RECORD_FPS 30
#define MAX_PK 160
#define N_BUF 4
#define N_ENV 32          // envs per buffer (OMP loop trip count)
#define N_ENV_COLS 16     // one row per OMP thread
#define N_ENV_ROWS 2
#define N_TH 2            // OMP threads per buffer = num_threads / num_buffers
#define HORIZON 8         // rollout timesteps stored in the 2D buffers
#define N_AGENTS 16       // train tensor rows (agents; buffers gone)
#define MB_ROWS 4         // contiguous agent-rows per minibatch
#define ROWS_PER_BUF (N_AGENTS / N_BUF)
#define N_GPU 8
#define STAGE_BASE 0.20f  // viz-seconds per stage (STEP/H2D/FWD/D2H)
#define STAGE_NOISE 0.25f
#define MB_PERIOD 0.45f
#define XFER_SEC 0.70f    // time to sync a full rollout into the trainer
#define LR_MB_SEC 4.00f
#define LR_AR_SEC 1.00f
#define LR_UPD_SEC 0.40f
#define LOOP_SEC 12.0f

// Colors — full-bright cyan + green. No mid-blues.
#define C_BG     ((Color){4, 14, 16, 255})
#define C_PANEL  ((Color){8, 24, 26, 245})
#define C_WHITE  ((Color){241, 241, 241, 255})
#define C_MUTED  ((Color){186, 210, 210, 255})
#define C_CYAN   ((Color){0, 187, 187, 255})      // Peak-level brand cyan
#define C_CYAN_HI ((Color){210, 250, 245, 255})   // pale — OBS only
#define C_CYAN_LO ((Color){0, 98, 104, 255})      // deep — TERMINALS only
#define C_ACTIVE ((Color){0, 240, 230, 255})      // bright cyan — live only
#define C_GREEN  ((Color){40, 230, 100, 255})     // rollout levels 1–3
#define C_TEAL   C_CYAN
#define C_VIOLET C_CYAN
#define C_AMBER  C_CYAN
#define C_BLUE   C_CYAN
#define C_PINK   C_CYAN_LO

// Fonts: 4 sizes for the whole figure.
#define FONT_ATLAS_UI   96
#define FONT_ATLAS_MONO 80
#define FONT_TITLE      40   // PufferLib 5.0
#define FONT_HEAD       28   // 5 levels, Peak, panel titles
#define FONT_MID        22   // descriptions, channels, chips, serial
#define FONT_SMALL      18   // cells, axis ticks, HUD
#define LOGO_S          64.0f
#define RING_THICK      4.5f
#define MOTE_SIZE       1.65f

enum { ST_FWD, ST_D2H, ST_STEP, ST_H2D, N_ST };
enum { LR_FWD, LR_BWD, LR_AR, LR_UPD, N_LR };

static const char *ST_NAME[N_ST] = {"FORWARD", "D2H", "ENV STEP", "H2D"};

// ─── tiny helpers ─────────────────────────────────────────────────────────────
static unsigned uhash(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
static float clampf(float x, float a, float b) {
    return x < a ? a : (x > b ? b : x);
}
static float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static float ease(float t) {
    t = clampf(t, 0, 1);
    return t * t * (3.0f - 2.0f * t);
}
static Color calpha(Color c, float a) {
    if (a < 0) a = 0;
    if (a > 1) a = 1;
    return (Color){c.r, c.g, c.b, (unsigned char)(c.a * a)};
}
static Vector2 vlerp(Vector2 a, Vector2 b, float t) {
    return (Vector2){lerpf(a.x, b.x, t), lerpf(a.y, b.y, t)};
}
static void text(Font f, const char *s, float x, float y, float sz, Color c) {
    DrawTextEx(f, s, (Vector2){x, y}, sz, 0.4f, c);
}
static float tw(Font f, const char *s, float sz) {
    return MeasureTextEx(f, s, sz, 0.4f).x;
}
static void text_c(Font f, const char *s, float cx, float y, float sz, Color c) {
    text(f, s, cx - tw(f, s, sz) * 0.5f, y, sz, c);
}
static void text_r(Font f, const char *s, float rx, float y, float sz, Color c) {
    text(f, s, rx - tw(f, s, sz), y, sz, c);
}

typedef struct {
    float x, y, size_scale;
    float r, g, b, a;
} StarVertex;

static void glow_circ(Vector2 p, float r, Color c, float a) {
    DrawCircleV(p, r * 2.4f, calpha(c, 0.10f * a));
    DrawCircleV(p, r * 1.45f, calpha(c, 0.28f * a));
    DrawCircleV(p, r, calpha(c, a));
}

static void draw_star_points(Shader *sh, StarVertex *verts, int n) {
    if (!sh || sh->id == 0 || n <= 0) return;
    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, n * (int)sizeof(StarVertex), verts, GL_STREAM_DRAW);
        glVertexAttribPointer(sh->locs[SHADER_LOC_VERTEX_POSITION],
            3, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void *)0);
        glEnableVertexAttribArray(sh->locs[SHADER_LOC_VERTEX_POSITION]);
        glVertexAttribPointer(sh->locs[SHADER_LOC_VERTEX_COLOR],
            4, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void *)(3 * sizeof(float)));
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

static StarVertex star_vert(Vector2 p, Color c, float size, float seed) {
    return (StarVertex){
        p.x, p.y, size,
        c.r / 255.0f * 0.32f, c.g / 255.0f * 0.32f, c.b / 255.0f * 0.32f, seed
    };
}

static void rounded(Rectangle r, float rnd, Color c) {
    DrawRectangleRounded(r, rnd, 8, c);
}

static void stroke(Rectangle r, float rnd, float thick, Color c) {
    DrawRectangleRoundedLinesEx(r, rnd, 8, thick, c);
}

static void dashed_rect(Rectangle r, Color c, float thick, float t) {
    float dash = 14.0f, gap = 9.0f, period = dash + gap;
    float perim = 2.0f * (r.width + r.height);
    float off = fmodf(t * 42.0f, period);
    for (float s = -period; s < perim; s += period) {
        float a = s + off;
        float b = a + dash;
        if (b < 0 || a > perim) continue;
        if (a < 0) a = 0;
        if (b > perim) b = perim;
        // Walk two points along the perimeter.
        Vector2 pts[2];
        float u[2] = {a, b};
        for (int k = 0; k < 2; k++) {
            float p = u[k];
            if (p < r.width) {
                pts[k] = (Vector2){r.x + p, r.y};
            } else if (p < r.width + r.height) {
                pts[k] = (Vector2){r.x + r.width, r.y + (p - r.width)};
            } else if (p < 2 * r.width + r.height) {
                pts[k] = (Vector2){r.x + r.width - (p - r.width - r.height), r.y + r.height};
            } else {
                pts[k] = (Vector2){r.x, r.y + r.height - (p - 2 * r.width - r.height)};
            }
        }
        DrawLineEx(pts[0], pts[1], thick, c);
    }
}

// ─── packets ──────────────────────────────────────────────────────────────────
typedef struct {
    Vector2 a, b;
    float t, speed;
    Color c;
    float r;
    float seed;
    int alive;
} Packet;

typedef struct {
    Rectangle mem, env, trn, learn, gpu;
} Layout;

typedef struct {
    int stage;
    int visit;
    float t;     // time in current stage (negative = not started)
    float dur;
    int ht;      // current horizon index 0..HORIZON
    int full;    // finished all HORIZON steps, waiting to sync
    float th_prog[N_TH];  // 0..1 of this thread's partition (STEP only)
    float th_speed[N_TH]; // partition-fractions per second
    int emit;             // fire obs/rew/term motes after STEP
    int emit_ht;
} BufSim;

struct Arch {
    float t;
    float dt;
    int paused;
    Font ui, mono;
    Texture2D logo;
    Shader star;
    Packet pk[MAX_PK];
    int hide_hud;
    int stars_n;
    Vector2 stars[420];
    float star_br[420];
    BufSim bufs[N_BUF];
    unsigned char written[N_BUF][HORIZON][N_ENV];
    unsigned char bank[N_BUF][HORIZON][N_ENV]; // last synced slot (trainer)
    float appear[N_BUF][HORIZON]; // rollout cell fade-in
    int bank_valid;
    float xfer;  // 0 idle, 0..1 transferring
    int lr_stage;
    float lr_t;
    int lr_mb;   // start row of current minibatch
};
typedef struct Arch Viz;

static Layout make_layout(void) {
    float x = MARGIN_X, y0 = HEADER_H, y1 = (float)SCREEN_H - MARGIN_BOT, gap = PANEL_GAP;
    float W = (float)SCREEN_W - 2.0f * MARGIN_X;
    float H = y1 - y0;
    float mem_w = W * 0.26f;
    float env_w = W - mem_w - gap;
    float env_h = (H - 2.0f * gap) / 3.0f;
    float rest = H - env_h - 2.0f * gap;
    float gpu_h = rest * 0.32f;
    float learn_h = rest - gpu_h;
    float rx = x + mem_w + gap;
    float y_learn = y0 + env_h + gap;
    float y_gpu = y_learn + learn_h + gap;
    return (Layout){
        {x, y0, mem_w, env_h},
        {rx, y0, env_w, env_h},
        {x, y_learn, mem_w, H - env_h - gap},
        {rx, y_learn, env_w, learn_h},
        {rx, y_gpu, env_w, gpu_h},
    };
}

static float stage_dur(int buf, int visit, int stage) {
    unsigned h = uhash((unsigned)(buf * 10007 + visit * 131 + stage * 17 + 0x9e3779b9u));
    float u = (h & 0xffff) / 65535.0f;
    return STAGE_BASE * (1.0f + STAGE_NOISE * (2.0f * u - 1.0f));
}

static float lr_dur(int stage) {
    if (stage == LR_AR) return LR_AR_SEC;
    if (stage == LR_UPD) return LR_UPD_SEC;
    // FWD and BWD split whatever is left of the minibatch.
    return 0.5f * (LR_MB_SEC - LR_AR_SEC - LR_UPD_SEC);
}

// Persistent per-thread speeds so T0/T1 drift across the buffer's horizon.
// Guarantee a >=1.5x gap so they never stay column-aligned.
static void resample_thread_speeds(BufSim *s, int b) {
    unsigned h0 = uhash((unsigned)(b * 10007 + s->visit * 131 + 0x51u));
    unsigned h1 = uhash((unsigned)(b * 10007 + s->visit * 131 + 0xC21u));
    float u0 = (h0 & 0xffff) / 65535.0f;
    float u1 = (h1 & 0xffff) / 65535.0f;
    float fast = (1.0f / STAGE_BASE) * (0.95f + 0.45f * u0);
    float slow = fast / (1.55f + 0.55f * u1);
    int t0_fast = (h0 >> 16) & 1;
    s->th_speed[0] = t0_fast ? fast : slow;
    s->th_speed[1] = t0_fast ? slow : fast;
    for (int th = 0; th < N_TH; th++) s->th_prog[th] = 0;
}

static void reset_step_threads(BufSim *s) {
    for (int th = 0; th < N_TH; th++) s->th_prog[th] = 0;
}

static int step_threads_done(const BufSim *s) {
    for (int th = 0; th < N_TH; th++) if (s->th_prog[th] < 1.0f) return 0;
    return 1;
}

// OMP schedule(static): thread th owns a contiguous half and walks it serially.
static int omp_cursor(const BufSim *s, int th) {
    if (s->t < 0 || s->stage != ST_STEP || s->full) return -1;
    if (s->th_prog[th] >= 1.0f) return -1; // finished this partition, idle
    int half = N_ENV / N_TH;
    int i = (int)(clampf(s->th_prog[th], 0, 0.999f) * half);
    return th * half + i;
}

static void reset_sim(Viz *v) {
    memset(v->written, 0, sizeof v->written);
    memset(v->appear, 0, sizeof v->appear);
    memset(v->bank, 1, sizeof v->bank); // previous slot already full (async)
    for (int b = 0; b < N_BUF; b++) {
        v->bufs[b].stage = ST_FWD;
        v->bufs[b].visit = 0;
        v->bufs[b].t = -(float)b * 0.18f;
        v->bufs[b].dur = stage_dur(b, 0, ST_FWD);
        v->bufs[b].ht = 0;
        v->bufs[b].full = 0;
        resample_thread_speeds(&v->bufs[b], b);
        v->bufs[b].emit = 0;
        v->bufs[b].emit_ht = 0;
    }
    v->xfer = 0;
    v->bank_valid = 1;
    v->lr_stage = 0;
    v->lr_t = 0;
    v->lr_mb = 0;
}

static float chunk_fill(const Viz *v, int b, int t) {
    int n = 0;
    for (int e = 0; e < N_ENV; e++) n += v->written[b][t][e];
    return n / (float)N_ENV;
}

static void mark_written(Viz *v, int b) {
    BufSim *s = &v->bufs[b];
    if (s->t < 0 || s->stage != ST_STEP || s->ht >= HORIZON) return;
    int half = N_ENV / N_TH;
    for (int th = 0; th < N_TH; th++) {
        int lo = th * half;
        int n = (s->th_prog[th] >= 1.0f)
                    ? half
                    : (int)(clampf(s->th_prog[th], 0, 0.999f) * half) + 1;
        if (s->th_prog[th] <= 0.0001f) n = 0;
        for (int k = 0; k < n && k < half; k++)
            v->written[b][s->ht][lo + k] = 1;
    }
}

static int all_full(const Viz *v) {
    for (int b = 0; b < N_BUF; b++) if (!v->bufs[b].full) return 0;
    return 1;
}

static void step_sim(Viz *v, float dt) {
    if (dt <= 0) return;

    if (v->xfer > 0) {
        v->xfer += dt / XFER_SEC;
        if (v->xfer >= 1.0f) {
            memcpy(v->bank, v->written, sizeof v->written);
            memset(v->written, 0, sizeof v->written);
            v->bank_valid = 1;
            v->xfer = 0;
            for (int b = 0; b < N_BUF; b++) {
                v->bufs[b].ht = 0;
                v->bufs[b].full = 0;
                v->bufs[b].stage = ST_FWD;
                v->bufs[b].t = 0;
                v->bufs[b].visit++;
                v->bufs[b].dur = stage_dur(b, v->bufs[b].visit, ST_FWD);
                resample_thread_speeds(&v->bufs[b], b);
            }
        }
    } else {
        for (int b = 0; b < N_BUF; b++) {
            BufSim *s = &v->bufs[b];
            if (s->full) continue;
            s->t += dt;
            if (s->t < 0) continue;

            if (s->stage == ST_STEP) {
                for (int th = 0; th < N_TH; th++) {
                    s->th_prog[th] += dt * s->th_speed[th];
                    if (s->th_prog[th] > 1.0f) s->th_prog[th] = 1.0f;
                }
                mark_written(v, b);
                if (!step_threads_done(s)) continue;
                if (s->ht < HORIZON) {
                    for (int e = 0; e < N_ENV; e++) v->written[b][s->ht][e] = 1;
                    s->emit = 1;
                    s->emit_ht = s->ht;
                }
                s->stage = ST_H2D;
                s->visit++;
                s->dur = stage_dur(b, s->visit, s->stage);
                s->t = 0;
                continue;
            }

            while (!s->full && s->stage != ST_STEP && s->t >= s->dur) {
                s->t -= s->dur;
                int prev = s->stage;
                s->stage = (s->stage + 1) % N_ST;
                s->visit++;
                s->dur = stage_dur(b, s->visit, s->stage);
                if (s->stage == ST_STEP) {
                    reset_step_threads(s);
                    s->t = 0;
                }
                // FWD -> D2H -> STEP -> H2D, then next t
                if (prev == ST_H2D) {
                    s->ht++;
                    if (s->ht >= HORIZON) {
                        s->ht = HORIZON;
                        s->full = 1;
                        s->t = 0;
                    }
                }
            }
        }
        if (all_full(v)) v->xfer = 0.001f;
    }

    {
        float k = 1.0f - expf(-dt / 0.16f);
        for (int b = 0; b < N_BUF; b++) {
            for (int t = 0; t < HORIZON; t++) {
                float tgt = chunk_fill(v, b, t);
                if (v->xfer > 0) tgt *= (1.0f - v->xfer);
                v->appear[b][t] += (tgt - v->appear[b][t]) * k;
            }
        }
    }

    // Train the ready slot even while the next rollout (and transpose) runs.
    if (v->bank_valid) {
        v->lr_t += dt;
        float dur = lr_dur(v->lr_stage);
        while (v->lr_t >= dur) {
            v->lr_t -= dur;
            v->lr_stage++;
            if (v->lr_stage >= N_LR) {
                v->lr_stage = 0;
                v->lr_mb = (v->lr_mb + MB_ROWS) % N_AGENTS;
            }
            dur = lr_dur(v->lr_stage);
        }
    }
}

static void panel(Rectangle r, Color accent, float a, const char *title,
                  const char *desc, Font mono) {
    float fill_a = lerpf(0.45f, 0.96f, a);
    rounded(r, 0.035f, calpha(C_PANEL, fill_a));
    DrawRectangleRounded(
        (Rectangle){r.x, r.y, 5, r.height}, 0.0f, 2,
        calpha(accent, 0.25f + 0.75f * a));
    stroke(r, 0.035f, 1.6f, calpha(accent, 0.22f + 0.55f * a));
    text(mono, title, r.x + 22, r.y + 12, FONT_HEAD, calpha(C_WHITE, a));
    if (desc && desc[0]) {
        float dx = r.x + 22 + tw(mono, title, FONT_HEAD) + 20;
        text(mono, desc, dx, r.y + 14, FONT_MID, calpha(C_MUTED, a));
    }
}

// ─── layer drawings ───────────────────────────────────────────────────────────
static Rectangle buf_tile(Rectangle body, int b) {
    int col = b % 2;
    int row = b / 2;
    float gw = body.width * 0.5f;
    float gh = body.height * 0.5f;
    return (Rectangle){body.x + col * gw + 8, body.y + row * gh + 8, gw - 16, gh - 16};
}

static const char *CHAN_NAME[3] = {"OBS", "REWARDS", "TERMINALS"};
// Pale / Peak / deep — three tensors side by side, same hue family.
static const Color CHAN_COL[3] = {C_CYAN_HI, C_CYAN, C_CYAN_LO};

typedef struct {
    float gx[3], gy[3];
    float cw, ch;
    float grid_w, grid_h;
} ChanGrid;

// 3 side-by-side channel grids, `rows` x `cols`, with a left/top label gutter.
static ChanGrid make_chan_grid(Rectangle r, int rows, int cols, int t_on_y) {
    (void)t_on_y;
    ChanGrid g = {0};
    float pad = 18, top = 46, bot = 16;
    float x0 = r.x + pad, y0 = r.y + top;
    float W = r.width - pad * 2, H = r.height - top - bot;
    float col_w = W / 3.0f;
    float left = 42, head = 42;
    g.cw = (col_w - left - 8) / cols;
    g.ch = (H - head) / rows;
    g.grid_w = g.cw * cols;
    g.grid_h = g.ch * rows;
    for (int c = 0; c < 3; c++) {
        g.gx[c] = x0 + c * col_w + left;
        g.gy[c] = y0 + head;
    }
    return g;
}

static Rectangle chan_cell(ChanGrid g, int c, int row, int col) {
    return (Rectangle){g.gx[c] + col * g.cw + 3, g.gy[c] + row * g.ch + 3,
                       g.cw - 6, g.ch - 6};
}

static void draw_chan_headers(Font mono, Rectangle r, ChanGrid g, int cols,
                              int t_on_x, float a) {
    float y0 = r.y + 46;
    for (int c = 0; c < 3; c++) {
        text_c(mono, CHAN_NAME[c], g.gx[c] + g.grid_w * 0.5f, y0, FONT_MID, calpha(C_WHITE, a));
        if (t_on_x && cols > 0) {
            float x0c = g.gx[c];
            float x1c = g.gx[c] + g.grid_w;
            float ly = g.gy[c] - 16;
            text(mono, "t0", x0c, ly, FONT_SMALL, calpha(C_MUTED, a));
            text_c(mono, "...", 0.5f * (x0c + x1c), ly, FONT_SMALL, calpha(C_MUTED, a));
            text_r(mono, "T", x1c, ly, FONT_SMALL, calpha(C_MUTED, a));
        }
    }
}

static void draw_mem(Viz *v, Rectangle r, float a) {
    panel(r, C_GREEN, a, "2  ROLLOUTS", "Split into async buffers", v->mono);

    // rows = time, cols = buffers (chunk of envs)
    ChanGrid g = make_chan_grid(r, HORIZON, N_BUF, 1);
    draw_chan_headers(v->mono, r, g, N_BUF, 0, a);
    for (int c = 0; c < 3; c++) {
        for (int t = 0; t < HORIZON; t++) {
            if (c == 0 && (t == 0 || t == HORIZON - 1)) {
                const char *lab = (t == 0) ? "t0" : "T";
                text(v->mono, lab, g.gx[c] - 36, g.gy[c] + t * g.ch + g.ch * 0.5f - 9,
                     FONT_MID, calpha(C_MUTED, a));
            }
            if (c == 0 && t == HORIZON / 2) {
                float mx = g.gx[c] - 22;
                float my = g.gy[c] + t * g.ch + g.ch * 0.5f;
                Color dc = calpha(C_MUTED, a);
                DrawCircleV((Vector2){mx, my - 8}, 1.7f, dc);
                DrawCircleV((Vector2){mx, my}, 1.7f, dc);
                DrawCircleV((Vector2){mx, my + 8}, 1.7f, dc);
            }
            for (int b = 0; b < N_BUF; b++) {
                if (t == 0) {
                    char blab[8];
                    snprintf(blab, sizeof blab, "%d", b);
                    text_c(v->mono, blab, g.gx[c] + (b + 0.5f) * g.cw, g.gy[c] - 18,
                           FONT_MID, calpha(C_WHITE, a));
                }
                float f = v->appear[b][t];
                Rectangle cell = chan_cell(g, c, t, b);
                rounded(cell, 0.10f, calpha(CHAN_COL[c], a * 0.035f));
                stroke(cell, 0.10f, 1.0f, calpha(CHAN_COL[c], a * 0.18f));
                if (f > 0.02f) {
                    rounded((Rectangle){cell.x + 2, cell.y + 2, cell.width - 4, cell.height - 4},
                            0.10f, calpha(CHAN_COL[c], a * (0.28f + 0.70f * f)));
                }
            }
        }
    }
    if (v->xfer > 0) {
        text_c(v->mono, "TRANSPOSE -> TRAIN", r.x + r.width * 0.5f,
               r.y + r.height - 20, FONT_MID, calpha(C_ACTIVE, a));
    }
}

static void draw_roll(Viz *v, Rectangle r, float a) {
    panel(r, C_GREEN, a, "1  ENVIRONMENTS", "Multithreaded", v->mono);

    Rectangle body = {r.x + 14, r.y + 48, r.width - 28, r.height - 62};
    int half = N_ENV / N_TH;
    int part_rows = N_ENV_ROWS / N_TH;

    for (int b = 0; b < N_BUF; b++) {
        BufSim *sim = &v->bufs[b];
        Rectangle tile = buf_tile(body, b);
        int live = (sim->t >= 0);
        int stepping = live && sim->stage == ST_STEP;
        int cur[N_TH];
        for (int th = 0; th < N_TH; th++) cur[th] = omp_cursor(sim, th);
        // Tile chrome is shared; stage lives on the chip + env cells.
        rounded(tile, 0.03f, calpha(C_CYAN, 0.055f * a));
        stroke(tile, 0.03f, 1.4f, calpha(C_CYAN, 0.38f * a));

        char blab[24];
        if (sim->full) snprintf(blab, sizeof blab, "Buffer %d  FULL", b);
        else snprintf(blab, sizeof blab, "Buffer %d  t=%d/T", b, sim->ht);
        text(v->mono, blab, tile.x + 10, tile.y + 6, FONT_MID, calpha(C_WHITE, a));

        const char *stname = sim->full ? "SYNC" : (live ? ST_NAME[sim->stage] : "...");
        int hot = sim->full || (live && sim->stage == ST_STEP);
        Color stc = hot ? C_ACTIVE : C_CYAN;
        float chip_w = tw(v->mono, stname, FONT_MID) + 16;
        if (chip_w < 56) chip_w = 56;
        Rectangle chip = {tile.x + tile.width - 10 - chip_w, tile.y + 5, chip_w, 24};
        rounded(chip, 0.3f, calpha(stc, a * (hot ? 0.55f : (live ? 0.14f : 0.06f))));
        text_c(v->mono, stname, chip.x + chip.width * 0.5f, chip.y + 1, FONT_MID,
               calpha(hot || live ? C_WHITE : C_MUTED, a));

        float rail = 100.0f;
        float gx = tile.x + 10 + rail;
        float gy = tile.y + 32;
        float gw = tile.width - 24 - rail;
        float gh = tile.height - 42;
        float cw = gw / N_ENV_COLS;
        float ch = gh / N_ENV_ROWS;
        float split_y = gy + part_rows * ch;
        text(v->mono, "thread 0", tile.x + 8, gy + part_rows * ch * 0.5f - FONT_SMALL * 0.5f,
             FONT_SMALL, calpha(C_WHITE, a));
        text(v->mono, "thread 1", tile.x + 8, split_y + part_rows * ch * 0.5f - FONT_SMALL * 0.5f,
             FONT_SMALL, calpha(C_WHITE, a));

        for (int i = 0; i < N_ENV; i++) {
            int col = i % N_ENV_COLS;
            int row = i / N_ENV_COLS;
            int th_owner = (i < half) ? 0 : 1;
            int active = 0;
            for (int th = 0; th < N_TH; th++) if (cur[th] == i) active = 1;
            int done = stepping && sim->ht < HORIZON && v->written[b][sim->ht][i] && !active;
            Rectangle cell = {gx + col * cw + 3, gy + row * ch + 3, cw - 6, ch - 6};
            // Cursor = live; done = Peak cyan; remaining/idle share dim chrome.
            float fill = active ? 1.00f : (done ? 0.48f : 0.045f);
            Color cc = active ? C_ACTIVE : C_CYAN;
            rounded(cell, 0.14f, calpha(cc, a * fill));
            stroke(cell, 0.14f, active ? 2.2f : 1.0f,
                   calpha(active ? C_WHITE : C_CYAN, a * (active ? 1.0f : 0.16f)));
            char elab[8];
            snprintf(elab, sizeof elab, "e%d", i);
            Color tc = active ? C_WHITE : calpha(done ? C_WHITE : C_MUTED, a * (done ? 0.90f : 0.32f));
            text_c(v->mono, elab, cell.x + cell.width * 0.5f,
                   cell.y + cell.height * 0.5f - 8, FONT_SMALL, tc);
            (void)th_owner;
        }
    }
}

static Rectangle train_band(ChanGrid g, int c, int b, int t) {
    int r0 = b * ROWS_PER_BUF;
    return (Rectangle){
        g.gx[c] + t * g.cw + 2,
        g.gy[c] + r0 * g.ch + 2,
        g.cw - 4,
        ROWS_PER_BUF * g.ch - 4
    };
}

static void draw_trn(Viz *v, Rectangle r, Rectangle mem, float a) {
    panel(r, C_GREEN, a, "3  TRAINING",
          v->xfer > 0 ? "TRANSPOSE" : "Async offset by 1 epoch", v->mono);

    ChanGrid g = make_chan_grid(r, N_AGENTS, HORIZON, 0);
    draw_chan_headers(v->mono, r, g, HORIZON, 1, a);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < N_AGENTS; i++) {
            int b = i / ROWS_PER_BUF;
            int in_mb = v->bank_valid && i >= v->lr_mb && i < v->lr_mb + MB_ROWS;
            for (int t = 0; t < HORIZON; t++) {
                float f = 0;
                if (v->bank_valid) {
                    int n = 0;
                    for (int e = 0; e < N_ENV; e++) n += v->bank[b][t][e];
                    f = n / (float)N_ENV;
                }
                Rectangle cell = chan_cell(g, c, i, t);
                rounded(cell, 0.08f, calpha(CHAN_COL[c], a * (in_mb ? 0.22f : 0.03f)));
                if (f > 0.001f) {
                    float fa = in_mb ? (0.58f + 0.40f * f) : (0.16f + 0.22f * f);
                    rounded((Rectangle){cell.x + 1, cell.y + 1, cell.width - 2, cell.height - 2},
                            0.08f, calpha(CHAN_COL[c], a * fa));
                }
            }
        }
        if (c == 0) {
            for (int i = 0; i < N_AGENTS; i += MB_ROWS) {
                char lab[8];
                snprintf(lab, sizeof lab, "%d", i);
                text(v->mono, lab, g.gx[c] - 26, g.gy[c] + i * g.ch + 1, FONT_SMALL, calpha(C_MUTED, a));
            }
        }
        if (v->bank_valid) {
            Rectangle mb = {
                g.gx[c], g.gy[c] + v->lr_mb * g.ch,
                g.grid_w, MB_ROWS * g.ch
            };
            stroke(mb, 0.04f, 2.2f, calpha(C_WHITE, a * 0.92f));
        }
    }

    if (v->xfer > 0) {
        ChanGrid src = make_chan_grid(mem, HORIZON, N_BUF, 1);
        float u = ease(v->xfer);
        for (int c = 0; c < 3; c++) {
            for (int b = 0; b < N_BUF; b++) {
                for (int t = 0; t < HORIZON; t++) {
                    float f = chunk_fill(v, b, t);
                    if (f < 0.5f) continue;
                    Rectangle from = chan_cell(src, c, t, b);
                    Rectangle to   = train_band(g, c, b, t);
                    Rectangle mid = {
                        lerpf(from.x, to.x, u),
                        lerpf(from.y, to.y, u),
                        lerpf(from.width, to.width, u),
                        lerpf(from.height, to.height, u),
                    };
                    rounded(mid, 0.10f, calpha(CHAN_COL[c], a * (0.55f + 0.35f * u)));
                }
            }
        }
    }
}

static void draw_learn(Viz *v, Rectangle r, float a) {
    panel(r, C_CYAN, a, "4  MODEL", "Linear recurrence", v->mono);
    int live = v->bank_valid;
    float dur = lr_dur(v->lr_stage);
    float frac = live ? clampf(v->lr_t / dur, 0, 1) : 0;
    int fwd = live && v->lr_stage == LR_FWD;
    int bwd = live && v->lr_stage == LR_BWD;

    {
        const char *labs[2] = {"BACKWARD", "FORWARD"};
        int on[2] = {bwd, fwd};
        float x = r.x + r.width - 22;
        float y = r.y + 12;
        for (int i = 0; i < 2; i++) {
            float cw = tw(v->mono, labs[i], FONT_MID) + 22;
            x -= cw;
            Rectangle chip = {x, y, cw, 26};
            Color ac = on[i] ? C_ACTIVE : C_CYAN;
            rounded(chip, 0.22f, calpha(ac, a * (on[i] ? 0.55f : 0.05f)));
            stroke(chip, 0.22f, on[i] ? 1.8f : 1.0f, calpha(ac, a * (on[i] ? 1.0f : 0.22f)));
            text_c(v->mono, labs[i], chip.x + chip.width * 0.5f, chip.y + 3, FONT_MID,
                   calpha(on[i] ? C_WHITE : C_MUTED, a));
            x -= 8;
        }
    }

    float pad = 22, top = 48;
    float x0 = r.x + pad;
    float y0 = r.y + top;
    float W = r.width - pad * 2;
    float H = r.height - top - 16;
    float rec_h = H;
    float by = y0;

    // FWD: L0 W (slow) -> L0 h over T (fast) -> L1 W -> L1 h.
    // BWD reverses each layer: L1 h (T-1..0) -> L1 W -> L0 h -> L0 W.
    const int nlay = 2;
    const float w_share = 0.80f;
    const float h_share = 1.0f - w_share;
    int pass = fwd || bwd;
    int cur_L = 0, is_h = 0;
    float subf = 0;
    if (pass) {
        float x = clampf(frac, 0, 0.9999f);
        float per = 1.0f / nlay;
        int seq = (int)(x / per); // 0 = first layer in this pass
        float u = (x - seq * per) / per;
        cur_L = bwd ? (nlay - 1 - seq) : seq;
        if (bwd) {
            is_h = u < h_share;
            subf = is_h ? u / h_share : (u - h_share) / w_share;
        } else {
            is_h = u >= w_share;
            subf = is_h ? (u - w_share) / h_share : u / w_share;
        }
    }

    float lh = rec_h / nlay;
    for (int L = 0; L < nlay; L++) {
        float ly = by + L * lh;
        char llab[8];
        snprintf(llab, sizeof llab, "L%d", L);
        text(v->mono, llab, x0, ly + 4, FONT_MID, calpha(C_WHITE, a));

        int seq_L = bwd ? (nlay - 1 - L) : L; // order this layer appears in the pass
        int seq_cur = bwd ? (nlay - 1 - cur_L) : cur_L;
        float w_fill = 0, h_fill = 0;
        if (pass) {
            if (seq_L < seq_cur) {
                w_fill = 1;
                h_fill = 1;
            } else if (seq_L == seq_cur) {
                if (bwd) {
                    h_fill = is_h ? subf : 1.0f;
                    w_fill = is_h ? 0 : subf;
                } else {
                    w_fill = is_h ? 1.0f : subf;
                    h_fill = is_h ? subf : 0;
                }
            }
        }

        float mm_h = lh * 0.34f;
        if (mm_h > 72) mm_h = 72;
        if (mm_h < 28) mm_h = 28;
        float h_h = lh * 0.40f;
        if (h_h > 56) h_h = 56;
        if (h_h < 32) h_h = 32;
        Rectangle mm = {x0 + 40, ly + 8, W - 40, mm_h};
        rounded(mm, 0.10f, calpha(C_CYAN, a * 0.035f));
        stroke(mm, 0.10f, 1.1f, calpha(C_CYAN, a * 0.28f));
        if (w_fill > 0.01f) {
            float fh = (mm.height - 4) * clampf(w_fill, 0, 1);
            // FWD fills top -> bottom; BWD fills bottom -> top.
            int sx = (int)mm.x;
            int sy = bwd ? (int)(mm.y + mm.height - 2 - fh) : (int)(mm.y + 2);
            int sw = (int)mm.width, sh = (int)(fh + 2);
            BeginScissorMode(sx, sy, sw, sh);
            rounded(mm, 0.10f, calpha(C_ACTIVE, a * (0.45f + 0.50f * w_fill)));
            EndScissorMode();
        }
        const char *wlab = (pass && L == cur_L && !is_h) ? "matmul W   filling (|| T)"
                                                         : "matmul W   parallel over T";
        text_c(v->mono, wlab, mm.x + mm.width * 0.5f,
               mm.y + mm.height * 0.5f - FONT_MID * 0.5f, FONT_MID, calpha(C_WHITE, a));

        float hy = ly + 14 + mm_h;
        text(v->mono, "h", x0 + 12, hy + h_h * 0.5f - FONT_MID * 0.5f, FONT_MID, calpha(C_MUTED, a));
        float cw = (W - 176) / HORIZON;
        for (int t = 0; t < HORIZON; t++) {
            Rectangle cell = {x0 + 40 + t * cw + 3, hy, cw - 6, h_h};
            int n_lit = (int)clampf(h_fill * HORIZON, 0, HORIZON);
            int cur_t = bwd ? (HORIZON - n_lit) : (n_lit - 1);
            int on = pass && L == cur_L && is_h && n_lit > 0 && n_lit < HORIZON && t == cur_t;
            int lit_cell = n_lit >= (bwd ? (HORIZON - t) : (t + 1));
            float lit = on ? 1.00f : (lit_cell ? 0.50f : 0.04f);
            Color hc = on ? C_ACTIVE : C_CYAN;
            rounded(cell, 0.14f, calpha(hc, a * lit));
            if (on) stroke(cell, 0.14f, 1.8f, calpha(C_WHITE, a));
            if (t == 0 || t == HORIZON - 1) {
                char lab[8];
                snprintf(lab, sizeof lab, "%s", t == 0 ? "t0" : "T");
                text_c(v->mono, lab, cell.x + cell.width * 0.5f,
                       cell.y + cell.height * 0.5f - FONT_SMALL * 0.5f, FONT_SMALL,
                       calpha(on || lit_cell ? C_WHITE : C_MUTED, a));
            }
        }
        float sx = x0 + 40 + HORIZON * cw + 8;
        float sy = hy + h_h * 0.5f - FONT_SMALL;
        text(v->mono, "serial", sx, sy, FONT_SMALL, calpha(C_MUTED, a));
        text(v->mono, "activations", sx, sy + FONT_SMALL + 2, FONT_SMALL, calpha(C_MUTED, a));
    }
}

// Neighbor hop on a row of GPUs; last->first wraps underneath.
static Vector2 ring_along(const Vector2 *c, int i, float u, float wrap_y) {
    int j = (i + 1) % N_GPU;
    if (i < N_GPU - 1) return vlerp(c[i], c[j], u);
    Vector2 a = c[i], b = c[j];
    Vector2 d = {a.x, wrap_y};
    Vector2 e = {b.x, wrap_y};
    if (u < 0.22f) return vlerp(a, d, u / 0.22f);
    if (u < 0.78f) return vlerp(d, e, (u - 0.22f) / 0.56f);
    return vlerp(e, b, (u - 0.78f) / 0.22f);
}

static void draw_gpu(Viz *v, Rectangle r, float a) {
    panel(r, C_CYAN, a, "5  MULTI-GPU", "Near linear scaling", v->mono);
    int live = v->bank_valid;
    int ar = live && v->lr_stage == LR_AR;
    int upd = live && v->lr_stage == LR_UPD;
    float dur = lr_dur(v->lr_stage);
    float frac = live ? clampf(v->lr_t / dur, 0, 1) : 0;

    {
        const char *ulab = "UPDATE";
        float uw = tw(v->mono, ulab, FONT_MID) + 22;
        Rectangle ub = {r.x + r.width - 26 - uw, r.y + 12, uw, 26};
        Color uc = upd ? C_ACTIVE : C_CYAN;
        rounded(ub, 0.22f, calpha(uc, a * (upd ? 0.55f : 0.05f)));
        stroke(ub, 0.22f, upd ? 1.8f : 1.0f, calpha(uc, a * (upd ? 1.0f : 0.22f)));
        text_c(v->mono, ulab, ub.x + ub.width * 0.5f, ub.y + 3, FONT_MID,
               calpha(upd ? C_WHITE : C_MUTED, a));
    }

    float pad = 20, top = 46;
    float x0 = r.x + pad;
    float y0 = r.y + top;
    float W = r.width - pad * 2;
    float H = r.height - top - 16;
    float wrap_h = 40;
    float gy = y0;
    float gh = H - wrap_h;
    float gw = W / N_GPU;
    float wrap_y = gy + gh + wrap_h - 10;
    Vector2 ctr[N_GPU];
    Rectangle box[N_GPU];
    for (int i = 0; i < N_GPU; i++) {
        box[i] = (Rectangle){x0 + i * gw + 6, gy, gw - 12, gh};
        ctr[i] = (Vector2){box[i].x + box[i].width * 0.5f,
                           box[i].y + box[i].height * 0.5f};
    }

    Color ringc = calpha(ar ? C_ACTIVE : C_CYAN, a * (ar ? 0.85f : 0.16f));
    for (int i = 0; i < N_GPU - 1; i++)
        DrawLineEx(ctr[i], ctr[i + 1], RING_THICK, ringc);
    DrawLineEx(ctr[N_GPU - 1], (Vector2){ctr[N_GPU - 1].x, wrap_y}, RING_THICK, ringc);
    DrawLineEx((Vector2){ctr[N_GPU - 1].x, wrap_y},
               (Vector2){ctr[0].x, wrap_y}, RING_THICK, ringc);
    DrawLineEx((Vector2){ctr[0].x, wrap_y}, ctr[0], RING_THICK, ringc);

    for (int i = 0; i < N_GPU; i++) {
        float pulse = 0.12f;
        if (ar) pulse = 0.50f + 0.50f * (0.5f + 0.5f * sinf(v->t * 8.0f + i));
        else if (upd) pulse = 0.85f;
        Color gc = (ar || upd) ? C_ACTIVE : C_CYAN;
        rounded(box[i], 0.14f, calpha(gc, a * pulse * 0.55f));
        stroke(box[i], 0.14f, ar || upd ? 1.8f : 1.0f, calpha(gc, a * (ar || upd ? 0.85f : 0.32f)));
        char lab[8];
        snprintf(lab, sizeof lab, "G%d", i);
        text_c(v->mono, lab, ctr[i].x, ctr[i].y - FONT_HEAD * 0.5f, FONT_HEAD, calpha(C_WHITE, a));
    }

    if (ar) {
        int npass = N_GPU - 1;
        float dist = frac * (2.0f * npass);
        int hops = (int)dist;
        float u = dist - hops;
        Color pc = C_ACTIVE;
        StarVertex verts[N_GPU];
        for (int i = 0; i < N_GPU; i++) {
            int at = (i + hops) % N_GPU;
            Vector2 p = ring_along(ctr, at, u, wrap_y);
            verts[i] = star_vert(p, calpha(pc, a), MOTE_SIZE, 0.17f * (float)(i + 1));
        }
        draw_star_points(&v->star, verts, N_GPU);
    }
}

static void spawn(Viz *v, Vector2 a, Vector2 b, Color c, float speed, float rad) {
    for (int i = 0; i < MAX_PK; i++) {
        if (v->pk[i].alive) continue;
        v->pk[i] = (Packet){a, b, 0, speed, c, rad, 0.13f * (float)(i + 3), 1};
        return;
    }
}

static Vector2 rect_bottom(Rectangle r, float u) {
    return (Vector2){r.x + r.width * u, r.y + r.height - 6};
}
static Vector2 rect_top(Rectangle r, float u) {
    return (Vector2){r.x + r.width * u, r.y + 8};
}
static Vector2 rect_right(Rectangle r, float v) {
    return (Vector2){r.x + r.width - 8, r.y + r.height * v};
}
static Vector2 rect_left(Rectangle r, float v) {
    return (Vector2){r.x + 8, r.y + r.height * v};
}

static void maybe_spawn(Viz *v, Layout L, unsigned mask) {
    (void)mask;
    Rectangle body = {L.env.x + 14, L.env.y + 48, L.env.width - 28, L.env.height - 62};
    ChanGrid g = make_chan_grid(L.mem, HORIZON, N_BUF, 1);
    for (int b = 0; b < N_BUF; b++) {
        if (!v->bufs[b].emit) continue;
        int ht = v->bufs[b].emit_ht;
        v->bufs[b].emit = 0;
        if (ht < 0 || ht >= HORIZON) continue;
        Rectangle tile = buf_tile(body, b);
        Vector2 from = {tile.x + tile.width * 0.5f, tile.y + tile.height * 0.5f};
        for (int ch = 0; ch < 3; ch++) {
            Rectangle cell = chan_cell(g, ch, ht, b);
            Vector2 to = {cell.x + cell.width * 0.5f, cell.y + cell.height * 0.5f};
            Vector2 src = {from.x + (ch - 1) * 14.0f, from.y};
            spawn(v, src, to, C_ACTIVE, 1.7f, MOTE_SIZE);
        }
    }
    (void)rect_bottom;
    (void)rect_right;
    (void)rect_left;
    (void)rect_top;
}

static void update_packets(Viz *v) {
    for (int i = 0; i < MAX_PK; i++) {
        if (!v->pk[i].alive) continue;
        v->pk[i].t += v->dt * v->pk[i].speed;
        if (v->pk[i].t >= 1.0f) v->pk[i].alive = 0;
    }
}

static void draw_packets(Viz *v, float a) {
    StarVertex verts[MAX_PK];
    int n = 0;
    for (int i = 0; i < MAX_PK; i++) {
        if (!v->pk[i].alive) continue;
        float t = ease(v->pk[i].t);
        Vector2 p = vlerp(v->pk[i].a, v->pk[i].b, t);
        float fade = 1.0f;
        if (v->pk[i].t < 0.08f) fade = v->pk[i].t / 0.08f;
        if (v->pk[i].t > 0.85f) fade = (1.0f - v->pk[i].t) / 0.15f;
        Color c = calpha(v->pk[i].c, a * fade);
        verts[n++] = star_vert(p, c, v->pk[i].r, v->pk[i].seed);
    }
    draw_star_points(&v->star, verts, n);
}

// ─── chrome ───────────────────────────────────────────────────────────────────
static void draw_header(Viz *v, float a) {
    const char *title = "PufferLib 5.0";
    const char *sub = "Five Levels of Parallelism";
    const char *tag = "Peak 63M steps/second/RTX5090";
    const char *levels = "1 Environments    2 Rollouts    3 Training    4 Model    5 Multi-GPU";
    float title_gap = 6.0f;
    float block_h = FONT_TITLE + title_gap + FONT_MID;
    float ty = (HEADER_H - block_h) * 0.5f;
    float tx = MARGIN_X + LOGO_S + 14.0f;
    if (v->logo.id) {
        Rectangle src = {0, 0, (float)v->logo.width * 0.5f, (float)v->logo.height};
        Rectangle dst = {MARGIN_X, (HEADER_H - LOGO_S) * 0.5f, LOGO_S, LOGO_S};
        DrawTexturePro(v->logo, src, dst, (Vector2){0, 0}, 0, calpha(C_WHITE, a));
    }
    text(v->ui, title, tx, ty, FONT_TITLE, calpha(C_WHITE, a));
    text(v->ui, sub, tx, ty + FONT_TITLE + title_gap, FONT_MID, calpha(C_WHITE, a));
    text_r(v->mono, tag, SCREEN_W - MARGIN_X, (HEADER_H - FONT_HEAD) * 0.5f, FONT_HEAD, calpha(C_CYAN, a));
    float left = tx + fmaxf(tw(v->ui, title, FONT_TITLE), tw(v->ui, sub, FONT_MID)) + 32;
    float right = SCREEN_W - MARGIN_X - tw(v->mono, tag, FONT_HEAD) - 32;
    text_c(v->ui, levels, 0.5f * (left + right), (HEADER_H - FONT_HEAD) * 0.5f, FONT_HEAD, calpha(C_WHITE, a));
}

static void draw_footer(Viz *v, float a) {
    DrawLine(0, SCREEN_H - MARGIN_BOT + 8, SCREEN_W, SCREEN_H - MARGIN_BOT + 8, calpha(C_CYAN, 0.18f * a));
    if (!v->hide_hud) {
        text_r(v->mono, "space pause   s shot   r restart",
               SCREEN_W - 40, SCREEN_H - 26, FONT_SMALL, calpha(C_MUTED, 0.7f * a));
    }
}

static void draw_stars(Viz *v, float a) {
    for (int i = 0; i < v->stars_n; i++) {
        float tw = 0.55f + 0.45f * sinf(v->t * (0.7f + (i % 9) * 0.11f) + i);
        DrawPixelV(v->stars[i], calpha(C_WHITE, v->star_br[i] * tw * 0.45f * a));
    }
}

static void paint(Viz *v, float a) {
    Layout L = make_layout();
    if (!v->hide_hud) {
        DrawRectangle(0, 0, SCREEN_W, SCREEN_H, calpha(C_BG, a));
        draw_stars(v, a);
    }
    draw_header(v, a);
    draw_mem(v, L.mem, a);
    draw_roll(v, L.env, a);
    draw_trn(v, L.trn, L.mem, a);
    draw_learn(v, L.learn, a);
    draw_gpu(v, L.gpu, a);
    draw_packets(v, a);
    draw_footer(v, a);
}

static void draw_frame(Viz *v, Layout L) {
    (void)L;
    BeginDrawing();
    ClearBackground(C_BG);
    draw_stars(v, 1.0f);
    paint(v, 1.0f);
    EndDrawing();
}

static void loop_reset(Viz *v);
static void update_viz(Viz *v, Layout L, float dt);

Arch *arch_create(Font ui, Font mono, Texture2D logo, Shader star) {
    Arch *v = (Arch *)calloc(1, sizeof(Arch));
    if (!v) return NULL;
    v->ui = ui;
    v->mono = mono;
    v->logo = logo;
    v->star = star;
    v->hide_hud = 1;
    v->stars_n = 420;
    for (int i = 0; i < v->stars_n; i++) {
        v->stars[i] = (Vector2){(float)(rand() % SCREEN_W), (float)(rand() % SCREEN_H)};
        v->star_br[i] = 0.15f + (rand() % 80) / 100.0f;
    }
    reset_sim(v);
    return v;
}

void arch_destroy(Arch *a) { free(a); }

void arch_reset(Arch *a) {
    if (a) loop_reset(a);
}

void arch_update(Arch *a, float dt) {
    if (!a) return;
    Layout L = make_layout();
    update_viz(a, L, dt);
}

void arch_draw(Arch *a, float alpha) {
    if (!a || alpha <= 0.001f) return;
    paint(a, alpha);
}

void arch_hide_hud(Arch *a, int hide) {
    if (a) a->hide_hud = hide;
}

float arch_loop_sec(void) { return LOOP_SEC; }

static void loop_reset(Viz *v) {
    memset(v->pk, 0, sizeof v->pk);
    reset_sim(v);
    v->t = 0;
}

static void update_viz(Viz *v, Layout L, float dt) {
    if (!v->paused) {
        v->dt = dt;
        v->t += dt;
    } else {
        v->dt = 0;
    }
    step_sim(v, v->dt);
    maybe_spawn(v, L, 0);
    update_packets(v);
}

static void seek_viz(Viz *v, Layout L, float t) {
    loop_reset(v);
    v->paused = 0;
    const float dt = 1.0f / 60.0f;
    int steps = (int)(t / dt);
    if (steps < 1) steps = 1;
    if (steps > 60 * 90) steps = 60 * 90;
    for (int i = 0; i < steps; i++) update_viz(v, L, dt);
}

static void export_png(const char *path) {
    Image img = LoadImageFromScreen();
    if (!ExportImage(img, path)) fprintf(stderr, "failed to write %s\n", path);
    else printf("wrote %s\n", path);
    UnloadImage(img);
}

static void ensure_shots_dir(void) {
    mkdir("trailer", 0755);
    mkdir("trailer/shots", 0755);
}

// ─── ffmpeg recorder (same pattern as trailer.c) ──────────────────────────────
typedef struct { int pid; int pipefd[2]; } VideoRecorder;

static bool open_video(VideoRecorder *rec, const char *filename, int w, int h) {
    if (pipe(rec->pipefd) < 0) return false;
    rec->pid = fork();
    if (rec->pid < 0) return false;
    if (rec->pid == 0) {
        close(rec->pipefd[1]);
        dup2(rec->pipefd[0], STDIN_FILENO);
        close(rec->pipefd[0]);
        char sz[32]; snprintf(sz, sizeof sz, "%dx%d", w, h);
        char fps[8]; snprintf(fps, sizeof fps, "%d", RECORD_FPS);
        execlp("ffmpeg", "ffmpeg", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", sz, "-r", fps, "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "medium", "-crf", "23",
               "-loglevel", "error",
               filename, (char *)NULL);
        _exit(1);
    }
    close(rec->pipefd[0]);
    return true;
}
static void write_frame(VideoRecorder *rec, int w, int h) {
    rlDrawRenderBatchActive();
    unsigned char *data = rlReadScreenPixels(w, h);
    write(rec->pipefd[1], data, (size_t)w * h * 4);
    RL_FREE(data);
}
static void close_video(VideoRecorder *rec) {
    close(rec->pipefd[1]);
    waitpid(rec->pid, NULL, 0);
}

static void usage(void) {
    fprintf(stderr,
        "usage: ./trailer/architecture [flags]\n"
        "  --shot [FILE]     export a still (default trailer/shots/loop.png)\n"
        "  --record [FILE]   record one %gs loop (default trailer/architecture.mp4)\n"
        "  --time SEC        seek before shot/play\n"
        "  --headless        hidden window\n"
        "  --once            play one loop and exit\n",
        LOOP_SEC);
}

#ifndef ARCHITECTURE_LIB
int main(int argc, char **argv) {
    const char *shot = NULL;
    int record = 0, once = 0, headless = 0;
    const char *record_path = "trailer/architecture.mp4";
    float seek = -1.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--shot") == 0) {
            shot = "trailer/shots/loop.png";
            if (i + 1 < argc && argv[i + 1][0] != '-') shot = argv[++i];
        } else if (strcmp(argv[i], "--record") == 0) {
            record = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') record_path = argv[++i];
        } else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc) {
            seek = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--headless") == 0) {
            headless = 1;
        } else if (strcmp(argv[i], "--once") == 0) {
            once = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        } else {
            fprintf(stderr, "unknown flag: %s\n", argv[i]);
            usage();
            return 1;
        }
    }

    if (shot) headless = 1;
    if (record) once = 1;

    unsigned flags = FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT;
    if (headless) flags |= FLAG_WINDOW_HIDDEN;
    SetConfigFlags(flags);
    InitWindow(SCREEN_W, SCREEN_H, "PufferLib 5.0's Five Levels of Parallelism");
    SetTargetFPS(60);
#ifndef GRAPHICS_API_OPENGL_ES2
    glEnable(GL_PROGRAM_POINT_SIZE);
#endif

    Viz v = {0};
    int cps[224];
    for (int i = 0; i < 224; i++) cps[i] = 32 + i;
    v.ui = LoadFontEx("resources/shared/Montserrat-Regular.ttf", FONT_ATLAS_UI, cps, 224);
    v.mono = LoadFontEx("resources/shared/JetBrainsMono-Medium.ttf", FONT_ATLAS_MONO, cps, 224);
    SetTextureFilter(v.ui.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(v.mono.texture, TEXTURE_FILTER_BILINEAR);
    if (FileExists("resources/shared/puffers_128.png")) {
        v.logo = LoadTexture("resources/shared/puffers_128.png");
    }
    v.star = LoadShader(
        TextFormat("resources/trailer/star_%i.vs", GLSL_VERSION),
        TextFormat("resources/constellation/point_particle_%i.fs", GLSL_VERSION)
    );
    v.stars_n = 420;
    for (int i = 0; i < v.stars_n; i++) {
        v.stars[i] = (Vector2){(float)(rand() % SCREEN_W), (float)(rand() % SCREEN_H)};
        v.star_br[i] = 0.15f + (rand() % 80) / 100.0f;
    }
    v.hide_hud = (shot || record);
    reset_sim(&v);

    Layout L = make_layout();

    if (shot) {
        ensure_shots_dir();
        float t = (seek >= 0) ? seek : 3.2f;
        seek_viz(&v, L, t);
        v.paused = 1;
        printf("architecture t=%.2f xfer=%.3f\n", t, v.xfer);
        for (int f = 0; f < 3; f++) draw_frame(&v, L);
        export_png(shot);
        goto cleanup;
    }

    if (seek >= 0) seek_viz(&v, L, seek);

    VideoRecorder rec = {0};
    int recording = 0;
    if (record) {
        recording = open_video(&rec, record_path, SCREEN_W, SCREEN_H);
        if (!recording) fprintf(stderr, "warning: ffmpeg record failed\n");
        else printf("recording %s\n", record_path);
    }

    int frame = 0;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_Q) || IsKeyPressed(KEY_ESCAPE)) break;
        if (IsKeyPressed(KEY_SPACE)) v.paused = !v.paused;
        if (IsKeyPressed(KEY_R)) loop_reset(&v);
        if (IsKeyPressed(KEY_S)) {
            ensure_shots_dir();
            draw_frame(&v, L);
            export_png("trailer/shots/loop.png");
        }

        float dt = recording ? (1.0f / 60.0f) : GetFrameTime();
        if (dt > 0.10f) dt = 0.10f;
        if (dt < 0) dt = 0;
        update_viz(&v, L, dt);
        draw_frame(&v, L);

        if (recording && (frame++ % 2 == 0))
            write_frame(&rec, SCREEN_W, SCREEN_H);

        if (once && v.t >= LOOP_SEC) break;
        if (!once && v.t >= LOOP_SEC) loop_reset(&v);
    }

    if (recording) {
        close_video(&rec);
        printf("wrote %s\n", record_path);
    }

cleanup:
    if (v.logo.id) UnloadTexture(v.logo);
    if (v.star.id) UnloadShader(v.star);
    UnloadFont(v.ui);
    UnloadFont(v.mono);
    CloseWindow();
    return 0;
}
#endif // ARCHITECTURE_LIB
