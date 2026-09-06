#include "trailer/stars.h"

#include <math.h>
#include <stdlib.h>

#include "raymath.h"
#include "rlgl.h"

#if defined(__APPLE__)
    #define GL_SILENCE_DEPRECATION
    #include <OpenGL/gl3.h>
#else
    #include "glad.h"
#endif

#define SCREEN_W 1920
#define SCREEN_H 1080

typedef struct {
    float x, y, size_scale;
    float r, g, b, a;
} StarVertex;

#define NUM_BG_STARS      32768
#define NUM_VISIBLE_STARS 16384
#define BG_DENSE_R        1200.0f
#define BG_FADE_R         2400.0f
#define BG_SPARSE_FRAC    0.02f
#define BG_MAX_R          4000.0f
#define SPIRAL_CX         (SCREEN_W * 0.5f)
#define SPIRAL_CY         (SCREEN_H * 0.5f)
#define SPIRAL_GM         16000000.0f
#define SPIRAL_OMEGA      0.1f
#define SPIRAL_OMEGA_R    1000.0f
#define NOVA_GM           5000000.0f

typedef struct { float r, angle, vr, brightness, size_scale; } BgStar;

static BgStar bg_stars[NUM_BG_STARS];
static StarVertex bg_verts[NUM_BG_STARS];
static int bg_draw_count;
static const Color C_WHITE = {220, 230, 255, 255};
static const Color C_CYAN  = {0, 240, 230, 255};

#define MAX_HEROES 8
typedef struct { int idx; float orig_size, orig_br; } Hero;
static Hero heroes[MAX_HEROES];
static int n_heroes;

static int is_hero(int i) {
    for (int h = 0; h < n_heroes; h++) if (heroes[h].idx == i) return 1;
    return 0;
}

static Vector2 star_xy(int i) {
    return (Vector2){
        SPIRAL_CX + bg_stars[i].r * cosf(bg_stars[i].angle),
        SPIRAL_CY + bg_stars[i].r * sinf(bg_stars[i].angle),
    };
}

static void draw_gl(StarVertex *verts, int n, Shader *sh) {
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

static void fill_verts(float alpha, int nova) {
    int n = nova ? NUM_BG_STARS : NUM_VISIBLE_STARS;
    for (int i = 0; i < n; i++) {
        int hero = is_hero(i);
        float br = bg_stars[i].brightness * alpha;
        if (hero) br *= 1.35f;
        Color col = hero ? C_CYAN : C_WHITE;
        bg_verts[i] = (StarVertex){
            SPIRAL_CX + bg_stars[i].r * cosf(bg_stars[i].angle),
            SPIRAL_CY + bg_stars[i].r * sinf(bg_stars[i].angle),
            bg_stars[i].size_scale,
            col.r / 255.0f * br,
            col.g / 255.0f * br,
            col.b / 255.0f * br,
            (float)i,
        };
    }
    bg_draw_count = n;
}

void stars_clear_heroes(void) {
    for (int h = 0; h < n_heroes; h++) {
        int i = heroes[h].idx;
        if (i >= 0 && i < NUM_BG_STARS) {
            bg_stars[i].size_scale = heroes[h].orig_size;
            bg_stars[i].brightness = heroes[h].orig_br;
        }
    }
    n_heroes = 0;
}

Vector2 stars_make_hero(float x, float y) {
    int best = -1;
    float best_d = 1e12f;
    for (int i = 0; i < NUM_VISIBLE_STARS; i++) {
        if (is_hero(i)) continue;
        Vector2 p = star_xy(i);
        if (p.x < 16 || p.x > SCREEN_W - 16 || p.y < 48 || p.y > SCREEN_H - 16) continue;
        float dx = p.x - x, dy = p.y - y;
        float d = dx * dx + dy * dy;
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    if (best < 0) return (Vector2){x, y};
    if (n_heroes >= MAX_HEROES) return star_xy(best);
    heroes[n_heroes].idx = best;
    heroes[n_heroes].orig_size = bg_stars[best].size_scale;
    heroes[n_heroes].orig_br = bg_stars[best].brightness;
    n_heroes++;
    // Same star, much bigger — shader twinkle scales with size_scale.
    bg_stars[best].size_scale = 5.6f;
    bg_stars[best].brightness = 1.55f;
    return star_xy(best);
}

Vector2 stars_hero_pos(int slot) {
    if (slot < 0 || slot >= n_heroes) return (Vector2){0, 0};
    return star_xy(heroes[slot].idx);
}

void stars_init(void) {
    stars_reset();
}

void stars_reset(void) {
    n_heroes = 0;
    float a_dense = BG_DENSE_R * BG_DENSE_R;
    float a_fade  = BG_FADE_R * BG_FADE_R - a_dense;
    float a_outer = BG_MAX_R * BG_MAX_R - BG_FADE_R * BG_FADE_R;
    float w_dense = 1.0f;
    float w_fade  = 0.5f * (1.0f + BG_SPARSE_FRAC);
    float w_outer = BG_SPARSE_FRAC;
    float total   = a_dense * w_dense + a_fade * w_fade + a_outer * w_outer;
    int n_dense   = (int)(NUM_VISIBLE_STARS * a_dense * w_dense / total);
    int n_fade    = (int)(NUM_VISIBLE_STARS * a_fade * w_fade / total);

    for (int i = 0; i < NUM_BG_STARS; i++) {
        float r;
        if (i >= NUM_VISIBLE_STARS) {
            r = 0.5f + ((float)rand() / (float)RAND_MAX) * 4.5f;
        } else if (i < n_dense) {
            float rnd = (float)rand() / (float)RAND_MAX;
            r = BG_DENSE_R * sqrtf(rnd);
        } else if (i < n_dense + n_fade) {
            float rnd = (float)rand() / (float)RAND_MAX;
            r = sqrtf(BG_DENSE_R * BG_DENSE_R + rnd * (BG_FADE_R * BG_FADE_R - BG_DENSE_R * BG_DENSE_R));
        } else {
            float rnd = (float)rand() / (float)RAND_MAX;
            r = sqrtf(BG_FADE_R * BG_FADE_R + rnd * (BG_MAX_R * BG_MAX_R - BG_FADE_R * BG_FADE_R));
        }
        bg_stars[i].r = r;
        bg_stars[i].angle = ((float)rand() / (float)RAND_MAX) * 2.0f * 3.14159265f;
        bg_stars[i].vr = 0.0f;
        bg_stars[i].brightness = 0.2f + (float)(rand() % 80) / 100.0f;
        float s = (float)rand() / (float)RAND_MAX;
        bg_stars[i].size_scale = (s < 0.85f)
            ? (0.2f + s * 0.4f) * 0.7f
            : (0.8f + (s - 0.85f) * 6.0f) * 0.7f;
    }
}

void stars_drift(float dt) {
    (void)dt; // static field — 4.0 spiral leftover
}

void stars_spiral(float dt) {
    for (int i = 0; i < NUM_BG_STARS; i++) {
        float r = bg_stars[i].r;
        bg_stars[i].vr -= SPIRAL_GM / (r * r) * dt;
        bg_stars[i].r += bg_stars[i].vr * dt;
        bg_stars[i].angle += SPIRAL_OMEGA * SPIRAL_OMEGA_R / (r + SPIRAL_OMEGA_R) * dt;
        if (bg_stars[i].r < 1.0f) bg_stars[i].r = 1.0f;
    }
}

void stars_nova(float dt) {
    for (int i = 0; i < NUM_BG_STARS; i++) {
        float r = fmaxf(bg_stars[i].r, 1.0f);
        bg_stars[i].vr += NOVA_GM / (r * r) * dt;
        bg_stars[i].r += bg_stars[i].vr * dt;
    }
}

void stars_draw(Shader *sh, float alpha) {
    if (alpha <= 0.001f) return;
    fill_verts(alpha, 0);
    draw_gl(bg_verts, bg_draw_count, sh);
}

void stars_draw_nova(Shader *sh, float alpha) {
    if (alpha <= 0.001f) return;
    fill_verts(alpha, 1);
    draw_gl(bg_verts, bg_draw_count, sh);
}
