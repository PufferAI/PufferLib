// PufferLib 5.0 release trailer.
// Build:  ./build.sh trailer
// Run:    ./trailer/trailer
// Record: ./trailer/trailer --record trailer/shots/puffer5.mp4
// Clips:  trailer/clips/<name>.mp4 (ffmpeg pipe; raylib has no video API)
//
// Keys:  space pause   n next   r restart   s shot   q quit

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#if defined(__APPLE__)
    #define GL_SILENCE_DEPRECATION
    #include <OpenGL/gl3.h>
#else
    #include "glad.h"
#endif

#include "trailer/architecture.h"
#include "trailer/plot_scale.h"
#include "trailer/stars.h"

#define GLSL_VERSION 330
#define SCREEN_W 1920
#define SCREEN_H 1080
#define RECORD_FPS 30
#define XFADE 0.70f

#define C_BG     ((Color){4, 8, 20, 255})
#define C_WHITE  ((Color){241, 241, 241, 255})
#define C_MUTED  ((Color){186, 210, 210, 255})
#define C_CYAN   ((Color){0, 187, 187, 255})

#define FONT_ATLAS 128
#define FONT_CARD  64
#define FONT_SUB   32
#define FONT_MID   22

enum { SC_ARCH, SC_FASTER, SC_RESULTS, SC_END };

typedef struct {
    int kind;
    float dur;
    const char *line1;
    const char *line2;
} Beat;

static const Beat BEATS[] = {
    { SC_ARCH,     8.0f, NULL, NULL },
    { SC_FASTER,   8.2f, "5x Faster Training", NULL },
    { SC_RESULTS, 10.0f, "5 Exciting new results", NULL },
    { SC_END,      6.0f, "PufferLib 5.0", "puffer.ai" },
};
#define N_BEATS ((int)(sizeof BEATS / sizeof BEATS[0]))
#define N_NEW   5



typedef struct { int pid; int pipefd[2]; } VideoRecorder;

// One GPU texture per clip. ffmpeg decodes the mp4 (raylib has no video API).
typedef struct {
    char path[256];
    Texture2D tex;
    unsigned char *pix;
    int w, h;
    float fps;
    int pid;
    int fd;
    int decoded;
    float ss;
    float dur;
    int loop;
    int punch_black;
} GameClip;

typedef struct {
    Font ui, mono;
    Texture2D logo;
    GameClip puffer;
    Shader field;
    Shader mote;
    Shader mark;
    Arch *arch;
    GameClip train;
    GameClip breakout;
    GameClip clip[N_NEW];
    float t;
    int paused;
    int hide_hud;
    int entered[N_BEATS];
    float beat_start[N_BEATS];
    float total;
} Trailer;

typedef struct {
    const char *key;
    const char *name;
} NewResult;

static const NewResult NEWS[N_NEW] = {
    {"drone",          "Drone"},
    {"robocode",       "Robocode"},
    {"craftax",        "Craftax"},
    {"zukbest_clip",   "OSRS Inferno"},
    {"colobest_clip",  "OSRS Colosseum"},
};

static Color calpha(Color c, float a) {
    if (a < 0) a = 0;
    if (a > 1) a = 1;
    return (Color){c.r, c.g, c.b, (unsigned char)(c.a * a)};
}

static float clampf(float x, float a, float b) {
    return x < a ? a : (x > b ? b : x);
}

static void text_c(Font f, const char *s, float cx, float y, float sz, Color c) {
    DrawTextEx(f, s, (Vector2){cx - MeasureTextEx(f, s, sz, 0.4f).x * 0.5f, y}, sz, 0.4f, c);
}

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

static void ensure_shots_dir(void) {
    mkdir("trailer", 0755);
    mkdir("trailer/shots", 0755);
}

static void export_png(const char *path) {
    Image img = LoadImageFromScreen();
    if (!ExportImage(img, path)) fprintf(stderr, "failed to write %s\n", path);
    else printf("wrote %s\n", path);
    UnloadImage(img);
}

static void layout_beats(Trailer *tr) {
    float t = 0;
    for (int i = 0; i < N_BEATS; i++) {
        tr->beat_start[i] = t;
        t += BEATS[i].dur;
        if (i + 1 < N_BEATS) t -= XFADE;
    }
    tr->total = tr->beat_start[N_BEATS - 1] + BEATS[N_BEATS - 1].dur;
}

static float alpha_in_out(float t, float start, float dur) {
    float fade = XFADE;
    if (dur < 2.0f * fade) fade = dur * 0.35f;
    float a = clampf((t - start) / fade, 0, 1);
    float b = clampf((start + dur - t) / fade, 0, 1);
    return a < b ? a : b;
}

static float ease(float t) {
    t = clampf(t, 0, 1);
    return t * t * (3.0f - 2.0f * t);
}



static void draw_clip(GameClip *c, float local, Rectangle panel, float alpha);

static void draw_faster(Trailer *tr, float local, float alpha) {
    if (alpha <= 0.001f) return;
    text_c(tr->ui, "5x Faster Training", SCREEN_W * 0.5f, 22, FONT_CARD, calpha(C_WHITE, alpha));

    Rectangle left = {48, 300, 884, 420};
    Rectangle right = {960, 88, 924, 956};

    float train_end = tr->train.dur;
    float pol_a = ease(clampf((local - train_end) / 0.45f, 0, 1));
    if (pol_a < 0.99f)
        draw_clip(&tr->train, local, left, alpha * (1.0f - 0.35f * pol_a));
    if (pol_a > 0.01f)
        draw_clip(&tr->breakout, local - train_end, left, alpha * pol_a);
    DrawRectangleRounded(right, 0.02f, 8, calpha((Color){6, 18, 20, 255}, alpha * 0.90f));
    plot_scale_draw_box(tr->ui, tr->mono, &tr->mark,
        clampf(local / 7.4f, 0, 1), alpha, right);
}

// 2x3 grid. Cell 0 is title; clips fill the other five.
static void results_layout(Rectangle cells[6]) {
    const float gap = 28.0f, mx = 48.0f;
    float w = (SCREEN_W - 2.0f * mx - 2.0f * gap) / 3.0f;
    float h = w * 9.0f / 16.0f;
    float y0 = (SCREEN_H - (2.0f * h + gap)) * 0.5f;
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            cells[r * 3 + c] = (Rectangle){
                mx + c * (w + gap), y0 + r * (h + gap), w, h
            };
        }
    }
}

static int read_full(int fd, unsigned char *p, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r == 0) return 0;
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        got += (size_t)r;
    }
    return 1;
}

static int probe_video(const char *path, int *w, int *h, float *fps, float *dur) {
    char cmd[512];
    snprintf(cmd, sizeof cmd,
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=width,height,avg_frame_rate -of csv=p=0 %s",
        path);
    FILE *fp = popen(cmd, "r");
    if (!fp) return 0;
    char line[128] = {0};
    if (!fgets(line, sizeof line, fp)) { pclose(fp); return 0; }
    pclose(fp);
    int num = 30, den = 1;
    if (sscanf(line, "%d,%d,%d/%d", w, h, &num, &den) < 3) return 0;
    if (den <= 0) den = 1;
    *fps = (float)num / (float)den;
    *dur = 0;
    snprintf(cmd, sizeof cmd,
        "ffprobe -v error -show_entries format=duration -of csv=p=0 %s", path);
    fp = popen(cmd, "r");
    if (fp) {
        line[0] = 0;
        if (fgets(line, sizeof line, fp)) *dur = (float)atof(line);
        pclose(fp);
    }
    return *w > 0 && *h > 0;
}

static void clip_close(GameClip *c) {
    if (c->fd > 0) { close(c->fd); c->fd = 0; }
    if (c->pid > 0) {
        kill(c->pid, SIGKILL);
        waitpid(c->pid, NULL, 0);
        c->pid = 0;
    }
    c->decoded = 0;
    c->ss = 0;
}

static void clip_open(GameClip *c, float ss) {
    if (c->path[0] == 0 || c->w <= 0) return;
    int pipefd[2];
    if (pipe(pipefd) < 0) return;
    fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);
    fcntl(pipefd[1], F_SETFD, FD_CLOEXEC);
    c->pid = fork();
    if (c->pid < 0) {
        close(pipefd[0]); close(pipefd[1]);
        c->pid = 0;
        return;
    }
    if (c->pid == 0) {
        dup2(pipefd[1], STDOUT_FILENO);
        int n = open("/dev/null", O_RDONLY);
        if (n >= 0) { dup2(n, STDIN_FILENO); close(n); }
        for (int i = 3; i < 256; i++) close(i);
        char ssbuf[32];
        snprintf(ssbuf, sizeof ssbuf, "%.3f", ss < 0 ? 0 : ss);
        if (c->loop && ss > 0.02f)
            execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-nostdin", "-ss", ssbuf, "-stream_loop", "-1", "-i", c->path,
                   "-f", "rawvideo", "-pix_fmt", "rgba", "-an", "-",
                   (char *)NULL);
        else if (c->loop)
            execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-nostdin", "-stream_loop", "-1", "-i", c->path,
                   "-f", "rawvideo", "-pix_fmt", "rgba", "-an", "-",
                   (char *)NULL);
        else
            execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-nostdin", "-i", c->path,
                   "-f", "rawvideo", "-pix_fmt", "rgba", "-an", "-",
                   (char *)NULL);
        _exit(1);
    }
    close(pipefd[1]);
    c->fd = pipefd[0];
    c->ss = ss;
    c->decoded = 0;
}

// Key only the black field connected to the frame edge. Interior black
// (pupils, spike roots) stays opaque.
static void punch_black_bg(unsigned char *pix, int w, int h) {
    size_t n = (size_t)w * (size_t)h;
    unsigned char *mark = (unsigned char *)calloc(n, 1);
    int *stack = (int *)malloc(n * sizeof(int));
    if (!mark || !stack) { free(mark); free(stack); return; }
    int sp = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char *p = pix + i * 4;
        if ((int)p[0] + (int)p[1] + (int)p[2] < 12) mark[i] = 2; // candidate
    }
    for (int x = 0; x < w; x++) {
        int a = x, b = (h - 1) * w + x;
        if (mark[a] == 2) { mark[a] = 1; stack[sp++] = a; }
        if (mark[b] == 2) { mark[b] = 1; stack[sp++] = b; }
    }
    for (int y = 0; y < h; y++) {
        int a = y * w, b = y * w + (w - 1);
        if (mark[a] == 2) { mark[a] = 1; stack[sp++] = a; }
        if (mark[b] == 2) { mark[b] = 1; stack[sp++] = b; }
    }
    while (sp) {
        int i = stack[--sp];
        int x = i % w, y = i / w;
        int nb[4];
        int nn = 0;
        if (x > 0) nb[nn++] = i - 1;
        if (x < w - 1) nb[nn++] = i + 1;
        if (y > 0) nb[nn++] = i - w;
        if (y < h - 1) nb[nn++] = i + w;
        for (int k = 0; k < nn; k++) {
            int j = nb[k];
            if (mark[j] == 2) { mark[j] = 1; stack[sp++] = j; }
        }
    }
    for (size_t i = 0; i < n; i++) {
        unsigned char *p = pix + i * 4;
        if (mark[i] != 1) { p[3] = 255; continue; }
        int s = (int)p[0] + (int)p[1] + (int)p[2];
        if (s < 10) p[3] = 0;
        else if (s < 28) p[3] = (unsigned char)((s - 10) * 255 / 18);
        else p[3] = 255;
    }
    free(mark);
    free(stack);
}

static int clip_read(GameClip *c) {
    size_t n = (size_t)c->w * (size_t)c->h * 4u;
    if (read_full(c->fd, c->pix, n) != 1) return 0;
    if (c->punch_black)
        punch_black_bg(c->pix, c->w, c->h);
    if (c->tex.id) UpdateTexture(c->tex, c->pix);
    c->decoded++;
    return 1;
}

static void clip_sync(GameClip *c, float local) {
    if (!c->tex.id || c->path[0] == 0) return;
    if (local < 0) local = 0;
    if (c->loop && c->dur > 0.05f)
        local = fmodf(local, c->dur);
    else if (!c->loop && c->dur > 0) {
        float last_t = c->dur - 1.0f / (c->fps > 1 ? c->fps : 30);
        if (last_t < 0) last_t = 0;
        if (local > last_t) local = last_t;
    }
    int want = (int)(local * c->fps);
    if (c->pid <= 0) {
        clip_open(c, c->loop ? local : 0);
        if (c->pid > 0) clip_read(c);
    }
    int last = (int)(c->ss * c->fps + 0.5f) + c->decoded - 1;
    if (c->decoded <= 0) last = -1;
    if (c->loop && (want < last || want - last > (int)(c->fps + 0.5f))) {
        clip_close(c);
        clip_open(c, local);
        if (c->pid > 0) clip_read(c);
        last = (int)(c->ss * c->fps + 0.5f) + c->decoded - 1;
        if (c->decoded <= 0) last = -1;
    }
    while (last < want) {
        if (!clip_read(c)) {
            clip_close(c);
            if (c->loop) {
                clip_open(c, local);
                if (c->pid > 0) clip_read(c);
            }
            return;
        }
        last++;
    }
}

static void load_clip(GameClip *c, const char *key) {
    memset(c, 0, sizeof *c);
    snprintf(c->path, sizeof c->path, "trailer/clips/%s.mp4", key);
    if (!FileExists(c->path)) {
        printf("clip %s: missing %s\n", key, c->path);
        c->path[0] = 0;
        return;
    }
    int iw = 0, ih = 0;
    float fps = 30.0f, dur = 0;
    if (!probe_video(c->path, &iw, &ih, &fps, &dur)) {
        fprintf(stderr, "clip %s: ffprobe failed for %s\n", key, c->path);
        c->path[0] = 0;
        return;
    }
    int w = iw & ~1, h = ih & ~1;
    if (w < 2) w = 2;
    if (h < 2) h = 2;
    c->w = w;
    c->h = h;
    c->fps = fps > 1.0f ? fps : 30.0f;
    c->dur = dur > 0 ? dur : 0;
    c->loop = 1;
    c->pix = (unsigned char *)calloc((size_t)w * (size_t)h, 4u);
    Image img = GenImageColor(w, h, BLACK);
    c->tex = LoadTextureFromImage(img);
    UnloadImage(img);
    SetTextureFilter(c->tex, TEXTURE_FILTER_BILINEAR);
    printf("clip %s: %s %dx%d @ %.1ffps %.2fs\n",
           key, c->path, w, h, c->fps, c->dur);
}

static void unload_clip(GameClip *c) {
    clip_close(c);
    if (c->tex.id) UnloadTexture(c->tex);
    free(c->pix);
    memset(c, 0, sizeof *c);
}

static void draw_clip(GameClip *c, float local, Rectangle panel, float alpha) {
    if (!c || alpha <= 0.001f) return;
    clip_sync(c, local);
    if (c->tex.id == 0) return;
    Texture2D tex = c->tex;
    float scale = fminf(panel.width / (float)tex.width, panel.height / (float)tex.height);
    float dw = tex.width * scale, dh = tex.height * scale;
    Rectangle dst = {
        panel.x + (panel.width - dw) * 0.5f,
        panel.y + (panel.height - dh) * 0.5f,
        dw, dh
    };
    DrawRectangleRounded(
        (Rectangle){dst.x - 10, dst.y - 10, dst.width + 20, dst.height + 20},
        0.04f, 8, calpha((Color){6, 18, 20, 255}, alpha * 0.9f));
    DrawTexturePro(tex,
        (Rectangle){0, 0, (float)tex.width, (float)tex.height},
        dst, (Vector2){0, 0}, 0, calpha(C_WHITE, alpha));
}

static void draw_results(Trailer *tr, float local, float alpha) {
    if (alpha <= 0.001f) return;
    float fa = clampf(local / 0.35f, 0, 1) * alpha;
    Rectangle cells[6];
    results_layout(cells);
    text_c(tr->ui, "5 Exciting",
           cells[0].x + cells[0].width * 0.5f,
           cells[0].y + cells[0].height * 0.5f - 48, FONT_CARD, calpha(C_WHITE, fa));
    text_c(tr->ui, "new results",
           cells[0].x + cells[0].width * 0.5f,
           cells[0].y + cells[0].height * 0.5f + 8, FONT_CARD, calpha(C_MUTED, fa));
    for (int i = 0; i < N_NEW; i++) {
        Rectangle card = cells[i + 1];
        DrawRectangleRounded(card, 0.03f, 8, calpha((Color){6, 18, 20, 255}, fa * 0.90f));
        DrawRectangleRoundedLinesEx(card, 0.03f, 8, 2.2f, calpha(C_CYAN, fa * 0.70f));
        if (tr->clip[i].tex.id) {
            Rectangle inner = {card.x + 10, card.y + 10, card.width - 20, card.height - 20};
            draw_clip(&tr->clip[i], local, inner, fa);
        }
        text_c(tr->ui, NEWS[i].name, card.x + card.width * 0.5f, card.y + 14, 26,
               calpha(C_WHITE, fa));
    }
}

static void draw_end(Trailer *tr, const Beat *b, float local, float alpha) {
    if (alpha <= 0.001f) return;
    if (tr->puffer.tex.id) {
        clip_sync(&tr->puffer, local);
        // Banner is 960x480, fish on the left. Crop at the inflated mouth
        // so the old 2D logo stream never enters frame.
        float src_w = tr->puffer.w > 500 ? 500.0f : (float)tr->puffer.w;
        float src_h = (float)tr->puffer.h;
        float dh = 620.0f;
        float dw = src_w * (dh / src_h);
        Rectangle dst = {
            SCREEN_W * 0.5f - dw * 0.5f,
            150.0f,
            dw, dh
        };
        DrawTexturePro(tr->puffer.tex,
            (Rectangle){0, 0, src_w, src_h},
            dst, (Vector2){0, 0}, 0, calpha(C_WHITE, alpha));
    }
    text_c(tr->ui, b->line1 ? b->line1 : "PufferLib 5.0",
           SCREEN_W * 0.5f, 860.0f, 72, calpha(C_WHITE, alpha));
    const char *left = "Available now at  ";
    const char *right = "puffer.ai";
    float w1 = MeasureTextEx(tr->ui, left, FONT_SUB, 0.4f).x;
    float w2 = MeasureTextEx(tr->ui, right, FONT_SUB, 0.4f).x;
    float x = SCREEN_W * 0.5f - 0.5f * (w1 + w2);
    float y = 860.0f + 72.0f;
    DrawTextEx(tr->ui, left, (Vector2){x, y}, FONT_SUB, 0.4f, calpha(C_WHITE, alpha));
    DrawTextEx(tr->ui, right, (Vector2){x + w1, y}, FONT_SUB, 0.4f, calpha(C_CYAN, alpha));
}

static void enter_beat(Trailer *tr, int i) {
    tr->entered[i] = 1;
    int kind = BEATS[i].kind;
    if (kind == SC_ARCH) arch_reset(tr->arch);
}

static void seek_trailer(Trailer *tr, float t) {
    tr->t = 0;
    memset(tr->entered, 0, sizeof tr->entered);
    stars_reset();
    arch_reset(tr->arch);
    const float dt = 1.0f / 60.0f;
    int n = (int)(t / dt);
    if (n < 0) n = 0;
    for (int k = 0; k < n; k++) {
        tr->t += dt;
        stars_drift(dt);
        for (int i = 0; i < N_BEATS; i++) {
            float a = alpha_in_out(tr->t, tr->beat_start[i], BEATS[i].dur);
            if (a <= 0.001f) continue;
            if (!tr->entered[i]) enter_beat(tr, i);
            if (BEATS[i].kind == SC_ARCH) arch_update(tr->arch, dt);
        }
    }
    tr->t = t;
}

static void tick(Trailer *tr, float dt) {
    if (tr->paused) return;
    tr->t += dt;
    if (tr->t > tr->total + 0.05f) {
        tr->t = 0;
        memset(tr->entered, 0, sizeof tr->entered);
        stars_reset();
        arch_reset(tr->arch);
    }
    int any_arch = 0;
    for (int i = 0; i < N_BEATS; i++) {
        float a = alpha_in_out(tr->t, tr->beat_start[i], BEATS[i].dur);
        if (a <= 0.001f) continue;
        if (!tr->entered[i]) enter_beat(tr, i);
        if (BEATS[i].kind == SC_ARCH) any_arch = 1;
    }
    if (any_arch) arch_update(tr->arch, dt);
}

static void draw_hud(Trailer *tr) {
    if (tr->hide_hud) return;
    char buf[96];
    snprintf(buf, sizeof buf, "%.1fs / %.1fs    space pause   n next   r restart",
             tr->t, tr->total);
    DrawTextEx(tr->mono, buf, (Vector2){40, SCREEN_H - 36}, 16, 0.4f, calpha(C_MUTED, 0.55f));
}

static void draw_trailer(Trailer *tr) {
    BeginDrawing();
    ClearBackground(C_BG);

    float star_a = 0;
    for (int i = 0; i < N_BEATS; i++) {
        float a = alpha_in_out(tr->t, tr->beat_start[i], BEATS[i].dur);
        if (a > star_a) star_a = a;
    }
    if (star_a > 0.001f)
        stars_draw(&tr->field, star_a);

    for (int i = 0; i < N_BEATS; i++) {
        float a = alpha_in_out(tr->t, tr->beat_start[i], BEATS[i].dur);
        if (a <= 0.001f) continue;
        float local = tr->t - tr->beat_start[i];
        const Beat *b = &BEATS[i];
        switch (b->kind) {
        case SC_ARCH:
            arch_draw(tr->arch, a);
            break;
        case SC_FASTER:
            draw_faster(tr, local, a);
            break;
        case SC_RESULTS:
            draw_results(tr, local, a);
            break;
        case SC_END:
            draw_end(tr, b, local, a);
            break;
        }
    }
    draw_hud(tr);
    EndDrawing();
}

static void skip_next(Trailer *tr) {
    int cur = 0;
    for (int i = 0; i < N_BEATS; i++) {
        if (tr->t + 0.01f >= tr->beat_start[i]) cur = i;
    }
    int n = cur + 1;
    if (n >= N_BEATS) n = 0;
    seek_trailer(tr, tr->beat_start[n] + 0.001f);
}

static void usage(void) {
    fprintf(stderr,
        "usage: ./trailer/trailer [flags]\n"
        "  --shot [FILE]     export a still (default trailer/shots/puffer5.png)\n"
        "  --plot [FILE]     full-screen scale plot still\n"
        "  --record [FILE]   record one pass (default trailer/shots/puffer5.mp4)\n"
        "  --time SEC        seek before play\n"
        "  --once            play once and exit\n"
        "  --headless        hidden window\n");
}

int main(int argc, char **argv) {
    int record = 0, once = 0, headless = 0, plot_only = 0;
    const char *shot = NULL;
    const char *record_path = "trailer/shots/puffer5.mp4";
    float seek = -1.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--shot") == 0) {
            shot = "trailer/shots/puffer5.png";
            if (i + 1 < argc && argv[i + 1][0] != '-') shot = argv[++i];
        } else if (strcmp(argv[i], "--plot") == 0) {
            plot_only = 1;
            shot = "trailer/shots/fig_scale.png";
            if (i + 1 < argc && argv[i + 1][0] != '-') shot = argv[++i];
        } else if (strcmp(argv[i], "--record") == 0) {
            record = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') record_path = argv[++i];
        } else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc) {
            seek = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--once") == 0) {
            once = 1;
        } else if (strcmp(argv[i], "--headless") == 0) {
            headless = 1;
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
    InitWindow(SCREEN_W, SCREEN_H, "PufferLib 5.0");
    SetTargetFPS(60);
#ifndef GRAPHICS_API_OPENGL_ES2
    glEnable(GL_PROGRAM_POINT_SIZE);
#endif
    {
        int samples = 0;
        glGetIntegerv(GL_SAMPLES, &samples);
        printf("GL_SAMPLES=%d (MSAA hint %s)\n", samples,
               samples >= 4 ? "ok" : "not applied — clips use bilinear, not framebuffer AA");
    }
    srand(4);

    Trailer tr = {0};
    layout_beats(&tr);
    tr.hide_hud = record || headless || shot;

    int cps[224];
    for (int i = 0; i < 224; i++) cps[i] = 32 + i;
    tr.ui = LoadFontEx("resources/shared/Montserrat-Regular.ttf", FONT_ATLAS, cps, 224);
    tr.mono = LoadFontEx("resources/shared/JetBrainsMono-Medium.ttf", 80, cps, 224);
    SetTextureFilter(tr.ui.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(tr.mono.texture, TEXTURE_FILTER_BILINEAR);
    if (FileExists("resources/shared/puffers_128.png")) {
        tr.logo = LoadTexture("resources/shared/puffers_128.png");
        SetTextureFilter(tr.logo, TEXTURE_FILTER_BILINEAR);
    }
    load_clip(&tr.puffer, "puffer");
    tr.puffer.loop = 1;
    tr.puffer.punch_black = 1;

    tr.field = LoadShader(
        TextFormat("resources/trailer/star_%i.vs", GLSL_VERSION),
        TextFormat("resources/trailer/star_%i.fs", GLSL_VERSION));
    tr.mote = LoadShader(
        TextFormat("resources/trailer/star_%i.vs", GLSL_VERSION),
        TextFormat("resources/constellation/point_particle_%i.fs", GLSL_VERSION));
    tr.mark = LoadShader(
        TextFormat("resources/constellation/point_particle_%i.vs", GLSL_VERSION),
        TextFormat("resources/constellation/point_particle_%i.fs", GLSL_VERSION));

    stars_init();
    plot_scale_init();
    tr.arch = arch_create(tr.ui, tr.mono, tr.logo, tr.mote);

    signal(SIGPIPE, SIG_IGN);
    load_clip(&tr.train, "breakout1s");
    tr.train.loop = 0;
    load_clip(&tr.breakout, "breakout");
    tr.breakout.loop = 0;
    for (int e = 0; e < N_NEW; e++)
        load_clip(&tr.clip[e], NEWS[e].key);
    arch_hide_hud(tr.arch, 1);

    if (seek >= 0) seek_trailer(&tr, seek);

    if (shot) {
        ensure_shots_dir();
        if (plot_only) {
            for (int f = 0; f < 3; f++) {
                BeginDrawing();
                ClearBackground(C_BG);
                plot_scale_draw(tr.ui, tr.mono, &tr.mark, 1.0f, 1.0f);
                EndDrawing();
            }
        } else {
            for (int f = 0; f < 3; f++) draw_trailer(&tr);
        }
        export_png(shot);
        goto cleanup;
    }

    VideoRecorder rec = {0};
    int recording = 0;
    if (record) {
        ensure_shots_dir();
        recording = open_video(&rec, record_path, SCREEN_W, SCREEN_H);
        if (!recording) fprintf(stderr, "warning: ffmpeg record failed\n");
        else printf("recording %s\n", record_path);
    }

    int frame = 0;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_Q) || IsKeyPressed(KEY_ESCAPE)) break;
        if (IsKeyPressed(KEY_SPACE)) tr.paused = !tr.paused;
        if (IsKeyPressed(KEY_R)) seek_trailer(&tr, 0);
        if (IsKeyPressed(KEY_N)) skip_next(&tr);
        if (IsKeyPressed(KEY_S)) {
            ensure_shots_dir();
            draw_trailer(&tr);
            export_png("trailer/shots/puffer5.png");
        }

        float dt = recording ? (1.0f / 60.0f) : GetFrameTime();
        if (dt > 0.10f) dt = 0.10f;
        if (dt < 0) dt = 0;
        tick(&tr, dt);
        draw_trailer(&tr);

        if (recording && (frame++ % 2 == 0))
            write_frame(&rec, SCREEN_W, SCREEN_H);

        if (once && tr.t >= tr.total - 0.001f) break;
    }

    if (recording) {
        close_video(&rec);
        printf("wrote %s\n", record_path);
    }

cleanup:
    arch_destroy(tr.arch);
    unload_clip(&tr.train);
    unload_clip(&tr.breakout);
    unload_clip(&tr.puffer);
    for (int e = 0; e < N_NEW; e++)
        unload_clip(&tr.clip[e]);
    if (tr.logo.id) UnloadTexture(tr.logo);
    UnloadShader(tr.field);
    UnloadShader(tr.mote);
    if (tr.mark.id) UnloadShader(tr.mark);
    UnloadFont(tr.ui);
    UnloadFont(tr.mono);
    CloseWindow();
    return 0;
}
