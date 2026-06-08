/* live_view.c — interactive single-env NetHack viewer for debugging.
 *
 * Loads libnethack.so, starts one env with a seed, and steps it with
 * either a recorded action stream, random actions, or interactive
 * keystrokes from stdin. Each step is rendered to the terminal with
 * ANSI colors driven by NetHack's colors[] grid, plus a sidebar
 * showing blstats / message / inventory letters.
 *
 * Use cases:
 *   1. Watch an env run with random actions to verify it's not stuck
 *      (catches "agent always picks invalid action and dies in 50 steps"
 *      class of bug):
 *        ./live_view --random --steps 500 --seed 42
 *   2. Replay a recorded action stream from a golden trajectory at a
 *      slow rate:
 *        ./live_view --replay golden_seed05_1k.bin --rate 5
 *   3. Step interactively (one keystroke = one action):
 *        ./live_view --interactive
 *
 * Build:
 *   clang -O2 -Wall -std=gnu11 -I./vendor/nle/include -I./ocean/nethack \
 *       ocean/nethack/live_view.c -o live_view -ldl -lpthread -lm
 *
 * Output ANSI escapes; pipe through `less -R` to scroll back through
 * a no-pause run, or pipe through `vhs`/`asciinema` to record a demo.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <time.h>
#include <termios.h>
#include <errno.h>
#define NLE_ALLOW_SEEDING
#include "nleobs.h"

typedef struct nle_ctx nle_ctx_t;
typedef nle_ctx_t *(*nle_start_fn)(nle_obs *, FILE *, nle_seeds_init_t *,
                                   nle_settings *);
typedef nle_ctx_t *(*nle_step_fn)(nle_ctx_t *, nle_obs *);
typedef void (*nle_end_fn)(nle_ctx_t *);

#define NH_ROWS 21
#define NH_COLS 79
#define NH_GRID (NH_ROWS * NH_COLS)

#define DEFAULT_OPTIONS                                                       \
    "name:Agent-mon-hum-neu-mal,"                                             \
    "autopickup,color,disclose:+i +a +v +g +c +o,"                            \
    "mention_walls,nobones,nocmdassist,nolegacy,nosparkle,"                   \
    "pickup_burden:unencumbered,pickup_types:$?!/,"                           \
    "runmode:teleport,showexp,showscore,time"

/* NetHack's CLR_* match ANSI fg codes 30-37. CLR_BLACK is rendered bright. */
static const char *ansi_fg[16] = {
    "\x1b[1;30m", "\x1b[0;31m", "\x1b[0;32m", "\x1b[0;33m",
    "\x1b[0;34m", "\x1b[0;35m", "\x1b[0;36m", "\x1b[0;37m",
    "\x1b[1;30m", "\x1b[1;31m", "\x1b[1;32m", "\x1b[1;33m",
    "\x1b[1;34m", "\x1b[1;35m", "\x1b[1;36m", "\x1b[1;37m",
};
#define ANSI_RESET "\x1b[0m"
#define ANSI_CLEAR "\x1b[2J\x1b[H"

/* 18-action default action set (NETHACK_ACTION_SET == 1) */
static const int ACTION_TABLE[18] = {
    'k', 'j', 'h', 'l', 'y', 'u', 'b', 'n',
    'K', 'J', 'H', 'L', 'Y', 'U', 'B', 'N',
    '.', 's',
};
#define NUM_ACTIONS 18

static int make_vardir(const char *src, char *out, size_t cap)
{
    char tmpl[] = "/tmp/nle-live-XXXXXX";
    char *d = mkdtemp(tmpl);
    if (!d) return -1;
    snprintf(out, cap, "%s", d);
    char a[2048], b[2048];
    snprintf(a, sizeof(a), "%s/nhdat", src);
    snprintf(b, sizeof(b), "%s/nhdat", d);
    if (symlink(a, b)) return -1;
    const char *t[] = {"perm", "record", "logfile", "xlogfile"};
    for (int i = 0; i < 4; i++) {
        snprintf(b, sizeof(b), "%s/%s", d, t[i]);
        int fd = open(b, O_CREAT | O_WRONLY, 0644);
        if (fd >= 0) close(fd);
    }
    snprintf(b, sizeof(b), "%s/save", d);
    mkdir(b, 0755);
    return 0;
}

typedef struct {
    nle_ctx_t *ctx;
    nle_obs obs;
    unsigned char chars[NH_GRID];
    signed char colors[NH_GRID];
    int misc[NLE_MISC_SIZE];
    int internal[NLE_INTERNAL_SIZE];
    long blstats[NLE_BLSTATS_SIZE];
    int8_t message[256];
    unsigned char inv_letters[NLE_INVENTORY_SIZE];
    unsigned char inv_oclasses[NLE_INVENTORY_SIZE];
    char vardir[256];
    nle_settings settings;
} Env;

static void bind_obs(Env *e)
{
    memset(&e->obs, 0, sizeof(e->obs));
    e->obs.chars = e->chars;
    e->obs.colors = e->colors;
    e->obs.misc = e->misc;
    e->obs.internal = e->internal;
    e->obs.blstats = e->blstats;
    e->obs.message = (unsigned char *) e->message;
    e->obs.inv_letters = e->inv_letters;
    e->obs.inv_oclasses = e->inv_oclasses;
}

static long prev_score = 0;
static int  prev_depth = 1;
static int  visited_level = 0;
static unsigned char visited[21 * 80 / 8 + 1];
static float ep_return = 0.0f;
static long ep_length  = 0;
static long ep_valid   = 0;
static long ep_illegal = 0;
static long ep_tiles   = 0;
static int  last_misc[3] = {0};

static void render(Env *e, long step, int action, float reward)
{
    fputs(ANSI_CLEAR, stdout);
    /* Header */
    printf("step=%-6ld  action=%-3d ('%c')  reward=%+.3f  HP=%ld/%ld  T=%ld  Dlvl=%ld  Score=%ld\n",
           step, action, (action >= 32 && action < 127) ? action : '?', reward,
           e->blstats[NLE_BL_HP], e->blstats[NLE_BL_HPMAX],
           e->blstats[NLE_BL_TIME], e->blstats[NLE_BL_DEPTH],
           e->blstats[NLE_BL_SCORE]);
    /* User stats (matches training Log struct) */
    printf("ep_return=%.2f  ep_len=%ld  depth=%ld  score=%ld  "
           "valid=%ld  illegal=%ld  tiles=%ld  "
           "misc=[%d,%d,%d]\n",
           ep_return, ep_length, e->blstats[NLE_BL_DEPTH],
           e->blstats[NLE_BL_SCORE],
           ep_valid, ep_illegal, ep_tiles,
           last_misc[0], last_misc[1], last_misc[2]);
    /* Message line */
    printf("msg: \"%.*s\"\n", 79, (char *) e->message);
    /* Chars grid w/ ANSI colors */
    for (int r = 0; r < NH_ROWS; r++) {
        int last_c = -2;
        for (int c = 0; c < NH_COLS; c++) {
            int color = e->colors[r * NH_COLS + c] & 0xF;
            unsigned char ch = e->chars[r * NH_COLS + c];
            if (color != last_c) {
                fputs(ansi_fg[color], stdout);
                last_c = color;
            }
            putchar(ch ? ch : ' ');
        }
        fputs(ANSI_RESET "\n", stdout);
    }
    /* Inventory letters */
    printf("inv: ");
    int any = 0;
    for (int i = 0; i < NLE_INVENTORY_SIZE; i++) {
        if (e->inv_letters[i]) {
            printf("%c ", e->inv_letters[i]);
            any = 1;
        }
    }
    if (!any) fputs("(empty)", stdout);
    putchar('\n');
    fflush(stdout);
}

static int drain(Env *e, nle_step_fn fn_step)
{
    int max_iter = 200;
    while (max_iter-- > 0 && e->obs.internal[3] != 0) {
        e->obs.action = '\r';
        e->ctx = fn_step(e->ctx, &e->obs);
        if (e->obs.done) return -1;
    }
    return 0;
}

static void usage(void)
{
    fprintf(stderr,
            "usage: live_view [options]\n"
            "  --random          random actions (default)\n"
            "  --interactive     read action keystrokes from stdin\n"
            "  --steps N         number of steps (random mode only, default 500)\n"
            "  --seed S          seed for RNG and env (default 42)\n"
            "  --rate HZ         pause 1/HZ seconds between frames (default 4)\n"
            "  --no-render       don't render (benchmark mode)\n"
            "  --replay FILE     replay an action stream from a binary file\n"
            "                    (uint32 actions, one per step)\n");
    exit(2);
}

int main(int argc, char **argv)
{
    int mode_random = 1, mode_interactive = 0, mode_replay = 0;
    int no_render = 0;
    long steps = 500;
    unsigned long seed = 42;
    double rate = 4.0;
    const char *replay_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--random")) {
            mode_random = 1;
            mode_interactive = 0;
            mode_replay = 0;
        } else if (!strcmp(argv[i], "--interactive")) {
            mode_interactive = 1;
            mode_random = 0;
        } else if (!strcmp(argv[i], "--replay") && i + 1 < argc) {
            replay_path = argv[++i];
            mode_replay = 1;
            mode_random = 0;
        } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            steps = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoul(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "--rate") && i + 1 < argc) {
            rate = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--no-render")) {
            no_render = 1;
        } else {
            usage();
        }
    }

    const char *libpath = getenv("NETHACK_LIBPATH");
    if (!libpath) libpath = "./vendor/nle/src/build/libnethack.so";
    const char *nhdir = getenv("NETHACKDIR");
    if (!nhdir) nhdir = "vendor/nle/nethackdir";

    void *h = dlopen(libpath, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "dlopen %s: %s\n", libpath, dlerror());
        return 1;
    }
    nle_start_fn fn_start = (nle_start_fn) dlsym(h, "nle_start");
    nle_step_fn fn_step = (nle_step_fn) dlsym(h, "nle_step");
    nle_end_fn fn_end = (nle_end_fn) dlsym(h, "nle_end");
    if (!fn_start || !fn_step || !fn_end) {
        fprintf(stderr, "missing libnethack symbols\n");
        return 1;
    }

    Env e = {0};
    if (make_vardir(nhdir, e.vardir, sizeof(e.vardir))) {
        fprintf(stderr, "vardir setup failed\n");
        return 1;
    }
    bind_obs(&e);
    memset(&e.settings, 0, sizeof(e.settings));
    snprintf(e.settings.hackdir, sizeof(e.settings.hackdir), "%s", e.vardir);
    snprintf(e.settings.scoreprefix, sizeof(e.settings.scoreprefix), "%s/",
             e.vardir);
    snprintf(e.settings.options, sizeof(e.settings.options), "%s",
             DEFAULT_OPTIONS);
    e.settings.spawn_monsters = 1;

    nle_seeds_init_t s = {0};
    s.seeds[0] = seed;
    s.seeds[1] = seed ^ 0x9E3779B97F4A7C15UL;
    e.ctx = fn_start(&e.obs, NULL, &s, &e.settings);
    if (!e.ctx) {
        fprintf(stderr, "nle_start failed\n");
        return 1;
    }
    if (drain(&e, fn_step) < 0) {
        fprintf(stderr, "drain failed\n");
        return 1;
    }

    /* Replay stream */
    int32_t *replay = NULL;
    long replay_len = 0;
    if (mode_replay) {
        FILE *f = fopen(replay_path, "rb");
        if (!f) {
            fprintf(stderr, "fopen %s: %s\n", replay_path, strerror(errno));
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long bytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        replay_len = bytes / 4;
        replay = malloc(bytes);
        if (fread(replay, 4, replay_len, f) != (size_t) replay_len) {
            fprintf(stderr, "short read on replay\n");
            return 1;
        }
        fclose(f);
        steps = replay_len;
    }

    /* Stash terminal mode for interactive */
    struct termios oldt, newt;
    if (mode_interactive) {
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    }

    unsigned r = (unsigned) seed;
    struct timespec frame = {0, 0};
    if (rate > 0 && !mode_interactive && !no_render) {
        long ns = (long) (1e9 / rate);
        frame.tv_sec = ns / 1000000000L;
        frame.tv_nsec = ns % 1000000000L;
    }

    /* Render initial state before first input */
    if (!no_render) {
        render(&e, 0, 0, 0.0f);
    }

    for (long t = 0; t < steps; t++) {
        int action;
        if (mode_interactive) {
            int ch = getchar();
            if (ch == EOF || ch == 'Q') break;
            action = ch;
        } else if (mode_replay) {
            action = replay[t];
        } else {
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            action = ACTION_TABLE[r % NUM_ACTIONS];
        }
        e.obs.action = action;
        e.ctx = fn_step(e.ctx, &e.obs);
        /* Auto-drain --More-- / xwaitforspace so level transitions
         * and multi-line messages don't leave the display stale. */
        /* Capture misc flags before drain clears them */
        if (e.obs.misc) {
            last_misc[0] = e.obs.misc[0];
            last_misc[1] = e.obs.misc[1];
            last_misc[2] = e.obs.misc[2];
        }
        /* Auto-drain yn prompts (answer 'y') and --More-- (send space) */
        for (int drain = 0; drain < 16; drain++) {
            int is_yn = e.obs.misc && e.obs.misc[0];
            int is_xwait = e.obs.misc && e.obs.misc[2];
            if (!is_yn && !is_xwait) break;
            e.obs.action = is_yn ? 'y' : ' ';
            e.ctx = fn_step(e.ctx, &e.obs);
            if (e.obs.done) break;
        }
        /* Track user stats (mirrors c_step reward shaping) */
        long score = e.blstats[NLE_BL_SCORE];
        float reward = 0.01f * (float)(score - prev_score);
        int depth = (int)e.blstats[NLE_BL_DEPTH];
        if (depth > prev_depth) reward += 10.0f * (float)(depth - prev_depth);
        if (depth != visited_level) {
            memset(visited, 0, sizeof(visited));
            visited_level = depth;
        }
        int px = (int)e.blstats[NLE_BL_X];
        int py = (int)e.blstats[NLE_BL_Y];
        if (px >= 0 && px < 80 && py >= 0 && py < 21) {
            int bit = py * 80 + px;
            if (!(visited[bit >> 3] & (1 << (bit & 7)))) {
                visited[bit >> 3] |= (1 << (bit & 7));
                ep_tiles++;
                reward += 0.1f;
            }
        }
        int was_illegal = last_misc[0] || last_misc[1];
        if (was_illegal) { ep_illegal++; reward += -0.1f; }
        else ep_valid++;
        ep_return += reward;
        ep_length++;
        if (!e.obs.done) {
            prev_score = score;
            prev_depth = depth;
        }
        if (e.obs.done) {
            prev_score = 0; prev_depth = 1; visited_level = 0;
            memset(visited, 0, sizeof(visited));
            ep_return = 0; ep_length = 0;
            ep_valid = 0; ep_illegal = 0; ep_tiles = 0;
        }
        if (!no_render) {
            render(&e, t + 1, action, reward);
        }
        if (e.obs.done) {
            printf("\n*** Env done at step %ld (how_done=%d) ***\n", t,
                   e.obs.how_done);
            break;
        }
        if (frame.tv_nsec || frame.tv_sec) nanosleep(&frame, NULL);
    }

    if (mode_interactive) tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fn_end(e.ctx);
    if (replay) free(replay);
    return 0;
}
