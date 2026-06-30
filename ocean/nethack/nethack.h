#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <errno.h>
#include "profile.h"

#ifndef SYS_memfd_create
#  ifdef __x86_64__
#    define SYS_memfd_create 319
#  endif
#endif
static inline int nethack_memfd_create(const char* name, unsigned int flags) {
    return (int)syscall(SYS_memfd_create, name, flags);
}

// ---------------------------------------------------------------------------
// NLE minimal API
// ---------------------------------------------------------------------------
// All per-env mutable state lives in nle_ctx_t. libnethack.so is linked
// directly (no dlopen). Each env gets its own nle_ctx_t + private memory
// arena; a thread-local pointer (current_nle_ctx) anchors field access.
#define NLE_ALLOW_SEEDING 1
#include "nleobs.h"

typedef struct nle_ctx nle_ctx_t;
typedef nle_ctx_t* (*nle_start_fn)(nle_obs*, FILE*, nle_seeds_init_t*, nle_settings*);
typedef nle_ctx_t* (*nle_step_fn)(nle_ctx_t*, nle_obs*);
typedef void       (*nle_end_fn)(nle_ctx_t*);

// Fast-reset extension (see vendor/nle/src/src/nle_fast_reset.c).
typedef void* (*nle_fr_snapshot_fn)(nle_ctx_t*);
typedef void  (*nle_fr_restore_fn) (nle_ctx_t*, void*);
typedef void  (*nle_fr_destroy_fn) (void*);

// Direct linkage: libnethack.so symbols are resolved at link time, not via
// dlopen. The wrapper retains env->fn_* function pointers for ABI stability,
// but they're assigned from these externs — no runtime symbol lookup.
#ifdef __cplusplus
extern "C" {
#endif
extern nle_ctx_t* nle_start(nle_obs*, FILE*, nle_seeds_init_t*, nle_settings*);
extern nle_ctx_t* nle_step(nle_ctx_t*, nle_obs*);
extern void       nle_end(nle_ctx_t*);
extern void*      nle_fr_snapshot(nle_ctx_t*);
extern void       nle_fr_restore(nle_ctx_t*, void*);
extern void       nle_fr_destroy(void*);
#ifdef __cplusplus
}
#endif

// Build flag: -DNETHACK_FAST_RESET=1 enables the snapshot/restore path.
// Default is OFF — the fast-reset machinery in
// vendor/nle/src/src/nle_fast_reset.c is fundamentally unsafe for N>1
// vecenv: nle_fr_restore memcpys back the SHARED bump-arena
// (alloc.c:75 — "All envs share one arena") and libnethack.so's
// .data/.bss segments. Under multi-env, that clobbers other envs' live
// arena allocations and rewinds the bump pointer past their data,
// producing 'double free in tcache 2' / 'munmap_chunk()' / SIGSEGV.
// train_bench at N=64 (which uses slow-only resets via nle_end + nle_start)
// runs 1M steps clean at 414K agg SPS, so slow-only is plenty fast.
// To re-enable for single-env benchmarks: build with -DNETHACK_FAST_RESET=1.
#ifndef NETHACK_FAST_RESET
#define NETHACK_FAST_RESET 0
#endif

// ---------------------------------------------------------------------------
// Geometry constants
// ---------------------------------------------------------------------------
#define NH_ROWS 21
#define NH_COLS 79
#define NH_GRID (NH_ROWS * NH_COLS)

// ---------------------------------------------------------------------------
// Compile-time observation selection
// ---------------------------------------------------------------------------
// Override any of these with -DNETHACK_USE_<FIELD>=0/1. Defaults: chars only.
// Each enabled field reserves its own slice of the ByteTensor observation
// buffer; disabled fields are not allocated, not bound, and NLE does not
// write to them (NLE's fill_obs guards every field with `if (ptr) ...`).
//
// Field widths (bytes per element):
//   chars     uint8     1
//   colors    uint8     1
//   specials  uint8     1
//   glyphs    int16     2  (packed as little-endian)
//   blstats   int64     4  (truncated to int32 to save space)
//   message   uint8     1
//   inv_letters / inv_oclasses  uint8  1  (size NLE_INVENTORY_SIZE)
//
// `inv` enables both inv_letters and inv_oclasses together (cheap pair).
// inv_glyphs and inv_strs are deliberately omitted for v1 to keep the
// observation a flat ByteTensor.

#ifndef NETHACK_USE_CHARS
#define NETHACK_USE_CHARS    1
#endif
#ifndef NETHACK_USE_COLORS
#define NETHACK_USE_COLORS   0
#endif
#ifndef NETHACK_USE_SPECIALS
#define NETHACK_USE_SPECIALS 0
#endif
#ifndef NETHACK_USE_GLYPHS
#define NETHACK_USE_GLYPHS   0
#endif
#ifndef NETHACK_USE_BLSTATS
#define NETHACK_USE_BLSTATS  0
#endif
#ifndef NETHACK_USE_MESSAGE
#define NETHACK_USE_MESSAGE  0
#endif
#ifndef NETHACK_USE_INV
#define NETHACK_USE_INV      0
#endif

// Auto-dismiss prompts (welcome screen, --More--, yes/no, getline) before
// applying the agent's action, by inspecting misc[]={in_yn_function,
// in_getlin, xwaitingforspace} on every step. These tiny buffers are
// always allocated (~48 bytes) regardless of obs-field selection. Cap the
// dismiss loop at NETHACK_AUTODISMISS_MAX iterations to avoid hanging on
// pathological prompt chains.
#ifndef NETHACK_AUTODISMISS
#define NETHACK_AUTODISMISS 1
#endif
#ifndef NETHACK_AUTODISMISS_MAX
#define NETHACK_AUTODISMISS_MAX 64
#endif

#ifndef NETHACK_MAX_EPISODE_STEPS
#define NETHACK_MAX_EPISODE_STEPS 10000
#endif

// Penalty applied to the reward when the agent's action triggers a sub-prompt
// (illegal/ambiguous action — e.g. "Apply what?" / "What direction?") that
// we then have to ESC out of. Default -0.01 — small enough that legitimate
// y/n confirmations during real play don't tank rewards.
// Per-step reward shaping (compile-time tunable).
//   reward = (score - prev_score)
//          + NETHACK_DEPTH_BONUS  if went deeper this step
//          + NETHACK_SCOUT_BONUS  if entered a previously-unvisited tile
//          + NETHACK_ILLEGAL_PENALTY  if action triggered a sub-prompt
#ifndef NETHACK_ILLEGAL_PENALTY
#define NETHACK_ILLEGAL_PENALTY -0.5f      // -invalid_moves/2 per user spec
#endif
#ifndef NETHACK_SCOUT_BONUS
#define NETHACK_SCOUT_BONUS      0.1f       // for each new tile visited
#endif
#ifndef NETHACK_DEPTH_BONUS
#define NETHACK_DEPTH_BONUS      1.0f       // for each new dungeon level
#endif

#define NETHACK_SZ_CHARS    (NETHACK_USE_CHARS    * NH_GRID)
#define NETHACK_SZ_COLORS   (NETHACK_USE_COLORS   * NH_GRID)
#define NETHACK_SZ_SPECIALS (NETHACK_USE_SPECIALS * NH_GRID)
#define NETHACK_SZ_GLYPHS   (NETHACK_USE_GLYPHS   * NH_GRID * 2)
#define NETHACK_SZ_BLSTATS  (NETHACK_USE_BLSTATS  * NLE_BLSTATS_SIZE * 4)
#define NETHACK_SZ_MESSAGE  (NETHACK_USE_MESSAGE  * NLE_MESSAGE_SIZE)
#define NETHACK_SZ_INV      (NETHACK_USE_INV      * NLE_INVENTORY_SIZE * 2)

#define NETHACK_OFF_CHARS    0
#define NETHACK_OFF_COLORS   (NETHACK_OFF_CHARS    + NETHACK_SZ_CHARS)
#define NETHACK_OFF_SPECIALS (NETHACK_OFF_COLORS   + NETHACK_SZ_COLORS)
#define NETHACK_OFF_GLYPHS   (NETHACK_OFF_SPECIALS + NETHACK_SZ_SPECIALS)
#define NETHACK_OFF_BLSTATS  (NETHACK_OFF_GLYPHS   + NETHACK_SZ_GLYPHS)
#define NETHACK_OFF_MESSAGE  (NETHACK_OFF_BLSTATS  + NETHACK_SZ_BLSTATS)
#define NETHACK_OFF_INV      (NETHACK_OFF_MESSAGE  + NETHACK_SZ_MESSAGE)
#define NETHACK_OBS_SIZE     (NETHACK_OFF_INV      + NETHACK_SZ_INV)

#if NETHACK_OBS_SIZE == 0
#error "At least one NETHACK_USE_* field must be enabled."
#endif

// Compile-time action set selection.
//   NETHACK_ACTION_SET == 0  (default, 23 actions) — full reduced set:
//       8 compass + 8 long-compass + > < . s \r ESC ,
//   NETHACK_ACTION_SET == 1  (18 actions) — drops the "rarely useful"
//       keystrokes (`>`, `<`, `,`, `\r`, `ESC`) that early-training
//       random play burns on no-op/menu states. Keeps movement + wait
//       + search. Helps lift SPS during the high-reset-rate phase.
#ifndef NETHACK_ACTION_SET
#define NETHACK_ACTION_SET 0
#endif

#if NETHACK_ACTION_SET == 1
#define NETHACK_NUM_ACTIONS 18
static const int NETHACK_ACTION_TABLE[NETHACK_NUM_ACTIONS] = {
    'k','j','h','l','y','u','b','n',
    'K','J','H','L','Y','U','B','N',
    '.','s',
};
#else
#define NETHACK_NUM_ACTIONS 23
static const int NETHACK_ACTION_TABLE[NETHACK_NUM_ACTIONS] = {
    'k','j','h','l','y','u','b','n',
    'K','J','H','L','Y','U','B','N',
    '>','<','.','s','\r',27,',',
};
#endif

#define NETHACK_DEFAULT_OPTIONS \
    "name:Agent-mon-hum-neu-mal," \
    "autopickup,color,disclose:+i +a +v +g +c +o," \
    "mention_walls,nobones,nocmdassist,nolegacy,nosparkle," \
    "pickup_burden:unencumbered,pickup_types:$?!/," \
    "runmode:teleport,showexp,showscore,time," \
    /* exp_039: disable bot() status renderer. The botl.c chain feeds
     * iflags.status_updates -> bot_via_windowport ->
     * eval_notify_windowport_field -> anything_to_s -> sprintf, eating
     * ~5% of user CPU formatting a status line nothing reads. The agent
     * gets stats via update_blstats() reading u.X / youmonst directly. */ \
    "!status_updates"

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float depth;
    float valid_moves;        // c_steps where the agent's action advanced NetHack's turn counter
    float illegal_actions;    // c_steps where the agent's action hit a sub-prompt we had to ESC out of
    float new_tiles;          // unique tiles entered this episode (sums over episodes via Log.n)
    float n;
} Log;

typedef struct Nethack {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    // Per-instance dynamic load of libnethack.so.
    void* dl_handle;
    int   dl_fd;
    nle_start_fn fn_start;
    nle_step_fn  fn_step;
    nle_end_fn   fn_end;
    nle_fr_snapshot_fn fn_fr_snapshot;  // NULL if patched lib not present
    nle_fr_restore_fn  fn_fr_restore;
    nle_fr_destroy_fn  fn_fr_destroy;
    void* fr_snapshot;  // populated after first c_reset

    // Lazy reset: c_reset sets this flag, c_step performs the actual reset.
    // Guarantees reset and step run on the same OMP thread (NLE's coroutine
    // captures the stack pointer; resetting on a different thread corrupts it).
    int pending_reset;

    // NLE state
    nle_ctx_t* ctx;
    nle_obs obs;
    nle_settings settings;
    char vardir[4096];

    // Backing storage — only fields with USE_* enabled are actually allocated.
    // Conditionally compile these to save memory.
#if NETHACK_USE_GLYPHS
    short          glyphs[NH_GRID];
#endif
#if NETHACK_USE_CHARS
    unsigned char  chars[NH_GRID];
#endif
#if NETHACK_USE_COLORS
    unsigned char  colors[NH_GRID];
#endif
#if NETHACK_USE_SPECIALS
    unsigned char  specials[NH_GRID];
#endif
#if NETHACK_USE_BLSTATS
    long           blstats[NLE_BLSTATS_SIZE];
#endif
#if NETHACK_USE_MESSAGE
    unsigned char  message[NLE_MESSAGE_SIZE];
#endif
#if NETHACK_USE_INV
    unsigned char  inv_letters[NLE_INVENTORY_SIZE];
    unsigned char  inv_oclasses[NLE_INVENTORY_SIZE];
#endif

    // Always-allocated hook buffers (independent of obs selection).
    // misc/internal: prompt-state flags for auto-dismiss
    // blstats: lets us read NLE_BL_TIME to count valid moves
    // message: scanned for '?' to catch single-key prompts (direction, item)
    //          that NLE does not expose via the misc[] flags.
    int           hook_misc[NLE_MISC_SIZE];
    int           hook_internal[NLE_INTERNAL_SIZE];
    long          hook_blstats[NLE_BLSTATS_SIZE];
    unsigned char hook_message[NLE_MESSAGE_SIZE];

    long episode_start_time;
    long episode_valid_moves;
    long episode_illegal_actions;
    long episode_new_tiles;       // unique tiles entered this episode

    // Exploration bitmap for the current dungeon level. Cleared on
    // dungeon-level change. One bit per (row, col); 21*79 = 1659 bits,
    // round up to 208 bytes.
    unsigned char visited[(NH_GRID + 7) / 8];
    int  visited_level;            // dungeon level the bitmap corresponds to

    int tick;
    long prev_score;
    int prev_depth;
    float episode_return;
    int episode_length;
    unsigned int rng;   // required by vecenv.h (seeded with env index)

    // ----- Per-instance reward shaping coefficients -----
    // Each reward term is independently toggleable: set its coef to 0.0 to
    // disable. Set from kwargs in my_init (binding.c). Plumb through
    // `[env]` in config/nethack.ini → CLI `--env.<key>` flags.
    //
    // reward = score_coef    * (score - prev_score)               [always >=0]
    //        + descent_coef  * max(0, depth - prev_depth)         [descent only]
    //        + scout_coef    * (1 if new tile this step else 0)
    //        + illegal_pen   * (1 if action triggered sub-prompt else 0)
    // Defaults below are also the fallbacks when the kwargs key is missing
    // (so the build still runs if config/nethack.ini hasn't been updated).
    float score_coef;       // default 0.01f  (R_SCORE)
    float descent_coef;     // default 1.0f   (R_DESCENT)
    float scout_coef;       // default NETHACK_SCOUT_BONUS (legacy)
    float illegal_penalty;  // default NETHACK_ILLEGAL_PENALTY (legacy, negative)

    // Per-env explicit RNG seed for nle_start. Without this, NetHack uses
    // its own internal seeding and some seeds hit infinite loops in level
    // generation (mklev/topologize). Seed bases known to work at N=32/64/96:
    // 0x111, 0x222, 0xCAFEBEEF.  Set via NETHACK_SEED_BASE env var or default.
    unsigned long seed_a;
    unsigned long seed_b;
} Nethack;

// ---------------------------------------------------------------------------
// Shared libnethack: DIRECT LINKAGE. No dlopen, no dlsym. The library is
// linked at build time (-lnethack); all envs share the same code and use
// per-env nle_ctx_t* for state. The migration in vendor/nle/src/* moved
// every per-game global into nle_ctx_t, so a single shared library
// supports N envs natively.
// ---------------------------------------------------------------------------
static int nethack_load_lib(Nethack* env) {
    PROF_INIT_IF_NEEDED();
    PROF_START(reload);
    // No dlopen — symbols are resolved at link time. Just assign the
    // function pointers so the rest of the wrapper code (env->fn_step etc.)
    // continues to work unchanged.
    env->dl_handle = (void*)1;  // non-NULL sentinel: "library is available"
    env->dl_fd     = 0;
    env->fn_start  = &nle_start;
    env->fn_step   = &nle_step;
    env->fn_end    = &nle_end;
    env->fn_fr_snapshot = &nle_fr_snapshot;
    env->fn_fr_restore  = &nle_fr_restore;
    env->fn_fr_destroy  = &nle_fr_destroy;
    PROF_END(reload, PROF_RESET_RELOAD);
    return 0;
}

static void nethack_unload_lib(Nethack* env) {
    // No per-env dlclose; the library handle is process-wide.
    env->dl_handle = NULL;
    env->dl_fd = 0;
    env->fn_start = NULL; env->fn_step = NULL; env->fn_end = NULL;
}

// ---------------------------------------------------------------------------
// Per-instance vardir (NetHack expects nhdat + a few touched files)
// ---------------------------------------------------------------------------
static void nethack_touch(const char* path) {
    int fd = open(path, O_CREAT | O_WRONLY, 0644);
    if (fd >= 0) close(fd);
}

static int nethack_make_vardir(const char* source_hackdir, char* out_buf, size_t out_cap) {
    PROF_START(vardir);
    char tmpl[] = "/tmp/nle-XXXXXX";
    char* dir = mkdtemp(tmpl);
    if (dir == NULL) return -1;
    if ((size_t)snprintf(out_buf, out_cap, "%s", dir) >= out_cap) return -1;

    char abs_source[4096];
    if (source_hackdir[0] == '/') {
        snprintf(abs_source, sizeof(abs_source), "%s", source_hackdir);
    } else {
        char cwd[2048];
        if (getcwd(cwd, sizeof(cwd)) == NULL) return -1;
        snprintf(abs_source, sizeof(abs_source), "%s/%s", cwd, source_hackdir);
    }

    char src[4096], dst[4096];
    snprintf(src, sizeof(src), "%s/nhdat", abs_source);
    snprintf(dst, sizeof(dst), "%s/nhdat", dir);

    // Fail-fast: verify the source nhdat actually exists before creating the
    // symlink. symlink(2) succeeds for dangling links, so without this check
    // a broken NETHACKDIR (or a cwd-shifted relative default) surfaces 100ms
    // later as a libnethack panic in init_dungeons("Cannot open dungeon
    // description - 'dungeon'..."), which looks like a concurrency crash but
    // is actually a config error. See exp_033_cluster_az/REDIAG.md.
    char resolved[4096];
    char cwd_dbg[2048] = "(unknown)";
    getcwd(cwd_dbg, sizeof(cwd_dbg));
    if (realpath(src, resolved) == NULL || access(resolved, R_OK) != 0) {
        fprintf(stderr,
                "nethack: NETHACKDIR is misconfigured.\n"
                "  Expected to find nhdat at: %s\n"
                "  (source_hackdir=\"%s\", cwd=\"%s\")\n"
                "  realpath/access failed: %s\n"
                "  Set NETHACKDIR to an ABSOLUTE path pointing at a directory\n"
                "  containing nhdat (e.g. <repo>/vendor/nle/nethackdir).\n",
                src, source_hackdir, cwd_dbg, strerror(errno));
        exit(1);
    }
    if (symlink(src, dst) != 0) return -1;
    const char* touched[] = {"perm", "record", "logfile", "xlogfile"};
    for (size_t i = 0; i < 4; i++) {
        snprintf(dst, sizeof(dst), "%s/%s", dir, touched[i]);
        nethack_touch(dst);
    }
    snprintf(dst, sizeof(dst), "%s/save", dir);
    mkdir(dst, 0755);
    PROF_END(vardir, PROF_RESET_VARDIR);
    return 0;
}

static void nethack_rm_vardir(const char* dir) {
    if (dir == NULL || dir[0] == '\0') return;
    char p[4096];
    const char* files[] = {"nhdat", "perm", "record", "logfile", "xlogfile", "paniclog", "save"};
    for (size_t i = 0; i < sizeof(files)/sizeof(files[0]); i++) {
        snprintf(p, sizeof(p), "%s/%s", dir, files[i]);
        if (i == 6) rmdir(p); else unlink(p);
    }
    rmdir(dir);
}

// ---------------------------------------------------------------------------
// Bind obs pointers — only enabled fields, others stay NULL so NLE skips them
// ---------------------------------------------------------------------------
static void nethack_bind_obs(Nethack* env) {
    nle_obs* o = &env->obs;
    memset(o, 0, sizeof(*o));
#if NETHACK_USE_CHARS
    o->chars = env->chars;
#endif
#if NETHACK_USE_COLORS
    o->colors = env->colors;
#endif
#if NETHACK_USE_SPECIALS
    o->specials = env->specials;
#endif
#if NETHACK_USE_GLYPHS
    o->glyphs = env->glyphs;
#endif
#if NETHACK_USE_BLSTATS
    o->blstats = env->blstats;
#endif
#if NETHACK_USE_MESSAGE
    o->message = env->message;
#endif
#if NETHACK_USE_INV
    o->inv_letters  = env->inv_letters;
    o->inv_oclasses = env->inv_oclasses;
#endif
    // Always bind misc/internal so the hook can read prompt-state flags
    // independent of which obs fields the user selected. blstats/message are
    // only bound to hook_* when the user opted them OUT of the obs tensor —
    // otherwise they are already bound to env->{blstats,message} above.
    o->misc     = env->hook_misc;
    o->internal = env->hook_internal;
#if !NETHACK_USE_BLSTATS
    o->blstats  = env->hook_blstats;
#endif
#if !NETHACK_USE_MESSAGE
    o->message  = env->hook_message;
#endif
}

// Pointers we can always read for the hook, regardless of NETHACK_USE_*.
static inline const unsigned char* nethack_msg(const Nethack* env) {
#if NETHACK_USE_MESSAGE
    return env->message;
#else
    return env->hook_message;
#endif
}

// Read NetHack's current turn counter regardless of which obs fields are bound.
static inline long nethack_current_time(const Nethack* env) {
#if NETHACK_USE_BLSTATS
    return env->blstats[NLE_BL_TIME];
#else
    return env->hook_blstats[NLE_BL_TIME];
#endif
}

// Auto-dismiss: while NLE reports an active prompt (welcome/--More--/yn/getlin),
// inject the appropriate dismiss keystroke until the game is back at the
// main command prompt. counter_id distinguishes the calling site so the
// profiler can attribute fn_step calls correctly. Returns # fn_step calls made.
static int nethack_drain_prompts_cat(Nethack* env, ProfCounterId fn_step_counter) {
    int n = 0;
#if NETHACK_AUTODISMISS
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX; i++) {
        int xwait = env->hook_misc[2];
        int yn    = env->hook_misc[0];
        int gl    = env->hook_misc[1];
        if (!xwait && !yn && !gl) break;
        if (xwait)      { env->obs.action = '\r'; PROF_COUNT(PROF_XWAIT_DRAIN, 1); }
        else if (yn)    { env->obs.action = 27;   PROF_COUNT(PROF_MISC_YN_ESC, 1); }
        else /* gl */   { env->obs.action = '\r'; PROF_COUNT(PROF_MISC_GETLIN_ESC, 1); }
        env->ctx = env->fn_step(env->ctx, &env->obs);
        n++;
        PROF_COUNT(PROF_FN_STEPS_TOTAL, 1);
        PROF_COUNT(fn_step_counter, 1);
        if (env->obs.done) break;
    }
#endif
    return n;
}

static void nethack_init_settings(Nethack* env) {
    memset(&env->settings, 0, sizeof(env->settings));
    const char* source = getenv("NETHACKDIR");
    if (source == NULL) source = "./vendor/nle/nethackdir";

    if (nethack_make_vardir(source, env->vardir, sizeof(env->vardir)) != 0) {
        fprintf(stderr, "nethack: failed to create vardir from source=%s\n", source);
        strncpy(env->settings.hackdir, source, sizeof(env->settings.hackdir) - 1);
    } else {
        strncpy(env->settings.hackdir, env->vardir, sizeof(env->settings.hackdir) - 1);
    }
    env->settings.scoreprefix[0] = '\0';
    env->settings.spawn_monsters = 1;
    env->settings.ttyrecname[0] = '\0';
    strncpy(env->settings.options, NETHACK_DEFAULT_OPTIONS, sizeof(env->settings.options) - 1);
    env->settings.wizkit[0] = '\0';
}

void init(Nethack* env) {
    env->ctx = NULL;
    env->tick = 0;
    env->prev_score = 0;
    env->prev_depth = 1;
    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->vardir[0] = '\0';
    env->dl_handle = NULL; env->dl_fd = 0;
    env->fn_fr_snapshot = NULL;
    env->fn_fr_restore = NULL;
    env->fn_fr_destroy = NULL;
    env->fr_snapshot = NULL;

    // Default reward shaping coefficients. Overridable from kwargs in my_init
    // (binding.c) before any reset/step occurs. If a binding chooses not to
    // populate these (e.g. standalone train_bench, replay_view), the defaults
    // below preserve the legacy compile-time behavior for descent + scout +
    // illegal, and use R_SCORE=0.01 per ocean/nethack/REWARDS.md.
    env->score_coef      = 0.01f;
    env->descent_coef    = NETHACK_DEPTH_BONUS;     // 1.0f
    env->scout_coef      = NETHACK_SCOUT_BONUS;     // 0.1f
    env->illegal_penalty = NETHACK_ILLEGAL_PENALTY; // -0.5f

    // Pick per-env seeds derived from env->rng (vecenv sets this to env index)
    // and a process-wide base. Same construction train_bench uses at N=64,
    // which avoids the upstream mklev hang.
    unsigned long base = 0xCAFEBEEFUL;
    const char* sb = getenv("NETHACK_SEED_BASE");
    if (sb) {
        char* end = NULL;
        unsigned long v = strtoul(sb, &end, 0);
        if (end && end != sb) base = v;
    }
    env->seed_a = base + (unsigned long)env->rng;
    env->seed_b = (base ^ 0x9E3779B97F4A7C15UL) + (unsigned long)env->rng;
    // Don't load_lib here — c_reset will do it on first call. Avoids a
    // redundant ~180 ms dlopen+nle_start before the user even resets.
    nethack_init_settings(env);
    nethack_bind_obs(env);
}

void c_close(Nethack* env) {
    if (env->fr_snapshot && env->fn_fr_destroy) {
        env->fn_fr_destroy(env->fr_snapshot);
        env->fr_snapshot = NULL;
    }
    if (env->ctx != NULL && env->fn_end) {
        env->fn_end(env->ctx);
        env->ctx = NULL;
    }
    nethack_unload_lib(env);
    nethack_rm_vardir(env->vardir);
    env->vardir[0] = '\0';
}

// ---------------------------------------------------------------------------
// Observation packing: copy enabled fields into the flat ByteTensor buffer
// ---------------------------------------------------------------------------
static void nethack_pack_obs(Nethack* env) {
    unsigned char* o = env->observations;
#if NETHACK_USE_CHARS
    memcpy(o + NETHACK_OFF_CHARS, env->chars, NETHACK_SZ_CHARS);
#endif
#if NETHACK_USE_COLORS
    memcpy(o + NETHACK_OFF_COLORS, env->colors, NETHACK_SZ_COLORS);
#endif
#if NETHACK_USE_SPECIALS
    memcpy(o + NETHACK_OFF_SPECIALS, env->specials, NETHACK_SZ_SPECIALS);
#endif
#if NETHACK_USE_GLYPHS
    memcpy(o + NETHACK_OFF_GLYPHS, env->glyphs, NETHACK_SZ_GLYPHS);
#endif
#if NETHACK_USE_BLSTATS
    // Pack 27 longs as 27 int32s (truncate; NetHack stat values fit in i32).
    int32_t* dst = (int32_t*)(o + NETHACK_OFF_BLSTATS);
    for (int i = 0; i < NLE_BLSTATS_SIZE; i++) dst[i] = (int32_t)env->blstats[i];
#endif
#if NETHACK_USE_MESSAGE
    memcpy(o + NETHACK_OFF_MESSAGE, env->message, NETHACK_SZ_MESSAGE);
#endif
#if NETHACK_USE_INV
    memcpy(o + NETHACK_OFF_INV,                          env->inv_letters,  NLE_INVENTORY_SIZE);
    memcpy(o + NETHACK_OFF_INV + NLE_INVENTORY_SIZE,     env->inv_oclasses, NLE_INVENTORY_SIZE);
#endif
}

static void nethack_add_log(Nethack* env) {
    env->log.perf            += (float)env->prev_score;
    env->log.score           += (float)env->prev_score;
    env->log.depth           += (float)env->prev_depth;
    env->log.valid_moves     += (float)env->episode_valid_moves;
    env->log.illegal_actions += (float)env->episode_illegal_actions;
    env->log.new_tiles       += (float)env->episode_new_tiles;
    env->log.episode_return  += env->episode_return;
    env->log.episode_length  += env->episode_length;
    env->log.n               += 1.0f;
}

// Per-episode bookkeeping reset (counters, exploration bitmap, obs pack).
// Called by both the slow and fast c_reset paths.
static void nethack_reset_bookkeeping(Nethack* env) {
    env->tick = 0;
    env->prev_score = 0;
#if NETHACK_USE_BLSTATS
    env->prev_depth = (int)env->blstats[NLE_BL_DEPTH];
#else
    env->prev_depth = 1;
#endif
    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->episode_start_time = nethack_current_time(env);
    env->episode_valid_moves = 0;
    env->episode_illegal_actions = 0;
    env->episode_new_tiles = 0;
    memset(env->visited, 0, sizeof(env->visited));
    env->visited_level = 0;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    PROF_START(reset_obs_pack);
    nethack_pack_obs(env);
    PROF_END(reset_obs_pack, PROF_OBS_PACK);
}

// Slow path: dlopen + nle_start + drain welcome. Called on first c_reset
// (and on every reset when NETHACK_FAST_RESET is disabled).
//
// The previous process-wide nethack_slow_reset_mu is removed.
// All named hazards it guarded are now safe under concurrent c_reset:
//   - choose_windows / windowprocs: idempotent CAS-guarded first-init-wins
//     in vendor/nle/src/src/windows.c.
//   - dlb_init / dlb_libs[]: idempotent CAS-guarded first-init-wins in
//     vendor/nle/src/src/dlb.c.
//   - nle_baseline: lazily allocated empty `struct nle_dungeon_save` whose
//     save/load functions are no-ops; even a
//     concurrent double-calloc only leaks one zero-filled blob.
//   - init_artifacts memset of artidisco[]: artidisco is now per-env on
//     nle_ctx_t (see previous commit).
//   - All "many libnethack `.data` globals" called out in the original
//     comment: now fully migrated. Every write reached from nle_start -> init_nle ->
//     mainloop -> unixmain -> moveloop -> init_nethack / u_init /
//     init_dungeons / init_objects / init_artifacts now targets a
//     current_nle_ctx->s_* field (or a same-value-every-time process-shared
//     register that is idempotent across envs).
//
// Removing the mutex restores concurrent reset: with N envs spread across
// pthread workers, env i's c_reset no longer serializes against env j's.
static void nethack_slow_reset(Nethack* env) {
    PROF_START(nle_end);
    if (env->ctx != NULL && env->fn_end) {
        env->fn_end(env->ctx);
        env->ctx = NULL;
    }
    if (env->dl_handle != NULL) {
        nethack_unload_lib(env);
    }
    PROF_END(nle_end, PROF_RESET_NLE_END);

    if (nethack_load_lib(env) != 0) {
        fprintf(stderr, "nethack: failed to reload libnethack on reset\n");
        return;
    }
    nethack_bind_obs(env);
    env->obs.action = 0;
    env->obs.done = 0;
    env->obs.in_normal_game = 0;
    env->obs.how_done = 0;

    if (env->fn_start == NULL) {
        fprintf(stderr, "nethack: fn_start is NULL — load_lib failed\n");
        return;
    }
    PROF_START(nle_start);
    // Advance the per-env LCG so each reset gets a fresh seed. Without
    // explicit seeds, NetHack's internal randomness can land on level-gen
    // seeds that infinite-loop in mklev/topologize. The LCG constants
    // (Numerical Recipes / MMIX) are the same as train_bench.
    env->seed_a = env->seed_a * 6364136223846793005UL + 1442695040888963407UL;
    env->seed_b = env->seed_b * 6364136223846793005UL + 1442695040888963407UL;
    nle_seeds_init_t seeds;
    memset(&seeds, 0, sizeof(seeds));
    seeds.seeds[0] = env->seed_a;
    seeds.seeds[1] = env->seed_b;
    seeds.reseed = 0;
    env->ctx = env->fn_start(&env->obs, NULL, &seeds, &env->settings);
    PROF_END(nle_start, PROF_RESET_NLE_START);

    PROF_START(reset_drain);
    nethack_drain_prompts_cat(env, PROF_FN_STEPS_RESET_DRAIN);
    PROF_END(reset_drain, PROF_RESET_DRAIN);
}

static void nethack_do_reset(Nethack* env) {
    PROF_INIT_IF_NEEDED();
    PROF_START(reset_total);
    PROF_COUNT(PROF_C_RESETS, 1);

#if NETHACK_FAST_RESET
    if (env->fr_snapshot && env->fn_fr_restore) {
        env->fn_fr_restore(env->ctx, env->fr_snapshot);
        nethack_bind_obs(env);
        env->obs.done = 0;
        env->obs.in_normal_game = 0;
        env->obs.how_done = 0;
        nethack_reset_bookkeeping(env);
        PROF_END(reset_total, PROF_C_RESET_TOTAL);
        return;
    }
#endif

    nethack_slow_reset(env);

#if NETHACK_FAST_RESET
    if (env->fn_fr_snapshot && env->fr_snapshot == NULL && env->ctx) {
        env->fr_snapshot = env->fn_fr_snapshot(env->ctx);
        if (env->fr_snapshot == NULL) {
            fprintf(stderr, "nethack: nle_fr_snapshot returned NULL; "
                            "fast-reset will fall back to slow path\n");
        }
    } else if (env->fn_fr_snapshot == NULL && env->fr_snapshot == NULL) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "nethack: NETHACK_FAST_RESET=1 but loaded "
                            "libnethack.so does not export nle_fr_snapshot; "
                            "falling back to slow path\n");
            warned = 1;
        }
    }
#endif

    nethack_reset_bookkeeping(env);
    PROF_END(reset_total, PROF_C_RESET_TOTAL);
}

void c_reset(Nethack* env) {
    env->pending_reset = 1;
}

void c_step(Nethack* env) {
    if (env->pending_reset) {
        env->pending_reset = 0;
        nethack_do_reset(env);
    }
    PROF_INIT_IF_NEEDED();
    PROF_START(c_step_total);
    PROF_COUNT(PROF_C_STEPS, 1);
#if NETHACK_PROFILE
    unsigned long fn_step_calls_before = g_prof.counters[PROF_FN_STEPS_TOTAL];
#endif
    int action_idx = (int)env->actions[0];
    if (action_idx < 0) action_idx = 0;
    if (action_idx >= NETHACK_NUM_ACTIONS) action_idx = NETHACK_NUM_ACTIONS - 1;
    env->obs.action = NETHACK_ACTION_TABLE[action_idx];

    long time_before = nethack_current_time(env);
    PROF_START(agent_fn_step);
    env->ctx = env->fn_step(env->ctx, &env->obs);
    PROF_END(agent_fn_step, PROF_AGENT_FN_STEP);
    PROF_COUNT(PROF_FN_STEPS_TOTAL, 1);
    PROF_COUNT(PROF_FN_STEPS_AGENT, 1);

    // Determine whether the agent's action landed in a sub-prompt:
    //   (a) misc[] flags (yn / getlin) — strong signal
    //   (b) message ends with '?' — catches single-key prompts NLE doesn't
    //       expose via misc, e.g. "In what direction do you want to throw?",
    //       "Apply what?", "What do you want to wield?". Legitimate game
    //       messages ("It's a wall.", "You feel...") don't end in '?'.
    int yn_or_getlin = (env->hook_misc[0] || env->hook_misc[1]);
    int xwait        = env->hook_misc[2];
    int msg_is_prompt = 0;
    const unsigned char* msg = nethack_msg(env);
    if (msg[0]) {
        int end = 0;
        while (end < NLE_MESSAGE_SIZE && msg[end]) end++;
        // Trim trailing spaces.
        while (end > 0 && msg[end-1] == ' ') end--;
        if (end > 0 && msg[end-1] == '?') msg_is_prompt = 1;
    }

    int illegal = yn_or_getlin || msg_is_prompt;
    PROF_START(post_drain);
    if (illegal) {
        // Auto-dismiss prompts the agent can't handle:
        //   yn_function (misc[0]): answer 'y' — commit to the action the agent chose
        //   getlin (misc[1]): ESC — can't type a string
        //   --More-- / trailing '?': ESC
        for (int i = 0; i < NETHACK_AUTODISMISS_MAX; i++) {
            int is_yn   = env->hook_misc[0];
            int is_getlin = env->hook_misc[1];
            int is_xwait  = env->hook_misc[2];
            int still_prompt = (is_yn || is_getlin || is_xwait);
            if (!still_prompt) {
                const unsigned char* m2 = nethack_msg(env);
                int e = 0; while (e < NLE_MESSAGE_SIZE && m2[e]) e++;
                while (e > 0 && m2[e-1] == ' ') e--;
                if (e == 0 || m2[e-1] != '?') break;
            }
            env->obs.action = is_yn ? 'y' : 27;  // yn→yes, everything else→ESC
            env->ctx = env->fn_step(env->ctx, &env->obs);
            PROF_COUNT(PROF_FN_STEPS_TOTAL, 1);
            PROF_COUNT(PROF_FN_STEPS_POST_DRAIN, 1);
            PROF_COUNT(PROF_MSG_PROMPT_ESC, msg_is_prompt && !yn_or_getlin ? 1 : 0);
            if (env->obs.done) break;
        }
        env->episode_illegal_actions++;
        PROF_COUNT(PROF_ILLEGAL_ACTIONS, 1);
    } else if (xwait) {
        nethack_drain_prompts_cat(env, PROF_FN_STEPS_POST_DRAIN);
    }
    PROF_END(post_drain, PROF_POST_DRAIN);

    long time_after = nethack_current_time(env);
    if (time_after > time_before) {
        env->episode_valid_moves++;
        PROF_COUNT(PROF_VALID_MOVES, 1);
    } else {
        // No time advance: classify why.
        if (illegal) {
            PROF_COUNT(PROF_NOADV_PROMPT_DETECTED, 1);
        } else {
            // No prompt detected. Did NetHack at least show a message?
            const unsigned char* m = nethack_msg(env);
            if (m[0]) PROF_COUNT(PROF_NOADV_HAS_MSG, 1);
            else      PROF_COUNT(PROF_NOADV_SILENT, 1);
        }
    }

    env->tick++;
    env->episode_length++;

    long score = 0; int depth = env->prev_depth;
    long px = 0, py = 0;
#if NETHACK_USE_BLSTATS
    score = env->blstats[NLE_BL_SCORE];
    depth = (int)env->blstats[NLE_BL_DEPTH];
    px = env->blstats[NLE_BL_X];
    py = env->blstats[NLE_BL_Y];
#else
    score = env->hook_blstats[NLE_BL_SCORE];
    depth = (int)env->hook_blstats[NLE_BL_DEPTH];
    px = env->hook_blstats[NLE_BL_X];
    py = env->hook_blstats[NLE_BL_Y];
#endif
    // ---- Reward shaping (each term independently toggleable) ----
    // See ocean/nethack/REWARDS.md for the recipe to add more terms.
    // score-delta term: positive whenever NetHack's score increases (kills,
    // depth, gold pickup, etc.). Score is monotonic non-decreasing modulo
    // some rare deaths/penalties; set score_coef=0 to disable entirely.
    float reward = env->score_coef * (float)(score - env->prev_score);

    // Depth-changed bonus + reset exploration bitmap.
    if (depth != env->visited_level) {
        memset(env->visited, 0, sizeof(env->visited));
        env->visited_level = depth;
    }
    // Descent term: one-shot bonus on each new max-depth step (positive only
    // — we never charge the agent for going back up, so it cannot game the
    // signal by yo-yo'ing between levels via score-delta on score loss).
    if (depth > env->prev_depth) reward += env->descent_coef * (float)(depth - env->prev_depth);

    // Scout bonus: reward each new (row,col) entered this level.
    // px is column (0..79), py is row (0..21). Clamp defensively.
    if (px >= 0 && px < NH_COLS && py >= 0 && py < NH_ROWS) {
        int bit_idx = (int)py * NH_COLS + (int)px;
        unsigned char* b = &env->visited[bit_idx >> 3];
        unsigned char mask = (unsigned char)(1 << (bit_idx & 7));
        if (!(*b & mask)) {
            *b |= mask;
            reward += env->scout_coef;
            env->episode_new_tiles++;
        }
    }

    if (illegal) reward += env->illegal_penalty;
    if (!env->obs.done) {
        env->prev_score = score;
        env->prev_depth = depth;
    }

    env->rewards[0] = reward;
    env->episode_return += reward;

    if (env->obs.done || env->episode_length >= NETHACK_MAX_EPISODE_STEPS) {
        env->terminals[0] = 1.0f;
        nethack_add_log(env);
        PROF_START(obs_pack_done);
        nethack_pack_obs(env);
        PROF_END(obs_pack_done, PROF_OBS_PACK);
        PROF_END(c_step_total, PROF_C_STEP_TOTAL);
        PROF_FN_STEPS_PER_C_STEP(g_prof.counters[PROF_FN_STEPS_TOTAL] - fn_step_calls_before);
        c_reset(env);
        return;
    }
    env->terminals[0] = 0.0f;
    PROF_START(obs_pack);
    nethack_pack_obs(env);
    PROF_END(obs_pack, PROF_OBS_PACK);
    PROF_END(c_step_total, PROF_C_STEP_TOTAL);
    PROF_FN_STEPS_PER_C_STEP(g_prof.counters[PROF_FN_STEPS_TOTAL] - fn_step_calls_before);
}

void c_render(Nethack* env) {
    printf("\x1b[H\x1b[2J");
#if NETHACK_USE_CHARS
    for (int r = 0; r < NH_ROWS; r++) {
        for (int c = 0; c < NH_COLS; c++) {
            unsigned char ch = env->chars[r * NH_COLS + c];
            putchar(ch ? ch : ' ');
        }
        putchar('\n');
    }
#else
    printf("(chars obs disabled)\n");
#endif
#if NETHACK_USE_BLSTATS
    printf("HP %ld/%ld  AC %ld  Dlvl %ld  Score %ld  T %ld\n",
           env->blstats[NLE_BL_HP], env->blstats[NLE_BL_HPMAX],
           env->blstats[NLE_BL_AC],
           env->blstats[NLE_BL_DEPTH],
           env->blstats[NLE_BL_SCORE],
           env->blstats[NLE_BL_TIME]);
#endif
#if NETHACK_USE_MESSAGE
    printf("Msg: %.*s\n", NLE_MESSAGE_SIZE, env->message);
#endif
    fflush(stdout);
}
