// Profiler for the NetHack env. Gated by -DNETHACK_PROFILE=1 so production
// builds pay no cost.
//
// Design:
//   * monotonic-clock nanoseconds, captured via clock_gettime(CLOCK_MONOTONIC)
//   * one struct NethackProfile per env (typically singleton during profiling)
//   * each metric tracks count + sum_ns + min/max + log2 histogram (24 buckets,
//     covers 1ns .. 16s)
//   * counters for fn_step calls (split agent vs drain vs reset-drain)
//   * JSON dump via prof_dump_json(path)
//
// Zero allocation in the hot path; the profile struct is allocated once and
// reused. Histograms are unsigned long, so 50M+ samples per bucket fit fine.

#ifndef NETHACK_PROFILE_H
#define NETHACK_PROFILE_H

#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifndef NETHACK_PROFILE
#define NETHACK_PROFILE 0
#endif

#define PROF_HIST_BUCKETS 24   // bucket k covers [2^k, 2^(k+1)) ns

typedef struct ProfMetric {
    const char* name;
    unsigned long count;
    unsigned long long sum_ns;
    unsigned long long min_ns;   // 0 means unset
    unsigned long long max_ns;
    unsigned long      hist[PROF_HIST_BUCKETS];
} ProfMetric;

// Categories of c_step phases we time.
typedef enum {
    PROF_C_STEP_TOTAL,         // whole c_step
    PROF_AGENT_FN_STEP,        // the one fn_step that consumes the agent action
    PROF_POST_DRAIN,            // fn_step calls after the agent action (xwait or ESC)
    PROF_OBS_PACK,             // memcpy obs fields into the tensor
    PROF_C_RESET_TOTAL,        // whole c_reset
    PROF_RESET_NLE_END,        // fn_end + dlclose
    PROF_RESET_RELOAD,         // open + memfd_create + copy + dlopen + dlsym
    PROF_RESET_VARDIR,         // mkdtemp + symlink + touch
    PROF_RESET_NLE_START,      // fn_start (NetHack init + first yield)
    PROF_RESET_DRAIN,          // welcome-screen drain after nle_start
    PROF_METRIC_COUNT,
} ProfMetricId;

static const char* PROF_METRIC_NAMES[PROF_METRIC_COUNT] = {
    "c_step_total",
    "agent_fn_step",
    "post_drain",
    "obs_pack",
    "c_reset_total",
    "reset_nle_end",
    "reset_reload",
    "reset_vardir",
    "reset_nle_start",
    "reset_drain",
};

// Counters (simple long, no histogram).
typedef enum {
    PROF_FN_STEPS_TOTAL,       // every fn_step call regardless of phase
    PROF_FN_STEPS_AGENT,
    PROF_FN_STEPS_POST_DRAIN,
    PROF_FN_STEPS_RESET_DRAIN,
    PROF_C_STEPS,
    PROF_C_RESETS,
    PROF_VALID_MOVES,
    PROF_ILLEGAL_ACTIONS,
    PROF_MSG_PROMPT_ESC,        // ESCed due to message-ending '?'
    PROF_MISC_YN_ESC,           // ESCed due to in_yn_function
    PROF_MISC_GETLIN_ESC,       // ESCed due to in_getlin
    PROF_XWAIT_DRAIN,           // \r sent to dismiss --More--
    PROF_AGENT_FN_STEPS_PER_C_STEP_SUM,  // reserved
    // No-time-advance diagnostics. After the agent's fn_step, if time did
    // not advance, classify what we found:
    PROF_NOADV_PROMPT_DETECTED, // misc flags or message-'?' -> illegal handled elsewhere
    PROF_NOADV_HAS_MSG,         // no detected prompt, but a message was shown
                                // (e.g. "It's a wall.", "There is nothing here.")
    PROF_NOADV_SILENT,          // no prompt, no message — action was an outright no-op
    PROF_COUNTER_COUNT,
} ProfCounterId;

static const char* PROF_COUNTER_NAMES[PROF_COUNTER_COUNT] = {
    "fn_steps_total",
    "fn_steps_agent",
    "fn_steps_post_drain",
    "fn_steps_reset_drain",
    "c_steps",
    "c_resets",
    "valid_moves",
    "illegal_actions",
    "msg_prompt_esc",
    "misc_yn_esc",
    "misc_getlin_esc",
    "xwait_drain",
    "agent_fn_steps_sum",
    "noadv_prompt_detected",
    "noadv_has_msg",
    "noadv_silent",
};

typedef struct NethackProfile {
    ProfMetric metrics[PROF_METRIC_COUNT];
    unsigned long long counters[PROF_COUNTER_COUNT];
    unsigned long fn_steps_per_c_step_hist[PROF_HIST_BUCKETS];
    int initialized;
} NethackProfile;

// Single global profile instance. We always allocate it (small: ~5KB) but the
// hot-path macros below only touch it if NETHACK_PROFILE=1.
static NethackProfile g_prof;

static inline void prof_init(void) {
    memset(&g_prof, 0, sizeof(g_prof));
    for (int i = 0; i < PROF_METRIC_COUNT; i++) {
        g_prof.metrics[i].name = PROF_METRIC_NAMES[i];
        g_prof.metrics[i].min_ns = 0;
    }
    g_prof.initialized = 1;
}

static inline unsigned long long prof_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
}

static inline int prof_log2_bucket(unsigned long long ns) {
    if (ns == 0) return 0;
    int b = 0;
    while (ns >= 2 && b < PROF_HIST_BUCKETS - 1) { ns >>= 1; b++; }
    return b;
}

static inline void prof_record(ProfMetricId id, unsigned long long ns) {
    ProfMetric* m = &g_prof.metrics[id];
    m->count++;
    m->sum_ns += ns;
    if (m->min_ns == 0 || ns < m->min_ns) m->min_ns = ns;
    if (ns > m->max_ns) m->max_ns = ns;
    m->hist[prof_log2_bucket(ns)]++;
}

static inline void prof_count(ProfCounterId id, unsigned long long n) {
    g_prof.counters[id] += n;
}

static inline void prof_record_fn_steps_per_c_step(unsigned long n) {
    g_prof.fn_steps_per_c_step_hist[prof_log2_bucket(n ? n : 1)]++;
}

// ---- gated macros for hot path ----
#if NETHACK_PROFILE
#define PROF_START(name) unsigned long long _pt_##name = prof_now_ns()
#define PROF_END(name, id) prof_record(id, prof_now_ns() - _pt_##name)
#define PROF_COUNT(id, n) prof_count(id, (unsigned long long)(n))
#define PROF_FN_STEPS_PER_C_STEP(n) prof_record_fn_steps_per_c_step(n)
#define PROF_INIT_IF_NEEDED() do { if (!g_prof.initialized) prof_init(); } while (0)
#else
#define PROF_START(name) ((void)0)
#define PROF_END(name, id) ((void)0)
#define PROF_COUNT(id, n) ((void)0)
#define PROF_FN_STEPS_PER_C_STEP(n) ((void)0)
#define PROF_INIT_IF_NEEDED() ((void)0)
#endif

// ---- JSON dump ----
static void prof_dump_json(FILE* f, double wall_seconds) {
    fprintf(f, "{\n");
    fprintf(f, "  \"wall_seconds\": %.6f,\n", wall_seconds);
    fprintf(f, "  \"counters\": {\n");
    for (int i = 0; i < PROF_COUNTER_COUNT; i++) {
        fprintf(f, "    \"%s\": %llu%s\n", PROF_COUNTER_NAMES[i],
                g_prof.counters[i], (i == PROF_COUNTER_COUNT - 1) ? "" : ",");
    }
    fprintf(f, "  },\n");

    // Derived
    double secs = wall_seconds > 0 ? wall_seconds : 1.0;
    fprintf(f, "  \"derived\": {\n");
    fprintf(f, "    \"c_steps_per_sec\": %.1f,\n",
            g_prof.counters[PROF_C_STEPS] / secs);
    fprintf(f, "    \"valid_moves_per_sec\": %.1f,\n",
            g_prof.counters[PROF_VALID_MOVES] / secs);
    fprintf(f, "    \"valid_moves_per_c_step\": %.4f,\n",
            g_prof.counters[PROF_C_STEPS] ?
              (double)g_prof.counters[PROF_VALID_MOVES] / g_prof.counters[PROF_C_STEPS] : 0.0);
    fprintf(f, "    \"illegal_per_c_step\": %.4f,\n",
            g_prof.counters[PROF_C_STEPS] ?
              (double)g_prof.counters[PROF_ILLEGAL_ACTIONS] / g_prof.counters[PROF_C_STEPS] : 0.0);
    fprintf(f, "    \"fn_steps_per_c_step\": %.3f,\n",
            g_prof.counters[PROF_C_STEPS] ?
              (double)(g_prof.counters[PROF_FN_STEPS_AGENT] +
                       g_prof.counters[PROF_FN_STEPS_POST_DRAIN]) /
              g_prof.counters[PROF_C_STEPS] : 0.0);
    fprintf(f, "    \"reset_p50_ms\": null,\n");
    fprintf(f, "    \"ns_per_valid_move\": %.1f\n",
            g_prof.counters[PROF_VALID_MOVES] ?
              wall_seconds * 1e9 / g_prof.counters[PROF_VALID_MOVES] : 0.0);
    fprintf(f, "  },\n");

    fprintf(f, "  \"metrics\": {\n");
    for (int i = 0; i < PROF_METRIC_COUNT; i++) {
        ProfMetric* m = &g_prof.metrics[i];
        double mean = m->count ? (double)m->sum_ns / m->count : 0.0;
        fprintf(f, "    \"%s\": {", m->name);
        fprintf(f, "\"count\": %lu, ", m->count);
        fprintf(f, "\"sum_ns\": %llu, ", m->sum_ns);
        fprintf(f, "\"mean_ns\": %.1f, ", mean);
        fprintf(f, "\"min_ns\": %llu, ", m->min_ns);
        fprintf(f, "\"max_ns\": %llu, ", m->max_ns);
        fprintf(f, "\"hist\": [");
        for (int b = 0; b < PROF_HIST_BUCKETS; b++) {
            fprintf(f, "%lu%s", m->hist[b], (b == PROF_HIST_BUCKETS - 1) ? "" : ",");
        }
        fprintf(f, "]}%s\n", (i == PROF_METRIC_COUNT - 1) ? "" : ",");
    }
    fprintf(f, "  },\n");

    fprintf(f, "  \"fn_steps_per_c_step_hist\": [");
    for (int b = 0; b < PROF_HIST_BUCKETS; b++) {
        fprintf(f, "%lu%s", g_prof.fn_steps_per_c_step_hist[b],
                (b == PROF_HIST_BUCKETS - 1) ? "" : ",");
    }
    fprintf(f, "],\n");

    fprintf(f, "  \"hist_buckets_ns\": [");
    for (int b = 0; b < PROF_HIST_BUCKETS; b++) {
        fprintf(f, "%llu%s", (1ull << b),
                (b == PROF_HIST_BUCKETS - 1) ? "" : ",");
    }
    fprintf(f, "]\n");
    fprintf(f, "}\n");
}

#endif // NETHACK_PROFILE_H
