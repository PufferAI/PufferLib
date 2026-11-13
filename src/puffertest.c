// puffertest: generic checks for any PufferLib env, built by ./build.sh ENV --test
// (ASan/UBSan) and run by puffer test. Random legal actions over several seeds,
// checking what the trainer assumes: finite data, 0/1 terminals, separate agent
// buffers, episodes that end, determinism. Then the env's tests/*.c (puf_test.h).
// Thresholds come from [test] in config/default.ini; --test.key=value overrides.

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include ENV_HEADER
#include "puf_test.h"

// The trainer reads Log as a flat float array and divides by n.
_Static_assert(sizeof(Log) % sizeof(float) == 0, "Log must hold only floats");
_Static_assert(_Generic(((Log*)0)->n, float: 1, default: 0), "Log.n must be a float");

#define PT_OBS_IS_FLOAT _Generic((obs_t)0, float: 1, default: 0)
// Env.agents is normally Agent agents[N]; a few envs allocate Agent* agents.
#define PT_AGENTS_IS_POINTER _Generic(&((Env*)0)->agents, Agent**: 1, default: 0)
#define PT_MAX_AGENTS ((int)(sizeof(((Env*)0)->agents) / sizeof(Agent)))

// Float obs are filled with this NaN before reset; a value still holding it
// was never written, which is told apart from a NaN the env computed.
static const uint32_t PT_POISON = 0x7FC0DEADu;

static int pt_is_poison(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return bits == PT_POISON;
}

// Envs live for the whole process, so leak reports would only drown the real
// findings; undefined behaviour, on the other hand, fails the run at the line.
const char* __asan_default_options(void) {
    return "detect_leaks=0";
}

const char* __ubsan_default_options(void) {
    return "halt_on_error=1:print_stacktrace=1";
}

static Ini pt_ini = {0};
static int pt_test_contiguous = 0;
static int pt_failures = 0;
static int pt_warnings = 0;

static void pt_fail(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("FAIL ");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
    pt_failures++;
}

static void pt_warn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("warn ");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
    pt_warnings++;
}

// Action stream RNG, separate from the env's so both are reproducible.
static uint32_t pt_rng_state = 1;

static uint32_t pt_rnd(void) {
    uint32_t x = pt_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    pt_rng_state = x;
    return x;
}

typedef struct {
    Env* env;
    unsigned int seed;
    int num_agents;
    int act_n;        // sum of ACT_SIZES: mask bytes per agent
    int continuous;   // every head has size 1
    void* slabs[5];   // one slab per buffer kind when [test] contiguous = 1
    int step;
    long terminals;
    uint64_t hash;    // digest of the last step's obs, rewards and terminals
    int mask_failed, finite_failed, terminal_failed, log_failed;
} PtEnv;

static const int pt_act_sizes[] = ACT_SIZES;
static const int pt_num_heads = sizeof(pt_act_sizes) / sizeof(pt_act_sizes[0]);

static long pt_steps_total = 0;
static double pt_episodes_logged = 0;  // sum of final log.n over runs
static unsigned char pt_obs_written[OBS_SIZE];
static int pt_reset_unwritten = -1;    // poison left after the first reset

// Trainer-style agent buffers, then puf_init and puf_reset; float obs poisoned first.
static void* pt_agent_block(PtEnv* t, int kind, int agent, size_t size) {
    if (!pt_test_contiguous) {
        return calloc(1, size);
    }
    if (agent == 0) {
        t->slabs[kind] = calloc(1, size * t->num_agents);
    }
    return (unsigned char*)t->slabs[kind] + size * agent;
}

static PtEnv pt_make_env(unsigned int seed) {
    PtEnv t = {0};
    t.seed = seed;
    t.env = calloc(1, sizeof(Env));
    t.env->rng = seed;
    srand(seed);
    puf_init(t.env, puf_ini_section(&pt_ini, "env", 0));
    t.num_agents = t.env->num_agents;
    if (t.num_agents < 1 || (!PT_AGENTS_IS_POINTER && t.num_agents > PT_MAX_AGENTS)) {
        pt_fail("num_agents is %d after puf_init; the agents array holds %d",
            t.num_agents, PT_MAX_AGENTS);
        exit(1);
    }
    t.continuous = 1;
    for (int h = 0; h < pt_num_heads; h++) {
        t.act_n += pt_act_sizes[h];
        if (pt_act_sizes[h] != 1) {
            t.continuous = 0;
        }
    }
    for (int i = 0; i < t.num_agents; i++) {
        Agent* a = &t.env->agents[i];
        a->observations = pt_agent_block(&t, 0, i, OBS_SIZE * sizeof(obs_t));
        a->actions = pt_agent_block(&t, 1, i, NUM_ATNS * sizeof(float));
        a->rewards = pt_agent_block(&t, 2, i, sizeof(float));
        a->terminals = pt_agent_block(&t, 3, i, sizeof(float));
        a->action_mask = pt_agent_block(&t, 4, i, t.act_n);
        memset(a->action_mask, 1, t.act_n);
        a->policy = 0;
        if (PT_OBS_IS_FLOAT) {
            for (int j = 0; j < OBS_SIZE; j++) {
                memcpy(&a->observations[j], &PT_POISON, sizeof(PT_POISON));
            }
        }
    }
    puf_reset(t.env);
    if (PT_OBS_IS_FLOAT && pt_reset_unwritten < 0) {
        pt_reset_unwritten = 0;
        for (int i = 0; i < t.num_agents; i++) {
            for (int j = 0; j < OBS_SIZE; j++) {
                pt_reset_unwritten += pt_is_poison((float)t.env->agents[i].observations[j]);
            }
        }
    }
    return t;
}

// puf_close may write the buffers (impulse_wars) and may free an Agent* agents
// array (nmmo3), so the buffer pointers are taken first and freed last.
static void pt_free_env(PtEnv* t) {
    int n = t->num_agents;
    void** bufs = calloc(5 * n, sizeof(void*));
    for (int i = 0; i < n; i++) {
        Agent* a = &t->env->agents[i];
        void* p[5] = {a->observations, a->actions, a->rewards, a->terminals, a->action_mask};
        memcpy(bufs + 5 * i, p, sizeof(p));
    }
    puf_close(t->env);
    // Contiguous: agent 0's five pointers are the slab bases.
    for (int k = 0; k < (pt_test_contiguous ? 5 : 5 * n); k++) {
        free(bufs[k]);
    }
    free(bufs);
    free(t->env);
}

// Uniform over the legal entries of each head; continuous heads get -1..1.
// Returns 0 when a head has no legal action.
static int pt_sample_actions(PtEnv* t, int agent) {
    Agent* a = &t->env->agents[agent];
    if (t->continuous) {
        for (int h = 0; h < pt_num_heads; h++) {
            a->actions[h] = (float)(pt_rnd() % 20001) / 10000.0f - 1.0f;
        }
        return 1;
    }
    int ok = 1;
    int offset = 0;
    for (int h = 0; h < pt_num_heads; h++) {
        int legal = 0;
        for (int k = 0; k < pt_act_sizes[h]; k++) {
            legal += a->action_mask[offset + k] != 0;
        }
        if (legal == 0) {
            a->actions[h] = 0;
            ok = 0;
        } else {
            int pick = pt_rnd() % legal;
            for (int k = 0; k < pt_act_sizes[h]; k++) {
                if (a->action_mask[offset + k] && pick-- == 0) {
                    a->actions[h] = (float)k;
                    break;
                }
            }
        }
        offset += pt_act_sizes[h];
    }
    return ok;
}

static uint64_t pt_fnv(uint64_t h, const void* data, size_t n) {
    const unsigned char* p = data;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ p[i]) * 1099511628211ULL;
    }
    return h;
}

// One puf_step with rewards and terminals cleared first, as the trainer does,
// then every per-step check; each failure kind reports once per run.
static void pt_step(PtEnv* t) {
    Env* env = t->env;
    unsigned int seed = t->seed;
    int s = t->step;
    for (int i = 0; i < t->num_agents; i++) {
        env->agents[i].rewards[0] = 0;
        env->agents[i].terminals[0] = 0;
    }
    puf_step(env);
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < t->num_agents; i++) {
        Agent* a = &env->agents[i];
        float r = a->rewards[0];
        float d = a->terminals[0];
        t->terminals += d != 0;
        if (!isfinite(r) && !t->finite_failed) {
            pt_fail("seed %u step %d: agent %d reward is not finite (%g)", seed, s, i, r);
            t->finite_failed = 1;
        }
        if (!(d == 0 || d == 1) && !t->terminal_failed) {
            pt_fail("seed %u step %d: agent %d terminal is %g, not 0 or 1", seed, s, i, d);
            t->terminal_failed = 1;
        }
        for (int j = 0; j < OBS_SIZE; j++) {
            float v = (float)a->observations[j];
            if (PT_OBS_IS_FLOAT && pt_is_poison(v)) {
                continue;
            }
            if (PT_OBS_IS_FLOAT && !isfinite(v)) {
                if (!t->finite_failed) {
                    pt_fail("seed %u step %d: agent %d obs[%d] is not finite (%g)",
                        seed, s, i, j, v);
                    t->finite_failed = 1;
                }
                continue;
            }
            pt_obs_written[j] = 1;
        }
        h = pt_fnv(h, a->observations, OBS_SIZE * sizeof(obs_t));
        h = pt_fnv(h, &r, sizeof(r));
        h = pt_fnv(h, &d, sizeof(d));
    }
    t->hash = h;
    float* lf = (float*)&env->log;
    int nlog = sizeof(Log) / sizeof(float);
    // A real float stat is never subnormal; a small int read as a float is.
    for (int k = 0; k < nlog && !t->log_failed; k++) {
        if (!isfinite(lf[k])) {
            pt_fail("seed %u step %d: Log field %d is not finite", seed, s, k);
            t->log_failed = 1;
        } else if (fpclassify(lf[k]) == FP_SUBNORMAL) {
            pt_fail("seed %u step %d: Log field %d reads as %g; an int in Log? (floats only)",
                seed, s, k, lf[k]);
            t->log_failed = 1;
        }
    }
    t->step++;
    pt_steps_total++;
}

// One run of [test] steps with random actions. Fills hashes[step] when given.
// Returns terminals seen.
static long pt_run(unsigned int seed, int steps, uint64_t* hashes) {
    pt_rng_state = seed * 2654435761u + 1;
    PtEnv t = pt_make_env(seed);
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < t.num_agents; i++) {
            if (!pt_sample_actions(&t, i) && !t.mask_failed) {
                pt_fail("seed %u step %d: action mask of agent %d rules out every action of a head",
                    seed, s, i);
                t.mask_failed = 1;
            }
        }
        pt_step(&t);
        if (hashes) {
            hashes[s] = t.hash;
        }
    }
    float n = t.env->log.n;
    if (t.terminals > 0 && n <= 0) {
        pt_warn("seed %u: %ld terminals but log.n is still 0; no episode was logged in %d steps",
            seed, t.terminals, steps);
    }
    pt_episodes_logged += n;
    long terminals = t.terminals;
    pt_free_env(&t);
    return terminals;
}

// Env-specific tests (puf_test.h). Tests register themselves at load time.
#define PT_MAX_TESTS 256
#define PT_MAX_LIVE 16

static struct {
    const char* name;
    PufTestFn fn;
} pt_tests[PT_MAX_TESTS];
static int pt_num_tests = 0;
static const char* pt_current_test = "";
static PtEnv pt_live[PT_MAX_LIVE];

void puf_test_register(const char* name, PufTestFn fn) {
    if (pt_num_tests == PT_MAX_TESTS) {
        fprintf(stderr, "puffertest: more than %d tests\n", PT_MAX_TESTS);
        exit(1);
    }
    pt_tests[pt_num_tests].name = name;
    pt_tests[pt_num_tests].fn = fn;
    pt_num_tests++;
}

void puf_test_fail(const char* file, int line, const char* what) {
    printf("FAIL %s: %s:%d: %s\n", pt_current_test, file, line, what);
    pt_failures++;
}

static PtEnv* pt_live_env(Env* env) {
    for (int i = 0; i < PT_MAX_LIVE; i++) {
        if (pt_live[i].env == env) {
            return &pt_live[i];
        }
    }
    fprintf(stderr, "%s: not an env from puf_test_env\n", pt_current_test);
    exit(1);
}

Env* puf_test_env(unsigned int seed) {
    for (int i = 0; i < PT_MAX_LIVE; i++) {
        if (!pt_live[i].env) {
            pt_live[i] = pt_make_env(seed);
            return pt_live[i].env;
        }
    }
    fprintf(stderr, "%s: more than %d envs open; call puf_test_close\n", pt_current_test,
        PT_MAX_LIVE);
    exit(1);
}

void puf_test_step(Env* env) {
    pt_step(pt_live_env(env));
}

void puf_test_close(Env* env) {
    PtEnv* t = pt_live_env(env);
    pt_free_env(t);
    memset(t, 0, sizeof(*t));
}

int main(int argc, char** argv) {
    puf_ini_load_env(&pt_ini, PUFFER_ENV_NAME, argc - 1, argv + 1);
    Dict* cfg = puf_ini_section(&pt_ini, "test", 0);
    int seeds = dict_get(cfg, "seeds");
    int steps = dict_get(cfg, "steps");
    int min_episodes = dict_get(cfg, "min_episodes");
    pt_test_contiguous = dict_get(cfg, "contiguous");
    unsigned int base_seed = puf_ini_get(&pt_ini, "base", "seed");
    printf("puffertest %s: %d seeds x %d steps, %d action heads, %s obs x %d\n",
        PUFFER_ENV_NAME, seeds, steps, pt_num_heads,
        PT_OBS_IS_FLOAT ? "float" : "integer", OBS_SIZE);

    long terminals = 0;
    uint64_t* first = calloc(steps, sizeof(uint64_t));
    for (int i = 0; i < seeds; i++) {
        terminals += pt_run(base_seed + i, steps, i == 0 ? first : NULL);
    }
    // Envs that respawn agents in place (snake) log episodes without terminals.
    if (min_episodes > 0 && terminals < min_episodes && pt_episodes_logged < min_episodes) {
        pt_fail("no episode ended: %ld terminals, log.n %g in %d x %d steps; the env must reset "
            "itself (or raise [test] steps / set min_episodes = 0)",
            terminals, pt_episodes_logged, seeds, steps);
    } else {
        printf("ok   %ld terminals, log.n %g, %ld steps\n", terminals, pt_episodes_logged,
            pt_steps_total);
    }

    uint64_t* again = calloc(steps, sizeof(uint64_t));
    pt_run(base_seed, steps, again);
    int diverged = -1;
    for (int s = 0; s < steps; s++) {
        if (first[s] != again[s]) {
            diverged = s;
            break;
        }
    }
    if (diverged >= 0) {
        pt_fail("seed %u is not deterministic: trajectories diverge at step %d "
            "(static state, uninitialised memory, or an RNG other than env->rng?)",
            base_seed, diverged);
    } else {
        printf("ok   deterministic for seed %u\n", base_seed);
    }
    free(again);
    free(first);

    int never = 0;
    for (int j = 0; j < OBS_SIZE; j++) {
        never += !pt_obs_written[j];
    }
    if (pt_reset_unwritten > 0) {
        pt_warn("reset leaves %d obs values unwritten; the first policy input is whatever the "
            "buffer held", pt_reset_unwritten);
    }
    if (never > 0) {
        pt_warn("%d of %d obs columns are never written (always 0 in the trainer)", never,
            OBS_SIZE);
    }

    for (int i = 0; i < pt_num_tests; i++) {
        pt_current_test = pt_tests[i].name;
        int failures = pt_failures;
        pt_tests[i].fn();
        for (int k = 0; k < PT_MAX_LIVE; k++) {
            if (pt_live[k].env) {
                puf_test_close(pt_live[k].env);
            }
        }
        if (pt_failures == failures) {
            printf("ok   %s\n", pt_tests[i].name);
        }
    }
    printf("puffertest %s: %d env tests, %d failed, %d warnings, %s\n", PUFFER_ENV_NAME,
        pt_num_tests, pt_failures, pt_warnings, pt_failures ? "FAILED" : "passed");
    puf_ini_free(&pt_ini);
    return pt_failures != 0;
}

#ifdef PUF_TEST_INCLUDE
#include PUF_TEST_INCLUDE
#endif
