#include <time.h>
#include <unistd.h>
#include <string.h>
// For the trajectory recorder we want full obs so the log is useful.
#define NETHACK_USE_BLSTATS 1
#define NETHACK_USE_MESSAGE 1
#define NETHACK_USE_COLORS  1
#include "nethack.h"

// Action index -> printable label for the trajectory log.
// Matches NETHACK_ACTION_TABLE: k j h l y u b n  K J H L Y U B N  > < . s \r ESC ,
static const char* NETHACK_ACTION_NAMES[NETHACK_NUM_ACTIONS] = {
    "N","S","W","E","NW","NE","SW","SE",
    "N_RUN","S_RUN","W_RUN","E_RUN","NW_RUN","NE_RUN","SW_RUN","SE_RUN",
    "DOWN_STAIRS","UP_STAIRS","WAIT(.)","SEARCH(s)","MORE(\\r)","ESC","PICKUP(,)",
};

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void run_interactive(int max_steps) {
    Nethack env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env.actions      = (float*)calloc(1, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));

    init(&env);
    c_reset(&env);
    c_render(&env);

    srand((unsigned)time(NULL));
    for (int t = 0; t < max_steps; t++) {
        env.actions[0] = (float)(rand() % NETHACK_NUM_ACTIONS);
        c_step(&env);
        c_render(&env);
        printf("step=%d action=%d reward=%.2f done=%.0f\n",
               t, (int)env.actions[0], env.rewards[0], env.terminals[0]);
        usleep(50 * 1000);
    }
    c_close(&env);
    free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
}

static void run_benchmark(long steps, int policy) {
    // policy: 0 = random, 1 = always wait '.' (action_idx 18)
    Nethack env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env.actions      = (float*)calloc(1, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));

    init(&env);
    c_reset(&env);
    srand(0xC0FFEE);
    long episodes = 1;
    double sum_reward = 0.0;

    double t0 = now_sec();
    for (long t = 0; t < steps; t++) {
        env.actions[0] = (policy == 1) ? 18.0f : (float)(rand() % NETHACK_NUM_ACTIONS);
        c_step(&env);
        sum_reward += env.rewards[0];
        if (env.terminals[0] > 0.5f) episodes++;
    }
    double dt = now_sec() - t0;

    printf("steps=%ld  episodes=%ld  time=%.3fs  steps/sec=%.0f  sum_reward=%.1f\n",
           steps, episodes, dt, steps / dt, sum_reward);
    printf("  valid_moves=%ld (%.1f%%)  illegal_actions=%ld (%.1f%%)\n",
           env.episode_valid_moves,
           100.0 * env.episode_valid_moves / (double)steps,
           env.episode_illegal_actions,
           100.0 * env.episode_illegal_actions / (double)steps);
#if NETHACK_USE_BLSTATS
    printf("  final HP=%ld/%ld  AC=%ld  Dlvl=%ld  Score=%ld  GameTime=%ld\n",
           env.blstats[NLE_BL_HP], env.blstats[NLE_BL_HPMAX],
           env.blstats[NLE_BL_AC],
           env.blstats[NLE_BL_DEPTH], env.blstats[NLE_BL_SCORE],
           env.blstats[NLE_BL_TIME]);
#endif

    c_close(&env);
    free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
}

// Dump one frame as plain text (chars grid + status + message + action label).
static void dump_frame(FILE* f, Nethack* env, long step, int action_idx, float reward) {
    fprintf(f, "=== step %ld | action=%s | reward=%+.2f | done=%.0f | in_normal_game=%d ===\n",
            step, NETHACK_ACTION_NAMES[action_idx], reward,
            env->terminals[0], env->obs.in_normal_game);
#if NETHACK_USE_BLSTATS
    fprintf(f, "HP %ld/%ld  AC %ld  Dlvl %ld  Score %ld  GameTime %ld  Hunger %ld\n",
            env->blstats[NLE_BL_HP], env->blstats[NLE_BL_HPMAX],
            env->blstats[NLE_BL_AC], env->blstats[NLE_BL_DEPTH],
            env->blstats[NLE_BL_SCORE], env->blstats[NLE_BL_TIME],
            env->blstats[NLE_BL_HUNGER]);
#endif
#if NETHACK_USE_MESSAGE
    fprintf(f, "Msg: %.256s\n", env->message);
#endif
    for (int r = 0; r < NH_ROWS; r++) {
        fputs("  |", f);
        for (int c = 0; c < NH_COLS; c++) {
            unsigned char ch = env->chars[r * NH_COLS + c];
            fputc(ch >= 32 && ch < 127 ? ch : (ch == 0 ? ' ' : '?'), f);
        }
        fputs("|\n", f);
    }
    fputc('\n', f);
}

static void run_record(const char* path, long steps, int policy) {
    FILE* f = (strcmp(path, "-") == 0) ? stdout : fopen(path, "w");
    if (!f) { perror("fopen"); return; }

    Nethack env; memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env.actions      = (float*)calloc(1, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));
    init(&env);
    c_reset(&env);
    dump_frame(f, &env, -1, 0, 0.0f);   // post-reset frame, "action" col is meaningless

    srand(0xC0FFEE);
    for (long t = 0; t < steps; t++) {
        int action_idx = (policy == 1) ? 18 : (rand() % NETHACK_NUM_ACTIONS);
        env.actions[0] = (float)action_idx;
        c_step(&env);
        dump_frame(f, &env, t, action_idx, env.rewards[0]);
        if (env.terminals[0] > 0.5f) {
            fprintf(f, "##### EPISODE END at step %ld #####\n\n", t);
            break;
        }
    }
    c_close(&env);
    if (f != stdout) fclose(f);
    free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
}

// policy: 0=random, 1=wait, 2=move-north (always 'k'), 3=safe cycle
static int policy_action(int policy, long t) {
    switch (policy) {
        case 0:  return rand() % NETHACK_NUM_ACTIONS;
        case 1:  return 18;                                 // WAIT '.'
        case 2:  return 0;                                  // MOVE N 'k'
        case 3: {
            // Cycle through 4 cardinal moves only — never picks risky actions
            // like '>' or ',' that trigger sub-prompts. Should keep character
            // alive longer than random and never trigger illegal-action handling.
            static const int safe[] = {0, 1, 2, 3};  // N S W E
            return safe[t & 3];
        }
        default: return 18;
    }
}

static void run_profile(const char* out_json, long steps, int policy, int seed) {
    Nethack env; memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env.actions      = (float*)calloc(1, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));

    double t0 = now_sec();
    init(&env);
    c_reset(&env);
    srand((unsigned)seed);

    long episodes = 1;
    double sum_reward = 0.0;
    for (long t = 0; t < steps; t++) {
        env.actions[0] = (float)policy_action(policy, t);
        c_step(&env);
        sum_reward += env.rewards[0];
        if (env.terminals[0] > 0.5f) episodes++;
    }
    double dt = now_sec() - t0;

    // Console summary
    printf("profile complete: steps=%ld episodes=%ld wall=%.3fs sum_reward=%.1f\n",
           steps, episodes, dt, sum_reward);

    // JSON dump
    FILE* f = (strcmp(out_json, "-") == 0) ? stdout : fopen(out_json, "w");
    if (!f) { perror("open profile output"); }
    else {
        prof_dump_json(f, dt);
        if (f != stdout) fclose(f);
        printf("wrote %s\n", out_json);
    }

    c_close(&env);
    free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
}

int main(int argc, char** argv) {
    if (argc >= 2 && strcmp(argv[1], "profile") == 0) {
        // Usage: ./nethack profile OUT_JSON [N_STEPS] [random|wait] [SEED]
        const char* out = (argc >= 3) ? argv[2] : "profile.json";
        long steps     = (argc >= 4) ? atol(argv[3]) : 100000;
        int policy     = 0;
        if (argc >= 5) {
            if      (strcmp(argv[4], "wait")   == 0) policy = 1;
            else if (strcmp(argv[4], "north")  == 0) policy = 2;
            else if (strcmp(argv[4], "safe")   == 0) policy = 3;
            else                                     policy = 0;
        }
        int seed       = (argc >= 6) ? atoi(argv[5]) : 0xC0FFEE;
        run_profile(out, steps, policy, seed);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "record") == 0) {
        // Usage: ./nethack record OUTFILE [N_STEPS] [random|wait]
        const char* path = (argc >= 3) ? argv[2] : "trajectory.txt";
        long steps      = (argc >= 4) ? atol(argv[3]) : 100;
        int policy      = (argc >= 5 && strcmp(argv[4], "wait") == 0) ? 1 : 0;
        run_record(path, steps, policy);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "bench") == 0) {
        long steps = (argc >= 3) ? atol(argv[2]) : 100000;
        int policy = (argc >= 4 && strcmp(argv[3], "wait") == 0) ? 1 : 0;
        printf("policy=%s\n", policy ? "wait" : "random");
        run_benchmark(steps, policy);
    } else if (argc >= 2 && strcmp(argv[1], "resets") == 0) {
        long n = (argc >= 3) ? atol(argv[2]) : 50;
        Nethack env; memset(&env, 0, sizeof(env));
        env.num_agents = 1;
        env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
        env.actions      = (float*)calloc(1, sizeof(float));
        env.rewards      = (float*)calloc(1, sizeof(float));
        env.terminals    = (float*)calloc(1, sizeof(float));
        init(&env);
        c_reset(&env);  // warm-up not counted
        double t0 = now_sec();
        for (long i = 0; i < n; i++) c_reset(&env);
        double dt = now_sec() - t0;
        printf("resets=%ld  time=%.3fs  per_reset=%.2fms  resets/sec=%.0f\n",
               n, dt, 1000.0 * dt / n, n / dt);
        c_close(&env);
        free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
    } else if (argc >= 2 && strcmp(argv[1], "stepreset") == 0) {
        // step+reset cycle test. Args: STEPS_PER_EPISODE NUM_RESETS
        long steps_per = (argc >= 3) ? atol(argv[2]) : 100;
        long n_resets  = (argc >= 4) ? atol(argv[3]) : 20;
        Nethack env; memset(&env, 0, sizeof(env));
        env.num_agents = 1;
        env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
        env.actions      = (float*)calloc(1, sizeof(float));
        env.rewards      = (float*)calloc(1, sizeof(float));
        env.terminals    = (float*)calloc(1, sizeof(float));
        init(&env);
        c_reset(&env);
        srand(0xC0FFEE);
        double t0 = now_sec();
        for (long r = 0; r < n_resets; r++) {
            for (long t = 0; t < steps_per; t++) {
                env.actions[0] = (float)(rand() % NETHACK_NUM_ACTIONS);
                c_step(&env);
            }
            c_reset(&env);
        }
        double dt = now_sec() - t0;
        long total = steps_per * n_resets;
        printf("stepreset: %ld cycles x %ld steps = %ld steps  time=%.3fs  sps=%.0f\n",
               n_resets, steps_per, total, dt, total / dt);
        c_close(&env);
        free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
    } else {
        int max_steps = (argc >= 2) ? atoi(argv[1]) : 50;
        run_interactive(max_steps);
    }
    return 0;
}
