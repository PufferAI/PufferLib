/* MIT, Copyright (c) 2026 Marc / Rage Industries.
 * Live float64 demo and headless evaluator for the frozen hybrid controller.
 */
#include <errno.h>
#include <limits.h>
#include "npole.h"
#include "npole_controller.h"
#ifdef PLATFORM_WEB
#include <emscripten.h>
#endif

static NpModel *demo_model;
static NpEpisode demo_episode;
static NpPolicy demo_policy;
static uint32_t demo_seed = 2026101401u;
static double demo_command;
static double demo_manual_force;
static int demo_manual;
static int demo_finished;
static int demo_failed;
static int demo_paused;

void np_demo_reset(uint32_t seed) {
    demo_seed = seed;
    np_reset_seed(&demo_episode, seed);
    memset(&demo_policy, 0, sizeof(demo_policy));
    demo_command = 0.0;
    demo_manual = 0;
    demo_finished = 0;
    demo_failed = 0;
    demo_paused = 0;
}

void np_demo_toggle_pause(void) {
    demo_paused = !demo_paused;
}

void np_demo_set_manual(double force) {
    if (isfinite(force)) {
        demo_manual_force = np_clip(force);
        demo_manual = 1;
    }
}

void np_demo_clear_manual(void) {
    demo_manual = 0;
}

int np_demo_advance(int ticks) {
    if (!demo_model || ticks < 0 || ticks > NP_HORIZON) {
        return -1;
    }
    for (int i = 0; i < ticks && !demo_finished && !demo_paused; i++) {
        if (demo_manual) {
            demo_command = demo_manual_force;
            demo_policy.tick++;
        } else if (np_policy_act(demo_model, &demo_policy,
                &demo_episode.state, &demo_command)) {
            demo_episode.numeric_ok = 0;
            demo_episode.hold = 0;
            demo_finished = 1;
            demo_failed = 1;
            break;
        }
        if (np_episode_tick(&demo_episode, demo_command)) {
            demo_finished = 1;
            demo_failed = 1;
        } else if (demo_episode.state.tick == NP_HORIZON) {
            demo_finished = 1;
            demo_failed = !np_success(&demo_episode);
        }
    }
    return demo_episode.state.tick;
}

double np_demo_state(int index) {
    return index >= 0 && index < NP_STATE ? demo_episode.state.y[index] : NAN;
}

double np_demo_estimate(int index) {
    return index >= 0 && index < NP_STATE ? demo_policy.estimate[index] : NAN;
}

int np_demo_tick(void) {
    return demo_episode.state.tick;
}

double np_demo_seed_value(void) {
    return (double)demo_seed;
}

int np_demo_ready(void) {
    return demo_model != NULL;
}

int np_demo_hold(void) {
    return demo_episode.hold;
}

int np_demo_success(void) {
    return np_success(&demo_episode);
}

int np_demo_failed(void) {
    return demo_failed;
}

int np_demo_paused(void) {
    return demo_paused;
}

double np_demo_command(void) {
    return demo_command;
}

double np_demo_applied(void) {
    return demo_episode.state.last_applied;
}

double np_demo_pending(void) {
    return demo_episode.state.pending;
}

double np_demo_residual(void) {
    return demo_policy.residual;
}

#ifndef NP_HEADLESS
static void np_demo_frame(void) {
    static double previous;
    static double accumulator;
    static double finished_at;
    static int keyboard_manual;
    double now = GetTime();
    double elapsed = previous == 0.0 ? 0.0 : now - previous;
    previous = now;
    if (IsKeyPressed(KEY_SPACE)) {
        np_demo_toggle_pause();
    }
    if (IsKeyPressed(KEY_R)) {
        np_demo_reset(demo_seed + 1u);
    }
    int left = IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A);
    int right = IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D);
    if (IsKeyDown(KEY_LEFT_SHIFT) && (left || right)) {
        np_demo_set_manual(right ? 1.0 : -1.0);
        keyboard_manual = 1;
    } else if (keyboard_manual) {
        np_demo_clear_manual();
        keyboard_manual = 0;
    }
    if (!demo_paused && !demo_finished) {
        accumulator = fmin(accumulator + fmax(0.0, elapsed), 5.0 * NP_DT);
        while (accumulator >= NP_DT && !demo_finished) {
            np_demo_advance(1);
            accumulator -= NP_DT;
        }
    } else {
        accumulator = 0.0;
    }
    if (demo_finished) {
        if (finished_at == 0.0) {
            finished_at = now;
        }
        if (!demo_paused && now - finished_at >= 3.0) {
            np_demo_reset(demo_seed + 1u);
        }
    } else {
        finished_at = 0.0;
    }
    const char *status = demo_finished
        ? (demo_failed ? "Trial failed - resetting shortly" : "Swing up and final hold passed")
        : (demo_paused ? "Paused" : (demo_manual ? "Manual cart force" : "Live swing up and balance"));
    np_draw(&demo_episode, status, demo_seed,
        "SPACE pause   R reset   SHIFT + arrows take control");
}
#endif

#ifdef PLATFORM_WEB
int np_demo_init(void) {
    if (demo_model) {
        return 0;
    }
    demo_model = np_load_model("resources/npole/npole_controller.bin");
    if (!demo_model) {
        return 1;
    }
    np_demo_reset(demo_seed);
    InitWindow(NP_WIDTH, NP_HEIGHT, "Ten Pendulums");
    emscripten_set_main_loop(np_demo_frame, 0, 0);
    return 0;
}
#else
static void np_csv_row(FILE *file, uint32_t seed) {
    const NpState *state = &demo_episode.state;
    fprintf(file, "%u,%d,%.17g,%.17g,%.17g,%.17g", seed, state->tick,
        demo_command, state->last_applied, state->pending, demo_policy.residual);
    for (int i = 0; i < NP_STATE; i++) {
        fprintf(file, ",%.17g", state->y[i]);
    }
    fprintf(file, "\n");
}

int main(int argc, char **argv) {
    const char *model_path = "resources/npole/npole_controller.bin";
    const char *record_path = NULL;
    int headless = 0;
    int episodes = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--headless")) {
            headless = 1;
        } else if (!strcmp(argv[i], "--help")) {
            puts("npole [--model FILE] [--headless --episodes N --seed N --record CSV]");
            return 0;
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--record") && i + 1 < argc) {
            record_path = argv[++i];
        } else if ((!strcmp(argv[i], "--seed") || !strcmp(argv[i], "--episodes")) && i + 1 < argc) {
            int is_seed = !strcmp(argv[i], "--seed");
            char *end;
            errno = 0;
            unsigned long long number = strtoull(argv[++i], &end, 10);
            if (errno || *end || !argv[i][0] || number > UINT32_MAX
                || (!is_seed && (number < 1 || number > 1024))) {
                fprintf(stderr, "Invalid numeric argument\n");
                return 1;
            }
            if (is_seed) {
                demo_seed = (uint32_t)number;
            } else {
                episodes = (int)number;
            }
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            return 1;
        }
    }
    demo_model = np_load_model(model_path);
    if (!demo_model) {
        fprintf(stderr, "Missing or invalid NP10v001 controller: %s\n", model_path);
        return 1;
    }
    np_demo_reset(demo_seed);
    if (headless) {
        FILE *record = record_path ? fopen(record_path, "w") : NULL;
        if (record_path && !record) {
            perror("Opening recording");
            free(demo_model);
            return 1;
        }
        if (record) {
            fprintf(record, "seed,tick,command,applied,pending,inverse_residual");
            for (int i = 0; i < NP_STATE; i++) {
                fprintf(record, ",state_%d", i);
            }
            fprintf(record, "\n");
        }
        int successes = 0;
        uint32_t first_seed = demo_seed;
        for (int episode = 0; episode < episodes; episode++) {
            np_demo_reset(first_seed + (uint32_t)episode);
            if (record) {
                np_csv_row(record, demo_seed);
            }
            while (!demo_finished) {
                np_demo_advance(1);
                if (record) {
                    np_csv_row(record, demo_seed);
                }
            }
            int success = np_demo_success();
            successes += success;
            printf("NP_TRIAL {\"seed\":%u,\"success\":%s,\"ticks\":%d,\"hold_ticks\":%d,\"rail_ok\":%s,\"numeric_ok\":%s}\n",
                demo_seed, success ? "true" : "false", demo_episode.state.tick,
                demo_episode.hold, demo_episode.rail_ok ? "true" : "false",
                demo_episode.numeric_ok ? "true" : "false");
        }
        if (record) {
            fclose(record);
        }
        printf("NP_EVAL {\"episodes\":%d,\"successes\":%d,\"control_hz\":50,\"physics_hz\":1000,\"delay_ms\":20}\n",
            episodes, successes);
        free(demo_model);
        return successes == episodes ? 0 : 2;
    }
#ifdef NP_HEADLESS
    fprintf(stderr, "This executable requires --headless\n");
    free(demo_model);
    return 1;
#else
    InitWindow(NP_WIDTH, NP_HEIGHT, "Ten Pendulums");
    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        np_demo_frame();
    }
    CloseWindow();
    free(demo_model);
    return 0;
#endif
}
#endif
