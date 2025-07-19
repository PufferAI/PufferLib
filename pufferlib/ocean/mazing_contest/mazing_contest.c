// Standalone C demo for mazing_contest environment
// Compile using: ./scripts/build_ocean.sh mazing_contest [local|fast]
// Run with: ./mazing_contest

#include "mazing_contest.h"
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

void generate_dummy_action(MazingContest* env) {
    env->actions[0] = rand() % TOTAL_ACTIONS;
}

#ifdef __EMSCRIPTEN__
typedef struct {
    MazingContest *env;
} WebRenderArgs;

void emscriptenStep(void *e) {
    WebRenderArgs *args = (WebRenderArgs *)e;
    MazingContest *env = args->env;

    generate_dummy_action(env);
    c_step(env);
    c_render(env);
    return;
}

WebRenderArgs *web_args = NULL;
#endif

int main() {
    srand(time(NULL));

    MazingContest *env = calloc(1, sizeof(MazingContest));
    
    env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    if (!env->observations || !env->actions || !env->rewards || !env->terminals) {
        fprintf(stderr, "ERROR: Failed to allocate memory for demo buffers.\n");
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        free(env);
        return 1;
    }

    env->build_time_limit = 600;
    env->max_moves = 3000000;
    env->max_rounds = 1;
    env->min_gold = 10;
    env->max_gold = 30;
    env->min_lumber = 2;
    env->max_lumber = 4;
    
    init(env);
    allocate(env);
    c_reset(env);

#ifdef __EMSCRIPTEN__
    WebRenderArgs *args = calloc(1, sizeof(WebRenderArgs));
    args->env = env;
    web_args = args;

    emscripten_set_main_loop_arg(emscriptenStep, args, 0, true);
#else
    c_render(env);

    while (!WindowShouldClose()) {
        generate_dummy_action(env);
        c_step(env);
        c_render(env);
    }

    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
#endif

    return 0;
}