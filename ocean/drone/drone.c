#include "drone.h"
#include "puffernet.h"
#include "render.h"
#include "task_hover.h"
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#ifdef __EMSCRIPTEN__
typedef struct {
    DroneEnv* env;
    PufferNet* net;
} WebRenderArgs;

void emscriptenStep(void* e) {
    WebRenderArgs* args = (WebRenderArgs*)e;
    forward_puffernet(args->net, args->env->observations, args->env->actions);
    c_step(args->env);
    c_render(args->env);
}
#endif

int main() {
    srand(time(NULL));

    DroneEnv* env = calloc(1, sizeof(DroneEnv));
    env->num_agents = 16;
    env->task = &TASK_HOVER;

    env->observations = (float*)calloc(env->num_agents * DRONE_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(env->num_agents * 4, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));

    init(env);

    // task config — hardcoded for demo
    HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
    cfg->target_dist = 5.0f;
    cfg->hover_dist = 0.1f;
    cfg->hover_omega = 0.1f;
    cfg->hover_vel = 0.1f;
    cfg->alpha_dist = 0.782192f;
    cfg->alpha_hover = 0.071445f;
    cfg->alpha_shaping = 3.9754f;
    cfg->alpha_omega = 0.00135588f;
    env->task_config = cfg;

    env->task->init(env);

    c_reset(env);

    Weights* weights = load_weights("resources/drone/drone_weights.bin");
    int logit_sizes[4] = {1, 1, 1, 1};
    PufferNet* net = make_puffernet(weights, env->num_agents, DRONE_OBS_SIZE, 64, 1, logit_sizes, 4);

#ifdef __EMSCRIPTEN__
    WebRenderArgs args = {.env = env, .net = net};
    emscripten_set_main_loop_arg(emscriptenStep, &args, 0, true);
#else
    c_render(env);
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        forward_puffernet(net, env->observations, env->actions);
        c_step(env);
        c_render(env);
    }

    c_close(env);
    free_puffernet(net);
    free(weights);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
#endif

    return 0;
}