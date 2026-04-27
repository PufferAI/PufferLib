#include "drone.h"
#include "puffernet.h"
#include "render.h"
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#ifdef __EMSCRIPTEN__
typedef struct {
    DroneEnv* env;
    PufferNet* net;
    Weights* weights;
} WebRenderArgs;

void emscriptenStep(void* e) {
    WebRenderArgs* args = (WebRenderArgs*)e;
    DroneEnv* env = args->env;
    PufferNet* net = args->net;

    forward_puffernet(net, env->observations, env->actions);
    c_step(env);
    c_render(env);
}

WebRenderArgs* web_args = NULL;
#endif

int main() {
    srand(time(NULL));

    DroneEnv* env = calloc(1, sizeof(DroneEnv));
    size_t obs_size = 23;

    env->num_agents = 16;
    env->max_rings = 10;
    env->task = HOVER;
    env->alpha_dist = 0.782192f;
    env->alpha_hover = 0.071445f;
    env->alpha_shaping = 3.9754f;
    env->alpha_omega = 0.00135588f;
    env->hover_target_dist = 5.0f;
    env->hover_dist = 0.1f;
    env->hover_omega = 0.1f;
    env->hover_vel = 0.1f;

    env->observations = (float*)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float*)calloc(env->num_agents * 4, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));

    Weights* weights = load_weights("resources/drone/drone_weights.bin");
    int logit_sizes[4] = {1, 1, 1, 1};
    // make_puffernet(weights, num_agents, obs_size, hidden_size, num_layers, logit_sizes, num_actions)
    PufferNet* net = make_puffernet(weights, env->num_agents, obs_size, 128, 3, logit_sizes, 4);

    init(env);
    c_reset(env);

#ifdef __EMSCRIPTEN__
    WebRenderArgs* args = calloc(1, sizeof(WebRenderArgs));
    args->env = env;
    args->net = net;
    args->weights = weights;
    web_args = args;

    emscripten_set_main_loop_arg(emscriptenStep, args, 0, true);
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
