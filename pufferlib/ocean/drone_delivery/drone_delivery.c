#include "drone_delivery.h"
#include "puffernet.h"
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

double randn(double mean, double std) {
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    }

    has_spare = 1;
    double u, v, s;
    do {
        u = 2.0 * rand() / RAND_MAX - 1.0;
        v = 2.0 * rand() / RAND_MAX - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

typedef struct LinearContLSTM LinearContLSTM;
struct LinearContLSTM {
    int num_agents;
    float *obs;
    float *log_std;
    Linear *encoder;
    GELU *gelu1;
    LSTM *lstm;
    Linear *actor;
    Linear *value_fn;
    int num_actions;
};

LinearContLSTM *make_linearcontlstm(Weights *weights, int num_agents, int input_dim,
                                    int logit_sizes[], int num_actions) {
    LinearContLSTM *net = calloc(1, sizeof(LinearContLSTM));
    net->num_agents = num_agents;
    net->obs = calloc(num_agents * input_dim, sizeof(float));
    net->num_actions = logit_sizes[0];
    net->log_std = weights->data;
    weights->idx += net->num_actions;
    net->encoder = make_linear(weights, num_agents, input_dim, 128);
    net->gelu1 = make_gelu(num_agents, 128);
    int atn_sum = 0;
    for (int i = 0; i < num_actions; i++) {
        atn_sum += logit_sizes[i];
    }
    net->actor = make_linear(weights, num_agents, 128, atn_sum);
    net->value_fn = make_linear(weights, num_agents, 128, 1);
    net->lstm = make_lstm(weights, num_agents, 128, 128);
    return net;
}

void free_linearcontlstm(LinearContLSTM *net) {
    free(net->obs);
    free(net->encoder);
    free(net->gelu1);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net);
}

void forward_linearcontlstm(LinearContLSTM *net, float *observations, float *actions) {
    linear(net->encoder, observations);
    gelu(net->gelu1, net->encoder->output);
    lstm(net->lstm, net->gelu1->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);
    for (int i = 0; i < net->num_actions; i++) {
        float std = expf(net->log_std[i]);
        float mean = net->actor->output[i];
        actions[i] = randn(mean, std);
    }
}

void generate_dummy_actions(DroneDelivery *env) {
    // Generate random floats in [-1, 1] range
    env->actions[0] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[1] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[2] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[3] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

#ifdef __EMSCRIPTEN__
typedef struct {
    DroneDelivery *env;
    LinearContLSTM *net;
    Weights *weights;
} WebRenderArgs;

void emscriptenStep(void *e) {
    WebRenderArgs *args = (WebRenderArgs *)e;
    DroneDelivery *env = args->env;
    LinearContLSTM *net = args->net;

    forward_linearcontlstm(net, env->observations, env->actions);
    c_step(env);
    c_render(env);
    return;
}

WebRenderArgs *web_args = NULL;
#endif

int main() {
    srand(42); // Seed random number generator

    DroneDelivery *env = calloc(1, sizeof(DroneDelivery));
    env->num_agents = 64;

    env->box_base_density = 50.0;
    env->box_k_growth = 0.25;

    env->dist_decay = 15.0;

    env->grip_k_decay = 5.0;
    env->grip_k_max = 17.105300970916993;
    env->grip_k_min = 1.0;

    env->pos_const = 0.7108647198043252;
    env->pos_penalty = 0.0009629600280207475;

    env->reward_grip = 0.9999;
    env->reward_ho_drop = 0.20890470430909128;
    env->reward_hover = 0.1603521816569735;

    env->reward_max_dist = 65.0;
    env->reward_min_dist = 0.9330297552248776;

    env->vel_penalty_clamp = 0.25;

    env->w_approach = 2.2462377881752698;
    env->w_position = 0.7312628607193232;
    env->w_stability = 1.6094417286807845;
    env->w_velocity = 0.0009999999999999731;

    init(env);

    size_t obs_size = 45;
    size_t act_size = 4;
    env->observations = (float *)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float *)calloc(env->num_agents * act_size, sizeof(float));
    env->rewards = (float *)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->num_agents, sizeof(float));

    //Weights *weights = load_weights("resources/drone/drone_weights.bin", 136073);
    //int logit_sizes[1] = {4};
    //LinearContLSTM *net = make_linearcontlstm(weights, env->num_agents, obs_size, logit_sizes, 1);

    if (!env->observations || !env->actions || !env->rewards) {
        fprintf(stderr, "ERROR: Failed to allocate memory for demo buffers.\n");
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        free(env);
        return 0;
    }

    init(env);
    c_reset(env);

#ifdef __EMSCRIPTEN__
    WebRenderArgs *args = calloc(1, sizeof(WebRenderArgs));
    args->env = env;
    args->net = net;
    args->weights = weights;
    web_args = args;

    emscripten_set_main_loop_arg(emscriptenStep, args, 0, true);
#else
    c_render(env);

    while (!WindowShouldClose()) {
        //forward_linearcontlstm(net, env->observations, env->actions);
        c_step(env);
        c_render(env);
    }

    c_close(env);
    //free_linearcontlstm(net);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
#endif

    return 0;
}
