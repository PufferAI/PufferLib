// Standalone C demo for DroneFork environment
// Compile using: ./scripts/build_ocean.sh drone [local|fast]
// Run with: ./drone

#include "rocket_drone.h"
#include "puffernet.h"
#include <time.h>
#include <sys/stat.h>

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

static size_t get_file_size(const char *path) {
    struct stat st;
    if(stat(path,&st)!=0) exit(1);
    return st.st_size;
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
    for (int agentid = 0; agentid < net->num_agents; agentid++) {
        for (int i = 0; i < net->num_actions; i++) {
            int idx = agentid * net->num_actions + i;
            float logstd = net->log_std[i];
            if (logstd < -10.0f) logstd = -10.0f;
            if (logstd > 0.0f) logstd = 0.0f;
            float std = expf(logstd);
            float mean = net->actor->output[idx];
            mean = tanhf(mean);
            
            float sample = randn(mean, std);
            if (sample < -1.0f) sample = -1.0f;
            if (sample > 1.0f) sample = 1.0f;
            if (i == 4) actions[idx] = sample > 0.0f ? 1.0f : 0.0f;
            else actions[idx] = sample;
        }
    }
}

void generate_dummy_actions(RocketDrone *env) {
    // Generate random floats in [-1, 1] range
    env->actions[0] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[1] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[2] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[3] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

#ifdef __EMSCRIPTEN__
typedef struct {
    RocketDrone *env;
    LinearContLSTM *net;
    Weights *weights;
} WebRenderArgs;

void emscriptenStep(void *e) {
    WebRenderArgs *args = (WebRenderArgs *)e;
    RocketDrone *env = args->env;
    LinearContLSTM *net = args->net;
    if (!player_active) forward_linearcontlstm(net, env->observations, env->actions);
    else player_character(env, player_idx);
    c_step(env);
    c_render(env);
    return;
}

WebRenderArgs *web_args = NULL;
#endif

int main(int argc, char** argv) {
    srand(time(NULL)); // Seed random number generator

    RocketDrone *env = calloc(1, sizeof(RocketDrone));
    env->num_agents = 8;

    init(env);

    size_t obs_size = 41;
    size_t act_size = 7;
    env->observations = (float *)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float *)calloc(env->num_agents * act_size, sizeof(float));
    env->rewards = (float *)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->num_agents, sizeof(float));


    // usage ./rocket_drone -> load default weights, ./rocket_drone {weights} -> load specified weights
    Weights *weights;
    char wpath[255];
    const char *weight_path;
    if (argc > 1) {
        snprintf(wpath, sizeof(wpath), "pufferlib/resources/rocket_drones/%s", argv[1]);
        weight_path = wpath;
    } else {
        weight_path = "pufferlib/resources/rocket_drones/drone_weights.bin";
    }
    size_t num_bytes = get_file_size(weight_path);
    size_t num_weights = num_bytes / sizeof(float);
    printf("Loading weights from: %s\n", weight_path);
    printf("File size: %zu bytes = %zu weights\n", num_bytes, num_weights);
    weights = load_weights(weight_path, num_weights);
    printf("Weights loaded successfully. Size: %d, Index: %d\n", weights->size, weights->idx);

    int logit_sizes[1] = {7};
    LinearContLSTM *net = make_linearcontlstm(weights, env->num_agents, obs_size, logit_sizes, 1);

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
        forward_linearcontlstm(net, env->observations, env->actions);

        c_step(env);
        c_render(env);
    }

    c_close(env);
    free_linearcontlstm(net);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
#endif

    return 0;
}
