#include "pong.h"
#include "puffernet.h"

typedef struct PongNet PongNet;
struct PongNet {
    int num_agents;
    float* obs;
    Linear* encoder;
    ReLU* relu1;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

PongNet* init_pongnet(Weights* weights, int num_agents) {
    PongNet* net = calloc(1, sizeof(PongNet));
    net->num_agents = num_agents;
    net->obs = calloc(num_agents*8, sizeof(float));
    net->encoder = make_linear(weights, num_agents, 8, 128);
    net->relu1 = make_relu(num_agents, 128);
    net->actor = make_linear(weights, num_agents, 128, 3);
    net->value_fn = make_linear(weights, num_agents, 128, 1);
    net->lstm = make_lstm(weights, num_agents, 128, 128);
    int logit_sizes[1] = {3};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

void free_pongnet(PongNet* net) {
    free_linear(net->encoder);
    free_linear(net->actor);
    free_linear(net->value_fn);
    free_lstm(net->lstm);
}

void forward(PongNet* net, float* observations, unsigned int* actions) {
    linear(net->encoder, observations);
    relu(net->relu1, net->encoder->output);
    lstm(net->lstm, net->relu1->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}

int main() {
    Weights* weights = load_weights("pong_weights.bin", 133764);
    PongNet* net = init_pongnet(weights, 1);

    Pong env = {
        .width = 500,
        .height = 640,
        .paddle_width = 20,
        .paddle_height = 70,
        .ball_width = 10,
        .ball_height = 15,
        .paddle_speed = 8,
        .ball_initial_speed_x = 10,
        .ball_initial_speed_y = 1,
        .ball_speed_y_increment = 3,
        .ball_max_speed_y = 13,
        .max_score = 21,
        .frameskip = 4,
    };
    allocate(&env);

    Client* client = make_client(&env);

    reset(&env);
    while (!WindowShouldClose()) {
        // User can take control of the paddle
        //env.actions[0] = 0;
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 1;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 2;

        step(&env);
        forward(net, env.observations, env.actions);
        render(client, &env);
    }
    free_pongnet(net);
    free_weights(weights);
    free_allocated(&env);
    close_client(client);
}

