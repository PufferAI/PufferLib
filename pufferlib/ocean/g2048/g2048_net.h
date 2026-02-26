#include "puffernet.h"

typedef struct G2048Net G2048Net;

struct G2048Net {
    int hidden_dim;
    float* obs;
    Linear* layer1;
    GELU* gelu1;
    Linear* layer2;
    GELU* gelu2;
    Linear* layer3;
    GELU* gelu3;
    Linear* actor_hidden;
    GELU* gelu_actor;
    Linear* actor_head;
    Linear* value_hidden;
    GELU* gelu_value;
    Linear* value_head;
    LSTM* lstm;
    Multidiscrete* multidiscrete;
};

G2048Net* make_g2048net(Weights* weights, int input_dim, int hidden_dim) {
    G2048Net* net = calloc(1, sizeof(G2048Net));
    const int num_agents = 1;
    const int num_actions = 1;
    const int atn_sum = 4;

    int logit_sizes[1] = {4};
    net->obs = calloc(num_agents*input_dim, sizeof(float));
    net->hidden_dim = hidden_dim;

    if (hidden_dim <= 256) {
        net->layer1 = make_linear(weights, num_agents, input_dim, 512);
        net->gelu1 = make_gelu(num_agents, 512);
        net->layer2 = make_linear(weights, num_agents, 512, 256);
        net->gelu2 = make_gelu(num_agents, 256);
        net->layer3 = make_linear(weights, num_agents, 256, hidden_dim);
        net->gelu3 = make_gelu(num_agents, hidden_dim);
    } else {
        net->layer1 = make_linear(weights, num_agents, input_dim, 2*hidden_dim);
        net->gelu1 = make_gelu(num_agents, 2*hidden_dim);
        net->layer2 = make_linear(weights, num_agents, 2*hidden_dim, hidden_dim);
        net->gelu2 = make_gelu(num_agents, hidden_dim);
        net->layer3 = make_linear(weights, num_agents, hidden_dim, hidden_dim);
        net->gelu3 = make_gelu(num_agents, hidden_dim);
    }

    net->actor_hidden = make_linear(weights, num_agents, hidden_dim, hidden_dim);
    net->gelu_actor = make_gelu(num_agents, hidden_dim);
    net->actor_head = make_linear(weights, num_agents, hidden_dim, atn_sum);

    net->value_hidden = make_linear(weights, num_agents, hidden_dim, hidden_dim);
    net->gelu_value = make_gelu(num_agents, hidden_dim);
    net->value_head = make_linear(weights, num_agents, hidden_dim, 1);

    net->lstm = make_lstm(weights, num_agents, hidden_dim, hidden_dim);
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, num_actions);
    return net;
}

void free_g2048net(G2048Net* net) {
    free(net->obs);
    free(net->layer1);
    free(net->gelu1);
    free(net->layer2);
    free(net->gelu2);
    free(net->layer3);
    free(net->gelu3);

    free(net->actor_hidden);
    free(net->gelu_actor);
    free(net->actor_head);
    free(net->value_hidden);
    free(net->gelu_value);
    free(net->value_head);

    free(net->lstm);
    free(net->multidiscrete);
    free(net);
}

void forward_g2048net(G2048Net* net, unsigned char* observations, int* actions) {
    for (int i = 0; i < net->layer1->input_dim; i++) {
        net->obs[i] = (float)observations[i];
        if (i < 16) net->obs[i] /= 100.0f;
    }

    linear(net->layer1, net->obs);
    gelu(net->gelu1, net->layer1->output);
    linear(net->layer2, net->gelu1->output);
    gelu(net->gelu2, net->layer2->output);
    linear(net->layer3, net->gelu2->output);
    gelu(net->gelu3, net->layer3->output);

    lstm(net->lstm, net->gelu3->output);

    // Actor only. Don't need critic in inference
    linear(net->actor_hidden, net->lstm->state_h);
    gelu(net->gelu_actor, net->actor_hidden->output);
    linear(net->actor_head, net->gelu_actor->output);
    softmax_multidiscrete(net->multidiscrete, net->actor_head->output, actions);
}
