#include "puffernet.h"

typedef struct G2048Net G2048Net;

#define G2048_EMBED_DIM 3
#define G2048_NUM_GRID_CELLS 16
#define G2048_NUM_TILE_VALUES 18
#define G2048_NUM_OBS (G2048_NUM_GRID_CELLS * G2048_EMBED_DIM)

struct G2048Net {
    int hidden_dim;
    int* value_indices;          // [16] - tile values for embedding lookup
    int* pos_indices;            // [16] - constant position indices 0-15
    float* embedded_obs;         // [48] - working buffer for embedded observations
    Embedding* value_embed;      // [18, 3] - tile value embeddings
    Embedding* pos_embed;        // [16, 3] - position embeddings
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

G2048Net* make_g2048net(Weights* weights, int hidden_dim) {
    G2048Net* net = calloc(1, sizeof(G2048Net));
    const int num_agents = 1;
    const int num_actions = 1;
    const int atn_sum = 4;

    int logit_sizes[1] = {4};
    net->hidden_dim = hidden_dim;

    net->value_indices = calloc(G2048_NUM_GRID_CELLS, sizeof(int));
    net->pos_indices = calloc(G2048_NUM_GRID_CELLS, sizeof(int));
    net->embedded_obs = calloc(G2048_NUM_OBS, sizeof(float));

    for (int i = 0; i < G2048_NUM_GRID_CELLS; i++) {
        net->pos_indices[i] = i;
    }

    net->value_embed = make_embedding(weights, G2048_NUM_GRID_CELLS, G2048_NUM_TILE_VALUES, G2048_EMBED_DIM);
    net->pos_embed = make_embedding(weights, G2048_NUM_GRID_CELLS, G2048_NUM_GRID_CELLS, G2048_EMBED_DIM);

    net->layer1 = make_linear(weights, num_agents, G2048_NUM_OBS, 2 * hidden_dim);
    net->gelu1 = make_gelu(num_agents, 2 * hidden_dim);
    net->layer2 = make_linear(weights, num_agents, 2 * hidden_dim, hidden_dim);
    net->gelu2 = make_gelu(num_agents, hidden_dim);
    net->layer3 = make_linear(weights, num_agents, hidden_dim, hidden_dim);
    net->gelu3 = make_gelu(num_agents, hidden_dim);

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
    free(net->value_indices);
    free(net->pos_indices);
    free(net->embedded_obs);
    free(net->value_embed);
    free(net->pos_embed);
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
    // Convert observations to value indices for embedding lookup
    for (int i = 0; i < G2048_NUM_GRID_CELLS; i++) {
        net->value_indices[i] = (int)observations[i];
    }

    // Embed tile values: value_embed(observations) -> [16, 3]
    embedding(net->value_embed, net->value_indices);

    // Embed positions: pos_embed([0,1,...,15]) -> [16, 3]
    embedding(net->pos_embed, net->pos_indices);

    // Add value and position embeddings, flatten to [48]
    // PyTorch: grid_obs = (value_obs + pos_obs).flatten(1)
    for (int i = 0; i < G2048_NUM_GRID_CELLS; i++) {
        for (int j = 0; j < G2048_EMBED_DIM; j++) {
            int idx = i * G2048_EMBED_DIM + j;
            net->embedded_obs[idx] = net->value_embed->output[idx] + net->pos_embed->output[idx];
        }
    }

    linear(net->layer1, net->embedded_obs);
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
