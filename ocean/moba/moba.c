#include "moba.h"
#include "puffernet.h"

typedef struct MOBANet MOBANet;
struct MOBANet {
    int num_agents;
    float* obs_2d;
    float* obs_1d;
    Conv2D* conv1;
    ReLU* relu1;
    Conv2D* conv2;
    Linear* flat;
    CatDim1* cat;
    ReLU* relu2;
    Linear* proj;
    ReLU* relu3;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

MOBANet* init_mobanet(Weights* weights, int num_agents) {
    MOBANet* net = calloc(1, sizeof(MOBANet));
    int hidden = 128;
    net->num_agents = num_agents;
    net->obs_2d = calloc(num_agents*11*11*19, sizeof(float));
    net->obs_1d = calloc(num_agents*26, sizeof(float));
    net->conv1 = make_conv2d(weights, num_agents, 11, 11, 19, hidden, 5, 3);
    net->relu1 = make_relu(num_agents, hidden*3*3);
    net->conv2 = make_conv2d(weights, num_agents, 3, 3, hidden, hidden, 3, 1);
    net->flat = make_linear(weights, num_agents, 26, hidden);
    net->cat = make_cat_dim1(num_agents, hidden, hidden);
    net->relu2 = make_relu(num_agents, 2*hidden);
    net->proj = make_linear(weights, num_agents, 2*hidden, 128);
    net->relu3 = make_relu(num_agents, 128);
    net->actor = make_linear(weights, num_agents, 128, 23);
    net->value_fn = make_linear(weights, num_agents, 128, 1);
    net->lstm = make_lstm(weights, num_agents, 128, 128);
    int logit_sizes[6] = {7, 7, 3, 2, 2, 2};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 6);
    return net;
}

void free_mobanet(MOBANet* net) {
    free(net->obs_2d);
    free(net->obs_1d);
    free(net->conv1);
    free(net->relu1);
    free(net->conv2);
    free(net->flat);
    free(net->cat);
    free(net->relu2);
    free(net->proj);
    free(net->relu3);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net->multidiscrete);
    free(net);
}

void forward(MOBANet* net, unsigned char* observations, int* actions) {
    memset(net->obs_2d, 0, net->num_agents*11*11*19*sizeof(float));
    float (*obs_2d)[19][11][11] = (float (*)[19][11][11])net->obs_2d;
    float (*obs_1d)[26] = (float (*)[26])net->obs_1d;
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b*(11*11*4 + 26);
        for (int i = 0; i < 11; i++) {
            for (int j = 0; j < 11; j++) {
                int elem_offset = 4*(i*11 + j);
                int one_hot_idx = observations[b_offset + elem_offset];
                obs_2d[b][one_hot_idx][i][j] = 1;
                obs_2d[b][16][i][j] = observations[b_offset + elem_offset+1] / 255.0f;
                obs_2d[b][17][i][j] = observations[b_offset + elem_offset+2] / 255.0f;
                obs_2d[b][18][i][j] = observations[b_offset + elem_offset+3] / 255.0f;
            }
        }
        for (int i = 0; i < 26; i++) {
            obs_1d[b][i] = observations[b_offset + 11*11*4 + i] / 255.0f;
        }
    }

    conv2d(net->conv1, net->obs_2d);
    relu(net->relu1, net->conv1->output);
    conv2d(net->conv2, net->relu1->output);

    linear(net->flat, net->obs_1d);

    cat_dim1(net->cat, net->conv2->output, net->flat->output);
    relu(net->relu2, net->cat->output);
    linear(net->proj, net->relu2->output);
    relu(net->relu3, net->proj->output);
    
    lstm(net->lstm, net->relu3->output);

    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);

    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
    for (int i = 0; i < net->num_agents; i++) {
        actions[i*6] = 100*(actions[i*6] - 3);
        actions[i*6 + 1] = 100*(actions[i*6 + 1] - 3);
    }
}

void demo() {
    Weights* weights = load_weights("resources/moba/moba_weights.bin", 380056);
    bool script_opponents = true;

    int num_agents = script_opponents ? 5 : 10;
    MOBANet* net = init_mobanet(weights, num_agents);

    MOBA env = {
        .vision_range = 5,
        .agent_speed = 1.0,
        .discretize = true,
        .reward_death = -1.0,
        .reward_xp = 0.006,
        .reward_distance = 0.05,
        .reward_tower = 3.0,
        .script_opponents = script_opponents,
    };
    allocate_moba(&env);
    c_reset(&env);
    c_render(&env);
    int frame = 1;
    while (!WindowShouldClose()) {
        if (frame % 12 == 0) {
            c_step(&env);
            forward(net, env.observations, env.actions);
        }
        c_render(&env);
        frame = (frame + 1) % 12;
    }
    free_mobanet(net);
    free(weights);
    free_allocated_moba(&env);
    //close_game_renderer(renderer);
}

void test_performance(float test_time) {
    bool script_opponents = true;
    int num_agents = script_opponents ? 5 : 10;

    MOBA env = {
        .vision_range = 5,
        .agent_speed = 1.0,
        .discretize = true,
        .reward_death = -1.0,
        .reward_xp = 0.006,
        .reward_distance = 0.05,
        .reward_tower = 3.0,
        .script_opponents = script_opponents,
    };
    allocate_moba(&env);

    c_reset(&env);
    int start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        for (int j = 0; j < num_agents; j++) {
            env.actions[6*j] = rand()%600 - 300;
            env.actions[6*j + 1] = rand()%600 - 300;
            env.actions[6*j + 2] = rand()%3;
            env.actions[6*j + 3] = rand()%2;
            env.actions[6*j + 4] = rand()%2;
            env.actions[6*j + 5] = rand()%2;
        }
        c_step(&env);
        i++;
    }
    int end = time(NULL);
    printf("SPS: %f\n", (float)num_agents*i / (end - start));
}

void test_bugs(float test_time) {
    Weights* weights = load_weights("resources/moba/moba_weights.bin", 380056);
    MOBANet* net = init_mobanet(weights, 10);

    MOBA env = {
        .vision_range = 5,
        .agent_speed = 1.0,
        .discretize = true,
        .reward_death = -1.0,
        .reward_xp = 0.006,
        .reward_distance = 0.05,
        .reward_tower = 3.0,
        .script_opponents = true,
    };
    allocate_moba(&env);

    c_reset(&env);
    int start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        c_step(&env);
        forward(net, env.observations, env.actions);
        i++;
    }
    int end = time(NULL);
    printf("SPS: %f\n", 10.0f*i / (end - start));
    printf("Frames: %i\n", i);
    free_mobanet(net);
    free(weights);
    free_allocated_moba(&env);
}


int main() {
    //test_bugs(2.0f);
    demo();
    //test_performance(30.0f);
    return 0;
}
