#include <time.h>
#include <unistd.h>
#include "tower_climb.h"
#include "puffernet.h"

typedef struct TowerClimbNet TowerClimbNet;
struct TowerClimbNet {
    int num_agents;
    float* obs_3d;
    float* obs_1d;
    Conv3D* conv1;
    ReLU* relu1;
    Conv3D* conv2;
    Linear* flat;
    CatDim1* cat;
    Linear* proj;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

TowerClimbNet* init_tower_climb_net(Weights* weights, int num_agents) {
    TowerClimbNet* net = calloc(1, sizeof(TowerClimbNet));
    int hidden_size = 256;
    int cnn_channels = 16;
    // Calculate correct output sizes for Conv3D layers
    // First conv: (5,5,9) -> (4,4,8) with kernel=2, stride=1
    // Second conv: (4,4,8) -> (3,3,7) with kernel=2, stride=1
    int cnn_flat_size = cnn_channels * 1 * 1 * 5;  // Match PyTorch size

    net->num_agents = num_agents;
    net->obs_3d = calloc(5 * 5 * 9, sizeof(float));
    net->obs_1d = calloc(3, sizeof(float));
    net->conv1 = make_conv3d(weights, num_agents, 9, 5, 5, 1, cnn_channels, 3, 1);
    net->relu1 = make_relu(num_agents, cnn_channels * 3 * 3 * 7);
    net->conv2 = make_conv3d(weights, num_agents, 7, 3, 3, cnn_channels, cnn_channels, 3, 1);
    net->flat = make_linear(weights, num_agents, 3, 16);
    net->cat = make_cat_dim1(num_agents, cnn_flat_size, 16);
    net->proj = make_linear(weights, num_agents, cnn_flat_size + 16, hidden_size);
    net->actor = make_linear(weights, num_agents, hidden_size, 6);
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_size, hidden_size);
    int logit_sizes[1] = {6};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

void forward(TowerClimbNet* net, unsigned char* observations, int* actions) {
    int vision_size = 5 * 5 * 9;
    int player_size = 3;
    // clear previous observations
    memset(net->obs_3d, 0, vision_size * sizeof(float));
    memset(net->obs_1d, 0, player_size * sizeof(float));
    // reshape board to 3d tensor
    float (*obs_3d)[1][5][5][9] = (float (*)[1][5][5][9])net->obs_3d;
    float (*obs_1d)[3] = (float (*)[3])net->obs_1d;
    // process vision board
    int obs_3d_idx = 0;
    for (int b = 0; b < 1; b++) {
        for (int d = 0; d < 5; d++) {
            for (int h = 0; h < 5; h++) {
                for (int w = 0; w < 9; w++) {
                    obs_3d[b][0][d][h][w] = observations[obs_3d_idx];
                    obs_3d_idx++;
                }
            }
        }
    }
    // process player board
    for (int i = 0; i < player_size; i++) {
        obs_1d[0][i] = observations[vision_size + i];
    }

    conv3d(net->conv1, net->obs_3d);
    relu(net->relu1, net->conv1->output);
    conv3d(net->conv2, net->relu1->output);
    linear(net->flat, net->obs_1d);
    cat_dim1(net->cat, net->conv2->output, net->flat->output);
    linear(net->proj, net->cat->output);
    lstm(net->lstm, net->proj->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}

void free_tower_climb_net(TowerClimbNet* net) {
    free(net->obs_3d);
    free(net->obs_1d);
    free(net->conv1);
    free(net->relu1);
    free(net->conv2);
    free(net->flat);
    free(net->cat);
    free(net->proj);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net->multidiscrete);
    free(net);
}

void demo() {   
    Weights* weights = load_weights("resources/tower_climb_weights.bin", 560407);
    TowerClimbNet* net = init_tower_climb_net(weights, 1);
    CTowerClimb* env = allocate();
    int seed = 0;
    srand(time(NULL));
    int random_level = 5 + (rand() % 4);
    init_random_level(env, random_level, 15, 10, seed);
    Client* client = make_client(env);
    client->enable_animations = 1;
    int tick = 0;
    while (!WindowShouldClose()) {
        int done = 0;
        if (tick % 6 == 0 && !client->isMoving) {
            tick = 0;
            int human_action = env->actions[0];
            forward(net, env->observations, env->actions);
            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                env->actions[0] = human_action;
            }
            done = c_step(env);
            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                env->actions[0] = NOOP;
            }
            if (done) {
                seed++;
                c_reset(env);
                srand(time(NULL));
                int random_level_next = 5 + (rand() % 4);
                init_random_level(env, random_level_next, 15, 10, seed);
            }
        }
        tick++;
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Camera controls
            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
                env->actions[0] = UP;
            }
            if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
                env->actions[0] = LEFT;
            }
            if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
                env->actions[0] = RIGHT;
            }
            if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)){
                env->actions[0] = DOWN;
            }
            if (IsKeyPressed(KEY_SPACE)){
                env->actions[0] = GRAB;
            }
            if (IsKeyPressed(KEY_RIGHT_SHIFT)){
                env->actions[0] = DROP;
            }
        }
        c_render(client, env);
    }
    close_client(client);
    free_allocated(env);
}

void performance_test() {
    long test_time = 10;
    CTowerClimb* env = allocate();
    int seed = 0;
    init_random_level(env, 8, 25, 15, seed);
    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env->actions[0] = rand() % 5;
        int done = 0;
        done = c_step(env);
        if (done) {
            seed++;
            c_reset(env);
            init_random_level(env, 8, 25, 15, seed);
        }
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(env);
}

int main() {
    demo();
    // performance_test();
    return 0;
}


