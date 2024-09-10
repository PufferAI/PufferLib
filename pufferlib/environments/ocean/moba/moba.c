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
    ArgmaxMultidiscrete* multidiscrete;
};

MOBANet* init_mobanet(Weights* weights, int num_agents) {
    MOBANet* net = calloc(1, sizeof(MOBANet));
    net->num_agents = num_agents;
    net->obs_2d = calloc(num_agents*11*11*19, sizeof(float));
    net->obs_1d = calloc(num_agents*26, sizeof(float));
    printf("First weight: %f\n", weights->data[0]);
    printf("Weight index: %d\n", weights->idx);
    net->conv1 = make_conv2d(weights, num_agents, 11, 11, 19, 32, 5, 3);
    net->relu1 = make_relu(num_agents, 32*3*3);
    net->conv2 = make_conv2d(weights, num_agents, 3, 3, 32, 32, 3, 1);
    net->flat = make_linear(weights, num_agents, 26, 32);
    net->cat = make_cat_dim1(num_agents, 32, 32);
    net->relu2 = make_relu(num_agents, 64);
    net->proj = make_linear(weights, num_agents, 64, 128);
    net->relu3 = make_relu(num_agents, 128);
    net->actor = make_linear(weights, num_agents, 128, 23);
    net->value_fn = make_linear(weights, num_agents, 128, 1);
    net->lstm = make_lstm(weights, num_agents, 128, 128);
    int logit_sizes[6] = {7, 7, 3, 2, 2, 2};
    net->multidiscrete = make_argmax_multidiscrete(num_agents, logit_sizes, 6);

    printf("conv1 weight 0: %f\n", net->conv1->weights[0]);
    printf("conv1 bias 0: %f\n", net->conv1->bias[0]);
    printf("conv2 weight 0: %f\n", net->conv2->weights[0]);
    printf("conv2 bias 0: %f\n", net->conv2->bias[0]);
    printf("flat weight 0: %f\n", net->flat->weights[0]);
    printf("flat bias 0: %f\n", net->flat->bias[0]);
    printf("proj weight 0: %f\n", net->proj->weights[0]);
    printf("proj bias 0: %f\n", net->proj->bias[0]);
    printf("actor weight 0: %f\n", net->actor->weights[0]);
    printf("actor bias 0: %f\n", net->actor->bias[0]);
    printf("value_fn weight 0: %f\n", net->value_fn->weights[0]);
    printf("value_fn bias 0: %f\n", net->value_fn->bias[0]);
    printf("lstm weight 0: %f\n", net->lstm->weights_input[0]);
    printf("lstm bias 0: %f\n", net->lstm->bias_input[0]);
    printf("lstm weight 1: %f\n", net->lstm->weights_state[0]);
    printf("lstm bias 1: %f\n", net->lstm->bias_state[0]);
    return net;
}

void free_mobanet(MOBANet* net) {
    free_conv2d(net->conv1);
    free_relu(net->relu1);
    free_conv2d(net->conv2);
    free_linear(net->flat);
    free_linear(net->proj);
    free_linear(net->actor);
    free_linear(net->value_fn);
}

void forward(MOBANet* net, unsigned char* env_obs_2d, unsigned char* env_obs_1d, int* actions) {
    memset(net->obs_2d, 0, net->num_agents*11*11*19*sizeof(float));
    float (*obs_2d)[19][11][11] = (float (*)[19][11][11])net->obs_2d;
    float (*obs_1d)[26] = (float (*)[26])net->obs_1d;
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b*(11*11*4);
        for (int i = 0; i < 11; i++) {
            for (int j = 0; j < 11; j++) {
                int elem_offset = 4*(i*11 + j);
                int one_hot_idx = env_obs_2d[b_offset + elem_offset];
                obs_2d[b][one_hot_idx][i][j] = 1;
                obs_2d[b][16][i][j] = env_obs_2d[b_offset + elem_offset+1] / 255.0f;
                obs_2d[b][17][i][j] = env_obs_2d[b_offset + elem_offset+2] / 255.0f;
                obs_2d[b][18][i][j] = env_obs_2d[b_offset + elem_offset+3] / 255.0f;
            }
        }
        for (int i = 0; i < 26; i++) {
            obs_1d[b][i] = env_obs_1d[b*26 + i] / 255.0f;
        }
    }

    /*
    printf("obs_2d: ");
    for (int i = 0; i < 11*11*19; i++) {
        printf("%f ", net->obs_2d[i]);
    }
    printf("\n");
    */

    //printf("obs_1d: %f\n", obs_1d[0][0]);
    conv2d(net->conv1, net->obs_2d);
    //printf("conv1: %f\n", net->conv1->output[0]);
    relu(net->relu1, net->conv1->output);
    //printf("relu1: %f\n", net->relu1->output[0]);
    conv2d(net->conv2, net->relu1->output);
    //printf("conv2: %f\n", net->conv2->output[0]);
    /*
    printf("cnn: ");
    for (int i = 0; i < 32; i++) {
        printf("%f ", net->conv2->output[i]);
    }
    printf("\n");
    */

    linear(net->flat, net->obs_1d);
    /*
    for (int i = 0; i < 32; i++) {
        printf("%f ", net->flat->output[i]);
    }
    printf("\n");
    */

    cat_dim1(net->cat, net->conv2->output, net->flat->output);
    //printf("cat: %f\n", net->cat->output[0]);
    relu(net->relu2, net->cat->output);
    //printf("relu2: %f\n", net->relu2->output[0]);
    linear(net->proj, net->relu2->output);
    //printf("proj: %f\n", net->proj->output[0]);
    relu(net->relu3, net->proj->output);
    //printf("relu3: %f\n", net->relu3->output[0]);

    /*
    for (int i=0; i<128; i++) {
        printf("%f ", net->relu3->output[i]);
    }
    */
    
    lstm(net->lstm, net->relu3->output);
    //printf("lstm: %f\n", net->lstm->state_h[0]);
    /*
    for (int i=0; i<128; i++) {
        printf("%f ", net->lstm->state_h[i]);
    }
    */

    linear(net->actor, net->lstm->state_h);
    //printf("actor: %f\n", net->actor->output[0]);
    linear(net->value_fn, net->lstm->state_h);
    //printf("value_fn: %f\n", net->value_fn->output[0]);

    argmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
    /*
    for (int i = 0; i < net->num_agents; i++) {
        printf("Actions for agent %i: ", i);
        for (int j = 0; j < 6; j++) {
            printf("%i ", actions[i*6 + j]);
        }
        printf("\n");
    }
    */

    for (int i = 0; i < net->num_agents; i++) {
        /*
        printf("Setting actions for agent %i: ", i);
        for (int j = 0; j < 6; j++) {
            printf("%i ", actions[i*6 + j]);
        }
        printf("\n");
        */

        actions[i*6] = 100*(actions[i*6] - 3);
        actions[i*6 + 1] = 100*(actions[i*6 + 1] - 3);
    }
}

int main() {
    int num_agents = 10;
    int num_creeps = 100;
    int num_neutrals = 72;
    int num_towers = 24;
    int vision_range = 5;
    float agent_speed = 1.0;
    bool discretize = true;
    float reward_death = -1.0;
    float reward_xp = 0.006;
    float reward_distance = 0.05;
    float reward_tower = 3.0;

    Weights* weights = load_weights("moba_weights.bin", 168856);
    MOBANet* net = init_mobanet(weights, num_agents);

    MOBA* env = init_moba(num_agents, num_creeps, num_neutrals, num_towers, vision_range,
        agent_speed, discretize, reward_death, reward_xp, reward_distance, reward_tower);

    GameRenderer* renderer = init_game_renderer(32, 41, 23);

    reset(env);
    int frame = 0;
    while (!WindowShouldClose()) {
        if (frame % 12 == 0) {
            step(env);

            /*
            for (int i = 0; i < (11*11*4+26); i++) {
                if (i < 11*11*4) {
                    env->observations_map[i] = i%16;
                } else {
                    env->observations_extra[i-11*11*4] = i%16;
                }
            }
            */
            forward(net, env->observations_map, env->observations_extra, env->actions);
        }
        render_game(renderer, env, frame);
        frame = (frame + 1) % 12;
    }
    free_mobanet(net);
    free_moba(env);
    close_game_renderer(renderer);
    return 0;
}
