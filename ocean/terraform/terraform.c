#include "terraform.h"
#include "puffernet.h"

void allocate(Terraform* env) {
    env->observations = (float*)calloc(env->num_agents*442, sizeof(float));
    env->actions = (int*)calloc(3*env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    init(env);
}

void free_allocated(Terraform* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free_initialized(env);
}

typedef struct TerraformNet TerraformNet;
struct TerraformNet {
    int num_agents;
    float* local_obs2d;
    float* global_obs2d;
    float* obs_1d;
    Conv2D* local_conv1;
    ReLU* relu1;
    Conv2D* local_conv2;
    ReLU* relu2;
    Conv2D* global_conv1;
    ReLU* relu3;
    Conv2D* global_conv2;
    ReLU* relu4;
    Linear* flat;
    CatDim1* cat1;
    CatDim1* cat2;
    Linear* proj;
    ReLU* relu5;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};
TerraformNet* init_terranet(Weights* weights, int num_agents, int vision_size, int quadrant_size) {
    TerraformNet* net = calloc(1, sizeof(TerraformNet));
    int hidden_size = 512;
    int cnn_channels = 32;
    int local_conv1_output_size = 3;
    int local_conv2_output_size = 1;
    int global_conv1_output_size = 4;
    int global_conv2_output_size = 2;
    int local_cnn_flat_size = cnn_channels * (local_conv2_output_size * local_conv2_output_size);
    int global_cnn_flat_size = cnn_channels * (global_conv2_output_size * global_conv2_output_size);

    net->num_agents = num_agents;
    net->local_obs2d = calloc(num_agents * vision_size * vision_size * 2, sizeof(float)); // 2 channels - height map & deltas
    net->global_obs2d = calloc(num_agents * quadrant_size * quadrant_size * 2, sizeof(float)); // 2 channels - global volume map and agent location
    net->obs_1d = calloc(num_agents * 5, sizeof(float)); // 2 additional features

    net->local_conv1 = make_conv2d(weights, num_agents, vision_size, vision_size, 2, cnn_channels, 5, 3);
    net->relu1 = make_relu(num_agents, cnn_channels * local_conv1_output_size * local_conv1_output_size);
    net->local_conv2 = make_conv2d(weights, num_agents, local_conv1_output_size, local_conv1_output_size, cnn_channels, cnn_channels, 3, 1);
    net->relu2 = make_relu(num_agents, cnn_channels * local_conv2_output_size * local_conv2_output_size);
    net->global_conv1 = make_conv2d(weights, num_agents, quadrant_size, quadrant_size, 2, cnn_channels, 3, 1);
    net->relu3 = make_relu(num_agents, cnn_channels * global_conv1_output_size * global_conv1_output_size);
    net->global_conv2 = make_conv2d(weights, num_agents, global_conv1_output_size, global_conv1_output_size, cnn_channels, cnn_channels, 3, 1);
    net->relu4 = make_relu(num_agents, cnn_channels * global_conv2_output_size * global_conv2_output_size);
    net->flat = make_linear(weights, num_agents, 5, hidden_size);
    net->cat1 = make_cat_dim1(num_agents, local_cnn_flat_size, global_cnn_flat_size);
    net->cat2 = make_cat_dim1(num_agents, local_cnn_flat_size + global_cnn_flat_size, hidden_size);
    net->proj = make_linear(weights, num_agents, local_cnn_flat_size + global_cnn_flat_size + hidden_size, hidden_size);
    net->relu5 = make_relu(num_agents, hidden_size);
    net->actor = make_linear(weights, num_agents, hidden_size, 13); // +1 for pass move
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_size, 512);
    int logit_sizes[3] = {5, 5, 3};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 3);
    return net;
}

void free_terranet(TerraformNet* net) {
    free(net->local_obs2d);
    free(net->global_obs2d);
    free(net->obs_1d);
    free(net->local_conv1);
    free(net->relu1);
    free(net->local_conv2);
    free(net->relu2);
    free(net->global_conv1);
    free(net->relu3);
    free(net->global_conv2);
    free(net->relu4);
    free(net->flat);
    free(net->cat1);
    free(net->cat2);
    free(net->proj);
    free(net->relu5);
    free(net->lstm);
    free(net->actor);
    free(net->value_fn);
    free(net);
}

void forward(TerraformNet* net, float* observations, int* actions, int vision_size, int quadrant_size) {
    int local_vision_size = vision_size * vision_size;  
    int global_quadrant_size = quadrant_size * quadrant_size;
    // Clear previous observations
    memset(net->local_obs2d, 0, net->num_agents * vision_size * vision_size * 2 * sizeof(float));
    memset(net->global_obs2d, 0, net->num_agents * quadrant_size * quadrant_size * 2 * sizeof(float));
    memset(net->obs_1d, 0, net->num_agents * 5 * sizeof(float));
    
    // Reshape observations into 2D boards and additional features
    float (*local_obs2d)[2][vision_size][vision_size] = (float (*)[2][vision_size][vision_size])net->local_obs2d;
    float (*global_obs2d)[2][quadrant_size][quadrant_size] = (float (*)[2][quadrant_size][quadrant_size])net->global_obs2d;
    float (*obs_1d)[5] = (float (*)[5])net->obs_1d;
    
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * (local_vision_size * 2 + global_quadrant_size * 2 + 5);  // offset for each batch
        
        // Process local vision board
        int obs_2d_idx = 0;
        for(int z = 0; z < 2; z++) {
            for (int i = 0; i < vision_size; i++) {
                for (int j = 0; j < vision_size; j++) {
                    local_obs2d[b][z][i][j] = observations[b_offset + obs_2d_idx];
                    obs_2d_idx++;
                }
            }
        }

        // Process additional features
        obs_1d[b][0] = observations[b_offset + obs_2d_idx];
        obs_1d[b][1] = observations[b_offset + obs_2d_idx + 1];
        obs_1d[b][2] = observations[b_offset + obs_2d_idx + 2];
        obs_1d[b][3] = observations[b_offset + obs_2d_idx + 3];
        obs_1d[b][4] = observations[b_offset + obs_2d_idx + 4];
        obs_2d_idx += 5;
        
        // Process global quadrant board
        for(int z = 0; z < 2; z++) {
            for (int i = 0; i < quadrant_size; i++) {
                for (int j = 0; j < quadrant_size; j++) {
                    global_obs2d[b][z][i][j] = observations[b_offset + obs_2d_idx];
                    obs_2d_idx++;
                }
            }
        }
    }

    // Forward pass through the network
    // local convs
    conv2d(net->local_conv1, net->local_obs2d);
    relu(net->relu1, net->local_conv1->output);
    conv2d(net->local_conv2, net->relu1->output);
    relu(net->relu2, net->local_conv2->output);
    // global convs
    conv2d(net->global_conv1, net->global_obs2d);
    relu(net->relu3, net->global_conv1->output);
    conv2d(net->global_conv2, net->relu3->output);
    relu(net->relu4, net->global_conv2->output);

    linear(net->flat, net->obs_1d);

    cat_dim1(net->cat1, net->relu2->output, net->relu4->output);
    cat_dim1(net->cat2, net->cat1->output, net->flat->output);
    linear(net->proj, net->cat2->output);
    relu(net->relu5, net->proj->output);
    
    lstm(net->lstm, net->relu5->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);

    // Get action by taking argmax of actor output
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);

}

void demo() {
    Weights* weights = load_weights("resources/terraform/puffer_terraform_weights.bin", 2476814);
    TerraformNet* net = init_terranet(weights, 1, 11, 6);
    srand(time(NULL));
    Terraform env = {.size = 64, .num_agents = 1, .reset_frequency = 8192, .reward_scale = 0.04f};
    allocate(&env);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        forward(net, env.observations, env.actions, 11, 6);
        int policy_actions[3] = {env.actions[0], env.actions[1], env.actions[2]};
        
        if(IsKeyDown(KEY_LEFT_SHIFT)) {
            // When shift is held, stop the dozer
            env.actions[0] = 2;  // Stop vertical movement
            env.actions[1] = 2;  // Stop horizontal movement
            env.actions[2] = 0;  // no scoop or drop
            // Override with keyboard controls if keys are pressed
            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) env.actions[0] = 4;
            if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) env.actions[0] = 0;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[1] = 0;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[1] = 4;
            if (IsKeyPressed(KEY_SPACE)) env.actions[2] = 1;
            if (IsKeyPressed(KEY_ENTER)) env.actions[2] = 2;
        }
        c_step(&env);
        c_render(&env);
    }
    free_allocated(&env);
    close_client(env.client);
    free_terranet(net);
    free(weights);
}

void test_performance(int timeout) {
    srand(time(NULL));
    Terraform env = {
        .size = 64,
        .num_agents = 8,
        .reset_frequency = 512,
        .reward_scale = 0.01f,
    };
    allocate(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[3*i] = rand() % 5;
            env.actions[3*i + 1] = rand() % 5;
            env.actions[3*i + 2] = rand() % 3;
        }

        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps * env.num_agents / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free_allocated(&env);
}

int main() {
    // test_performance(10);
    demo();
}

