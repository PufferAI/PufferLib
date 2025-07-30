#include <time.h>
#include <unistd.h>
#include "drive.h"
#include "puffernet.h"

typedef struct DriveNet DriveNet;
struct DriveNet {
    int num_agents;
    float* obs_self;
    float* obs_partner;
    float* obs_road;
    float* partner_linear_output;
    float* road_linear_output;
    float* partner_layernorm_output;
    float* road_layernorm_output;
    float* partner_linear_output_two;
    float* road_linear_output_two;
    Linear* ego_encoder;
    Linear* road_encoder;
    Linear* partner_encoder;
    LayerNorm* ego_layernorm;
    LayerNorm* road_layernorm;
    LayerNorm* partner_layernorm;
    Linear* ego_encoder_two;
    Linear* road_encoder_two;
    Linear* partner_encoder_two;
    MaxDim1* partner_max;
    MaxDim1* road_max;
    CatDim1* cat1;
    CatDim1* cat2;
    GELU* gelu;
    Linear* shared_embedding;
    ReLU* relu;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

DriveNet* init_drivenet(Weights* weights, int num_agents) {
    DriveNet* net = calloc(1, sizeof(DriveNet));
    int hidden_size = 256;
    int input_size = 64;

    net->num_agents = num_agents;
    net->obs_self = calloc(num_agents*7, sizeof(float)); // 7 features
    net->obs_partner = calloc(num_agents*63*7, sizeof(float)); // 63 objects, 7 features
    net->obs_road = calloc(num_agents*200*13, sizeof(float)); // 200 objects, 13 features
    net->partner_linear_output = calloc(num_agents*63*input_size, sizeof(float));
    net->road_linear_output = calloc(num_agents*200*input_size, sizeof(float));
    net->partner_linear_output_two = calloc(num_agents*63*input_size, sizeof(float));
    net->road_linear_output_two = calloc(num_agents*200*input_size, sizeof(float));
    net->partner_layernorm_output = calloc(num_agents*63*input_size, sizeof(float));
    net->road_layernorm_output = calloc(num_agents*200*input_size, sizeof(float));
    net->ego_encoder = make_linear(weights, num_agents, 7, input_size);
    net->ego_layernorm = make_layernorm(weights, num_agents, input_size);
    net->ego_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->road_encoder = make_linear(weights, num_agents, 13, input_size);
    net->road_layernorm = make_layernorm(weights, num_agents, input_size);
    net->road_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->partner_encoder = make_linear(weights, num_agents, 7, input_size);
    net->partner_layernorm = make_layernorm(weights, num_agents, input_size);
    net->partner_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->partner_max = make_max_dim1(num_agents, 63, input_size);
    net->road_max = make_max_dim1(num_agents, 200, input_size);
    net->cat1 = make_cat_dim1(num_agents, input_size, input_size);
    net->cat2 = make_cat_dim1(num_agents, input_size + input_size, input_size);
    net->gelu = make_gelu(num_agents, 3*input_size);
    net->shared_embedding = make_linear(weights, num_agents, input_size*3, hidden_size);
    net->relu = make_relu(num_agents, hidden_size);
    net->actor = make_linear(weights, num_agents, hidden_size, 20); 
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_size, 256);
    memset(net->lstm->state_h, 0, num_agents*256*sizeof(float));
    memset(net->lstm->state_c, 0, num_agents*256*sizeof(float));
    int logit_sizes[2] = {7, 13};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 2);
    return net;
}

void free_drivenet(DriveNet* net) {
    free(net->obs_self);
    free(net->obs_partner);
    free(net->obs_road);
    free(net->partner_linear_output);
    free(net->road_linear_output);
    free(net->partner_linear_output_two);
    free(net->road_linear_output_two);
    free(net->ego_encoder);
    free(net->road_encoder);
    free(net->partner_encoder); 
    free(net->ego_layernorm);
    free(net->road_layernorm);
    free(net->partner_layernorm);
    free(net->ego_encoder_two);
    free(net->road_encoder_two);
    free(net->partner_encoder_two);
    free(net->partner_max);
    free(net->road_max);
    free(net->cat1);
    free(net->cat2);
    free(net->gelu);
    free(net->shared_embedding);
    free(net->relu);
    free(net->multidiscrete);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net);
}

void forward(DriveNet* net, float* observations, int* actions) {
    // Clear previous observations
    memset(net->obs_self, 0, net->num_agents * 7 * sizeof(float));
    memset(net->obs_partner, 0, net->num_agents * 63 * 7 * sizeof(float));
    memset(net->obs_road, 0, net->num_agents * 200 * 13 * sizeof(float));
    
    // Reshape observations into 2D boards and additional features
    float (*obs_self)[7] = (float (*)[7])net->obs_self;
    float (*obs_partner)[63][7] = (float (*)[63][7])net->obs_partner;
    float (*obs_road)[200][13] = (float (*)[200][13])net->obs_road;
    
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * (7 + 63*7 + 200*7);  // offset for each batch
        int partner_offset = b_offset + 7;
        int road_offset = b_offset + 7 + 63*7;
        // Process self observation
        for(int i = 0; i < 7; i++) {
            obs_self[b][i] = observations[b_offset + i];
        }

        // Process partner observation
        for(int i = 0; i < 63; i++) {
            for(int j = 0; j < 7; j++) {
                obs_partner[b][i][j] = observations[partner_offset + i*7 + j];
            }
        }

        // Process road observation
        for(int i = 0; i < 200; i++) {
            for(int j = 0; j < 7; j++) {
                obs_road[b][i][j] = observations[road_offset + i*7 + j];
            }
            for(int j = 0; j < 7; j++) {
                if(j == observations[road_offset+i*7 + 6]) {
                    obs_road[b][i][6 + j] = 1.0f;
                } else {
                    obs_road[b][i][6 + j] = 0.0f;
                }
            }
        }
    }

    // Forward pass through the network
    linear(net->ego_encoder, net->obs_self);
    layernorm(net->ego_layernorm, net->ego_encoder->output);
    linear(net->ego_encoder_two, net->ego_layernorm->output);
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 63; obj++) {
            // Get the 7 features for this object
            float* obj_features = &net->obs_partner[b*63*7 + obj*7];
            // Apply linear layer to this object
            _linear(obj_features, net->partner_encoder->weights, net->partner_encoder->bias,
                   &net->partner_linear_output[b*63*64 + obj*64], 1, 7, 64);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 63; obj++) {
            float* after_first = &net->partner_linear_output[b*63*64 + obj*64];
            _layernorm(after_first, net->partner_layernorm->weights, net->partner_layernorm->bias,
                        &net->partner_layernorm_output[b*63*64 + obj*64], 1, 64);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 63; obj++) {
            // Get the 7 features for this object
            float* obj_features = &net->partner_layernorm_output[b*63*64 + obj*64];
            // Apply linear layer to this object
            _linear(obj_features, net->partner_encoder_two->weights, net->partner_encoder_two->bias,
                   &net->partner_linear_output_two[b*63*64 + obj*64], 1, 64, 64);
            
        }
    }
    
    // Process road objects: apply linear to each object individually  
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 200; obj++) {
            // Get the 13 features for this object
            float* obj_features = &net->obs_road[b*200*13 + obj*13];
            // Apply linear layer to this object
            _linear(obj_features, net->road_encoder->weights, net->road_encoder->bias,
                   &net->road_linear_output[b*200*64 + obj*64], 1, 13, 64);
        }
    }
    
    // Apply layer norm and second linear to each road object
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 200; obj++) {
            float* after_first = &net->road_linear_output[b*200*64 + obj*64];
            _layernorm(after_first, net->road_layernorm->weights, net->road_layernorm->bias,
                        &net->road_layernorm_output[b*200*64 + obj*64], 1, 64);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 200; obj++) {
            float* after_first = &net->road_layernorm_output[b*200*64 + obj*64];
            _linear(after_first, net->road_encoder_two->weights, net->road_encoder_two->bias,
                    &net->road_linear_output_two[b*200*64 + obj*64], 1, 64, 64);
        }
    }
    
    max_dim1(net->partner_max, net->partner_linear_output_two);
    max_dim1(net->road_max, net->road_linear_output_two);
    cat_dim1(net->cat1, net->ego_encoder_two->output, net->road_max->output);
    cat_dim1(net->cat2, net->cat1->output, net->partner_max->output);
    gelu(net->gelu, net->cat2->output);
    linear(net->shared_embedding, net->gelu->output);
    relu(net->relu, net->shared_embedding->output);
    lstm(net->lstm, net->relu->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);

    // Get action by taking argmax of actor output
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}
void demo() {

    Drive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
        .reward_vehicle_collision = -0.1f,
        .reward_offroad_collision = -0.1f,
	    .map_name = "resources/drive/binaries/map_942.bin",
        .spawn_immunity_timer = 50
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin", 595925);
    DriveNet* net = init_drivenet(weights, env.active_agent_count);
    //Client* client = make_client(&env);
    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        // Handle camera controls
        int (*actions)[2] = (int(*)[2])env.actions;
        forward(net, env.observations, env.actions);
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            actions[env.human_agent_idx][0] = 3;
            actions[env.human_agent_idx][1] = 6;
            if(IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)){
                actions[env.human_agent_idx][0] += accel_delta;
                // Cap acceleration to maximum of 6
                if(actions[env.human_agent_idx][0] > 6) {
                    actions[env.human_agent_idx][0] = 6;
                }
            }
            if(IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)){
                actions[env.human_agent_idx][0] -= accel_delta;
                // Cap acceleration to minimum of 0
                if(actions[env.human_agent_idx][0] < 0) {
                    actions[env.human_agent_idx][0] = 0;
                }
            }
            if(IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)){
                actions[env.human_agent_idx][1] += steer_delta;
                // Cap steering to minimum of 0
                if(actions[env.human_agent_idx][1] < 0) {
                    actions[env.human_agent_idx][1] = 0;
                }
            }
            if(IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)){
                actions[env.human_agent_idx][1] -= steer_delta;
                // Cap steering to maximum of 12
                if(actions[env.human_agent_idx][1] > 12) {
                    actions[env.human_agent_idx][1] = 12;
                }
            }   
            if(IsKeyPressed(KEY_TAB)){
                env.human_agent_idx = (env.human_agent_idx + 1) % env.active_agent_count;
            }
        }
        c_step(&env);
        c_render(&env);
    }

    close_client(env.client);
    free_allocated(&env);
    free_drivenet(net);
    free(weights);
}

void performance_test() {
    long test_time = 10;
    Drive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
	    .map_name = "resources/drive/binaries/map_942.bin"
    };
    clock_t start_time, end_time;
    double cpu_time_used;
    start_time = clock();
    allocate(&env);
    c_reset(&env);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Init time: %f\n", cpu_time_used);

    long start = time(NULL);
    int i = 0;
    int (*actions)[2] = (int(*)[2])env.actions;
    
    while (time(NULL) - start < test_time) {
        // Set random actions for all agents
        for(int j = 0; j < env.active_agent_count; j++) {
            int accel = rand() % 7;
            int steer = rand() % 13;
            actions[j][0] = accel;  // -1, 0, or 1
            actions[j][1] = steer;  // Random steering
        }
        
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (i*env.active_agent_count) / (end - start));
    free_allocated(&env);
}

int main() {
    demo();
    // performance_test();
    return 0;
}
