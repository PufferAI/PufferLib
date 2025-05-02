#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "puffernet.h"
#include "nmmo3.h"

// Only rens a few agents in the C
// version, and reduces for web.
// You can run the full 1024 on GPU
// with PyTorch.
#if defined(PLATFORM_WEB)
    #define NUM_AGENTS 4
#else
    #define NUM_AGENTS 16
#endif


typedef struct MMONet MMONet;
struct MMONet {
    int num_agents;
    float* ob_map;
    int* ob_player_discrete;
    float* ob_player_continuous;
    float* ob_reward;
    Conv2D* map_conv1;
    ReLU* map_relu;
    Conv2D* map_conv2;
    Embedding* player_embed;
    float* proj_buffer;
    Linear* proj;
    ReLU* proj_relu;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

MMONet* init_mmonet(Weights* weights, int num_agents) {
    MMONet* net = calloc(1, sizeof(MMONet));
    int hidden = 256;
    net->num_agents = num_agents;
    net->ob_map = calloc(num_agents*11*15*59, sizeof(float));
    net->ob_player_discrete = calloc(num_agents*47, sizeof(int));
    net->ob_player_continuous = calloc(num_agents*47, sizeof(float));
    net->ob_reward = calloc(num_agents*10, sizeof(float));
    net->map_conv1 = make_conv2d(weights, num_agents, 15, 11, 59, 64, 5, 3);
    net->map_relu = make_relu(num_agents, 64*3*4);
    net->map_conv2 = make_conv2d(weights, num_agents, 4, 3, 64, 64, 3, 1);
    net->player_embed = make_embedding(weights, num_agents*47, 128, 32);
    net->proj_buffer = calloc(num_agents*1689, sizeof(float));
    net->proj = make_linear(weights, num_agents, 1689, hidden);
    net->proj_relu = make_relu(num_agents, hidden);
    net->actor = make_linear(weights, num_agents, hidden, 26);
    net->value_fn = make_linear(weights, num_agents, hidden, 1);
    net->lstm = make_lstm(weights, num_agents, hidden, hidden);
    int logit_sizes[1] = {26};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

void free_mmonet(MMONet* net) {
    free(net->ob_map);
    free(net->ob_player_discrete);
    free(net->ob_player_continuous);
    free(net->ob_reward);
    free(net->map_conv1);
    free(net->map_relu);
    free(net->map_conv2);
    free(net->player_embed);
    free(net->proj_buffer);
    free(net->proj);
    free(net->proj_relu);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net->multidiscrete);
    free(net);
}

void forward(MMONet* net, unsigned char* observations, int* actions) {
    memset(net->ob_map, 0, net->num_agents*11*15*59*sizeof(float));

    // CNN subnetwork
    int factors[10] = {4, 4, 17, 5, 3, 5, 5, 5, 7, 4};
    float (*ob_map)[59][11][15] = (float (*)[59][11][15])net->ob_map;
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b*(11*15*10 + 47 + 10);
        for (int i = 0; i < 11; i++) {
            for (int j = 0; j < 15; j++) {
                int f_offset = 0;
                for (int f = 0; f < 10; f++) {
                    int obs_idx = f_offset + observations[b_offset + i*15*10 + j*10 + f];
                    ob_map[b][obs_idx][i][j] = 1;
                    f_offset += factors[f];
                }
            }
        }
    }
    conv2d(net->map_conv1, net->ob_map);
    relu(net->map_relu, net->map_conv1->output);
    conv2d(net->map_conv2, net->map_relu->output);

    // Player embedding subnetwork
    for (int b = 0; b < net->num_agents; b++) {
        for (int i = 0; i < 47; i++) {
            unsigned char ob = observations[b*(11*15*10 + 47 + 10) + 11*15*10 + i];
            net->ob_player_discrete[b*47 + i] = ob;
            net->ob_player_continuous[b*47 + i] = ob;
        }
    }
    embedding(net->player_embed, net->ob_player_discrete);

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b*1689;
        for (int i = 0; i < 128; i++) {
            net->proj_buffer[b_offset + i] = net->map_conv2->output[b*128 + i];
        }

        b_offset += 128;
        for (int i = 0; i < 47*32; i++) {
            net->proj_buffer[b_offset + i] = net->player_embed->output[b*47*32 + i];
        }

        b_offset += 47*32;
        for (int i = 0; i < 47; i++) {
            net->proj_buffer[b_offset + i] = net->ob_player_continuous[b*47 + i];
        }

        b_offset += 47;
        for (int i = 0; i < 10; i++) {
            net->proj_buffer[b_offset + i] = net->ob_reward[b*10 + i];
        }
    }

    linear(net->proj, net->proj_buffer);
    relu(net->proj_relu, net->proj->output);

    lstm(net->lstm, net->proj_relu->output);

    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);

    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}

void demo(int num_players) {
    Weights* weights = load_weights("resources/nmmo3/nmmo_2025.bin", 1101403);
    MMONet* net = init_mmonet(weights, num_players);

    MMO env = {
        .client = NULL,
        .width = 512,
        .height = 512,
        .num_players = num_players,
        .num_enemies = 2048,
        .num_resources = 2048,
        .num_weapons = 1024,
        .num_gems = 512,
        .tiers = 5,
        .levels = 40,
        .teleportitis_prob = 0.0,
        .enemy_respawn_ticks = 2,
        .item_respawn_ticks = 100,
        .x_window = 7,
        .y_window = 5,
    };
    allocate_mmo(&env);

    c_reset(&env);
    c_render(&env, 0);

    int human_action = ATN_NOOP;
    bool human_mode = false;
    int i = 0;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_LEFT_CONTROL)) {
            human_mode = !human_mode;
        }
        if (i % 36 == 0) {
            forward(net, env.observations, env.actions);
            if (human_mode) {
                env.actions[0] = human_action;
            }

            c_step(&env);
            //printf("Reward: %f\n\tDeath: %f\n\tProf: %f\n\tComb: %f\n\tItem: %f\n", env.rewards[0].death, env.rewards[0].death, env.rewards[0].prof_lvl, env.rewards[0].comb_lvl, env.rewards[0].item_atk_lvl);
            human_action = ATN_NOOP;
        } else {
            int atn = c_render(&env, i/36.0f);
            if (atn != ATN_NOOP) {
                human_action = atn;
            }
        }
        i = (i + 1) % 36;
    }

    free_mmonet(net);
    free(weights);
    free_allocated_mmo(&env);
    //close_client(client);
}

void test_mmonet_performance(int num_players, int timeout) {
    Weights* weights = load_weights("nmmo3_weights.bin", 1101403);
    MMONet* net = init_mmonet(weights, num_players);

    MMO env = {
        .width = 512,
        .height = 512,
        .num_players = num_players,
        .num_enemies = 128,
        .num_resources = 32,
        .num_weapons = 32,
        .num_gems = 32,
        .tiers = 5,
        .levels = 7,
        .teleportitis_prob = 0.001,
        .enemy_respawn_ticks = 10,
        .item_respawn_ticks = 200,
        .x_window = 7,
        .y_window = 5,
    };
    allocate_mmo(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        forward(net, env.observations, env.actions);
        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_players * num_steps / (end - start);
    printf("Test Environment Performance FPS: %f\n", sps);
    free_allocated_mmo(&env);
    free_mmonet(net);
    free(weights);
}

void copy_cast(float* input, unsigned char* output, int width, int height) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            output[adr] = 255*input[adr];
        }
    }
}

void raylib_grid(unsigned char* grid, int width, int height, int tile_size) {
    InitWindow(width*tile_size, height*tile_size, "Raylib Grid");
    SetTargetFPS(1);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                int adr = r*width + c;
                unsigned char val = grid[adr];

                Color color = (Color){val, val, val, 255};

                int x = c*tile_size;
                int y = r*tile_size;
                DrawRectangle(x, y, tile_size, tile_size, color);
            } }
        EndDrawing();
    }
    CloseWindow();
}

void raylib_grid_colored(unsigned char* grid, int width, int height, int tile_size) {
    InitWindow(width*tile_size, height*tile_size, "Raylib Grid");
    SetTargetFPS(1);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                int adr = 3*(r*width + c);
                unsigned char red = grid[adr];
                unsigned char green = grid[adr+1];
                unsigned char blue = grid[adr+2];

                Color color = (Color){red, green, blue, 255};

                int x = c*tile_size;
                int y = r*tile_size;
                DrawRectangle(x, y, tile_size, tile_size, color);
            }
        }
        EndDrawing();
    }
    CloseWindow();
}

void test_perlin_noise(int width, int height,
        float base_frequency, int octaves, int seed) {
    float terrain[width*height];
    perlin_noise((float*)terrain, width, height, base_frequency, octaves, seed, seed);

    unsigned char map[width*height];
    copy_cast((float*)terrain, (unsigned char*)map, width, height);
    raylib_grid((unsigned char*)map, width, height, 1024.0/width);
}

void test_flood_fill(int width, int height, int colors) {
    unsigned char unfilled[width][height];
    memset(unfilled, 0, width*height);

    // Draw some squares
    for (int i = 0; i < 32; i++) {
        int w = rand() % width/4;
        int h = rand() % height/4;
        int start_r = rand() % (3*height/4);
        int start_c = rand() % (3*width/4);
        int end_r = start_r + h;
        int end_c = start_c + w;
        for (int r = start_r; r < end_r; r++) {
            unfilled[r][start_c] = 1;
            unfilled[r][end_c] = 1;
        }
        for (int c = start_c; c < end_c; c++) {
            unfilled[start_r][c] = 1;
            unfilled[end_r][c] = 1;
        }
    }

    char filled[width*height];
    flood_fill((unsigned char*)unfilled, (char*)filled,
        width, height, colors, width*height);

    // Cast and colorize
    unsigned char output[width*height];
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            int val = filled[adr];
            if (val == 0) {
                output[adr] = 0;
            }
            output[adr] = 128 + (128/colors)*val;
        }
    }

    raylib_grid((unsigned char*)output, width, height, 1024.0/width);
}

void test_cellular_automata(int width, int height, int colors, int max_fill) {
    char grid[width][height];
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            grid[r][c] = -1;
        }
    }

    // Fill some squares
    for (int i = 0; i < 32; i++) {
        int w = rand() % width/4;
        int h = rand() % height/4;
        int start_r = rand() % (3*height/4);
        int start_c = rand() % (3*width/4);
        int end_r = start_r + h;
        int end_c = start_c + w;
        int color = rand() % colors;
        for (int r = start_r; r < end_r; r++) {
            for (int c = start_c; c < end_c; c++) {
                grid[r][c] = color;
            }
        }
    }

    cellular_automata((char*)grid, width, height, colors, max_fill);

    // Colorize
    unsigned char output[width*height];
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int val = grid[r][c];
            int adr = r*width + c;
            if (val == 0) {
                output[adr] = 0;
            }
            output[adr] = (255/colors)*val;
        }
    }

    raylib_grid((unsigned char*)output, width, height, 1024.0/width);
}

void test_generate_terrain(int width, int height, int x_border, int y_border) {
    char terrain[width][height];
    unsigned char rendered[width][height][3];
    generate_terrain((char*)terrain, (unsigned char*)rendered, width, height, x_border, y_border);


    // Colorize
    /*
    unsigned char output[width*height];
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int val = terrain[r][c];
            int adr = r*width + c;
            if (val == 0) {
                output[adr] = 0;
            }
            output[adr] = (255/4)*val;
        }
    }
    */

    raylib_grid_colored((unsigned char*)rendered, width, height, 1024.0/width);
}

void test_performance(int num_players, int timeout) {
    MMO env = {
        .width = 512,
        .height = 512,
        .num_players = num_players,
        .num_enemies = 128,
        .num_resources = 32,
        .num_weapons = 32,
        .num_gems = 32,
        .tiers = 5,
        .levels = 7,
        .teleportitis_prob = 0.001,
        .enemy_respawn_ticks = 10,
        .item_respawn_ticks = 200,
        .x_window = 7,
        .y_window = 5,
    };
    allocate_mmo(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        for (int i = 0; i < num_players; i++) {
            env.actions[i] = rand() % 23;
        }
        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_players * num_steps / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free_allocated_mmo(&env);
}

int main() {

    /*
    int width = 512;
    int height = 512;
    float base_frequency = 1.0/64.0;
    int octaves = 2;
    int seed = 0;
    test_perlin_noise(width, height, base_frequency, octaves, seed);
    test_flood_fill(width, height, 4);
    test_cellular_automata(width, height, 4, 4000);
    test_generate_terrain(width, height, 8, 8);
    */
    //test_performance(64, 10);
    demo(NUM_AGENTS);
    //test_mmonet_performance(1024, 10);
}
