#include <stdlib.h>
#include <math.h>
#include "raylib.h"
#include <stdio.h>

#define NOOP 0
#define SELECT_CARD_1 1
#define SELECT_CARD_2 2
#define SELECT_CARD_3 3
#define SELECT_CARD_4 4
#define SELECT_CARD_5 5
#define PLACE_CARD_1 6
#define PLACE_CARD_2 7
#define PLACE_CARD_3 8
#define PLACE_CARD_4 9
#define PLACE_CARD_5 10
#define PLACE_CARD_6 11
#define PLACE_CARD_7 12
#define PLACE_CARD_8 13
#define PLACE_CARD_9 14
#define WIN_CONDITION 4
#define Y_OFFSET 10
#define TICK_RATE 1.0f/60.0f

// how to start game compile - LD_LIBRARY_PATH=raylib-5.0_linux_amd64/lib ./tripletriadgame 

typedef struct CTripleTriad CTripleTriad;
struct CTripleTriad {
    float* observations;
    unsigned char* actions;
    float* rewards;
    unsigned char* dones;
    int card_width;
    int card_height;
    float* board_x;
    float* board_y;
    int board_states[3][3];
    int width;
    int height;
    int game_over;
    int num_cards;
    int*** cards_in_hand;
    int* card_selected;
    int** card_locations;
    int* action_masks;
};

void generate_board_positions(CTripleTriad* env) {
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            int idx = row * 3 + col;
            env->board_x[idx] = col* env->card_width;
            env->board_y[idx] = row*env->card_height;
        }
    }
}

void generate_cards_in_hand(CTripleTriad* env) {
    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            for(int k=0; k< 4; k++) {
                env->cards_in_hand[i][j][k] = rand() % 7;
            }
        }
    }
}



void generate_card_locations(CTripleTriad* env) {
    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            env->card_locations[i][j] = 0;
        }
    }
}

void generate_card_selections(CTripleTriad* env) {
    for(int i=0; i< 2; i++) {
        env->card_selected[i] = -1;
    }
}



CTripleTriad* init_ctripletriad( unsigned char* actions,
        float* observations, float* rewards, unsigned char* dones,
        int width, int height, int card_width, int card_height, int game_over, int num_cards) {

    CTripleTriad* env = (CTripleTriad*)calloc(1, sizeof(CTripleTriad));

    env->actions = actions;
    env->observations = observations;
    env->rewards = rewards;
    env->dones = dones;
    env->width = width;
    env->height = height;
    env->card_width = card_width;
    env->card_height = card_height;
    env->game_over = game_over;
    // Allocate memory for board_x, board_y, and board_states
    env->board_x = (float*)calloc(9, sizeof(float));
    env->board_y = (float*)calloc(9, sizeof(float));
    env->cards_in_hand = (int***)calloc(2, sizeof(int**));
    env->card_selected = (int*)calloc(2, sizeof(int));
    env->card_locations = (int**)calloc(2, sizeof(int*));
    env->action_masks = (int*)calloc(15, sizeof(int));
    for(int i=0; i< 2; i++) {
        env->cards_in_hand[i] = (int**)calloc(5, sizeof(int*));
        for(int j=0; j< 5; j++) {
            env->cards_in_hand[i][j] = (int*)calloc(4, sizeof(int));
        }
    }
    for(int i=0; i< 3; i++) {
        for(int j=0; j< 3; j++) {
            env->board_states[i][j] = 0;
        }
    }
    for(int i=0; i< 2; i++) {
        env->card_locations[i] = (int*)calloc(5, sizeof(int));
    }
    env->num_cards = num_cards;

    generate_board_positions(env);
    generate_cards_in_hand(env);
    generate_card_locations(env);
    generate_card_selections(env);
    return env;
}

CTripleTriad* allocate_ctripletriad(int width, int height,
    int card_width, int card_height, int game_over, int num_cards) {

    unsigned char* actions = (unsigned char*)calloc(1, sizeof(unsigned char));
    float* observations = (float*)calloc(width * height, sizeof(float));
    unsigned char* dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    float* rewards = (float*)calloc(1, sizeof(float));

    CTripleTriad* env = init_ctripletriad(actions,
        observations, rewards, dones, width, height,
            card_width, card_height, game_over, num_cards);

    return env;
}

void free_ctripletriad(CTripleTriad* env) {
    free(env->board_x);
    free(env->board_y);
    free(env->cards_in_hand);
    free(env->card_locations);
    free(env->card_selected);
    free(env->action_masks);
    free(env);
}

void free_allocated_ctripletriad(CTripleTriad* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_ctripletriad(env);
}

void compute_observations(CTripleTriad* env) {
    int idx=0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            env->observations[idx] = env->board_states[i][j];
            idx++;
        }
    }
}



void reset(CTripleTriad* env) {

    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            for(int k=0; k< 4; k++) {
                env->cards_in_hand[i][j][k] = rand() % 7;
            }
        }
    }
    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            env->card_locations[i][j] = 0;
        }
    }
    for(int i=0; i< 2; i++) {
        env->card_selected[i] = -1;
    }
    for(int i=0; i< 3; i++) {
        for(int j=0; j< 3; j++) {
            env->board_states[i][j] = 0;
        }
    }
    env->dones[0] = 0;
}

void select_card(CTripleTriad* env, int card_selected, int player) {
    int player_idx = (player == 1) ? 0 : 1;
    env->card_selected[player_idx] = card_selected-1;

    // print the selected card
    // printf("Player: %d\n", player);
    // printf("Card selected: %d\n", env->card_selected[player_idx]);
    // printf("Card Values: %d %d %d %d\n", env->cards_in_hand[player_idx][env->card_selected[player_idx]][0],
    //     env->cards_in_hand[player_idx][env->card_selected[player_idx]][1],
    //     env->cards_in_hand[player_idx][env->card_selected[player_idx]][2],
    //     env->cards_in_hand[player_idx][env->card_selected[player_idx]][3]);

}

void place_card(CTripleTriad* env, int card_placement, int player) {
    int player_idx = (player == 1) ? 0 : 1;
    env->card_locations[player_idx][env->card_selected[player_idx]] = card_placement;
    env->board_states[(card_placement-1)/3][(card_placement-1)%3] = player;

    // printf("Player: %d\n", player);
    // printf("Board States: %d %d %d %d %d %d %d %d %d\n", env->board_states[0][0], env->board_states[0][1], env->board_states[0][2],
    //     env->board_states[1][0], env->board_states[1][1], env->board_states[1][2],
    //     env->board_states[2][0], env->board_states[2][1], env->board_states[2][2]);
    // // print the card value
    // printf("Team: %d\n", player);
    // printf("Card selected: %d\n", env->card_selected[player_idx]);
    // printf("Card placed at: %d\n", card_placement);
    // printf("Card Values: %d %d %d %d\n", env->cards_in_hand[player_idx][env->card_selected[player_idx]][0],
    //     env->cards_in_hand[player_idx][env->card_selected[player_idx]][1],
    //     env->cards_in_hand[player_idx][env->card_selected[player_idx]][2],
    //     env->cards_in_hand[player_idx][env->card_selected[player_idx]][3]);
}

void update_action_masks(CTripleTriad* env) {
    for (int i =0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            if (env->card_locations[i][j] != 0) {
                int action_idx = env->card_locations[i][j] + 5;
                env->action_masks[action_idx] = 1;
            }
        }
    }
    printf("Action masks: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", env->action_masks[0], env->action_masks[1], env->action_masks[2],
        env->action_masks[3], env->action_masks[4], env->action_masks[5], env->action_masks[6], env->action_masks[7],
        env->action_masks[8], env->action_masks[9], env->action_masks[10], env->action_masks[11], env->action_masks[12], env->action_masks[13], env->action_masks[14]);
}





void check_win_condition(CTripleTriad* env, int player) {
    int count = 0;
    for (int i=0; i< 3; i++) {
        for (int j=0; j< 3; j++) {
            if (env->board_states[i][j] == player) {
                count++;
            } 
        }
    }
    if (count == 5) {
        env->dones[0] = 1;
        env->rewards[0] = player; // 1 for player win, -1 for opponent win
        printf("Player %d wins!\n", player);
    }
    return;
    
}

void step(CTripleTriad* env) {
    env->rewards[0] = 0.0;
    int action = env->actions[0];
    // reset the game if game over
    if (env->game_over == 1) {
        reset(env);
        env->game_over = 0;
        return;
    }
    // select a card if the card is in the range of 1-5 and the card is not placed
    if (action >= SELECT_CARD_1 && action <= SELECT_CARD_5 ) {
        int card_selected = action;
        if(env->card_locations[0][card_selected-1] == 0) {
            select_card(env,card_selected, 1);
        }

    }
    // place a card if the card is in the range of 1-9 and the card is selected
    else if (action >= PLACE_CARD_1 && action <= PLACE_CARD_9 ) {
        int card_placement = action -5;
        if(env->card_selected[0] >= 0) {
            place_card(env,card_placement, 1);
            check_win_condition(env, 1);
            update_action_masks(env);
            env->card_selected[0] = -1;
        }

        // opponent turn 
        if (env->dones[0] == 0) {
            int bot_card_selected = rand() % 5 + 1;
            select_card(env,bot_card_selected, -1);
            int bot_card_placement = rand() % 8 + 1;
            place_card(env,bot_card_placement, -1);
            check_win_condition(env, -1);
            update_action_masks(env);
            env->card_selected[1] = -1;
        }
    }
    if (env->dones[0] == 1) {
        env->game_over=1;
    }

    compute_observations(env);
}


typedef struct Client Client;
struct Client {
    float width;
    float height;
};

Client* make_client(int width, int height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;

    InitWindow(width, height, "PufferLib Ray Connect4");
    SetTargetFPS(15);

    //sound_path = os.path.join(*self.__module__.split(".")[:-1], "hit.wav")
    //self.sound = rl.LoadSound(sound_path.encode())

    return client;
}

void render(Client* client, CTripleTriad* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});


    // create 3x3 board for triple triad
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            int board_idx = row * 3 + col;
            Color piece_color=PURPLE;

            if (env->board_states[row][col] == 0.0) {
                piece_color = BLACK;
            } else if (env->board_states[row][col] == 1.0) {
                piece_color = RED;
            } else if (env->board_states[row][col] == -1.0) {
                piece_color = BLUE;
            }
            int x = env->board_x[board_idx];
            int y = env->board_y[board_idx];
            Color board_color = DARKBLUE;
            DrawRectangle(x+196+10 , y+30 , env->card_width, env->card_height, piece_color);
            DrawRectangleLines(x+196+10 , y+30 , env->card_width, env->card_height, WHITE);

            
        }
    }
    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            // starting locations for cards in hand
            int card_x = (i == 0) ? 10 : (env->width - env->card_width - 10);
            int card_y = 30 + env->card_height/2*j;

            // locations if card is placed
            if (env->card_locations[i][j] != 0) {
                card_x = env->board_x[env->card_locations[i][j]-1] + 196 + 10;
                card_y = env->board_y[env->card_locations[i][j]-1] + 30;
            }
            // Draw card background
            Color card_color = (i == 0) ? RED : BLUE;
            DrawRectangle(card_x, card_y, env->card_width, env->card_height, card_color);
            // change background if card is selected, highlight it
            if (env->card_selected[i] == j) {
                DrawRectangleLines(card_x, card_y, env->card_width, env->card_height, YELLOW);
            } else {
                DrawRectangleLines(card_x, card_y, env->card_width, env->card_height, WHITE);
            }
        
            for(int k=0; k< 4; k++) {
                char str[20];
                sprintf(str, "%d", env->cards_in_hand[i][j][k]);
                // calculate positiion of x and y offsets
                int x_offset, y_offset;
                switch(k) {
                    case 0: // North (top)
                        x_offset = card_x + 25;
                        y_offset = card_y + 5;
                        break;
                    case 1: // South (bottom)
                        x_offset = card_x + 25;
                        y_offset = card_y + 45;
                        break;
                    case 2: // East (right)
                        x_offset = card_x + 45;
                        y_offset = card_y + 25;
                        break;
                    case 3: // West (left)
                        x_offset = card_x + 5;
                        y_offset = card_y + 25;
                        break;
                }

                Color text_color = (i == 0) ? YELLOW : WHITE;
                DrawText(str, x_offset, y_offset, 20, text_color);
            }
        }
    }
    DrawText("Triple Triad", 20, 10, 20, WHITE);

    // give instructions to player 1: 
    DrawText("Use 1-5 to select a card", 20, env->height - 70, 20, WHITE);
    DrawText("Click an empty space on the board to place a card", 20, env->height - 40, 20, WHITE);
    EndDrawing();

    //PlaySound(client->sound);
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}
