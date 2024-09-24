#include "connect4.h"

int main() {
    int width = 672;
    int height = 576;
    int piece_width = 96;
    int piece_height = 96;
    int longest_connected = 4;
    int game_over = 0;
    int pieces_placed = 0;

    CConnect4* env = allocate_cconnect4(width, height, piece_width, piece_height,
        longest_connected, game_over, pieces_placed);
    reset(env);
 
    Client* client = make_client(width, height);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env->actions[0] = 0;
        // user inputs 1 - 7 key pressed
        if(IsKeyPressed(KEY_ONE)) env->actions[0] = 1;
        if(IsKeyPressed(KEY_TWO)) env->actions[0] = 2;
        if(IsKeyPressed(KEY_THREE)) env->actions[0] = 3;
        if(IsKeyPressed(KEY_FOUR)) env->actions[0] = 4;
        if(IsKeyPressed(KEY_FIVE)) env->actions[0] = 5;
        if(IsKeyPressed(KEY_SIX)) env->actions[0] = 6;
        if(IsKeyPressed(KEY_SEVEN)) env->actions[0] = 7;

        step(env);
        render(client, env);
    }
    close_client(client);
    free_allocated_cconnect4(env);
    return 0;
}

