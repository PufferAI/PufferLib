#include "connect4.h"

int main() {
    CConnect4 env = {
        .width = 672,
        .height = 576,
        .piece_width = 96,
        .piece_height = 96,
        .game_over = 0,
        .pieces_placed = 0,
    };
    allocate_cconnect4(&env);
    reset(&env);
 
    Client* client = make_client(env.width, env.height);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env.actions[0] = 0;
        // user inputs 1 - 7 key pressed
        if(IsKeyPressed(KEY_ONE)) env.actions[0] = 1;
        if(IsKeyPressed(KEY_TWO)) env.actions[0] = 2;
        if(IsKeyPressed(KEY_THREE)) env.actions[0] = 3;
        if(IsKeyPressed(KEY_FOUR)) env.actions[0] = 4;
        if(IsKeyPressed(KEY_FIVE)) env.actions[0] = 5;
        if(IsKeyPressed(KEY_SIX)) env.actions[0] = 6;
        if(IsKeyPressed(KEY_SEVEN)) env.actions[0] = 7;

        if (env.actions[0] != 0) {
            step(&env);
        }
        render(client, &env);
    }
    close_client(client);
    free_allocated_cconnect4(&env);
    return 0;
}

