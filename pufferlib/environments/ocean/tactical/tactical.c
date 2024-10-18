#include "tactical.h"


int main() {
    Tactical* env = init_tactical();
    // allocate(&env);

    GameRenderer* client = init_game_renderer(env);

    reset(env);
    while (true) { // Escape key is used in the game
        if (IsKeyPressed(KEY_Q) || IsKeyPressed(KEY_BACKSPACE)) break;
        step(env);
        render_game(client, env);
    }
    // free_linearlstm(net);
    // free(weights);
    // free_allocated(&env);
    // close_client(client);
}

