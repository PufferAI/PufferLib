#include "tactical.h"


int main() {
    Tactical* env = init_tactical();
    // allocate(&env);
    
    env->client = init_client(env);

    c_reset(env);
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_Q) || IsKeyPressed(KEY_BACKSPACE)) break;
        c_step(env);
        c_render(env);
    }

    close_client(env->client);
    free_tactical(env);
    
    // free_linearlstm(net);
    // free(weights);
    // free_allocated(&env);
    // close_client(client);
}

