#include "snake.h"
#include "puffernet.h"

int main() {
    CSnake env = {
        .num_snakes = 64,
        .width = 135,
        .height = 90,
        .max_snake_length = 200,
        .food = 256,
        .vision = 5,
        .leave_corpse_on_death = true,
        .reward_food = 1.0f,
        .reward_corpse = 0.5f,
        .reward_death = -1.0f,
    };
    allocate_csnake(&env);

    int batch_size = env.num_snakes;
    int input_dim = 11*11;
    int output_dim = 4;

    //Weights* weights = load_weights("resources/snake_weights.bin", 133764);
    //LinearLSTM* net = make_linearlstm(weights, batch_size, input_dim, output_dim);

    int render_cell_size = 8;
    Client* client = make_client(render_cell_size, env.width, env.height);

    reset(&env);
    while (!WindowShouldClose()) {
        // User can take control of the first snake
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 0;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 1;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 2;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 3;
        } else {
            /*
            for (int i = 0; i < batch_size*input_dim; i++) {
                net->obs[i] = (float)env.observations[i];
            }
            forward_linearlstm(net, net->obs, env.actions);
            */
        }
        step(&env);
        render(client, &env);
    }
    //free_linearlstm(net);
    //free(weights);
    close_client(client);
    free_csnake(&env);
    return 0;
}

