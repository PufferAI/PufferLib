#include "snake.h"

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

    int render_cell_size = 8;
    Client* client = make_client(render_cell_size, env.width, env.height);

    int batch_size = env.num_snakes;
    int input_dim = 11*11;
    int hidden_dim = 128;
    int output_dim = 4;
    MLP* mlp = allocate_mlp(batch_size, input_dim, hidden_dim, output_dim);
    if (load_weights("resources/snake_weights.bin", mlp))
        return 1;

    reset(&env);
    while (!WindowShouldClose()) {
        for (int i = 0; i < batch_size*input_dim; i++)
			mlp->observations[i] = (float)env.observations[i];

		mlp_forward(mlp, env.actions);

        // User can take control of the first snake
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 0;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 1;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 2;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 3;

        step(&env);
        render(client, &env);
    }
    close_client(client);
    free_csnake(&env);
    free_mlp(mlp);
    return 0;
}

