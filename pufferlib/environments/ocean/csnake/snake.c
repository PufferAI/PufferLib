#include "snake.h"

int main() {
    int num_snakes = 128;
    int width = 160;
    int height = 160;
    int max_snake_length = 200;
    int food = 128;
    int vision = 5;
    bool leave_corpse_on_death = true;
    float reward_food = 1.0f;
    float reward_corpse = 0.5f;
    float reward_death = -1.0f;

    int render_cell_size = 5;

    int batch_size = num_snakes;
    int input_dim = 11*11;
    int hidden_dim = 128;
    int output_dim = 4;

    MLP* mlp = allocate_mlp(batch_size, input_dim, hidden_dim, output_dim);
    int err = load_weights("flat_snake.pt", mlp);
    if (err)
        return 1;

    CSnake* env = allocate_csnake(num_snakes, width, height, max_snake_length, 
        food, vision, leave_corpse_on_death, reward_food, reward_corpse, reward_death);
    Renderer* renderer = init_renderer(render_cell_size, width, height);

    reset(env);
    while (!WindowShouldClose()) {
        for (int i = 0; i < batch_size * input_dim; i++)
			mlp->observations[i] = (float)env->observations[i];

		mlp_forward(mlp, env->actions);

        // User can take control of the first snake
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env->actions[0] = 0;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env->actions[0] = 1;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env->actions[0] = 2;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env->actions[0] = 3;

        step(env);
        render(renderer, env);
    }
    close_renderer(renderer);
    free_csnake(env);
    free_mlp(mlp);
    return 0;
}

