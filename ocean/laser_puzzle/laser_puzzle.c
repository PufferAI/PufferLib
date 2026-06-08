#include "laser_puzzle.h"
#include "puffernet.h"

#define WEIGHTS_PATH "resources/laser_puzzle/laser_puzzle_weights.bin"

static void copy_observations(float* out, const unsigned char* in) {
    for (int i = 0; i < LASER_PUZZLE_OBS_SIZE; i++) {
        out[i] = (float)in[i];
    }
}

int demo() {
    Weights* weights = load_weights(WEIGHTS_PATH);
    if (weights == NULL) {
        return 1;
    }

    int logit_sizes[1] = {NUM_ACTIONS};
    PufferNet* net = make_puffernet(
        weights, 1, LASER_PUZZLE_OBS_SIZE, 128, 2, logit_sizes, 1);
    float observations[LASER_PUZZLE_OBS_SIZE] = {0};

    LaserPuzzle env = {0};

    // allocate memory, initialize the client
    allocate(&env);
    c_reset(&env);
    env.client = make_client();

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
        }

        if (IsKeyDown(KEY_LEFT_SHIFT) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            int gridWidth = env.COLS * CELL_SIZE;
            int gridHeight = env.ROWS * CELL_SIZE;
            int offsetX = (GetScreenWidth() - gridWidth) / 2;
            int offsetY = (GetScreenHeight() - gridHeight) / 2;

            Vector2 mouse = GetMousePosition();
            int c = (mouse.x - offsetX) / CELL_SIZE;
            int r = (mouse.y - offsetY) / CELL_SIZE;

            if (r >= 1 && r < env.ROWS - 1 && c >= 1 && c < env.COLS - 1) {
                Cell* cell = &env.board[BOARD_IDX(env.COLS, r, c)];
                int mirror_action = (cell->mirror + 1) % ACTIONS_PER_CELL;
                int cell_idx = (r - 1) * INNER_COLS + (c - 1);
                env.actions[0] = (float)(cell_idx * ACTIONS_PER_CELL + mirror_action);
                c_step(&env);
            }
        } else {
            copy_observations(observations, env.observations);
            forward_puffernet(net, observations, env.actions);
            c_step(&env);
        }

        c_render(&env);
    }

    free_puffernet(net);
    free(weights);

    // call closing procedures
    c_close(&env);
    return 0;
}

int main() {
    demo();
    return 0;
}
