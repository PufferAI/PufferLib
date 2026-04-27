#include <time.h>
#include "go.h"
#include "puffernet.h"

void demo(int grid_size) {

    CGo env = {
        .width = 950,
        .height = 750,
        .grid_size = grid_size,
        .board_width = 600,
        .board_height = 600,
        .grid_square_size = 64,
        .komi = 7.5,
        .reward_move_pass = -0.518441,
        .reward_move_valid = 0,
        .reward_move_invalid = -0.0864746,
        .reward_player_capture = 0.553628,
        .reward_opponent_capture = -0.102283,
        .selfplay = 0,
        .side = 1,
    };

    Weights* weights = load_weights("resources/go/go_weights.bin");
    int logit_sizes[1] = {grid_size * grid_size + 1};
    int obs_size = grid_size * grid_size * 4 + 2;
    PufferNet* net = make_puffernet(weights, 1, obs_size, 512, 1, logit_sizes, 1);
    allocate(&env);
    c_reset(&env);
    c_render(&env);

    int tick = 0;
    while (!WindowShouldClose()) {
        if(tick % 3 == 0) {
            tick = 0;
            int human_action = env.actions[0];
            forward_puffernet(net, env.observations, env.actions);
            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                env.actions[0] = human_action;
            }
            c_step(&env);
            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                env.actions[0] = -1;
            }
        }
        tick++;
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                Vector2 mousePos = GetMousePosition();

                // Calculate the offset for the board
                int boardOffsetX = env.grid_square_size;
                int boardOffsetY = env.grid_square_size;

                // Adjust mouse position relative to the board
                int relativeX = mousePos.x - boardOffsetX;
                int relativeY = mousePos.y - boardOffsetY;

                // Calculate cell indices for the corners
                int cellX = (relativeX + env.grid_square_size / 2) / env.grid_square_size;
                int cellY = (relativeY + env.grid_square_size / 2) / env.grid_square_size;

                // Ensure the click is within the game board
                if (cellX >= 0 && cellX <= env.grid_size && cellY >= 0 && cellY <= env.grid_size) {
                    // Calculate the point index (1-19) based on the click position
                    int pointIndex = cellY * (env.grid_size) + cellX + 1;
                    env.actions[0] = (unsigned short)pointIndex;
                }
            // Check if pass button is clicked
                int passButtonX = env.width - 300;
                int passButtonY = 200;
                int passButtonWidth = 100;
                int passButtonHeight = 50;

                if (mousePos.x >= passButtonX && mousePos.x <= passButtonX + passButtonWidth &&
                    mousePos.y >= passButtonY && mousePos.y <= passButtonY + passButtonHeight) {
                    env.actions[0] = 0; // Send action 0 for pass
                }
            }
        }
        c_render(&env);
    }
    free_puffernet(net);
    free(weights);
    free_allocated(&env);
}

int main() {
    demo(9);
    return 0;
}
