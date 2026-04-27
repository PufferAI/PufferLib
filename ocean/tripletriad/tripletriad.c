#include "tripletriad.h"
#include "puffernet.h"
#include <time.h>

#define NOOP -1

void demo() {
    Weights* weights = load_weights("resources/tripletriad/tripletriad_weights.bin");
    int logit_sizes[1] = {14};
    PufferNet* net = make_puffernet(weights, 1, 114, 256, 2, logit_sizes, 1);

    CTripleTriad env = {
        .width = 990,
        .height = 690,
        .card_width = 576 / 3,
        .card_height = 672 / 3,
        .game_over = 0,
        .num_cards = 10,
    };
    allocate_ctripletriad(&env);
    c_reset(&env); 
    env.client = make_client(env.width, env.height);

    int tick = 0;
    int action;
    while (!WindowShouldClose()) {
        action = NOOP;

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyPressed(KEY_ONE)) action = SELECT_CARD_1;
            if (IsKeyPressed(KEY_TWO)) action = SELECT_CARD_2;
            if (IsKeyPressed(KEY_THREE)) action = SELECT_CARD_3;
            if (IsKeyPressed(KEY_FOUR)) action = SELECT_CARD_4;
            if (IsKeyPressed(KEY_FIVE)) action = SELECT_CARD_5;

            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                Vector2 mousePos = GetMousePosition();
                int boardOffsetX = 196 + 10;
                int boardOffsetY = 30;
                int relativeX = mousePos.x - boardOffsetX;
                int relativeY = mousePos.y - boardOffsetY;
                int cellX = relativeX / env.card_width;
                int cellY = relativeY / env.card_height;
                int cellIndex = cellY * 3 + cellX + 1; 
                if (cellX >= 0 && cellX < 3 && cellY >= 0 && cellY < 3) {
                    action = cellIndex + 4;
                }
            }
        } else if (tick % 45 == 0) {
            forward_puffernet(net, env.observations, env.actions);
            action = env.actions[0];
        }

        tick = (tick + 1) % 45;

        if (action != NOOP) {
            env.actions[0] = action;
            c_step(&env);
        }

        c_render(&env);
    }
    free_puffernet(net);
    free(weights);
    close_client(env.client);
    free_allocated_ctripletriad(&env);
}

int main() {
    demo();
    return 0;
}
