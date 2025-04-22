#include "tripletriad.h"
#include "puffernet.h"

int main() {
    Weights* weights = load_weights("resources/tripletriad_weights.bin", 148880);
    LinearLSTM* net = make_linearlstm(weights, 1, 114, 15);

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
    Client* client = make_client(env.width, env.height);

    int tick = 0;
    int action;
    while (!WindowShouldClose()) {
        action = -1;

        // User can take control of the player
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Handle Card Selection ( 1-5 for selecting a card)
            if (IsKeyPressed(KEY_ONE)) action = SELECT_CARD_1;
            if (IsKeyPressed(KEY_TWO)) action = SELECT_CARD_2;
            if (IsKeyPressed(KEY_THREE)) action = SELECT_CARD_3;
            if (IsKeyPressed(KEY_FOUR)) action = SELECT_CARD_4;
            if (IsKeyPressed(KEY_FIVE)) action = SELECT_CARD_5;

            // Handle Card Placement ( 1-9 for placing a card)
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                Vector2 mousePos = GetMousePosition();
        
                // Calculate the offset for the board
                int boardOffsetX = 196 + 10; // 196 from the DrawRectangle call in render(), plus 10 for padding
                int boardOffsetY = 30; // From the DrawRectangle call in render()
                
                // Adjust mouse position relative to the board
                int relativeX = mousePos.x - boardOffsetX;
                int relativeY = mousePos.y - boardOffsetY;
                
                // Calculate cell indices
                int cellX = relativeX / env.card_width;
                int cellY = relativeY / env.card_height;
                
                // Calculate the cell index (1-9) based on the click position
                int cellIndex = cellY * 3 + cellX+1; 
                
                // Ensure the click is within the game board
                if (cellX >= 0 && cellX < 3 && cellY >= 0 && cellY < 3) {
                    action = cellIndex + 4;
                }
            }
        } else if (tick % 45 == 0) {
            forward_linearlstm(net, env.observations, env.actions);
            action = env.actions[0];
        }

        tick = (tick + 1) % 45;
        if (env.actions[0] != NOOP) {
            c_step(&env);
        }

        c_render(client, &env);
    }
    free_linearlstm(net);
    free(weights);
    close_client(client);
    free_allocated_ctripletriad(&env);
    return 0;
}
