#include "tripletriad.h"

int main() {
    CTripleTriad env = {
        .width = 990,
        .height = 1000,
        .card_width = 576 / 3,
        .card_height = 672 / 3,
        .game_over = 0,
        .num_cards = 10,
    };
    allocate_ctripletriad(&env);
    reset(&env); 
 
    Client* client = make_client(env.width, env.height);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env.actions[0] = 0;
        // Handle Card Selection ( 1-5 for selecting a card)
        if (IsKeyPressed(KEY_ONE)) env.actions[0] = 1;
        if (IsKeyPressed(KEY_TWO)) env.actions[0] = 2;
        if (IsKeyPressed(KEY_THREE)) env.actions[0] = 3;
        if (IsKeyPressed(KEY_FOUR)) env.actions[0] = 4;
        if (IsKeyPressed(KEY_FIVE)) env.actions[0] = 5;

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
                env.actions[0] = cellIndex + 5;
            }
        }

        step(&env);
        printf("Reward: %f, Tick: %f\n", env.rewards[0], env.log.episode_length);
        render(client, &env);
    }
    close_client(client);
    free_allocated_ctripletriad(&env);

    return 0;
}
