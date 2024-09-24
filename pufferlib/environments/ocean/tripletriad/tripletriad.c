#include "tripletriad.h"

int main() {
    int width = 990;
    int height = 1000;
    int board_width = 576;
    int board_height = 672;
    int piece_width = board_width /3;
    int piece_height = board_height / 3;
    int game_over = 0;
    int num_cards = 10;
    CTripleTriad* env = allocate_ctripletriad(width, height, piece_width, piece_height,
        game_over,num_cards);
    reset(env); 
 
    Client* client = make_client(width, height);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env->actions[0] = 0;
        // Handle Card Selection ( 1-5 for selecting a card)
        if (IsKeyPressed(KEY_ONE)) env->actions[0] = 1;
        if (IsKeyPressed(KEY_TWO)) env->actions[0] = 2;
        if (IsKeyPressed(KEY_THREE)) env->actions[0] = 3;
        if (IsKeyPressed(KEY_FOUR)) env->actions[0] = 4;
        if (IsKeyPressed(KEY_FIVE)) env->actions[0] = 5;

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
            int cellX = relativeX / env->card_width;
            int cellY = relativeY / env->card_height;
            
            // Calculate the cell index (1-9) based on the click position
            int cellIndex = cellY * 3 + cellX+1; 
            
            // Ensure the click is within the game board
            if (cellX >= 0 && cellX < 3 && cellY >= 0 && cellY < 3) {
                env->actions[0] = cellIndex + 5;
            }
        }



        step(env);
        render(client, env);
    }
    close_client(client);
    free_allocated_ctripletriad(env);

    return 0;
}
