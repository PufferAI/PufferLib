#include "go.h"

int main() {
    CGo env = {
        .width = 1200,
        .height = 1200,
        .grid_size = 18,
        .board_width = 1000,
        .board_height = 1000,
        .grid_square_size = 1000/18
    };
    allocate(&env);
    reset(&env);
 
    Client* client = make_client(&env);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env.actions[0] = 0;

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            Vector2 mousePos = GetMousePosition();
    
            // Calculate the offset for the board
            int boardOffsetX = 100; // 196 from the DrawRectangle call in render(), plus 10 for padding
            int boardOffsetY = 100; // From the DrawRectangle call in render()
            
            // Adjust mouse position relative to the board
            int relativeX = mousePos.x - boardOffsetX;
            int relativeY = mousePos.y - boardOffsetY;
            
            // Calculate cell indices for the corners
            int cellX = (relativeX + env.grid_square_size / 2) / env.grid_square_size;
            int cellY = (relativeY + env.grid_square_size / 2) / env.grid_square_size;
            
            // Ensure the click is within the game board
            if (cellX >= 0 && cellX <= env.grid_size && cellY >= 0 && cellY <= env.grid_size) {
                // Calculate the point index (1-19) based on the click position
                int pointIndex = cellY * (env.grid_size + 1) + cellX + 1; 

                printf("pointIndex: %d\n", pointIndex);
                env.actions[0] = (unsigned short)pointIndex;
            }
        }
        step(&env);
        render(client, &env);
    }
    close_client(client);
    free_allocated(&env);
    return 0;
}

