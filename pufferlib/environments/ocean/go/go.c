#include <time.h>
#include "go.h"

void demo() {
    CGo env = {
        .width = 1000,
        .height = 800,
        .grid_size = 9,
        .board_width = 600,
        .board_height = 600,
        .grid_square_size = 600/9,
        .moves_made = 0,
        .komi = 7.5
    };
    allocate(&env);
    reset(&env);
 
    Client* client = make_client(env.width, env.height);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env.actions[0] = -1;

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
        step(&env);
        render(client,&env);
    }
    close_client(client);
    free_allocated(&env);
}

void performance_test() {
    long test_time = 10;
    CGo env = {
        .width = 1000,
        .height = 800,
        .grid_size = 9,
        .board_width = 600,
        .board_height = 600,
        .grid_square_size = 600/9,
        .moves_made = 0,
        .komi = 7.5
    };
    allocate(&env);
    reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % (env.grid_size)*(env.grid_size);
        step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(&env);
}

int main() {
    demo();
    // performance_test();
    return 0;
}
