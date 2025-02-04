#include "raylib.h"
#include <stdio.h>
#include "flappy_bird.h"

  //Flappy_env env = ..;
  //allocate(&env)?
  //Client* client = make_client for the env make_client(&env)
  //c_reset(&env)
  //while(!window close)
  //if (pressing left shift)
  // if (action)
  //  env.action = action
  //c_render(client, &env)

int y_pos = 200;
int gravity = -5;
   
int main(void)
{
    SetTargetFPS(60);
    InitWindow(800, 400, "Flappy bird");

    while (!WindowShouldClose())
    {
        BeginDrawing();
            DrawCircle(200, y_pos, 15, RED);
            ClearBackground(SKYBLUE);
        EndDrawing();

        y_pos -= gravity;
        if (y_pos > 400) {
          y_pos = 0;
    }
    }

    CloseWindow();

    return 0;
}

