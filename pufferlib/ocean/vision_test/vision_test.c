#include "raylib.h"
#include "rcamera.h"
#include "raymath.h"
#include "vision_test.h"
#include "puffernet.h"

int main(void) {
    VisionTest env = {};

    env.observations = (uint8_t*)calloc(3*64, sizeof(uint8_t));  // Alloc our 16x16 window compressed to float
    env.actions = (int*)calloc(5, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0;
            if      (IsKeyDown(KEY_W))    { env.actions[0] = S_UP;    }
            else if (IsKeyDown(KEY_S))  { env.actions[0] = S_DOWN;  }
            else if (IsKeyDown(KEY_A))  { env.actions[0] = S_LEFT;  }
            else if (IsKeyDown(KEY_D)) { env.actions[0] = S_RIGHT; }
        } else {
            env.actions[0] = rand() % 5;
        }

        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
