#include "tcg.h"

int main() {
    TCG env = {0}; // MUST ZERO
    allocate_tcg(&env);
    reset(&env);

    init_client(&env);

    int atn = -1;
    while (!WindowShouldClose()) {
        if (atn != -1) {
            step(&env, atn);
            atn = -1;
        }

        if (IsKeyPressed(KEY_ONE)) atn = 0;
        if (IsKeyPressed(KEY_TWO)) atn = 1;
        if (IsKeyPressed(KEY_THREE)) atn = 2;
        if (IsKeyPressed(KEY_FOUR)) atn = 3;
        if (IsKeyPressed(KEY_FIVE)) atn = 4;
        if (IsKeyPressed(KEY_SIX)) atn = 5;
        if (IsKeyPressed(KEY_SEVEN)) atn = 6;
        if (IsKeyPressed(KEY_EIGHT)) atn = 7;
        if (IsKeyPressed(KEY_NINE)) atn = 8;
        if (IsKeyPressed(KEY_ZERO)) atn = 9;
        if (IsKeyPressed(KEY_ENTER)) atn = 10;

        if (env.turn == 1) {
            atn = rand() % 11;
        }
 
        render(&env);
    }
    free_tcg(&env);
    return 0;
}
