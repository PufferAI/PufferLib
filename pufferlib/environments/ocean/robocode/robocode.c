#include "robocode.h"

int main() {
    Env env = {0};
    env.num_agents = 1;
    env.width = 1080;
    env.height = 720;
    allocate_env(&env);
    reset(&env);

    init_client(&env);

    while (true) {
        int atn = 5;
        if (IsKeyPressed(KEY_ESCAPE)) break;
        if (IsKeyDown(KEY_W)) atn = 0;
        if (IsKeyDown(KEY_S)) atn = 1;
        if (IsKeyDown(KEY_A)) atn = 2;
        if (IsKeyDown(KEY_D)) atn = 3;
        env.actions[0] = atn;

        step(&env);
        render(&env);
    }
    CloseWindow();
    return 0;
}
