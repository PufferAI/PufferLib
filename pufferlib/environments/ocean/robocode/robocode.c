#include "robocode.h"

int main() {
    Env env = {0};
    env.num_agents = 2;
    env.width = 768;
    env.height = 576;
    allocate_env(&env);
    reset(&env);

    Client* client = make_client(&env);

    while (true) {
        for (int i = 0; i < NUM_ACTIONS; i++) {
            env.actions[i] = 0;
        }

        env.actions[0] = 16.0f;
        float x = env.robots[0].x;
        float y = env.robots[0].y;
        float op_x = env.robots[1].x;
        float op_y = env.robots[1].y;
        float gun_heading = env.robots[0].gun_heading;
        float angle_to_op = 180*atan2(op_y - y, op_x - x)/M_PI;
        float gun_delta = angle_to_op - gun_heading;
        if (gun_delta < -180) gun_delta += 360;
        env.actions[2] = (gun_delta > 0) ? 1.0f : -1.0f;
        if (gun_delta < 5 && gun_delta > -5) env.actions[4] = 1.0;

        env.actions[5] = 16.0f;
        x = env.robots[1].x;
        y = env.robots[1].y;
        op_x = env.robots[0].x;
        op_y = env.robots[0].y;
        gun_heading = env.robots[1].gun_heading;
        angle_to_op = 180*atan2(op_y - y, op_x - x)/M_PI;
        gun_delta = angle_to_op - gun_heading;
        if (gun_delta < -180) gun_delta += 360;
        env.actions[7] = (gun_delta > 0) ? 1.0f : -1.0f;
        if (gun_delta < 5 && gun_delta > -5) env.actions[9] = 1.0;


        if (IsKeyPressed(KEY_ESCAPE)) break;
        if (IsKeyDown(KEY_W)) env.actions[0] = 16.0f;
        if (IsKeyDown(KEY_S)) env.actions[0] = -16.0f;
        if (IsKeyDown(KEY_A)) env.actions[1] = -2.0f;
        if (IsKeyDown(KEY_D)) env.actions[1] = 2.0f;
        if (IsKeyDown(KEY_Q)) env.actions[2] = -1.0f;
        if (IsKeyDown(KEY_E)) env.actions[2] = 1.0f;
        if (IsKeyDown(KEY_SPACE)) env.actions[4] = 1.0f;

        step(&env);
        render(client, &env);
    }
    CloseWindow();
    return 0;
}
