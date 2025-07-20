/* FlappyBirdMini: a simple 2-block tall grid environment.
 * Agent can move up or down. Observes an 8x2 grid with moving obstacles.
 * +1 reward per obstacle passed.
 */

#include "flappy_bird_mini.h"

void add_log(FlappyBirdMini* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

void c_reset(FlappyBirdMini* env) {
    env->agent_pos = 0; // Start on the floor
    env->wall_pos = rand() % 2;
    env->tick = 0;

    env->observations[0] = (env->wall_pos == 0) ? WALL : EMPTY; // Floor observation
    env->observations[1] = (env->wall_pos == 1) ? WALL : EMPTY; // Ceiling observation
}

void c_step(FlappyBirdMini* env) {
    env->tick += 1;

    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    if (action == UP) {
        env->agent_pos = 1;
    } else if (action == DOWN) {
        env->agent_pos = 0;
    }

    // Check for collision with wall
    if (env->agent_pos == env->wall_pos) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    env->observations[0] = (env->wall_pos == 0) ? WALL : EMPTY; // Floor observation
    env->observations[1] = (env->wall_pos == 1) ? WALL : EMPTY; // Ceiling observation

    if (env->tick >= 1000) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }
}

void c_render(FlappyBirdMini* env) {
    if (!IsWindowReady()) {
        InitWindow(64*2, 64*2, "PufferLib Flappy Bird Mini");
        SetTargetFPS(5);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int px = 64;

    // Draw floor and ceiling
    DrawRectangle(0, px, px*2, px, (Color){100, 100, 100, 255});
    DrawRectangle(0, 0, px*2, px, (Color){100, 100, 100, 255});

    // Draw agent
    Color agent_color = (Color){0, 187, 187, 255};
    if (env->agent_pos == 0) {
        DrawRectangle(px/2, px + px/2, px, px/2, agent_color); // On floor
    } else {
        DrawRectangle(px/2, px/2, px, px/2, agent_color); // On ceiling
    }

    // Draw wall
    Color wall_color = (Color){187, 0, 0, 255};
    if (env->wall_pos == 0) {
        DrawRectangle(px + px/2, px, px/2, px, wall_color); // Wall on floor
    } else {
        DrawRectangle(px + px/2, 0, px/2, px, wall_color); // Wall on ceiling
    }

    EndDrawing();
}

void c_close(FlappyBirdMini* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

int main() {
    FlappyBirdMini env = {};
    env.observations = (unsigned char*)calloc(2, sizeof(unsigned char)); // 2 observations: floor and ceiling
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = NOOP;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
        } else {
            env.actions[0] = rand() % 3; // 0: NOOP, 1: UP, 2: DOWN
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
