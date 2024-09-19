#include "grid.h"

unsigned int actions[41] = {NORTH, NORTH, NORTH, NORTH, NORTH, NORTH,
    EAST, EAST, EAST, EAST, EAST, EAST, SOUTH, WEST, WEST, WEST, NORTH, WEST,
    WEST, WEST, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH,
    SOUTH, SOUTH, SOUTH, EAST, EAST, EAST, EAST, EAST, EAST, EAST, EAST, SOUTH
};

void test_multiple_envs() {
    Env** envs = (Env**)calloc(10, sizeof(Env*));
    for (int i = 0; i < 10; i++) {
        envs[i] = alloc_locked_room_env();
        reset_locked_room(envs[i]);
    }

    for (int i = 0; i < 41; i++) {
        for (int j = 0; j < 10; j++) {
            envs[j]->actions[0] = actions[i];
            step(envs[j]);
        }
    }
    for (int i = 0; i < 10; i++) {
        free_allocated_grid(envs[i]);
    }
    free(envs);
    printf("Done\n");
}

int main() {
    int width = 32;
    int height = 32;
    int num_agents = 1;
    int horizon = 128;
    float agent_speed = 1;
    int vision = 5;
    bool discretize = true;

    int render_cell_size = 32;
    int seed = 42;

    //test_multiple_envs();
    //exit(0);

    Env* env = alloc_locked_room_env();
    reset_locked_room(env);
    /*
    Env* env = allocate_grid(width, height, num_agents, horizon,
        vision, agent_speed, discretize);
    env->agents[0].spawn_y = 16;
    env->agents[0].spawn_x = 16;
    env->agents[0].color = AGENT_2;
    Env* env = alloc_locked_room_env();
    reset_locked_room(env);
    */
 
    Renderer* renderer = init_renderer(render_cell_size, width, height);

    int t = 0;
    while (!WindowShouldClose()) {
        // User can take control of the first agent
        env->actions[0] = PASS;
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env->actions[0] = NORTH;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env->actions[0] = SOUTH;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env->actions[0] = WEST;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env->actions[0] = EAST;

        //for (int i = 0; i < num_agents; i++) {
        //    env->actions[i] = rand() % 4;
        //}
        //env->actions[0] = actions[t];
        bool done = step(env);
        if (done) {
            printf("Done\n");
            reset_locked_room(env);
        }
        render_global(renderer, env);

        /*
        t++;
        if (t == 41) {
            exit(0);
        }
        */
    }
    close_renderer(renderer);
    free_allocated_grid(env);
    return 0;
}

