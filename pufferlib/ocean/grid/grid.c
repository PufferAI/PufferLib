#include "grid.h"

int main() {
    int max_size = 32;
    int width = 32;
    int height = 32;
    int num_agents = 1;
    int horizon = 128;
    float speed = 1;
    int vision = 5;
    bool discretize = true;

    int render_cell_size = 32;
    int seed = 0;

    Grid* env = allocate_grid(max_size, num_agents, horizon,
        vision, speed, discretize);

    //env->width = 32;
    //env->height = 32; env->agents[0].spawn_x = 16;
    //env->agents[0].spawn_y = 16;
    //env->agents[0].color = 6;
    //reset(env, seed);
    //load_locked_room_preset(env);
 
    create_maze_level(env, 5, 5, 0.85, seed);
    //generate_locked_room(env);
    //State state;
    //init_state(&state, env->max_size, env->num_agents);
    //get_state(env, &state);

    /*
    width = height = 31;
    env->width=31;
    env->height=31;
    env->agents[0].spawn_x = 1;
    env->agents[0].spawn_y = 1;
    reset(env, seed);
    generate_growing_tree_maze(env->grid, env->width, env->height, max_size, 0.85, 0);
    env->grid[(env->height-2)*env->max_size + (env->width - 2)] = GOAL;
    */
 
    Renderer* renderer = init_renderer(render_cell_size, width, height);

    int tick = 0;
    while (!WindowShouldClose()) {
        // User can take control of the first agent
        env->actions[0] = ATN_FORWARD;
        Agent* agent = &env->agents[0];

        // TODO: Why are up and down flipped?
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)){
                //env->actions[0] = ATN_FORWARD;
                agent->direction = 3.0*PI/2.0;
            } else if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) {
                //env->actions[0] = ATN_BACK;
                agent->direction = PI/2.0;
            } else if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) {
                //env->actions[0] = ATN_LEFT;
                agent->direction = PI;
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                //env->actions[0] = ATN_RIGHT;
                agent->direction = 0;
            } else {
                env->actions[0] = ATN_PASS;
            }
        } else {
            for (int i = 0; i < num_agents; i++) {
                env->actions[i] = rand() % 5;
            }
        }

        //env->actions[0] = actions[t];
        tick = (tick + 1)%12;
        bool done = false;
        if (tick % 12 == 0) {
            done = step(env);
            printf("direction: %f\n", env->agents[0].direction);

        }
        if (done) {
            printf("Done, reward: %f\n", env->rewards[0]);
            seed++;
            reset(env, seed);
            create_maze_level(env, 5, 5, 0.85, seed);
        }
        render_global(renderer, env, (float)tick/12.0);
    }
    close_renderer(renderer);
    free_allocated_grid(env);
    return 0;
}

