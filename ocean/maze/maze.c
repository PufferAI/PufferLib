#include "maze.h"
#include "puffernet.h"

void demo() {
    Weights* weights = load_weights("resources/maze/maze_weights.bin");
    int logit_sizes[1] = {5};
    PufferNet* net = make_puffernet(weights, 1, 121, 512, 5, logit_sizes, 1);

    int max_size = 47;
    int num_maps = 64;
    int num_agents = 1;
    int horizon = 256;
    float speed = 1;
    int vision = 5;
    bool discretize = true;

    Grid* env = allocate_maze(max_size, num_agents, horizon,
        vision, speed, discretize);

    // Generate maps matching binding.c: random odd sizes, random difficulty
    State* levels = calloc(num_maps, sizeof(State));
    Grid temp_env;
    temp_env.max_size = max_size;
    init_maze(&temp_env);
    unsigned int map_rng = 42;
    for (int i = 0; i < num_maps; i++) {
        int sz = 5 + (rand_r(&map_rng) % (max_size - 5));
        if (sz % 2 == 0) sz -= 1;
        float difficulty = (float)rand_r(&map_rng) / (float)(RAND_MAX);
        create_maze_level(&temp_env, sz, sz, difficulty, i);
        init_state(&levels[i], max_size, num_agents);
        get_state(&temp_env, &levels[i]);
    }
    free(temp_env.maze);

    env->num_maps = num_maps;
    env->levels = levels;

    c_reset(env);
    c_render(env);
    while (!WindowShouldClose()) {
        env->actions[0] = ATN_PASS;
        Agent* agent = &env->agents[0];

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)){
                agent->direction = 3.0*PI/2.0;
                env->actions[0] = ATN_FORWARD;
            } else if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) {
                agent->direction = PI/2.0;
                env->actions[0] = ATN_FORWARD;
            } else if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) {
                agent->direction = PI;
                env->actions[0] = ATN_FORWARD;
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                agent->direction = 0;
                env->actions[0] = ATN_FORWARD;
            } else {
                env->actions[0] = ATN_PASS;
            }
        } else {
            float obs[121];
            for (int i = 0; i < 121; i++) obs[i] = env->observations[i];
            forward_puffernet(net, obs, env->actions);
        }

        c_step(env);
        c_render(env);
    }
    
    free_puffernet(net);
    free(weights);
    free_allocated_maze(env);
    for (int i = 0; i < num_maps; i++) free_state(&levels[i]);
    free(levels);
}

int main() {
    demo();
    return 0;
}
