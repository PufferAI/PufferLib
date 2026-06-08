#include "maze.h"
#include "puffernet.h"

void demo() {
    Weights* weights = load_weights("resources/maze/maze_weights.bin");
    int logit_sizes[1] = {5};
    PufferNet* net = make_puffernet(weights, 1, 121, 512, 5, logit_sizes, 1);

    int num_maps = 64;
    int horizon = 256;
    float speed = 1;
    int vision = 5;
    bool discretize = true;

    Grid* env = (Grid*)calloc(1, sizeof(Grid));
    env->num_agents = 1;
    env->rng = 73;
    env->observations = calloc(WINDOW*WINDOW, sizeof(unsigned char));
    env->actions = calloc(1, sizeof(float));
    env->rewards = calloc(1, sizeof(float));
    env->terminals = calloc(1, sizeof(float));

    // Generate maps matching binding.c: random odd sizes, random difficulty
    State* levels = calloc(num_maps, sizeof(State));
    unsigned int map_rng = 42;
    for (int i = 0; i < num_maps; i++) {
        int sz = 5 + (rand_r(&map_rng) % (MAX_SIZE - 5));
        if (sz % 2 == 0) sz -= 1;
        float difficulty = (float)rand_r(&map_rng) / (float)(RAND_MAX);
        State* level = &levels[i];
        level->width = sz;
        level->height = sz;
        create_maze_level(level, difficulty, i);
    }

    env->num_levels = num_maps;
    env->levels = levels;

    c_reset(env);
    c_render(env);
    while (!WindowShouldClose()) {
        env->actions[0] = ATN_PASS;
        env->actions[0] = ATN_SOUTH;
        State* s = &env->state;

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)){
                env->actions[0] = ATN_NORTH;
            } else if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) {
                env->actions[0] = ATN_SOUTH;
            } else if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) {
                s->direction = PI;
                env->actions[0] = ATN_WEST;
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                s->direction = 0;
                env->actions[0] = ATN_EAST;
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
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
    free(levels);
}

int main() {
    demo();
    return 0;
}
