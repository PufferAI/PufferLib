#include <time.h>
#include <unistd.h>
#include "tower_climb.h"
#include "puffernet.h"

void demo() {   
    Weights* weights = load_weights("resources/tower_climb/tower_climb_weights.bin");
    int logit_sizes[1] = {6};
    PufferNet* net = make_puffernet(weights, 1, 228, 256, 5, logit_sizes, 1);
    const char* path = "resources/tower_climb/maps.bin";

    int num_maps = 0;
    Level* levels = load_levels_from_file(&num_maps, path);
    if (levels == NULL || num_maps == 0) {
        fprintf(stderr, "Failed to load maps for demo. Pre-generate maps with generate_maps.py and put maps.bin under resources/tower_climb.\n");
        return;
    }

    PuzzleState* puzzle_states = calloc(num_maps, sizeof(PuzzleState));

    srand(time(NULL));
    
    for (int i = 0; i < num_maps; i++) {
        init_puzzle_state(&puzzle_states[i]);
        levelToPuzzleState(&levels[i], &puzzle_states[i]);
    }

    CTowerClimb* env = allocate();
    env->num_maps = num_maps;
    env->all_levels = levels;
    env->all_puzzles = puzzle_states;
    c_reset(env);
    c_render(env);
    Client* client = env->client;
    client->enable_animations = 1;
    int tick = 0;
    while (!WindowShouldClose()) {
        if (tick % 6 == 0 && !client->isMoving) {
            tick = 0;
            int human_action = env->actions[0];
            
            // Convert unsigned char observations to float
            for (int i = 0; i < 228; i++) {
                net->obs[i] = (float)env->observations[i];
            }
            forward_puffernet(net, net->obs, env->actions);

            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                env->actions[0] = human_action;
            }
            c_step(env);
            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                env->actions[0] = NOOP;
            }
        }
        tick++;
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Camera controls
            if (IsKeyPressed(KEY_UP)) { // || IsKeyPressed(KEY_W)) {
                env->actions[0] = UP;
            }
            if (IsKeyPressed(KEY_LEFT)) { //|| IsKeyPressed(KEY_A)) {
                env->actions[0] = LEFT;
            }
            if (IsKeyPressed(KEY_RIGHT)) { //|| IsKeyPressed(KEY_D)) {
                env->actions[0] = RIGHT;
            }
            if (IsKeyPressed(KEY_DOWN)) { //|| IsKeyPressed(KEY_S)){
                env->actions[0] = DOWN;
            }
            if (IsKeyPressed(KEY_SPACE)){
                env->actions[0] = GRAB;
            }
            if (IsKeyPressed(KEY_RIGHT_SHIFT)){
                env->actions[0] = DROP;
            }
        }
        c_render(env);
        
        // Handle delayed level reset after puffer animation finishes
        if (env->pending_reset) {
            bool shouldReset = false;
            
            if (env->celebrationStarted) {
                // Wait for full celebration sequence: 0.8s climbing + 0.4s beam + 0.7s banner = 1.9s total
                float celebrationDuration = GetTime() - env->celebrationStartTime;
                shouldReset = (celebrationDuration >= 1.9f);
            } else {
                // No celebration; reset when banner finishes
                shouldReset = (!client->showBanner || client->bannerType != 1);
            }
            
            if (shouldReset) {
                env->pending_reset = false;
                c_reset(env);
            }
        }
    }
    close_client(client);
    free_allocated(env);
    free_puffernet(net);
    free(weights);
    for (int i = 0; i < num_maps; i++) {
        free(levels[i].map);
    }
    free(levels);
    free(puzzle_states);
}

void performance_test() {
    long test_time = 10;
    CTowerClimb* env = allocate();
    int seed = 0;
    init_random_level(env, 8, 25, 15, seed);
    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env->actions[0] = rand() % 5;
        c_step(env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(env);
}

int main() {
    demo();
    return 0;
}
