#include "rubiks.h"
#include <unistd.h>
#include <string.h>
#include "puffernet.h"

//Specific functions for user mode only

//To convert highlights to actions
static inline int axis_layer_to_face(int axis, int layer, int N) {
    int outer = (layer == N-1); // 1 if the positive side slab
    switch (axis) {
        case 0: return outer ? R : L; // +X is R, -X is L
        case 1: return outer ? U : D; // +Y is U, -Y is D
        case 2: return outer ? F : B; // +Z is F, -Z is B
        default: return -1;
    }
}

static inline int face_dir_to_action(int face, int cw) {
    // decode_action: even -> +1 turn, odd -> -1 turn
    // treat cw as +1
    return face * 2 + (cw ? 0 : 1);
}

// Directly from highlight to action
static inline int highlight_to_action(const Cube *env, int cw) {
    int face = axis_layer_to_face(env->highlight_axis, env->highlight_layer, env->N);
    return face < 0 ? -1 : face_dir_to_action(face, cw);
}

int main() {
    int N = 3;
    int num_obs = 6*N*N*6;


    Cube env = {
        .N = N,
        .shuffles = 0,
        .size = num_obs
    };
    init(&env);



    env.observations = calloc(num_obs, sizeof(float));
    env.actions = calloc(12, sizeof(int));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(unsigned char));
    env.max_episode_steps = 1000;
    

    c_reset(&env);
    c_render(&env);
   
    env.user_mode = 1;
     while (!WindowShouldClose()) {
            c_render(&env);

            if (IsKeyPressed(KEY_ENTER)) {            // CW
                    int a = highlight_to_action(&env, 1);
                    env.actions[0] = a;
                    c_step(&env);
                }
            if (IsKeyPressed(KEY_BACKSPACE)) {        // CCW
                int a = highlight_to_action(&env, 0);
                env.actions[0] = a;
                c_step(&env);
            }
        }
      
       
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    printf("Done\n");
    
}

