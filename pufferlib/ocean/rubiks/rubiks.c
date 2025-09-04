/* Pure C demo file for Target. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#include "rubiks.h"

/* Puffernet is our lightweight cpu inference library that
 * lets you load basic PyTorch model architectures so that
 * you can run them in pure C or on the web via WASM
 */
#include "puffernet.h"

int main() {
    int N = 3;
    int num_obs = 6*N*N*6;


    Cube env = {
        .N = N,
        .shuffles = 0,
        .size = num_obs
    };
    init(&env);

    // Allocate these manually since they aren't being passed from Python
    env.observations = calloc(num_obs, sizeof(float));
    env.actions = calloc(12, sizeof(int));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(unsigned char));
    env.max_episode_steps = 1000;
    

    // Always call reset and render first
    c_reset(&env);
   // check_face_mapping_bijection(&env);
    //check_projection_solved(&env);
    //
   
   c_render(&env);

   /* int a=0;
    for (int i=0; i<12; i++) {
        printf("Action %d\n", a);
        env.actions[0] = a:;
        c_step(&env);
        a++;

        }*/
   env.actions[0] = 1;
    c_step(&env);
    env.actions[0] = 0;
    c_step(&env);

    env.actions[0] =2;
    c_step(&env);
    env.actions[0] =3;
    c_step(&env);
    env.actions[0] =4;
    c_step(&env);       
    env.actions[0] =5;
    c_step(&env);
    env.actions[0] =6;
    c_step(&env);
    env.actions[0] =7;
    c_step(&env);
    env.actions[0] =8;      
    c_step(&env);
    env.actions[0] =9;
    c_step(&env);
    env.actions[0] =10;
    c_step(&env);
    env.actions[0] =11;
    c_step(&env);

   // print_strips(&env);
    //test_moves(&env);
   
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    printf("Done\n");
    
}

