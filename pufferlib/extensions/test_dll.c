#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <time.h>
#include "vecenv.h"

vec_send_fn vec_send;
vec_recv_fn vec_recv;

int timeout = 10;
 
float perf_test(VecEnv* vec, int buffers) {
    int start = time(NULL);
    int i = 0;
    while (time(NULL) - start < timeout) {
        int buf = i % buffers;
        vec_recv(vec, buf);
        vec_send(vec, buf);
        i++;
    }

    return (float)i / (float)timeout;
}
 
int main() {
    void* handle = dlopen("./breakout.so", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen error: %s\n", dlerror());
        return 1;
    }

    // Clear any existing errors
    dlerror();

    // Load the function pointer
    create_environments_fn create_environments = (create_environments_fn)dlsym(handle, "create_environments");
    env_init_fn env_init = (env_init_fn)dlsym(handle, "env_init");
    vec_reset_fn vec_reset = (vec_reset_fn)dlsym(handle, "vec_reset");
    vec_step_fn vec_step = (vec_step_fn)dlsym(handle, "vec_step");
    vec_send = (vec_send_fn)dlsym(handle, "vec_send");
    vec_recv = (vec_recv_fn)dlsym(handle, "vec_recv");
    env_close_fn env_close = (env_close_fn)dlsym(handle, "env_close");
    vec_close_fn vec_close = (vec_close_fn)dlsym(handle, "vec_close");
    vec_log_fn vec_log = (vec_log_fn)dlsym(handle, "vec_log");
    vec_render_fn vec_render = (vec_render_fn)dlsym(handle, "vec_render");
    int obs_n = *(int*)dlsym(handle, "OBS_N");
    int act_n = *(int*)dlsym(handle, "ACT_N");
    int obs_t = *(int*)dlsym(handle, "OBS_T");
    int act_t = *(int*)dlsym(handle, "ACT_T");
    
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        fprintf(stderr, "dlsym error: %s\n", dlsym_error);
        dlclose(handle);
        return 1;
    }

    Dict* kwargs = create_dict(32);
    dict_set_int(kwargs, "frameskip", 4);
    dict_set_int(kwargs, "width", 576);
    dict_set_int(kwargs, "height", 330);
    dict_set_int(kwargs, "paddle_width", 62);
    dict_set_int(kwargs, "paddle_height", 8);
    dict_set_int(kwargs, "ball_width", 32);
    dict_set_int(kwargs, "ball_height", 32);
    dict_set_int(kwargs, "brick_width", 32);
    dict_set_int(kwargs, "brick_height", 12);
    dict_set_int(kwargs, "brick_rows", 6);
    dict_set_int(kwargs, "brick_cols", 18);
    dict_set_int(kwargs, "initial_ball_speed", 256);
    dict_set_int(kwargs, "max_ball_speed", 448);
    dict_set_int(kwargs, "paddle_speed", 620);
    dict_set_int(kwargs, "continuous", 0);

    int num_envs = 4096;
    int threads = 8;
    int buffers = 4;
    int block_size = 256;

    /*
    int num_envs = 32;
    int threads = 0;
    int buffers = 2;
    int block_size = 2;
    */

    VecEnv* vec1 = create_environments(num_envs, threads, buffers, block_size, true, 0, kwargs);
    vec_reset(vec1);

    VecEnv* vec2 = create_environments(num_envs, threads, buffers, block_size, true, 1, kwargs);
    vec_reset(vec2);

    float sps = perf_test(vec1, buffers) * num_envs / (float)buffers;
    printf("Performance: %f\n M SPS (%f GB/s)\n", sps/1e6f, 118.0f*sps/1e9f);
    exit(0);


    /*
    for (int i = 0; i < vec1->size; i++) {
        float* obs = vec1->observations + i*obs_n;
        obs[0] = i;

        obs = vec2->observations + i*obs_n;
        obs[0] = i;
    }
    */

    for (int i = 0; i < 10000; i++) {
        vec_recv(vec1, i%buffers);
        vec_recv(vec2, i%buffers);

        /*
        if (i % 2 == 0) {
            vec_recv(vec1, 0);
         }
        vec_recv(vec2, i%2);
        */

        int start = (i % buffers) * (num_envs / buffers);
        int end = start + num_envs / buffers;
        // Doesnt work for 1 buffer (bad end index)
        for (int j = start; j < end; j++) {
            float* obs1 = vec1->observations + j*obs_n;
            float* obs2 = vec2->observations + j*obs_n;
            for (int k = 0; k < obs_n; k++) {
                if (obs1[k] != obs2[k]) {
                    sleep(1);
                    printf("Observation mismatch at index %d\n", j);
                    exit(1);
                }
            }
            assert(vec1->actions[j] == vec2->actions[j]);
            assert(vec1->rewards[j] == vec2->rewards[j]);
            assert(vec1->terminals[j] == vec2->terminals[j]);
        }
        printf("Passed %d\n", i);

        /*
        if (i % 2 == 1) {
            vec_send(vec1, 0);
        }
        vec_send(vec2, i%2);
        */
        vec_send(vec1, i%buffers);
        vec_send(vec2, i%buffers);
    }

    /*
    VecEnv* vec = create_environments(num_envs, threads, buffers, block_size, kwargs);
    for (int i = 0; i < 10000; i++) {
        int buf = i % buffers;
        vec_recv(vec, buf);
        vec_render(vec, 0);
        vec_send(vec, buf);
    }
    */

    //printf("Created VecEnv with %d environments\n", vec->size);
    // TODO: Add a `close_vecenv` function to clean up
    // vec.envs, etc.
    printf("Done\n");

    // Close the library
    //dlclose(handle);
    return 0;
}
