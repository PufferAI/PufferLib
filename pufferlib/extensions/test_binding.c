#include "binding.h"

int main() {
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

    int num_envs = 1024;
    int threads = 8;
    int buffers = 2;

    VecEnv* vec = create_environments(num_envs, threads, buffers, kwargs);
    vec_reset(vec);

    for (int i = 0; i < 10000; i++) {
        int buf = i % buffers;
        vec_recv(vec, buf);
        for (int j = 0; j < num_envs; j++) {
            Env* env = &vec->envs[j];
            env->actions[j] = rand() % 3;
        }
        vec_send(vec, buf);

        /*
        Env* env = &vec.envs[0];
        c_render(env);
        env->actions[0] = rand() % 3;
        c_step(env);
        */
    }
    return 0;
}

