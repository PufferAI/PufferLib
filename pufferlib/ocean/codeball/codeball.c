#include <stdio.h>
#include "codeball.h"
#include "puffernet.h"
#include "renderer.h"
#include <sys/time.h>

#if defined(PLATFORM_DESKTOP)
#define GLSL_VERSION 330
#else  // PLATFORM_ANDROID, PLATFORM_WEB
#define GLSL_VERSION 100
#endif

#define NETWORK_CONTROLLED 0

#define SLOWDOWN 1

int main() {
    srand(time(NULL)); // Seed the random number generator
    Client* client = make_client();

    int n_robots = 8;
    int obs_size = (n_robots + 2) * 9;
    int action_size = 8;
    #if NETWORK_CONTROLLED
    Weights* weights =
        load_weights("../../resources/codeball_weights.bin", 142601);
    LinearLSTM* net = make_linearlstm(weights, n_robots, obs_size, action_size);
    float observation_buffer[n_robots * obs_size];
    int action_buffer[n_robots * action_size];
    #endif

    CodeBall env;
    env.n_robots = n_robots;
    env.n_nitros = 4;
    env.frame_skip = 1;
    allocate(&env);
    env.actions = (float*)calloc(n_robots * 4, sizeof(float));
    reset(&env);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int initial_steps = 2;

    for (int i = 0; i < 10000; i++) {
        if (WindowShouldClose()) break;

        if (i % SLOWDOWN == 0) {
            #if NETWORK_CONTROLLED
            if (env.terminal) {
                free_linearlstm(net);
                weights->idx = 0;
                net = make_linearlstm(weights, n_robots, obs_size, action_size);
            }
            make_observation(&env, observation_buffer);
            forward_linearlstm(net, observation_buffer, action_buffer);
            for (int j = 0; j < n_robots; j++) {
                int vel_action = action_buffer[j];
                if (vel_action == 4) {
                    vel_action = 8;
                }
                int vel_x = vel_action % 3 - 1;
                int vel_z = vel_action / 3 - 1;
                env.actions[j * 4] = vel_x * ROBOT_MAX_GROUND_SPEED;
                env.actions[j * 4 + 1] = 0;
                env.actions[j * 4 + 2] = vel_z * ROBOT_MAX_GROUND_SPEED;
                env.actions[j * 4 + 3] = 0;
            }
            #else
            for (int j = 0; j < env.n_robots; j++)
            {
                Vec3D tgt =
                    vec3d_subtract(env.ball.position, env.robots[j].position);
                for (int k = 0; k < env.n_robots; k++) {
                    if (k != j) {
                        Vec3D diff = vec3d_subtract(env.robots[k].position, env.robots[j].position);
                        double diff_len = vec3d_length(diff);
                        if (diff_len < 2.5) {
                            tgt = vec3d_multiply(diff, -1.0);
                        }
                    }
                }
                tgt = vec3d_multiply(tgt, ROBOT_MAX_GROUND_SPEED);
                env.actions[j * 4] = tgt.x;
                env.actions[j * 4 + 1] = tgt.z;
            }
            #endif
            step(&env);
        }
        if (i == initial_steps) {
            gettimeofday(&end, NULL);
            double elapsed = (end.tv_sec - start.tv_sec) +
                            (end.tv_usec - start.tv_usec) / 1000000.0;
            printf("%d steps took %f seconds\n", initial_steps, elapsed);
            printf("SPS: \t%f\n", ((double)initial_steps) / elapsed);
        }
        if (i > initial_steps) {
            render(client, &env);
        }
    }

    close_client(client);
    free(env.actions);
    free_allocated(&env);

    return 0;
}
