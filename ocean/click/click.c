#include "click.h"
#include "puffernet.h"

int main() {

    float observations[OBS_SIZE];
    float actions[ACTION_SIZE];
    float rewards[1];
    float terminals[1]; 
    
    ClickEnv env = {
        .width = 800,
        .height = 600,
        .target_spawn_duration = 200,
        .episode_length = 1000,
        .rng = 1234,
        .observations = observations,
        .actions = actions,
        .rewards = rewards,
        .terminals = terminals,
    };

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        env.human_input = get_human_input(&env);
        c_step(&env);
        c_render(&env);
    }
}