#include "froggy.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ncurses.h"

int main() {
    srand(time(NULL));
    Froggy env = {0};
    env.width = 20;
    env.height = 15;
    env.episode_length = 1000; // Set episode_length
    env.map_seed = time(NULL); // Initialize map seed
    
    allocate(&env);
    c_reset(&env);
    c_render(&env);

    while (1) {
        int action = froggy_ui_get_input();
        if (action == -1) {
            free_allocated(&env);
            c_close(&env);
            return 0;
        }
        env.actions[0] = action;
        c_step(&env);
        c_render(&env);
    }

    return 0;
}

