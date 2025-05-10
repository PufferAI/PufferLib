#include "gpudrive_test.h"
#include <time.h>

void demo(){
    long test_time = 10;
    test_struct env = {
        .num_agents = 128,
        .active_agents =15,
    };
    init(&env);
    long start = time(NULL);
    int i = 0;
    int mean_obs = 4000;
    int max_obs = 20000;
    while (time(NULL) - start < test_time) {
        step(&env);
        // return;
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (i *env.active_agents) / (end - start));
    free_initialized(&env);

}

int main() {
    demo();
    return 0;
}
