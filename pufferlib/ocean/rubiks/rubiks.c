/* Pure C demo file for Target. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#include "rubiks.h"
#include <unistd.h>
#include <string.h>

/* Puffernet is our lightweight cpu inference library that
 * lets you load basic PyTorch model architectures so that
 * you can run them in pure C or on the web via WASM
 */
#include "puffernet.h"


//For checks when we break stuff

int compare_logs(const char *ref_path, const char *new_path) {
    FILE *ref = fopen(ref_path, "r");
    FILE *newf = fopen(new_path, "r");
    if (!ref || !newf) {
        perror("fopen");
        return -1;
    }

    char a[512], b[512];
    int line = 1;
    int diff_found = 0;

    while (1) {
        char *ra = fgets(a, sizeof a, ref);
        char *rb = fgets(b, sizeof b, newf);

        if (!ra || !rb) {
            if (ra != rb) {
                printf("Length mismatch starting at line %d\n", line);
                diff_found = 1;
            }
            break;
        }

        if (strcmp(a, b) != 0) {
            printf("Line %d differs:\n", line);
            printf("  ref: %s", a);
            printf("  new: %s", b);
            diff_found = 1;
        }
        line++;
    }

    fclose(ref);
    fclose(newf);

    if (remove(new_path) != 0) {
        perror("remove");
    }

    if (!diff_found) {
        printf("Logs match exactly\n");
        return 0;
    } else {
        return 1;
    }
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

    //TESTING
    /*

    FILE *log = fopen("stickers_checking.log", "a");

    for (int i=0; i<12; i++) {
    print_stickers_file(&env, log);
    env.actions[0] = 0;
    c_step(&env);
    fprintf(log, "Step %d\n", i);
    print_stickers_file(&env, log);
}
    fclose(log);


    int res = compare_logs("pufferlib/ocean/rubiks/stickers.log", "stickers_checking.log");
    if (res == 0) {
        printf("Logs match\n");
    } else {
        printf("Logs differ\n");
    }
    */
    //END TESTING
    //
    //
    env.user_mode = 1;
     while (!WindowShouldClose()) {
            c_render(&env);
        }
      
       
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    printf("Done\n");
    
}

