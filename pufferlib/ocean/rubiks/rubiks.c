#include "rubiks.h"
#include <unistd.h>
#include <string.h>
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

// Pack face + direction into env action [0..11]
// cw=1 for clockwise as seen from outside the face, cw=0 for counter-clockwise
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

