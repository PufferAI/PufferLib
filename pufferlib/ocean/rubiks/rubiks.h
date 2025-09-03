
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

typedef struct {
    int face;  // face index
    int row;   // starting row
    int col;   // starting col
    int dr;    // row step
    int dc;    // col step
} strip_t;


typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard 
    float score; 
    int tick;
    int N; // size of cube NxNxN
    int size;
    int shuffles; // number of random moves to shuffle at reset
    char obs_type[10];
    int *stickers; // 6xNxN stickers
    strip_t strips[6][4]; // Precomputed strips for each face
} Cube;


#define OBS(env,f,r,c,color) \
        ((env)->observations[ ((f)*(env)->N*(env)->N*6) + ((r)*(env)->N*6) + ((c)*6) + (color) ])
#define STICKER(env,f,r,c) ((env)->stickers[(f)*(env)->N*(env)->N + (r)*(env)->N + (c)])

enum { U=0, D=1, R=2, L=3, F=4, B=5 };


// Precompute strips for l=0 (outer layer)
void precompute_strips(Cube *env) {
    int N = env->N;
    // Front
    env->strips[F][0] = (strip_t){U, N-1, 0, 0, 1};
    env->strips[F][1] = (strip_t){L, N-1, N-1, -1, 0};
    env->strips[F][2] = (strip_t){D, 0, 0, 0, 1};
    env->strips[F][3] = (strip_t){R, 0, 0, 1, 0};
    // Back
    env->strips[B][0] = (strip_t){U, 0, 0, 0, 1};
    env->strips[B][1] = (strip_t){R, 0, N-1, 1, 0};
    env->strips[B][2] = (strip_t){D, N-1, 0, 0, 1};
    env->strips[B][3] = (strip_t){L, N-1, 0, -1, 0};
    // Up
    env->strips[U][0] = (strip_t){B, 0, 0, 0, 1};
    env->strips[U][1] = (strip_t){R, 0, 0, 0, 1};
    env->strips[U][2] = (strip_t){F, 0, 0, 0, 1};
    env->strips[U][3] = (strip_t){L, 0, 0, 0, 1};
    // Down
    env->strips[D][0] = (strip_t){F, N-1, 0, 0, 1};
    env->strips[D][1] = (strip_t){R, N-1, 0, 0, 1};
    env->strips[D][2] = (strip_t){B, N-1, 0, 0, 1};
    env->strips[D][3] = (strip_t){L, N-1, 0, 0, 1};
    // Right
    env->strips[R][0] = (strip_t){U, 0, N-1, 1, 0};
    env->strips[R][1] = (strip_t){F, 0, N-1, 1, 0};
    env->strips[R][2] = (strip_t){D, 0, N-1, 1, 0};
    env->strips[R][3] = (strip_t){B, N-1, 0, -1, 0};
    // Left
    env->strips[L][0] = (strip_t){U, 0, 0, 1, 0};
    env->strips[L][1] = (strip_t){B, N-1, N-1, -1, 0};
    env->strips[L][2] = (strip_t){D, 0, 0, 1, 0};
    env->strips[L][3] = (strip_t){F, 0, 0, 1, 0};
}



/* Recommended to have an init function of some kind if you allocate 
 * extra memory. This should be freed by c_close. Don't forget to call
 * this in binding.c!
 */
void init(Cube* env) {
    // Allocate any extra memory you need here. Don't forget to free in c_close!
    env->stickers = malloc(6 * env->N * env->N * sizeof(int));
    precompute_strips(env);
}

void reset_stickers(Cube* env) {
    for(int i = 0; i < 6; i++) {
            for(int j = 0; j < env->N; j++) {
               for(int k = 0; k < env->N; k++) {
                   STICKERS(i,j,k) = i;
               }
            }
    }
}

static inline void set_color(Cube *env, int f, int r, int c, int k) {
    for (int ch=0; ch<6; ch++)
        OBS(env,f,r,c,ch) = (ch == k) ? 1.0f : 0.0f;
}

void compute_observations(Cube* env) {
    for (int f=0; f<6; f++) {
        for (int r=0; r<env->N; r++) {
            for (int c=0; c<env->N; c++) {
                int colour = STICKER(env,f,r,c);
                set_color(env, f, r, c, colour);
            }
        }
    }
}
//Just rotates the strips CLOCKWISE, not the face itself
static void rotate_strips(Cube *env, strip_t s[4]) {
    int N = env->N;
    int tmp[N];
    for (int k=0;k<N;k++)
        tmp[k] = STICKER(env, s[0].face, s[0].row + s[0].dr*k, s[0].col + s[0].dc*k);

    for (int j=0;j<3;j++) {
        for (int k=0;k<N;k++) {
            STICKER(env, s[j].face, s[j].row + s[j].dr*k, s[j].col + s[j].dc*k) =
                STICKER(env, s[j+1].face, s[j+1].row + s[j+1].dr*k, s[j+1].col + s[j+1].dc*k);
        }
    }

    for (int k=0;k<N;k++)
        STICKER(env, s[3].face, s[3].row + s[3].dr*k, s[3].col + s[3].dc*k) = tmp[k];
}

// Just rotates the face CLOCKWISE
static void rotate_face(Cube *env, int f) {
    int N = env->N;
    int tmp[N][N]; // VLA; use malloc if your compiler forbids VLAs
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            tmp[j][N-1-i] = STICKER(env,f,i,j);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            STICKER(env,f,i,j) = tmp[i][j];
}


void move(Cube *env, int face, int turns) {
    // Normalize turns to [0,3] if -1 then = 3 so use -ves for anti-clockwise 
    turns = (turns % 4 + 4) % 4; 
    for (int t=0;t<turns;t++) {
        rotate_strips(env, env->strips[face]);
        rotate_face(env, face);
    }
}

void shuffle(Cube* env, int shuffles){
    for (int i=0;i<shuffles;i++) {
        int face = rand() % 6;
        int turns = (rand() % 3) + 1; // 1,2,3 turns
        move(env, face, turns);
    }
}

static inline void decode_action(int action, int *face, int *turns) {
    *face = action / 2;
    *turns = (action % 2 == 0) ? +1 : -1;
}

// Required function
void c_reset(Cube* env) {

    memset(env->observations, 0, sizeof(float) * env->size); //reset obs to 0s
    reset_stickers(env); //reset cube to solved state
    shuffle(env, env->shuffles);

    env -> tick = 0;
    env -> score = 0;
    compute_observations(env);
}

// Required function
void c_step(Cube* env) {

    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->tick += 1;
    int face, turns;
    decode_action(env->actions[0], &face, &turns);
    move(env, face, turns);

    //main below
    //
    //
    //Then update reward and terminals as needed
    //env->rewards[0] = ...;
    //env->terminals[0] = ...;
    //env->score += env->rewards[0];
    compute_observations(env);
}

// Required function. Should handle creating the client on first call
void c_render(Cube* env) {
 
    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Cube* env) {
    //free memory you allocated in init
    free(env->stickers);
   if (IsWindowReady()) {
        CloseWindow();      
    }
}
