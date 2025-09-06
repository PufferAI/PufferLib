
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"
#include <stdio.h>

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
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
    Vector3 pos;
    Color faces[6];
    int ix, iy, iz;   // grid coordinates 0..N-1
} Cubelet_r;



typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard 
    float score;
    int max_episode_steps;
    int tick;
    int N; // size of cube NxNxN
    int size;
    int shuffles; // number of random moves to shuffle at reset
    int *stickers; // 6xNxN stickers
    strip_t strips[6][4]; // Precomputed strips for each face
    Cubelet_r *cubelets; // for rendering
    int total_cubelets;
    int render; //global OpenGL window so only render if called but we need render stuff in step for the animation
} Cube;


#define OBS(env,f,r,c,color) \
        ((env)->observations[ ((f)*(env)->N*(env)->N*6) + ((r)*(env)->N*6) + ((c)*6) + (color) ])
#define STICKER(env,f,r,c) ((env)->stickers[(f)*(env)->N*(env)->N + (r)*(env)->N + (c)])


//Faces are Up, Down, Right, Left, Front, Back
enum { U=0, D=1, R=2, L=3, F=4, B=5 };


static Color sticker_colors[6] = {
    WHITE,  // U
    YELLOW, // D
    RED,    // R
    ORANGE, // L
    GREEN,  // F
    BLUE    // B
};




typedef struct {
    int rotating;      // 0 idle, 1 anim
    float elapsed;
    float duration;    // e.g. 0.5f
    int axis;          // 0=X,1=Y,2=Z
    int layer;         // 0..N-1
    int dir;           // +1 or -1
} MoveState;

static MoveState anim = {0};

void add_log(Cube* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->score;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}


// Precompute strips for l=0 (outer layer)





void precompute_strips(Cube *env) {
    int N = env->N;

   // face row col drow dcol 
    // FRONT (F): U row 0 → R col N-1 → D row N-1 → L col N-1
     env->strips[F][0] = (strip_t){U, N-1, 0,   0,  1};   // U bottom row, L→R
    env->strips[F][1] = (strip_t){R, 0,   0,   1,  0};   // R left col, T→B
    env->strips[F][2] = (strip_t){D, 0,   N-1, 0, -1};   // D top row, R→L
    env->strips[F][3] = (strip_t){L, N-1, N-1,-1,  0};   // L right col, B→T                                                       // 
       
    // BACK (B): U back row → L left col → D top row → R left col  (clockwise, viewed from back)
  // BACK (B): U top → L left → D bottom → R right  (clockwise, viewed from back)
    env->strips[B][0] = (strip_t){U, 0,    N-1, 0, -1};   // U row 0,     R→L
    env->strips[B][1] = (strip_t){L, 0,    0,   1,  0};   // L col 0,     T→B
    env->strips[B][2] = (strip_t){D, N-1,  0,   0,  1};   // D row N-1,   L→R
    env->strips[B][3] = (strip_t){R, N-1,  N-1, -1, 0};   // R col N-1,   B→T
//UP
 /* env->strips[U][0] = (strip_t){F, 0, 0,   0,  1};    // F top row, L→R
    env->strips[U][1] = (strip_t){L, 0, N-1, 0, -1};    // L top row, R→L
    env->strips[U][2] = (strip_t){B, 0, 0,   0,  1};    // B top row, L→R
    env->strips[U][3] = (strip_t){R, 0, N-1, 0, -1};    // R top row, R→L*/
                                                        //
// UP face (looking down on U): cycle is F → L → B → R
env->strips[U][0] = (strip_t){F, 0, 0,   0, +1};  // F top row, left→right
env->strips[U][1] = (strip_t){L, 0, 0,   0, +1};  // L top row, left→right
env->strips[U][2] = (strip_t){B, 0, 0,   0, +1};  // B top row, left→right
env->strips[U][3] = (strip_t){R, 0, 0,   0, +1};  // R top row, left→right

    // DOWN (D): F bottom → L bottom → B bottom → R bottom  (clockwise by your rule)
    env->strips[D][0] = (strip_t){F, N-1, 0, 0, 1};  // F row N-1, L→R
    env->strips[D][1] = (strip_t){L, N-1, 0, 0, 1};  // L row N-1, L→R
    env->strips[D][2] = (strip_t){B, N-1, 0, 0, 1};  // B row N-1, L→R
    env->strips[D][3] = (strip_t){R, N-1, 0, 0, 1};  // R row N-1, L→R
   
  
    // RIGHT (R): U right → B left → D right → F right
    env->strips[R][0] = (strip_t){U, 0,   N-1, 1, 0};    // U col N-1, T→B
    env->strips[R][1] = (strip_t){B, N-1, 0,  -1,0};     // B col 0,   B→T
    env->strips[R][2] = (strip_t){D, 0,   N-1, 1, 0};    // D col N-1, T→B
    env->strips[R][3] = (strip_t){F, 0,   N-1, 1, 0};    // F col N-1, T→B

    // LEFT (L): U left → F left → D left → B right
    env->strips[L][0] = (strip_t){U, 0,   0, 1, 0};        // U col 0,   T→B
    env->strips[L][1] = (strip_t){F, 0,   0, 1, 0};        // F col 0,   T→B
    env->strips[L][2] = (strip_t){D, 0,   0, 1, 0};        // D col 0,   T→B
    env->strips[L][3] = (strip_t){B, N-1, N-1,-1,0};       // B col N-1, B→T
}




void init(Cube* env) {
    // Allocate any extra memory you need here. Don't forget to free in c_close!
    env->stickers = malloc(6 * env->N * env->N * sizeof(int));
    env->cubelets = NULL;
    env->total_cubelets = 0;
    precompute_strips(env);
    env->render = 0;
    
    for (int f = 0; f < 6; f++) {
        for (int s = 0; s < 4; s++) {
            strip_t idx = env->strips[f][s];
            if (idx.face < 0 || idx.face >= 6 ||
                idx.row  < 0 || idx.row  >= 3 ||
                idx.col  < 0 || idx.col  >= 3) {
                printf("Bad strip: face %d slot %d -> (%d,%d,%d)\n",
                       f, s, idx.face, idx.row, idx.col);
            }
        }
}

//fprintf(stderr, "sizeof(Cube)=%zu sizeof(strip_t)=%zu\n",
       // sizeof(Cube), sizeof(strip_t));

}

void reset_stickers(Cube* env) {
    for(int i = 0; i < 6; i++) {
        int col = i;
            for(int j = 0; j < env->N; j++) {
               for(int k = 0; k < env->N; k++) {
                   STICKER(env, i,j,k) = col;
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
        tmp[k] = STICKER(env, s[3].face, s[3].row + s[3].dr*k, s[3].col + s[3].dc*k);

    for (int j=3;j>0;j--) {
        for (int k=0;k<N;k++) {
            STICKER(env, s[j].face, s[j].row + s[j].dr*k, s[j].col + s[j].dc*k) =
                STICKER(env, s[j-1].face, s[j-1].row + s[j-1].dr*k, s[j-1].col + s[j-1].dc*k);
        }
    }

    for (int k=0;k<N;k++)
        STICKER(env, s[0].face, s[0].row + s[0].dr*k, s[0].col + s[0].dc*k) = tmp[k];
}
  
// Rotates the strips COUNTER-CLOCKWISE
static void rotate_strips_ccw(Cube *env, strip_t s[4]) {
    int N = env->N;
    int tmp[N];
    // copy first strip
    for (int k=0;k<N;k++)
        tmp[k] = STICKER(env, s[0].face,
                         s[0].row + s[0].dr*k,
                         s[0].col + s[0].dc*k);

    // shift others down
    for (int j=0;j<3;j++) {
        for (int k=0;k<N;k++) {
            STICKER(env, s[j].face,
                    s[j].row + s[j].dr*k,
                    s[j].col + s[j].dc*k) =
                STICKER(env, s[j+1].face,
                        s[j+1].row + s[j+1].dr*k,
                        s[j+1].col + s[j+1].dc*k);
        }
    }

    // put saved strip into last
    for (int k=0;k<N;k++)
        STICKER(env, s[3].face,
                s[3].row + s[3].dr*k,
                s[3].col + s[3].dc*k) = tmp[k];
}

// rotate face counter-clockwise
static void rotate_face_ccw(Cube *env, int f) {
    int N = env->N;
    int tmp[N][N];
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            tmp[N-1-j][i] = STICKER(env,f,i,j);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            STICKER(env,f,i,j) = tmp[i][j];
}


// Just rotates the face CLOCKWISE
static void rotate_face(Cube *env, int f) {
    int N = env->N;
    int tmp[N][N];     
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            tmp[j][N-1-i] = STICKER(env,f,i,j);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            STICKER(env,f,i,j) = tmp[i][j];
}



void move(Cube *env, int face, int turns) {
    int dir = (turns > 0) ? +1 : -1;
    turns = abs(turns) % 4;
    for (int t=0; t<turns; t++) {
        if (dir > 0) {
            rotate_strips(env, env->strips[face]);
            rotate_face(env, face);
        } else {
            rotate_strips_ccw(env, env->strips[face]);
            rotate_face_ccw(env, face);
        }
    }
}


/*void move(Cube *env, int face, int turns) {
    // Normalize turns to [0,3] if -1 then = 3 so use -ves for anti-clockwise 
    turns = (turns % 4 + 4) % 4; 
    for (int t=0;t<turns;t++) {
        rotate_strips(env, env->strips[face]);
        rotate_face(env, face);
    }
}*/

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

/*static inline bool rendering_enabled(void) {
    return IsWindowReady();
}*/

float score(Cube* env) {

    float temp_score = 1.0f;
    for (int f=0;f<6; f++) {
        int t_colour = f;
        int face_score = 0;
        for (int r = 0; r < env->N; r++) {
            for (int c = 0; c < env->N; c++) {
                if (STICKER(env, f,r,c) == t_colour) {
                    face_score += 1.0;
                }
            }
        }
        temp_score *= face_score;
    }   
    return temp_score;
}

int is_solved(Cube *env) {
    for (int f = 0; f < 6; f++) {
        int color = f;
        for (int r = 0; r < env->N; r++) {
            for (int c = 0; c < env->N; c++) {
                if (STICKER(env, f, r, c) != color) {
                    return 0; 
                }
            }
        }
    }
    return 1; 
}

void print_stickers(Cube* env) {
    for (int f=0;f<6; f++) {
        printf("Face %d:\n", f);
        for (int r=0; r<env->N; r++) {
            for (int c=0; c<env->N; c++) {
                printf("%d ", STICKER(env,f,r,c));
            }
            printf("\n");
        }
        printf("\n");
    }
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


void print_strips(Cube *env) {
    const char *names[6] = {"U","D","R","L","F","B"};
    for (int f=0; f<6; f++) {
        printf("Face %s strips:\n", names[f]);
        for (int s=0; s<4; s++) {
            printf("  Strip %d: ", s);
            for (int k=0; k<env->N; k++) {
                int r = env->strips[f][s].row + env->strips[f][s].dr * k;
                int c = env->strips[f][s].col + env->strips[f][s].dc * k;
                printf("(%d,%d,%d) ", env->strips[f][s].face, r, c);
            }
            printf("\n");
        }
    }
}
// --- test harness ---
void test_moves(Cube *env) {
    const char *names[6] = {"U","D","R","L","F","B"};

    reset_stickers(env);
    printf("Initial (solved):\n");
    print_stickers(env);

    for (int f=0; f<6; f++) {
        printf("\n=== Test face %s ===\n", names[f]);

        // apply one clockwise turn
        move(env, f, +1);
        printf("After %s:\n", names[f]);
        print_stickers(env);

        // undo with 3 clockwise (equivalent to one counter-clockwise)
        move(env, f, +3);
        printf("After %s + %s':\n", names[f], names[f]);
        print_stickers(env);

        // check restored
        if (!is_solved(env))
            printf("ERROR: %s rotation did not restore\n", names[f]);
        else
            printf("OK: %s rotation restores cube\n", names[f]);
    }
}


//RENDER STUFF
//
//
//





static inline Vector3 axis_vector(int axis) {
    return (axis==0)? (Vector3){1,0,0} :
           (axis==1)? (Vector3){0,1,0} :
                      (Vector3){0,0,1};
}

static inline int in_layer(Vector3 pos, int axis, int layer, int N) {
    float half = (N - 1) / 2.0f;
    // spacing must match cubelet spacing in c_render
    float spacing = 1.1f;
    int coord = (axis==0)? (int)roundf(pos.x/spacing + half) :
                (axis==1)? (int)roundf(pos.y/spacing + half) :
                           (int)roundf(pos.z/spacing + half);
    return coord == layer;
}


#include "rlgl.h"

static void DrawQuad(Vector3 v1, Vector3 v2, Vector3 v3, Vector3 v4, Color color) {
    rlBegin(RL_QUADS);
        rlColor4ub(color.r, color.g, color.b, color.a);
        rlVertex3f(v1.x, v1.y, v1.z);
        rlVertex3f(v2.x, v2.y, v2.z);
        rlVertex3f(v3.x, v3.y, v3.z);
        rlVertex3f(v4.x, v4.y, v4.z);
    rlEnd();
}

void DrawCubelet(Vector3 pos, float size, Color faceColors[6]) {
    float h = size * 0.5f;

    // +X
    DrawQuad(
        (Vector3){pos.x+h, pos.y-h, pos.z+h},
        (Vector3){pos.x+h, pos.y-h, pos.z-h},
        (Vector3){pos.x+h, pos.y+h, pos.z-h},
        (Vector3){pos.x+h, pos.y+h, pos.z+h},
        faceColors[0]);

    // -X
    DrawQuad(
        (Vector3){pos.x-h, pos.y-h, pos.z-h},
        (Vector3){pos.x-h, pos.y-h, pos.z+h},
        (Vector3){pos.x-h, pos.y+h, pos.z+h},
        (Vector3){pos.x-h, pos.y+h, pos.z-h},
        faceColors[1]);

    // +Y
    DrawQuad(
        (Vector3){pos.x-h, pos.y+h, pos.z+h},
        (Vector3){pos.x+h, pos.y+h, pos.z+h},
        (Vector3){pos.x+h, pos.y+h, pos.z-h},
        (Vector3){pos.x-h, pos.y+h, pos.z-h},
        faceColors[2]);

    // -Y
    DrawQuad(
        (Vector3){pos.x-h, pos.y-h, pos.z-h},
        (Vector3){pos.x+h, pos.y-h, pos.z-h},
        (Vector3){pos.x+h, pos.y-h, pos.z+h},
        (Vector3){pos.x-h, pos.y-h, pos.z+h},
        faceColors[3]);

    // +Z
    DrawQuad(
        (Vector3){pos.x-h, pos.y-h, pos.z+h},
        (Vector3){pos.x+h, pos.y-h, pos.z+h},
        (Vector3){pos.x+h, pos.y+h, pos.z+h},
        (Vector3){pos.x-h, pos.y+h, pos.z+h},
        faceColors[4]);

    // -Z
    DrawQuad(
        (Vector3){pos.x+h, pos.y-h, pos.z-h},
        (Vector3){pos.x-h, pos.y-h, pos.z-h},
        (Vector3){pos.x-h, pos.y+h, pos.z-h},
        (Vector3){pos.x+h, pos.y+h, pos.z-h},
        faceColors[5]);
}


/*static void rotate_layer_indices(Cubelet_r *cubelets, int total, int N,
                                 int axis, int layer, int dir) {
    for (int i=0; i<total; i++) {
        int x = cubelets[i].ix;
        int y = cubelets[i].iy;
        int z = cubelets[i].iz;

        if ((axis==0 && x==layer) ||
            (axis==1 && y==layer) ||
            (axis==2 && z==layer)) {

            int nx=x, ny=y, nz=z;
            if (axis==0) { // rotate around X
                ny = (dir>0) ? (N-1 - z) : z;
                nz = (dir>0) ? y         : (N-1 - y);
            } else if (axis==1) { // rotate around Y
                nx = (dir>0) ? z         : (N-1 - z);
                nz = (dir>0) ? (N-1 - x) : x;
            } else { // axis==2
                nx = (dir>0) ? (N-1 - y) : y;
                ny = (dir>0) ? x         : (N-1 - x);
            }
            cubelets[i].ix = nx;
            cubelets[i].iy = ny;
            cubelets[i].iz = nz;
        }
    }
}*/



// Required function. Should handle creating the client on first call

void c_render(Cube* env) {
    env->render = 1;
    static int initialized = 0;
    static Camera camera;

    float half = (env->N - 1) / 2.0f;
    float spacing = 1.1f;

     // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (!initialized) {
        if (!IsWindowReady()) {
            InitWindow(800, 600, "PufferLib Rubik's");
            SetTargetFPS(60);
        }

        camera.position = (Vector3){10.0f,10.0f,10.0f};
        camera.target   = (Vector3){0.0f,0.0f,0.0f};
        camera.up       = (Vector3){0.0f,1.0f,0.0f};
        camera.fovy     = 45.0f;
        camera.projection = CAMERA_PERSPECTIVE;

        initialized = 1;
    }

    BeginDrawing();
    ClearBackground((Color){6,24,24,255});
    BeginMode3D(camera);
    UpdateCamera(&camera, CAMERA_THIRD_PERSON);

    for (int x=0; x<env->N; x++) {
        for (int y=0; y<env->N; y++) {
            for (int z=0; z<env->N; z++) {

                Vector3 pos = (Vector3){
                    (x-half)*spacing,
                    (y-half)*spacing,
                    (z-half)*spacing
                };

                Color faces[6] = { BLACK, BLACK, BLACK, BLACK, BLACK, BLACK };

                // Right (+X)
                if (x == env->N - 1)
                    faces[0] = sticker_colors[ STICKER(env, R, env->N - 1 - y, env->N - 1 - z) ];
                // Left (−X)
                if (x == 0)
                    faces[1] = sticker_colors[ STICKER(env, L, env->N - 1 - y, z) ];
                // Up (+Y)
                if (y == env->N - 1)
                    faces[2] = sticker_colors[ STICKER(env, U, z, x) ];
                // Down (−Y)
                if (y == 0)
                    faces[3] = sticker_colors[ STICKER(env, D, env->N-1-z, x) ];
                // Front (+Z)
                if (z == env->N - 1)
                    faces[4] = sticker_colors[ STICKER(env, F, env->N - 1 - y, x) ];
                // Back (−Z)
                if (z == 0)
                    faces[5] = sticker_colors[ STICKER(env, B, env->N - 1 - y, env->N - 1 - x) ];
                rlPushMatrix();
                // rotate only the turning layer while animating
                if (anim.rotating && in_layer(pos, anim.axis, anim.layer, env->N)) {
                    Vector3 axis = axis_vector(anim.axis);
                    rlRotatef(anim.dir * (anim.elapsed / anim.duration) * 90.0f,
                              axis.x, axis.y, axis.z);
                }
                rlTranslatef(pos.x, pos.y, pos.z);
                DrawCubelet((Vector3){0,0,0}, 1.0f, faces);
                rlPopMatrix();
            }
        }
    }

    EndMode3D();
    char buf[50];
    snprintf(buf, sizeof(buf), "Tick %d", env->tick);
    DrawText(buf, 10, 10, 20, WHITE);

    snprintf(buf, sizeof(buf), "Score %.2f", env->score);
    DrawText(buf, 10, 40, 20, WHITE);

    EndDrawing();
}

// Required function


void c_step(Cube* env) {
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->tick += 1;

    int face, turns;
    decode_action(env->actions[0], &face, &turns);
    //printf("Action: face=%d turns=%d\n", face, turns);
    //printf("Before move:\n");
    //print_stickers(env);



    //printf("ENTER c_step tick=%d\n", env->tick);

    //anim.dir = dir;
    //if (face == U || face == R || face == F)
     //anim.dir = -dir;
     
    // Faces: U=0, D=1, R=2, L=3, F=4, B=5
    //
    //
    static const int FACE_AXIS[6]  = {1, 1, 0, 0, 2, 2};          // Y,Y,X,X,Z,Z
    static const int FACE_LAYER[6] = {1, 0, 1, 0, 1, 0};          // 1->N-1, 0->0
    static const int FACE_SIGN[6]  = {-1,-1,-1,+1,-1,+1};         // anim sign
    int dir = (turns > 0) ? +1 : -1;

    if (env->render) {

        anim.rotating = 1;
        anim.axis     = FACE_AXIS[face];
        anim.layer    = FACE_LAYER[face] ? env-> N-1 : 0 ;
        anim.dir      = FACE_SIGN[face] * dir;
        anim.elapsed  = 0.0f;
        anim.duration = 1.0f; // seconds per move
                                                     //
        // animate with OLD stickers
        while (anim.elapsed < anim.duration) {
            if (WindowShouldClose()) break;
            anim.elapsed += GetFrameTime();
            c_render(env);
        }

        // only now commit the move
        move(env, face, turns);
        anim.rotating = 0;
    } else {
        // no animation, just commit directly
        move(env, face, turns);
    }
    //c_render(env);

   // printf("After move:\n");
   // print_stickers(env);

    env->score = score(env);
    env->rewards[0] -= 1.0f;

    /*if (is_solved(env)) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0f;
        add_log(env);
        c_reset(env);
        return;
    }*/
    if (env->tick >= env->max_episode_steps) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    compute_observations(env);
}



// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Cube* env) {
    //free memory you allocated in init
    free(env->stickers);
    free(env->cubelets);
   if (IsWindowReady()) {
        CloseWindow();      
    }
}
