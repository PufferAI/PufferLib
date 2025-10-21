//Some code inspired by https://github.com/Princeton-RL/CRTR/blob/main/envs/rubik/gym_rubik/envs/cube.py

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"
#include <stdio.h>
#include "rlgl.h"


typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    float n; // Required as the last field 
} Log;


//Describes how to find the edge strips
typedef struct {
    int face;  // face index
    int row;   // starting row
    int col;   // starting col
    int dr;    // row step
    int dc;    // col step
} strip_t;


//Holds individual cublets (27 in 3x3 cube)
typedef struct {
    Vector3 pos;
    Color faces[6];
    int ix, iy, iz;   // grid coordinates 0..N-1
} Cubelet_r;


//Main env
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
    float anim_time;
    int user_mode;
    int highlight_layer;
    int highlight_axis;
    float episode_return;
} Cube;



//Faces are Up, Down, Right, Left, Front, Back
enum { U=0, D=1, R=2, L=3, F=4, B=5 };

static Color sticker_colors[6];

void init_sticker_colors(void) {
    sticker_colors[0] = WHITE;
    sticker_colors[1] = YELLOW;
    sticker_colors[2] = RED;
    sticker_colors[3] = ORANGE;
    sticker_colors[4] = GREEN;
    sticker_colors[5] = BLUE;
}


//Holds info for the animation
typedef struct {
    int rotating;      // 0 idle, 1 anim
    float elapsed;
    float duration;    // e.g. 0.5f
    int axis;          // 0=X,1=Y,2=Z
    int layer;         // 0..N-1
    int dir;           // +1 or -1
} MoveState;

static MoveState anim = {0};

//Puffer logging 
void add_log(Cube* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->score;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.n++;
    init_sticker_colors();
}

#define OBS(env,f,r,c,color) \
        ((env)->observations[ ((f)*(env)->N*(env)->N*6) + ((r)*(env)->N*6) + ((c)*6) + (color) ])
#define STICKER(env,f,r,c) ((env)->stickers[(f)*(env)->N*(env)->N + (r)*(env)->N + (c)])

static int *tmp; //temp array for computing strips and rotations later
static int *r_tmp; //temp array for computing strips and rotations later
#define R_TMP(i,j) tmp[(i)*(env)->N + (j)]


// Precompute strips that surround each face
void precompute_strips(Cube *env) {
    int N = env->N;
    // For each face looking at it moving clockwise. Strips on other faces that rotate 
    // with the face. Order is for clockwise rotation. Descrives how to start traversing the strip
    // {Face, starting row, starting col, direction row, direction col}
    // NB inconsistent use of directions here possibly better to use consistent schema but it works
    // for now so I don't want to break it!
    // FRONT (F):
    env->strips[F][0] = (strip_t){U, N-1, 0,   0,  1};  
    env->strips[F][1] = (strip_t){R, 0,   0,   1,  0};   
    env->strips[F][2] = (strip_t){D, 0,   N-1, 0, -1};   
    env->strips[F][3] = (strip_t){L, N-1, N-1,-1,  0};                                                          
    // BACK (B): 
    env->strips[B][0] = (strip_t){U, 0,    N-1, 0, -1};   
    env->strips[B][1] = (strip_t){L, 0,    0,   1,  0};   
    env->strips[B][2] = (strip_t){D, N-1,  0,   0,  1};  
    env->strips[B][3] = (strip_t){R, N-1,  N-1, -1, 0};  
    // UP face 
    env->strips[U][0] = (strip_t){F, 0, 0,   0, +1}; 
    env->strips[U][1] = (strip_t){L, 0, 0,   0, +1};  
    env->strips[U][2] = (strip_t){B, 0, 0,   0, +1};  
    env->strips[U][3] = (strip_t){R, 0, 0,   0, +1};  

    // DOWN (D): 
    env->strips[D][0] = (strip_t){F, N-1, 0, 0, 1}; 
    env->strips[D][1] = (strip_t){L, N-1, 0, 0, 1};  
    env->strips[D][2] = (strip_t){B, N-1, 0, 0, 1};  
    env->strips[D][3] = (strip_t){R, N-1, 0, 0, 1};  
  
    // RIGHT (R):
    env->strips[R][0] = (strip_t){U, 0,   N-1, 1, 0};    
    env->strips[R][1] = (strip_t){B, N-1, 0,  -1,0};    
    env->strips[R][2] = (strip_t){D, 0,   N-1, 1, 0};   
    env->strips[R][3] = (strip_t){F, 0,   N-1, 1, 0};  

    // LEFT (L):
    env->strips[L][0] = (strip_t){U, 0,   0, 1, 0};  
    env->strips[L][1] = (strip_t){F, 0,   0, 1, 0}; 
    env->strips[L][2] = (strip_t){D, 0,   0, 1, 0}; 
    env->strips[L][3] = (strip_t){B, N-1, N-1,-1,0}; 
}

//Main init
void init(Cube* env) {
    env->stickers = malloc(6 * env->N * env->N * sizeof(int));
    env->total_cubelets = 0;
    precompute_strips(env);
    env->render = 0;
    env->user_mode = 0;
    env->highlight_axis = 0;  // 0=X,1=Y,2=Z
    env->highlight_layer = 0;
    env->anim_time = 0.5;
    tmp = malloc(env->N * sizeof(int));
    r_tmp = malloc(env->N * env->N * sizeof(int));
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

//To set OHE in the obs space
static inline void set_color(Cube *env, int f, int r, int c, int k) {
    for (int ch=0; ch<6; ch++)
        OBS(env,f,r,c,ch) = (ch == k) ? 1.0f : 0.0f;
}

void compute_observations(Cube* env) {
    for (int f=0; f<6; f++) { //face
        for (int r=0; r<env->N; r++) { //row
            for (int c=0; c<env->N; c++) { //col
                int colour = STICKER(env,f,r,c);
                set_color(env, f, r, c, colour);
            }
        }
    }
}


//Just rotates the strips CLOCKWISE, not the face itself
static void rotate_strips(Cube *env, strip_t s[4]) {
    int N = env->N;
    //Copy last strip
    for (int k=0;k<N;k++)
        tmp[k] = STICKER(env, s[3].face, s[3].row + s[3].dr*k, s[3].col + s[3].dc*k);
    //Shift
    for (int j=3;j>0;j--) {
        for (int k=0;k<N;k++) {
            STICKER(env, s[j].face, s[j].row + s[j].dr*k, s[j].col + s[j].dc*k) =
                STICKER(env, s[j-1].face, s[j-1].row + s[j-1].dr*k, s[j-1].col + s[j-1].dc*k);
        }
    }
    //Copy last back
    for (int k=0;k<N;k++)
        STICKER(env, s[0].face, s[0].row + s[0].dr*k, s[0].col + s[0].dc*k) = tmp[k];
}
  
// Rotates the strips COUNTER-CLOCKWISE
static void rotate_strips_ccw(Cube *env, strip_t s[4]) {
    int N = env->N;
    //Copy first strip
    for (int k=0;k<N;k++)
        tmp[k] = STICKER(env, s[0].face,s[0].row + s[0].dr*k,s[0].col + s[0].dc*k);
    //shift others
    for (int j=0;j<3;j++) {
        for (int k=0;k<N;k++) {
            STICKER(env, s[j].face,s[j].row + s[j].dr*k,s[j].col + s[j].dc*k) =
                STICKER(env, s[j+1].face,s[j+1].row + s[j+1].dr*k,s[j+1].col + s[j+1].dc*k);
        }
    }
    //Copy first
    for (int k=0;k<N;k++)
        STICKER(env, s[3].face,s[3].row + s[3].dr*k,s[3].col + s[3].dc*k) = tmp[k];
}

//Just rotates face stickers counter-clockwise
static void rotate_face_ccw(Cube *env, int f) {
    int N = env->N;
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            R_TMP(N-1-j,i) = STICKER(env,f,i,j);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            STICKER(env,f,i,j) = R_TMP(i,j);
}

//Just rotates the face stickers CLOCKWISE
static void rotate_face(Cube *env, int f) {
    int N = env->N;
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            R_TMP(j,N-1-i) = STICKER(env,f,i,j);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            STICKER(env,f,i,j) = R_TMP(i,j);
}

//Execute move for face, rotate face rotate strips in theory supports multiple turns but only 1 tested
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

//Distance from solved based on centre sticker as the colour for that face
//VERY rough heuristic
float score(Cube *env) {
    float temp_score = 1.0f;
    for (int f = 0; f < 6; f++) {
        int t_colour = f;
        int face_score = 0;
        for (int r = 0; r < env->N; r++) {
            for (int c = 0; c < env->N; c++) {
                if (STICKER(env, f, r, c) == t_colour)
                    face_score++;
            }
        }
        temp_score *= (float)face_score;
    }
    return temp_score;
}

//NB in this code we dont move centre stickers so face colour = centre sticker as in score
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

// Required function
void c_reset(Cube* env) {
    memset(env->observations, 0, sizeof(float) * env->size); 
    reset_stickers(env); 
    shuffle(env, env->shuffles);
    env->tick = 0;
    env->score = 0;
    env->episode_return = 0;
    compute_observations(env);
}

//Some debugging functions

void print_stickers_file(Cube* env, FILE *out) {
    for (int f=0; f<6; f++) {
        fprintf(out, "Face %d:\n", f);
        for (int r=0; r<env->N; r++) {
            for (int c=0; c<env->N; c++) {
                fprintf(out, "%d ", STICKER(env,f,r,c));
            }
            fprintf(out, "\n");
        }
        fprintf(out, "\n");
    }
}

void print_stickers(Cube* env) {
    print_stickers_file(env, stdout);
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


// Step Code at bottom as unfortunately we need to call render code in step for animations


/* MAIN RENDERING CODE */

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

void c_render(Cube* env) {
    env->render = 1; //Important global window for anims so need to turn on for this env only
    static int initialized = 0;
    static Camera camera;
    float half = (env->N - 1) / 2.0f;
    float spacing = 1.1f; //Needs to match 'in layer' code

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

    //BACKGROUND
    float size = 20.0f;   // half-size of the room
    int steps = 20;       // subdivisions per wall
    float step = (2*size) / steps;
    Color cyan = (Color){0,255,255,255};

    // XY planes at z = ±size
    for (int i = 0; i <= steps; i++) {
        float x = -size + i*step;
        DrawLine3D((Vector3){x,-size,-size}, (Vector3){x,size,-size}, cyan);
        DrawLine3D((Vector3){x,-size, size}, (Vector3){x,size, size}, cyan);
    }
    for (int j = 0; j <= steps; j++) {
        float y = -size + j*step;
        DrawLine3D((Vector3){-size,y,-size}, (Vector3){ size,y,-size}, cyan);
        DrawLine3D((Vector3){-size,y, size}, (Vector3){ size,y, size}, cyan);
    }

    // XZ planes at y = ±size
    for (int i = 0; i <= steps; i++) {
        float x = -size + i*step;
        DrawLine3D((Vector3){x,-size,-size}, (Vector3){x,-size, size}, cyan);
        DrawLine3D((Vector3){x, size,-size}, (Vector3){x, size, size}, cyan);
    }
    for (int j = 0; j <= steps; j++) {
        float z = -size + j*step;
        DrawLine3D((Vector3){-size,-size,z}, (Vector3){ size,-size,z}, cyan);
        DrawLine3D((Vector3){-size, size,z}, (Vector3){ size, size,z}, cyan);
    }

    // YZ planes at x = ±size
    for (int i = 0; i <= steps; i++) {
        float y = -size + i*step;
        DrawLine3D((Vector3){-size,y,-size}, (Vector3){-size,y, size}, cyan);
        DrawLine3D((Vector3){ size,y,-size}, (Vector3){ size,y, size}, cyan);
    }
    for (int j = 0; j <= steps; j++) {
        float z = -size + j*step;
        DrawLine3D((Vector3){-size,-size,z}, (Vector3){-size, size,z}, cyan);
        DrawLine3D((Vector3){ size,-size,z}, (Vector3){ size, size,z}, cyan);
    }
    

    //CUBE
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
    //for highlights
    if (env->user_mode){
       
        rlDisableDepthTest();
        rlDisableBackfaceCulling();

        // change axis
        if (IsKeyPressed(KEY_UP))    env->highlight_axis = (env->highlight_axis + 1) % 3;
        if (IsKeyPressed(KEY_DOWN))  env->highlight_axis = (env->highlight_axis + 2) % 3;

        // toggle between external layers
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_LEFT)) {
            env->highlight_layer = (env->highlight_layer == 0) ? env->N - 1 : 0;


}        // draw highlight
        float spacing = 1.1f;
        float half = (env->N-1)/2.0f;
        float coord = (env->highlight_layer-half)*spacing;
        float extent = (env->N*spacing)/2.0f + 0.1f;
        Color highlight = (Color){0,255,255,100}; // translucent yellow


        float thickness = spacing;  // slab thickness
        Vector3 pos = {0,0,0};
        float dx = 2*extent, dy = 2*extent, dz = 2*extent;

        if (env->highlight_axis == 0) {
            pos = (Vector3){coord, 0, 0};
            dx = thickness;   // thin along X
        }
        else if (env->highlight_axis == 1) {
            pos = (Vector3){0, coord, 0};
            dy = thickness;   // thin along Y
        }
        else {
            pos = (Vector3){0, 0, coord};
            dz = thickness;   // thin along Z
        }
        float hx = dx * 0.5f;
        float hy = dy * 0.5f;
        float hz = dz * 0.5f;

        // +X
        DrawQuad((Vector3){pos.x+hx, pos.y-hy, pos.z-hz},
                 (Vector3){pos.x+hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z-hz}, highlight);

        // -X
        DrawQuad((Vector3){pos.x-hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x-hx, pos.y-hy, pos.z-hz},
                 (Vector3){pos.x-hx, pos.y+hy, pos.z-hz},
                 (Vector3){pos.x-hx, pos.y+hy, pos.z+hz}, highlight);

        // +Y
        DrawQuad((Vector3){pos.x-hx, pos.y+hy, pos.z-hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z-hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z+hz},
                 (Vector3){pos.x-hx, pos.y+hy, pos.z+hz}, highlight);

        // -Y
        DrawQuad((Vector3){pos.x-hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y-hy, pos.z-hz},
                 (Vector3){pos.x-hx, pos.y-hy, pos.z-hz}, highlight);

        // +Z
        DrawQuad((Vector3){pos.x-hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z+hz},
                 (Vector3){pos.x-hx, pos.y+hy, pos.z+hz}, highlight);

        // -Z
        DrawQuad((Vector3){pos.x+hx, pos.y-hy, pos.z-hz},
                 (Vector3){pos.x-hx, pos.y-hy, pos.z-hz},
                 (Vector3){pos.x-hx, pos.y+hy, pos.z-hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z-hz}, highlight);
       
    rlEnableDepthTest();  
    }
    EndMode3D();

   
    rlEnableBackfaceCulling();
    char buf[50];
    snprintf(buf, sizeof(buf), "Tick %d", env->tick);
    DrawText(buf, 10, 10, 20, WHITE);

    snprintf(buf, sizeof(buf), "Score %.2f", env->score);
    DrawText(buf, 10, 40, 20, WHITE);

    EndDrawing();
}


void c_step(Cube* env) {
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->tick += 1;

    int face, turns;
    decode_action(env->actions[0], &face, &turns);

    static const int FACE_AXIS[6]  = {1, 1, 0, 0, 2, 2};          
    static const int FACE_LAYER[6] = {1, 0, 1, 0, 1, 0};         
    static const int FACE_SIGN[6]  = {-1,-1,-1,+1,-1,+1};        
    int dir = (turns > 0) ? +1 : -1;

    if (env->render) {

        anim.rotating = 1;
        anim.axis     = FACE_AXIS[face];
        anim.layer    = FACE_LAYER[face] ? env-> N-1 : 0 ;
        anim.dir      = FACE_SIGN[face] * dir;
        anim.elapsed  = 0.0f;
        anim.duration = env->anim_time; // seconds per move
        // animate with OLD stickers
        while (anim.elapsed < anim.duration) {
            if (WindowShouldClose()) break;
            anim.elapsed += GetFrameTime();
            c_render(env);
        }
        move(env, face, turns);
        anim.rotating = 0;
    } else {
        move(env, face, turns);
    }


    env->score = score(env);
    env->rewards[0] -= 1.0f;

   if (is_solved(env)) {
       env->terminals[0] = 1;
       env->rewards[0] = 1.0f;
       env->episode_return += env->rewards[0];
       add_log(env);
       c_reset(env);
       return;
   }

   if (env->tick >= env->max_episode_steps) {
       env->terminals[0] = 1;
       env->episode_return += env->rewards[0];
       add_log(env);
       c_reset(env);
       return;
   }
    env->episode_return += env->rewards[0];
    compute_observations(env);
}

void c_close(Cube* env) {
    free(env->stickers);
    free(tmp);
    free(r_tmp);
   if (IsWindowReady()) {
        CloseWindow();      
   }
}
