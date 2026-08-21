// Raylib 3D rendering of a physics.h model: geoms, contacts and a tracking
// camera. MuJoCo is z-up, raylib y-up: (x, y, z) -> (x, z, -y).
#include <stdlib.h>
#include "raylib.h"
#include "rlgl.h"

const Color MJ_BACKGROUND = (Color){6, 24, 24, 255};
const Color MJ_BODY = (Color){0, 187, 187, 255};
const Color MJ_CONTACT = (Color){187, 0, 0, 255};

Vector3 mj_rl(const float* p) {
    return (Vector3){p[0], p[2], -p[1]};
}

// Draw one frame from qpos: window setup, camera `back` behind and `up` above
// the target (MuJoCo coords), geoms, contacts and a HUD line
void mj_render(const MjModel* m, MjData* d, const char* title, const float* target, float back,
    float up, const char* hud) {
    if (!IsWindowReady()) {
        InitWindow(1280, 720, title);
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    mj_kinematics(m, d);
    mj_collision(m, d);
    float eye[3] = {target[0], target[1] - back, target[2] + up};
    Camera3D camera = {.position = mj_rl(eye), .target = mj_rl(target), .up = {0, 1, 0},
        .fovy = 45.0f, .projection = CAMERA_PERSPECTIVE};
    BeginDrawing();
    ClearBackground(MJ_BACKGROUND);
    BeginMode3D(camera);
    for (int g = 0; g < m->ngeom; g++) {
        const float* size = m->geom_size[g];
        float* pos = d->geom_xpos[g];
        float* mat = d->geom_xmat[g];
        float a[3] = {pos[0] - size[1]*mat[2], pos[1] - size[1]*mat[5], pos[2] - size[1]*mat[8]};
        float b[3] = {pos[0] + size[1]*mat[2], pos[1] + size[1]*mat[5], pos[2] + size[1]*mat[8]};
        int type = m->geom_type[g];
        if (type == MJ_GEOM_PLANE) {
            rlPushMatrix();
            rlTranslatef(pos[0], pos[2], -pos[1]);
            DrawGrid(400, 1.0f);
            rlPopMatrix();
        } else if (type == MJ_GEOM_SPHERE) {
            DrawSphere(mj_rl(pos), size[0], MJ_BODY);
        } else if (type == MJ_GEOM_CAPSULE) {
            DrawCapsule(mj_rl(a), mj_rl(b), size[0], 12, 4, MJ_BODY);
        } else if (type == MJ_GEOM_CYLINDER) {
            DrawCylinderEx(mj_rl(a), mj_rl(b), size[0], size[0], 16, MJ_BODY);
        } else if (type == MJ_GEOM_BOX) {
            // column-major OpenGL matrix of P R P^T with P the z-up to y-up map
            float t[16] = {mat[0], mat[6], -mat[3], 0.0f, mat[2], mat[8], -mat[5], 0.0f,
                -mat[1], -mat[7], mat[4], 0.0f, pos[0], pos[2], -pos[1], 1.0f};
            rlPushMatrix();
            rlMultMatrixf(t);
            DrawCube((Vector3){0, 0, 0}, 2.0f*size[0], 2.0f*size[2], 2.0f*size[1], MJ_BODY);
            rlPopMatrix();
        }
    }
    for (int c = 0; c < d->ncon; c++) {
        DrawSphere(mj_rl(d->contact[c].pos), 0.02f, MJ_CONTACT);
    }
    EndMode3D();
    DrawText(hud, 20, 20, 20, RAYWHITE);
    EndDrawing();
    puf_web_vsync();
}
