#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "raylib.h"
#include "rcamera.h"
#include "raymath.h"

const unsigned char NOOP    = 0;
const unsigned char S_DOWN  = 1;
const unsigned char S_UP    = 2;
const unsigned char S_LEFT  = 3;
const unsigned char S_RIGHT = 4;

const float RAY_LENGTH = 100.0f;
const Color RAY_COLOR  = GREEN;

const int HEIGHT = 450;
const int WIDTH  = 800;

// Required struct. Only use floats!
typedef struct {
    float episode_length;
    float score;
    float n;
} Log;

typedef struct {
    Color pixels[64];
    BoundingBox bounding_boxes[6];
    Vector3 square_positions[6];
    Color square_colors[6];
    Log log;
    Camera camera;
    Ray target_ray;

    unsigned char* terminals;
    float* rewards;
    float score;
    int* actions;
    int tick;
    int color_indices[6];
    int selected_color_name;
    int time_since_score;
    int selected_box;
    int score_tick;
    int last_hit;
    uint8_t* observations;
} VisionTest;

const Color BASE_COLORS[6] = { RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE };
const char *COLOR_NAMES[6] = { "RED", "BLUE", "GREEN", "YELLOW", "PURPLE", "ORANGE" };

const float FOVY= 60.0;
const int BOX_SIZE = 20;

const float CELL_WIDTH = (float)WIDTH/8;
const float CELL_HEIGHT = (float)HEIGHT/8;
const float TAN_FOVY_2 = 0.577350;
const float ASPECT_RATIO = (float)WIDTH/HEIGHT;

#define FLT_MAX 3.40282346638528859811704183484516925440e+38f

void add_log(VisionTest* env) {
    env->log.episode_length += env->tick;
    env->log.score += env->score; 
    env->log.n++;
}

void Respawn(VisionTest* env) {
    Vector3 base_positions[6] = {
        (Vector3){ 0.0f, 0.0f, -2*BOX_SIZE },
        (Vector3){ 0.0f, 0.0f, 2*BOX_SIZE },
        (Vector3){ -2*BOX_SIZE, 0.0f, 0.0f },
        (Vector3){ 2*BOX_SIZE, 0.0f, 0.0f },
        (Vector3){ 0.0f, 2*BOX_SIZE, 0.0f }, 
        (Vector3){ 0.0f, -2*BOX_SIZE, 0.0f }
    };

    int color_indices[6] = { 0, 1, 2, 3, 4, 5 };
    for (int i = 5; i > 0; i--) { // Fisher-Yates shuffle for color indices
        int j = GetRandomValue(0, i);
        int temp = color_indices[i];
        color_indices[i] = color_indices[j];
        color_indices[j] = temp;
    }

    env->selected_box = GetRandomValue(0, 5);
    env->selected_color_name = color_indices[env->selected_box];
    env->observations[64*3] = floor(256 * env->selected_color_name)/6 ;

    for (int i = 0; i < 6; i++) {
        env->square_positions[i] = base_positions[i];
        env->square_colors[i] = BASE_COLORS[color_indices[i]];
        env->color_indices[i] = color_indices[i];
        env->bounding_boxes[i].min = (Vector3){
            env->square_positions[i].x - BOX_SIZE/2,
            env->square_positions[i].y - BOX_SIZE/2,
            env->square_positions[i].z - BOX_SIZE/2
        };
        env->bounding_boxes[i].max = (Vector3){
            env->square_positions[i].x + BOX_SIZE/2,
            env->square_positions[i].y + BOX_SIZE/2,
            env->square_positions[i].z + BOX_SIZE/2
        };
    }

}

void c_reset(VisionTest* env) {
    memset(env->observations, 0, sizeof(uint8_t)*(64*3 + 1));
    memset(env->pixels, 0, sizeof(Color)*(64));

    Camera camera = env->camera;
    camera.target = (Vector3){ 0.0f, 2.0f, 0.0f };
    camera.position = (Vector3){ 0.0f, 2.0f, 4.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 60.0f;
    camera.projection = CAMERA_CUSTOM;
    env->camera = camera;

    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->tick = 0;
    env->score = 0;
    env->time_since_score = 0;
    env->score_tick = 0;

    Respawn(env);
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(VisionTest* env) {
    if(IsWindowReady()){
        CloseWindow();
    }
}

// Required function. Should handle creating the client on first call
void c_render(VisionTest* env) {
    if (!IsWindowReady()) {
        Camera camera = env->camera;

        camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
        camera.fovy = 60.0f;
        camera.position = (Vector3){ 0.0f, 2.0f, 4.0f };
        camera.target = (Vector3){ 0.0f, 2.0f, 0.0f };
        camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
        camera.fovy = 60.0f;
        camera.projection = CAMERA_CUSTOM;
        env->camera = camera;
        InitWindow(WIDTH, HEIGHT, "PufferLib VisionTest");
        SetTargetFPS(30);
        Respawn(env);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    RenderTexture2D rt = LoadRenderTexture(WIDTH, HEIGHT);

    Camera camera = env->camera;
    BeginTextureMode(rt);
        ClearBackground(BLACK);
        BeginMode3D(camera);
            for (int i = 0; i < 6; i++) {
                DrawCube(env->square_positions[i], BOX_SIZE, BOX_SIZE, BOX_SIZE, env->square_colors[i]);
                DrawCubeWires(env->square_positions[i], BOX_SIZE, BOX_SIZE, BOX_SIZE, WHITE);
            }
            Vector3 rayEnd = Vector3Add(camera.position, Vector3Scale(env->target_ray.direction, RAY_LENGTH));
            DrawCylinderWiresEx(camera.position, rayEnd, 0.1f, 0.1f, 4, GREEN);
        EndMode3D();

        // Draw 2D UI overlays here (on top of the 3D render) well use this later for vision purposes
        DrawRectangle(600, 5, 190, 40, Fade(SKYBLUE, 0.5f));
        DrawRectangleLines(600, 5, 190, 40, BLUE);
        DrawText(TextFormat("- Score: %f", env->score), 610, 10, 10, BLACK);
        DrawText(TextFormat("- Find Color: %s", COLOR_NAMES[env->selected_color_name]), 610, 25, 10, BLACK);
    EndTextureMode();

    env->camera = camera;
    
    BeginDrawing();
        DrawTextureRec(rt.texture, (Rectangle){ 0, 0, (float)rt.texture.width, (float)-rt.texture.height }, (Vector2){ 0, 0 }, WHITE);
        int debugX = 10;
        int debugY = 120;
        int scale = 8; // Magnification (16=128x128 grid; adjust as needed)
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                int index = y * 8 + x;
                Color c = env->pixels[index];
                DrawRectangle(debugX + x * scale, debugY + y * scale, scale, scale, c);
            }
        }
        DrawRectangleLines(debugX, debugY, 8 * scale, 8 * scale, RED);
        DrawText("Puffer POV:", debugX, debugY - 15, 10, WHITE);
    EndDrawing();
}

// I know you see this and think you should do AABB. This is 10% faster from my testing probably because of the near even spread of hits / non hits
bool CheckRayCollisionBox(Ray ray, BoundingBox box) {
    // Quadrant and candidate planes
    int quadrant_x, quadrant_y, quadrant_z;
    float candidatePlane_x, candidatePlane_y, candidatePlane_z;
    int inside = 1;

    // X-axis: Check if origin is left, right, or inside box
    if (ray.position.x < box.min.x) {
        quadrant_x = 0;  // LEFT
        candidatePlane_x = box.min.x;
        inside = 0;
    } else if (ray.position.x > box.max.x) {
        quadrant_x = 1;  // RIGHT
        candidatePlane_x = box.max.x;
        inside = 0;
    } else {
        quadrant_x = 2;  // MIDDLE
    }

    // Y-axis
    if (ray.position.y < box.min.y) {
        quadrant_y = 0;
        candidatePlane_y = box.min.y;
        inside = 0;
    } else if (ray.position.y > box.max.y) {
        quadrant_y = 1;
        candidatePlane_y = box.max.y;
        inside = 0;
    } else {
        quadrant_y = 2;
    }
    
    // Z-axis
    if (ray.position.z < box.min.z) {
        quadrant_z = 0;
        candidatePlane_z = box.min.z;
        inside = 0;
    } else if (ray.position.z > box.max.z) {
        quadrant_z = 1;
        candidatePlane_z = box.max.z;
        inside = 0;
    } else {
        quadrant_z = 2;
    }

    // Early return if origin is inside box
    if (inside) return true;

    // Calculate T distances to candidate planes
    float maxT_x = (quadrant_x != 2 && ray.direction.x != 0.0f) ? (candidatePlane_x - ray.position.x) / ray.direction.x : -1.0f;
    float maxT_y = (quadrant_y != 2 && ray.direction.y != 0.0f) ? (candidatePlane_y - ray.position.y) / ray.direction.y : -1.0f;
    float maxT_z = (quadrant_z != 2 && ray.direction.z != 0.0f) ? (candidatePlane_z - ray.position.z) / ray.direction.z : -1.0f;

    // Find largest maxT
    int whichPlane = 0;
    if (maxT_x < maxT_y) whichPlane = 1;
    if (maxT_x < maxT_z && maxT_y < maxT_z) whichPlane = 2;

    // Early out if maxT is negative (ray points away)
    float maxT = whichPlane == 0 ? maxT_x : whichPlane == 1 ? maxT_y : maxT_z;
    if (maxT < 0.0f) return false;

    // Check hit point bounds for other axes
    if (whichPlane != 0) {  // Check X if not X-plane
        float coord_x = ray.position.x + maxT * ray.direction.x;
        if (coord_x < box.min.x || coord_x > box.max.x) return false;
    }
    if (whichPlane != 1) {  // Check Y if not Y-plane
        float coord_y = ray.position.y + maxT * ray.direction.y;
        if (coord_y < box.min.y || coord_y > box.max.y) return false;
    }
    if (whichPlane != 2) {  // Check Z if not Z-plane
        float coord_z = ray.position.z + maxT * ray.direction.z;
        if (coord_z < box.min.z || coord_z > box.max.z) return false;
    }

    return true;  // Hit confirmed, maxT >= 0.0f already checked
}

void c_step(VisionTest* env) {
    int action = env->actions[0];
    env->tick++;
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    Camera camera = env->camera;
    float rotationSpeed = 0.1;
    if (action == S_DOWN) CameraPitch(&camera, -rotationSpeed, true, false, true);
    if (action == S_UP) CameraPitch(&camera, rotationSpeed, true, false, true);
    if (action == S_RIGHT) CameraYaw(&camera, -rotationSpeed, false);
    if (action == S_LEFT) CameraYaw(&camera, rotationSpeed, false);
    // Set target ray
    env->target_ray.position = camera.position;
    env->target_ray.direction = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 up = Vector3Normalize(camera.up);
    Vector3 right = Vector3CrossProduct(forward, up);

    // Compute closest 3 boxes using dot product since max 3 on screen
    int closest1 = -1, closest2 = -1, closest3 = -1;
    float dot1 = -FLT_MAX, dot2 = -FLT_MAX, dot3 = -FLT_MAX;
    for (int k = 0; k < 6; k++) {
        BoundingBox bb = env->bounding_boxes[k];
        Vector3 center = { (bb.min.x + bb.max.x) * 0.5f, (bb.min.y + bb.max.y) * 0.5f, (bb.min.z + bb.max.z) * 0.5f };
        Vector3 vec_to_center = Vector3Subtract(center, camera.position);
        float dist_sq = Vector3LengthSqr(vec_to_center);
        float dot_k;
        if (dist_sq == 0.0f) {
            dot_k = 1.0f;
        } else {
            Vector3 norm_vec = Vector3Normalize(vec_to_center);
            dot_k = Vector3DotProduct(norm_vec, forward);
        }
        if (dot_k > dot1) {
            dot3 = dot2; closest3 = closest2;
            dot2 = dot1; closest2 = closest1;
            dot1 = dot_k; closest1 = k;
        } else if (dot_k > dot2) {
            dot3 = dot2; closest3 = closest2;
            dot2 = dot_k; closest2 = k;
        } else if (dot_k > dot3) {
            dot3 = dot_k; closest3 = k;
        }
    }

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            int index = j * 8 + i;
            float view_x = (i/4.0 - 7.0/8.0) * ASPECT_RATIO * TAN_FOVY_2;
            float view_y = (7.0/8.0 - j/4.0) * TAN_FOVY_2;
            Vector3 view_dir = { view_x, view_y, 1.0f };
            Vector3 world_dir = {
                right.x * view_dir.x + up.x * view_dir.y + forward.x * view_dir.z,
                right.y * view_dir.x + up.y * view_dir.y + forward.y * view_dir.z,
                right.z * view_dir.x + up.z * view_dir.y + forward.z * view_dir.z
            };
            Ray tempRay;
            Color c;
            tempRay.position = camera.position;
            tempRay.direction = Vector3Normalize(world_dir);

            bool hit = false;
            if (CheckRayCollisionBox(tempRay, env->bounding_boxes[closest1])) {
                c = env->square_colors[closest1];
                hit = true;
            } else if (CheckRayCollisionBox(tempRay, env->bounding_boxes[closest2])) {
                c = env->square_colors[closest2];
                hit = true;
            } else if (CheckRayCollisionBox(tempRay, env->bounding_boxes[closest3])) {
                c = env->square_colors[closest3];
                hit = true;
            } else {
                c = BLACK;
            }
            env->pixels[index] = c;
            env->observations[index] = c.r;
            env->observations[8*8 + index] = c.g;
            env->observations[8*8*2 + index] = c.b;
        }

    }
    env->camera = camera;
    float closestDistance = RAY_LENGTH;
    int hitType = 0;
    RayCollision colCollision = GetRayCollisionBox(env->target_ray, env->bounding_boxes[env->selected_box]);
    if (colCollision.hit && colCollision.distance < closestDistance) {
        closestDistance = colCollision.distance;
        hitType = 1;
    }
    env->time_since_score++;
    if (hitType == 1) {
        env->score_tick++;
        env->score += 1.0f;
        env->rewards[0] = 1.0f;
    }
    if(env->score_tick >= 16){
        env->score_tick = 0;
        env->time_since_score = 0;
        Respawn(env);
    }
    if(env->score >= 128){ // WIN
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    } else if(env->time_since_score >= 128) { // LOSE
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
    env->last_hit = hitType;
}
