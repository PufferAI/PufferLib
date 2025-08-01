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

const float RAY_LENGTH    = 100.0f;
const Color RAY_COLOR     = GREEN;

const int HEIGHT = 450;
const int WIDTH  = 800;

// Required struct. Only use floats!
typedef struct {
    // float episode_return;
    float episode_length;
    float score;
    float n;
} Log;

typedef struct {
    Log log;
    uint8_t* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    float score;
    int tick;
    int shoot_delta;

    Camera camera;

    Ray ray;
    float collision_distance;
    RenderTexture2D rt;

    BoundingBox bounding_box;
    Vector3 square_positions[6];
    Color square_colors[6];

    int score_tick;
    int time_since_score;
    int swap_timer;
    int selected_box;
    int selected_color_name;
    int blue_pos;
} VisionTest;

const Color BASE_COLORS[6] = { RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE };
const char *COLOR_NAMES[6] = { "RED", "BLUE", "GREEN", "YELLOW", "PURPLE", "ORANGE" };

void add_log(VisionTest* env) {
    env->log.episode_length += env->tick;
    env->log.score += env->score; 
    env->log.n++;
}

void Respawn(VisionTest* env) {
    float box_size = 20.0f;
    env->blue_pos = 1;

    Vector3 base_positions[6] = {
        (Vector3){ 0.0f, 0.0f, -2*box_size },
        (Vector3){ 0.0f, 0.0f, 2*box_size },
        (Vector3){ -2*box_size, 0.0f, 0.0f },
        (Vector3){ 2*box_size, 0.0f, 0.0f },
        (Vector3){ 0.0f, 2*box_size, 0.0f }, 
        (Vector3){ 0.0f, -2*box_size, 0.0f }
    };

    int color_indices[6] = { 0, 1, 2, 3, 4, 5 };

    for (int i = 5; i > 0; i--) { // Fisher-Yates shuffle for color indices
        int j = GetRandomValue(0, i);
        int temp = color_indices[i];
        color_indices[i] = color_indices[j];
        color_indices[j] = temp;
    }

    // allows for randomization of the color needs an added obs
    // env->selected_box = GetRandomValue(0, 5);
    // env->selected_color_name = color_indices[env->selected_box];
    // env->observations[64*3] = floor(256 * env->selected_color_name)/6 ;

    bool should_set = true;
    for (int i = 0; i < 6; i++) {
        env->square_positions[i] = base_positions[i];
        env->square_colors[i] = BASE_COLORS[color_indices[i]];
        if(color_indices[i] == env->blue_pos && should_set){
            env->blue_pos = i;
            should_set = false;
        }
    }

    env->bounding_box.min = (Vector3){
        env->square_positions[env->blue_pos].x - box_size/2,
        env->square_positions[env->blue_pos].y - box_size/2,
        env->square_positions[env->blue_pos].z - box_size/2
    };
    env->bounding_box.max = (Vector3){
        env->square_positions[env->blue_pos].x + box_size/2,
        env->square_positions[env->blue_pos].y + box_size/2,
        env->square_positions[env->blue_pos].z + box_size/2
    };
}

void c_reset(VisionTest* env) {
    memset(env->observations, 0, sizeof(uint8_t)*(64*3));

    Camera camera = env->camera;
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 60.0f;
    camera.position = (Vector3){ 0.0f, 2.0f, 4.0f };
    camera.target = (Vector3){ 0.0f, 2.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 60.0f;
    camera.projection = CAMERA_CUSTOM;
    env->camera = camera;


    env->rewards[0] = 0;
    env->tick = 0;
    env->score = 0;
    env->time_since_score = 0;
    env->swap_timer = 0;
    env->score_tick = 0;

    Respawn(env);
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(VisionTest* env) {
    UnloadRenderTexture(env->rt);
    memset(&env->rt, 0, sizeof(RenderTexture2D));
    if(IsWindowReady()){
        CloseWindow();
    }
}

void c_step(VisionTest* env) {
    int action = env->actions[0];
    env->tick++;
    env->rewards[0] = 0;
    env->terminals[0] = 0;

    Camera camera = env->camera;
    float rotationSpeed = 0.8;

    if (action == S_DOWN)  CameraPitch(&camera, -rotationSpeed, true, false, true);
    if (action == S_UP)    CameraPitch(&camera, rotationSpeed, true, false, true);
    if (action == S_RIGHT) CameraYaw(&camera, -rotationSpeed, false);
    if (action == S_LEFT)  CameraYaw(&camera, rotationSpeed, false);

    env->ray.position = camera.position;
    env->ray.direction = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    env->camera = camera;

    float closestDistance = RAY_LENGTH;
    int hitType = 0;

    env->collision_distance = (hitType > 0) ? closestDistance : -1.0f;

    RayCollision colCollision = GetRayCollisionBox(env->ray, env->bounding_box);
    if (colCollision.hit && colCollision.distance < closestDistance) {
        closestDistance = colCollision.distance;
        hitType = 1;
    }

    env->time_since_score++;
    if (hitType == 1) {
        env->score_tick++;
        env->score += 1.0f;
        env->rewards[0] = 1.0f;
    }else{
        env->score_tick = 0;
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
    }else if(env->time_since_score >= 128){ // LOSE
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
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
        SetTargetFPS(60);
        Respawn(env);
        env->rt = LoadRenderTexture(WIDTH, HEIGHT);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    Camera camera = env->camera;
    float box_size = 10.0f; // Size of each square side
    BeginTextureMode(env->rt);
        ClearBackground(BLACK);
        BeginMode3D(camera);

            for (int i = 0; i < 6; i++) {
                DrawCube(env->square_positions[i], 1.5*box_size, 1.5*box_size, 1.5*box_size, env->square_colors[i]);
                DrawCubeWires(env->square_positions[i], 1.5*box_size, 1.5*box_size, 1.5*box_size, WHITE);
            }

            Vector3 rayEnd = Vector3Add(camera.position, Vector3Scale(env->ray.direction, env->collision_distance > 0 ? env->collision_distance : RAY_LENGTH));
            DrawCylinderWiresEx(camera.position, rayEnd, 0.1f, 0.1f, 4, GREEN);
        EndMode3D();

        // Draw 2D UI overlays here (on top of the 3D render) well use this later for vision purposes
        DrawRectangle(600, 5, 190, 40, Fade(SKYBLUE, 0.5f));
        DrawRectangleLines(600, 5, 190, 40, BLUE);
        DrawText(TextFormat("- Score: %f", env->score), 610, 10, 10, BLACK);
        DrawText(TextFormat("- Find Color: %s", COLOR_NAMES[1]), 610, 25, 10, BLACK);
    EndTextureMode();

    env->camera = camera;

    BeginDrawing();
        DrawTextureRec(env->rt.texture, (Rectangle){ 0, 0, (float)env->rt.texture.width, (float)-env->rt.texture.height }, (Vector2){ 0, 0 }, WHITE);
        Image image = LoadImageFromTexture(env->rt.texture);  // Captures the full rendered frame (scene + UI)
        ImageResize(&image, 8, 8);
        ImageFlipVertical(&image);

        Color *pixels = LoadImageColors(image);

        int debugX = 10;
        int debugY = 120;
        int scale = 8; // Magnification (16=128x128 grid; adjust as needed)

        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                int index = y * 8 + x;
                Color c = pixels[index];
                DrawRectangle(debugX + x * scale, debugY + y * scale, scale, scale, c);
                env->observations[index] = c.r;
                env->observations[8*8+index] = c.g;
                env->observations[8*8*2+index] = c.b;
            }
        }
        DrawRectangleLines(debugX, debugY, 8 * scale, 8 * scale, RED);
        DrawText("Puffer POV:", debugX, debugY - 15, 10, WHITE);
        UnloadImageColors(pixels);
        UnloadImage(image);
    EndDrawing();
}
