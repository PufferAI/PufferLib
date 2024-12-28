#pragma once
#include "raylib.h"
#include "codeball.h"
#include "raymath.h"
#include "rlgl.h"

#include "base_vs.h"
#include "fragment_fs.h"

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Color robot_color[2];
    Color ball_color;
    Color nitro_color;
    Camera3D camera;
    Shader shader;
    Model sphere;
};

Client* make_client() {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 800;
    client->height = 600;
    client->robot_color[0] = RED;
    client->robot_color[1] = BLUE;
    client->ball_color = WHITE;
    client->nitro_color = GREEN;
    client->camera =
        (Camera3D){(Vector3){0.0f, 60.0f, -90.0f}, (Vector3){0.0f, 0.0f, 0.0f},
                   (Vector3){0.0f, 1.0f, 0.0f}, 45.0f, 0};

    InitWindow(client->width, client->height, "CodeBall");

    // client->shader = LoadShader("base.vs", "fragment.fs");
    char *vs_code = malloc(base_vs_len + 1);
    memcpy(vs_code, base_vs, base_vs_len);
    vs_code[base_vs_len] = '\0';
    char *fs_code = malloc(fragment_fs_len + 1);
    memcpy(fs_code, fragment_fs, fragment_fs_len);
    fs_code[fragment_fs_len] = '\0';
    client->shader = LoadShaderFromMemory(vs_code, fs_code);
    free(vs_code);
    free(fs_code);

    Mesh sphere = GenMeshSphere(1.0f, 32, 32);
    for (int v = 0; v < sphere.vertexCount; v++) {
        sphere.normals[v * 3] = v % 2 == 0 ? 1.0f : -1.0f;
        sphere.normals[v * 3 + 1] = v % 3 == 0 ? 1.0f : -1.0f;
        sphere.normals[v * 3 + 2] = 0.95f;
    }
    client->sphere = LoadModelFromMesh(sphere);
    client->sphere.materials[0].shader = client->shader;

    SetTargetFPS(TICKS_PER_SECOND);

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

Vector3 cvt(Vec3D vec) {
    return (Vector3){vec.x, vec.y, vec.z};
}

// TODO instancing...
void MyDrawSphere(Client *client, Vector3 center, float radius,
               Color color) {
                DrawModel(client->sphere, center, radius, color);
                // DrawSphere(center, radius, color);
}

// Custom DrawPlane function for arbitrary orientation
void MyDrawPlane(Vector3 center, Vector3 normal, float width, float height,
               Color color) {
    Vector3 u, v;

    if (fabsf(normal.y) < 0.999f) {
        u = (Vector3){normal.z, 0, -normal.x};
    } else {
        u = (Vector3){1, 0, 0};
    }
    v = Vector3CrossProduct(normal, u);

    u = Vector3Normalize(u);
    v = Vector3Normalize(v);

    Vector3 p1 =
        Vector3Subtract(Vector3Subtract(center, Vector3Scale(u, width / 2)),
                        Vector3Scale(v, height / 2));
    Vector3 p2 = Vector3Add(Vector3Subtract(center, Vector3Scale(u, width / 2)),
                            Vector3Scale(v, height / 2));
    Vector3 p3 = Vector3Add(Vector3Add(center, Vector3Scale(u, width / 2)),
                            Vector3Scale(v, height / 2));
    Vector3 p4 = Vector3Subtract(Vector3Add(center, Vector3Scale(u, width / 2)),
                                 Vector3Scale(v, height / 2));

    DrawTriangleStrip3D((Vector3[]){p1, p2, p4, p3}, 4, color);
}

void render(Client* client, CodeBall* env) {
    Entity* robots = env->robots;
    Entity ball = env->ball;
    NitroPack* nitro_packs = env->nitro_packs;

    BeginDrawing();

    ClearBackground(DARKGRAY);
    BeginMode3D(client->camera);  // Begin 3D mode

    BeginShaderMode(client->shader);

    // Arena dimensions
    Vector3 arena_size = {arena.width, arena.height,
                          arena.depth};

    // Draw arena (box)
    // DrawCube((Vector3){0, arena.height / 2, 0}, arena_size.x, arena_size.y,
    //          arena_size.z, LIGHTGRAY);
    // DrawCube((Vector3){0, -arena.height / 2, 0}, arena_size.x, arena_size.y,
    //          arena_size.z, LIGHTGRAY);

    MyDrawPlane((Vector3){0, 0, 0}, (Vector3){0, -1, 0}, arena_size.x,
              arena_size.z, LIGHTGRAY);  // Bottom (CORRECTED NORMAL)
    MyDrawPlane((Vector3){arena_size.x / 2, arena_size.y / 2, 0},
              (Vector3){1, 0, 0}, arena_size.z, arena_size.y,
              LIGHTGRAY);  // Right
    MyDrawPlane((Vector3){-arena_size.x / 2, arena_size.y / 2, 0},
              (Vector3){-1, 0, 0}, arena_size.z, arena_size.y,
              LIGHTGRAY);  // Left
    
    for (int fb = 0; fb < 2; fb++) {
        int sign = fb == 0 ? 1 : -1;
        Vector3 normal = {0, 0, sign};
        MyDrawPlane(
            (Vector3){0, arena.goal_height + (arena_size.y - arena.goal_height) / 2,
                    arena_size.z / 2 * sign},
            normal, arena.goal_width,
            (arena_size.y - arena.goal_height),
            LIGHTGRAY);  // Back middle
        double remaining = (arena_size.x - arena.goal_width) / 2;
        MyDrawPlane((Vector3){-arena.goal_width / 2 - remaining / 2, arena_size.y / 2,
                            arena_size.z / 2 * sign},
                    normal, remaining, arena_size.y,
                    LIGHTGRAY);  // Back right
        MyDrawPlane((Vector3){arena.goal_width / 2 + remaining / 2,
                              arena_size.y / 2, arena_size.z / 2 * sign},
                    normal, remaining, arena_size.y,
                    LIGHTGRAY);  // Back left
    }

    // Draw Goals (as boxes)
    Vector3 goal_size = {arena.goal_width, arena.goal_height, arena.goal_depth};

    Color goal_colors[2] = {BLUE, RED};
    for (int gi = 0; gi < 2; gi++) {
        Color goal_color = goal_colors[gi];
        int sign = gi == 0 ? 1 : -1;
        MyDrawPlane(
            (Vector3){0, goal_size.y / 2, sign * (arena.depth / 2 + arena.goal_depth)},
            (Vector3){0, 0, sign}, goal_size.x, goal_size.y, goal_color);  // Back
        MyDrawPlane((Vector3){-goal_size.x / 2, goal_size.y / 2,
                              sign * (arena.depth / 2 + goal_size.z / 2)},
                    (Vector3){-1, 0, 0}, goal_size.z, goal_size.y,
                    goal_color);  // Left
        MyDrawPlane((Vector3){goal_size.x / 2, goal_size.y / 2,
                              sign * (arena.depth / 2 + goal_size.z / 2)},
                    (Vector3){1, 0, 0}, goal_size.z, goal_size.y,
                    goal_color);  // Right
        MyDrawPlane(
            (Vector3){0, goal_size.y, sign * (arena.depth / 2 + goal_size.z / 2)},
            (Vector3){0, 1, 0}, goal_size.x, goal_size.z, goal_color);  // Top

        MyDrawPlane((Vector3){0, 0, sign * (arena.depth / 2 + goal_size.z / 2)},
                    (Vector3){0, -1, 0}, goal_size.x, goal_size.z, goal_color);  // Bottom
    }

    // Draw robots (colored by side)
    for (int i = 0; i < env->n_robots; i++) {
        Color robot_color =
            robots[i].side ? client->robot_color[1] : client->robot_color[0];
        // DrawCube(cvt(robots[i].position), robots[i].radius * 2, robots[i].radius * 2,
        //          robots[i].radius * 2, robot_color);
        MyDrawSphere(client, cvt(robots[i].position), robots[i].radius, robot_color);
    }

    // Draw ball
    MyDrawSphere(client, cvt(ball.position), ball.radius, client->ball_color);

    // Draw nitro packs
    for (int i = 0; i < env->n_nitros; i++) {
        if (nitro_packs[i].alive) {
            MyDrawSphere(client, cvt(nitro_packs[i].position), NITRO_PACK_RADIUS * 2,
                     client->nitro_color);
        }
    }

    EndShaderMode();

    EndMode3D();

    DrawFPS(10, 10);
    
    char rew_str[100];
    snprintf(rew_str, sizeof(rew_str), "Rewards: %f %f", env->rewards[0],
             env->rewards[1]);
    DrawText(rew_str, 100, 10, 20, RED);

    EndDrawing();

    // Camera controls
    Vector3 addition = {0, 0, 0};
    Vector3 camera_forward = Vector3Normalize(
        Vector3Subtract(client->camera.target, client->camera.position));
    Vector3 camera_right = Vector3Normalize(
        Vector3CrossProduct(camera_forward, (Vector3){0, 1, 0}));
    Vector3 camera_up =
        Vector3Normalize(Vector3CrossProduct(camera_right, camera_forward));

    if (IsKeyDown(KEY_W)) addition = Vector3Add(addition, camera_forward);
    if (IsKeyDown(KEY_S)) addition = Vector3Subtract(addition, camera_forward);
    if (IsKeyDown(KEY_D)) addition = Vector3Add(addition, camera_right);
    if (IsKeyDown(KEY_A)) addition = Vector3Subtract(addition, camera_right);
    if (IsKeyDown(KEY_E)) addition = Vector3Add(addition, camera_up);
    if (IsKeyDown(KEY_Q)) addition = Vector3Subtract(addition, camera_up);
    addition = Vector3Scale(Vector3Normalize(addition), 0.5f);
    client->camera.position = Vector3Add(client->camera.position, addition);
}