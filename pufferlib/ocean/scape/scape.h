/* Scape: a sample multiagent env about puffers eating stars.
 * Use this as a tutorial and template for your own multiagent envs.
 * We suggest starting with the Squared env for a simpler intro.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "raylib.h"
#include "rlgl.h"

static float offset_x = -6.5f;
static float offset_z = 12.5f;

static Vector3 scene_pos = (Vector3){-6.5, -2.5, 60.5};

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
    Texture2D puffer;
    Camera3D camera;
    Model scene;
    Model player;
} Client;

typedef struct {
    float x;
    float y;
    float dest_x;
    float dest_y;
    int size;
    int height;
    Color color;
} Entity;

typedef struct {
    float x;
    float y;
} Goal;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Client* client;
    Entity* entities;
    Goal* goals;
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    double* actions; // Required. double* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    float* terminals; // Required. We don't yet have truncations as standard yet
    int width;
    int height;
    int num_agents;
    int num_npcs;
} Scape;

Vector3 to_world(Vector2 pos) {
    return (Vector3){
        .x = pos.x - 6.5f,
        .y = -2.5f,
        .z = pos.y + 12.5f
    };
}

Vector2 from_world(Vector3 pos) {
    return (Vector2){
        .x = pos.x + 6.5f,
        .y = pos.z - 12.5f
    };
}

float clampf(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

void move_agent(Scape* env) {
    float dx = env->entities[0].dest_x - env->entities[0].x;
    float dy = env->entities[0].dest_y - env->entities[0].y;
    env->entities[0].x += clampf(dx, -2, 2);
    env->entities[0].y += clampf(dy, -2, 2);
}

bool overlaps(Entity* a, Entity* b) {
  return a->x < b->x + b->size && b->x < a->x + a->size
    && a->y > b->y - b->size && b->y > a->y - a->size;
}

void move_npc(Scape* env, int idx) {
    Entity* player = &env->entities[0];
    Entity* npc = &env->entities[idx];
    float x = npc->x;
    float y = npc->y;

    bool collide_x = false;
    bool collide_y = false;

    int dst_x = npc->x - clampf(x - npc->dest_x, -1, 1);
    int dst_y = npc->y - clampf(y - npc->dest_y, -1, 1);

    npc->x = dst_x;
    npc->y = dst_y;
    if (overlaps(npc, player)) {
        collide_y = true;
    }
    npc->x = x;
    npc->y = y;

    npc->y = dst_y;
    for (int i=1; i<env->num_npcs + env->num_agents; i++) {
        Entity* other = &env->entities[i];
        if (npc == other) {
            continue;
        }
        if (overlaps(npc, other)) {
            env->entities[idx].y = y;
            collide_y = true;
            break;
        }
    }
    npc->y = y;

    npc->x = dst_x;
    for (int i=1; i<env->num_npcs + env->num_agents; i++) {
        Entity* other = &env->entities[i];
        if (npc == other) {
            continue;
        }
        if (overlaps(npc, other)) {
            env->entities[idx].x = x;
            collide_x = true;
            break;
        }
    }
    npc->x = x;

    if (!collide_x) {
        npc->x = dst_x;
    }
    if (!collide_y) {
        npc->y = dst_y;
    }
}

/* Recommended to have an init function of some kind if you allocate 
 * extra memory. This should be freed by c_close. Don't forget to call
 * this in binding.c!
 */
void init(Scape* env) {
    env->num_agents = 1;
    env->num_npcs = 6;
    env->entities = calloc(env->num_agents + env->num_npcs, sizeof(Entity));
}

/* Recommended to have an observation function of some kind because
 * you need to compute agent observations in both reset and in step.
 * If using float obs, try to normalize to roughly -1 to 1 by dividing
 * by an appropriate constant.
 */
void compute_observations(Scape* env) {
}

// Required function
void c_reset(Scape* env) {
    env->entities[0].x = 28;
    env->entities[0].dest_x = 28;
    env->entities[0].y = 17;
    env->entities[0].dest_y = 17;
    env->entities[0].size = 1;
    env->entities[0].height = 2;
    env->entities[0].color = WHITE;

    // South pillar
    env->entities[1].x = 21;
    env->entities[1].dest_x = 21;
    env->entities[1].y = 37;
    env->entities[1].dest_y = 37;
    env->entities[1].size = 3;
    env->entities[1].height = 3;
    env->entities[1].color = YELLOW;

    // West pillar
    env->entities[2].x = 11;
    env->entities[2].dest_x = 11;
    env->entities[2].y = 23;
    env->entities[2].dest_y = 23;
    env->entities[2].size = 3;
    env->entities[2].height = 3;
    env->entities[2].color = YELLOW;

    // North pillar
    env->entities[3].x = 28;
    env->entities[3].dest_x = 28;
    env->entities[3].y = 21;
    env->entities[3].dest_y = 21;
    env->entities[3].size = 3;
    env->entities[3].height = 3;
    env->entities[3].color = YELLOW;

    // Test enemy
    env->entities[4].x = 12;
    env->entities[4].y = 19;
    env->entities[4].size = 4;
    env->entities[4].height = 2;
    env->entities[4].color = RED;

    env->entities[5].x = 26;
    env->entities[5].y = 42;
    env->entities[5].size = 4;
    env->entities[5].height = 2;
    env->entities[5].color = BLUE;

    env->entities[6].x = 14;
    env->entities[6].y = 25;
    env->entities[6].size = 3;
    env->entities[6].height = 2;
    env->entities[6].color = GREEN;

    compute_observations(env);
}

// Required function
void c_step(Scape* env) {
    move_agent(env);

    // Set npc target to player pos
    for (int i=4; i<env->num_agents + env->num_npcs; i++) {
        Entity* npc = &env->entities[i];
        npc->dest_x = env->entities[0].x;
        npc->dest_y = env->entities[0].y;
    }

    for (int i=env->num_agents; i<env->num_agents + env->num_npcs; i++) {
        move_npc(env, i);
    }
    compute_observations(env);
}

void handle_camera_controls(Client* client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_MIDDLE)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_MIDDLE)) {
        is_dragging = false;
    }

    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {
            -(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
            (current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed
        };

        // Apply 45-degree rotation to the movement
        // For a -45 degree rotation (clockwise)
        float cos45 = -0.7071f;  // cos(-45°)
        float sin45 = 0.7071f; // sin(-45°)
        Vector2 rotated_delta = {
            delta.x * cos45 - delta.y * sin45,
            delta.x * sin45 + delta.y * cos45
        };

        // Update camera position (only X and Y)
        client->camera.position.z += rotated_delta.x;
        client->camera.position.x += rotated_delta.y;

        // Update camera target (only X and Y)
        client->camera.target.z += rotated_delta.x;
        client->camera.target.x += rotated_delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {
            client->camera.position.x - client->camera.target.x,
            client->camera.position.y - client->camera.target.y,
            client->camera.position.z - client->camera.target.z
        };

        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

Vector2 get_hovered_tile(Camera3D* camera) {
    Ray ray = GetMouseRay(GetMousePosition(), *camera);
    float plane_y = -0.5f;
    float t = (plane_y - ray.position.y) / ray.direction.y;
    Vector3 floor_pos = {
        .x = ray.position.x + ray.direction.x*t,
        .y = plane_y,
        .z = ray.position.z + ray.direction.z*t
    };
    return (Vector2){
        .x = floorf(floor_pos.x + 0.5f),
        .y = floorf(floor_pos.z + 0.5f)
    };
}

Vector3 tile_to_mesh_pos(float x, float y) {
    return (Vector3){
        .x = x,
        .y = -0.49f,
        .z = y
    };
}

void draw_tile(Client* client, Vector2 hovered_tile) {
    Vector3 corner = tile_to_mesh_pos(hovered_tile.x, hovered_tile.y);

    Vector3 w00 = (Vector3){corner.x - 0.5, corner.y, corner.z + 0.5};
    Vector3 w01 = (Vector3){corner.x + 0.5, corner.y, corner.z + 0.5};
    Vector3 w10 = (Vector3){corner.x + 0.5, corner.y, corner.z - 0.5};
    Vector3 w11 = (Vector3){corner.x - 0.5, corner.y, corner.z - 0.5};

    Vector2 s00 = GetWorldToScreen(w00, client->camera);
    Vector2 s01 = GetWorldToScreen(w01, client->camera);
    Vector2 s10 = GetWorldToScreen(w10, client->camera);
    Vector2 s11 = GetWorldToScreen(w11, client->camera);

    DrawLineEx(s00, s01, 2.0f, RED);
    DrawLineEx(s00, s11, 2.0f, RED);
    DrawLineEx(s10, s11, 2.0f, RED);
    DrawLineEx(s10, s01, 2.0f, RED);
}


// Required function. Should handle creating the client on first call
void c_render(Scape* env) {
    Client* client = env->client;
    if (client == NULL) {
        InitWindow(env->width, env->height, "PufferLib Scape");
        SetTargetFPS(60);

        client = (Client*)calloc(1, sizeof(Client));
        env->client = client;

        // Don't do this before calling InitWindow
        client->puffer = LoadTexture("resources/shared/puffers_128.png");

        Camera3D camera = { 0 };
        camera.position = (Vector3){ 28.0f, 10.0f, 12.0f };
        camera.target = (Vector3){ 28.0f, -0.5, 17.0f };
        //camera.position = (Vector3){ 25.0f, 20.0f, 0.0f };
        //camera.target = (Vector3){ 25.0f, 0.0, 28.0f };
        camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
        camera.fovy = 70.0f;
        camera.projection = CAMERA_PERSPECTIVE;
        client->camera = camera;
        client->scene = LoadModel("resources/scape/inferno_compress.glb");
        client->player = LoadModel("resources/scape/player_twisted_bow_decompressed.glb");
    }

    handle_camera_controls(client);

    if (IsKeyDown(KEY_RIGHT)) offset_x += 0.1f;
    if (IsKeyDown(KEY_LEFT)) offset_x -= 0.1f;
    if (IsKeyDown(KEY_UP)) offset_z -= 0.1f;
    if (IsKeyDown(KEY_DOWN)) offset_z += 0.1f;

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    BeginMode3D(client->camera);
    Vector2 origin = (Vector2){0.0f, 57.0f};
    //DrawModel(env->client->scene, to_world(origin), 1.0f, WHITE);
    if (IsKeyPressed(KEY_W)) scene_pos.x += 1.0f;
    if (IsKeyPressed(KEY_S)) scene_pos.x -= 1.0f;
    if (IsKeyPressed(KEY_A)) scene_pos.z -= 1.0f;
    if (IsKeyPressed(KEY_D)) scene_pos.z += 1.0f;

    DrawModel(env->client->scene, (Vector3)scene_pos, 1.0f, WHITE);
    /*
    rlDisableBackfaceCulling();
    DrawModelEx(env->client->scene,
        (Vector3){ -6.0f, -2.5f, 12.0f },
        (Vector3){ 1.0f, 0.0f, 1.0f },
        0.0f,
        (Vector3){ 1.0f, 1.0f, -1.0f },
        WHITE);
    rlEnableBackfaceCulling();
    */
    //DrawModel(env->client->player, (Vector3){ 32.0f, 10.0f, -26.0f }, 1.0f, WHITE);
    

    //Vector2 spawn = (Vector2){28.0f, 17.0f};
    //Vector3 player_pos = to_world(spawn);
    Vector3 player_pos = (Vector3){env->entities[0].x, 0, env->entities[0].y};
    DrawCube(player_pos, 1.0f, 1.0f, 1.0f, RED);

    for (int i=0; i<env->num_agents + env->num_npcs; i++) {
        Entity* entity = &env->entities[i];
        float entity_x = entity->x + entity->size/2.0f - 0.5f;
        float entity_y = entity->y - entity->size/2.0f + 0.5f;
        Vector3 entity_pos = (Vector3){entity_x, 0, entity_y};
        DrawCube(entity_pos, entity->size, 2*entity->height, entity->size, entity->color);
    }

    // Draw axes
    DrawLine3D((Vector3){0, 0, 0}, (Vector3){50, 0, 0}, RED);
    DrawLine3D((Vector3){0, 0, 0}, (Vector3){0, 0, 50}, BLUE);
    DrawLine3D((Vector3){0, 0, 0}, (Vector3){0, 50, 0}, GREEN);

    EndMode3D();

    Vector2 hovered_tile = get_hovered_tile(&client->camera);
    draw_tile(client, hovered_tile);

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        env->entities[0].dest_x = hovered_tile.x;
        env->entities[0].dest_y = hovered_tile.y;
    }

    DrawText(TextFormat("scene_pos.x: %f scene_pos.z: %f", scene_pos.x, scene_pos.z), 10, 30, 20, RAYWHITE);
    DrawText(TextFormat("offset_x: %f offset_z: %f", offset_x, offset_z), 10, 10, 20, RAYWHITE);

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Scape* env) {
    free(env->entities);
    if (env->client != NULL) {
        Client* client = env->client;
        UnloadTexture(client->puffer);
        CloseWindow();
        free(client);
    }
}
