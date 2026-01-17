# Dogfight Rendering Guide

Reference patterns extracted from PufferLib ocean environments for Phase 4 implementation.

## Current State
- `dogfight.h` lines 375-406: basic `c_render()` skeleton with placeholder camera
- Need: chase camera, plane drawing, ground, bounds, HUD

## Client Struct

Update the existing Client struct (~line 104) to support camera controls:

```c
typedef struct Client {
    Camera3D camera;
    float width;
    float height;
    // Camera orbit state (for mouse control)
    float cam_distance;
    float cam_azimuth;
    float cam_elevation;
    bool is_dragging;
    float last_mouse_x, last_mouse_y;
} Client;
```

## Chase Camera

Calculate camera position behind and above player using quaternion orientation:

```c
// Get player forward vector from quaternion
Vec3 fwd = quat_rotate(player->ori, vec3(1, 0, 0));

// Camera position: behind and above player
float dist = 80.0f, height = 30.0f;
Vector3 cam_pos = {
    player->pos.x - fwd.x * dist,
    player->pos.y - fwd.y * dist,
    player->pos.z + height
};

// Look at player
Vector3 cam_target = {player->pos.x, player->pos.y, player->pos.z};
```

## Raylib Quick Reference

| Task | Code |
|------|------|
| Init window | `InitWindow(1280, 720, "Dogfight"); SetTargetFPS(60);` |
| Camera setup | `camera.up = (Vector3){0, 0, 1}; camera.fovy = 45; camera.projection = CAMERA_PERSPECTIVE;` |
| Draw sphere | `DrawSphere((Vector3){x, y, z}, radius, color);` |
| Draw line | `DrawLine3D(start, end, color);` |
| Draw ground | `DrawPlane((Vector3){0, 0, 0}, (Vector2){4000, 4000}, DARKGREEN);` |
| Draw bounds | `DrawCubeWires((Vector3){0, 0, 1500}, 4000, 4000, 3000, WHITE);` |
| HUD text | `DrawText(TextFormat("Speed: %.0f", speed), 10, 10, 20, WHITE);` |

## Mouse Orbit Controls

Pattern from `drone_race.h` - allows user to orbit camera with mouse drag:

```c
void handle_camera_controls(Client *c) {
    Vector2 mouse = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_LEFT)) {
        c->is_dragging = true;
        c->last_mouse_x = mouse.x;
        c->last_mouse_y = mouse.y;
    }
    if (IsMouseButtonReleased(MOUSE_LEFT)) {
        c->is_dragging = false;
    }

    if (c->is_dragging) {
        float sensitivity = 0.005f;
        c->cam_azimuth -= (mouse.x - c->last_mouse_x) * sensitivity;
        c->cam_elevation += (mouse.y - c->last_mouse_y) * sensitivity;
        c->cam_elevation = clampf(c->cam_elevation, -1.4f, 1.4f);  // prevent gimbal lock
        c->last_mouse_x = mouse.x;
        c->last_mouse_y = mouse.y;
    }

    // Mouse wheel zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        c->cam_distance = clampf(c->cam_distance - wheel * 5.0f, 30.0f, 200.0f);
    }
}
```

## Complete c_render() Template

```c
void c_render(Dogfight *env) {
    // 1. Lazy init
    if (env->client == NULL) {
        env->client = (Client *)calloc(1, sizeof(Client));
        env->client->width = 1280;
        env->client->height = 720;
        env->client->cam_distance = 80.0f;
        env->client->cam_azimuth = 0.0f;
        env->client->cam_elevation = 0.3f;

        InitWindow(1280, 720, "Dogfight");
        SetTargetFPS(60);

        env->client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        env->client->camera.fovy = 45.0f;
        env->client->camera.projection = CAMERA_PERSPECTIVE;
    }

    // 2. Handle window close
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    // 3. Update chase camera
    Plane *p = &env->player;
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    float dist = env->client->cam_distance;

    env->client->camera.position = (Vector3){
        p->pos.x - fwd.x * dist,
        p->pos.y - fwd.y * dist,
        p->pos.z + dist * 0.4f
    };
    env->client->camera.target = (Vector3){p->pos.x, p->pos.y, p->pos.z};

    // 4. Optional: handle mouse orbit
    // handle_camera_controls(env->client);

    // 5. Draw
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    BeginMode3D(env->client->camera);

    // Ground plane at z=0
    DrawPlane((Vector3){0, 0, 0}, (Vector2){4000, 4000}, (Color){20, 60, 20, 255});

    // World bounds wireframe
    DrawCubeWires((Vector3){0, 0, 1500}, 4000, 4000, 3000, (Color){100, 100, 100, 255});

    // Player plane (green)
    Vector3 player_pos = {p->pos.x, p->pos.y, p->pos.z};
    DrawSphere(player_pos, 5.0f, GREEN);
    // Forward direction indicator
    Vector3 player_fwd = {p->pos.x + fwd.x * 30, p->pos.y + fwd.y * 30, p->pos.z + fwd.z * 30};
    DrawLine3D(player_pos, player_fwd, GREEN);

    // Opponent plane (red)
    Plane *o = &env->opponent;
    Vector3 opp_pos = {o->pos.x, o->pos.y, o->pos.z};
    DrawSphere(opp_pos, 5.0f, RED);
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vector3 opp_fwd_end = {o->pos.x + opp_fwd.x * 30, o->pos.y + opp_fwd.y * 30, o->pos.z + opp_fwd.z * 30};
    DrawLine3D(opp_pos, opp_fwd_end, RED);

    EndMode3D();

    // HUD
    float speed = norm3(p->vel);
    float dist_to_opp = norm3(sub3(o->pos, p->pos));

    DrawText(TextFormat("Speed: %.0f m/s", speed), 10, 10, 20, WHITE);
    DrawText(TextFormat("Alt: %.0f m", p->pos.z), 10, 40, 20, WHITE);
    DrawText(TextFormat("Throttle: %.0f%%", p->throttle * 100.0f), 10, 70, 20, WHITE);
    DrawText(TextFormat("Distance: %.0f m", dist_to_opp), 10, 100, 20, WHITE);
    DrawText(TextFormat("Tick: %d", env->tick), 10, 130, 20, WHITE);
    DrawText(TextFormat("Return: %.2f", env->episode_return), 10, 160, 20, WHITE);

    // Camera controls hint
    DrawText("ESC: Exit", 10, env->client->height - 30, 16, GRAY);

    EndDrawing();
}
```

## Coordinate System

- Dogfight uses: **X=forward, Y=right, Z=up**
- Set `camera.up = {0, 0, 1}` to match
- World bounds: ±2000 X/Y, 0-3000 Z (from `dogfight.h` defines)

## Reference Environments

| File | Key Patterns |
|------|--------------|
| `drone_race/drone_race.h` | Spherical orbit camera, mouse controls, trail effects |
| `drive/drive.h` | FPV chase camera using heading angle |
| `battle/battle.h` | Quaternion orientation, `CAMERA_THIRD_PERSON`, 3D models |
| `impulse_wars/render.h` | Smooth camera lerp, bloom effects, sophisticated UI |

## Build & Test

```bash
# Build
python setup.py build_ext --inplace --force

# Run with rendering
python -m pufferlib.pufferl train puffer_dogfight --render

# Run tests
gcc -I raylib-5.5_linux_amd64/include -o pufferlib/ocean/dogfight/dogfight_test pufferlib/ocean/dogfight/dogfight_test.c raylib-5.5_linux_amd64/lib/libraylib.a -lm -lpthread -ldl && ./pufferlib/ocean/dogfight/dogfight_test
```
