#ifndef ORBITAL_DOCK_RENDER_H
#define ORBITAL_DOCK_RENDER_H

#include "raymath.h"

// ============================================================================
// Render Client Struct
// ============================================================================

#define RENDER_WIDTH 1080
#define RENDER_HEIGHT 720
#define TRAIL_LENGTH 256
#define NUM_STARS 300

typedef struct {
    double x, y, z;
} Vec3d;

typedef struct {
    Vec3d pos[TRAIL_LENGTH];
    int index;
    int count;
} Trail;

struct Client {
    Camera3D camera;
    float width;
    float height;
    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;
    Trail trail;
    float scale;
    int last_step;
    double orbit_angle;
    double earth_angle;
    Vec3d stars[NUM_STARS];
    Model earth_model;
    Texture2D earth_texture;
};

// Additional colors
static const Color PUFF_GREEN = {0, 187, 0, 255};
static const Color PUFF_MAGENTA = {187, 0, 187, 255};

static inline Vec3d vec3d(double x, double y, double z) {
    Vec3d v = {x, y, z};
    return v;
}

static inline float clampf_render(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void update_camera_position(Client *c) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

    float x = r * cosf(el) * cosf(az);
    float y = r * cosf(el) * sinf(az);
    float z = r * sinf(el);

    c->camera.position = (Vector3){x, y, z};
    c->camera.target = (Vector3){0, 0, 0};
}

static void handle_camera_controls(Client *client) {
    Vector2 mouse_pos = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = true;
        client->last_mouse_pos = mouse_pos;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = false;
    }

    if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_delta = {
            mouse_pos.x - client->last_mouse_pos.x,
            mouse_pos.y - client->last_mouse_pos.y
        };

        float sensitivity = 0.005f;
        client->camera_azimuth -= mouse_delta.x * sensitivity;
        client->camera_elevation += mouse_delta.y * sensitivity;
        client->camera_elevation = clampf_render(client->camera_elevation,
                                                  -M_PI / 2.0f + 0.1f,
                                                  M_PI / 2.0f - 0.1f);
        client->last_mouse_pos = mouse_pos;
        update_camera_position(client);
    }

    // Zoom with mouse wheel
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf_render(client->camera_distance, 5.0f, 200.0f);
        update_camera_position(client);
    }
}

Client* make_client(OrbitalDock *env) {
    Client *client = (Client*)calloc(1, sizeof(Client));

    client->width = RENDER_WIDTH;
    client->height = RENDER_HEIGHT;

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(RENDER_WIDTH, RENDER_HEIGHT, "PufferLib Orbital Dock");
    SetTargetFPS(60);

    if (!IsWindowReady()) {
        free(client);
        return NULL;
    }

    client->camera_distance = 50.0f;
    client->camera_azimuth = M_PI / 4.0f;
    client->camera_elevation = M_PI / 6.0f;
    client->is_dragging = false;
    client->last_mouse_pos = (Vector2){0, 0};

    client->camera.up = (Vector3){0, 0, 1};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;

    update_camera_position(client);

    client->scale = 10.0f;

    client->trail.index = 0;
    client->trail.count = 0;
    client->last_step = -1;
    for (int i = 0; i < TRAIL_LENGTH; i++) {
        client->trail.pos[i] = vec3d(0, 0, 0);
    }

    /* Earth + starfield disabled for now
    client->orbit_angle = 0.0;
    client->earth_angle = 0.0;
    srand(12345);
    for (int i = 0; i < NUM_STARS; i++) {
        double x, y, z, r2;
        do {
            x = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            y = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            z = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            r2 = x*x + y*y + z*z;
        } while (r2 > 1.0 || r2 < 0.001);
        double inv_r = 400.0 / sqrt(r2);
        client->stars[i] = vec3d(x * inv_r, y * inv_r, z * inv_r);
    }
    client->earth_model = LoadModel("resources/orbital_dock/ps1_style_low_poly_earth.glb");
    */

    return client;
}

void close_client(Client *client) {
    if (client) {
        /* UnloadModel(client->earth_model); */
        CloseWindow();
        free(client);
    }
}

// Convert LVLH position to render coordinates
// LVLH: x=R-bar(radial), y=V-bar(prograde), z=H-bar(normal)
// Render: X=V-bar, Y=H-bar, Z=R-bar (nice viewing angle)
static Vector3 lvlh_to_render(Client *client, double lx, double ly, double lz) {
    float s = client->scale;
    return (Vector3){
        (float)(ly / s),  // X = V-bar (prograde)
        (float)(lz / s),  // Y = H-bar (normal)
        (float)(lx / s)   // Z = R-bar (radial)
    };
}

void render_orbital_dock(OrbitalDock *env, Client *client) {
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        close_client(client);
        exit(0);
    }

    handle_camera_controls(client);
    WaitTime(0.033);  // ~30fps max

    // CW state is already in LVLH relative to station at origin
    double dist = sqrt(
        (env->x - env->dock_x) * (env->x - env->dock_x) +
        (env->y - env->dock_y) * (env->y - env->dock_y) +
        (env->z - env->dock_z) * (env->z - env->dock_z));
    double rel_speed = sqrt(env->vx*env->vx + env->vy*env->vy + env->vz*env->vz);

    // Set scale once at episode start
    if (env->step_count <= 1) {
        double pos_mag = sqrt(env->x*env->x + env->y*env->y + env->z*env->z);
        client->scale = fmax(10.0f, fmin(10000.0f, (float)pos_mag / 10.0f));
    }

    // Update trail
    if (env->step_count < client->last_step || env->step_count == 0) {
        client->trail.count = 0;
        client->trail.index = 0;
    }
    client->last_step = env->step_count;

    if (env->step_count % 5 == 0) {
        double s = client->scale;
        client->trail.pos[client->trail.index] = vec3d(
            env->y / s, env->z / s, env->x / s);
        client->trail.index = (client->trail.index + 1) % TRAIL_LENGTH;
        if (client->trail.count < TRAIL_LENGTH) {
            client->trail.count++;
        }
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    BeginMode3D(client->camera);

    // Bounding box: matches drone env scale (~60 render units at camera dist 50)
    DrawCubeWires((Vector3){0, 0, 0}, 60.0f, 60.0f, 60.0f, (Color){80, 80, 80, 255});

    /* Earth + starfield disabled for now
    client->orbit_angle += 0.005;
    float es = 15.0f;
    float theta = (float)client->orbit_angle;
    Vector3 earth_pos = {0.0f, 0.0f, -30.0f};
    {
        Matrix m = MatrixScale(es, es, es);
        m = MatrixMultiply(m, MatrixRotateY(theta));
        m = MatrixMultiply(m, MatrixTranslate(earth_pos.x, earth_pos.y, earth_pos.z));
        client->earth_model.transform = m;
        DrawModel(client->earth_model, (Vector3){0, 0, 0}, 1.0f, WHITE);
    }
    {
        double cos_a = cos(client->orbit_angle);
        double sin_a = sin(client->orbit_angle);
        for (int i = 0; i < NUM_STARS; i++) {
            double sx = client->stars[i].x;
            double sy = client->stars[i].y;
            double sz = client->stars[i].z;
            float rx = (float)(sx * cos_a + sz * sin_a);
            float ry = (float)sy;
            float rz = (float)(-sx * sin_a + sz * cos_a);
            float edx = rx - earth_pos.x;
            float edy = ry - earth_pos.y;
            float edz = rz - earth_pos.z;
            if (edx*edx + edy*edy + edz*edz < 18.0f*18.0f) continue;
            unsigned char b = (unsigned char)(180 + (i * 73) % 76);
            DrawCube((Vector3){rx, ry, rz}, 0.8f, 0.8f, 0.8f,
                     (Color){b, b, (unsigned char)(b + 15), 255});
        }
    }
    */

    // LVLH axes at station (origin)
    float axis_len = 8.0f;
    DrawLine3D((Vector3){0, 0, 0}, (Vector3){axis_len, 0, 0}, PUFF_GREEN);
    DrawSphere((Vector3){axis_len, 0, 0}, 0.2f, PUFF_GREEN);
    DrawLine3D((Vector3){0, 0, 0}, (Vector3){0, axis_len, 0}, PUFF_CYAN);
    DrawSphere((Vector3){0, axis_len, 0}, 0.2f, PUFF_CYAN);
    DrawLine3D((Vector3){0, 0, 0}, (Vector3){0, 0, axis_len}, PUFF_YELLOW);
    DrawSphere((Vector3){0, 0, axis_len}, 0.2f, PUFF_YELLOW);

    // Station (cyan cube at origin)
    DrawCube((Vector3){0, 0, 0}, 0.5f, 0.5f, 0.5f, PUFF_CYAN);
    DrawCubeWires((Vector3){0, 0, 0}, 0.6f, 0.6f, 0.6f, PUFF_WHITE);

    // Docking point (small green sphere)
    Vector3 dock_pos = lvlh_to_render(client, env->dock_x, env->dock_y, env->dock_z);
    DrawSphere(dock_pos, 0.2f, PUFF_GREEN);

    // Chaser
    Vector3 chaser_pos = lvlh_to_render(client, env->x, env->y, env->z);
    DrawSphere(chaser_pos, 0.3f, PUFF_WHITE);

    // Velocity vector (magenta)
    if (rel_speed > 0.01) {
        float vel_vis_scale = 50.0f;
        Vector3 vel_end = lvlh_to_render(client,
            env->x + env->vx * vel_vis_scale,
            env->y + env->vy * vel_vis_scale,
            env->z + env->vz * vel_vis_scale);
        DrawLine3D(chaser_pos, vel_end, PUFF_MAGENTA);
    }

    // Thrust actions
    float act_x = env->actions[0];  // R-bar
    float act_y = env->actions[1];  // V-bar
    float act_z = env->actions[2];  // H-bar

    // Thrust command vector (orange)
    float act_mag = sqrtf(act_x*act_x + act_y*act_y + act_z*act_z);
    if (act_mag > 0.05f) {
        float cmd_scale = 3.0f;
        Vector3 cmd_end = {
            chaser_pos.x + act_y * cmd_scale,  // V-bar -> X
            chaser_pos.y + act_z * cmd_scale,  // H-bar -> Y
            chaser_pos.z + act_x * cmd_scale   // R-bar -> Z
        };
        DrawLine3D(chaser_pos, cmd_end, PUFF_ORANGE);
        DrawSphere(cmd_end, 0.15f, PUFF_ORANGE);
    }

    // Docking zone (wireframe sphere around dock point)
    float dock_zone_radius = (float)(env->dock_dist / client->scale);
    DrawSphereWires(dock_pos, dock_zone_radius, 12, 12, (Color){0, 255, 0, 100});

    // LOS cone (wireframe from dock point along +y in LVLH)
    {
        double ha = env->los_half_angle;
        double extent = env->los_extent;
        int n_seg = 24;
        int n_rings = 4;
        int n_edges = 8;
        Color cone_color = {0, 187, 0, 60};

        // Draw rings at evenly spaced distances along the cone
        for (int ri = 1; ri <= n_rings; ri++) {
            double d = extent * ri / n_rings;
            double r = d * tan(ha);
            for (int i = 0; i < n_seg; i++) {
                double a0 = 2.0 * M_PI * i / n_seg;
                double a1 = 2.0 * M_PI * (i + 1) / n_seg;
                Vector3 p0 = lvlh_to_render(client,
                    env->dock_x + r * cos(a0),
                    env->dock_y + d,
                    env->dock_z + r * sin(a0));
                Vector3 p1 = lvlh_to_render(client,
                    env->dock_x + r * cos(a1),
                    env->dock_y + d,
                    env->dock_z + r * sin(a1));
                DrawLine3D(p0, p1, cone_color);
            }
        }

        // Draw edge lines from dock point to cone rim
        double r_end = extent * tan(ha);
        for (int i = 0; i < n_edges; i++) {
            double a = 2.0 * M_PI * i / n_edges;
            Vector3 tip = dock_pos;
            Vector3 rim = lvlh_to_render(client,
                env->dock_x + r_end * cos(a),
                env->dock_y + extent,
                env->dock_z + r_end * sin(a));
            DrawLine3D(tip, rim, cone_color);
        }
    }

    // Trail
    if (client->trail.count > 2) {
        for (int j = 0; j < client->trail.count - 1; j++) {
            int idx0 = (client->trail.index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
            int idx1 = (client->trail.index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;

            float age_factor = (float)j / (float)client->trail.count;
            unsigned char brightness = (unsigned char)(187 * (1.0f - age_factor * 0.8f));
            Color trail_color = {0, brightness, brightness, 255};

            Vector3 p0 = {(float)client->trail.pos[idx0].x,
                          (float)client->trail.pos[idx0].y,
                          (float)client->trail.pos[idx0].z};
            Vector3 p1 = {(float)client->trail.pos[idx1].x,
                          (float)client->trail.pos[idx1].y,
                          (float)client->trail.pos[idx1].z};
            DrawLine3D(p0, p1, trail_color);
        }
    }

    EndMode3D();

    // HUD
    int y = 20;
    int dy = 22;
    DrawText(TextFormat("Distance to dock: %.1f m", dist), 20, y, 20, PUFF_WHITE); y += dy;
    DrawText(TextFormat("Rel Speed: %.2f m/s", rel_speed), 20, y, 20, PUFF_WHITE); y += dy;
    DrawText(TextFormat("Fuel: %.1f%%", 100.0 * env->fuel / env->fuel_budget), 20, y, 20, PUFF_WHITE); y += dy;
    DrawText(TextFormat("Step: %d / %d", env->step_count, env->max_steps), 20, y, 20, PUFF_WHITE); y += dy;
    y += dy/2;
    DrawText(TextFormat("Pos: [%.0f, %.0f, %.0f] m", env->x, env->y, env->z), 20, y, 18, (Color){150, 150, 150, 255}); y += dy - 4;
    DrawText(TextFormat("Vel: [%.2f, %.2f, %.2f] m/s", env->vx, env->vy, env->vz), 20, y, 18, (Color){150, 150, 150, 255}); y += dy;
    DrawText(TextFormat("Scale: %.0f m/unit", client->scale), 20, y, 20, PUFF_WHITE); y += dy;

    y += dy;
    DrawText("Thrust Cmd:", 20, y, 20, PUFF_ORANGE); y += dy;
    DrawText(TextFormat("  R-bar: %+.0f%%", act_x * 100.0f), 20, y, 18,
             act_x > 0.05 ? PUFF_GREEN : (act_x < -0.05 ? PUFF_RED : PUFF_WHITE)); y += dy - 4;
    DrawText(TextFormat("  V-bar: %+.0f%%", act_y * 100.0f), 20, y, 18,
             act_y > 0.05 ? PUFF_GREEN : (act_y < -0.05 ? PUFF_RED : PUFF_WHITE)); y += dy - 4;
    DrawText(TextFormat("  H-bar: %+.0f%%", act_z * 100.0f), 20, y, 18,
             act_z > 0.05 ? PUFF_GREEN : (act_z < -0.05 ? PUFF_RED : PUFF_WHITE)); y += dy;

    // Axis legend (right side)
    int rx = RENDER_WIDTH - 180;
    y = RENDER_HEIGHT - 100;
    DrawText("LVLH Axes:", rx, y, 18, PUFF_WHITE); y += 20;
    DrawText("  V-bar (prograde)", rx, y, 16, PUFF_GREEN); y += 18;
    DrawText("  H-bar (normal)", rx, y, 16, PUFF_CYAN); y += 18;
    DrawText("  R-bar (radial)", rx, y, 16, PUFF_YELLOW);

    // Controls
    DrawText("Left click + drag: Rotate camera", 20, RENDER_HEIGHT - 40, 16, (Color){150, 150, 150, 255});
    DrawText("Mouse wheel: Zoom | ESC: Exit", 20, RENDER_HEIGHT - 20, 16, (Color){150, 150, 150, 255});

    // Dock status
    if (dist < env->dock_dist && rel_speed < env->dock_speed) {
        DrawText("DOCKED!", RENDER_WIDTH/2 - 60, RENDER_HEIGHT/2 - 20, 40, PUFF_GREEN);
    } else if (dist < env->dock_dist) {
        DrawText("TOO FAST!", RENDER_WIDTH/2 - 70, RENDER_HEIGHT/2 - 20, 40, PUFF_RED);
    }

    EndDrawing();
}

#endif // ORBITAL_DOCK_RENDER_H
