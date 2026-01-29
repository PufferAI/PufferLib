#ifndef DOGFIGHT_RENDER_H
#define DOGFIGHT_RENDER_H


#include "raymath.h"

// Convert our Quat (w,x,y,z) to Raylib Quaternion (x,y,z,w)
static inline Quaternion quat_to_raylib(Quat q) {
    return (Quaternion){q.x, q.y, q.z, q.w};
}

// Scheme 0: OBS_MOMENTUM (15 obs) - baseline
static const char* OBS_LABELS_MOMENTUM[15] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect"
};

// Scheme 1: OBS_MOMENTUM_BETA (17 obs)
static const char* OBS_LABELS_MOMENTUM_BETA[17] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "beta",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 2: OBS_MOMENTUM_GFORCE (17 obs)
static const char* OBS_LABELS_MOMENTUM_GFORCE[17] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "g_force",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 3: OBS_MOMENTUM_FULL (19 obs)
static const char* OBS_LABELS_MOMENTUM_FULL[19] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "beta", "g_force", "throttle",
    "tgt_az", "tgt_el", "range", "closure",
    "tgt_pitch_r", "tgt_roll_r", "E_adv"
};

// Scheme 4: OBS_MINIMAL (11 obs)
static const char* OBS_LABELS_MINIMAL[11] = {
    "fwd_spd", "aoa", "roll_r", "pitch_r", "yaw_r", "altitude",
    "tgt_az", "tgt_el", "range", "closure", "E_adv"
};

// Scheme 5: OBS_CARTESIAN (15 obs)
static const char* OBS_LABELS_CARTESIAN[15] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "tgt_x", "tgt_y", "tgt_z", "closure",
    "E_adv", "aspect"
};

// Scheme 6: OBS_DRONE_STYLE (22 obs)
static const char* OBS_LABELS_DRONE_STYLE[22] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "up_x", "up_y", "up_z",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect"
};

// Scheme 7: OBS_QBAR (16 obs)
static const char* OBS_LABELS_QBAR[16] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "q_bar",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect"
};

// Scheme 8: OBS_KITCHEN_SINK (25 obs)
static const char* OBS_LABELS_KITCHEN_SINK[25] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "beta", "g_force", "q_bar", "altitude", "energy", "throttle",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "up_x", "up_y", "up_z",
    "tgt_az", "tgt_el", "range", "closure", "E_adv"
};

void draw_plane_model(Client *client, Vec3 pos, Quat ori, Color tint, float scale_factor, float prop_angle) {
    Vector3 position = {pos.x, pos.y, pos.z};

    // Convert our quaternion (w,x,y,z) to Raylib (x,y,z,w)
    Quaternion model_rot = quat_to_raylib(ori);

    // GLB model is Y-up, we use Z-up
    // Rotate 90 deg around X to convert Y-up to Z-up
    // Then rotate to align nose with +X (model nose might point +Z or -Z)
    Vector3 x_axis = {1, 0, 0};
    Vector3 z_axis = {0, 0, 1};
    Quaternion coord_fix, nose_fix, full_fix, final_rot;

    coord_fix = QuaternionFromAxisAngle(x_axis, PI / 2);  // Y-up to Z-up
    nose_fix = QuaternionFromAxisAngle(z_axis, PI / 2);   // Rotate nose to +X
    full_fix = QuaternionMultiply(nose_fix, coord_fix);

    final_rot = QuaternionMultiply(model_rot, full_fix);

    Matrix scale_mat = MatrixScale(scale_factor, scale_factor, scale_factor);
    Matrix rot_mat = QuaternionToMatrix(final_rot);
    Matrix trans_mat = MatrixTranslate(position.x, position.y, position.z);
    Matrix base_transform = MatrixMultiply(MatrixMultiply(scale_mat, rot_mat), trans_mat);

    // Propeller rotation around model's forward axis (Y in model space before coord fix)
    // After coord_fix: model Y becomes world Z, but we want rotation around nose axis
    // Propeller spins around model's local Z axis (forward in GLB space)
    Quaternion prop_rot = QuaternionFromAxisAngle((Vector3){0, 0, 1}, prop_angle);
    Quaternion prop_final = QuaternionMultiply(final_rot, prop_rot);
    Matrix prop_rot_mat = QuaternionToMatrix(prop_final);
    Matrix prop_transform = MatrixMultiply(MatrixMultiply(scale_mat, prop_rot_mat), trans_mat);

    rlDisableColorBlend();
    Model model = client->plane_model;
    for (int i = 0; i < model.meshCount; i++) {
        if (i >= 3) continue;
        if (i == 2 && client->camera_mode == 3) continue;  // Skip blur prop in cockpit view
        Matrix transform = (i == 2) ? prop_transform : base_transform;
        int mat_idx = model.meshMaterial[i];
        Color original = model.materials[mat_idx].maps[MATERIAL_MAP_DIFFUSE].color;
        model.materials[mat_idx].maps[MATERIAL_MAP_DIFFUSE].color = (Color){
            (unsigned char)(original.r * tint.r / 255),
            (unsigned char)(original.g * tint.g / 255),
            (unsigned char)(original.b * tint.b / 255),
            original.a
        };
        DrawMesh(model.meshes[i], model.materials[mat_idx], transform);
        model.materials[mat_idx].maps[MATERIAL_MAP_DIFFUSE].color = original;
    }
    rlEnableColorBlend();
}

void handle_camera_controls(Client *c) {
    Vector2 mouse = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        c->is_dragging = true;
        c->last_mouse_x = mouse.x;
        c->last_mouse_y = mouse.y;
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
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

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        c->cam_distance = clampf(c->cam_distance - wheel * 10.0f, 30.0f, 300.0f);
    }
}

// Draw a single observation bar
// x, y: top-left position
// label: observation name
// value: the observation value
// is_01_range: true for [0,1] range, false for [-1,1] range
void draw_obs_bar(int x, int y, const char* label, float value, bool is_01_range) {
    DrawText(label, x, y, 14, WHITE);

    int bar_x = x + 80;
    int bar_w = 150;
    int bar_h = 14;

    DrawRectangle(bar_x, y, bar_w, bar_h, DARKGRAY);

    float norm_val;
    int fill_x, fill_w;

    if (is_01_range) {
        norm_val = clampf(value, 0.0f, 1.0f);
        fill_x = bar_x;
        fill_w = (int)(norm_val * bar_w);
    } else {
        norm_val = clampf(value, -1.0f, 1.0f);
        int center = bar_x + bar_w / 2;
        if (norm_val >= 0) {
            fill_x = center;
            fill_w = (int)(norm_val * bar_w / 2);
        } else {
            fill_w = (int)(-norm_val * bar_w / 2);
            fill_x = center - fill_w;
        }
    }

    Color fill_color = GREEN;
    if (fabsf(value) > 0.9f) fill_color = YELLOW;
    if (fabsf(value) > 1.0f) fill_color = RED;

    DrawRectangle(fill_x, y, fill_w, bar_h, fill_color);

    if (!is_01_range) {
        int center = bar_x + bar_w / 2;
        DrawLine(center, y, center, y + bar_h, WHITE);
    }

    DrawText(TextFormat("%+.2f", value), bar_x + bar_w + 5, y, 14, WHITE);
}

void draw_obs_monitor(Dogfight *env) {
    int start_x = 1540;
    int start_y = 10;
    int row_height = 18;

    const char** labels = NULL;
    int num_obs = env->obs_size;

    switch (env->obs_scheme) {
        case OBS_MOMENTUM:
            labels = OBS_LABELS_MOMENTUM;
            break;
        case OBS_MOMENTUM_BETA:
            labels = OBS_LABELS_MOMENTUM_BETA;
            break;
        case OBS_MOMENTUM_GFORCE:
            labels = OBS_LABELS_MOMENTUM_GFORCE;
            break;
        case OBS_MOMENTUM_FULL:
            labels = OBS_LABELS_MOMENTUM_FULL;
            break;
        case OBS_MINIMAL:
            labels = OBS_LABELS_MINIMAL;
            break;
        case OBS_CARTESIAN:
            labels = OBS_LABELS_CARTESIAN;
            break;
        case OBS_DRONE_STYLE:
            labels = OBS_LABELS_DRONE_STYLE;
            break;
        case OBS_QBAR:
            labels = OBS_LABELS_QBAR;
            break;
        case OBS_KITCHEN_SINK:
            labels = OBS_LABELS_KITCHEN_SINK;
            break;
        default:
            labels = OBS_LABELS_MOMENTUM;
            break;
    }

    DrawText(TextFormat("OBS (scheme %d)", env->obs_scheme),
             start_x, start_y, 16, YELLOW);
    start_y += 22;

    for (int i = 0; i < num_obs; i++) {
        float val = env->observations[i];
        bool is_01 = false;
        switch (env->obs_scheme) {
            case OBS_MOMENTUM:
                // fwd_spd(0), altitude(7), energy(8), range(11) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 11);
                break;
            case OBS_MOMENTUM_BETA:
            case OBS_MOMENTUM_GFORCE:
            case OBS_QBAR:
                // fwd_spd(0), altitude(7), energy(8), range(12) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 12);
                break;
            case OBS_MOMENTUM_FULL:
                // fwd_spd(0), altitude(7), energy(8), throttle(11), range(14) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 11 || i == 14);
                break;
            case OBS_MINIMAL:
                // fwd_spd(0), altitude(5), range(8) are [0,1]
                is_01 = (i == 0 || i == 5 || i == 8);
                break;
            case OBS_CARTESIAN:
                // fwd_spd(0), altitude(7), energy(8) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8);
                break;
            case OBS_DRONE_STYLE:
                // fwd_spd(0), altitude(7), energy(8), range(18) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 18);
                break;
            case OBS_KITCHEN_SINK:
                // fwd_spd(0), q_bar(9), altitude(10), energy(11), throttle(12), range(22) are [0,1]
                is_01 = (i == 0 || i == 9 || i == 10 || i == 11 || i == 12 || i == 22);
                break;
            default:
                break;
        }
        int y = start_y + i * row_height;
        draw_obs_bar(start_x, y, labels[i], val, is_01);

        if (env->obs_highlight[i]) {
            int arrow_x = start_x - 20;
            int arrow_y = y + 7;
            DrawTriangle(
                (Vector2){arrow_x, arrow_y - 5},
                (Vector2){arrow_x, arrow_y + 5},
                (Vector2){arrow_x + 12, arrow_y},
                RED
            );
        }
    }
}

void c_render(Dogfight *env) {
    if (env->client == NULL) {
        env->client = (Client *)calloc(1, sizeof(Client));
        env->client->width = 1280;
        env->client->height = 720;
        env->client->cam_distance = 80.0f;
        env->client->cam_azimuth = 0.0f;
        env->client->cam_elevation = 0.3f;
        env->client->camera_mode = 0;  // 0 = follow target, 1 = midpoint view, 2 = chase, 3 = cockpit
        env->client->is_dragging = false;

        InitWindow(1920, 1080, "Dogfight");
        SetTargetFPS(60);

        env->client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        env->client->camera.fovy = 45.0f;
        env->client->camera.projection = CAMERA_PERSPECTIVE;

        env->client->plane_model = LoadModel("pufferlib/ocean/dogfight/p40.glb");
        env->client->model_loaded = (env->client->plane_model.meshCount > 0);
    }

    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    if (IsKeyPressed(KEY_C)) {
        env->client->camera_mode = (env->client->camera_mode + 1) % 4;
    }

    handle_camera_controls(env->client);

    Plane *p = &env->player;
    Plane *o = &env->opponent;
    Plane *cam_target = env->camera_follow_opponent ? o : p;
    Vec3 fwd = quat_rotate(cam_target->ori, vec3(1, 0, 0));
    float dist = env->client->cam_distance;

    float az = env->client->cam_azimuth;
    float el = env->client->cam_elevation;

    if (env->client->camera_mode == 2) {
        // Mode 2: Direct chase - fully body-aligned (roll + pitch + yaw)
        Vec3 up = quat_rotate(cam_target->ori, vec3(0, 0, 1));
        // Position: behind and slightly above in body frame (0.25 up, was 0.5)
        float chase_dist = dist * 0.7f;
        float cam_x = cam_target->pos.x - fwd.x * chase_dist + up.x * chase_dist * 0.25f;
        float cam_y = cam_target->pos.y - fwd.y * chase_dist + up.y * chase_dist * 0.25f;
        float cam_z = cam_target->pos.z - fwd.z * chase_dist + up.z * chase_dist * 0.25f;
        env->client->camera.position = (Vector3){cam_x, cam_y, cam_z};
        // Target ahead and above plane so plane appears lower on screen
        float tgt_x = cam_target->pos.x + fwd.x * 20.0f + up.x * 20.0f;
        float tgt_y = cam_target->pos.y + fwd.y * 20.0f + up.y * 20.0f;
        float tgt_z = cam_target->pos.z + fwd.z * 20.0f + up.z * 20.0f;
        env->client->camera.target = (Vector3){tgt_x, tgt_y, tgt_z};
        env->client->camera.up = (Vector3){up.x, up.y, up.z};
    } else if (env->client->camera_mode == 3) {
        // Mode 3: Cockpit POV - at plane center, 1.25m up in body frame
        Vec3 up = quat_rotate(cam_target->ori, vec3(0, 0, 1));
        float cam_x = cam_target->pos.x + up.x * 1.11f;
        float cam_y = cam_target->pos.y + up.y * 1.11f;
        float cam_z = cam_target->pos.z + up.z * 1.11f;
        env->client->camera.position = (Vector3){cam_x, cam_y, cam_z};
        // Look forward and ~15 degrees up (tan(15°) ≈ 0.268, so 27 up per 100 forward)
        float tgt_x = cam_target->pos.x + fwd.x * 100.0f + up.x * 27.0f;
        float tgt_y = cam_target->pos.y + fwd.y * 100.0f + up.y * 27.0f;
        float tgt_z = cam_target->pos.z + fwd.z * 100.0f + up.z * 27.0f;
        env->client->camera.target = (Vector3){tgt_x, tgt_y, tgt_z};
        env->client->camera.up = (Vector3){up.x, up.y, up.z};
    } else {
        // Reset to world up for other modes
        env->client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        // Modes 0 and 1: Orbit camera position
        float cam_x = cam_target->pos.x - fwd.x * dist * cosf(el) * cosf(az) + fwd.y * dist * sinf(az);
        float cam_y = cam_target->pos.y - fwd.y * dist * cosf(el) * cosf(az) - fwd.x * dist * sinf(az);
        float cam_z = cam_target->pos.z + dist * sinf(el) + 20.0f;
        env->client->camera.position = (Vector3){cam_x, cam_y, cam_z};

        if (env->client->camera_mode == 0) {
            // Mode 0: Orbit - look at cam_target
            env->client->camera.target = (Vector3){cam_target->pos.x, cam_target->pos.y, cam_target->pos.z};
        } else {
            // Mode 1: Midpoint - look at midpoint between both planes
            float mid_x = (p->pos.x + o->pos.x) / 2.0f;
            float mid_y = (p->pos.y + o->pos.y) / 2.0f;
            float mid_z = (p->pos.z + o->pos.z) / 2.0f;
            env->client->camera.target = (Vector3){mid_x, mid_y, mid_z};
        }
    }

    WaitTime(0.02);  // Sync with sim DT (50 FPS) for proper GIF frame timing
    BeginDrawing();
    ClearBackground((Color){0, 100, 120, 255});

    rlSetClipPlanes(1.0, 15000.0);
    BeginMode3D(env->client->camera);

    // DrawPlane uses raylib's Y-up convention (XZ plane), so we draw triangles instead
    Vector3 g1 = {-4000, -4000, 0};
    Vector3 g2 = {4000, -4000, 0};
    Vector3 g3 = {4000, 4000, 0};
    Vector3 g4 = {-4000, 4000, 0};
    Color ground_color = (Color){20, 60, 20, 255};
    DrawTriangle3D(g1, g2, g3, ground_color);
    DrawTriangle3D(g1, g3, g4, ground_color);

    DrawCubeWires((Vector3){0, 0, 2500}, 8000, 8000, 5000, (Color){100, 100, 100, 255});

    float player_scale = env->camera_follow_opponent ? 4.0f : 1.0f;
    float opponent_scale = env->camera_follow_opponent ? 1.0f : 4.0f;

    float prop_speed = 150.0f + 130.0f * p->throttle;
    env->client->propeller_angle += prop_speed * 0.0167f;
    if (env->client->propeller_angle > 2.0f * PI) {
        env->client->propeller_angle -= 2.0f * PI;
    }

    draw_plane_model(env->client, p->pos, p->ori, WHITE, player_scale, env->client->propeller_angle);

    draw_plane_model(env->client, o->pos, o->ori, RED, opponent_scale, env->client->propeller_angle);

    if (p->fire_cooldown >= FIRE_COOLDOWN - 2) {
        Vec3 nose = add3(p->pos, quat_rotate(p->ori, vec3(15, 0, 0)));
        Vec3 tracer_end = add3(p->pos, quat_rotate(p->ori, vec3(GUN_RANGE, 0, 0)));
        Vector3 nose_r = {nose.x, nose.y, nose.z};
        Vector3 end_r = {tracer_end.x, tracer_end.y, tracer_end.z};
        DrawLine3D(nose_r, end_r, YELLOW);
    }

    EndMode3D();

    float speed = norm3(p->vel);
    float dist_to_opp = norm3(sub3(o->pos, p->pos));

    DrawText(TextFormat("Speed: %.0f m/s", speed), 10, 10, 20, WHITE);
    DrawText(TextFormat("Altitude: %.0f m", p->pos.z), 10, 40, 20, WHITE);
    DrawText(TextFormat("Throttle: %.0f%%", p->throttle * 100.0f), 10, 70, 20, WHITE);
    DrawText(TextFormat("Distance: %.0f m", dist_to_opp), 10, 100, 20, WHITE);
    DrawText(TextFormat("Tick: %d / %d", env->tick, env->max_steps), 10, 130, 20, WHITE);
    DrawText(TextFormat("Return: %.2f", env->episode_return), 10, 160, 20, WHITE);
    DrawText(TextFormat("Perf: %.1f%% | Shots: %.0f", env->log.perf / fmaxf(env->log.n, 1.0f) * 100.0f, env->log.shots_fired), 10, 190, 20, YELLOW);
    DrawText(TextFormat("Stage: %d", env->stage), 10, 220, 20, LIME);

    draw_obs_monitor(env);

    DrawText("Mouse drag: Orbit | Scroll: Zoom | ESC: Exit", 10, (int)env->client->height - 30, 16, GRAY);

    EndDrawing();
}

void c_close(Dogfight *env) {
    if (env->client != NULL) {
        if (env->client->model_loaded) {
            UnloadModel(env->client->plane_model);
        }
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

#endif // DOGFIGHT_RENDER_H
