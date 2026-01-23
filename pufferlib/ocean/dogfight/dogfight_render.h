// dogfight_render.h - Rendering functions for dogfight environment
// Extracted from dogfight.h to reduce file size
//
// Contains:
//   - draw_plane_shape() - 3D wireframe airplane
//   - handle_camera_controls() - Mouse orbit/zoom
//   - draw_obs_bar() - Single observation bar
//   - draw_obs_monitor() - Full observation HUD
//   - c_render() - Main render loop
//   - c_close() - Cleanup

#ifndef DOGFIGHT_RENDER_H
#define DOGFIGHT_RENDER_H

// Requires: raylib.h, rlgl.h, flightlib.h (Vec3, Quat), Dogfight struct

// Observation labels for each scheme (for HUD display)
// Scheme 0: OBS_ANGLES (12 obs)
static const char* OBS_LABELS_ANGLES[12] = {
    "px", "py", "pz", "speed", "pitch", "roll", "yaw",
    "tgt_az", "tgt_el", "dist", "closure", "opp_hdg"
};

// Scheme 1: OBS_PURSUIT (13 obs)
static const char* OBS_LABELS_PURSUIT[13] = {
    "speed", "potential", "pitch", "roll", "energy",
    "tgt_az", "tgt_el", "dist", "closure",
    "tgt_roll", "tgt_pitch", "aspect", "E_adv"
};

// Scheme 2: OBS_REALISTIC (10 obs)
static const char* OBS_LABELS_REALISTIC[10] = {
    "airspeed", "altitude", "pitch", "roll",
    "tgt_az", "tgt_el", "tgt_size",
    "aspect", "horizon", "dist"
};

// Scheme 3: OBS_REALISTIC_RANGE (10 obs)
static const char* OBS_LABELS_REALISTIC_RANGE[10] = {
    "airspeed", "altitude", "pitch", "roll",
    "tgt_az", "tgt_el", "range_km",
    "aspect", "horizon", "closure"
};

// Scheme 4: OBS_REALISTIC_ENEMY_STATE (13 obs)
static const char* OBS_LABELS_REALISTIC_ENEMY_STATE[13] = {
    "airspeed", "altitude", "pitch", "roll",
    "tgt_az", "tgt_el", "range_km",
    "aspect", "horizon", "closure",
    "emy_pitch", "emy_roll", "emy_hdg"
};

// Scheme 5: OBS_REALISTIC_FULL (15 obs)
static const char* OBS_LABELS_REALISTIC_FULL[15] = {
    "airspeed", "altitude", "pitch", "roll",
    "tgt_az", "tgt_el", "range_km",
    "aspect", "horizon", "closure",
    "emy_pitch", "emy_roll", "emy_hdg",
    "turn_rate", "g_load"
};

// Scheme 6: OBS_MOMENTUM (15 obs) - for mode 1 physics
static const char* OBS_LABELS_MOMENTUM[15] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect"
};

// Draw airplane shape using lines - shows roll/pitch/yaw clearly
// Body frame: X=forward, Y=right, Z=up
void draw_plane_shape(Vec3 pos, Quat ori, Color body_color, Color wing_color) {
    // Body frame points (scaled for visibility: ~20m wingspan, ~25m length)
    Vec3 nose = vec3(15, 0, 0);
    Vec3 tail = vec3(-10, 0, 0);
    Vec3 left_wing = vec3(0, -12, 0);
    Vec3 right_wing = vec3(0, 12, 0);
    Vec3 vtail_top = vec3(-8, 0, 8);       // Vertical stabilizer
    Vec3 htail_left = vec3(-10, -5, 0);    // Horizontal stabilizer
    Vec3 htail_right = vec3(-10, 5, 0);

    // Rotate all points by orientation and translate to world position
    Vec3 nose_w = add3(pos, quat_rotate(ori, nose));
    Vec3 tail_w = add3(pos, quat_rotate(ori, tail));
    Vec3 lwing_w = add3(pos, quat_rotate(ori, left_wing));
    Vec3 rwing_w = add3(pos, quat_rotate(ori, right_wing));
    Vec3 vtop_w = add3(pos, quat_rotate(ori, vtail_top));
    Vec3 htl_w = add3(pos, quat_rotate(ori, htail_left));
    Vec3 htr_w = add3(pos, quat_rotate(ori, htail_right));

    // Convert to Raylib Vector3
    Vector3 nose_r = {nose_w.x, nose_w.y, nose_w.z};
    Vector3 tail_r = {tail_w.x, tail_w.y, tail_w.z};
    Vector3 lwing_r = {lwing_w.x, lwing_w.y, lwing_w.z};
    Vector3 rwing_r = {rwing_w.x, rwing_w.y, rwing_w.z};
    Vector3 vtop_r = {vtop_w.x, vtop_w.y, vtop_w.z};
    Vector3 htl_r = {htl_w.x, htl_w.y, htl_w.z};
    Vector3 htr_r = {htr_w.x, htr_w.y, htr_w.z};

    // Fuselage (nose to tail)
    DrawLine3D(nose_r, tail_r, body_color);

    // Main wings (left to right, through center for visibility)
    DrawLine3D(lwing_r, rwing_r, wing_color);
    // Wing to fuselage connections (makes it look more solid)
    DrawLine3D(lwing_r, nose_r, wing_color);
    DrawLine3D(rwing_r, nose_r, wing_color);

    // Vertical stabilizer (tail to top)
    DrawLine3D(tail_r, vtop_r, body_color);

    // Horizontal stabilizer
    DrawLine3D(htl_r, htr_r, body_color);
    DrawLine3D(htl_r, tail_r, body_color);
    DrawLine3D(htr_r, tail_r, body_color);

    // Small sphere at nose to show front clearly
    DrawSphere(nose_r, 2.0f, body_color);
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

    // Mouse wheel zoom
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
    // Draw label (fixed width)
    DrawText(label, x, y, 14, WHITE);

    // Bar dimensions
    int bar_x = x + 80;
    int bar_w = 150;
    int bar_h = 14;

    // Draw background
    DrawRectangle(bar_x, y, bar_w, bar_h, DARKGRAY);

    // Calculate fill position
    float norm_val;
    int fill_x, fill_w;

    if (is_01_range) {
        // [0, 1] range - fill from left
        norm_val = clampf(value, 0.0f, 1.0f);
        fill_x = bar_x;
        fill_w = (int)(norm_val * bar_w);
    } else {
        // [-1, 1] range - fill from center
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

    // Color based on magnitude
    Color fill_color = GREEN;
    if (fabsf(value) > 0.9f) fill_color = YELLOW;
    if (fabsf(value) > 1.0f) fill_color = RED;

    DrawRectangle(fill_x, y, fill_w, bar_h, fill_color);

    // Draw center line for [-1,1] range
    if (!is_01_range) {
        int center = bar_x + bar_w / 2;
        DrawLine(center, y, center, y + bar_h, WHITE);
    }

    // Draw value text
    DrawText(TextFormat("%+.2f", value), bar_x + bar_w + 5, y, 14, WHITE);
}

// Draw observation monitor showing all observation values as bars
void draw_obs_monitor(Dogfight *env) {
    int start_x = 900;
    int start_y = 10;
    int row_height = 18;

    const char** labels = NULL;
    int num_obs = env->obs_size;

    // Select labels based on scheme
    switch (env->obs_scheme) {
        case OBS_ANGLES:
            labels = OBS_LABELS_ANGLES;
            break;
        case OBS_PURSUIT:
            labels = OBS_LABELS_PURSUIT;
            break;
        case OBS_REALISTIC:
            labels = OBS_LABELS_REALISTIC;
            break;
        case OBS_REALISTIC_RANGE:
            labels = OBS_LABELS_REALISTIC_RANGE;
            break;
        case OBS_REALISTIC_ENEMY_STATE:
            labels = OBS_LABELS_REALISTIC_ENEMY_STATE;
            break;
        case OBS_REALISTIC_FULL:
            labels = OBS_LABELS_REALISTIC_FULL;
            break;
        case OBS_MOMENTUM:
            labels = OBS_LABELS_MOMENTUM;
            break;
        default:
            labels = OBS_LABELS_ANGLES;
            break;
    }

    // Title
    DrawText(TextFormat("OBS (scheme %d)", env->obs_scheme),
             start_x, start_y, 16, YELLOW);
    start_y += 22;

    // Draw each observation bar
    for (int i = 0; i < num_obs; i++) {
        float val = env->observations[i];
        // Determine if this observation is [0,1] range
        // Based on observation scheme and index:
        // - Scheme 0 (ANGLES): index 3 (speed) is [0,1]
        // - Scheme 1 (PURSUIT): indices 0 (speed), 1 (potential), 4 (energy) are [0,1]
        // - Scheme 2-5 (REALISTIC*): indices 0 (airspeed), 1 (altitude) are [0,1]
        bool is_01 = false;
        switch (env->obs_scheme) {
            case OBS_ANGLES:
                is_01 = (i == 3);  // speed
                break;
            case OBS_PURSUIT:
                is_01 = (i == 0 || i == 1 || i == 4);  // speed, potential, energy
                break;
            case OBS_REALISTIC:
            case OBS_REALISTIC_RANGE:
            case OBS_REALISTIC_ENEMY_STATE:
            case OBS_REALISTIC_FULL:
                is_01 = (i == 0 || i == 1);  // airspeed, altitude
                // Also range_km (index 6) is [0,1]
                if (env->obs_scheme != OBS_REALISTIC && i == 6) is_01 = true;
                break;
            case OBS_MOMENTUM:
                // fwd_spd(0), altitude(7), energy(8), range(11) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 11);
                break;
            default:
                break;
        }
        int y = start_y + i * row_height;
        draw_obs_bar(start_x, y, labels[i], val, is_01);

        // Draw red arrow for highlighted observations
        if (env->obs_highlight[i]) {
            // Draw arrow pointing right at the label (triangle)
            int arrow_x = start_x - 20;
            int arrow_y = y + 7;  // Center vertically
            // Triangle pointing right: 3 points
            DrawTriangle(
                (Vector2){arrow_x, arrow_y - 5},      // Top
                (Vector2){arrow_x, arrow_y + 5},      // Bottom
                (Vector2){arrow_x + 12, arrow_y},     // Tip (right)
                RED
            );
        }
    }
}

void c_render(Dogfight *env) {
    // 1. Lazy initialization
    if (env->client == NULL) {
        env->client = (Client *)calloc(1, sizeof(Client));
        env->client->width = 1280;
        env->client->height = 720;
        env->client->cam_distance = 80.0f;
        env->client->cam_azimuth = 0.0f;
        env->client->cam_elevation = 0.3f;
        env->client->is_dragging = false;

        InitWindow(1280, 720, "Dogfight");
        SetTargetFPS(60);

        // Z-up coordinate system
        env->client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        env->client->camera.fovy = 45.0f;
        env->client->camera.projection = CAMERA_PERSPECTIVE;
    }

    // 2. Handle window close
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    // 3. Handle mouse controls for camera orbit
    handle_camera_controls(env->client);

    // 4. Update chase camera
    Plane *p = &env->player;
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    float dist = env->client->cam_distance;

    // Apply orbit offsets from mouse drag
    float az = env->client->cam_azimuth;
    float el = env->client->cam_elevation;

    // Base chase position (behind and above player)
    float cam_x = p->pos.x - fwd.x * dist * cosf(el) * cosf(az) + fwd.y * dist * sinf(az);
    float cam_y = p->pos.y - fwd.y * dist * cosf(el) * cosf(az) - fwd.x * dist * sinf(az);
    float cam_z = p->pos.z + dist * sinf(el) + 20.0f;

    env->client->camera.position = (Vector3){cam_x, cam_y, cam_z};
    env->client->camera.target = (Vector3){p->pos.x, p->pos.y, p->pos.z};

    // 5. Begin drawing
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});  // Dark blue-green sky

    // Set clip planes for long-range visibility (default far=1000 is too close)
    rlSetClipPlanes(1.0, 10000.0);  // near=1m, far=10km
    BeginMode3D(env->client->camera);

    // 6. Draw ground plane at z=0 (XY plane, since we use Z-up)
    // DrawPlane uses raylib's Y-up convention (XZ plane), so we draw triangles instead
    Vector3 g1 = {-2000, -2000, 0};
    Vector3 g2 = {2000, -2000, 0};
    Vector3 g3 = {2000, 2000, 0};
    Vector3 g4 = {-2000, 2000, 0};
    Color ground_color = (Color){20, 60, 20, 255};
    DrawTriangle3D(g1, g2, g3, ground_color);
    DrawTriangle3D(g1, g3, g4, ground_color);

    // 7. Draw world bounds wireframe
    // Bounds: X +/-2000, Y +/-2000, Z 0-3000 -> center at (0, 0, 1500)
    DrawCubeWires((Vector3){0, 0, 1500}, 4000, 4000, 3000, (Color){100, 100, 100, 255});

    // 8. Draw player plane (cyan wireframe airplane)
    Color cyan = {0, 255, 255, 255};
    Color light_cyan = {100, 255, 255, 255};
    draw_plane_shape(p->pos, p->ori, cyan, light_cyan);

    // 9. Draw opponent plane (red wireframe airplane)
    Plane *o = &env->opponent;
    draw_plane_shape(o->pos, o->ori, RED, ORANGE);

    // 10. Draw tracer when firing (cooldown just set = just fired)
    if (p->fire_cooldown >= FIRE_COOLDOWN - 2) {  // Show for 2 frames
        Vec3 nose = add3(p->pos, quat_rotate(p->ori, vec3(15, 0, 0)));
        Vec3 tracer_end = add3(p->pos, quat_rotate(p->ori, vec3(GUN_RANGE, 0, 0)));
        Vector3 nose_r = {nose.x, nose.y, nose.z};
        Vector3 end_r = {tracer_end.x, tracer_end.y, tracer_end.z};
        DrawLine3D(nose_r, end_r, YELLOW);
    }

    EndMode3D();

    // 10. Draw HUD
    float speed = norm3(p->vel);
    float dist_to_opp = norm3(sub3(o->pos, p->pos));

    DrawText(TextFormat("Speed: %.0f m/s", speed), 10, 10, 20, WHITE);
    DrawText(TextFormat("Altitude: %.0f m", p->pos.z), 10, 40, 20, WHITE);
    DrawText(TextFormat("Throttle: %.0f%%", p->throttle * 100.0f), 10, 70, 20, WHITE);
    DrawText(TextFormat("Distance: %.0f m", dist_to_opp), 10, 100, 20, WHITE);
    DrawText(TextFormat("Tick: %d / %d", env->tick, env->max_steps), 10, 130, 20, WHITE);
    DrawText(TextFormat("Return: %.2f", env->episode_return), 10, 160, 20, WHITE);
    DrawText(TextFormat("Perf: %.1f%% | Shots: %.0f", env->log.perf / fmaxf(env->log.n, 1.0f) * 100.0f, env->log.shots_fired), 10, 190, 20, YELLOW);

    // 11. Draw observation monitor (right side)
    draw_obs_monitor(env);

    // Controls hint
    DrawText("Mouse drag: Orbit | Scroll: Zoom | ESC: Exit", 10, (int)env->client->height - 30, 16, GRAY);

    EndDrawing();
}

void c_close(Dogfight *env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

#endif // DOGFIGHT_RENDER_H
