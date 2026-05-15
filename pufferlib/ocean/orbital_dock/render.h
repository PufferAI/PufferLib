#ifndef ORBITAL_DOCK_RENDER_H
#define ORBITAL_DOCK_RENDER_H

#include "raymath.h"
#include <float.h>

// ============================================================================
// Render Client Struct
// ============================================================================

#define RENDER_WIDTH 1080
#define RENDER_HEIGHT 720
#define TRAIL_LENGTH 256
#define NUM_STARS 300
#define EARTH_RADIUS_M 6371000.0
#define GEO_RADIUS_M 42164000.0
#define VISUAL_ORBIT_TIME_SCALE 30.0
#define VISUAL_PATH_DIRECTION -1.0  // Orbit path direction around Earth in render
#define VISUAL_VBAR_DIRECTION 1.0   // Keep LVLH +V-bar orientation as configured
#define VISUAL_SPIN_TIME_SCALE 2.0
#define VISUAL_EARTH_SPIN_MULT 60.0
#define LVLH_OVERLAY_SCALE_M 40.0
#define LVLH_VIS_MAX_M 900.0

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
    float camera_min_distance;
    float camera_max_distance;
    bool is_dragging;
    Vector2 last_mouse_pos;

    Trail trail;
    float far_scale;          // Far background scale for docking mode
    float star_radius;
    int last_step;

    // Render-only analytical orbital state
    double mean_motion;
    double theta;
    double earth_spin_angle;
    Vec3d rs;
    Vec3d rhat;
    Vec3d vhat;
    Vec3d hhat;

    Vec3d stars[NUM_STARS];
    Model earth_model;
    bool earth_loaded;
    Vector3 earth_center;
    float earth_model_radius;
};

// Additional colors
static const Color PUFF_GREEN = {0, 187, 0, 255};
static const Color PUFF_MAGENTA = {187, 0, 187, 255};

static inline Vec3d vec3d(double x, double y, double z) {
    Vec3d v = {x, y, z};
    return v;
}

static inline Vec3d vec3d_add(Vec3d a, Vec3d b) {
    return vec3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline Vec3d vec3d_scale(Vec3d a, double s) {
    return vec3d(a.x * s, a.y * s, a.z * s);
}

static inline double vec3d_dot(Vec3d a, Vec3d b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vec3d vec3d_cross(Vec3d a, Vec3d b) {
    return vec3d(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static inline double vec3d_norm(Vec3d a) {
    return sqrt(vec3d_dot(a, a));
}

static inline Vec3d vec3d_normalize(Vec3d a, Vec3d fallback) {
    double n = vec3d_norm(a);
    if (n < 1e-12) return fallback;
    return vec3d_scale(a, 1.0 / n);
}

static inline float clampf_render(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline unsigned int render_xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline double render_rand_signed(unsigned int* state) {
    return 2.0 * ((double)render_xorshift32(state) / (double)0xFFFFFFFF) - 1.0;
}

static void set_default_camera(Client *client) {
    client->camera_distance = 170.0f;
    client->camera_azimuth = -0.72f;
    client->camera_elevation = 0.26f;
    client->camera_min_distance = 12.0f;
    client->camera_max_distance = 360.0f;
}

static void update_camera_position(Client *c, Vector3 target) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

    float x = r * cosf(el) * cosf(az);
    float y = r * cosf(el) * sinf(az);
    float z = r * sinf(el);

    c->camera.position = (Vector3){target.x + x, target.y + y, target.z + z};
    c->camera.target = target;
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
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf_render(client->camera_distance,
                                                client->camera_min_distance,
                                                client->camera_max_distance);
    }
}

static void update_orbital_basis(Client *client, OrbitalDock *env) {
    double R = env->station_radius;
    client->mean_motion = sqrt(env->mu / (R * R * R));

    // Use global_step so visual orbit is continuous across episode resets.
    double t = (double)env->global_step * env->dt * VISUAL_ORBIT_TIME_SCALE;
    double path_dir = VISUAL_PATH_DIRECTION;
    client->theta = fmod(path_dir * t * client->mean_motion, 2.0 * M_PI);
    if (client->theta < 0.0) client->theta += 2.0 * M_PI;

    double c = cos(client->theta);
    double s = sin(client->theta);
    // Orbit in X-Y plane (normal +Z) to match renderer's Z-up convention.
    client->rs = vec3d(R * c, R * s, 0.0);
    client->rhat = vec3d_normalize(client->rs, vec3d(1.0, 0.0, 0.0));
    client->hhat = vec3d(0.0, 0.0, 1.0);
    client->vhat = vec3d_normalize(vec3d_cross(client->hhat, client->rhat),
                                   vec3d(0.0, 1.0, 0.0));
    if (VISUAL_VBAR_DIRECTION < 0.0) {
        client->vhat = vec3d_scale(client->vhat, -1.0);
    }
}

static Vector3 eci_to_render(Vec3d p_eci, float meters_per_unit) {
    return (Vector3){
        (float)(p_eci.x / meters_per_unit),
        (float)(p_eci.y / meters_per_unit),
        (float)(p_eci.z / meters_per_unit)
    };
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

    client->is_dragging = false;
    client->last_mouse_pos = (Vector2){0, 0};

    client->camera.up = (Vector3){0, 0, 1};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;

    set_default_camera(client);
    update_camera_position(client, (Vector3){0, 0, 0});

    client->far_scale = (float)fmax(1.0, env->station_radius / 42.0);
    client->star_radius = 520.0f;

    client->trail.index = 0;
    client->trail.count = 0;
    client->last_step = -1;
    for (int i = 0; i < TRAIL_LENGTH; i++) {
        client->trail.pos[i] = vec3d(0, 0, 0);
    }

    update_orbital_basis(client, env);

    unsigned int seed = 0xC0FFEE01u;
    for (int i = 0; i < NUM_STARS; i++) {
        double x, y, z, r2;
        do {
            x = render_rand_signed(&seed);
            y = render_rand_signed(&seed);
            z = render_rand_signed(&seed);
            r2 = x*x + y*y + z*z;
        } while (r2 > 1.0 || r2 < 0.001);
        double inv_r = 1.0 / sqrt(r2);
        client->stars[i] = vec3d(x * inv_r, y * inv_r, z * inv_r);
    }

    client->earth_model = LoadModel("resources/orbital_dock/ps1_style_low_poly_earth.glb");
    client->earth_loaded = client->earth_model.meshCount > 0;
    client->earth_center = (Vector3){0.0f, 0.0f, 0.0f};
    client->earth_model_radius = 1.0f;
    if (client->earth_loaded && client->earth_model.meshCount > 0) {
        Vector3 bb_min = (Vector3){FLT_MAX, FLT_MAX, FLT_MAX};
        Vector3 bb_max = (Vector3){-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (int i = 0; i < client->earth_model.meshCount; i++) {
            BoundingBox bb = GetMeshBoundingBox(client->earth_model.meshes[i]);
            bb_min.x = fminf(bb_min.x, bb.min.x);
            bb_min.y = fminf(bb_min.y, bb.min.y);
            bb_min.z = fminf(bb_min.z, bb.min.z);
            bb_max.x = fmaxf(bb_max.x, bb.max.x);
            bb_max.y = fmaxf(bb_max.y, bb.max.y);
            bb_max.z = fmaxf(bb_max.z, bb.max.z);
        }
        client->earth_center = (Vector3){
            0.5f * (bb_min.x + bb_max.x),
            0.5f * (bb_min.y + bb_max.y),
            0.5f * (bb_min.z + bb_max.z)
        };
        float hx = 0.5f * (bb_max.x - bb_min.x);
        float hy = 0.5f * (bb_max.y - bb_min.y);
        float hz = 0.5f * (bb_max.z - bb_min.z);
        float r = sqrtf(hx*hx + hy*hy + hz*hz);
        client->earth_model_radius = fmaxf(1e-4f, r);
        Matrix center = MatrixTranslate(
            -client->earth_center.x,
            -client->earth_center.y,
            -client->earth_center.z
        );
        Matrix align = MatrixRotateX(PI / 2.0f);
        client->earth_model.transform = MatrixMultiply(align, center);
    }
    client->earth_spin_angle = 0.0;

    return client;
}

void close_client(Client *client) {
    if (client) {
        if (client->earth_loaded) {
            UnloadModel(client->earth_model);
        }
        CloseWindow();
        free(client);
    }
}

static void draw_earth_model(Client *client, Vector3 render_pos, float model_scale) {
    if (!client->earth_loaded) {
        DrawSphere(render_pos, model_scale, (Color){30, 90, 180, 255});
        return;
    }

    float draw_scale = model_scale / client->earth_model_radius;
    DrawModelEx(client->earth_model,
                render_pos,
                (Vector3){0.0f, 0.0f, 1.0f},
                RAD2DEG * (float)client->earth_spin_angle,
                (Vector3){draw_scale, draw_scale, draw_scale},
                WHITE);
}

static void draw_orbit_ring_scaled(OrbitalDock *env, float eci_meters_per_unit, Color color) {
    const int n_seg = 192;
    for (int i = 0; i < n_seg; i++) {
        double t0 = 2.0 * M_PI * i / n_seg;
        double t1 = 2.0 * M_PI * (i + 1) / n_seg;
        Vec3d p0 = vec3d(env->station_radius * cos(t0), env->station_radius * sin(t0), 0.0);
        Vec3d p1 = vec3d(env->station_radius * cos(t1), env->station_radius * sin(t1), 0.0);
        DrawLine3D(eci_to_render(p0, eci_meters_per_unit),
                   eci_to_render(p1, eci_meters_per_unit), color);
    }
}

static void draw_station_trail_scaled(Client *client, OrbitalDock *env, float eci_meters_per_unit, Color base_color) {
    const int n_pts = 220;
    double dtheta = client->mean_motion * env->dt * 3.0;
    for (int i = 0; i < n_pts - 1; i++) {
        double t0 = client->theta - dtheta * i;
        double t1 = client->theta - dtheta * (i + 1);
        Vec3d p0 = vec3d(env->station_radius * cos(t0), env->station_radius * sin(t0), 0.0);
        Vec3d p1 = vec3d(env->station_radius * cos(t1), env->station_radius * sin(t1), 0.0);
        float age = (float)i / (float)n_pts;
        float fade = 1.0f - age;
        unsigned char r = (unsigned char)(base_color.r * (0.3f + 0.7f * fade));
        unsigned char g = (unsigned char)(base_color.g * (0.3f + 0.7f * fade));
        unsigned char b = (unsigned char)(base_color.b * (0.3f + 0.7f * fade));
        Color c = {r, g, b, (unsigned char)(230.0f * fade)};
        DrawLine3D(eci_to_render(p0, eci_meters_per_unit),
                   eci_to_render(p1, eci_meters_per_unit), c);
    }
}

// Render LVLH offsets as a local overlay around the station:
// station position uses far-scene scale, relative offsets use local overlay scale.
static Vector3 lvlh_to_orbit_overlay_render(Client *client,
                                            double x, double y, double z,
                                            float station_m_per_unit,
                                            float overlay_m_per_unit) {
    Vec3d delta_eci = vec3d_add(
        vec3d_scale(client->rhat, x),
        vec3d_add(vec3d_scale(client->vhat, y), vec3d_scale(client->hhat, z)));
    Vector3 station = eci_to_render(client->rs, station_m_per_unit);
    Vector3 delta = eci_to_render(delta_eci, overlay_m_per_unit);
    return (Vector3){station.x + delta.x, station.y + delta.y, station.z + delta.z};
}

// Convert a pure LVLH direction/offset (meters) into render delta for overlay scale.
static Vector3 lvlh_overlay_delta_render(Client *client,
                                         double x, double y, double z,
                                         float overlay_m_per_unit) {
    Vec3d delta_eci = vec3d_add(
        vec3d_scale(client->rhat, x),
        vec3d_add(vec3d_scale(client->vhat, y), vec3d_scale(client->hhat, z)));
    return eci_to_render(delta_eci, overlay_m_per_unit);
}

static Vector3 lvlh_to_orbit_render_clamped(Client *client,
                                            double x, double y, double z,
                                            float station_m_per_unit,
                                            float overlay_m_per_unit,
                                            double max_range_m,
                                            bool *clipped_out) {
    double r = sqrt(x*x + y*y + z*z);
    double k = 1.0;
    bool clipped = false;
    if (r > max_range_m && r > 1e-6) {
        k = max_range_m / r;
        clipped = true;
    }
    if (clipped_out) *clipped_out = clipped;
    return lvlh_to_orbit_overlay_render(client, x * k, y * k, z * k,
                                        station_m_per_unit, overlay_m_per_unit);
}

static void draw_stars(Client *client) {
    for (int i = 0; i < NUM_STARS; i++) {
        Vector3 p = (Vector3){
            (float)(client->stars[i].x * client->star_radius),
            (float)(client->stars[i].y * client->star_radius),
            (float)(client->stars[i].z * client->star_radius)
        };
        unsigned char b = (unsigned char)(180 + (i * 73) % 76);
        DrawCubeV(p, (Vector3){1.00f, 1.00f, 1.00f},
                  (Color){b, b, b, 255});
    }
}

void render_orbital_dock(OrbitalDock *env, Client *client) {
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        close_client(client);
        exit(0);
    }

    handle_camera_controls(client);
    WaitTime(0.033);  // ~30fps max

    update_orbital_basis(client, env);
    const double earth_spin_rate = 7.2921159e-5;  // rad/s
    client->earth_spin_angle = fmod(
        (double)env->global_step * env->dt * earth_spin_rate
            * VISUAL_SPIN_TIME_SCALE * VISUAL_EARTH_SPIN_MULT,
        2.0 * M_PI
    );

    // CW state is already in LVLH relative to station at origin
    double dist = sqrt(
        (env->x - env->dock_x) * (env->x - env->dock_x) +
        (env->y - env->dock_y) * (env->y - env->dock_y) +
        (env->z - env->dock_z) * (env->z - env->dock_z));
    double rel_speed = sqrt(env->vx*env->vx + env->vy*env->vy + env->vz*env->vz);

    // Update local LVLH trail for docking mode
    if (env->step_count < client->last_step || env->step_count == 0) {
        client->trail.count = 0;
        client->trail.index = 0;
    }
    client->last_step = env->step_count;

    if (env->step_count % 5 == 0) {
        client->trail.pos[client->trail.index] = vec3d(env->x, env->y, env->z);
        client->trail.index = (client->trail.index + 1) % TRAIL_LENGTH;
        if (client->trail.count < TRAIL_LENGTH) {
            client->trail.count++;
        }
    }

    float act_x = env->actions[0];  // R-bar
    float act_y = env->actions[1];  // V-bar
    float act_z = env->actions[2];  // H-bar

    // Station position in the far-orbit visual scale (used by docking mode).
    Vector3 station_far = eci_to_render(client->rs, client->far_scale);
    float lvlh_scale_overlay = LVLH_OVERLAY_SCALE_M;

    BeginDrawing();
    ClearBackground(BLACK);

    update_camera_position(client, station_far);

    BeginMode3D(client->camera);

    // DOCKING view: Earth-centered, LVLH frame moves along station orbit.
    float orbit_radius = (float)(env->station_radius / client->far_scale);
    float earth_radius = orbit_radius * (float)(EARTH_RADIUS_M / GEO_RADIUS_M);

    draw_stars(client);
    draw_earth_model(client, (Vector3){0, 0, 0}, earth_radius);
    draw_orbit_ring_scaled(env, client->far_scale, (Color){85, 115, 165, 95});
    draw_station_trail_scaled(client, env, client->far_scale, (Color){0, 230, 255, 255});

    // Station marker (match old LVLH cue)
    DrawCube(station_far, 0.45f, 0.45f, 0.45f, PUFF_CYAN);
    DrawCubeWires(station_far, 0.58f, 0.58f, 0.58f, PUFF_WHITE);

    // LVLH triad at station in ECI-aligned render coordinates.
    {
        float axis_len = 6.0f;
        double axis_m = axis_len * client->far_scale;
        Vector3 r_axis = eci_to_render(vec3d_add(client->rs, vec3d_scale(client->rhat, axis_m)), client->far_scale);
        Vector3 v_axis = eci_to_render(vec3d_add(client->rs, vec3d_scale(client->vhat, axis_m)), client->far_scale);
        Vector3 h_axis = eci_to_render(vec3d_add(client->rs, vec3d_scale(client->hhat, axis_m)), client->far_scale);
        DrawLine3D(station_far, v_axis, PUFF_GREEN);
        DrawLine3D(station_far, h_axis, PUFF_CYAN);
        DrawLine3D(station_far, r_axis, PUFF_YELLOW);
        DrawSphere(v_axis, 0.18f, PUFF_GREEN);
        DrawSphere(h_axis, 0.18f, PUFF_CYAN);
        DrawSphere(r_axis, 0.18f, PUFF_YELLOW);
    }

    Vector3 dock_pos = lvlh_to_orbit_overlay_render(client, env->dock_x, env->dock_y, env->dock_z,
                                                    client->far_scale, lvlh_scale_overlay);
    DrawSphere(dock_pos, 0.24f, PUFF_GREEN);

    Vector3 chaser_pos = lvlh_to_orbit_render_clamped(client, env->x, env->y, env->z,
                                                      client->far_scale, lvlh_scale_overlay,
                                                      LVLH_VIS_MAX_M,
                                                      NULL);
    DrawSphere(chaser_pos, 0.32f, PUFF_WHITE);

    // Purple heading/velocity cue: always draw from visible chaser marker.
    if (rel_speed > 0.001) {
        double vnorm = 1.0 / rel_speed;
        double hx = env->vx * vnorm;
        double hy = env->vy * vnorm;
        double hz = env->vz * vnorm;
        Vector3 hdelta = lvlh_overlay_delta_render(client,
            hx * 160.0, hy * 160.0, hz * 160.0, lvlh_scale_overlay);
        Vector3 heading_end = {
            chaser_pos.x + hdelta.x,
            chaser_pos.y + hdelta.y,
            chaser_pos.z + hdelta.z
        };
        DrawLine3D(chaser_pos, heading_end, PUFF_MAGENTA);
        DrawSphere(heading_end, 0.12f, PUFF_MAGENTA);
    }

    float act_mag = sqrtf(act_x*act_x + act_y*act_y + act_z*act_z);
    // Orange thrust cue: always draw from visible chaser marker.
    if (act_mag > 0.02f) {
        Vector3 cdelta = lvlh_overlay_delta_render(client,
            act_x * 160.0, act_y * 160.0, act_z * 160.0, lvlh_scale_overlay);
        Vector3 cmd_end = {
            chaser_pos.x + cdelta.x,
            chaser_pos.y + cdelta.y,
            chaser_pos.z + cdelta.z
        };
        DrawLine3D(chaser_pos, cmd_end, PUFF_ORANGE);
        DrawSphere(cmd_end, 0.20f, PUFF_ORANGE);
    }

    float dock_zone_radius = fmaxf(0.12f, (float)(env->dock_dist / lvlh_scale_overlay));
    DrawSphereWires(dock_pos, dock_zone_radius, 12, 12, (Color){0, 255, 0, 100});

    // LOS cone (wireframe from dock point along +V-bar in LVLH)
    {
        double ha = env->los_half_angle;
        double extent = env->los_extent;
        int n_seg = 32;
        int n_rings = 5;
        int n_edges = 12;
        Color cone_color = {0, 220, 80, 220};

        for (int ri = 1; ri <= n_rings; ri++) {
            double d = extent * ri / n_rings;
            double r = d * tan(ha);
            for (int i = 0; i < n_seg; i++) {
                double a0 = 2.0 * M_PI * i / n_seg;
                double a1 = 2.0 * M_PI * (i + 1) / n_seg;
                Vector3 p0 = lvlh_to_orbit_overlay_render(client,
                    env->dock_x + r * cos(a0),
                    env->dock_y + d,
                    env->dock_z + r * sin(a0),
                    client->far_scale, lvlh_scale_overlay);
                Vector3 p1 = lvlh_to_orbit_overlay_render(client,
                    env->dock_x + r * cos(a1),
                    env->dock_y + d,
                    env->dock_z + r * sin(a1),
                    client->far_scale, lvlh_scale_overlay);
                DrawLine3D(p0, p1, cone_color);
                if ((ri == n_rings) && (i % 8 == 0)) {
                    DrawSphere(p0, 0.06f, (Color){0, 230, 90, 255});
                }
            }
        }

        double r_end = extent * tan(ha);
        for (int i = 0; i < n_edges; i++) {
            double a = 2.0 * M_PI * i / n_edges;
            Vector3 rim = lvlh_to_orbit_overlay_render(client,
                env->dock_x + r_end * cos(a),
                env->dock_y + extent,
                env->dock_z + r_end * sin(a),
                client->far_scale, lvlh_scale_overlay);
            DrawLine3D(dock_pos, rim, cone_color);
        }
    }

    // Chaser trail in LVLH, projected through current LVLH->ECI transform.
    if (client->trail.count > 2) {
        for (int j = 0; j < client->trail.count - 1; j++) {
            int idx0 = (client->trail.index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
            int idx1 = (client->trail.index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;

            float age_factor = (float)j / (float)client->trail.count;
            unsigned char brightness = (unsigned char)(187 * (1.0f - age_factor * 0.8f));
            Color trail_color = {0, brightness, brightness, 255};

            Vec3d p0_lvlh = client->trail.pos[idx0];
            Vec3d p1_lvlh = client->trail.pos[idx1];
            Vector3 p0 = lvlh_to_orbit_render_clamped(client, p0_lvlh.x, p0_lvlh.y, p0_lvlh.z,
                                                      client->far_scale, lvlh_scale_overlay,
                                                      LVLH_VIS_MAX_M, NULL);
            Vector3 p1 = lvlh_to_orbit_render_clamped(client, p1_lvlh.x, p1_lvlh.y, p1_lvlh.z,
                                                      client->far_scale, lvlh_scale_overlay,
                                                      LVLH_VIS_MAX_M, NULL);
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

    y += 8;
    DrawText(TextFormat("Altitude: %.0f km | Phase: %.1f deg",
            env->station_radius / 1000.0 - 6371.0,
            client->theta * 180.0 / M_PI),
            20, y, 18, (Color){190, 190, 210, 255}); y += dy;

    y += 4;
    DrawText(TextFormat("Pos: [%.0f, %.0f, %.0f] m", env->x, env->y, env->z),
            20, y, 18, (Color){150, 150, 150, 255}); y += dy - 4;
    DrawText(TextFormat("Vel: [%.2f, %.2f, %.2f] m/s", env->vx, env->vy, env->vz),
            20, y, 18, (Color){150, 150, 150, 255}); y += dy;

    DrawText("Thrust Cmd:", 20, y, 20, PUFF_ORANGE); y += dy;
    DrawText(TextFormat("  R-bar: %+.0f%%", act_x * 100.0f), 20, y, 18,
             act_x > 0.05 ? PUFF_GREEN : (act_x < -0.05 ? PUFF_RED : PUFF_WHITE)); y += dy - 4;
    DrawText(TextFormat("  V-bar: %+.0f%%", act_y * 100.0f), 20, y, 18,
             act_y > 0.05 ? PUFF_GREEN : (act_y < -0.05 ? PUFF_RED : PUFF_WHITE)); y += dy - 4;
    DrawText(TextFormat("  H-bar: %+.0f%%", act_z * 100.0f), 20, y, 18,
             act_z > 0.05 ? PUFF_GREEN : (act_z < -0.05 ? PUFF_RED : PUFF_WHITE)); y += dy;

    int rx = RENDER_WIDTH - 230;
    y = RENDER_HEIGHT - 120;
    DrawText("LVLH Axes:", rx, y, 18, PUFF_WHITE); y += 20;
    DrawText("  V-bar (prograde)", rx, y, 16, PUFF_GREEN); y += 18;
    DrawText("  H-bar (normal)", rx, y, 16, PUFF_CYAN); y += 18;
    DrawText("  R-bar (radial)", rx, y, 16, PUFF_YELLOW);

    DrawText("Left click + drag: Rotate camera", 20, RENDER_HEIGHT - 56, 16,
             (Color){150, 150, 150, 255});
    DrawText("Mouse wheel: Zoom | ESC: Exit",
             20, RENDER_HEIGHT - 34, 16, (Color){150, 150, 150, 255});

    if (dist < env->dock_dist && rel_speed < env->dock_speed) {
        DrawText("DOCKED!", RENDER_WIDTH/2 - 60, RENDER_HEIGHT/2 - 20, 40, PUFF_GREEN);
    } else if (dist < env->dock_dist) {
        DrawText("TOO FAST!", RENDER_WIDTH/2 - 70, RENDER_HEIGHT/2 - 20, 40, PUFF_RED);
    }

    EndDrawing();
}

#endif // ORBITAL_DOCK_RENDER_H
