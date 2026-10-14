#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "material_render.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef MAP_OPTIX
#include "car_render.h"
extern void racing_map_query(float x, float y, float z, float dx, float dz, float* result);
extern void racing_map_query_close(void);
#endif

typedef struct {
    Mesh mesh;
    BoundingBox bounds;
    uint32_t group, node, primitive, material;
    char name[160];
} Surface;

typedef struct {
    float s;
    Vector3 center, left, right;
    uint32_t flags;
} RoutePoint;

typedef struct {
    uint32_t index;
    char name[64];
} Landmark;

typedef struct { Vector3 left, right; } TrackEdge;
typedef struct { uint32_t index; Vector3 forward; TrackEdge edge; } TrackGate;

int main(int argc, char **argv) {
    const char *map_path = argc > 1 ? argv[1] : "ocean/racing/map.bin";
    const char *route_path = argc > 2 ? argv[2] : "ocean/racing/route_draft.csv";
    FILE *file = fopen(map_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open %s; run ocean/racing/prepare_map.py first\n", map_path);
        return 1;
    }
    char magic[8];
    uint32_t count;
    assert(fread(magic, 1, 8, file) == 8 && memcmp(magic, "PFMAP001", 8) == 0);
    assert(fread(&count, 4, 1, file) == 1 && count > 0 && count < 10000);
    Surface *surfaces = calloc(count, sizeof(Surface));
    assert(surfaces);
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1440, 960, "Racing map inspection | provisional surface groups");
    SetTargetFPS(60);
    SetExitKey(KEY_ESCAPE);
    FILE* visual = fopen("ocean/racing/map_visual.bin", "rb");
    if (!visual) {
        fprintf(stderr, "Missing map_visual.bin; rerun ocean/racing/view_map.sh\n");
        fclose(file);
        CloseWindow();
        free(surfaces);
        return 1;
    }
    uint32_t visual_count, texture_count, material_count;
    assert(fread(magic, 1, 8, visual) == 8 && memcmp(magic, "PFVIS004", 8) == 0);
    assert(fread(&visual_count, 4, 1, visual) == 1 && visual_count == count);
    assert(fread(&texture_count, 4, 1, visual) == 1 && texture_count <= 1024);
    assert(fread(&material_count, 4, 1, visual) == 1 && material_count <= 1024);
    Texture2D* textures = calloc(texture_count, sizeof(Texture2D));
    uint32_t* material_textures = calloc(material_count, sizeof(uint32_t));
    uint32_t* alpha_modes = calloc(material_count, sizeof(uint32_t));
    unsigned* draw_order = calloc(count, sizeof(unsigned));
    float* draw_depth = calloc(count, sizeof(float));
    RacingMaterialDetail* details = calloc(material_count, sizeof(*details));
    Color* material_colors = calloc(material_count, sizeof(Color));
    assert(details && textures && material_textures && material_colors && alpha_modes && draw_order && draw_depth);
    for (uint32_t i = 0; i < texture_count; i++) {
        uint32_t bytes;
        assert(fread(&bytes, 4, 1, visual) == 1 && bytes > 0 && bytes < 100000000);
        unsigned char* encoded = malloc(bytes);
        assert(encoded && fread(encoded, 1, bytes, visual) == bytes);
        Image image = LoadImageFromMemory(".png", encoded, bytes);
        free(encoded);
        if (!IsImageValid(image)) {
            fprintf(stderr, "Cannot decode map texture %u\n", i);
            exit(1);
        }
        textures[i] = LoadTextureFromImage(image);
        UnloadImage(image);
        assert(IsTextureValid(textures[i]));
        GenTextureMipmaps(&textures[i]);
        SetTextureFilter(textures[i], TEXTURE_FILTER_TRILINEAR);
        SetTextureFilter(textures[i], TEXTURE_FILTER_ANISOTROPIC_8X);
        SetTextureWrap(textures[i], TEXTURE_WRAP_REPEAT);
    }
    for (uint32_t i = 0; i < material_count; i++) {
        float rgba[4];
        assert(fread(&material_textures[i], 4, 1, visual) == 1);
        assert(fread(&alpha_modes[i], 4, 1, visual) == 1 && alpha_modes[i] <= 2);
        assert(fread(&details[i].normal, 4, 1, visual) == 1);
        assert(fread(&details[i].detail, 4, 1, visual) == 1);
        assert(details[i].normal < texture_count && details[i].detail < texture_count);
        assert(material_textures[i] < texture_count && fread(rgba, 4, 4, visual) == 4);
        assert(fread(&details[i].normal_strength, 4, 6, visual) == 6);
        material_colors[i] = ColorFromNormalized((Vector4){rgba[0], rgba[1], rgba[2], rgba[3]});
    }
    FILE *render_bindings=fopen("ocean/racing/render_materials.bin","rb");
    if (!render_bindings) { fprintf(stderr,"Run python3 ocean/racing/prepare_render.py --track\n"); return 1; }
    uint32_t binding_count;
    assert(fread(magic,1,8,render_bindings)==8 && memcmp(magic,"PFRMAT01",8)==0);
    assert(fread(&binding_count,4,1,render_bindings)==1 && binding_count==material_count);
    RacingRenderBinding *bindings=calloc(material_count,sizeof(*bindings));
    assert(bindings && fread(bindings,sizeof(*bindings),material_count,render_bindings)==material_count);
    fclose(render_bindings);
    BoundingBox bounds = {{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
    for (uint32_t i = 0; i < count; i++) {
        Surface *surface = &surfaces[i];
        uint32_t vertices;
        assert(fread(&vertices, 4, 1, file) == 1 && vertices % 3 == 0);
        assert(vertices > 0 && vertices < 10000000);
        assert(fread(&surface->group, 4, 1, file) == 1 && surface->group < 7);
        assert(fread(&surface->node, 4, 1, file) == 1);
        assert(fread(&surface->primitive, 4, 1, file) == 1);
        assert(fread(&surface->material, 4, 1, file) == 1);
        assert(fread(surface->name, 1, 160, file) == 160 && surface->name[159] == 0);
        surface->mesh.vertexCount = vertices;
        surface->mesh.triangleCount = vertices / 3;
        surface->mesh.vertices = MemAlloc(vertices * 3 * sizeof(float));
        assert(surface->mesh.vertices);
        assert(fread(surface->mesh.vertices, 12, vertices, file) == vertices);
        uint32_t uv_count;
        assert(fread(&uv_count, 4, 1, visual) == 1 && uv_count == vertices);
        assert(surface->material < material_count);
        surface->mesh.texcoords = MemAlloc(vertices * 2 * sizeof(float));
        assert(surface->mesh.texcoords);
        assert(fread(surface->mesh.texcoords, 8, vertices, visual) == vertices);
        surface->mesh.normals = MemAlloc(vertices * 3 * sizeof(float));
        assert(surface->mesh.normals);
        assert(fread(surface->mesh.normals, 12, vertices, visual) == vertices);
        surface->bounds = GetMeshBoundingBox(surface->mesh);
        bounds.min = Vector3Min(bounds.min, surface->bounds.min);
        bounds.max = Vector3Max(bounds.max, surface->bounds.max);
        UploadMesh(&surface->mesh, false);
    }
    fclose(file);
    fclose(visual);
    Material material = LoadMaterialDefault();
    Shader default_shader = material.shader;
    Shader texture_shader = LoadShader("ocean/racing/shaders/track.vs",
        "ocean/racing/shaders/track.fs");
    texture_shader.locs[SHADER_LOC_MATRIX_MODEL] = GetShaderLocation(texture_shader, "matModel");
    texture_shader.locs[SHADER_LOC_MATRIX_NORMAL] = GetShaderLocation(texture_shader, "matNormal");
    Shader sky_shader = LoadShader(NULL, "ocean/racing/shaders/sky.fs");
    int lit_location = GetShaderLocation(texture_shader, "lit");
    int eye_location = GetShaderLocation(texture_shader, "eyePosition");
    int fog_location = GetShaderLocation(texture_shader, "fogDensity");
    int sky_resolution = GetShaderLocation(sky_shader, "resolution");
    int sky_forward = GetShaderLocation(sky_shader, "viewForward");
    int sky_right = GetShaderLocation(sky_shader, "viewRight");
    int sky_up = GetShaderLocation(sky_shader, "viewUp");
    int sky_scale = GetShaderLocation(sky_shader, "perspectiveScale");
    Vector3 sun = Vector3Normalize((Vector3){-0.4f, 0.75f, -0.35f});
    assert(IsShaderValid(sky_shader) && sky_resolution >= 0 && lit_location >= 0);
    SetShaderValue(texture_shader, GetShaderLocation(texture_shader, "sunDirection"),
        &sun, SHADER_UNIFORM_VEC3);
    SetShaderValue(sky_shader, GetShaderLocation(sky_shader, "sunDirection"),
        &sun, SHADER_UNIFORM_VEC3);
    bool lighting = true;
    int alpha_location = GetShaderLocation(texture_shader, "alphaMode");
    int ground_location = GetShaderLocation(texture_shader, "groundSurface");
    assert(IsShaderValid(texture_shader) && alpha_location >= 0);
    racing_shader_maps(texture_shader);
    material.shader = texture_shader;
    Texture2D default_texture = material.maps[MATERIAL_MAP_DIFFUSE].texture;
    bool textured = true;
    bool diagnostics = false;
    bool show_hud = true;
    Color colors[] = {GRAY, RED, DARKGREEN, ORANGE, SKYBLUE, DARKGRAY, PURPLE};
    const char *names[] = {"Road", "Curbs", "Runoff", "Barriers / fences", "Structures",
        "Other / unclassified", "Groove / decals"};
    bool visible[] = {true, true, true, true, true, true, false};
    bool top = true, wire = false;
    Vector3 target = Vector3Scale(Vector3Add(bounds.min, bounds.max), 0.5f);
    float extent = fmaxf(bounds.max.x - bounds.min.x, bounds.max.z - bounds.min.z);
    rlSetClipPlanes(0.1, extent * 12);
    float distance = extent * 0.8f, yaw = 0, pitch = 0.9f;
    Camera3D camera = {0};
    RoutePoint candidate[4096];
    Landmark landmarks[64];
    uint32_t candidate_count = 0, landmark_count = 0;
    TrackEdge legal[4096];
    TrackGate gates[128];
    uint32_t legal_count = 0, gate_count = 0;
    bool show_legal = true, show_gates = true;
#ifdef MAP_OPTIX
    float query_lines[260][7];
    bool have_queries = false, show_queries = true;
    RacingCarRender car_render = {0};
    RacingCarFrame car_frame = {0};
    bool driving = false, car_paused = false, chase_ready = false;
    float car_distance = 8;
    double car_accumulator = 0;
#endif
    int landmark = 0, review_point = -1;
    bool show_route = true, show_edges = true, show_names = true;
    file = fopen("ocean/racing/route.bin", "rb");
    if (file) {
        assert(fread(magic, 1, 8, file) == 8 && memcmp(magic, "PFROUTE1", 8) == 0);
        assert(fread(&candidate_count, 4, 1, file) == 1 && candidate_count <= 4096);
        assert(fread(&landmark_count, 4, 1, file) == 1 && landmark_count <= 64);
        for (uint32_t i = 0; i < candidate_count; i++) {
            assert(fread(&candidate[i].s, 4, 1, file) == 1);
            assert(fread(&candidate[i].center, 4, 3, file) == 3);
            assert(fread(&candidate[i].left, 4, 3, file) == 3);
            assert(fread(&candidate[i].right, 4, 3, file) == 3);
            assert(fread(&candidate[i].flags, 4, 1, file) == 1);
        }
        for (uint32_t i = 0; i < landmark_count; i++) {
            assert(fread(&landmarks[i].index, 4, 1, file) == 1);
            assert(landmarks[i].index < candidate_count);
            assert(fread(landmarks[i].name, 1, 64, file) == 64 && landmarks[i].name[63] == 0);
        }
        fclose(file);
    }
    file = fopen("ocean/racing/track.bin", "rb");
    if (file) {
        assert(fread(magic, 1, 8, file) == 8 && memcmp(magic, "PFTRACK2", 8) == 0);
        assert(fread(&legal_count, 4, 1, file) == 1 && legal_count == candidate_count);
        assert(fread(&gate_count, 4, 1, file) == 1 && gate_count <= 128);
        for (uint32_t i = 0; i < legal_count; i++) {
            assert(fread(&legal[i].left, 4, 3, file) == 3);
            assert(fread(&legal[i].right, 4, 3, file) == 3);
        }
        for (uint32_t i = 0; i < gate_count; i++) {
            assert(fread(&gates[i].index, 4, 1, file) == 1 && gates[i].index < legal_count);
            assert(fread(&gates[i].forward, 4, 3, file) == 3);
            assert(fread(&gates[i].edge.left, 4, 3, file) == 3);
            assert(fread(&gates[i].edge.right, 4, 3, file) == 3);
        }
        fclose(file);
    }
    RayCollision selected = {0};
    int picked = -1, route_count = 0;
    Vector3 route[4096];
    file = fopen(route_path, "r");
    if (file) {
        while (route_count < 4096 && fscanf(file, "%f,%f,%f", &route[route_count].x,
            &route[route_count].y, &route[route_count].z) == 3) {
            route_count++;
        }
        fclose(file);
    }
    const char *status = "Route is an open draft; no direction or legal corridor is verified.";
    while (!WindowShouldClose()) {
#ifdef MAP_OPTIX
        bool reset_car = false;
        if (IsKeyPressed(KEY_C) && candidate_count) {
            driving = !driving;
            chase_ready = false;
            car_accumulator = 0;
            if (driving && !car_render.loaded) {
                racing_car_load(&car_render);
                reset_car = true;
            }
        }
        if (car_render.loaded && IsKeyPressed(KEY_SPACE)) {
            car_paused = !car_paused;
            car_accumulator = 0;
        }
        bool trial_start = car_render.loaded && IsKeyPressed(KEY_F6);
        if (trial_start || (car_render.loaded && IsKeyPressed(KEY_F5))) {
            reset_car = true;
        }
        if (reset_car) {
            int index = !trial_start && landmark_count ? landmarks[landmark].index : 0;
            // Start behind gate zero so the first forward crossing arms the timer.
            if (index == 0 && candidate_count > 3) index = candidate_count - 3;
            Vector3 position = candidate[index].center;
            Vector3 delta = Vector3Subtract(candidate[(index+1)%candidate_count].center, position);
            racing_car_reset(position.x, position.y, position.z, atan2f(delta.x, delta.z), &car_frame);
            racing_car_lidar(&query_lines[0][0]);
            if (trial_start) {
                driving = true;
                car_paused = false;
            }
            have_queries = true;
            car_accumulator = 0;
            chase_ready = false;
        }
        if (driving && !car_paused && !reset_car && !car_frame.crashed && !car_frame.task_done) {
            car_accumulator += fminf(GetFrameTime(), 1.0f/30);
            int steps = (int)(car_accumulator*240);
            if (steps > 0) {
                car_accumulator -= steps/240.0;
                float throttle = IsKeyDown(KEY_W) || IsKeyDown(KEY_UP);
                float brake = IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN);
                float steering = (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) -
                    (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT));
                racing_car_step(steps, throttle, brake, steering, &car_frame, &query_lines[0][0]);
            }
        }
#endif
        if (IsKeyPressed(KEY_T)) {
            top = !top;
        }
        if (IsKeyPressed(KEY_F)) {
            wire = !wire;
        }
        if (IsKeyPressed(KEY_V)) {
            textured = !textured;
        }
        if (IsKeyPressed(KEY_J)) {
            lighting = !lighting;
        }
        if (IsKeyPressed(KEY_F1)) {
            show_hud = !show_hud;
        }
        if (IsKeyPressed(KEY_H)) {
            diagnostics = !diagnostics;
        }
        if (IsKeyPressed(KEY_R)) {
            show_route = !show_route;
        }
        if (IsKeyPressed(KEY_B)) {
            show_edges = !show_edges;
        }
        if (IsKeyPressed(KEY_N)) {
            show_names = !show_names;
        }
        if (IsKeyPressed(KEY_K)) {
            show_legal = !show_legal;
        }
        if (IsKeyPressed(KEY_G)) {
            show_gates = !show_gates;
        }
#ifdef MAP_OPTIX
        if (IsKeyPressed(KEY_O)) {
            show_queries = !show_queries;
        }
        if (IsKeyPressed(KEY_L) && candidate_count) {
            int index = review_point >= 0 ? review_point :
                (landmark_count ? (int)landmarks[landmark].index : 0);
            Vector3 point = picked >= 0 ? selected.point : candidate[index].center;
            Vector3 forward = Vector3Subtract(candidate[(index+1)%candidate_count].center,
                candidate[index].center);
            if (car_render.loaded) {
                racing_car_lidar(&query_lines[0][0]);
            } else {
                racing_map_query(point.x, point.y, point.z, forward.x, forward.z,
                    &query_lines[0][0]);
                diagnostics = true;
            }
            have_queries = true;
            show_queries = true;
            status = "OptiX snapshot: 256 LiDAR + 4 diagnostic support rays.";
        }
#endif
        if (landmark_count && (IsKeyPressed(KEY_LEFT_BRACKET) ||
            IsKeyPressed(KEY_RIGHT_BRACKET))) {
            landmark = (landmark + (IsKeyPressed(KEY_RIGHT_BRACKET) ? 1 : -1) +
                (int)landmark_count) % (int)landmark_count;
            target = candidate[landmarks[landmark].index].center;
            picked = -1;
            review_point = -1;
            distance = 100;
            status = landmarks[landmark].name;
        }
        if (IsKeyPressed(KEY_P) && candidate_count) {
            for (uint32_t i = 0; i < candidate_count; i++) {
                review_point = (review_point + 1) % candidate_count;
                if (candidate[review_point].flags) {
                    target = candidate[review_point].center;
                    picked = -1;
                    distance = 60;
                    break;
                }
            }
        }
        if (IsKeyPressed(KEY_HOME)) {
            target = Vector3Scale(Vector3Add(bounds.min, bounds.max), 0.5f);
            distance = extent * 0.8f;
        }
        for (int group = 0; group < 7; group++) {
            if (IsKeyPressed(KEY_ONE + group)) {
                visible[group] = !visible[group];
            }
        }
        if (IsKeyPressed(KEY_BACKSPACE) && route_count > 0) {
            route_count--;
        }
        if (IsKeyPressed(KEY_ENTER)) {
            file = fopen(route_path, "w");
            bool saved = file != NULL;
            if (file) {
                for (int i = 0; i < route_count; i++) {
                    if (fprintf(file, "%.6f,%.6f,%.6f\n", route[i].x, route[i].y,
                        route[i].z) < 0) {
                        saved = false;
                    }
                }
                if (fclose(file) != 0) {
                    saved = false;
                }
            }
            status = saved ? "Saved open route draft (asset units)." : "Route save failed.";
        }
        distance = Clamp(distance * expf(-GetMouseWheelMove() * 0.12f), 2, extent * 3);
        if (!top && IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Vector2 delta = GetMouseDelta();
            yaw -= delta.x * 0.005f;
            pitch = Clamp(pitch + delta.y * 0.005f, 0.04f, 1.55f);
        }
        float speed = distance * GetFrameTime() * 0.7f;
        Vector3 right = top ? (Vector3){1, 0, 0} : (Vector3){cosf(yaw), 0, -sinf(yaw)};
        Vector3 forward = top ? (Vector3){0, 0, -1} : (Vector3){-sinf(yaw), 0, -cosf(yaw)};
        float horizontal = IsKeyDown(KEY_D) - IsKeyDown(KEY_A);
        float vertical = IsKeyDown(KEY_W) - IsKeyDown(KEY_S);
#ifdef MAP_OPTIX
        if (driving) {
            horizontal = vertical = 0;
        }
#endif
        target = Vector3Add(target, Vector3Scale(right, horizontal * speed));
        target = Vector3Add(target, Vector3Scale(forward, vertical * speed));
        target.y += (IsKeyDown(KEY_E) - IsKeyDown(KEY_Q)) * speed;
        camera.target = target;
        camera.position = Vector3Add(target, top ? (Vector3){0, extent * 2, 0} :
            (Vector3){distance * cosf(pitch) * sinf(yaw), distance * sinf(pitch),
                distance * cosf(pitch) * cosf(yaw)});
        camera.up = top ? (Vector3){0, 0, -1} : (Vector3){0, 1, 0};
        camera.projection = top ? CAMERA_ORTHOGRAPHIC : CAMERA_PERSPECTIVE;
        camera.fovy = top ? distance * 2 : 60;
#ifdef MAP_OPTIX
        if (driving) {
            car_distance = Clamp(car_distance-GetMouseWheelMove(), 4, 20);
            Quaternion rotation = {car_frame.rotation[0], car_frame.rotation[1],
                car_frame.rotation[2], car_frame.rotation[3]};
            Vector3 car_position = {car_frame.position[0],car_frame.position[1],car_frame.position[2]};
            Vector3 direction = Vector3RotateByQuaternion((Vector3){0,0,1},rotation);
            Vector3 desired = Vector3Add(Vector3Subtract(car_position,
                Vector3Scale(direction,car_distance)),(Vector3){0,car_distance*0.4f,0});
            // Stable horizon while the rigid chassis pitches and rolls independently.
            static Vector3 chase_position;
            chase_position = chase_ready ? Vector3Lerp(chase_position,desired,
                1-expf(-8*fminf(GetFrameTime(),0.05f))) : desired;
            chase_ready = true;
            camera.position = chase_position;
            camera.target = Vector3Add(car_position,Vector3Add(Vector3Scale(direction,3),
                (Vector3){0,0.4f,0}));
            camera.up = (Vector3){0,1,0};
            camera.projection = CAMERA_PERSPECTIVE;
            camera.fovy = 65;
        }
#endif
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && GetMouseY() > (show_hud ? 260 : 0)) {
            Ray ray = GetScreenToWorldRay(GetMousePosition(), camera);
            picked = -1;
            selected = (RayCollision){.distance = 1e30f};
            for (uint32_t i = 0; i < count; i++) {
                if (!visible[surfaces[i].group] ||
                    !GetRayCollisionBox(ray, surfaces[i].bounds).hit) {
                    continue;
                }
                RayCollision hit = GetRayCollisionMesh(ray, surfaces[i].mesh, MatrixIdentity());
                if (hit.hit && hit.distance < selected.distance) {
                    selected = hit;
                    picked = i;
                }
            }
            if (picked >= 0 && IsKeyDown(KEY_LEFT_SHIFT) && route_count < 4096) {
                route[route_count++] = selected.point;
            }
        }
        Vector3 view_forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
        Vector3 view_right = Vector3Normalize(Vector3CrossProduct(view_forward, camera.up));
        Vector3 view_up = Vector3CrossProduct(view_right, view_forward);
        Vector2 resolution = {(float)GetScreenWidth(), (float)GetScreenHeight()};
        float perspective_scale = camera.projection == CAMERA_ORTHOGRAPHIC ? 0 :
            tanf(camera.fovy * DEG2RAD * 0.5f);
        float fog = camera.projection == CAMERA_ORTHOGRAPHIC ? 0 : 0.00012f;
        int lit = lighting && textured;
        SetShaderValue(texture_shader, lit_location, &lit, SHADER_UNIFORM_INT);
        SetShaderValue(texture_shader, eye_location, &camera.position, SHADER_UNIFORM_VEC3);
        SetShaderValue(texture_shader, fog_location, &fog, SHADER_UNIFORM_FLOAT);
        SetShaderValue(sky_shader, sky_resolution, &resolution, SHADER_UNIFORM_VEC2);
        SetShaderValue(sky_shader, sky_forward, &view_forward, SHADER_UNIFORM_VEC3);
        SetShaderValue(sky_shader, sky_right, &view_right, SHADER_UNIFORM_VEC3);
        SetShaderValue(sky_shader, sky_up, &view_up, SHADER_UNIFORM_VEC3);
        SetShaderValue(sky_shader, sky_scale, &perspective_scale, SHADER_UNIFORM_FLOAT);
        BeginDrawing();
        ClearBackground(SKYBLUE);
        BeginShaderMode(sky_shader);
        DrawRectangle(0, 0, GetScreenWidth(), GetScreenHeight(), WHITE);
        EndShaderMode();
        BeginMode3D(camera);
        rlDisableBackfaceCulling();
        if (wire) {
            rlEnableWireMode();
        }
        for (uint32_t i = 0; i < count; i++) {
            draw_order[i] = i;
            Vector3 center = Vector3Scale(Vector3Add(surfaces[i].bounds.min,
                surfaces[i].bounds.max), 0.5f);
            draw_depth[i] = Vector3DotProduct(Vector3Subtract(center, camera.position),
                Vector3Subtract(camera.target, camera.position));
        }
        // Only blended surfaces need sorting. They draw after opaque/cutout depth is established.
        for (uint32_t i = 1; i < count; i++) {
            unsigned value = draw_order[i], j = i;
            bool blended = textured && alpha_modes[surfaces[value].material] == 2;
            while (j > 0) {
                unsigned prev = draw_order[j-1];
                bool prev_blended = textured && alpha_modes[surfaces[prev].material] == 2;
                if (!(prev_blended && (!blended || draw_depth[value] > draw_depth[prev]))) {
                    break;
                }
                draw_order[j] = prev;
                j--;
            }
            draw_order[j] = value;
        }
        for (uint32_t n = 0; n < count; n++) {
            unsigned i = draw_order[n];
            if (visible[surfaces[i].group]) {
                unsigned id = surfaces[i].material;
                int mode = textured ? alpha_modes[id] : 0;
                SetShaderValue(texture_shader, alpha_location, &mode, SHADER_UNIFORM_INT);
                int ground = textured && (id == 1 || id == 50);
                SetShaderValue(texture_shader, ground_location, &ground, SHADER_UNIFORM_INT);
                if (mode == 2) {
                    rlDisableDepthMask();
                } else {
                    rlEnableDepthMask();
                }
                material.maps[MATERIAL_MAP_DIFFUSE].texture = textured ?
                    textures[bindings[id].base] : default_texture;
                material.maps[MATERIAL_MAP_DIFFUSE].color = textured ?
                    material_colors[id] : colors[surfaces[i].group];
                RacingMaterialDetail values = details[id];
                values.ground = textured ? ground : 0;
                values.normal_strength = bindings[id].normal_strength;
                if (ground) values.detail_strength=0;
                if (bindings[id].flags&2) values.normal_strength=values.detail_strength=0;
                if (!textured) values.normal_strength = values.detail_strength = 0;
                racing_material_values(texture_shader, &values);
                material.maps[MATERIAL_MAP_NORMAL].texture = textures[bindings[id].normal];
                material.maps[MATERIAL_MAP_METALNESS].texture =
                    textures[bindings[id].detail];
                DrawMesh(surfaces[i].mesh, material, MatrixIdentity());
            }
        }
        int ground = 0;
        SetShaderValue(texture_shader, ground_location, &ground, SHADER_UNIFORM_INT);
        rlEnableDepthMask();
        if (wire) {
            rlDisableWireMode();
        }
        rlEnableBackfaceCulling();
#ifdef MAP_OPTIX
        if (car_render.loaded) {
            int car_lit = lighting;
            SetShaderValue(texture_shader,lit_location,&car_lit,SHADER_UNIFORM_INT);
            racing_car_draw(&car_render,&car_frame,texture_shader,alpha_location,camera.position);
        }
#endif
        Vector3 lift = {0, 0.25f, 0};
        for (uint32_t i = 0; diagnostics && i < candidate_count; i++) {
            RoutePoint a = candidate[i], b = candidate[(i + 1) % candidate_count];
            Vector3 point = Vector3Add(a.center, lift);
            Vector3 next = Vector3Add(b.center, lift);
            if (show_route) {
                DrawLine3D(point, next, LIME);
                if (i % 30 == 0) {
                    Vector3 forward = Vector3Normalize(Vector3Subtract(next, point));
                    Vector3 side = {-forward.z, 0, forward.x};
                    Vector3 tip = Vector3Add(point, Vector3Scale(forward, 6));
                    DrawLine3D(tip, Vector3Add(point, Vector3Scale(side, 2)), LIME);
                    DrawLine3D(tip, Vector3Add(point, Vector3Scale(side, -2)), LIME);
                }
            }
            if (show_edges) {
                Color edge_color = (a.flags || b.flags) ? ORANGE : SKYBLUE;
                DrawLine3D(Vector3Add(a.left, lift), Vector3Add(b.left, lift), edge_color);
                DrawLine3D(Vector3Add(a.right, lift), Vector3Add(b.right, lift), edge_color);
                if (a.flags && i % 4 == 0) {
                    DrawLine3D(Vector3Add(a.left, lift), Vector3Add(a.right, lift), ORANGE);
                }
            }
        }
        if (diagnostics && candidate_count && show_route) {
            DrawLine3D(Vector3Add(candidate[0].left, lift),
                Vector3Add(candidate[0].right, lift), MAGENTA);
        }
        if (diagnostics && show_legal) {
            for (uint32_t i = 0; i < legal_count; i++) {
                unsigned next = (i+1)%legal_count;
                DrawLine3D(Vector3Add(legal[i].left, lift),
                    Vector3Add(legal[next].left, lift), PINK);
                DrawLine3D(Vector3Add(legal[i].right, lift),
                    Vector3Add(legal[next].right, lift), PINK);
            }
        }
        if (diagnostics && show_gates) {
            for (uint32_t i = 0; i < gate_count; i++) {
                TrackEdge edge = gates[i].edge;
                DrawLine3D(Vector3Add(edge.left, lift), Vector3Add(edge.right, lift),
                    i ? VIOLET : MAGENTA);
            }
        }
#ifdef MAP_OPTIX
        if (car_render.loaded && gate_count > 1) {
            unsigned next = car_frame.lap_active ? car_frame.next_gate : 0;
            if (next < gate_count) {
                TrackEdge edge = gates[next].edge;
                Vector3 up = {0, 2, 0};
                Color color = car_frame.lap_invalid ? ORANGE : LIME;
                DrawLine3D(edge.left, Vector3Add(edge.left, up), color);
                DrawLine3D(edge.right, Vector3Add(edge.right, up), color);
                DrawLine3D(Vector3Add(edge.left, up), Vector3Add(edge.right, up), color);
            }
        }
        if (have_queries && show_queries && (diagnostics || car_render.loaded)) {
            for (int i = 0; i < 260; i++) {
                float* line = query_lines[i];
                Vector3 start = {line[0], line[1], line[2]}, end = {line[3], line[4], line[5]};
                Color color = line[6] < 0 ? DARKGRAY : (i < 256 ? SKYBLUE : MAGENTA);
                DrawLine3D(start, end, color);
                if (line[6] >= 0) {
                    DrawSphere(end, i < 256 ? 0.15f : 0.3f, color);
                }
            }
        }
#endif
        for (int i = 0; diagnostics && i < route_count; i++) {
            Vector3 point = Vector3Add(route[i], (Vector3){0, 0.5f, 0});
            DrawSphere(point, fmaxf(0.2f, distance * 0.002f), i ? YELLOW : GREEN);
            if (i > 0) {
                DrawLine3D(Vector3Add(route[i - 1], (Vector3){0, 0.5f, 0}), point, YELLOW);
            }
        }
        if (diagnostics && picked >= 0) {
            DrawSphere(selected.point, fmaxf(0.2f, distance * 0.002f), MAGENTA);
            DrawLine3D(selected.point, Vector3Add(selected.point,
                Vector3Scale(selected.normal, fmaxf(2, distance * 0.02f))), MAGENTA);
        }
        EndMode3D();
        if (diagnostics && show_names && show_route) {
            for (uint32_t i = 0; i < landmark_count; i++) {
                Vector3 point = candidate[landmarks[i].index].center;
                if (Vector3DotProduct(Vector3Subtract(point, camera.position),
                    Vector3Subtract(camera.target, camera.position)) <= 0) {
                    continue;
                }
                Vector2 screen = GetWorldToScreen(point, camera);
                if (screen.x < 0 || screen.x > GetScreenWidth() || screen.y < (show_hud ? 260 : 0) ||
                    screen.y > GetScreenHeight()) {
                    continue;
                }
                DrawText(landmarks[i].name, screen.x + 5, screen.y, 14, YELLOW);
            }
        }
        if (diagnostics && show_gates && distance < 350) {
            for (uint32_t i = 0; i < gate_count; i++) {
                Vector3 point = candidate[gates[i].index].center;
                if (Vector3DotProduct(Vector3Subtract(point, camera.position),
                    Vector3Subtract(camera.target, camera.position)) <= 0) {
                    continue;
                }
                Vector2 screen = GetWorldToScreen(point, camera);
                if (screen.x >= 0 && screen.x < GetScreenWidth() && screen.y > (show_hud ? 260 : 0) &&
                    screen.y < GetScreenHeight()) {
                    DrawText(TextFormat("Gate %u", i), screen.x, screen.y, 16, VIOLET);
                }
            }
        }
        if (show_hud) {
            DrawRectangle(0, 0, GetScreenWidth(), 260, Fade(BLACK, 0.85f));
            DrawText(TextFormat("F1: HUD | V: %s | H: diagnostics %s | J: lighting | coordinates in asset units",
                textured ? "textures" : "surface colors", diagnostics ? "on" : "off"),
                12, 8, 18, RAYWHITE);
            const char* movement_help =
                "WASD pan | Q/E height | wheel zoom | T top/orbit | RMB orbit | Home frame | F wire";
#ifdef MAP_OPTIX
            if (driving) {
                movement_help = "W/up throttle | S/down brake | A/D steer | wheel camera zoom | C inspect";
            }
#endif
            DrawText(movement_help, 12, 32, 16, RAYWHITE);
            DrawText("Click pick + normal | Shift-click append route | Backspace undo | Enter save draft",
                12, 54, 16, RAYWHITE);
            int x = 12;
            for (int i = 0; i < 7; i++) {
                const char *label = TextFormat("%d %s%s", i + 1, names[i], visible[i] ? "" : " [off]");
                DrawText(label, x, 78, 15, visible[i] ? colors[i] : GRAY);
                x += MeasureText(label, 15) + 18;
            }
            if (picked >= 0) {
                Surface *s = &surfaces[picked];
                DrawText(TextFormat("%s | node %u primitive %u material %u", s->name, s->node,
                    s->primitive, s->material), 12, 102, 16, WHITE);
                DrawText(TextFormat("Hit %.3f, %.3f, %.3f | normal %.3f, %.3f, %.3f",
                    selected.point.x, selected.point.y, selected.point.z,
                    selected.normal.x, selected.normal.y, selected.normal.z), 12, 124, 16, WHITE);
            }
            DrawText(TextFormat("Route points: %d | %s", route_count, status), 12, 151, 16, YELLOW);
            DrawText("R route/arrows | B pavement edges | N names | [ / ] landmarks | P next flagged edge",
                12, 175, 16, RAYWHITE);
            DrawText("Green: guidance | Blue: pavement extent | Orange: review | Magenta: start candidate",
                12, 198, 16, RAYWHITE);
            DrawText("K task corridor (pink, max 10 units) | G full-pavement gates (violet)",
                12, 220, 16, PINK);
#ifdef MAP_OPTIX
            DrawText("Car LiDAR: automatic 30 Hz | L refresh / map probe | O show/hide rays",
                12, 240, 15, SKYBLUE);
#else
            DrawText("OptiX preview: launch bash ocean/racing/view_map.sh --optix", 12, 240, 15, GRAY);
#endif
        } else {
            DrawText("F1: show HUD", 12, 8, 16, RAYWHITE);
        }
        if (review_point >= 0) {
            RoutePoint point = candidate[review_point];
            DrawText(TextFormat("Review point %d | s %.1f | elevation %.2f | flags %u",
                review_point, point.s, point.center.y, point.flags),
                20, GetScreenHeight() - 58, 16, ORANGE);
        }
        if (top) {
            float units = powf(10, floorf(log10f(distance * 0.15f)));
            int pixels = (int)(units * GetScreenHeight() / (2 * distance));
            DrawLine(20, GetScreenHeight() - 35, 20 + pixels, GetScreenHeight() - 35, WHITE);
            DrawText(TextFormat("%.3g asset units | +X right, -Z up", units),
                20, GetScreenHeight() - 25, 16, WHITE);
        }
#ifdef MAP_OPTIX
        DrawText("C: drive / inspect (pauses car) | WASD/arrows: drive | Space: pause | F5: reset | F6: start line",
            12, GetScreenHeight()-92, 16, WHITE);
        if (car_render.loaded) {
            const char* lap_status = !car_frame.lap_active ? "Cross START to begin" :
                (car_frame.lap_invalid == 1 ? "INVALID: reverse gate crossing" :
                (car_frame.lap_invalid == 2 ? "INVALID: missed gate" :
                (car_frame.lap_invalid == 3 ? "INVALID: impact / rollover" :
                (car_frame.lap_invalid == 4 ? "INVALID: off road" : "ROAD-VALID LAP"))));
            const char* task_reasons[] = {"running", "crash", "off road", "invalid gate/route",
                "wrong way", "stalled", "time limit", "lap complete"};
            DrawText(TextFormat("Reward %.4f | Return %.2f | New progress %.1f m | %s%s",
                car_frame.reward, car_frame.episode_return, car_frame.progress,
                task_reasons[car_frame.task_done], car_frame.task_done ? " - F6 restart" : ""),
                12, GetScreenHeight()-199, 16, car_frame.task_done ? ORANGE : WHITE);
            DrawRectangle(0, GetScreenHeight()-176, GetScreenWidth(), 52, Fade(BLACK, 0.8f));
            DrawText(TextFormat("Lap %.3f s | Last %.3f | Best %.3f | Finished %d | Next %d / %u",
                car_frame.lap_time, car_frame.last_lap, car_frame.best_lap,
                car_frame.completed_laps, car_frame.next_gate, gate_count ? gate_count-1 : 0),
                12, GetScreenHeight()-170, 18, WHITE);
            DrawText(lap_status, 12, GetScreenHeight()-147, 16,
                car_frame.lap_invalid ? ORANGE : LIME);
            DrawRectangle(0,GetScreenHeight()-70,GetScreenWidth(),70,Fade(BLACK,0.8f));
            DrawText(TextFormat("%.1f km/h | gear %d | %.0f RPM | throttle %.0f%% | %s",
                car_frame.speed*3.6f,car_frame.gear,car_frame.rpm,car_frame.throttle*100,
                car_frame.crashed ? "IMPACT / ROLLOVER - F5 reset" :
                (car_frame.task_done ? "EPISODE ENDED - F6 restart" :
                (!driving || car_paused ? "PAUSED" : "DRIVING"))),12,GetScreenHeight()-63,20,WHITE);
            const char* surface[4];
            for (int i=0;i<4;i++) {
                int m=car_frame.material[i];
                surface[i]=!car_frame.contacts[i] ? "air" :
                    (m==53 || m==58 ? "grass" : (m==32 || m==59 ? "gravel" :
                    (m==44 || m==56 ? "curb" : "tarmac")));
            }
            DrawText(TextFormat("FL %s %.2f | FR %s %.2f | RL %s %.2f | RR %s %.2f (grip)",
                surface[0],car_frame.grip[0],surface[1],car_frame.grip[1],
                surface[2],car_frame.grip[2],surface[3],car_frame.grip[3]),
                12,GetScreenHeight()-32,18,LIME);
            int nearest_landmark = 0;
            float best_distance = 1e30f;
            Vector3 position = {car_frame.position[0],car_frame.position[1],car_frame.position[2]};
            for (uint32_t i=0;i<landmark_count;i++) {
                float d = Vector3DistanceSqr(position,candidate[landmarks[i].index].center);
                if (d<best_distance) {
                    nearest_landmark=i;
                    best_distance=d;
                }
            }
            if (landmark_count) {
                DrawText(TextFormat("Near %s",landmarks[nearest_landmark].name),
                    12,GetScreenHeight()-116,16,WHITE);
            }
        }
#endif
        DrawFPS(GetScreenWidth() - 90, GetScreenHeight() - 25);
        EndDrawing();
    }
    for (uint32_t i = 0; i < count; i++) {
        UnloadMesh(surfaces[i].mesh);
    }
    material.maps[MATERIAL_MAP_DIFFUSE].texture = default_texture;
    material.shader = default_shader;
    material.maps[MATERIAL_MAP_NORMAL].texture.id = 0;
    material.maps[MATERIAL_MAP_METALNESS].texture.id = 0;
    UnloadMaterial(material);
    UnloadShader(texture_shader);
    UnloadShader(sky_shader);
    for (uint32_t i = 0; i < texture_count; i++) {
        UnloadTexture(textures[i]);
    }
    free(textures);
    free(material_textures);
    free(material_colors);
    free(details);
    free(bindings);
    free(alpha_modes);
    free(draw_order);
    free(draw_depth);
#ifdef MAP_OPTIX
    if (car_render.loaded) {
        racing_car_unload(&car_render);
    }
    racing_map_query_close();
#endif
    free(surfaces);
    CloseWindow();
    return 0;
}
