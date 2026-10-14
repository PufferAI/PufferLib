#ifndef PUFFER_ROBOT_ARM_GPU_CU
#define PUFFER_ROBOT_ARM_GPU_CU

#define PUF_BACKEND PUF_GPU

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Keep the entry-file observation declaration for native build-script discovery.
#if defined(from_float) && !defined(PRECISION_FLOAT)
typedef precision_t obs_t;
#else
typedef float obs_t;
#endif

#include "robot_arm_cuda.cuh"

static int g_ra_no_timeout;
static int g_ra_stack;
static int g_ra_basketball;
static const char* g_ra_model_glb = "resources/robot_arm/franka_panda.glb";
static RaRenderHost g_ra_render_host;

static struct {
    Env* envs;
    int n;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    cudaStream_t stream;
} g_gpu;

static int ra_flag(Dict* kwargs, const char* key) {
    DictItem* item = dict_find(kwargs, key);
    return item != NULL && item->value != 0.0;
}

static void ra_fill(Env* env, unsigned int rng) {
    memset(env, 0, sizeof(*env));
    env->num_agents = 1;
    env->rng = rng ? rng : 1u;
    env->world.state.rng = env->rng;
    env->world.state.no_timeout = g_ra_no_timeout;
    env->world.state.stack_mode = g_ra_stack;
    env->world.state.basketball_mode = g_ra_basketball;
    ra_reset(&env->world.state);
    ra_rbrst(&env->world.rigid, ra_topo(&env->world.state));
}

Env* puf_vec_create(int n, Dict* env_kwargs, obs_t* observations, float* actions, float* rewards,
    float* terminals) {
    g_ra_no_timeout = ra_flag(env_kwargs, "no_timeout");
    g_ra_stack = ra_flag(env_kwargs, "stack");
    g_ra_basketball = ra_flag(env_kwargs, "basketball");
    assert(!(g_ra_stack && g_ra_basketball));
    DictItem* model = dict_find(env_kwargs, "model_glb");
    if (model != NULL && model->str != NULL && model->str[0] != '\0'
        && strcmp(model->str, "None") != 0) {
        g_ra_model_glb = model->str;
    }
    g_ra_render_host.model_glb = g_ra_model_glb;
    g_ra_render_host.camera_distance = g_ra_basketball ? 2.35f : 1.55f;
    g_ra_render_host.camera_yaw = 0.78f;
    g_ra_render_host.camera_pitch = 0.48f;

    Env* host_envs = (Env*)calloc(n, sizeof(Env));
    for (int i = 0; i < n; i++) {
        ra_fill(&host_envs[i], i + 1);
    }
    Env* envs = NULL;
    assert(cudaMalloc((void**)&envs, n * sizeof(Env)) == cudaSuccess);
    assert(cudaMemcpy(envs, host_envs, n * sizeof(Env), cudaMemcpyHostToDevice) == cudaSuccess);
    free(host_envs);
    g_gpu.envs = envs;
    g_gpu.n = n;
    g_gpu.observations = observations;
    g_gpu.actions = actions;
    g_gpu.rewards = rewards;
    g_gpu.terminals = terminals;
    g_gpu.stream = 0;
    return envs;
}

void puf_bind_stream(cudaStream_t stream) {
    g_gpu.stream = stream;
}

void puf_init(Env* env, Dict*) {
    env->num_agents = 1;
}

void puf_reset(Env*) {
    ra_kinit<<<(g_gpu.n + RA_CUDA_BLOCK_SIZE - 1) / RA_CUDA_BLOCK_SIZE, RA_CUDA_BLOCK_SIZE>>>(
        g_gpu.envs, g_gpu.observations, g_gpu.rewards, g_gpu.terminals, g_gpu.n);
    assert(cudaGetLastError() == cudaSuccess);
}

void puf_step(Env*) {
    dim3 grid((g_gpu.n + RA_CUDA_BLOCK_SIZE - 1) / RA_CUDA_BLOCK_SIZE);
    dim3 block(RA_CUDA_BLOCK_SIZE);
    ra_kbegin<<<grid, block, 0, g_gpu.stream>>>(g_gpu.envs, 0, g_gpu.n, g_gpu.actions);
    assert(cudaGetLastError() == cudaSuccess);
    ra_kphys<<<grid, block, 0, g_gpu.stream>>>(g_gpu.envs, 0, g_gpu.n);
    assert(cudaGetLastError() == cudaSuccess);
    ra_kfin<<<grid, block, 0, g_gpu.stream>>>(
        g_gpu.envs, 0, g_gpu.n, g_gpu.observations, g_gpu.rewards, g_gpu.terminals);
    assert(cudaGetLastError() == cudaSuccess);
}

void puf_close(Env*) {
    if (g_ra_render_host.renderer.loaded) {
        UnloadModel(g_ra_render_host.renderer.arm);
    }
    if (g_ra_render_host.renderer.attempted) {
        UnloadModel(g_ra_render_host.renderer.cube);
        UnloadModel(g_ra_render_host.renderer.sphere);
        UnloadModel(g_ra_render_host.renderer.cylinder);
        UnloadShader(g_ra_render_host.renderer.skin_shader);
        UnloadShader(g_ra_render_host.renderer.depth_shader);
        rlUnloadFramebuffer(g_ra_render_host.renderer.shadow.id);
    }
    memset(&g_ra_render_host.renderer, 0, sizeof(g_ra_render_host.renderer));
    if (IsWindowReady()) {
        CloseWindow();
    }
    cudaFree(g_gpu.envs);
    g_gpu.envs = NULL;
}

void puf_render(Env*) {
    if (g_gpu.stream) {
        cudaStreamSynchronize(g_gpu.stream);
    }
    RaState state;
    assert(cudaMemcpy(&state, &g_gpu.envs->world.state, sizeof(RaState), cudaMemcpyDeviceToHost)
        == cudaSuccess);
    RaPose links[RA_LINKS];
    ra_fk(state.q, state.gripper_width, links, NULL, NULL, &state.end_effector);
    static int screenshot_taken = 0;
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(1180, 760, "PufferLib - CUDA Robot Arm Manipulation");
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_R)) {
        g_ra_render_host.reset_requested = 1;
    }

    RaRenderer* renderer = &g_ra_render_host.renderer;
    if (!renderer->attempted) {
        renderer->attempted = 1;
        if (!FileExists(g_ra_render_host.model_glb)) {
            fprintf(stderr, "Robot arm model not found: %s\n", g_ra_render_host.model_glb);
        } else {
            renderer->arm = LoadModel(g_ra_render_host.model_glb);
        }
        if (renderer->arm.meshCount <= 0) {
            if (FileExists(g_ra_render_host.model_glb)) {
                fprintf(stderr, "Robot arm GLB failed to load: %s\n", g_ra_render_host.model_glb);
            }
        } else {
            float home[RA_DOF] = RA_MODEL_HOME;
            RaPose bind[RA_LINKS];
            ra_fk(home, 0.08f, bind, NULL, NULL, NULL);
            for (int link = 0; link < RA_LINKS; ++link) {
                renderer->inverse_bind[link] = MatrixInvert(ra_matrix(bind[link]));
            }
            renderer->loaded = 1;
            if (renderer->arm.meshCount != RA_EXPECTED_MESHES) {
                fprintf(stderr,
                    "Robot arm GLB has %d meshes; expected %d. "
                    "Rendering without articulated group map.\n",
                    renderer->arm.meshCount, RA_EXPECTED_MESHES);
            }
        }
    }

    if (renderer->shadow.id == 0) {
        renderer->cube = LoadModelFromMesh(GenMeshCube(1, 1, 1));
        renderer->sphere = LoadModelFromMesh(GenMeshSphere(1, 32, 32));
        renderer->cylinder = LoadModelFromMesh(GenMeshCylinder(1, 1, 24));
        renderer->skin_shader = LoadShader(
            "resources/robot_arm/panda_lighting.vs", "resources/robot_arm/panda_lighting.fs");
        renderer->depth_shader =
            LoadShader("resources/robot_arm/panda_depth.vs", "resources/robot_arm/panda_depth.fs");
        assert(renderer->skin_shader.id != rlGetShaderIdDefault());
        assert(renderer->depth_shader.id != rlGetShaderIdDefault());
        renderer->light_direction_loc = GetShaderLocation(renderer->skin_shader, "lightDirection");
        renderer->view_position_loc = GetShaderLocation(renderer->skin_shader, "viewPosition");
        renderer->light_matrix_loc =
            GetShaderLocation(renderer->skin_shader, "lightViewProjection");
        renderer->surface_kind_loc = GetShaderLocation(renderer->skin_shader, "surfaceKind");
        renderer->skin_shader.locs[SHADER_LOC_MAP_METALNESS] =
            GetShaderLocation(renderer->skin_shader, "shadowMap");
        renderer->shadow.id = rlLoadFramebuffer();
        renderer->shadow.texture.width = renderer->shadow.texture.height = RA_SHADOW_SIZE;
        renderer->shadow.depth =
            (Texture2D){rlLoadTextureDepth(RA_SHADOW_SIZE, RA_SHADOW_SIZE, false), RA_SHADOW_SIZE,
                RA_SHADOW_SIZE, 1, PIXELFORMAT_UNCOMPRESSED_R32};
        rlFramebufferAttach(renderer->shadow.id, renderer->shadow.depth.id, RL_ATTACHMENT_DEPTH,
            RL_ATTACHMENT_TEXTURE2D, 0);
        assert(rlFramebufferComplete(renderer->shadow.id));
        rlDisableFramebuffer();
        SetTextureFilter(renderer->shadow.depth, TEXTURE_FILTER_POINT);
        SetTextureWrap(renderer->shadow.depth, TEXTURE_WRAP_CLAMP);
    }

    if (!g_ra_render_host.camera_initialized) {
        g_ra_render_host.camera.target =
            state.basketball_mode ? (Vector3){0.75f, 0.38f, -0.16f} : (Vector3){0.23f, 0.29f, 0.0f};
        g_ra_render_host.camera.up = (Vector3){0, 1, 0};
        g_ra_render_host.camera.fovy = 42.0f;
        g_ra_render_host.camera.projection = CAMERA_PERSPECTIVE;
        g_ra_render_host.camera_initialized = 1;
    }
    float dt = GetFrameTime();
    Vector2 mouse_delta = GetMouseDelta();
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        g_ra_render_host.camera_yaw += mouse_delta.x * 0.006f;
        g_ra_render_host.camera_pitch += mouse_delta.y * 0.006f;
    }
    if (IsKeyDown(KEY_LEFT)) {
        g_ra_render_host.camera_yaw -= 0.9f * dt;
    }
    if (IsKeyDown(KEY_RIGHT)) {
        g_ra_render_host.camera_yaw += 0.9f * dt;
    }
    if (IsKeyDown(KEY_UP)) {
        g_ra_render_host.camera_pitch += 0.65f * dt;
    }
    if (IsKeyDown(KEY_DOWN)) {
        g_ra_render_host.camera_pitch -= 0.65f * dt;
    }
    g_ra_render_host.camera_pitch = ra_clamp(g_ra_render_host.camera_pitch, 0.16f, 1.15f);
    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        float sin_yaw = sinf(g_ra_render_host.camera_yaw);
        float cos_yaw = cosf(g_ra_render_host.camera_yaw);
        float sin_pitch = sinf(g_ra_render_host.camera_pitch);
        float cos_pitch = cosf(g_ra_render_host.camera_pitch);
        Vector3 right = (Vector3){sin_yaw, 0.0f, -cos_yaw};
        Vector3 up = (Vector3){-cos_yaw * sin_pitch, cos_pitch, -sin_yaw * sin_pitch};
        float pan_scale = 0.001f * g_ra_render_host.camera_distance;
        g_ra_render_host.camera.target = Vector3Add(g_ra_render_host.camera.target,
            Vector3Add(Vector3Scale(right, -mouse_delta.x * pan_scale),
                Vector3Scale(up, mouse_delta.y * pan_scale)));
    }
    g_ra_render_host.camera_distance =
        ra_clamp(g_ra_render_host.camera_distance - 0.12f * GetMouseWheelMove(), 0.85f, 3.2f);
    if (IsKeyPressed(KEY_HOME)) {
        g_ra_render_host.camera.target =
            state.basketball_mode ? (Vector3){0.75f, 0.38f, -0.16f} : (Vector3){0.23f, 0.29f, 0.0f};
        g_ra_render_host.camera_distance = state.basketball_mode ? 2.35f : 1.55f;
        g_ra_render_host.camera_yaw = 0.78f;
        g_ra_render_host.camera_pitch = 0.48f;
    }
    float horizontal = g_ra_render_host.camera_distance * cosf(g_ra_render_host.camera_pitch);
    g_ra_render_host.camera.position = (Vector3){
        g_ra_render_host.camera.target.x + horizontal * cosf(g_ra_render_host.camera_yaw),
        g_ra_render_host.camera.target.y
            + g_ra_render_host.camera_distance * sinf(g_ra_render_host.camera_pitch),
        g_ra_render_host.camera.target.z + horizontal * sinf(g_ra_render_host.camera_yaw),
    };

    RaDrawList scene = {};
    ra_box(&scene, renderer,
        (Vector3){RA_TABLE_CENTER_X, RA_TABLE_TOP - 0.5f * RA_TABLE_THICKNESS, 0},
        (Vector3){RA_TABLE_SIZE_X, RA_TABLE_THICKNESS, RA_TABLE_SIZE_Z}, (Color){68, 78, 89, 255},
        1);
    if (renderer->loaded) {
        const signed char mesh_link[RA_EXPECTED_MESHES] = {-1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9};
        int articulated = renderer->arm.meshCount == RA_EXPECTED_MESHES;
        for (int mesh = 0; mesh < renderer->arm.meshCount; ++mesh) {
            assert(scene.count < RA_DRAW_ITEMS);
            int link = articulated ? mesh_link[mesh] : -1;
            Matrix transform = link >= 0
                ? MatrixMultiply(renderer->inverse_bind[link], ra_matrix(links[link]))
                : MatrixIdentity();
            int material = renderer->arm.meshMaterial[mesh];
            assert(material >= 0 && material < renderer->arm.materialCount);
            scene.items[scene.count++] = (RaDrawItem){&renderer->arm.meshes[mesh],
                &renderer->arm.materials[material], transform, WHITE, 0};
        }
    }
    if (state.basketball_mode) {
        ra_box(&scene, renderer,
            (Vector3){RA_HOOP_CENTER_X, RA_BACKBOARD_CENTER_Y, RA_BACKBOARD_CENTER_Z},
            (Vector3){2 * RA_BACKBOARD_HALF_X, 2 * RA_BACKBOARD_HALF_Y, 2 * RA_BACKBOARD_HALF_Z},
            (Color){226, 232, 238, 255}, 0);
        ra_cylinder(&scene, renderer,
            (Vector3){RA_HOOP_CENTER_X, RA_TABLE_TOP, RA_BACKBOARD_CENTER_Z - 0.025f},
            (Vector3){RA_HOOP_CENTER_X, RA_BACKBOARD_CENTER_Y, RA_BACKBOARD_CENTER_Z - 0.025f},
            0.012f, (Color){72, 82, 94, 255});
        for (int segment = 0; segment < 64; ++segment) {
            float a = 2 * PI * segment / 64;
            float b = 2 * PI * (segment + 1) / 64;
            ra_cylinder(&scene, renderer,
                (Vector3){RA_HOOP_CENTER_X + RA_RIM_MAJOR_RADIUS * cosf(a), RA_HOOP_CENTER_Y,
                    RA_HOOP_CENTER_Z + RA_RIM_MAJOR_RADIUS * sinf(a)},
                (Vector3){RA_HOOP_CENTER_X + RA_RIM_MAJOR_RADIUS * cosf(b), RA_HOOP_CENTER_Y,
                    RA_HOOP_CENTER_Z + RA_RIM_MAJOR_RADIUS * sinf(b)},
                RA_RIM_TUBE_RADIUS, (Color){235, 91, 31, 255});
        }
        for (int strand = 0; strand < 12; ++strand) {
            float angle = 2 * PI * strand / 12;
            ra_cylinder(&scene, renderer,
                (Vector3){RA_HOOP_CENTER_X + RA_RIM_MAJOR_RADIUS * cosf(angle), RA_HOOP_CENTER_Y,
                    RA_HOOP_CENTER_Z + RA_RIM_MAJOR_RADIUS * sinf(angle)},
                (Vector3){RA_HOOP_CENTER_X + 0.038f * cosf(angle + 0.20f), RA_HOOP_CENTER_Y - 0.11f,
                    RA_HOOP_CENTER_Z + 0.038f * sinf(angle + 0.20f)},
                0.0008f, (Color){235, 235, 225, 255});
        }
    }
    for (int object = 0; object < (state.stack_mode ? 2 : 1); ++object) {
        RaPose pose = object == 0 ? (RaPose){state.cube_position, state.cube_rotation}
                                  : (RaPose){state.base_cube_position, state.base_cube_rotation};
        Color color = object ? (Color){224, 85, 91, 255}
            : state.grasped  ? (Color){245, 181, 58, 255}
                             : (Color){64, 146, 224, 255};
        float scale = 2 * RA_CUBE_HALF;
        Model* model = &renderer->cube;
        if (state.basketball_mode) {
            scale = RA_BALL_RADIUS;
            model = &renderer->sphere;
            color = state.grasped ? (Color){245, 181, 58, 255} : (Color){225, 112, 31, 255};
        }
        ra_mesh(&scene, model, MatrixMultiply(MatrixScale(scale, scale, scale), ra_matrix(pose)),
            color, 0);
    }
    Vector3 light = Vector3Normalize((Vector3){-0.42f, 0.82f, -0.38f});
    Camera3D light_camera = {};
    light_camera.target = (Vector3){state.basketball_mode ? 0.65f : 0.20f, 0.30f, 0};
    light_camera.position = Vector3Add(light_camera.target, Vector3Scale(light, 6));
    light_camera.up = (Vector3){0, 1, 0};
    light_camera.fovy = 4.4f;
    light_camera.projection = CAMERA_ORTHOGRAPHIC;
    SetShaderValue(
        renderer->skin_shader, renderer->light_direction_loc, &light, SHADER_UNIFORM_VEC3);
    SetShaderValue(renderer->skin_shader, renderer->view_position_loc,
        &g_ra_render_host.camera.position, SHADER_UNIFORM_VEC3);
    BeginDrawing();
    ClearBackground((Color){24, 30, 39, 255});
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 0) {
            BeginTextureMode(renderer->shadow);
            ClearBackground(WHITE);
            rlSetClipPlanes(0.1, 12.0);
            BeginMode3D(light_camera);
            Matrix light_matrix = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection());
            SetShaderValueMatrix(renderer->skin_shader, renderer->light_matrix_loc, light_matrix);
        } else {
            rlSetClipPlanes(RL_CULL_DISTANCE_NEAR, RL_CULL_DISTANCE_FAR);
            BeginMode3D(g_ra_render_host.camera);
        }
        for (int item = 0; item < scene.count; ++item) {
            RaDrawItem* draw = &scene.items[item];
            Material material = *draw->material;
            material.shader = pass == 0 ? renderer->depth_shader : renderer->skin_shader;
            Color saved_color = material.maps[MATERIAL_MAP_ALBEDO].color;
            Texture2D saved_texture = material.maps[MATERIAL_MAP_METALNESS].texture;
            material.maps[MATERIAL_MAP_ALBEDO].color =
                (Color){(unsigned char)(saved_color.r * draw->color.r / 255),
                    (unsigned char)(saved_color.g * draw->color.g / 255),
                    (unsigned char)(saved_color.b * draw->color.b / 255), saved_color.a};
            if (pass == 1) {
                material.maps[MATERIAL_MAP_METALNESS].texture = renderer->shadow.depth;
                SetShaderValue(renderer->skin_shader, renderer->surface_kind_loc,
                    &draw->surface_kind, SHADER_UNIFORM_INT);
            }
            DrawMesh(*draw->mesh, material, draw->transform);
            material.maps[MATERIAL_MAP_ALBEDO].color = saved_color;
            material.maps[MATERIAL_MAP_METALNESS].texture = saved_texture;
        }
        EndMode3D();
        if (pass == 0) {
            EndTextureMode();
        }
    }
    BeginMode3D(g_ra_render_host.camera);
    if (state.basketball_mode) {
        float quality = ra_btq(state.cube_position, state.cube_velocity);
        unsigned char red = (unsigned char)(235.0f - 175.0f * quality);
        unsigned char green = (unsigned char)(70.0f + 175.0f * quality);
        Color path_color = (Color){red, green, 55, 255};
        RaVec3 position = state.cube_position;
        RaVec3 velocity = state.cube_velocity;
        RaVec3 visual_apex = position;
        for (int frame = 0; frame < 120; ++frame) {
            RaVec3 previous = position;
            for (int substep = 0; substep < RA_SUBSTEPS; ++substep) {
                velocity = ra_bvel(velocity, RA_PHYSICS_DT);
                position = ra_add(position, ra_scale(velocity, RA_PHYSICS_DT));
            }
            if (position.y > visual_apex.y) {
                visual_apex = position;
            }
            DrawLine3D(ra_vector3(previous), ra_vector3(position), path_color);
            if (position.y <= RA_TABLE_TOP + RA_BALL_RADIUS && frame > 1) {
                break;
            }
        }
        DrawSphere(ra_vector3(visual_apex), 0.012f, (Color){255, 215, 70, 255});

        RaVec3 crossing;
        if (ra_bxing(state.cube_position, state.cube_velocity, &crossing, NULL, NULL)) {
            RaVec3 hoop = ra_hoop();
            DrawSphere(ra_vector3(crossing), 0.016f, path_color);
            DrawLine3D(ra_vector3(crossing), ra_vector3(hoop), path_color);
        }
    } else if (!state.stack_mode) {
        Vector3 target = ra_vector3(state.target_position);
        DrawCylinderEx((Vector3){target.x, RA_TABLE_TOP + 0.002f, target.z},
            (Vector3){target.x, RA_TABLE_TOP + 0.004f, target.z}, 0.066f, 0.066f, 64,
            (Color){31, 205, 150, 140});
    }
    DrawSphere(ra_vector3(state.end_effector), 0.012f, (Color){255, 218, 80, 220});
    EndMode3D();
    if (state.basketball_mode) {
        DrawText(TextFormat("Baskets: %d", state.baskets), 24, 22, 28, (Color){245, 245, 240, 255});
    }
    EndDrawing();

    const char* screenshot = getenv("PUFFER_ROBOT_ARM_SCREENSHOT");
    if (!screenshot_taken && screenshot != NULL && screenshot[0] != '\0') {
        TakeScreenshot(screenshot);
        screenshot_taken = 1;
    }
    if (!g_ra_render_host.reset_requested) {
        return;
    }
    g_ra_render_host.reset_requested = 0;
    Env host_env;
    ra_fill(&host_env, state.rng);
    assert(cudaMemcpy(g_gpu.envs, &host_env, sizeof(Env), cudaMemcpyHostToDevice) == cudaSuccess);
}

#endif
