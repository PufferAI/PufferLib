#include <time.h>
#include <stdio.h>
#include "drone.h"
#include "puffercpu.h"
#include "rlgl.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

// Match raylib's built-in GIF recorder (SUPPORT_GIF_RECORDING / Ctrl+F12):
// same msf_gif path, full-res rlReadScreenPixels, 10fps, bitDepth 16.
// Own symbols so we don't clash with the copy inside libraylib.
#define msf_gif_begin drone_msf_gif_begin
#define msf_gif_frame drone_msf_gif_frame
#define msf_gif_end drone_msf_gif_end
#define msf_gif_free drone_msf_gif_free
#define msf_gif_begin_to_file drone_msf_gif_begin_to_file
#define msf_gif_frame_to_file drone_msf_gif_frame_to_file
#define msf_gif_end_to_file drone_msf_gif_end_to_file
#define msf_gif_bgra_flag drone_msf_gif_bgra_flag
#define msf_gif_alpha_threshold drone_msf_gif_alpha_threshold
#define MSF_GIF_IMPL
#include "msf_gif.h"

// Raylib defaults used by EndDrawing GIF recording
#define GIF_RECORD_FRAMERATE 10
#define GIF_RECORD_BITRATE 16
#define GIF_CAMERA_ZOOM 1.25f
#define GIF_CAMERA_DISTANCE_DEFAULT 40.0f

// Standalone demo (./build.sh drone --cpu|--debug|--web).
// Fin multitask policy from PR #599 — TAB cycles tasks.
//   ./drone --gif [out.gif]   automated capture schedule

static void setup_task(DroneEnv* env, TaskType task) {
    task_close(env);
    env->task = task;

    if (task == TASK_RACE) {
        RaceConfig* cfg = (RaceConfig*)calloc(1, sizeof(RaceConfig));
        cfg->max_rings = 10;
        cfg->ring_reward = 2.4450236350884f;
        cfg->alpha_dist = 2.8630645575928786f;
        cfg->horizon = 2048;
        env->task_config = cfg;
    } else {
        HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
        cfg->target_dist = 5.0f;
        cfg->alpha_hover = 1.0f;
        cfg->alpha_dist = 0.8120191629018807f;
        cfg->sphere_radius = 4.0f;
        cfg->horizon = 1024;
        env->task_config = cfg;
    }
    task_init(env);
    puf_reset(env);
}

// Same as train: one forward + one ACTION_DT tick.
static void step_once(DroneEnv* env, PufferNet* net, float* observations, float* actions) {
    forward_puffernet(net, observations, actions);
    puf_step(env);
}

// 100 Hz of (forward+tick). Display is rAF. Re-forward every tick.
static void step_display_frame(DroneEnv* env, PufferNet* net, float* observations, float* actions) {
    static double prev = -1.0;
    static double accum = 0.0;
#ifdef __EMSCRIPTEN__
    double now = emscripten_get_now() * 0.001;
#else
    double now = GetTime();
#endif
    int n = 1;
    if (prev >= 0.0) {
        double dt = now - prev;
        if (dt <= 0.0 || dt > 0.25) {
            dt = ACTION_DT;
        }
        accum += dt;
        n = 0;
        while (accum >= ACTION_DT && n < 5) {
            accum -= ACTION_DT;
            n++;
        }
        if (n < 1) {
            n = 1;
        }
    }
    prev = now;
    for (int i = 0; i < n; i++) {
        step_once(env, net, observations, actions);
    }
}

// Fixed-dt stepping for deterministic capture (ignore wall clock jitter).
static void step_fixed(DroneEnv* env, PufferNet* net, float* observations, float* actions,
                       float frame_dt) {
    static double accum = 0.0;
    accum += frame_dt;
    if (accum > 0.25) accum = 0.25;
    while (accum >= ACTION_DT) {
        forward_puffernet(net, observations, actions);
        puf_step(env);
        accum -= ACTION_DT;
    }
}

static bool tab_swap_pressed(void) {
    static bool prev_down = false;
    bool down = IsKeyDown(KEY_TAB);
    bool edge = down && !prev_down;
    prev_down = down;
    return edge;
}

static TaskType next_demo_task(TaskType cur) {
    // Cycle race <-> hover only in interactive mode
    return (cur == TASK_RACE) ? TASK_HOVER : TASK_RACE;
}

#ifdef __EMSCRIPTEN__
typedef struct {
    DroneEnv* env;
    PufferNet* net;
    float* observations;
    float* actions;
} WebRenderArgs;

void emscriptenStep(void* e) {
    WebRenderArgs* args = (WebRenderArgs*)e;
    if (tab_swap_pressed()) {
        setup_task(args->env, next_demo_task(args->env->task));
    }
    step_display_frame(args->env, args->net, args->observations, args->actions);
    puf_render(args->env);
}
#endif

// Capture schedule: race 5s, then each other task 2.5s.
typedef struct {
    TaskType task;
    float seconds;
} GifSegment;

static const GifSegment GIF_SCHEDULE[] = {
    {TASK_RACE, 5.0f},
    {TASK_HOVER, 2.5f},
    {TASK_SPHERE, 2.5f},
    {TASK_CUBE, 2.5f},
    {TASK_FLAG, 2.5f},
};
static const int GIF_NUM_SEGMENTS = (int)(sizeof(GIF_SCHEDULE) / sizeof(GIF_SCHEDULE[0]));

// Apply capture framing once the client exists (make_client is lazy in puf_render).
static void apply_gif_camera(DroneEnv* env) {
    if (env->client == NULL) return;
    // 1.25x zoom = pull camera closer along the look vector
    env->client->camera_distance = GIF_CAMERA_DISTANCE_DEFAULT / GIF_CAMERA_ZOOM;
    env->client->follow_mode = false;
    update_camera_position(env->client, (Vec3){0, 0, 0});
}

// Exact same grab path as raylib EndDrawing() GIF recording.
static unsigned char* capture_frame_raylib(int* out_w, int* out_h) {
    Vector2 scale = GetWindowScaleDPI();
    int w = (int)((float)GetScreenWidth() * scale.x);
    int h = (int)((float)GetScreenHeight() * scale.y);
    *out_w = w;
    *out_h = h;
    return rlReadScreenPixels(w, h);
}

static int run_gif_capture(DroneEnv* env, PufferNet* net, float* observations, float* actions,
                           const char* out_path) {
    // Mirror raylib SUPPORT_GIF_RECORDING defaults (see rcore.c EndDrawing)
    const int fps = GIF_RECORD_FRAMERATE;
    const int centi = 10;  // centiseconds per frame (raylib 5.0 fixed; 10fps)
    const int bit_depth = GIF_RECORD_BITRATE;
    const float frame_dt = 1.0f / (float)fps;

    SetTargetFPS(fps);

    // Open window + apply zoom before we size the GIF
    setup_task(env, GIF_SCHEDULE[0].task);
    puf_render(env);
    apply_gif_camera(env);
    puf_render(env);

    Vector2 scale = GetWindowScaleDPI();
    const int gif_w = (int)((float)GetScreenWidth() * scale.x);
    const int gif_h = (int)((float)GetScreenHeight() * scale.y);

    MsfGifState gif = {0};
    if (!msf_gif_begin(&gif, gif_w, gif_h)) {
        fprintf(stderr, "msf_gif_begin failed (%dx%d)\n", gif_w, gif_h);
        return 1;
    }
    printf("GIF recording %dx%d @ %dfps (raylib-style msf_gif, zoom=%.2fx)\n",
           gif_w, gif_h, fps, GIF_CAMERA_ZOOM);

    int total_frames = 0;
    for (int s = 0; s < GIF_NUM_SEGMENTS; s++) {
        const GifSegment* seg = &GIF_SCHEDULE[s];
        printf("GIF segment %d/%d: task=%s (%.1fs)\n", s + 1, GIF_NUM_SEGMENTS,
               task_name(seg->task), seg->seconds);
        // setup_task before any render — task_horizon() needs task_config
        if (s > 0) setup_task(env, seg->task);
        apply_gif_camera(env);  // task reset should not lose framing

        int frames = (int)(seg->seconds * fps + 0.5f);
        for (int f = 0; f < frames; f++) {
            step_fixed(env, net, observations, actions, frame_dt);
            puf_render(env);

            int w = 0, h = 0;
            unsigned char* pixels = capture_frame_raylib(&w, &h);
            if (pixels == NULL || w != gif_w || h != gif_h) {
                fprintf(stderr, "rlReadScreenPixels failed at frame %d (%dx%d, expected %dx%d)\n",
                        total_frames, w, h, gif_w, gif_h);
                free(pixels);
                msf_gif_end(&gif);
                return 1;
            }
            // raylib: msf_gif_frame(&gifState, screenData, 10, 16, width*4)
            if (!msf_gif_frame(&gif, pixels, centi, bit_depth, w * 4)) {
                fprintf(stderr, "msf_gif_frame failed at frame %d\n", total_frames);
                free(pixels);
                msf_gif_end(&gif);
                return 1;
            }
            free(pixels);
            total_frames++;
            if ((f + 1) % fps == 0) {
                printf("  ... %s %.1fs\n", task_name(seg->task), (f + 1) * frame_dt);
                fflush(stdout);
            }
        }
    }

    MsfGifResult result = msf_gif_end(&gif);
    if (!result.data || result.dataSize == 0) {
        fprintf(stderr, "msf_gif_end failed\n");
        msf_gif_free(result);
        return 1;
    }

    FILE* fp = fopen(out_path, "wb");
    if (!fp) {
        perror(out_path);
        msf_gif_free(result);
        return 1;
    }
    fwrite(result.data, 1, result.dataSize, fp);
    fclose(fp);
    msf_gif_free(result);

    printf("Wrote %s (%d frames @ %dfps, %dx%d, %.2fx zoom)\n",
           out_path, total_frames, fps, gif_w, gif_h, GIF_CAMERA_ZOOM);
    return 0;
}

void demo(int gif_mode, const char* gif_path) {
    srand(time(NULL));

    // Match resources/drone/drone_weights.bin from PR #599 (obs=21, H=64, L=2)
    Weights* weights = load_weights("resources/drone/drone_weights.bin");
    if (!weights) {
        fprintf(stderr, "failed to load resources/drone/drone_weights.bin\n");
        exit(1);
    }

    int num_agents = 64;
    int logit_sizes[4] = {1, 1, 1, 1};
    PufferNet* net = make_puffernet(weights, num_agents, OBS_SIZE, 64, 2, logit_sizes, 4);

    DroneEnv env = {0};
    env.num_agents = num_agents;
    env.rng = 1;
    env.dr = 0.05f;
    env.integrator = 0;
    env.alpha_vel = 0.0f;
    env.alpha_omega = 0.0f;
    env.alpha_action = 0.0f;

    // Physics + drones allocated once; setup_task owns task config/state
    init(&env);

    float* observations = (float*)calloc(num_agents * OBS_SIZE, sizeof(float));
    float* actions = (float*)calloc(num_agents * NUM_ATNS, sizeof(float));
    float* rewards = (float*)calloc(num_agents, sizeof(float));
    float* terminals = (float*)calloc(num_agents, sizeof(float));
    for (int i = 0; i < num_agents; i++) {
        env.agents[i].observations = observations + i * OBS_SIZE;
        env.agents[i].actions = actions + i * NUM_ATNS;
        env.agents[i].rewards = rewards + i;
        env.agents[i].terminals = terminals + i;
        env.agents[i].action_mask = NULL;
        env.agents[i].policy = 0;
    }

    if (gif_mode) {
        int rc = run_gif_capture(&env, net, (float*)observations, actions, gif_path);
        puf_close(&env);
        free_puffernet(net);
        free(weights);
        free(observations);
        free(actions);
        free(rewards);
        free(terminals);
        if (rc != 0) exit(rc);
        return;
    }

    setup_task(&env, TASK_RACE);
    puf_render(&env);
#ifndef __EMSCRIPTEN__
    SetTargetFPS(60);
#endif

#ifdef __EMSCRIPTEN__
    static WebRenderArgs args;
    args.env = &env;
    args.net = net;
    args.observations = (float*)observations;
    args.actions = actions;
    emscripten_set_main_loop_arg(emscriptenStep, &args, 0, true);
#else
    while (!WindowShouldClose()) {
        if (tab_swap_pressed()) {
            setup_task(&env, next_demo_task(env.task));
        }
        step_display_frame(&env, net, (float*)observations, actions);
        puf_render(&env);
    }

    puf_close(&env);
    free_puffernet(net);
    free(weights);
    free(observations);
    free(actions);
    free(rewards);
    free(terminals);
#endif
}

int main(int argc, char** argv) {
    int gif_mode = 0;
    const char* gif_path = "captures/drone_multitask.gif";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gif") == 0) {
            gif_mode = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                gif_path = argv[++i];
            }
        }
    }
    demo(gif_mode, gif_path);
    return 0;
}
