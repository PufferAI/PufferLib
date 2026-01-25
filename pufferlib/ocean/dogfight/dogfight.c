// Standalone C demo for Dogfight environment
// Build: ./scripts/build_ocean.sh dogfight local
// Run:   ./dogfight
//
// Controls (hold LEFT_SHIFT):
//   W/S     - Pitch down/up (nose down/up)
//   A/D     - Roll left/right
//   Q/E     - Yaw left/right (rudder)
//   R/Up    - Throttle up
//   F/Down  - Throttle down
//   Space   - Fire
//   ESC     - Quit

#include <time.h>
#include "dogfight.h"
#include "puffernet.h"

void demo() {
    // TODO: Load trained weights when available
    // Weights* weights = load_weights("resources/dogfight/dogfight_weights.bin", SIZE);
    // LinearContLSTM* net = make_linearcontlstm(weights, ...);

    int obs_scheme = OBS_MOMENTUM;  // Default: 15 observations
    int obs_size = OBS_SIZES[obs_scheme];

    Dogfight env = {
        .max_steps = 3000,
    };

    // Allocate buffers
    env.observations = (float*)calloc(obs_size, sizeof(float));
    env.actions = (float*)calloc(5, sizeof(float));  // throttle, elevator, aileron, rudder, trigger
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    RewardConfig rcfg = {
        .aim_scale = 0.05f,
        .closing_scale = 0.003f,
        .neg_g = 0.02f,
        .speed_min = 50.0f,
    };

    // curriculum_enabled=1, curriculum_randomize=1 for variety
    init(&env, obs_scheme, &rcfg, 1, 1, 0);
    c_reset(&env);
    c_render(&env);  // Initialize window (lazy init inside)

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        // ============================================
        // HUMAN CONTROL (hold LEFT_SHIFT)
        // ============================================
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Initialize to neutral
            env.actions[0] = 0.0f;   // throttle (0 = 50% cruise)
            env.actions[1] = 0.0f;   // elevator
            env.actions[2] = 0.0f;   // ailerons
            env.actions[3] = 0.0f;   // rudder
            env.actions[4] = -1.0f;  // trigger (not firing)

            // Pitch: elevator (+1 = push forward = nose DOWN)
            if (IsKeyDown(KEY_W)) env.actions[1] = 1.0f;   // Nose down
            if (IsKeyDown(KEY_S)) env.actions[1] = -1.0f;  // Nose up

            // Roll: ailerons (+1 = roll RIGHT)
            if (IsKeyDown(KEY_A)) env.actions[2] = -1.0f;  // Roll left
            if (IsKeyDown(KEY_D)) env.actions[2] = 1.0f;   // Roll right

            // Rudder: (+1 = yaw LEFT)
            if (IsKeyDown(KEY_Q)) env.actions[3] = 1.0f;   // Yaw left
            if (IsKeyDown(KEY_E)) env.actions[3] = -1.0f;  // Yaw right

            // Throttle: actions[0] in [-1, 1] maps to [0%, 100%]
            if (IsKeyDown(KEY_R) || IsKeyDown(KEY_UP)) env.actions[0] = 1.0f;     // Full throttle
            if (IsKeyDown(KEY_F) || IsKeyDown(KEY_DOWN)) env.actions[0] = -1.0f;  // Idle

            // Fire
            if (IsKeyDown(KEY_SPACE)) env.actions[4] = 1.0f;
        } else {
            // ============================================
            // AI CONTROL (when SHIFT not held)
            // ============================================
            // TODO: Use neural network when weights available
            // forward_linearcontlstm(net, env.observations, env.actions);

            // For now: simple cruise autopilot
            env.actions[0] = 0.0f;   // 50% throttle (neutral maps to 50%)
            env.actions[1] = 0.0f;   // neutral elevator
            env.actions[2] = 0.0f;   // neutral ailerons
            env.actions[3] = 0.0f;   // neutral rudder
            env.actions[4] = -1.0f;  // don't fire
        }

        c_step(&env);
        c_render(&env);
    }

    c_close(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
}

int main() {
    demo();
    return 0;
}
