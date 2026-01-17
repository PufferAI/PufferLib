/**
 * autopilot.h - Target aircraft flight maneuvers
 *
 * Provides autopilot modes for opponent aircraft during training.
 * Can be set randomly at reset or forced via API for curriculum learning.
 */

#ifndef AUTOPILOT_H
#define AUTOPILOT_H

#include "flightlib.h"
#include <math.h>

// Autopilot mode enumeration
typedef enum {
    AP_STRAIGHT = 0,  // Fly straight (current/default behavior)
    AP_LEVEL,         // Level flight with PD on vz
    AP_TURN_LEFT,     // Coordinated left turn
    AP_TURN_RIGHT,    // Coordinated right turn
    AP_CLIMB,         // Constant climb rate
    AP_DESCEND,       // Constant descent rate
    AP_HARD_TURN_LEFT,   // Aggressive 70° left turn
    AP_HARD_TURN_RIGHT,  // Aggressive 70° right turn
    AP_WEAVE,            // Sine wave jinking (S-turns)
    AP_EVASIVE,          // Break turn when threat behind
    AP_RANDOM,        // Random mode selection at reset
    AP_COUNT
} AutopilotMode;

// PID gains (from test_flight.py)
#define AP_LEVEL_KP       0.001f
#define AP_LEVEL_KD       0.001f
#define AP_TURN_ELEV_KP  -0.05f
#define AP_TURN_ELEV_KD   0.005f
#define AP_TURN_ROLL_KP  -2.0f
#define AP_TURN_ROLL_KD  -0.1f

// Default parameters
#define AP_DEFAULT_THROTTLE   1.0f
#define AP_DEFAULT_BANK_DEG   30.0f   // Base gentle turns
#define AP_DEFAULT_CLIMB_RATE 5.0f

// Stage-specific bank angles (curriculum progression)
#define AP_STAGE4_BANK_DEG    30.0f   // MANEUVERING - gentle 30° turns
#define AP_STAGE5_BANK_DEG    45.0f   // FULL_RANDOM - medium 45° turns
#define AP_STAGE6_BANK_DEG    60.0f   // HARD_MANEUVERING - steep 60° turns
#define AP_HARD_BANK_DEG      70.0f   // EVASIVE - aggressive 70° turns
#define AP_WEAVE_AMPLITUDE    0.6f    // ~35° bank amplitude (radians)
#define AP_WEAVE_PERIOD       3.0f    // 3 second full cycle

// Autopilot state for a plane
typedef struct {
    AutopilotMode mode;
    int randomize_on_reset;  // If true, pick random mode each reset
    float throttle;          // Target throttle [0,1]
    float target_bank;       // Target bank angle (radians)
    float target_vz;         // Target vertical velocity (m/s)

    // Curriculum: mode selection weights (sum to 1.0)
    float mode_weights[AP_COUNT];

    // Own RNG state (not affected by srand() calls)
    unsigned int rng_state;

    // PID gains
    float pitch_kp, pitch_kd;
    float roll_kp, roll_kd;

    // PID state (for derivative terms)
    float prev_vz;
    float prev_bank_error;

    // AP_WEAVE state
    float phase;             // Sine wave phase for weave oscillation

    // AP_EVASIVE state (set by caller each step)
    Vec3 threat_pos;         // Position of threat to evade
} AutopilotState;

// Simple LCG random for autopilot (not affected by srand)
static inline float ap_rand(AutopilotState* ap) {
    ap->rng_state = ap->rng_state * 1103515245 + 12345;
    return (float)((ap->rng_state >> 16) & 0x7FFF) / 32767.0f;
}

// Initialize autopilot with defaults
static inline void autopilot_init(AutopilotState* ap) {
    ap->mode = AP_STRAIGHT;
    ap->randomize_on_reset = 0;
    ap->throttle = AP_DEFAULT_THROTTLE;
    ap->target_bank = AP_DEFAULT_BANK_DEG * (PI / 180.0f);
    ap->target_vz = AP_DEFAULT_CLIMB_RATE;

    // Default: uniform weights for modes 1-5 (skip STRAIGHT and RANDOM)
    for (int i = 0; i < AP_COUNT; i++) {
        ap->mode_weights[i] = 0.0f;
    }
    float uniform = 1.0f / 5.0f;  // 5 modes: LEVEL, TURN_L, TURN_R, CLIMB, DESCEND
    ap->mode_weights[AP_LEVEL] = uniform;
    ap->mode_weights[AP_TURN_LEFT] = uniform;
    ap->mode_weights[AP_TURN_RIGHT] = uniform;
    ap->mode_weights[AP_CLIMB] = uniform;
    ap->mode_weights[AP_DESCEND] = uniform;

    // Seed autopilot RNG from system rand (called once at init, not affected by later srand)
    ap->rng_state = (unsigned int)rand();

    ap->pitch_kp = AP_LEVEL_KP;
    ap->pitch_kd = AP_LEVEL_KD;
    ap->roll_kp = AP_TURN_ROLL_KP;
    ap->roll_kd = AP_TURN_ROLL_KD;

    ap->prev_vz = 0.0f;
    ap->prev_bank_error = 0.0f;

    // New mode state
    ap->phase = 0.0f;
    ap->threat_pos = vec3(0, 0, 0);
}

// Set autopilot mode with parameters
static inline void autopilot_set_mode(AutopilotState* ap, AutopilotMode mode,
                                      float throttle, float bank_deg, float climb_rate) {
    ap->mode = mode;
    ap->randomize_on_reset = (mode == AP_RANDOM) ? 1 : 0;
    ap->throttle = throttle;
    ap->target_bank = bank_deg * (PI / 180.0f);
    ap->target_vz = climb_rate;

    // Reset PID state on mode change
    ap->prev_vz = 0.0f;
    ap->prev_bank_error = 0.0f;

    // Set appropriate gains based on mode
    if (mode == AP_LEVEL || mode == AP_CLIMB || mode == AP_DESCEND) {
        ap->pitch_kp = AP_LEVEL_KP;
        ap->pitch_kd = AP_LEVEL_KD;
    } else if (mode == AP_TURN_LEFT || mode == AP_TURN_RIGHT) {
        ap->pitch_kp = AP_TURN_ELEV_KP;
        ap->pitch_kd = AP_TURN_ELEV_KD;
        ap->roll_kp = AP_TURN_ROLL_KP;
        ap->roll_kd = AP_TURN_ROLL_KD;
    }
}

// Randomize autopilot mode using weighted selection (for AP_RANDOM at reset)
static inline void autopilot_randomize(AutopilotState* ap) {
    float r = ap_rand(ap);  // Use own RNG, not affected by srand()
    float cumsum = 0.0f;
    AutopilotMode selected = AP_LEVEL;  // Default fallback

    for (int i = 1; i < AP_COUNT - 1; i++) {  // Skip STRAIGHT(0) and RANDOM(6)
        cumsum += ap->mode_weights[i];
        if (r <= cumsum) {
            selected = (AutopilotMode)i;
            break;
        }
    }

    // Save randomize flag (autopilot_set_mode would clear it)
    int save_randomize = ap->randomize_on_reset;
    autopilot_set_mode(ap, selected, AP_DEFAULT_THROTTLE,
                      AP_DEFAULT_BANK_DEG, AP_DEFAULT_CLIMB_RATE);
    ap->randomize_on_reset = save_randomize;
}

// Get bank angle from plane orientation
// Returns positive for right bank, negative for left bank
static inline float ap_get_bank_angle(Plane* p) {
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float bank = acosf(fminf(fmaxf(up.z, -1.0f), 1.0f));
    if (up.y < 0) bank = -bank;
    return bank;
}

// Get vertical velocity from plane
static inline float ap_get_vz(Plane* p) {
    return p->vel.z;
}

// Clamp value to range
static inline float ap_clamp(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

// Main autopilot step function
// Computes actions[5] = [throttle, elevator, ailerons, rudder, trigger]
static inline void autopilot_step(AutopilotState* ap, Plane* p, float* actions, float dt) {
    // Initialize all actions to zero
    actions[0] = 0.0f;  // throttle (will be set below)
    actions[1] = 0.0f;  // elevator
    actions[2] = 0.0f;  // ailerons
    actions[3] = 0.0f;  // rudder
    actions[4] = -1.0f; // trigger (never fire)

    // Set throttle (convert from [0,1] to [-1,1] action space)
    actions[0] = ap->throttle * 2.0f - 1.0f;

    float vz = ap_get_vz(p);
    float bank = ap_get_bank_angle(p);

    switch (ap->mode) {
        case AP_STRAIGHT:
            // Do nothing - just fly straight with throttle
            break;

        case AP_LEVEL: {
            // PD control on vz to maintain level flight
            float vz_error = -vz;  // Target vz = 0
            float vz_deriv = (vz - ap->prev_vz) / dt;
            float elevator = ap->pitch_kp * vz_error + ap->pitch_kd * vz_deriv;
            actions[1] = ap_clamp(elevator, -1.0f, 1.0f);
            ap->prev_vz = vz;
            break;
        }

        case AP_TURN_LEFT:
        case AP_TURN_RIGHT: {
            // Dual PID: roll to target bank, pitch to maintain altitude
            float target_bank = ap->target_bank;
            if (ap->mode == AP_TURN_LEFT) target_bank = -target_bank;

            // Elevator PID (maintain vz = 0)
            float vz_error = -vz;
            float vz_deriv = (vz - ap->prev_vz) / dt;
            float elevator = ap->pitch_kp * vz_error + ap->pitch_kd * vz_deriv;
            actions[1] = ap_clamp(elevator, -1.0f, 1.0f);
            ap->prev_vz = vz;

            // Aileron PID (achieve target bank)
            float bank_error = target_bank - bank;
            float bank_deriv = (bank_error - ap->prev_bank_error) / dt;
            float aileron = ap->roll_kp * bank_error + ap->roll_kd * bank_deriv;
            actions[2] = ap_clamp(aileron, -1.0f, 1.0f);
            ap->prev_bank_error = bank_error;
            break;
        }

        case AP_CLIMB: {
            // PD control to maintain target climb rate
            float vz_error = ap->target_vz - vz;
            float vz_deriv = (vz - ap->prev_vz) / dt;
            // Negative because nose-up pitch (negative elevator) increases climb
            float elevator = -ap->pitch_kp * vz_error + ap->pitch_kd * vz_deriv;
            actions[1] = ap_clamp(elevator, -1.0f, 1.0f);
            ap->prev_vz = vz;
            break;
        }

        case AP_DESCEND: {
            // PD control to maintain target descent rate
            float vz_error = -ap->target_vz - vz;  // Target is negative vz
            float vz_deriv = (vz - ap->prev_vz) / dt;
            float elevator = -ap->pitch_kp * vz_error + ap->pitch_kd * vz_deriv;
            actions[1] = ap_clamp(elevator, -1.0f, 1.0f);
            ap->prev_vz = vz;
            break;
        }

        case AP_HARD_TURN_LEFT:
        case AP_HARD_TURN_RIGHT: {
            // Aggressive turn with high bank angle (70°)
            float target_bank = AP_HARD_BANK_DEG * (PI / 180.0f);
            if (ap->mode == AP_HARD_TURN_LEFT) target_bank = -target_bank;

            // Hard pull to maintain altitude in steep bank
            float vz_error = -vz;
            float elevator = -0.5f + ap->pitch_kp * vz_error;  // Base pull + PD
            actions[1] = ap_clamp(elevator, -1.0f, 1.0f);
            ap->prev_vz = vz;

            // Aggressive aileron to achieve bank (50% more aggressive)
            float bank_error = target_bank - bank;
            float aileron = ap->roll_kp * bank_error * 1.5f;
            actions[2] = ap_clamp(aileron, -1.0f, 1.0f);
            break;
        }

        case AP_WEAVE: {
            // Sine wave banking - oscillates left/right, hard to lead
            ap->phase += dt * (2.0f * PI / AP_WEAVE_PERIOD);
            if (ap->phase > 2.0f * PI) ap->phase -= 2.0f * PI;

            float target_bank = AP_WEAVE_AMPLITUDE * sinf(ap->phase);

            // Elevator PID for level flight (maintain vz = 0)
            float vz_error = -vz;
            float vz_deriv = (vz - ap->prev_vz) / dt;
            float elevator = ap->pitch_kp * vz_error + ap->pitch_kd * vz_deriv;
            actions[1] = ap_clamp(elevator, -1.0f, 1.0f);
            ap->prev_vz = vz;

            // Aileron PID to track oscillating bank
            float bank_error = target_bank - bank;
            float bank_deriv = (bank_error - ap->prev_bank_error) / dt;
            float aileron = ap->roll_kp * bank_error + ap->roll_kd * bank_deriv;
            actions[2] = ap_clamp(aileron, -1.0f, 1.0f);
            ap->prev_bank_error = bank_error;
            break;
        }

        case AP_EVASIVE: {
            // Break turn away from threat when close and behind
            Vec3 to_threat = sub3(ap->threat_pos, p->pos);
            float dist = norm3(to_threat);
            Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));
            float dot_fwd = dot3(normalize3(to_threat), fwd);

            float target_bank = 0.0f;
            float base_elevator = 0.0f;

            // Check if threat is close (<600m) and not in front (behind or side)
            if (dist < 600.0f && dot_fwd < 0.3f) {
                // Threat close and behind - BREAK TURN!
                // Determine which side threat is on
                Vec3 right = quat_rotate(p->ori, vec3(0, -1, 0));
                float dot_right = dot3(normalize3(to_threat), right);

                // Turn away from threat (opposite side)
                target_bank = (dot_right > 0) ? -1.2f : 1.2f;  // ~70° break
                base_elevator = -0.6f;  // Pull hard
            }

            // Elevator: base pull + PD for altitude
            float vz_error = -vz;
            float elevator = base_elevator + ap->pitch_kp * vz_error;
            actions[1] = ap_clamp(elevator, -1.0f, 1.0f);
            ap->prev_vz = vz;

            // Aileron to achieve break bank (aggressive)
            float bank_error = target_bank - bank;
            float aileron = ap->roll_kp * bank_error * 1.5f;
            actions[2] = ap_clamp(aileron, -1.0f, 1.0f);
            break;
        }

        case AP_RANDOM:
            // Should have been randomized at reset, fall through to straight
            break;

        default:
            break;
    }
}

#endif // AUTOPILOT_H
