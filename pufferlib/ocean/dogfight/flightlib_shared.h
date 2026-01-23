// flightlib_shared.h - Shared flight physics types and constants
// Used by both flightlib.h (rate-based) and physics_momentum.h (momentum-based)

#ifndef FLIGHTLIB_SHARED_H
#define FLIGHTLIB_SHARED_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Allow DEBUG to be defined before including this header
#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef PI
#define PI 3.14159265358979f
#endif

// ============================================================================
// MATH TYPES
// ============================================================================

typedef struct { float x, y, z; } Vec3;
typedef struct { float w, x, y, z; } Quat;

// ============================================================================
// MATH UTILITIES
// ============================================================================

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float rndf(float a, float b) {
    return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

// --- Vec3 operations ---

static inline Vec3 vec3(float x, float y, float z) { return (Vec3){x, y, z}; }
static inline Vec3 add3(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 mul3(Vec3 a, float s) { return (Vec3){a.x * s, a.y * s, a.z * s}; }
static inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline Vec3 normalize3(Vec3 v) {
    float n = norm3(v);
    if (n < 1e-8f) return vec3(0, 0, 0);
    return mul3(v, 1.0f / n);
}

static inline Vec3 cross3(Vec3 a, Vec3 b) {
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// --- Quaternion operations ---

static inline Quat quat(float w, float x, float y, float z) { return (Quat){w, x, y, z}; }

static inline Quat quat_mul(Quat a, Quat b) {
    return (Quat){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

static inline Quat quat_add(Quat a, Quat b) {
    return (Quat){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Quat quat_scale(Quat q, float s) {
    return (Quat){q.w * s, q.x * s, q.y * s, q.z * s};
}

static inline void quat_normalize(Quat* q) {
    float n = sqrtf(q->w*q->w + q->x*q->x + q->y*q->y + q->z*q->z);
    if (n > 1e-8f) {
        float inv = 1.0f / n;
        q->w *= inv; q->x *= inv; q->y *= inv; q->z *= inv;
    }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = {0.0f, v.x, v.y, v.z};
    Quat q_conj = {q.w, -q.x, -q.y, -q.z};
    Quat tmp = quat_mul(q, qv);
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float half = angle * 0.5f;
    float s = sinf(half);
    return (Quat){cosf(half), axis.x * s, axis.y * s, axis.z * s};
}

// ============================================================================
// AIRCRAFT PARAMETERS - P-51D Mustang Reference
// ============================================================================
// Based on P51d_REFERENCE_DATA.md - validated against historical data
// Test condition: 9,000 lb (4,082 kg) combat weight, sea level ISA
//
// THEORETICAL PERFORMANCE (P-51D targets):
//   Max speed (SL, Military): 355 mph (159 m/s)
//   Max speed (SL, WEP):      368 mph (164 m/s)
//   Stall speed (clean):      100 mph (45 m/s)
//   ROC (SL, Military):       3,030 ft/min (15.4 m/s)
//
// LIFT MODEL:
//   C_L = C_L_alpha * (alpha + incidence - alpha_zero)
//   The P-51D has a cambered airfoil (NAA 45-100) with alpha_zero = -1.2°
//   Wing incidence is +1.5° relative to fuselage datum
//   At 0° body pitch: effective AOA = 1.5° - (-1.2°) = 2.7°, C_L ~ 0.26
//
// DRAG POLAR: Cd = Cd0 + K * Cl^2
//   - Cd0 = 0.0163 (P-51D published value, very clean laminar flow wing)
//   - K = 0.072 = 1/(pi * e * AR) where e=0.75, AR=5.86
// ============================================================================

#define MASS 4082.0f           // kg (P-51D combat weight: 9,000 lb)
#define WING_AREA 21.65f       // m^2 (P-51D: 233 ft^2)
#define WINGSPAN 11.28f        // m (P-51D: 37 ft)
#define CHORD 2.02f            // m (MAC - mean aerodynamic chord)

// Moments of inertia (estimated for P-51D, kg⋅m²)
// Fighter aircraft: Iyy >> Ixx ≈ Izz
#define IXX 6500.0f    // Roll inertia (wings not very long)
#define IYY 22000.0f   // Pitch inertia (long fuselage, largest)
#define IZZ 27000.0f   // Yaw inertia (fuselage + vertical tail)

// Aerodynamic coefficients
#define C_D0 0.0163f           // parasitic drag coefficient (P-51D laminar flow)
#define K 0.072f               // induced drag factor: 1/(pi*0.75*5.86)
#define K_SIDESLIP 0.7f        // sideslip drag factor (JSBSim: 0.05 CD at 15 deg)
#define C_L_MAX 1.48f          // max lift coefficient before stall (P-51D clean)
#define C_L_ALPHA 5.56f        // lift curve slope (P-51D: 0.097/deg = 5.56/rad)
#define ALPHA_ZERO -0.021f     // zero-lift angle (rad), -1.2° for cambered airfoil
#define WING_INCIDENCE 0.026f  // wing incidence angle (rad), +1.5° (P-51D)

// Propulsion
#define ENGINE_POWER 1112000.0f // watts (P-51D Military: 1,490 hp)
#define ETA_PROP 0.80f         // propeller efficiency (P-51D cruise: 0.80-0.85)

// Environment
#define GRAVITY 9.81f          // m/s^2
#define RHO 1.225f             // air density kg/m^3 (sea level ISA)

// G-limits
#define G_LIMIT_POS 6.0f       // max positive G (pulling up) - pilot limit
#define G_LIMIT_NEG 1.5f       // max negative G (pushing over) - blood to head is painful

// Inverse constants for faster computation (multiply instead of divide)
#define INV_MASS     0.000245f   // 1/4082
#define INV_GRAVITY  0.10197f    // 1/9.81
#define RAD_TO_DEG   57.2957795f // 180/PI

// Rate limits
#define MAX_PITCH_RATE 2.5f    // rad/s
#define MAX_ROLL_RATE 3.0f     // rad/s
#define MAX_YAW_RATE 0.50f     // rad/s (~29 deg/s command, realistic ~7 deg/s achieved)

// ============================================================================
// PLANE STRUCT - Flight object state
// ============================================================================

typedef struct {
    Vec3 pos;
    Vec3 vel;
    Vec3 prev_vel;      // Previous velocity for acceleration calculation
    Vec3 omega;         // Angular velocity in body frame (for momentum physics)
    Quat ori;
    float throttle;
    float g_force;      // Current G-loading (for reward calculation)
    float yaw_from_rudder;  // Accumulated yaw from rudder (for damping)
    int fire_cooldown;  // Ticks until can fire again (0 = ready)
} Plane;

// ============================================================================
// SIMPLE OPPONENT PHYSICS - used by both physics modes
// ============================================================================
// Opponent doesn't need full physics - just forward motion

static inline void step_plane(Plane *p, float dt) {
    // Save previous velocity for acceleration calculation
    p->prev_vel = p->vel;

    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    float speed = norm3(p->vel);
    if (speed < 1.0f) speed = 80.0f;
    p->vel = mul3(forward, speed);
    p->pos = add3(p->pos, mul3(p->vel, dt));

    if (DEBUG >= 10) printf("=== TARGET ===\n");
    if (DEBUG >= 10) printf("target_speed=%.1f m/s (expected=80)\n", speed);
    if (DEBUG >= 10) printf("target_pos=(%.1f, %.1f, %.1f)\n", p->pos.x, p->pos.y, p->pos.z);
    if (DEBUG >= 10) printf("target_fwd=(%.2f, %.2f, %.2f)\n", forward.x, forward.y, forward.z);
}

#endif // FLIGHTLIB_SHARED_H
