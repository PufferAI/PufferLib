// flightlib.h - Rate-based flight physics for dogfight environment
// Modeled after dronelib.h pattern - self-contained physics simulation module
//
// Contains:
//   - Rate-based attitude control (direct rate commands)
//   - Point-mass aerodynamics (no moments/stability derivatives)
//   - Propeller thrust model (T = P*eta/V, capped at static thrust)
//   - Drag polar: Cd = Cd0 + K*Cl^2

#ifndef FLIGHTLIB_H
#define FLIGHTLIB_H

#include "flightlib_shared.h"

// ============================================================================
// RESET FUNCTION (Rate-based)
// ============================================================================

static inline void reset_plane_rate(Plane *p, Vec3 pos, Vec3 vel) {
    p->pos = pos;
    p->vel = vel;
    p->prev_vel = vel;  // Initialize to current vel (no acceleration at start)
    p->omega = vec3(0, 0, 0);  // No angular velocity at start
    p->ori = quat(1, 0, 0, 0);
    p->throttle = 0.5f;
    p->g_force = 1.0f;  // 1G at start (level flight)
    p->yaw_from_rudder = 0.0f;  // No accumulated rudder yaw at start
    p->fire_cooldown = 0;
}

// ============================================================================
// PHYSICS MODEL - step_plane_with_physics_rate()
// ============================================================================
// This implements a simplified 6-DOF flight model with:
//   - Rate-based attitude control (not position control)
//   - Point-mass aerodynamics (no moments/stability derivatives)
//   - Propeller thrust model (T = P*eta/V, capped at static thrust)
//   - Drag polar: Cd = Cd0 + K*Cl^2
//   - Wing incidence angle (built-in AOA for near-level cruise)
//
// COORDINATE SYSTEM:
//   World frame: X=East, Y=North, Z=Up (right-handed, Z-up)
//   Body frame:  X=Forward (nose), Y=Right (wing), Z=Up (canopy)
//
// WING INCIDENCE:
//   The wing is mounted at WING_INCIDENCE (~2 deg) relative to fuselage.
//   Effective AOA for lift = body_alpha + WING_INCIDENCE
//   This allows near-level flight at cruise speed with zero pitch input.
//
// REMAINING LIMITATIONS:
//   - No pitching moment / static stability (Cm_alpha)
//   - Rate-based controls (not position-based)
//   - Symmetric stall model (real stall is asymmetric)
// ============================================================================
static inline void step_plane_with_physics_rate(Plane *p, float *actions, float dt) {
    // Save previous velocity for acceleration calculation (v²/r)
    p->prev_vel = p->vel;

    // ========================================================================
    // 1. BODY FRAME AXES (transform from body to world coordinates)
    // ========================================================================
    // These are the aircraft's body axes expressed in world coordinates
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));  // Nose direction
    Vec3 right = quat_rotate(p->ori, vec3(0, 1, 0));    // Right wing direction
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));       // Canopy direction

    // ========================================================================
    // 2. CONTROL INPUTS -> ANGULAR RATES
    // ========================================================================
    // Actions are [-1, 1], mapped to physical rates
    // NOTE: These are RATE commands, not POSITION commands!
    // Holding elevator=0.5 pitches DOWN continuously (standard joystick convention)
    float throttle = (actions[0] + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
    float pitch_rate = actions[1] * MAX_PITCH_RATE;  // rad/s, + = nose down (push fwd)
    float roll_rate = actions[2] * MAX_ROLL_RATE;    // rad/s, + = roll right

    // ========================================================================
    // 2a. RUDDER DAMPING - Realistic sideslip-limited yaw
    // ========================================================================
    // Real rudder physics: deflection creates sideslip angle (β), NOT sustained
    // yaw rate. Vertical tail creates restoring moment that limits β to ~10°.
    // Once equilibrium sideslip is reached, yaw rate approaches zero.
    //
    // Implementation: Track accumulated yaw from rudder, reduce effectiveness
    // as it accumulates toward max sideslip angle (~10°).
    float rudder_yaw_cmd = -actions[3] * MAX_YAW_RATE;  // Commanded yaw rate
    float max_rudder_yaw = 0.175f;  // ~10 degrees max sideslip (0.175 rad)

    // Damping: effectiveness drops to 0 as accumulated yaw approaches limit
    float damping = 1.0f - clampf(fabsf(p->yaw_from_rudder) / max_rudder_yaw, 0.0f, 1.0f);
    float yaw_rate = rudder_yaw_cmd * damping;

    // Update accumulated yaw state
    if (fabsf(actions[3]) < 0.1f) {
        // Rudder released: weathervane tendency returns nose to airflow
        // Vertical tail realigns with velocity, sideslip decays
        p->yaw_from_rudder *= 0.95f;
    } else {
        // Rudder active: accumulate yaw (limited by damping above)
        p->yaw_from_rudder += yaw_rate * dt;
    }

    // ========================================================================
    // 3. ATTITUDE INTEGRATION (Quaternion kinematics)
    // ========================================================================
    // q_dot = 0.5 * q * w  where w is angular velocity in body frame
    // This is the standard quaternion derivative formula
    Vec3 omega_body = vec3(roll_rate, pitch_rate, yaw_rate);  // body-frame w
    p->omega = omega_body;  // Store for consistency (used by momentum physics)
    Quat omega_quat = quat(0, omega_body.x, omega_body.y, omega_body.z);
    Quat q_dot = quat_mul(p->ori, omega_quat);
    p->ori.w += 0.5f * q_dot.w * dt;
    p->ori.x += 0.5f * q_dot.x * dt;
    p->ori.y += 0.5f * q_dot.y * dt;
    p->ori.z += 0.5f * q_dot.z * dt;
    quat_normalize(&p->ori);  // Prevent drift from numerical integration

    // ========================================================================
    // 4. ANGLE OF ATTACK (AOA, a)
    // ========================================================================
    // AOA = angle between velocity vector and body X-axis (nose)
    // Positive a = nose above flight path = generating positive lift
    //
    // SIGN CONVENTION:
    //   If velocity has component opposite to body Z (up), nose is above
    //   flight path, so a is positive.
    float V = norm3(p->vel);
    if (V < 1.0f) V = 1.0f;  // Prevent division by zero

    Vec3 vel_norm = normalize3(p->vel);
    float cos_alpha = dot3(vel_norm, forward);
    cos_alpha = clampf(cos_alpha, -1.0f, 1.0f);
    float alpha = acosf(cos_alpha);  // Always positive [0, pi]

    // Determine sign: positive when nose is ABOVE velocity vector
    // If vel dot up < 0, velocity is "below" the body frame -> nose above -> a > 0
    float sign_alpha = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
    alpha *= sign_alpha;

    // ========================================================================
    // 5. LIFT COEFFICIENT (Linear + Stall Clamp)
    // ========================================================================
    // C_L = C_L_alpha * (alpha - alpha_zero)
    // For cambered airfoils, alpha_zero < 0 (generates lift at 0° AOA)
    // P-51D NAA 45-100 airfoil: alpha_zero = -1.2°
    //
    // Effective AOA for lift = body_alpha + wing_incidence - alpha_zero
    // At 0° body pitch: alpha_eff = 0 + 1.5° - (-1.2°) = 2.7°
    // This gives C_L = 5.56 * 0.047 = 0.26, allowing near-level cruise
    //
    // Stall occurs at alpha_eff ~ 19° (P-51D clean), C_L_max = 1.48
    float alpha_effective = alpha + WING_INCIDENCE - ALPHA_ZERO;
    float C_L = C_L_ALPHA * alpha_effective;
    C_L = clampf(C_L, -C_L_MAX, C_L_MAX);  // Stall limiting (symmetric)

    // ========================================================================
    // 6. DYNAMIC PRESSURE
    // ========================================================================
    // q = 0.5*rho*V^2 [Pa or N/m^2]
    // This is the "pressure" available for aerodynamic forces
    // At 100 m/s: q = 0.5 * 1.225 * 10000 = 6,125 Pa
    float q_dyn = 0.5f * RHO * V * V;

    // ========================================================================
    // 7. LIFT FORCE
    // ========================================================================
    // L = Cl * q * S  [Newtons]
    // For level flight: L = W = m*g = 29,430 N
    // Required Cl at 100 m/s: Cl = 29430 / (6125 * 22) = 0.218
    // Required a = 0.218 / 5.7 = 0.038 rad ~ 2.2 deg
    float L_mag = C_L * q_dyn * WING_AREA;

    // ========================================================================
    // 8. DRAG FORCE (Drag Polar)
    // ========================================================================
    // Cd = Cd0 + K * Cl^2 + K_SIDESLIP * beta^2
    float C_D_sideslip = K_SIDESLIP * p->yaw_from_rudder * p->yaw_from_rudder;
    float C_D = C_D0 + K * C_L * C_L + C_D_sideslip;
    float D_mag = C_D * q_dyn * WING_AREA;

    // ========================================================================
    // 9. THRUST FORCE (Propeller Model)
    // ========================================================================
    // Power-based: P = T * V  ->  T = P * eta / V
    // At low speed, thrust is limited by static thrust capability
    //
    // At V=80 m/s, full throttle: T = 800,000 / 80 = 10,000 N
    // At V=143 m/s (max speed):   T = 800,000 / 143 = 5,594 N ~ D
    float P_avail = ENGINE_POWER * throttle;
    float T_dynamic = (P_avail * ETA_PROP) / V;   // Thrust from power equation
    float T_static = 0.3f * P_avail;              // Static thrust limit
    float T_mag = fminf(T_static, T_dynamic);     // Can't exceed either limit

    // ========================================================================
    // 10. FORCE DIRECTIONS (All in world frame)
    // ========================================================================
    Vec3 drag_dir = mul3(vel_norm, -1.0f);  // Opposite to velocity
    Vec3 thrust_dir = forward;               // Along body X-axis (nose)

    // Lift direction: perpendicular to velocity, in plane of velocity & wing
    // lift_dir = vel x right, then normalized
    // This ensures lift is perpendicular to V and perpendicular to span
    Vec3 lift_dir = cross3(vel_norm, right);
    float lift_dir_mag = norm3(lift_dir);
    if (lift_dir_mag > 0.01f) {
        lift_dir = mul3(lift_dir, 1.0f / lift_dir_mag);
    } else {
        lift_dir = up;  // Fallback if velocity parallel to wing (rare)
    }

    // ========================================================================
    // 11. WEIGHT (Gravity)
    // ========================================================================
    Vec3 weight = vec3(0, 0, -MASS * GRAVITY);  // Always -Z in world frame

    // ========================================================================
    // 12. SUM FORCES -> ACCELERATION
    // ========================================================================
    Vec3 F_thrust = mul3(thrust_dir, T_mag);
    Vec3 F_lift = mul3(lift_dir, L_mag);
    Vec3 F_drag = mul3(drag_dir, D_mag);

    // Aerodynamic forces only (what pilot feels - "specific force")
    // In level flight: lift ≈ weight, so F_aero_up ≈ m*g, giving g_force ≈ 1.0
    Vec3 F_aero = add3(F_thrust, add3(F_lift, F_drag));

    // Body-up axis (perpendicular to wings, toward canopy)
    Vec3 body_up = quat_rotate(p->ori, vec3(0, 0, 1));

    // G-force = aero force along body-up / (mass * g)
    // This is what the pilot feels (pushed into seat = positive G)
    float g_force = dot3(F_aero, body_up) * INV_MASS * INV_GRAVITY;

    // Total force includes weight for actual physics
    Vec3 F_total = add3(F_aero, weight);

    // ========================================================================
    // 13. G-LIMIT (Asymmetric for Positive/Negative G)
    // ========================================================================
    // Pilots can handle much more positive G (blood to feet, 6G+) than
    // negative G (blood to head, -1.5G is very uncomfortable).
    // Limit the body-normal acceleration asymmetrically.
    Vec3 accel = mul3(F_total, INV_MASS);

    // Asymmetric limits on felt G
    if (g_force > G_LIMIT_POS) {
        // Positive G exceeded - clamp
        float excess = (g_force - G_LIMIT_POS) * GRAVITY;  // Excess accel in m/s^2
        accel = sub3(accel, mul3(body_up, excess));
        g_force = G_LIMIT_POS;
    } else if (g_force < -G_LIMIT_NEG) {
        // Negative G exceeded - clamp (need to ADD acceleration along body_up)
        float deficit = (-G_LIMIT_NEG - g_force) * GRAVITY;  // How much to add back
        accel = add3(accel, mul3(body_up, deficit));  // ADD, not subtract!
        g_force = -G_LIMIT_NEG;
    }

    if (DEBUG >= 10) printf("=== PHYSICS ===\n");
    if (DEBUG >= 10) printf("speed=%.1f m/s (stall~45, max~159 P-51D)\n", V);
    if (DEBUG >= 10) printf("throttle=%.2f\n", throttle);
    if (DEBUG >= 10) printf("alpha_body=%.2f deg, alpha_eff=%.2f deg (inc=%.1f, a0=%.1f), C_L=%.3f\n",
                      alpha * RAD_TO_DEG, alpha_effective * RAD_TO_DEG,
                      WING_INCIDENCE * RAD_TO_DEG, ALPHA_ZERO * RAD_TO_DEG, C_L);
    if (DEBUG >= 10) printf("thrust=%.0f N, lift=%.0f N, drag=%.0f N, weight=%.0f N\n", T_mag, L_mag, D_mag, MASS * GRAVITY);
    if (DEBUG >= 10) printf("g_force=%.2f g (limit=+%.1f/-%.1f)\n", g_force, G_LIMIT_POS, G_LIMIT_NEG);

    // ========================================================================
    // 14. INTEGRATION (Semi-implicit Euler)
    // ========================================================================
    // v(t+dt) = v(t) + a * dt
    // x(t+dt) = x(t) + v(t+dt) * dt  (using NEW velocity)
    p->vel = add3(p->vel, mul3(accel, dt));
    p->pos = add3(p->pos, mul3(p->vel, dt));

    p->throttle = throttle;
    p->g_force = g_force;  // Store for reward calculation
}

#endif // FLIGHTLIB_H
