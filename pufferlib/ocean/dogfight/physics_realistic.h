// physics_realistic.h - Realistic RK4 flight physics for dogfight environment
//
// Full 6-DOF flight model with:
//   - Angular momentum as state variable (omega integrated, not commanded)
//   - RK4 integration (4th-order Runge-Kutta)
//   - Aerodynamic moments from stability derivatives
//   - Control surface effectiveness (elevator, aileron, rudder)
//   - Euler's equations for rotational dynamics

#ifndef PHYSICS_REALISTIC_H
#define PHYSICS_REALISTIC_H

#include "flightlib_shared.h"

// ============================================================================
// DEBUG CONTROL
// ============================================================================
// Set DEBUG_REALISTIC to enable debug output:
//   0 = off
//   1 = high-level per-step summary
//   2 = forces and moments
//   3 = all intermediate calculations
//   5 = RK4 stages
//  10 = everything (very verbose)

#ifndef DEBUG_REALISTIC
#define DEBUG_REALISTIC 0
#endif

// Step counter for debug output (to limit spam)
static int _realistic_step_count = 0;
static int _realistic_rk4_stage = 0;  // Which RK4 stage (0=k1, 1=k2, 2=k3, 3=k4)

// ============================================================================
// STABILITY DERIVATIVES (body-axis, per radian)
// ============================================================================
// These create aerodynamic moments proportional to angles and rates

// Static stability (moment vs angle)
// CM_0: Pitch trim offset. Negative counters nose-up from wing incidence/lift.
// With WING_INCIDENCE=+1.5° and cambered airfoil, lift creates nose-up moment.
// Tuning: 0.025f->2.26G, -0.03f->0.16G. Targeting ~1.0G, linear interpolation suggests -0.005f.
#define CM_0 -0.005f        // Pitch trim offset (fine-tuned for ~1.0G level flight)
#define CM_ALPHA -1.2f      // Pitch stability (negative = stable, nose-up creates nose-down moment)
#define CL_BETA -0.08f      // Dihedral effect (negative = stable, sideslip creates restoring roll)
#define CN_BETA 0.12f       // Weathervane stability (positive = stable, sideslip creates restoring yaw)

// Damping derivatives (dimensionless, multiplied by q*c/2V or p*b/2V)
#define CM_Q -10.0f         // Pitch damping (matches JSBSim P-51D)
#define CL_P -0.4f          // Roll damping (opposes roll rate)
#define CN_R -0.15f         // Yaw damping (opposes yaw rate)

// Control derivatives (per radian deflection)
// Tuned for P-51D target performance (see test results)
#define CM_DELTA_E -0.5f    // Elevator: negative = nose UP with positive (back stick) deflection
#define CL_DELTA_A 0.20f    // Aileron: positive = roll RIGHT with positive deflection
                            // Tuning: 0.04f->19°, 0.15f->70°, need 90°, try 0.20f
#define CN_DELTA_R 0.015f   // Rudder: positive = nose RIGHT with positive (right pedal) deflection
                            // Tuning: 0.015f should give 2-20° heading change with full rudder

// Cross-coupling derivatives
#define CN_DELTA_A -0.007f  // Adverse yaw from aileron (negative = right aileron causes left yaw)
#define CL_DELTA_R -0.003f  // Roll from rudder (negative = right rudder causes left roll, rudder is above roll axis)

// Control surface deflection limits (radians)
#define MAX_ELEVATOR_DEFLECTION 0.35f   // ±20°
#define MAX_AILERON_DEFLECTION 0.35f    // ±20°
#define MAX_RUDDER_DEFLECTION 0.35f     // ±20°

// ============================================================================
// STATE DERIVATIVE STRUCT (for RK4)
// ============================================================================

typedef struct {
    Vec3 vel;       // d(pos)/dt = velocity
    Vec3 v_dot;     // d(vel)/dt = acceleration
    Quat q_dot;     // d(quat)/dt = quaternion rate
    Vec3 w_dot;     // d(omega)/dt = angular acceleration
} StateDerivative;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Compute angle of attack from state
static inline float compute_aoa(Plane* p) {
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));

    float V = norm3(p->vel);
    if (V < 1.0f) return 0.0f;

    Vec3 vel_norm = normalize3(p->vel);
    float cos_alpha = dot3(vel_norm, forward);
    cos_alpha = clampf(cos_alpha, -1.0f, 1.0f);
    float alpha = acosf(cos_alpha);  // Always positive [0, pi]

    // Sign: positive when nose is ABOVE velocity vector
    // If vel dot up < 0, velocity is "below" the body frame -> nose above -> alpha > 0
    float vel_dot_up = dot3(p->vel, up);
    float sign = (vel_dot_up < 0) ? 1.0f : -1.0f;

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("  [AOA] forward=(%.3f,%.3f,%.3f) up=(%.3f,%.3f,%.3f)\n",
               forward.x, forward.y, forward.z, up.x, up.y, up.z);
        printf("  [AOA] vel=(%.1f,%.1f,%.1f) |vel|=%.1f\n",
               p->vel.x, p->vel.y, p->vel.z, V);
        printf("  [AOA] vel_norm=(%.4f,%.4f,%.4f)\n",
               vel_norm.x, vel_norm.y, vel_norm.z);
        printf("  [AOA] cos_alpha=%.4f (vel_norm·forward)\n", cos_alpha);
        printf("  [AOA] acos(cos_alpha)=%.4f rad = %.2f deg\n", alpha, alpha * RAD_TO_DEG);
        printf("  [AOA] vel·up=%.4f -> sign=%.0f\n", vel_dot_up, sign);
        printf("  [AOA] FINAL alpha=%.4f rad = %.2f deg\n", alpha * sign, alpha * sign * RAD_TO_DEG);
    }

    return alpha * sign;
}

// Compute sideslip angle from state
static inline float compute_sideslip(Plane* p) {
    Vec3 right = quat_rotate(p->ori, vec3(0, 1, 0));

    float V = norm3(p->vel);
    if (V < 1.0f) return 0.0f;

    Vec3 vel_norm = normalize3(p->vel);

    // beta = arcsin(v · right / |v|) - positive when velocity has component to the right
    float sin_beta = dot3(vel_norm, right);
    float beta = asinf(clampf(sin_beta, -1.0f, 1.0f));

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("  [BETA] right=(%.3f,%.3f,%.3f)\n", right.x, right.y, right.z);
        printf("  [BETA] sin_beta=%.4f (vel_norm·right)\n", sin_beta);
        printf("  [BETA] FINAL beta=%.4f rad = %.2f deg\n", beta, beta * RAD_TO_DEG);
    }

    return beta;
}

// Compute lift direction (perpendicular to velocity, in lift plane)
static inline Vec3 compute_lift_direction(Vec3 vel_norm, Vec3 right, Vec3 body_up) {
    Vec3 lift_dir = cross3(vel_norm, right);
    float mag = norm3(lift_dir);

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("  [LIFT_DIR] vel_norm×right=(%.3f,%.3f,%.3f) |mag|=%.4f\n",
               lift_dir.x, lift_dir.y, lift_dir.z, mag);
    }

    if (mag > 0.01f) {
        Vec3 result = mul3(lift_dir, 1.0f / mag);
        if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
            printf("  [LIFT_DIR] normalized=(%.3f,%.3f,%.3f)\n", result.x, result.y, result.z);
        }
        return result;
    }
    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("  [LIFT_DIR] FALLBACK to world_up=(0,0,1)\n");
    }
    return (Vec3){0, 0, 1};  // Fallback to world-frame up (lift perpendicular to ground)
}

// Compute thrust from power model
static inline float compute_thrust(float throttle, float V) {
    float P_avail = ENGINE_POWER * throttle;
    float T_dynamic = (P_avail * ETA_PROP) / V;   // Thrust from power equation
    float T_static = 0.3f * P_avail;              // Static thrust limit
    float T = fminf(T_static, T_dynamic);         // Can't exceed either limit

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("  [THRUST] throttle=%.2f P_avail=%.0f W\n", throttle, P_avail);
        printf("  [THRUST] T_dynamic=%.0f N, T_static=%.0f N -> T=%.0f N\n",
               T_dynamic, T_static, T);
    }

    return T;
}

// Helper: apply derivative to state (for RK4 intermediate stages)
static inline void step_temp(Plane* state, StateDerivative* d, float dt, Plane* out) {
    out->pos = add3(state->pos, mul3(d->vel, dt));
    out->vel = add3(state->vel, mul3(d->v_dot, dt));
    out->ori = quat_add(state->ori, quat_scale(d->q_dot, dt));
    quat_normalize(&out->ori);
    out->omega = add3(state->omega, mul3(d->w_dot, dt));
    out->throttle = state->throttle;
    out->g_force = state->g_force;
    out->yaw_from_rudder = state->yaw_from_rudder;
    out->fire_cooldown = state->fire_cooldown;
    out->prev_vel = state->prev_vel;

    if (DEBUG_REALISTIC >= 5) {
        printf("    [STEP_TEMP] dt=%.4f\n", dt);
        printf("    [STEP_TEMP] d->vel=(%.2f,%.2f,%.2f) d->v_dot=(%.2f,%.2f,%.2f)\n",
               d->vel.x, d->vel.y, d->vel.z, d->v_dot.x, d->v_dot.y, d->v_dot.z);
        printf("    [STEP_TEMP] d->w_dot=(%.4f,%.4f,%.4f)\n",
               d->w_dot.x, d->w_dot.y, d->w_dot.z);
        printf("    [STEP_TEMP] out->vel=(%.2f,%.2f,%.2f)\n", out->vel.x, out->vel.y, out->vel.z);
        printf("    [STEP_TEMP] out->omega=(%.4f,%.4f,%.4f)\n",
               out->omega.x, out->omega.y, out->omega.z);
        printf("    [STEP_TEMP] out->ori=(%.4f,%.4f,%.4f,%.4f)\n",
               out->ori.w, out->ori.x, out->ori.y, out->ori.z);
    }
}

// ============================================================================
// CORE PHYSICS: compute_derivatives()
// ============================================================================
// This is called 4 times per RK4 step. Computes all state derivatives.

static inline void compute_derivatives(Plane* state, float* actions, float dt, StateDerivative* deriv) {

    if (DEBUG_REALISTIC >= 5) {
        const char* stage_names[] = {"k1", "k2", "k3", "k4"};
        printf("\n  === COMPUTE_DERIVATIVES (RK4 stage %s) ===\n", stage_names[_realistic_rk4_stage]);
    }

    // ========================================================================
    // 1. Extract state
    // ========================================================================
    float V = norm3(state->vel);
    if (V < 1.0f) V = 1.0f;  // Prevent div-by-zero

    Vec3 vel_norm = normalize3(state->vel);
    Vec3 forward = quat_rotate(state->ori, vec3(1, 0, 0));  // Body X-axis
    Vec3 right = quat_rotate(state->ori, vec3(0, 1, 0));    // Body Y-axis
    Vec3 body_up = quat_rotate(state->ori, vec3(0, 0, 1));  // Body Z-axis

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- STATE ---\n");
        printf("  pos=(%.1f, %.1f, %.1f)\n", state->pos.x, state->pos.y, state->pos.z);
        printf("  vel=(%.2f, %.2f, %.2f) |V|=%.2f m/s\n",
               state->vel.x, state->vel.y, state->vel.z, V);
        printf("  vel_norm=(%.4f, %.4f, %.4f)\n", vel_norm.x, vel_norm.y, vel_norm.z);
        printf("  ori=(w=%.4f, x=%.4f, y=%.4f, z=%.4f) |ori|=%.6f\n",
               state->ori.w, state->ori.x, state->ori.y, state->ori.z,
               sqrtf(state->ori.w*state->ori.w + state->ori.x*state->ori.x +
                     state->ori.y*state->ori.y + state->ori.z*state->ori.z));
        printf("  omega=(%.4f, %.4f, %.4f) rad/s = (%.2f, %.2f, %.2f) deg/s\n",
               state->omega.x, state->omega.y, state->omega.z,
               state->omega.x * RAD_TO_DEG, state->omega.y * RAD_TO_DEG, state->omega.z * RAD_TO_DEG);
        printf("  forward=(%.4f, %.4f, %.4f)\n", forward.x, forward.y, forward.z);
        printf("  right=(%.4f, %.4f, %.4f)\n", right.x, right.y, right.z);
        printf("  body_up=(%.4f, %.4f, %.4f)\n", body_up.x, body_up.y, body_up.z);

        // Compute pitch angle from forward vector
        float pitch_from_forward = asinf(-forward.z) * RAD_TO_DEG;  // nose up = positive
        printf("  pitch_from_forward=%.2f deg (nose %s)\n",
               pitch_from_forward, pitch_from_forward > 0 ? "UP" : "DOWN");

        // Velocity direction
        float vel_pitch = asinf(vel_norm.z) * RAD_TO_DEG;  // climbing = positive
        printf("  vel_pitch=%.2f deg (%s)\n", vel_pitch, vel_pitch > 0 ? "CLIMBING" : "DESCENDING");
    }

    // ========================================================================
    // 2. Compute aerodynamic angles
    // ========================================================================
    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("\n  --- AERODYNAMIC ANGLES ---\n");
    }
    float alpha = compute_aoa(state);      // Angle of attack
    float beta = compute_sideslip(state);  // Sideslip angle

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("  alpha=%.4f rad = %.2f deg (%s)\n", alpha, alpha * RAD_TO_DEG,
               alpha > 0 ? "nose ABOVE vel" : "nose BELOW vel");
        printf("  beta=%.4f rad = %.2f deg\n", beta, beta * RAD_TO_DEG);
    }

    // ========================================================================
    // 3. Dynamic pressure
    // ========================================================================
    float q_bar = 0.5f * RHO * V * V;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- DYNAMIC PRESSURE ---\n");
        printf("  q_bar = 0.5 * %.4f * %.1f^2 = %.1f Pa\n", RHO, V, q_bar);
    }

    // ========================================================================
    // 4. Map actions to control surface deflections
    // ========================================================================
    // Actions are [-1, 1], mapped to deflection in radians
    // Sign conventions (M_moment is negated later for Z-up frame):
    //   - Elevator: actions[1] > 0 (push forward) → nose DOWN
    //   - Aileron: actions[2] > 0 → roll RIGHT
    //   - Rudder: actions[3] > 0 → yaw LEFT
    float throttle = clampf((actions[0] + 1.0f) * 0.5f, 0.0f, 1.0f);  // [0, 1]
    float delta_e = clampf(actions[1], -1.0f, 1.0f) * MAX_ELEVATOR_DEFLECTION;  // Elevator
    float delta_a = clampf(actions[2], -1.0f, 1.0f) * MAX_AILERON_DEFLECTION;   // Aileron
    float delta_r = clampf(actions[3], -1.0f, 1.0f) * MAX_RUDDER_DEFLECTION;    // Rudder

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- CONTROLS ---\n");
        printf("  actions=[%.3f, %.3f, %.3f, %.3f]\n",
               actions[0], actions[1], actions[2], actions[3]);
        printf("  throttle=%.3f (%.0f%%)\n", throttle, throttle * 100);
        printf("  delta_e=%.4f rad = %.2f deg (elevator, %s)\n",
               delta_e, delta_e * RAD_TO_DEG,
               delta_e > 0 ? "push=nose DOWN" : delta_e < 0 ? "pull=nose UP" : "neutral");
        printf("  delta_a=%.4f rad = %.2f deg (aileron)\n", delta_a, delta_a * RAD_TO_DEG);
        printf("  delta_r=%.4f rad = %.2f deg (rudder)\n", delta_r, delta_r * RAD_TO_DEG);
    }

    // ========================================================================
    // 5. Compute lift coefficient
    // ========================================================================
    float alpha_effective = alpha + WING_INCIDENCE - ALPHA_ZERO;
    float C_L_raw = C_L_ALPHA * alpha_effective;
    float C_L = clampf(C_L_raw, -C_L_MAX, C_L_MAX);  // Stall limiting

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- LIFT COEFFICIENT ---\n");
        printf("  alpha=%.4f + WING_INCIDENCE=%.4f - ALPHA_ZERO=%.4f = alpha_eff=%.4f rad\n",
               alpha, WING_INCIDENCE, ALPHA_ZERO, alpha_effective);
        printf("  C_L_raw = C_L_ALPHA(%.2f) * alpha_eff(%.4f) = %.4f\n",
               C_L_ALPHA, alpha_effective, C_L_raw);
        printf("  C_L = clamp(%.4f, -%.2f, %.2f) = %.4f%s\n",
               C_L_raw, C_L_MAX, C_L_MAX, C_L,
               (C_L != C_L_raw) ? " (STALL CLAMPED!)" : "");
    }

    // ========================================================================
    // 6. Compute drag coefficient (drag polar)
    // ========================================================================
    float C_D0_term = C_D0;
    float induced_term = K * C_L * C_L;
    float sideslip_term = K_SIDESLIP * beta * beta;
    float C_D = C_D0_term + induced_term + sideslip_term;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- DRAG COEFFICIENT ---\n");
        printf("  C_D0=%.4f + K*C_L^2=%.4f + K_sideslip*beta^2=%.4f = C_D=%.4f\n",
               C_D0_term, induced_term, sideslip_term, C_D);
        printf("  L/D ratio = %.2f\n", (C_D > 0.0001f) ? C_L / C_D : 0.0f);
    }

    // ========================================================================
    // 7. Compute aerodynamic FORCES
    // ========================================================================
    float L_mag = C_L * q_bar * WING_AREA;  // Lift magnitude
    float D_mag = C_D * q_bar * WING_AREA;  // Drag magnitude

    // Lift direction: perpendicular to velocity, in plane with right wing
    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("\n  --- LIFT DIRECTION ---\n");
    }
    Vec3 lift_dir = compute_lift_direction(vel_norm, right, body_up);
    Vec3 F_lift = mul3(lift_dir, L_mag);

    // Drag direction: opposite to velocity
    Vec3 F_drag = mul3(vel_norm, -D_mag);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- AERODYNAMIC FORCES ---\n");
        printf("  L_mag = C_L(%.4f) * q_bar(%.1f) * S(%.1f) = %.1f N\n",
               C_L, q_bar, WING_AREA, L_mag);
        printf("  D_mag = C_D(%.4f) * q_bar(%.1f) * S(%.1f) = %.1f N\n",
               C_D, q_bar, WING_AREA, D_mag);
        printf("  lift_dir=(%.4f, %.4f, %.4f)\n", lift_dir.x, lift_dir.y, lift_dir.z);
        printf("  F_lift=(%.1f, %.1f, %.1f) N\n", F_lift.x, F_lift.y, F_lift.z);
        printf("  F_drag=(%.1f, %.1f, %.1f) N (opposite to vel)\n", F_drag.x, F_drag.y, F_drag.z);
    }

    // ========================================================================
    // 8. Compute THRUST force
    // ========================================================================
    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("\n  --- THRUST ---\n");
    }
    float T_mag = compute_thrust(throttle, V);
    Vec3 F_thrust = mul3(forward, T_mag);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("  F_thrust=(%.1f, %.1f, %.1f) N (along forward)\n",
               F_thrust.x, F_thrust.y, F_thrust.z);
    }

    // ========================================================================
    // 9. Gravity (world frame)
    // ========================================================================
    Vec3 F_gravity = vec3(0, 0, -MASS * GRAVITY);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- GRAVITY ---\n");
        printf("  F_gravity=(%.1f, %.1f, %.1f) N\n", F_gravity.x, F_gravity.y, F_gravity.z);
    }

    // ========================================================================
    // 10. Total force → linear acceleration
    // ========================================================================
    Vec3 F_aero = add3(F_lift, F_drag);
    Vec3 F_aero_thrust = add3(F_aero, F_thrust);
    Vec3 F_total = add3(F_aero_thrust, F_gravity);
    deriv->v_dot = mul3(F_total, INV_MASS);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- TOTAL FORCE & ACCELERATION ---\n");
        printf("  F_aero (lift+drag)=(%.1f, %.1f, %.1f) N\n", F_aero.x, F_aero.y, F_aero.z);
        printf("  F_aero+thrust=(%.1f, %.1f, %.1f) N\n", F_aero_thrust.x, F_aero_thrust.y, F_aero_thrust.z);
        printf("  F_total=(%.1f, %.1f, %.1f) N\n", F_total.x, F_total.y, F_total.z);
        printf("  |F_total|=%.1f N\n", norm3(F_total));
        printf("  v_dot = F/m = (%.3f, %.3f, %.3f) m/s^2\n", deriv->v_dot.x, deriv->v_dot.y, deriv->v_dot.z);
        printf("  |v_dot|=%.3f m/s^2 = %.3f g\n", norm3(deriv->v_dot), norm3(deriv->v_dot) / GRAVITY);

        // Break down vertical component
        printf("  v_dot.z=%.3f m/s^2 (%s)\n", deriv->v_dot.z,
               deriv->v_dot.z > 0 ? "accelerating UP" : "accelerating DOWN");

        // What's contributing to vertical acceleration?
        printf("  Vertical breakdown: lift_z=%.1f + drag_z=%.1f + thrust_z=%.1f + grav_z=%.1f = %.1f N\n",
               F_lift.z, F_drag.z, F_thrust.z, F_gravity.z, F_total.z);
    }

    // ========================================================================
    // 11. Compute aerodynamic MOMENTS (body frame)
    // ========================================================================
    // Body angular rates
    float p = state->omega.x;  // roll rate
    float q = state->omega.y;  // pitch rate
    float r = state->omega.z;  // yaw rate

    // Non-dimensional rates for damping derivatives
    float p_hat = p * WINGSPAN / (2.0f * V);
    float q_hat = q * CHORD / (2.0f * V);
    float r_hat = r * WINGSPAN / (2.0f * V);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- ANGULAR RATES ---\n");
        printf("  p=%.4f, q=%.4f, r=%.4f rad/s (body: roll, pitch, yaw)\n", p, q, r);
        printf("  p_hat=%.6f, q_hat=%.6f, r_hat=%.6f (non-dimensional)\n", p_hat, q_hat, r_hat);
    }

    // Rolling moment coefficient (Cl)
    // Components: dihedral effect + roll damping + aileron control + rudder coupling
    float Cl_beta = CL_BETA * beta;
    float Cl_p = CL_P * p_hat;
    float Cl_da = CL_DELTA_A * delta_a;
    float Cl_dr = CL_DELTA_R * delta_r;
    float Cl = Cl_beta + Cl_p + Cl_da + Cl_dr;

    // Pitching moment coefficient (Cm)
    // Components: static stability + pitch damping + elevator control
    float Cm_0 = CM_0;  // Trim offset
    float Cm_alpha = CM_ALPHA * alpha;
    float Cm_q = CM_Q * q_hat;
    float Cm_de = CM_DELTA_E * delta_e;
    float Cm = Cm_0 + Cm_alpha + Cm_q + Cm_de;

    // Yawing moment coefficient (Cn)
    // Components: weathervane stability + yaw damping + rudder control + adverse yaw
    float Cn_beta = CN_BETA * beta;
    float Cn_r = CN_R * r_hat;
    float Cn_dr = CN_DELTA_R * delta_r;
    float Cn_da = CN_DELTA_A * delta_a;
    float Cn = Cn_beta + Cn_r + Cn_dr + Cn_da;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- MOMENT COEFFICIENTS ---\n");
        printf("  Cl = CL_BETA*beta(%.6f) + CL_P*p_hat(%.6f) + CL_DELTA_A*da(%.6f) + CL_DELTA_R*dr(%.6f) = %.6f\n",
               Cl_beta, Cl_p, Cl_da, Cl_dr, Cl);
        printf("  Cm = CM_0(%.6f) + CM_ALPHA*alpha(%.6f) + CM_Q*q_hat(%.6f) + CM_DELTA_E*de(%.6f) = %.6f\n",
               Cm_0, Cm_alpha, Cm_q, Cm_de, Cm);
        printf("       CM_0=%.4f (trim), CM_ALPHA=%.2f, alpha=%.4f rad -> Cm_alpha=%.6f\n", CM_0, CM_ALPHA, alpha, Cm_alpha);
        printf("       (alpha>0 means nose ABOVE vel, CM_ALPHA<0 means nose-down restoring moment)\n");
        printf("       (Cm_alpha %.6f is %s)\n", Cm_alpha,
               Cm_alpha > 0 ? "nose-UP moment" : Cm_alpha < 0 ? "nose-DOWN moment" : "zero");
        printf("  Cn = CN_BETA*beta(%.6f) + CN_R*r_hat(%.6f) + CN_DELTA_R*dr(%.6f) + CN_DELTA_A*da(%.6f) = %.6f\n",
               Cn_beta, Cn_r, Cn_dr, Cn_da, Cn);
    }

    // Convert to dimensional moments (N⋅m)
    // Note: Cm sign convention is for aircraft Z-down frame (positive Cm = nose up)
    // In our Z-up frame, positive omega.y = nose DOWN, so we negate Cm
    float L_moment = Cl * q_bar * WING_AREA * WINGSPAN;  // Roll moment
    float M_moment = -Cm * q_bar * WING_AREA * CHORD;    // Pitch moment (negated for Z-up frame)
    float N_moment = Cn * q_bar * WING_AREA * WINGSPAN;  // Yaw moment

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- DIMENSIONAL MOMENTS ---\n");
        printf("  L_moment (roll) = Cl(%.6f) * q_bar(%.1f) * S(%.1f) * b(%.1f) = %.1f N⋅m\n",
               Cl, q_bar, WING_AREA, WINGSPAN, L_moment);
        printf("  M_moment (pitch) = -Cm(%.6f) * q_bar(%.1f) * S(%.1f) * c(%.2f) = %.1f N⋅m\n",
               Cm, q_bar, WING_AREA, CHORD, M_moment);
        printf("       Note: M_moment negated because our Z is up (positive omega.y = nose DOWN)\n");
        printf("       Cm=%.6f -> -Cm=%.6f -> M_moment=%.1f (will cause omega.y to %s)\n",
               Cm, -Cm, M_moment, M_moment > 0 ? "INCREASE (nose DOWN)" : "DECREASE (nose UP)");
        printf("  N_moment (yaw) = Cn(%.6f) * q_bar(%.1f) * S(%.1f) * b(%.1f) = %.1f N⋅m\n",
               Cn, q_bar, WING_AREA, WINGSPAN, N_moment);
    }

    // ========================================================================
    // 12. Angular acceleration (Euler's equations)
    // ========================================================================
    // τ = I⋅α + ω × (I⋅ω)  →  α = I⁻¹(τ - ω × (I⋅ω))
    // For diagonal inertia tensor, the gyroscopic coupling terms are:
    // (I_yy - I_zz) * q * r  for roll
    // (I_zz - I_xx) * r * p  for pitch
    // (I_xx - I_yy) * p * q  for yaw

    float gyro_roll = (IYY - IZZ) * q * r;
    float gyro_pitch = (IZZ - IXX) * r * p;
    float gyro_yaw = (IXX - IYY) * p * q;

    deriv->w_dot.x = (L_moment + gyro_roll) / IXX;
    deriv->w_dot.y = (M_moment + gyro_pitch) / IYY;
    deriv->w_dot.z = (N_moment + gyro_yaw) / IZZ;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- ANGULAR ACCELERATION (Euler's equations) ---\n");
        printf("  Gyroscopic: roll=%.3f, pitch=%.3f, yaw=%.3f N⋅m\n", gyro_roll, gyro_pitch, gyro_yaw);
        printf("  I = (Ixx=%.0f, Iyy=%.0f, Izz=%.0f) kg⋅m^2\n", IXX, IYY, IZZ);
        printf("  w_dot.x (roll)  = (L=%.1f + gyro=%.3f) / Ixx = %.6f rad/s^2 = %.3f deg/s^2\n",
               L_moment, gyro_roll, deriv->w_dot.x, deriv->w_dot.x * RAD_TO_DEG);
        printf("  w_dot.y (pitch) = (M=%.1f + gyro=%.3f) / Iyy = %.6f rad/s^2 = %.3f deg/s^2\n",
               M_moment, gyro_pitch, deriv->w_dot.y, deriv->w_dot.y * RAD_TO_DEG);
        printf("  w_dot.z (yaw)   = (N=%.1f + gyro=%.3f) / Izz = %.6f rad/s^2 = %.3f deg/s^2\n",
               N_moment, gyro_yaw, deriv->w_dot.z, deriv->w_dot.z * RAD_TO_DEG);
        printf("  w_dot.y=%.6f means omega.y will %s -> nose will pitch %s\n",
               deriv->w_dot.y,
               deriv->w_dot.y > 0 ? "INCREASE" : "DECREASE",
               deriv->w_dot.y > 0 ? "DOWN" : "UP");
    }

    // ========================================================================
    // 13. Quaternion kinematics
    // ========================================================================
    // q_dot = 0.5 * q * [0, ω]  where ω is angular velocity in body frame
    Quat omega_q = {0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->ori, omega_q);
    deriv->q_dot.w = 0.5f * q_dot.w;
    deriv->q_dot.x = 0.5f * q_dot.x;
    deriv->q_dot.y = 0.5f * q_dot.y;
    deriv->q_dot.z = 0.5f * q_dot.z;

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("\n  --- QUATERNION KINEMATICS ---\n");
        printf("  omega_q=(%.4f, %.4f, %.4f, %.4f)\n", omega_q.w, omega_q.x, omega_q.y, omega_q.z);
        printf("  q_dot (before 0.5)=(%.6f, %.6f, %.6f, %.6f)\n", q_dot.w, q_dot.x, q_dot.y, q_dot.z);
        printf("  q_dot (final)=(%.6f, %.6f, %.6f, %.6f)\n",
               deriv->q_dot.w, deriv->q_dot.x, deriv->q_dot.y, deriv->q_dot.z);
    }

    // ========================================================================
    // 14. Position derivative = velocity
    // ========================================================================
    deriv->vel = state->vel;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- DERIVATIVE SUMMARY ---\n");
        printf("  vel = (%.2f, %.2f, %.2f) m/s\n", deriv->vel.x, deriv->vel.y, deriv->vel.z);
        printf("  v_dot = (%.3f, %.3f, %.3f) m/s^2\n", deriv->v_dot.x, deriv->v_dot.y, deriv->v_dot.z);
        printf("  q_dot = (%.6f, %.6f, %.6f, %.6f)\n",
               deriv->q_dot.w, deriv->q_dot.x, deriv->q_dot.y, deriv->q_dot.z);
        printf("  w_dot = (%.6f, %.6f, %.6f) rad/s^2\n", deriv->w_dot.x, deriv->w_dot.y, deriv->w_dot.z);
    }
}

// ============================================================================
// RK4 INTEGRATION
// ============================================================================

static inline void rk4_step(Plane* state, float* actions, float dt) {
    StateDerivative k1, k2, k3, k4;
    Plane temp;

    if (DEBUG_REALISTIC >= 5) {
        printf("\n========== RK4 STEP (dt=%.4f) ==========\n", dt);
    }

    // k1: derivative at current state
    _realistic_rk4_stage = 0;
    compute_derivatives(state, actions, dt, &k1);

    if (DEBUG_REALISTIC >= 5) {
        printf("\n  k1: v_dot=(%.3f,%.3f,%.3f) w_dot=(%.6f,%.6f,%.6f)\n",
               k1.v_dot.x, k1.v_dot.y, k1.v_dot.z, k1.w_dot.x, k1.w_dot.y, k1.w_dot.z);
    }

    // k2: derivative at state + k1*dt/2
    _realistic_rk4_stage = 1;
    step_temp(state, &k1, dt * 0.5f, &temp);
    compute_derivatives(&temp, actions, dt, &k2);

    if (DEBUG_REALISTIC >= 5) {
        printf("  k2: v_dot=(%.3f,%.3f,%.3f) w_dot=(%.6f,%.6f,%.6f)\n",
               k2.v_dot.x, k2.v_dot.y, k2.v_dot.z, k2.w_dot.x, k2.w_dot.y, k2.w_dot.z);
    }

    // k3: derivative at state + k2*dt/2
    _realistic_rk4_stage = 2;
    step_temp(state, &k2, dt * 0.5f, &temp);
    compute_derivatives(&temp, actions, dt, &k3);

    if (DEBUG_REALISTIC >= 5) {
        printf("  k3: v_dot=(%.3f,%.3f,%.3f) w_dot=(%.6f,%.6f,%.6f)\n",
               k3.v_dot.x, k3.v_dot.y, k3.v_dot.z, k3.w_dot.x, k3.w_dot.y, k3.w_dot.z);
    }

    // k4: derivative at state + k3*dt
    _realistic_rk4_stage = 3;
    step_temp(state, &k3, dt, &temp);
    compute_derivatives(&temp, actions, dt, &k4);

    if (DEBUG_REALISTIC >= 5) {
        printf("  k4: v_dot=(%.3f,%.3f,%.3f) w_dot=(%.6f,%.6f,%.6f)\n",
               k4.v_dot.x, k4.v_dot.y, k4.v_dot.z, k4.w_dot.x, k4.w_dot.y, k4.w_dot.z);
    }

    _realistic_rk4_stage = 0;  // Reset for next step

    // Weighted average: (k1 + 2*k2 + 2*k3 + k4) / 6
    float dt_6 = dt / 6.0f;

    // Save pre-update values for debug
    Vec3 old_vel = state->vel;
    Vec3 old_omega = state->omega;
    Quat old_ori = state->ori;

    // Position update
    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;

    // Velocity update
    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;

    // Quaternion update
    state->ori.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->ori.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->ori.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->ori.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;

    // Angular velocity update
    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;

    // Normalize quaternion to prevent drift
    quat_normalize(&state->ori);

    if (DEBUG_REALISTIC >= 5) {
        printf("\n  --- RK4 WEIGHTED AVERAGE ---\n");
        printf("  vel: (%.2f,%.2f,%.2f) -> (%.2f,%.2f,%.2f) delta=(%.3f,%.3f,%.3f)\n",
               old_vel.x, old_vel.y, old_vel.z,
               state->vel.x, state->vel.y, state->vel.z,
               state->vel.x - old_vel.x, state->vel.y - old_vel.y, state->vel.z - old_vel.z);
        printf("  omega: (%.4f,%.4f,%.4f) -> (%.4f,%.4f,%.4f) delta=(%.6f,%.6f,%.6f)\n",
               old_omega.x, old_omega.y, old_omega.z,
               state->omega.x, state->omega.y, state->omega.z,
               state->omega.x - old_omega.x, state->omega.y - old_omega.y, state->omega.z - old_omega.z);
        printf("  ori: (%.4f,%.4f,%.4f,%.4f) -> (%.4f,%.4f,%.4f,%.4f)\n",
               old_ori.w, old_ori.x, old_ori.y, old_ori.z,
               state->ori.w, state->ori.x, state->ori.y, state->ori.z);
    }
}

// ============================================================================
// MAIN INTERFACE: step_plane_with_physics_realistic()
// ============================================================================

static inline void step_plane_with_physics_realistic(Plane *p, float *actions, float dt) {
    _realistic_step_count++;

    if (DEBUG_REALISTIC >= 1) {
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║ REALISTIC PHYSICS STEP %d (dt=%.4f)                                           \n", _realistic_step_count, dt);
        printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    }

    // Save previous velocity for G-force calculation
    p->prev_vel = p->vel;

    if (DEBUG_REALISTIC >= 1) {
        printf("\n=== BEFORE RK4 ===\n");
        printf("pos=(%.1f, %.1f, %.1f) alt=%.1f m\n", p->pos.x, p->pos.y, p->pos.z, p->pos.z);
        printf("vel=(%.2f, %.2f, %.2f) |V|=%.2f m/s\n", p->vel.x, p->vel.y, p->vel.z, norm3(p->vel));
        printf("ori=(w=%.4f, x=%.4f, y=%.4f, z=%.4f)\n", p->ori.w, p->ori.x, p->ori.y, p->ori.z);
        printf("omega=(%.4f, %.4f, %.4f) rad/s\n", p->omega.x, p->omega.y, p->omega.z);

        // Compute pitch angle
        Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
        float pitch = asinf(-forward.z) * RAD_TO_DEG;
        Vec3 vel_norm = normalize3(p->vel);
        float vel_pitch = asinf(vel_norm.z) * RAD_TO_DEG;
        float alpha = compute_aoa(p) * RAD_TO_DEG;

        printf("pitch=%.2f deg (nose %s), vel_pitch=%.2f deg (%s), alpha=%.2f deg\n",
               pitch, pitch > 0 ? "UP" : "DOWN",
               vel_pitch, vel_pitch > 0 ? "CLIMBING" : "DESCENDING",
               alpha);
        printf("actions=[thr=%.2f, elev=%.2f, ail=%.2f, rud=%.2f]\n",
               actions[0], actions[1], actions[2], actions[3]);
    }

    // Clamp actions to [-1, 1]
    float clamped_actions[4];
    for (int i = 0; i < 4; i++) {
        clamped_actions[i] = clampf(actions[i], -1.0f, 1.0f);
    }

    // Run RK4 integration
    rk4_step(p, clamped_actions, dt);

    // Update throttle state for display/logging
    p->throttle = (clamped_actions[0] + 1.0f) * 0.5f;

    // Clamp angular velocity to prevent runaway
    float old_omega_y = p->omega.y;
    p->omega.x = clampf(p->omega.x, -5.0f, 5.0f);  // ~286 deg/s max roll
    p->omega.y = clampf(p->omega.y, -5.0f, 5.0f);  // ~286 deg/s max pitch
    p->omega.z = clampf(p->omega.z, -2.0f, 2.0f);  // ~115 deg/s max yaw (less authority)

    if (DEBUG_REALISTIC >= 1 && old_omega_y != p->omega.y) {
        printf("  WARNING: omega.y clamped from %.4f to %.4f\n", old_omega_y, p->omega.y);
    }

    // ========================================================================
    // G-FORCE CALCULATION
    // ========================================================================
    // G-force = aerodynamic acceleration along body-up axis / g
    // In level flight, lift ≈ weight, so g_force ≈ 1.0

    Vec3 dv = sub3(p->vel, p->prev_vel);
    Vec3 accel = mul3(dv, 1.0f / dt);
    Vec3 body_up = quat_rotate(p->ori, vec3(0, 0, 1));

    // Total acceleration in body-up direction, converted to G
    // Add 1G because we're measuring from inertial frame (gravity already in accel)
    float accel_up = dot3(accel, body_up);
    p->g_force = accel_up * INV_GRAVITY + 1.0f;

    if (DEBUG_REALISTIC >= 1) {
        printf("\n=== G-FORCE CALCULATION ===\n");
        printf("dv=(%.3f, %.3f, %.3f) over dt=%.4f\n", dv.x, dv.y, dv.z, dt);
        printf("accel=(%.3f, %.3f, %.3f) m/s^2\n", accel.x, accel.y, accel.z);
        printf("body_up=(%.4f, %.4f, %.4f)\n", body_up.x, body_up.y, body_up.z);
        printf("accel·body_up=%.3f m/s^2 / g=%.3f + 1.0 = %.3f G\n",
               accel_up, accel_up * INV_GRAVITY, p->g_force);
    }

    // ========================================================================
    // G-LIMIT ENFORCEMENT (clamp velocity change, energy-conserving)
    // ========================================================================
    // If G-force exceeds limits, reduce the velocity change to stay within limits.
    // IMPORTANT: The correction must be perpendicular to velocity to preserve kinetic energy.
    // If body_up has a component along velocity, applying the full correction would
    // change speed, violating conservation of energy.

    float speed_before_glimit = norm3(p->vel);

    if (p->g_force > G_LIMIT_POS) {
        // Positive G exceeded - reduce upward acceleration
        float excess_g = p->g_force - G_LIMIT_POS;
        float excess_accel = excess_g * GRAVITY;

        if (DEBUG_REALISTIC >= 1) {
            printf("G-LIMIT: +%.2f G exceeded limit +%.1f by %.2f G, reducing vel\n",
                   p->g_force, G_LIMIT_POS, excess_g);
        }

        // Calculate the correction vector
        Vec3 correction = mul3(body_up, excess_accel * dt);

        // Project out the component along velocity to preserve speed (energy)
        Vec3 vel_norm = normalize3(p->vel);
        float correction_along_vel = dot3(correction, vel_norm);
        Vec3 correction_perp = sub3(correction, mul3(vel_norm, correction_along_vel));

        // Apply only the perpendicular correction
        p->vel = sub3(p->vel, correction_perp);
        p->g_force = G_LIMIT_POS;

    } else if (p->g_force < -G_LIMIT_NEG) {
        // Negative G exceeded - reduce downward acceleration
        float deficit_g = -G_LIMIT_NEG - p->g_force;
        float deficit_accel = deficit_g * GRAVITY;

        if (DEBUG_REALISTIC >= 1) {
            printf("G-LIMIT: %.2f G exceeded limit -%.1f by %.2f G, reducing vel\n",
                   p->g_force, G_LIMIT_NEG, -deficit_g);
        }

        // Calculate the correction vector
        Vec3 correction = mul3(body_up, deficit_accel * dt);

        // Project out the component along velocity to preserve speed (energy)
        Vec3 vel_norm = normalize3(p->vel);
        float correction_along_vel = dot3(correction, vel_norm);
        Vec3 correction_perp = sub3(correction, mul3(vel_norm, correction_along_vel));

        // Apply only the perpendicular correction
        p->vel = add3(p->vel, correction_perp);
        p->g_force = -G_LIMIT_NEG;
    }

    // Verify energy was preserved (speed should not have changed)
    if (DEBUG_REALISTIC >= 1) {
        float speed_after_glimit = norm3(p->vel);
        if (fabsf(speed_after_glimit - speed_before_glimit) > 0.01f) {
            printf("WARNING: G-limit changed speed from %.2f to %.2f!\n",
                   speed_before_glimit, speed_after_glimit);
        }
    }

    // Update yaw_from_rudder for backward compatibility
    // In momentum physics, this approximates sideslip angle
    p->yaw_from_rudder = compute_sideslip(p);

    if (DEBUG_REALISTIC >= 1) {
        printf("\n=== AFTER RK4 ===\n");
        printf("pos=(%.1f, %.1f, %.1f) alt=%.1f m (Δalt=%.2f m)\n",
               p->pos.x, p->pos.y, p->pos.z, p->pos.z, p->pos.z - (p->pos.z - p->vel.z * dt));
        printf("vel=(%.2f, %.2f, %.2f) |V|=%.2f m/s\n", p->vel.x, p->vel.y, p->vel.z, norm3(p->vel));
        printf("ori=(w=%.4f, x=%.4f, y=%.4f, z=%.4f)\n", p->ori.w, p->ori.x, p->ori.y, p->ori.z);
        printf("omega=(%.4f, %.4f, %.4f) rad/s = (%.2f, %.2f, %.2f) deg/s\n",
               p->omega.x, p->omega.y, p->omega.z,
               p->omega.x * RAD_TO_DEG, p->omega.y * RAD_TO_DEG, p->omega.z * RAD_TO_DEG);
        printf("g_force=%.2f G (limits: +%.1f/-%.1f)\n", p->g_force, G_LIMIT_POS, G_LIMIT_NEG);

        // Compute final pitch and alpha
        Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
        float pitch = asinf(-forward.z) * RAD_TO_DEG;
        float alpha = compute_aoa(p) * RAD_TO_DEG;
        Vec3 vel_norm = normalize3(p->vel);
        float vel_pitch = asinf(vel_norm.z) * RAD_TO_DEG;

        printf("final: pitch=%.2f deg, vel_pitch=%.2f deg, alpha=%.2f deg\n",
               pitch, vel_pitch, alpha);

        // Key insight: what's happening to orientation vs velocity?
        printf("\n=== STEP SUMMARY ===\n");
        printf("vel.z changed: %.3f -> %.3f (Δ=%.3f m/s, %s)\n",
               p->prev_vel.z, p->vel.z, p->vel.z - p->prev_vel.z,
               p->vel.z > p->prev_vel.z ? "CLIMBING MORE" : "DIVING MORE");
        printf("omega.y = %.4f rad/s = %.2f deg/s (nose pitching %s)\n",
               p->omega.y, p->omega.y * RAD_TO_DEG,
               p->omega.y > 0 ? "DOWN" : "UP");
    }

    if (DEBUG >= 10) {
        float V = norm3(p->vel);
        float alpha = compute_aoa(p) * RAD_TO_DEG;
        float beta = compute_sideslip(p) * RAD_TO_DEG;
        printf("=== REALISTIC PHYSICS ===\n");
        printf("speed=%.1f m/s\n", V);
        printf("throttle=%.2f\n", p->throttle);
        printf("alpha=%.2f deg, beta=%.2f deg\n", alpha, beta);
        printf("omega=(%.3f, %.3f, %.3f) rad/s\n", p->omega.x, p->omega.y, p->omega.z);
        printf("g_force=%.2f g (limit=+%.1f/-%.1f)\n", p->g_force, G_LIMIT_POS, G_LIMIT_NEG);
    }
}

// ============================================================================
// RESET FUNCTION (Realistic)
// ============================================================================

static inline void reset_plane_realistic(Plane *p, Vec3 pos, Vec3 vel) {
    p->pos = pos;
    p->vel = vel;
    p->prev_vel = vel;  // Initialize to current vel (no acceleration at start)
    p->omega = vec3(0, 0, 0);  // No angular velocity at start
    p->ori = quat(1, 0, 0, 0);
    p->throttle = 0.5f;
    p->g_force = 1.0f;  // 1G at start (level flight)
    p->yaw_from_rudder = 0.0f;
    p->fire_cooldown = 0;

    // Reset debug counter
    _realistic_step_count = 0;

    if (DEBUG_REALISTIC >= 1) {
        printf("\n=== RESET_PLANE_REALISTIC ===\n");
        printf("pos=(%.1f, %.1f, %.1f)\n", pos.x, pos.y, pos.z);
        printf("vel=(%.2f, %.2f, %.2f) |V|=%.2f m/s\n", vel.x, vel.y, vel.z, norm3(vel));
        printf("ori=(1, 0, 0, 0) (identity)\n");
        printf("omega=(0, 0, 0)\n");
    }
}

#endif // PHYSICS_REALISTIC_H
