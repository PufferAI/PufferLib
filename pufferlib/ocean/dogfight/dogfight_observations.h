// dogfight_observations.h - Observation computation for dogfight environment
// Extracted from dogfight.h to reduce file size
//
// Observation Schemes (for realistic physics - physics mode 1):
//   Scheme 0: OBS_MOMENTUM       - Baseline (15 obs)
//   Scheme 1: OBS_MOMENTUM_BETA  - + sideslip angle (16 obs)
//   Scheme 2: OBS_MOMENTUM_GFORCE - + G-force (16 obs)
//   Scheme 3: OBS_MOMENTUM_FULL  - + sideslip + G + throttle + tgt rates (19 obs)
//   Scheme 4: OBS_MINIMAL        - stripped down essentials (11 obs)
//   Scheme 5: OBS_CARTESIAN      - cartesian target position (15 obs)
//   Scheme 6: OBS_DRONE_STYLE    - + quaternion + up vector (22 obs)
//   Scheme 7: OBS_QBAR           - + dynamic pressure (16 obs)
//   Scheme 8: OBS_KITCHEN_SINK   - everything (25 obs)

#ifndef DOGFIGHT_OBSERVATIONS_H
#define DOGFIGHT_OBSERVATIONS_H

// Requires: flightlib.h (Vec3, Quat, math), Dogfight struct defined before include

// Normalization constants
#define MAX_OMEGA 3.0f              // ~172 deg/s, reasonable for aggressive maneuvering
#define INV_MAX_OMEGA (1.0f / MAX_OMEGA)
#define MAX_AOA 0.5f                // ~28 deg, beyond this is deep stall
#define INV_MAX_AOA (1.0f / MAX_AOA)
#define MAX_SIDESLIP 0.5f           // ~28 degrees
#define INV_MAX_SIDESLIP (1.0f / MAX_SIDESLIP)
#define MAX_QBAR 38281.0f           // 0.5 * 1.225 * 250^2 at sea level, max speed
#define INV_MAX_QBAR (1.0f / MAX_QBAR)
#define MAX_RANGE 2000.0f           // Normalization range for target distance
#define INV_MAX_RANGE (1.0f / MAX_RANGE)

// ============================================================================
// Scheme 0: OBS_MOMENTUM - Baseline (15 obs)
// ============================================================================
// Body-frame velocity + omega + AoA + energy + target spherical + tactical
// [0-2]   Body-frame velocity (forward speed, sideslip, climb rate)
// [3-5]   Angular velocity (roll rate, pitch rate, yaw rate)
// [6]     Angle of attack
// [7-8]   Altitude + own energy
// [9-12]  Target spherical (azimuth, elevation, range, closure)
// [13-14] Tactical (energy advantage, target aspect)
void compute_obs_momentum(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // === OWN FLIGHT STATE ===
    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    float speed = norm3(p->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(p->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Energy state
    float potential = p->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // === TARGET STATE ===
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Closure rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // === TACTICAL ===
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    float opp_speed = norm3(o->vel);
    float opp_potential = o->pos.z * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (9 obs)
    env->observations[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);  // Forward speed [0,1]
    env->observations[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f); // Sideslip [-1,1]
    env->observations[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f); // Climb rate [-1,1]
    env->observations[i++] = clampf(p->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f); // Roll rate [-1,1]
    env->observations[i++] = clampf(p->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f); // Pitch rate [-1,1]
    env->observations[i++] = clampf(p->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f); // Yaw rate [-1,1]
    env->observations[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);          // AoA [-1,1]
    env->observations[i++] = potential;                                        // Altitude [0,1]
    env->observations[i++] = own_energy;                                       // Own energy [0,1]

    // Target state - spherical (4 obs)
    env->observations[i++] = target_az * INV_PI;                               // Azimuth [-1,1]
    env->observations[i++] = target_el * INV_HALF_PI;                          // Elevation [-1,1]
    env->observations[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);        // Range [0,1]
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);    // Closure [-1,1]

    // Tactical (2 obs)
    env->observations[i++] = energy_advantage;                                 // Energy advantage [-1,1]
    env->observations[i++] = target_aspect;                                    // Aspect [-1,1]
    // OBS_SIZE = 15
}

// ============================================================================
// Scheme 1: OBS_MOMENTUM_BETA - + sideslip angle (16 obs)
// ============================================================================
// Hypothesis: Explicit sideslip angle helps coordinated flight
void compute_obs_momentum_beta(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    float speed = norm3(p->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(p->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Sideslip angle (beta)
    // beta = asin(vy / speed), positive = nose left of velocity
    float beta = 0.0f;
    if (speed > 1.0f) {
        beta = asinf(clampf(vel_body.y / speed, -1.0f, 1.0f));
    }

    // Energy state
    float potential = p->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // Target state
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Tactical
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    float opp_speed = norm3(o->vel);
    float opp_potential = o->pos.z * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (9 obs - same as MOMENTUM)
    env->observations[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);
    env->observations[i++] = potential;
    env->observations[i++] = own_energy;

    // NEW: Sideslip angle
    env->observations[i++] = clampf(beta * INV_MAX_SIDESLIP, -1.0f, 1.0f);    // Beta [-1,1]

    // Target state (4 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Tactical (2 obs)
    env->observations[i++] = energy_advantage;
    env->observations[i++] = target_aspect;
    // OBS_SIZE = 16
}

// ============================================================================
// Scheme 2: OBS_MOMENTUM_GFORCE - + G-force (16 obs)
// ============================================================================
// Hypothesis: G-force awareness enables better high-G maneuvering
void compute_obs_momentum_gforce(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    float speed = norm3(p->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(p->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Energy state
    float potential = p->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // G-force normalization: 0G=0, 1G=0.2, 5G=1.0, -2.5G=-0.5
    float g_norm = clampf(p->g_force / 5.0f, -0.5f, 1.0f);

    // Target state
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Tactical
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    float opp_speed = norm3(o->vel);
    float opp_potential = o->pos.z * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (9 obs)
    env->observations[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);
    env->observations[i++] = potential;
    env->observations[i++] = own_energy;

    // NEW: G-force
    env->observations[i++] = g_norm;                                           // G-force [-0.5,1]

    // Target state (4 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Tactical (2 obs)
    env->observations[i++] = energy_advantage;
    env->observations[i++] = target_aspect;
    // OBS_SIZE = 16
}

// ============================================================================
// Scheme 3: OBS_MOMENTUM_FULL - + sideslip + G + throttle + target rates (19 obs)
// ============================================================================
// Hypothesis: Maximum relevant information is optimal
void compute_obs_momentum_full(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    float speed = norm3(p->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(p->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Sideslip angle
    float beta = 0.0f;
    if (speed > 1.0f) {
        beta = asinf(clampf(vel_body.y / speed, -1.0f, 1.0f));
    }

    // Energy state
    float potential = p->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // G-force
    float g_norm = clampf(p->g_force / 5.0f, -0.5f, 1.0f);

    // Target state
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Tactical
    float opp_speed = norm3(o->vel);
    float opp_potential = o->pos.z * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (9 obs)
    env->observations[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);
    env->observations[i++] = potential;
    env->observations[i++] = own_energy;

    // Extended own state (3 obs)
    env->observations[i++] = clampf(beta * INV_MAX_SIDESLIP, -1.0f, 1.0f);     // Beta
    env->observations[i++] = g_norm;                                            // G-force
    env->observations[i++] = p->throttle;                                       // Throttle [0,1]

    // Target state (4 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Target angular rates (2 obs) - for predicting opponent maneuvers
    env->observations[i++] = clampf(o->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // Target pitch rate
    env->observations[i++] = clampf(o->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // Target roll rate

    // Energy advantage (1 obs)
    env->observations[i++] = energy_advantage;
    // OBS_SIZE = 19
}

// ============================================================================
// Scheme 4: OBS_MINIMAL - stripped down essentials (11 obs)
// ============================================================================
// Hypothesis: Simpler observations learn faster and generalize better
void compute_obs_minimal(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    float speed = norm3(p->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(p->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Altitude
    float potential = p->pos.z * INV_WORLD_MAX_Z;

    // Target state
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Energy advantage
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    float opp_speed = norm3(o->vel);
    float opp_potential = o->pos.z * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Minimal own state (6 obs)
    env->observations[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);   // Forward speed
    env->observations[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);           // AoA
    env->observations[i++] = clampf(p->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // Roll rate
    env->observations[i++] = clampf(p->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // Pitch rate
    env->observations[i++] = clampf(p->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);  // Yaw rate
    env->observations[i++] = potential;                                         // Altitude

    // Target (4 obs)
    env->observations[i++] = target_az * INV_PI;                                // Azimuth
    env->observations[i++] = target_el * INV_HALF_PI;                           // Elevation
    env->observations[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);         // Range
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);     // Closure

    // Tactical (1 obs)
    env->observations[i++] = energy_advantage;
    // OBS_SIZE = 11
}

// ============================================================================
// Scheme 5: OBS_CARTESIAN - cartesian target position (15 obs)
// ============================================================================
// Hypothesis: Cartesian target coords better for lead computing
void compute_obs_cartesian(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    float speed = norm3(p->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(p->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Energy state
    float potential = p->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // Target in body frame - CARTESIAN instead of spherical
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);

    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Tactical
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    float opp_speed = norm3(o->vel);
    float opp_potential = o->pos.z * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (9 obs)
    env->observations[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);
    env->observations[i++] = potential;
    env->observations[i++] = own_energy;

    // Target state - CARTESIAN (4 obs)
    env->observations[i++] = clampf(rel_pos_body.x * INV_MAX_RANGE, -1.0f, 1.0f);  // Target X (forward)
    env->observations[i++] = clampf(rel_pos_body.y * INV_MAX_RANGE, -1.0f, 1.0f);  // Target Y (right)
    env->observations[i++] = clampf(rel_pos_body.z * INV_MAX_RANGE, -1.0f, 1.0f);  // Target Z (up)
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Tactical (2 obs)
    env->observations[i++] = energy_advantage;
    env->observations[i++] = target_aspect;
    // OBS_SIZE = 15
}

// ============================================================================
// Scheme 6: OBS_DRONE_STYLE - + quaternion + up vector (22 obs)
// ============================================================================
// Hypothesis: Quaternion + up vector (drone_race style) helps 3D maneuvers
void compute_obs_drone_style(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    float speed = norm3(p->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(p->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Energy state
    float potential = p->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // Up vector in world frame (derived from quaternion)
    Vec3 world_up = quat_rotate(p->ori, vec3(0, 0, 1));

    // Target state
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Tactical
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    float opp_speed = norm3(o->vel);
    float opp_potential = o->pos.z * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (9 obs)
    env->observations[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);
    env->observations[i++] = potential;
    env->observations[i++] = own_energy;

    // Quaternion (4 obs) - raw orientation for NN to reason about 3D
    env->observations[i++] = p->ori.w;
    env->observations[i++] = p->ori.x;
    env->observations[i++] = p->ori.y;
    env->observations[i++] = p->ori.z;

    // Up vector in world frame (3 obs) - gravity-relative maneuvers
    env->observations[i++] = world_up.x;
    env->observations[i++] = world_up.y;
    env->observations[i++] = world_up.z;

    // Target state (4 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Tactical (2 obs)
    env->observations[i++] = energy_advantage;
    env->observations[i++] = target_aspect;
    // OBS_SIZE = 22
}

// ============================================================================
// Scheme 7: OBS_QBAR - + dynamic pressure (16 obs)
// ============================================================================
// Hypothesis: Dynamic pressure helps understand control authority
void compute_obs_qbar(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    float speed = norm3(p->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(p->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Energy state
    float potential = p->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // Dynamic pressure q_bar = 0.5 * rho * V^2
    // At sea level rho ≈ 1.225 kg/m³
    float rho = 1.225f;
    float q_bar = 0.5f * rho * speed * speed;
    float q_bar_norm = clampf(q_bar * INV_MAX_QBAR, 0.0f, 1.0f);

    // Target state
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Tactical
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    float opp_speed = norm3(o->vel);
    float opp_potential = o->pos.z * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (9 obs)
    env->observations[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);
    env->observations[i++] = potential;
    env->observations[i++] = own_energy;

    // Dynamic pressure (1 obs)
    env->observations[i++] = q_bar_norm;                                        // q_bar [0,1]

    // Target state (4 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Tactical (2 obs)
    env->observations[i++] = energy_advantage;
    env->observations[i++] = target_aspect;
    // OBS_SIZE = 16
}

// ============================================================================
// Scheme 8: OBS_KITCHEN_SINK - everything (25 obs)
// ============================================================================
// Hypothesis: Maximum information with everything is optimal
void compute_obs_kitchen_sink(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    float speed = norm3(p->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up_body = quat_rotate(p->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(p->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(p->vel, up_body) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Sideslip angle
    float beta = 0.0f;
    if (speed > 1.0f) {
        beta = asinf(clampf(vel_body.y / speed, -1.0f, 1.0f));
    }

    // G-force
    float g_norm = clampf(p->g_force / 5.0f, -0.5f, 1.0f);

    // Dynamic pressure
    float rho = 1.225f;
    float q_bar = 0.5f * rho * speed * speed;
    float q_bar_norm = clampf(q_bar * INV_MAX_QBAR, 0.0f, 1.0f);

    // Energy state
    float potential = p->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // Up vector in world frame
    Vec3 world_up = quat_rotate(p->ori, vec3(0, 0, 1));

    // Target state
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Energy advantage
    float opp_speed = norm3(o->vel);
    float opp_potential = o->pos.z * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Body-frame velocity (3 obs)
    env->observations[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);
    env->observations[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);

    // Angular velocity (3 obs)
    env->observations[i++] = clampf(p->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);
    env->observations[i++] = clampf(p->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);

    // Flight angles (2 obs)
    env->observations[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);
    env->observations[i++] = clampf(beta * INV_MAX_SIDESLIP, -1.0f, 1.0f);

    // Flight state (4 obs)
    env->observations[i++] = g_norm;
    env->observations[i++] = q_bar_norm;
    env->observations[i++] = potential;
    env->observations[i++] = own_energy;

    // Controls (1 obs)
    env->observations[i++] = p->throttle;

    // Quaternion (4 obs)
    env->observations[i++] = p->ori.w;
    env->observations[i++] = p->ori.x;
    env->observations[i++] = p->ori.y;
    env->observations[i++] = p->ori.z;

    // Up vector in world frame (3 obs)
    env->observations[i++] = world_up.x;
    env->observations[i++] = world_up.y;
    env->observations[i++] = world_up.z;

    // Target spherical (4 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Energy advantage (1 obs)
    env->observations[i++] = energy_advantage;
    // OBS_SIZE = 25
}

// ============================================================================
// Dispatcher function
// ============================================================================
void compute_observations(Dogfight *env) {
    switch (env->obs_scheme) {
        case OBS_MOMENTUM:        compute_obs_momentum(env); break;
        case OBS_MOMENTUM_BETA:   compute_obs_momentum_beta(env); break;
        case OBS_MOMENTUM_GFORCE: compute_obs_momentum_gforce(env); break;
        case OBS_MOMENTUM_FULL:   compute_obs_momentum_full(env); break;
        case OBS_MINIMAL:         compute_obs_minimal(env); break;
        case OBS_CARTESIAN:       compute_obs_cartesian(env); break;
        case OBS_DRONE_STYLE:     compute_obs_drone_style(env); break;
        case OBS_QBAR:            compute_obs_qbar(env); break;
        case OBS_KITCHEN_SINK:    compute_obs_kitchen_sink(env); break;
        default:                  compute_obs_momentum(env); break;
    }
}

// ============================================================================
// Debug labels for print_observations
// ============================================================================
#if DEBUG >= 5

// Scheme 0: OBS_MOMENTUM (15 obs)
static const char* DEBUG_OBS_LABELS_MOMENTUM[15] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect"
};

// Scheme 1: OBS_MOMENTUM_BETA (16 obs)
static const char* DEBUG_OBS_LABELS_MOMENTUM_BETA[16] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "beta",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect"
};

// Scheme 2: OBS_MOMENTUM_GFORCE (16 obs)
static const char* DEBUG_OBS_LABELS_MOMENTUM_GFORCE[16] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "g_force",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect"
};

// Scheme 3: OBS_MOMENTUM_FULL (19 obs)
static const char* DEBUG_OBS_LABELS_MOMENTUM_FULL[19] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "beta", "g_force", "throttle",
    "tgt_az", "tgt_el", "range", "closure",
    "tgt_pitch_r", "tgt_roll_r", "E_adv"
};

// Scheme 4: OBS_MINIMAL (11 obs)
static const char* DEBUG_OBS_LABELS_MINIMAL[11] = {
    "fwd_spd", "aoa", "roll_r", "pitch_r", "yaw_r", "altitude",
    "tgt_az", "tgt_el", "range", "closure", "E_adv"
};

// Scheme 5: OBS_CARTESIAN (15 obs)
static const char* DEBUG_OBS_LABELS_CARTESIAN[15] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "tgt_x", "tgt_y", "tgt_z", "closure",
    "E_adv", "aspect"
};

// Scheme 6: OBS_DRONE_STYLE (22 obs)
static const char* DEBUG_OBS_LABELS_DRONE_STYLE[22] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "up_x", "up_y", "up_z",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect"
};

// Scheme 7: OBS_QBAR (16 obs)
static const char* DEBUG_OBS_LABELS_QBAR[16] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "q_bar",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect"
};

// Scheme 8: OBS_KITCHEN_SINK (25 obs)
static const char* DEBUG_OBS_LABELS_KITCHEN_SINK[25] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "beta", "g_force", "q_bar", "altitude", "energy", "throttle",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "up_x", "up_y", "up_z",
    "tgt_az", "tgt_el", "range", "closure", "E_adv"
};

void print_observations(Dogfight *env) {
    const char** labels = NULL;
    int num_obs = env->obs_size;

    // Select labels based on scheme
    switch (env->obs_scheme) {
        case OBS_MOMENTUM:        labels = DEBUG_OBS_LABELS_MOMENTUM; break;
        case OBS_MOMENTUM_BETA:   labels = DEBUG_OBS_LABELS_MOMENTUM_BETA; break;
        case OBS_MOMENTUM_GFORCE: labels = DEBUG_OBS_LABELS_MOMENTUM_GFORCE; break;
        case OBS_MOMENTUM_FULL:   labels = DEBUG_OBS_LABELS_MOMENTUM_FULL; break;
        case OBS_MINIMAL:         labels = DEBUG_OBS_LABELS_MINIMAL; break;
        case OBS_CARTESIAN:       labels = DEBUG_OBS_LABELS_CARTESIAN; break;
        case OBS_DRONE_STYLE:     labels = DEBUG_OBS_LABELS_DRONE_STYLE; break;
        case OBS_QBAR:            labels = DEBUG_OBS_LABELS_QBAR; break;
        case OBS_KITCHEN_SINK:    labels = DEBUG_OBS_LABELS_KITCHEN_SINK; break;
        default:                  labels = DEBUG_OBS_LABELS_MOMENTUM; break;
    }

    printf("=== OBS (scheme %d, %d obs) ===\n", env->obs_scheme, num_obs);

    for (int i = 0; i < num_obs; i++) {
        float val = env->observations[i];

        // Determine range based on scheme and index
        bool is_01 = false;
        switch (env->obs_scheme) {
            case OBS_MOMENTUM:
                // fwd_spd(0), altitude(7), energy(8), range(11) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 11);
                break;
            case OBS_MOMENTUM_BETA:
            case OBS_MOMENTUM_GFORCE:
            case OBS_QBAR:
                // fwd_spd(0), altitude(7), energy(8), range(12) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 12);
                break;
            case OBS_MOMENTUM_FULL:
                // fwd_spd(0), altitude(7), energy(8), throttle(11), range(14) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 11 || i == 14);
                break;
            case OBS_MINIMAL:
                // fwd_spd(0), altitude(5), range(8) are [0,1]
                is_01 = (i == 0 || i == 5 || i == 8);
                break;
            case OBS_CARTESIAN:
                // fwd_spd(0), altitude(7), energy(8) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8);
                break;
            case OBS_DRONE_STYLE:
                // fwd_spd(0), altitude(7), energy(8), range(18) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 18);
                break;
            case OBS_KITCHEN_SINK:
                // fwd_spd(0), q_bar(9), altitude(10), energy(11), throttle(12), range(22) are [0,1]
                is_01 = (i == 0 || i == 9 || i == 10 || i == 11 || i == 12 || i == 22);
                break;
            default:
                break;
        }

        const char* range_str = is_01 ? "[0,1]" : "[-1,1]";
        printf("[%2d] %-12s = %+.3f  %s\n", i, labels[i], val, range_str);
    }
}
#endif // DEBUG >= 5

#endif // DOGFIGHT_OBSERVATIONS_H
