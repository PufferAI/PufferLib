// dogfight_observations.h - Observation computation for dogfight environment
// Extracted from dogfight.h to reduce file size
//
// Observation Schemes (for realistic physics - physics mode 1):
// All schemes include timer observation at the end: tick/(max_steps+1) [0,~1)
//   Scheme 0: OBS_MOMENTUM       - Baseline (16 obs)
//   Scheme 1: OBS_MOMENTUM_BETA  - + sideslip angle (17 obs)
//   Scheme 2: OBS_MOMENTUM_GFORCE - + G-force (17 obs)
//   Scheme 3: OBS_MOMENTUM_FULL  - + sideslip + G + throttle + tgt rates (20 obs)
//   Scheme 4: OBS_MINIMAL        - stripped down essentials (12 obs)
//   Scheme 5: OBS_CARTESIAN      - cartesian target position (16 obs)
//   Scheme 6: OBS_DRONE_STYLE    - + quaternion + up vector (23 obs)
//   Scheme 7: OBS_QBAR           - + dynamic pressure (17 obs)
//   Scheme 8: OBS_KITCHEN_SINK   - everything (26 obs)

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
// Generalized observation computation for self-play
// ============================================================================
// Computes observations from 'self' plane's perspective looking at 'other' plane.
// Used for both player and opponent observations.
//
// Note: Timer observation uses env->tick/max_steps which is shared between both
// planes. This is correct for self-play where both see the same episode timer.

void compute_obs_momentum_for_plane(Dogfight *env, Plane *self, Plane *other, float *obs_buffer) {
    Quat q_inv = {self->ori.w, -self->ori.x, -self->ori.y, -self->ori.z};

    // === OWN FLIGHT STATE ===
    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, self->vel);
    float speed = norm3(self->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(self->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(self->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(self->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(self->vel, up) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Energy state
    float potential = self->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // === TARGET STATE ===
    Vec3 rel_pos = sub3(other->pos, self->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Closure rate (positive = closing)
    Vec3 rel_vel = sub3(self->vel, other->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // === TACTICAL ===
    Vec3 other_fwd = quat_rotate(other->ori, vec3(1, 0, 0));
    Vec3 to_self = normalize3(sub3(self->pos, other->pos));
    float target_aspect = dot3(other_fwd, to_self);

    float other_speed = norm3(other->vel);
    float other_potential = other->pos.z * INV_WORLD_MAX_Z;
    float other_kinetic = (other_speed * other_speed) / (MAX_SPEED * MAX_SPEED);
    float other_energy = (other_potential + other_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - other_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (9 obs)
    obs_buffer[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);  // Forward speed [0,1]
    obs_buffer[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f); // Sideslip [-1,1]
    obs_buffer[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f); // Climb rate [-1,1]
    obs_buffer[i++] = clampf(self->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f); // Roll rate [-1,1]
    obs_buffer[i++] = clampf(self->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f); // Pitch rate [-1,1]
    obs_buffer[i++] = clampf(self->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f); // Yaw rate [-1,1]
    obs_buffer[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);          // AoA [-1,1]
    obs_buffer[i++] = potential;                                        // Altitude [0,1]
    obs_buffer[i++] = own_energy;                                       // Own energy [0,1]

    // Target state - spherical (4 obs)
    obs_buffer[i++] = target_az * INV_PI;                               // Azimuth [-1,1]
    obs_buffer[i++] = target_el * INV_HALF_PI;                          // Elevation [-1,1]
    obs_buffer[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);        // Range [0,1]
    obs_buffer[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);    // Closure [-1,1]

    // Tactical (2 obs)
    obs_buffer[i++] = energy_advantage;                                 // Energy advantage [-1,1]
    obs_buffer[i++] = target_aspect;                                    // Aspect [-1,1]

    // Timer (1 obs) - how much time left before episode ends
    obs_buffer[i++] = (float)env->tick / (float)(env->max_steps + 1);   // Timer [0,~1)
    // OBS_SIZE = 16
}

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
    compute_obs_momentum_for_plane(env, &env->player, &env->opponent, env->observations);
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

    // Timer (1 obs)
    env->observations[i++] = (float)env->tick / (float)(env->max_steps + 1);
    // OBS_SIZE = 17
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

    // Timer (1 obs)
    env->observations[i++] = (float)env->tick / (float)(env->max_steps + 1);
    // OBS_SIZE = 17
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

    // Timer (1 obs)
    env->observations[i++] = (float)env->tick / (float)(env->max_steps + 1);
    // OBS_SIZE = 20
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

    // Timer (1 obs)
    env->observations[i++] = (float)env->tick / (float)(env->max_steps + 1);
    // OBS_SIZE = 12
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

    // Timer (1 obs)
    env->observations[i++] = (float)env->tick / (float)(env->max_steps + 1);
    // OBS_SIZE = 16
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

    // Timer (1 obs)
    env->observations[i++] = (float)env->tick / (float)(env->max_steps + 1);
    // OBS_SIZE = 23
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

    // Timer (1 obs)
    env->observations[i++] = (float)env->tick / (float)(env->max_steps + 1);
    // OBS_SIZE = 17
}

// ============================================================================
// Scheme 8: OBS_KITCHEN_SINK - everything (25 obs)
// ============================================================================
// Hypothesis: Maximum information with everything is optimal
// ============================================================================
// Scheme 8: OBS_KITCHEN_SINK - Optimized high-information (30 obs)
// ============================================================================
// Design: Maximum non-redundant information for long training runs
// Includes: all flight state + orientation + control feedback + opponent prediction
//
// Layout (30 obs):
// [0-12]  Own flight state: vel(3), omega(3), aoa, beta, g, q_bar, alt, energy, throttle
// [13-16] Orientation: quaternion (4) - explicitly clamped
// [17-19] Control surfaces: elevator, aileron, rudder commands (actuator feedback)
// [20-23] Target: azimuth, elevation, range, closure (spherical)
// [24-27] Opponent state: roll/pitch/yaw rates (3), target aspect (1)
// [28-29] Tactical: energy advantage, timer
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

    // Target state
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Target aspect - is opponent facing toward or away from us?
    // +1 = nose-on (coming at us), -1 = tail-on (running away)
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    // Energy advantage
    float opp_speed = norm3(o->vel);
    float opp_potential = o->pos.z * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;

    // === OWN FLIGHT STATE (13 obs) ===
    // Body-frame velocity (3 obs)
    env->observations[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);   // [0] Forward speed [0,1]
    env->observations[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);  // [1] Sideslip [-1,1]
    env->observations[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);  // [2] Climb rate [-1,1]

    // Angular velocity (3 obs)
    env->observations[i++] = clampf(p->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // [3] Roll rate [-1,1]
    env->observations[i++] = clampf(p->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // [4] Pitch rate [-1,1]
    env->observations[i++] = clampf(p->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);  // [5] Yaw rate [-1,1]

    // Flight angles (2 obs)
    env->observations[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);           // [6] AoA [-1,1]
    env->observations[i++] = clampf(beta * INV_MAX_SIDESLIP, -1.0f, 1.0f);     // [7] Beta [-1,1]

    // Flight physics (2 obs)
    env->observations[i++] = g_norm;                                            // [8] G-force [-0.5,1]
    env->observations[i++] = q_bar_norm;                                        // [9] Dynamic pressure [0,1]

    // Energy state (2 obs)
    env->observations[i++] = potential;                                         // [10] Altitude [0,1]
    env->observations[i++] = own_energy;                                        // [11] Own energy [0,1]

    // Throttle (1 obs)
    env->observations[i++] = p->throttle;                                       // [12] Throttle [0,1]

    // === ORIENTATION (4 obs) ===
    // Quaternion - explicitly clamped for consistent normalization
    env->observations[i++] = clampf(p->ori.w, -1.0f, 1.0f);                    // [13] Quat w [-1,1]
    env->observations[i++] = clampf(p->ori.x, -1.0f, 1.0f);                    // [14] Quat x [-1,1]
    env->observations[i++] = clampf(p->ori.y, -1.0f, 1.0f);                    // [15] Quat y [-1,1]
    env->observations[i++] = clampf(p->ori.z, -1.0f, 1.0f);                    // [16] Quat z [-1,1]

    // === CONTROL SURFACES (3 obs) - actuator feedback ===
    // Like drone_race motor RPMs - agent sees its own control commands
    env->observations[i++] = clampf(env->actions[1], -1.0f, 1.0f);             // [17] Elevator cmd [-1,1]
    env->observations[i++] = clampf(env->actions[2], -1.0f, 1.0f);             // [18] Aileron cmd [-1,1]
    env->observations[i++] = clampf(env->actions[3], -1.0f, 1.0f);             // [19] Rudder cmd [-1,1]

    // === TARGET INFO (4 obs) ===
    env->observations[i++] = target_az * INV_PI;                                // [20] Azimuth [-1,1]
    env->observations[i++] = target_el * INV_HALF_PI;                           // [21] Elevation [-1,1]
    env->observations[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);         // [22] Range [0,1]
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);     // [23] Closure [-1,1]

    // === OPPONENT STATE (4 obs) ===
    // Opponent angular rates - for predicting maneuvers
    env->observations[i++] = clampf(o->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // [24] Opp roll rate [-1,1]
    env->observations[i++] = clampf(o->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // [25] Opp pitch rate [-1,1]
    env->observations[i++] = clampf(o->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);  // [26] Opp yaw rate [-1,1]

    // Target aspect - critical tactical info
    env->observations[i++] = target_aspect;                                     // [27] Aspect [-1,1]

    // === TACTICAL (2 obs) ===
    env->observations[i++] = energy_advantage;                                  // [28] E advantage [-1,1]
    env->observations[i++] = (float)env->tick / (float)(env->max_steps + 1);   // [29] Timer [0,~1)

    // OBS_SIZE = 30
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
// Opponent observations (for self-play)
// ============================================================================
// Scheme 1 generalized for self-play (opponent perspective)
// ============================================================================
void compute_obs_momentum_beta_for_plane(Dogfight *env, Plane *self, Plane *other, float *obs_buffer) {
    Quat q_inv = {self->ori.w, -self->ori.x, -self->ori.y, -self->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, self->vel);
    float speed = norm3(self->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(self->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(self->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(self->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(self->vel, up) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // Sideslip angle (beta)
    float beta = 0.0f;
    if (speed > 1.0f) {
        beta = asinf(clampf(vel_body.y / speed, -1.0f, 1.0f));
    }

    // Energy state
    float potential = self->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // Target state
    Vec3 rel_pos = sub3(other->pos, self->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    Vec3 rel_vel = sub3(self->vel, other->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Tactical
    Vec3 other_fwd = quat_rotate(other->ori, vec3(1, 0, 0));
    Vec3 to_self = normalize3(sub3(self->pos, other->pos));
    float target_aspect = dot3(other_fwd, to_self);

    float other_speed = norm3(other->vel);
    float other_potential = other->pos.z * INV_WORLD_MAX_Z;
    float other_kinetic = (other_speed * other_speed) / (MAX_SPEED * MAX_SPEED);
    float other_energy = (other_potential + other_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - other_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (9 obs)
    obs_buffer[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);
    obs_buffer[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);
    obs_buffer[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);
    obs_buffer[i++] = clampf(self->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);
    obs_buffer[i++] = clampf(self->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);
    obs_buffer[i++] = clampf(self->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);
    obs_buffer[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);
    obs_buffer[i++] = potential;
    obs_buffer[i++] = own_energy;

    // Sideslip angle (scheme 1 addition)
    obs_buffer[i++] = clampf(beta * INV_MAX_SIDESLIP, -1.0f, 1.0f);

    // Target state (4 obs)
    obs_buffer[i++] = target_az * INV_PI;
    obs_buffer[i++] = target_el * INV_HALF_PI;
    obs_buffer[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);
    obs_buffer[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Tactical (2 obs)
    obs_buffer[i++] = energy_advantage;
    obs_buffer[i++] = target_aspect;

    // Timer (1 obs)
    obs_buffer[i++] = (float)env->tick / (float)(env->max_steps + 1);
    // OBS_SIZE = 17
}

// ============================================================================
// Computes observations from opponent's perspective looking at player.
// Supports scheme 0 (MOMENTUM) and scheme 1 (MOMENTUM_BETA).
void compute_opponent_observations(Dogfight *env, float *opp_obs_buffer) {
    if (env->obs_scheme == 1) {
        compute_obs_momentum_beta_for_plane(env, &env->opponent, &env->player, opp_obs_buffer);
    } else {
        // Default to scheme 0 for other schemes (may need to add more _for_plane variants)
        compute_obs_momentum_for_plane(env, &env->opponent, &env->player, opp_obs_buffer);
    }
}

// ============================================================================
// Debug labels for print_observations
// ============================================================================
#if DEBUG >= 5

// Scheme 0: OBS_MOMENTUM (16 obs)
static const char* DEBUG_OBS_LABELS_MOMENTUM[16] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 1: OBS_MOMENTUM_BETA (17 obs)
static const char* DEBUG_OBS_LABELS_MOMENTUM_BETA[17] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "beta",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 2: OBS_MOMENTUM_GFORCE (17 obs)
static const char* DEBUG_OBS_LABELS_MOMENTUM_GFORCE[17] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "g_force",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 3: OBS_MOMENTUM_FULL (20 obs)
static const char* DEBUG_OBS_LABELS_MOMENTUM_FULL[20] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "beta", "g_force", "throttle",
    "tgt_az", "tgt_el", "range", "closure",
    "tgt_pitch_r", "tgt_roll_r", "E_adv", "timer"
};

// Scheme 4: OBS_MINIMAL (12 obs)
static const char* DEBUG_OBS_LABELS_MINIMAL[12] = {
    "fwd_spd", "aoa", "roll_r", "pitch_r", "yaw_r", "altitude",
    "tgt_az", "tgt_el", "range", "closure", "E_adv", "timer"
};

// Scheme 5: OBS_CARTESIAN (16 obs)
static const char* DEBUG_OBS_LABELS_CARTESIAN[16] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "tgt_x", "tgt_y", "tgt_z", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 6: OBS_DRONE_STYLE (23 obs)
static const char* DEBUG_OBS_LABELS_DRONE_STYLE[23] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "up_x", "up_y", "up_z",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 7: OBS_QBAR (17 obs)
static const char* DEBUG_OBS_LABELS_QBAR[17] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "q_bar",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 8: OBS_KITCHEN_SINK (26 obs)
static const char* DEBUG_OBS_LABELS_KITCHEN_SINK[30] = {
    // Own flight state (13)
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "beta", "g_force", "q_bar", "altitude", "energy", "throttle",
    // Orientation (4)
    "quat_w", "quat_x", "quat_y", "quat_z",
    // Control surfaces (3)
    "elev_cmd", "ail_cmd", "rud_cmd",
    // Target (4)
    "tgt_az", "tgt_el", "range", "closure",
    // Opponent state (4)
    "opp_roll_r", "opp_pitch_r", "opp_yaw_r", "aspect",
    // Tactical (2)
    "E_adv", "timer"
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
                // fwd_spd(0), altitude(7), energy(8), range(11), timer(15) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 11 || i == 15);
                break;
            case OBS_MOMENTUM_BETA:
            case OBS_MOMENTUM_GFORCE:
            case OBS_QBAR:
                // fwd_spd(0), altitude(7), energy(8), range(12), timer(16) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 12 || i == 16);
                break;
            case OBS_MOMENTUM_FULL:
                // fwd_spd(0), altitude(7), energy(8), throttle(11), range(14), timer(19) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 11 || i == 14 || i == 19);
                break;
            case OBS_MINIMAL:
                // fwd_spd(0), altitude(5), range(8), timer(11) are [0,1]
                is_01 = (i == 0 || i == 5 || i == 8 || i == 11);
                break;
            case OBS_CARTESIAN:
                // fwd_spd(0), altitude(7), energy(8), timer(15) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 15);
                break;
            case OBS_DRONE_STYLE:
                // fwd_spd(0), altitude(7), energy(8), range(18), timer(22) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 18 || i == 22);
                break;
            case OBS_KITCHEN_SINK:
                // fwd_spd(0), q_bar(9), altitude(10), energy(11), throttle(12), range(22), timer(29) are [0,1]
                is_01 = (i == 0 || i == 9 || i == 10 || i == 11 || i == 12 || i == 22 || i == 29);
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
