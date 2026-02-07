// dogfight_observations.h - Observation computation for dogfight environment
// Extracted from dogfight.h to reduce file size
//
// Observation Schemes:
// All schemes include timer observation at the end: tick/(max_steps+1) [0,~1)
//   Scheme 0: OBS_MOMENTUM_GFORCE - G-force awareness (17 obs) — proven winner
//   Scheme 1: OBS_DRONE_STYLE     - + quaternion + up vector (23 obs)
//   Scheme 2: OBS_QBAR            - + dynamic pressure (17 obs)
//   Scheme 3: OBS_PILOT_QUAT      - Pilot + quaternion (25 obs)
//   Scheme 4: OBS_PILOT           - Pilot awareness (21 obs) — lean hypothesis

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
// Scheme 0: OBS_MOMENTUM_GFORCE - G-force awareness (17 obs)
// ============================================================================
// Proven winner from df24 sweep (0.989 max ultimate2)
// [0-2]   vel_body (fwd, sideslip, climb)
// [3-5]   omega (roll, pitch, yaw rate)
// [6]     AoA
// [7]     altitude
// [8]     energy
// [9]     G-force
// [10-13] target (az, el, range, closure)
// [14]    E_advantage
// [15]    aspect
// [16]    timer
void compute_obs_momentum_gforce_for_plane(Dogfight *env, Plane *self, Plane *other, float *obs_buffer) {
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

    // Energy state
    float potential = self->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // G-force normalization: 0G=0, 1G=0.2, 5G=1.0, -2.5G=-0.5
    float g_norm = clampf(self->g_force / 5.0f, -0.5f, 1.0f);

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
    Vec3 to_self_dir = normalize3(sub3(self->pos, other->pos));
    float target_aspect = dot3(other_fwd, to_self_dir);

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

    // G-force (1 obs)
    obs_buffer[i++] = g_norm;                                           // G-force [-0.5,1]

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

void compute_obs_momentum_gforce(Dogfight *env) {
    compute_obs_momentum_gforce_for_plane(env, &env->player, &env->opponent, env->observations);
}

// ============================================================================
// Scheme 1: OBS_DRONE_STYLE - + quaternion + up vector (23 obs)
// ============================================================================
// [0-2]   vel_body
// [3-5]   omega
// [6]     AoA
// [7]     altitude
// [8]     energy
// [9-12]  quaternion (w, x, y, z)
// [13-15] up_vector (x, y, z)
// [16-19] target (az, el, range, closure)
// [20]    E_advantage
// [21]    aspect
// [22]    timer
void compute_obs_drone_style_for_plane(Dogfight *env, Plane *self, Plane *other, float *obs_buffer) {
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

    // Energy state
    float potential = self->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // Up vector in world frame (derived from quaternion)
    Vec3 world_up = quat_rotate(self->ori, vec3(0, 0, 1));

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
    Vec3 to_self_dir = normalize3(sub3(self->pos, other->pos));
    float target_aspect = dot3(other_fwd, to_self_dir);

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

    // Quaternion (4 obs) - raw orientation for NN to reason about 3D
    obs_buffer[i++] = self->ori.w;
    obs_buffer[i++] = self->ori.x;
    obs_buffer[i++] = self->ori.y;
    obs_buffer[i++] = self->ori.z;

    // Up vector in world frame (3 obs) - gravity-relative maneuvers
    obs_buffer[i++] = world_up.x;
    obs_buffer[i++] = world_up.y;
    obs_buffer[i++] = world_up.z;

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
    // OBS_SIZE = 23
}

void compute_obs_drone_style(Dogfight *env) {
    compute_obs_drone_style_for_plane(env, &env->player, &env->opponent, env->observations);
}

// ============================================================================
// Scheme 2: OBS_QBAR - + dynamic pressure (17 obs)
// ============================================================================
// [0-2]   vel_body
// [3-5]   omega
// [6]     AoA
// [7]     altitude
// [8]     energy
// [9]     q_bar (dynamic pressure)
// [10-13] target (az, el, range, closure)
// [14]    E_advantage
// [15]    aspect
// [16]    timer
void compute_obs_qbar_for_plane(Dogfight *env, Plane *self, Plane *other, float *obs_buffer) {
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

    // Energy state
    float potential = self->pos.z * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // Dynamic pressure q_bar = 0.5 * rho * V^2
    // At sea level rho ≈ 1.225 kg/m³
    float rho = 1.225f;
    float q_bar = 0.5f * rho * speed * speed;
    float q_bar_norm = clampf(q_bar * INV_MAX_QBAR, 0.0f, 1.0f);

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
    Vec3 to_self_dir = normalize3(sub3(self->pos, other->pos));
    float target_aspect = dot3(other_fwd, to_self_dir);

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

    // Dynamic pressure (1 obs)
    obs_buffer[i++] = q_bar_norm;                                        // q_bar [0,1]

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

void compute_obs_qbar(Dogfight *env) {
    compute_obs_qbar_for_plane(env, &env->player, &env->opponent, env->observations);
}

// ============================================================================
// Scheme 3: OBS_PILOT_QUAT - Pilot + quaternion (25 obs)
// ============================================================================
// PILOT + quaternion(4). Tests whether quat helps on top of a good obs set.
// [0-2]   vel_body (fwd, sideslip, climb)
// [3-5]   omega (roll, pitch, yaw rate)
// [6]     AoA
// [7]     altitude
// [8]     G-force
// [9-11]  up_vector (x, y, z)
// [12-15] quaternion (w, x, y, z)
// [16-19] target (az, el, range, closure)
// [20]    E_advantage
// [21]    aspect
// [22]    opp_pitch_rate
// [23]    opp_roll_rate
// [24]    timer
void compute_obs_pilot_quat_for_plane(Dogfight *env, Plane *self, Plane *other, float *obs_buffer) {
    Quat q_inv = {self->ori.w, -self->ori.x, -self->ori.y, -self->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, self->vel);
    float speed = norm3(self->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(self->ori, vec3(1, 0, 0));
    Vec3 up_body = quat_rotate(self->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(self->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(self->vel, up_body) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // G-force
    float g_norm = clampf(self->g_force / 5.0f, -0.5f, 1.0f);

    // Altitude
    float potential = self->pos.z * INV_WORLD_MAX_Z;

    // Up vector in world frame
    Vec3 world_up = quat_rotate(self->ori, vec3(0, 0, 1));

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
    Vec3 to_self_dir = normalize3(sub3(self->pos, other->pos));
    float target_aspect = dot3(other_fwd, to_self_dir);

    float other_speed = norm3(other->vel);
    float other_potential = other->pos.z * INV_WORLD_MAX_Z;
    float other_kinetic = (other_speed * other_speed) / (MAX_SPEED * MAX_SPEED);
    float own_kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + own_kinetic) * 0.5f;
    float other_energy = (other_potential + other_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - other_energy, -1.0f, 1.0f);

    int i = 0;

    // Own flight state (9 obs)
    obs_buffer[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);   // [0] Forward speed
    obs_buffer[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);  // [1] Sideslip
    obs_buffer[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);  // [2] Climb rate
    obs_buffer[i++] = clampf(self->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // [3] Roll rate
    obs_buffer[i++] = clampf(self->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // [4] Pitch rate
    obs_buffer[i++] = clampf(self->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);  // [5] Yaw rate
    obs_buffer[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);           // [6] AoA
    obs_buffer[i++] = potential;                                         // [7] Altitude
    obs_buffer[i++] = g_norm;                                            // [8] G-force

    // Up vector (3 obs)
    obs_buffer[i++] = world_up.x;                                        // [9] Up X
    obs_buffer[i++] = world_up.y;                                        // [10] Up Y
    obs_buffer[i++] = world_up.z;                                        // [11] Up Z

    // Quaternion (4 obs)
    obs_buffer[i++] = clampf(self->ori.w, -1.0f, 1.0f);                    // [12] Quat w
    obs_buffer[i++] = clampf(self->ori.x, -1.0f, 1.0f);                    // [13] Quat x
    obs_buffer[i++] = clampf(self->ori.y, -1.0f, 1.0f);                    // [14] Quat y
    obs_buffer[i++] = clampf(self->ori.z, -1.0f, 1.0f);                    // [15] Quat z

    // Target state (4 obs)
    obs_buffer[i++] = target_az * INV_PI;                                // [16] Azimuth
    obs_buffer[i++] = target_el * INV_HALF_PI;                           // [17] Elevation
    obs_buffer[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);         // [18] Range
    obs_buffer[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);     // [19] Closure

    // Tactical (2 obs)
    obs_buffer[i++] = energy_advantage;                                  // [20] E advantage
    obs_buffer[i++] = target_aspect;                                     // [21] Aspect

    // Opponent rates (2 obs)
    obs_buffer[i++] = clampf(other->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // [22] Opp pitch rate
    obs_buffer[i++] = clampf(other->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // [23] Opp roll rate

    // Timer (1 obs)
    obs_buffer[i++] = (float)env->tick / (float)(env->max_steps + 1);   // [24] Timer
    // OBS_SIZE = 25
}

void compute_obs_pilot_quat(Dogfight *env) {
    compute_obs_pilot_quat_for_plane(env, &env->player, &env->opponent, env->observations);
}

// ============================================================================
// Scheme 4: OBS_PILOT - Pilot awareness (21 obs) — NEW lean hypothesis
// ============================================================================
// Best of scheme 0 + drone_race up_vector + opp rates - redundancy
// [0-2]   vel_body (fwd, sideslip, climb)
// [3-5]   omega (roll, pitch, yaw rate)
// [6]     AoA
// [7]     altitude
// [8]     G-force
// [9-11]  up_vector (x, y, z)
// [12-15] target (az, el, range, closure)
// [16]    E_advantage
// [17]    aspect
// [18]    opp_pitch_rate
// [19]    opp_roll_rate
// [20]    timer
void compute_obs_pilot_for_plane(Dogfight *env, Plane *self, Plane *other, float *obs_buffer) {
    Quat q_inv = {self->ori.w, -self->ori.x, -self->ori.y, -self->ori.z};

    // Body-frame velocity
    Vec3 vel_body = quat_rotate(q_inv, self->vel);
    float speed = norm3(self->vel);

    // Angle of attack
    Vec3 forward = quat_rotate(self->ori, vec3(1, 0, 0));
    Vec3 up_body = quat_rotate(self->ori, vec3(0, 0, 1));
    float aoa = 0.0f;
    if (speed > 1.0f) {
        Vec3 vel_norm = normalize3(self->vel);
        float cos_alpha = clampf(dot3(vel_norm, forward), -1.0f, 1.0f);
        float alpha = acosf(cos_alpha);
        float sign = (dot3(self->vel, up_body) < 0) ? 1.0f : -1.0f;
        aoa = alpha * sign;
    }

    // G-force
    float g_norm = clampf(self->g_force / 5.0f, -0.5f, 1.0f);

    // Altitude
    float potential = self->pos.z * INV_WORLD_MAX_Z;

    // Up vector in world frame
    Vec3 world_up = quat_rotate(self->ori, vec3(0, 0, 1));

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
    Vec3 to_self_dir = normalize3(sub3(self->pos, other->pos));
    float target_aspect = dot3(other_fwd, to_self_dir);

    float other_speed = norm3(other->vel);
    float other_potential = other->pos.z * INV_WORLD_MAX_Z;
    float other_kinetic = (other_speed * other_speed) / (MAX_SPEED * MAX_SPEED);
    float own_kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + own_kinetic) * 0.5f;
    float other_energy = (other_potential + other_kinetic) * 0.5f;
    float energy_advantage = clampf(own_energy - other_energy, -1.0f, 1.0f);

    int i = 0;

    // Own flight state (9 obs)
    obs_buffer[i++] = clampf(vel_body.x * INV_MAX_SPEED, 0.0f, 1.0f);   // [0] Forward speed
    obs_buffer[i++] = clampf(vel_body.y * INV_MAX_SPEED, -1.0f, 1.0f);  // [1] Sideslip
    obs_buffer[i++] = clampf(vel_body.z * INV_MAX_SPEED, -1.0f, 1.0f);  // [2] Climb rate
    obs_buffer[i++] = clampf(self->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // [3] Roll rate
    obs_buffer[i++] = clampf(self->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // [4] Pitch rate
    obs_buffer[i++] = clampf(self->omega.z * INV_MAX_OMEGA, -1.0f, 1.0f);  // [5] Yaw rate
    obs_buffer[i++] = clampf(aoa * INV_MAX_AOA, -1.0f, 1.0f);           // [6] AoA
    obs_buffer[i++] = potential;                                         // [7] Altitude
    obs_buffer[i++] = g_norm;                                            // [8] G-force

    // Up vector (3 obs)
    obs_buffer[i++] = world_up.x;                                        // [9] Up X
    obs_buffer[i++] = world_up.y;                                        // [10] Up Y
    obs_buffer[i++] = world_up.z;                                        // [11] Up Z

    // Target state (4 obs)
    obs_buffer[i++] = target_az * INV_PI;                                // [12] Azimuth
    obs_buffer[i++] = target_el * INV_HALF_PI;                           // [13] Elevation
    obs_buffer[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);         // [14] Range
    obs_buffer[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);     // [15] Closure

    // Tactical (2 obs)
    obs_buffer[i++] = energy_advantage;                                  // [16] E advantage
    obs_buffer[i++] = target_aspect;                                     // [17] Aspect

    // Opponent rates (2 obs)
    obs_buffer[i++] = clampf(other->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // [18] Opp pitch rate
    obs_buffer[i++] = clampf(other->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // [19] Opp roll rate

    // Timer (1 obs)
    obs_buffer[i++] = (float)env->tick / (float)(env->max_steps + 1);   // [20] Timer
    // OBS_SIZE = 21
}

void compute_obs_pilot(Dogfight *env) {
    compute_obs_pilot_for_plane(env, &env->player, &env->opponent, env->observations);
}

// ============================================================================
// Dispatcher function
// ============================================================================
void compute_observations(Dogfight *env) {
    switch (env->obs_scheme) {
        case OBS_MOMENTUM_GFORCE: compute_obs_momentum_gforce(env); break;
        case OBS_DRONE_STYLE:     compute_obs_drone_style(env); break;
        case OBS_QBAR:            compute_obs_qbar(env); break;
        case OBS_PILOT_QUAT:      compute_obs_pilot_quat(env); break;
        case OBS_PILOT:           compute_obs_pilot(env); break;
        default:                  compute_obs_momentum_gforce(env); break;
    }
}

// ============================================================================
// Opponent observations (for self-play)
// ============================================================================
// Uses same obs scheme as player, from opponent's perspective.
// In self-play, both player and opponent feed into the same policy,
// so they must see identically-structured observations.
void compute_opponent_observations(Dogfight *env, float *opp_obs_buffer) {
    switch (env->obs_scheme) {
        case OBS_MOMENTUM_GFORCE: compute_obs_momentum_gforce_for_plane(env, &env->opponent, &env->player, opp_obs_buffer); break;
        case OBS_DRONE_STYLE:     compute_obs_drone_style_for_plane(env, &env->opponent, &env->player, opp_obs_buffer); break;
        case OBS_QBAR:            compute_obs_qbar_for_plane(env, &env->opponent, &env->player, opp_obs_buffer); break;
        case OBS_PILOT_QUAT:      compute_obs_pilot_quat_for_plane(env, &env->opponent, &env->player, opp_obs_buffer); break;
        case OBS_PILOT:           compute_obs_pilot_for_plane(env, &env->opponent, &env->player, opp_obs_buffer); break;
        default:                  compute_obs_momentum_gforce_for_plane(env, &env->opponent, &env->player, opp_obs_buffer); break;
    }
}

// ============================================================================
// Debug labels for print_observations
// ============================================================================
#if DEBUG >= 5

// Scheme 0: OBS_MOMENTUM_GFORCE (17 obs)
static const char* DEBUG_OBS_LABELS_MOMENTUM_GFORCE[17] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "g_force",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 1: OBS_DRONE_STYLE (23 obs)
static const char* DEBUG_OBS_LABELS_DRONE_STYLE[23] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "up_x", "up_y", "up_z",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 2: OBS_QBAR (17 obs)
static const char* DEBUG_OBS_LABELS_QBAR[17] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "q_bar",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect", "timer"
};

// Scheme 3: OBS_PILOT_QUAT (25 obs)
static const char* DEBUG_OBS_LABELS_PILOT_QUAT[25] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "g_force",
    "up_x", "up_y", "up_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect",
    "opp_pitch_r", "opp_roll_r", "timer"
};

// Scheme 4: OBS_PILOT (21 obs)
static const char* DEBUG_OBS_LABELS_PILOT[21] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "g_force",
    "up_x", "up_y", "up_z",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect",
    "opp_pitch_r", "opp_roll_r", "timer"
};

void print_observations(Dogfight *env) {
    const char** labels = NULL;
    int num_obs = env->obs_size;

    // Select labels based on scheme
    switch (env->obs_scheme) {
        case OBS_MOMENTUM_GFORCE: labels = DEBUG_OBS_LABELS_MOMENTUM_GFORCE; break;
        case OBS_DRONE_STYLE:     labels = DEBUG_OBS_LABELS_DRONE_STYLE; break;
        case OBS_QBAR:            labels = DEBUG_OBS_LABELS_QBAR; break;
        case OBS_PILOT_QUAT:      labels = DEBUG_OBS_LABELS_PILOT_QUAT; break;
        case OBS_PILOT:           labels = DEBUG_OBS_LABELS_PILOT; break;
        default:                  labels = DEBUG_OBS_LABELS_MOMENTUM_GFORCE; break;
    }

    printf("=== OBS (scheme %d, %d obs) ===\n", env->obs_scheme, num_obs);

    for (int i = 0; i < num_obs; i++) {
        float val = env->observations[i];

        // Determine range based on scheme and index
        bool is_01 = false;
        switch (env->obs_scheme) {
            case OBS_MOMENTUM_GFORCE:
            case OBS_QBAR:
                // fwd_spd(0), altitude(7), energy(8), range(12), timer(16) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 12 || i == 16);
                break;
            case OBS_DRONE_STYLE:
                // fwd_spd(0), altitude(7), energy(8), range(18), timer(22) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 18 || i == 22);
                break;
            case OBS_PILOT_QUAT:
                // fwd_spd(0), altitude(7), range(18), timer(24) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 18 || i == 24);
                break;
            case OBS_PILOT:
                // fwd_spd(0), altitude(7), range(14), timer(20) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 14 || i == 20);
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
