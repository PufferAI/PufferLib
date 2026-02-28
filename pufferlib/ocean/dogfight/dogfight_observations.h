// dogfight_observations.h - Observation computation for dogfight environment
// Extracted from dogfight.h to reduce file size
//
// Observation Schemes:
// All schemes include timer observation at the end: tick/(max_steps+1) [0,~1)
//   Scheme 0: OBS_MOMENTUM_GFORCE  - G-force awareness (17 obs) — proven winner
//   Scheme 1: OBS_PILOT            - Pilot awareness (22 obs)
//   Scheme 2: OBS_OPPONENT_AWARE   - S1 + opp up vector + opp speed (26 obs)
//
// Preserved (unwired) rate schemes from df24-df31:
//   OBS_RATES_LEAN  - Scheme 0 + tactical rates (22 obs) — harmful per df32 analysis
//   OBS_RATES_FULL  - Scheme 1 + tactical rates (27 obs) — harmful per df32 analysis

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
#define MAX_LOS_RATE 2.0f           // ~115 deg/s, aggressive close-range maneuvering
#define INV_MAX_LOS_RATE (1.0f / MAX_LOS_RATE)
#define MAX_ASPECT_RATE 2.0f        // Max rate of cos(angle) change per second
#define INV_MAX_ASPECT_RATE (1.0f / MAX_ASPECT_RATE)
#define MAX_EADV_RATE 0.5f          // Energy advantage changes slowly
#define INV_MAX_EADV_RATE (1.0f / MAX_EADV_RATE)

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
// Scheme 1: OBS_PILOT - Pilot awareness (22 obs) — lean hypothesis
// ============================================================================
// Best of scheme 0 + drone_race up_vector + opp rates
// [0-2]   vel_body (fwd, sideslip, climb)
// [3-5]   omega (roll, pitch, yaw rate)
// [6]     AoA
// [7]     altitude
// [8]     G-force
// [9]     energy
// [10-12] up_vector (x, y, z)
// [13-16] target (az, el, range, closure)
// [17]    E_advantage
// [18]    aspect
// [19]    opp_pitch_rate
// [20]    opp_roll_rate
// [21]    timer
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
    obs_buffer[i++] = own_energy;                                        // [9] Energy

    // Up vector (3 obs)
    obs_buffer[i++] = world_up.x;                                        // [10] Up X
    obs_buffer[i++] = world_up.y;                                        // [11] Up Y
    obs_buffer[i++] = world_up.z;                                        // [12] Up Z

    // Target state (4 obs)
    obs_buffer[i++] = target_az * INV_PI;                                // [13] Azimuth
    obs_buffer[i++] = target_el * INV_HALF_PI;                           // [14] Elevation
    obs_buffer[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);         // [15] Range
    obs_buffer[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);     // [16] Closure

    // Tactical (2 obs)
    obs_buffer[i++] = energy_advantage;                                  // [17] E advantage
    obs_buffer[i++] = target_aspect;                                     // [18] Aspect

    // Opponent rates (2 obs)
    obs_buffer[i++] = clampf(other->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // [19] Opp pitch rate
    obs_buffer[i++] = clampf(other->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // [20] Opp roll rate

    // Timer (1 obs)
    obs_buffer[i++] = (float)env->tick / (float)(env->max_steps + 1);   // [21] Timer
    // OBS_SIZE = 22
}

void compute_obs_pilot(Dogfight *env) {
    compute_obs_pilot_for_plane(env, &env->player, &env->opponent, env->observations);
}

// ============================================================================
// Scheme 2: OBS_RATES_LEAN - Scheme 0 + tactical rates (22 obs)
// ============================================================================
// Scheme 0 base (g-force awareness) + LOS rates, aspect rate, energy adv rate, opp speed
// [0-2]   vel_body (fwd, sideslip, climb)
// [3-5]   omega (roll, pitch, yaw rate)
// [6]     AoA
// [7]     altitude
// [8]     energy
// [9]     G-force
// [10-13] target (az, el, range, closure)
// [14]    E_advantage
// [15]    aspect
// [16]    az_rate  (LOS rate horizontal)
// [17]    el_rate  (LOS rate vertical)
// [18]    asp_rate (aspect rate)
// [19]    eadv_rate (energy advantage rate)
// [20]    opp_spd  (opponent airspeed)
// [21]    timer
void compute_obs_rates_lean_for_plane(Dogfight *env, Plane *self, Plane *other, float *obs_buffer,
                                       float *prev_az, float *prev_el, float *prev_aspect, float *prev_eadv) {
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

    // G-force
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

    // Rate computations (finite differences with azimuth wraparound)
    float az_delta = target_az - *prev_az;
    if (az_delta > M_PI) az_delta -= 2.0f * M_PI;
    if (az_delta < -M_PI) az_delta += 2.0f * M_PI;
    float azimuth_rate = az_delta / DT;   // rad/s

    float el_delta = target_el - *prev_el;
    float elevation_rate = el_delta / DT;  // rad/s

    float aspect_rate = (target_aspect - *prev_aspect) / DT;
    float eadv_rate = (energy_advantage - *prev_eadv) / DT;

    // Update previous values
    *prev_az = target_az;
    *prev_el = target_el;
    *prev_aspect = target_aspect;
    *prev_eadv = energy_advantage;

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
    obs_buffer[i++] = g_norm;

    // Target state (4 obs)
    obs_buffer[i++] = target_az * INV_PI;
    obs_buffer[i++] = target_el * INV_HALF_PI;
    obs_buffer[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);
    obs_buffer[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Tactical (2 obs)
    obs_buffer[i++] = energy_advantage;
    obs_buffer[i++] = target_aspect;

    // Rates (4 obs)
    obs_buffer[i++] = clampf(azimuth_rate * INV_MAX_LOS_RATE, -1.0f, 1.0f);     // [16] az_rate
    obs_buffer[i++] = clampf(elevation_rate * INV_MAX_LOS_RATE, -1.0f, 1.0f);    // [17] el_rate
    obs_buffer[i++] = clampf(aspect_rate * INV_MAX_ASPECT_RATE, -1.0f, 1.0f);    // [18] asp_rate
    obs_buffer[i++] = clampf(eadv_rate * INV_MAX_EADV_RATE, -1.0f, 1.0f);        // [19] eadv_rate

    // Opponent speed (1 obs)
    obs_buffer[i++] = clampf(other_speed * INV_MAX_SPEED, 0.0f, 1.0f);           // [20] opp_spd

    // Timer (1 obs)
    obs_buffer[i++] = (float)env->tick / (float)(env->max_steps + 1);
    // OBS_SIZE = 22
}

void compute_obs_rates_lean(Dogfight *env) {
    compute_obs_rates_lean_for_plane(env, &env->player, &env->opponent, env->observations,
                                      &env->prev_player_target_az, &env->prev_player_target_el,
                                      &env->prev_player_aspect, &env->prev_player_eadv);
}

// ============================================================================
// Scheme 3: OBS_RATES_FULL - Scheme 1 + tactical rates (27 obs)
// ============================================================================
// Scheme 1 base (pilot awareness) + LOS rates, aspect rate, energy adv rate, opp speed
// [0-2]   vel_body (fwd, sideslip, climb)
// [3-5]   omega (roll, pitch, yaw rate)
// [6]     AoA
// [7]     altitude
// [8]     G-force
// [9]     energy
// [10-12] up_vector (x, y, z)
// [13-16] target (az, el, range, closure)
// [17]    E_advantage
// [18]    aspect
// [19]    opp_pitch_rate
// [20]    opp_roll_rate
// [21]    az_rate  (LOS rate horizontal)
// [22]    el_rate  (LOS rate vertical)
// [23]    asp_rate (aspect rate)
// [24]    eadv_rate (energy advantage rate)
// [25]    opp_spd  (opponent airspeed)
// [26]    timer
void compute_obs_rates_full_for_plane(Dogfight *env, Plane *self, Plane *other, float *obs_buffer,
                                       float *prev_az, float *prev_el, float *prev_aspect, float *prev_eadv) {
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

    // Rate computations (finite differences with azimuth wraparound)
    float az_delta = target_az - *prev_az;
    if (az_delta > M_PI) az_delta -= 2.0f * M_PI;
    if (az_delta < -M_PI) az_delta += 2.0f * M_PI;
    float azimuth_rate = az_delta / DT;   // rad/s

    float el_delta = target_el - *prev_el;
    float elevation_rate = el_delta / DT;  // rad/s

    float aspect_rate = (target_aspect - *prev_aspect) / DT;
    float eadv_rate = (energy_advantage - *prev_eadv) / DT;

    // Update previous values
    *prev_az = target_az;
    *prev_el = target_el;
    *prev_aspect = target_aspect;
    *prev_eadv = energy_advantage;

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
    obs_buffer[i++] = own_energy;                                        // [9] Energy

    // Up vector (3 obs)
    obs_buffer[i++] = world_up.x;                                        // [10] Up X
    obs_buffer[i++] = world_up.y;                                        // [11] Up Y
    obs_buffer[i++] = world_up.z;                                        // [12] Up Z

    // Target state (4 obs)
    obs_buffer[i++] = target_az * INV_PI;                                // [13] Azimuth
    obs_buffer[i++] = target_el * INV_HALF_PI;                           // [14] Elevation
    obs_buffer[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);         // [15] Range
    obs_buffer[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);     // [16] Closure

    // Tactical (2 obs)
    obs_buffer[i++] = energy_advantage;                                  // [17] E advantage
    obs_buffer[i++] = target_aspect;                                     // [18] Aspect

    // Opponent rates (2 obs)
    obs_buffer[i++] = clampf(other->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // [19] Opp pitch rate
    obs_buffer[i++] = clampf(other->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // [20] Opp roll rate

    // Tactical rates (4 obs)
    obs_buffer[i++] = clampf(azimuth_rate * INV_MAX_LOS_RATE, -1.0f, 1.0f);     // [21] az_rate
    obs_buffer[i++] = clampf(elevation_rate * INV_MAX_LOS_RATE, -1.0f, 1.0f);    // [22] el_rate
    obs_buffer[i++] = clampf(aspect_rate * INV_MAX_ASPECT_RATE, -1.0f, 1.0f);    // [23] asp_rate
    obs_buffer[i++] = clampf(eadv_rate * INV_MAX_EADV_RATE, -1.0f, 1.0f);        // [24] eadv_rate

    // Opponent speed (1 obs)
    obs_buffer[i++] = clampf(other_speed * INV_MAX_SPEED, 0.0f, 1.0f);           // [25] opp_spd

    // Timer (1 obs)
    obs_buffer[i++] = (float)env->tick / (float)(env->max_steps + 1);   // [26] Timer
    // OBS_SIZE = 27
}

void compute_obs_rates_full(Dogfight *env) {
    compute_obs_rates_full_for_plane(env, &env->player, &env->opponent, env->observations,
                                      &env->prev_player_target_az, &env->prev_player_target_el,
                                      &env->prev_player_aspect, &env->prev_player_eadv);
}

// ============================================================================
// Scheme 2: OBS_OPPONENT_AWARE - S1 + opp up vector + opp speed (26 obs)
// ============================================================================
// Scheme 1 base (pilot awareness) + opponent up vector (world frame) + opponent speed
// No finite differences — all direct state reads
// [0-2]   vel_body (fwd, sideslip, climb)
// [3-5]   omega (roll, pitch, yaw rate)
// [6]     AoA
// [7]     altitude
// [8]     G-force
// [9]     energy
// [10-12] up_vector (x, y, z)
// [13-16] target (az, el, range, closure)
// [17]    E_advantage
// [18]    aspect
// [19]    opp_pitch_rate
// [20]    opp_roll_rate
// [21-23] opp_up_vector (x, y, z)
// [24]    opp_spd
// [25]    timer
void compute_obs_opponent_aware_for_plane(Dogfight *env, Plane *self, Plane *other, float *obs_buffer) {
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

    // Up vector in world frame (self)
    Vec3 world_up = quat_rotate(self->ori, vec3(0, 0, 1));

    // Opponent up vector in world frame
    Vec3 opp_world_up = quat_rotate(other->ori, vec3(0, 0, 1));

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
    obs_buffer[i++] = own_energy;                                        // [9] Energy

    // Up vector (3 obs)
    obs_buffer[i++] = world_up.x;                                        // [10] Up X
    obs_buffer[i++] = world_up.y;                                        // [11] Up Y
    obs_buffer[i++] = world_up.z;                                        // [12] Up Z

    // Target state (4 obs)
    obs_buffer[i++] = target_az * INV_PI;                                // [13] Azimuth
    obs_buffer[i++] = target_el * INV_HALF_PI;                           // [14] Elevation
    obs_buffer[i++] = clampf(dist * INV_MAX_RANGE, 0.0f, 1.0f);         // [15] Range
    obs_buffer[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);     // [16] Closure

    // Tactical (2 obs)
    obs_buffer[i++] = energy_advantage;                                  // [17] E advantage
    obs_buffer[i++] = target_aspect;                                     // [18] Aspect

    // Opponent rates (2 obs)
    obs_buffer[i++] = clampf(other->omega.y * INV_MAX_OMEGA, -1.0f, 1.0f);  // [19] Opp pitch rate
    obs_buffer[i++] = clampf(other->omega.x * INV_MAX_OMEGA, -1.0f, 1.0f);  // [20] Opp roll rate

    // Opponent up vector (3 obs)
    obs_buffer[i++] = opp_world_up.x;                                   // [21] Opp Up X
    obs_buffer[i++] = opp_world_up.y;                                   // [22] Opp Up Y
    obs_buffer[i++] = opp_world_up.z;                                   // [23] Opp Up Z

    // Opponent speed (1 obs)
    obs_buffer[i++] = clampf(other_speed * INV_MAX_SPEED, 0.0f, 1.0f);  // [24] Opp speed

    // Timer (1 obs)
    obs_buffer[i++] = (float)env->tick / (float)(env->max_steps + 1);   // [25] Timer
    // OBS_SIZE = 26
}

void compute_obs_opponent_aware(Dogfight *env) {
    compute_obs_opponent_aware_for_plane(env, &env->player, &env->opponent, env->observations);
}

// ============================================================================
// Dispatcher function
// ============================================================================
void compute_observations(Dogfight *env) {
    switch (env->obs_scheme) {
        case OBS_MOMENTUM_GFORCE:  compute_obs_momentum_gforce(env); break;
        case OBS_PILOT:            compute_obs_pilot(env); break;
        case OBS_OPPONENT_AWARE:   compute_obs_opponent_aware(env); break;
        default:                   compute_obs_momentum_gforce(env); break;
    }
}

// ============================================================================
// Opponent observations (for self-play)
// ============================================================================
// Uses same obs scheme as player, from opponent's perspective.
// In self-play, both player and opponent feed into the same policy,
// so they must see identically-structured observations.
void compute_opponent_observations(Dogfight *env, float *opp_obs_buffer) {
    // Use opponent_obs_scheme if set, otherwise fall back to player's obs_scheme
    int scheme = (env->opponent_obs_scheme >= 0) ? env->opponent_obs_scheme : env->obs_scheme;
    switch (scheme) {
        case OBS_MOMENTUM_GFORCE:  compute_obs_momentum_gforce_for_plane(env, &env->opponent, &env->player, opp_obs_buffer); break;
        case OBS_PILOT:            compute_obs_pilot_for_plane(env, &env->opponent, &env->player, opp_obs_buffer); break;
        case OBS_OPPONENT_AWARE:   compute_obs_opponent_aware_for_plane(env, &env->opponent, &env->player, opp_obs_buffer); break;
        default:                   compute_obs_momentum_gforce_for_plane(env, &env->opponent, &env->player, opp_obs_buffer); break;
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

// Scheme 1: OBS_PILOT (22 obs)
static const char* DEBUG_OBS_LABELS_PILOT[22] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "g_force", "energy",
    "up_x", "up_y", "up_z",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect",
    "opp_pitch_r", "opp_roll_r", "timer"
};

// Scheme 2: OBS_OPPONENT_AWARE (26 obs)
static const char* DEBUG_OBS_LABELS_OPPONENT_AWARE[26] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "g_force", "energy",
    "up_x", "up_y", "up_z",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect",
    "opp_pitch_r", "opp_roll_r",
    "opp_up_x", "opp_up_y", "opp_up_z",
    "opp_spd", "timer"
};

// Preserved: OBS_RATES_LEAN (22 obs) — unwired
static const char* DEBUG_OBS_LABELS_RATES_LEAN[22] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "energy", "g_force",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect",
    "az_rate", "el_rate", "asp_rate", "eadv_rate",
    "opp_spd", "timer"
};

// Preserved: OBS_RATES_FULL (27 obs) — unwired
static const char* DEBUG_OBS_LABELS_RATES_FULL[27] = {
    "fwd_spd", "sideslip", "climb", "roll_r", "pitch_r", "yaw_r",
    "aoa", "altitude", "g_force", "energy",
    "up_x", "up_y", "up_z",
    "tgt_az", "tgt_el", "range", "closure",
    "E_adv", "aspect",
    "opp_pitch_r", "opp_roll_r",
    "az_rate", "el_rate", "asp_rate", "eadv_rate",
    "opp_spd", "timer"
};

void print_observations(Dogfight *env) {
    const char** labels = NULL;
    int num_obs = env->obs_size;

    // Select labels based on scheme
    switch (env->obs_scheme) {
        case OBS_MOMENTUM_GFORCE:  labels = DEBUG_OBS_LABELS_MOMENTUM_GFORCE; break;
        case OBS_PILOT:            labels = DEBUG_OBS_LABELS_PILOT; break;
        case OBS_OPPONENT_AWARE:   labels = DEBUG_OBS_LABELS_OPPONENT_AWARE; break;
        default:                   labels = DEBUG_OBS_LABELS_MOMENTUM_GFORCE; break;
    }

    printf("=== OBS (scheme %d, %d obs) ===\n", env->obs_scheme, num_obs);

    for (int i = 0; i < num_obs; i++) {
        float val = env->observations[i];

        // Determine range based on scheme and index
        bool is_01 = false;
        switch (env->obs_scheme) {
            case OBS_MOMENTUM_GFORCE:
                // fwd_spd(0), altitude(7), energy(8), range(12), timer(16) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 8 || i == 12 || i == 16);
                break;
            case OBS_PILOT:
                // fwd_spd(0), altitude(7), energy(9), range(15), timer(21) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 9 || i == 15 || i == 21);
                break;
            case OBS_OPPONENT_AWARE:
                // fwd_spd(0), altitude(7), energy(9), range(15), opp_spd(24), timer(25) are [0,1]
                is_01 = (i == 0 || i == 7 || i == 9 || i == 15 || i == 24 || i == 25);
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
