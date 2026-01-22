// dogfight_observations.h - Observation computation for dogfight environment
// Extracted from dogfight.h to reduce file size
//
// Contains:
//   - compute_obs_angles() - Scheme 0: Spherical coordinates
//   - compute_obs_pursuit() - Scheme 1: Energy-aware pursuit
//   - compute_obs_realistic() - Scheme 2: Cockpit instruments
//   - compute_obs_realistic_range() - Scheme 3: With explicit range
//   - compute_obs_realistic_enemy_state() - Scheme 4: + enemy state
//   - compute_obs_realistic_full() - Scheme 5: Full instrumentation
//   - compute_observations() - Dispatcher

#ifndef DOGFIGHT_OBSERVATIONS_H
#define DOGFIGHT_OBSERVATIONS_H

// Requires: flightlib.h (Vec3, Quat, math), Dogfight struct defined before include

// Scheme 0: Angles observations (spherical coordinates)
void compute_obs_angles(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles from quaternion
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));
    float yaw = atan2f(2.0f * (p->ori.w * p->ori.z + p->ori.x * p->ori.y),
                       1.0f - 2.0f * (p->ori.y * p->ori.y + p->ori.z * p->ori.z));

    // Target in body frame -> spherical
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float azimuth = atan2f(rel_pos_body.y, rel_pos_body.x);  // -pi to pi
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float elevation = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));  // -pi/2 to pi/2

    // Closing rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closing_rate = dot3(rel_vel, normalize3(rel_pos));

    // Opponent heading relative to player
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 opp_fwd_body = quat_rotate(q_inv, opp_fwd);
    float opp_heading = atan2f(opp_fwd_body.y, opp_fwd_body.x);

    int i = 0;
    // Player state
    env->observations[i++] = p->pos.x * INV_WORLD_HALF_X;
    env->observations[i++] = p->pos.y * INV_WORLD_HALF_Y;
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = clampf(norm3(p->vel) * INV_MAX_SPEED, 0.0f, 1.0f);  // Speed scalar
    env->observations[i++] = pitch * INV_PI;      // -0.5 to 0.5
    env->observations[i++] = roll * INV_PI;       // -1 to 1
    env->observations[i++] = yaw * INV_PI;        // -1 to 1

    // Target angles
    env->observations[i++] = azimuth * INV_PI;    // -1 to 1
    env->observations[i++] = elevation * INV_HALF_PI;  // -1 to 1
    env->observations[i++] = clampf(dist * INV_GUN_RANGE, 0.0f, 2.0f) - 1.0f;  // [-1,1]
    env->observations[i++] = clampf(closing_rate * INV_MAX_SPEED, -1.0f, 1.0f);  // Clamped to [-1,1]

    // Opponent info
    env->observations[i++] = opp_heading * INV_PI;  // -1 to 1
    // OBS_SIZE = 12
}

// Scheme 1: OBS_PURSUIT - Energy-aware pursuit observations (13 obs)
// Better than old OBS_CONTROL_ERROR: no spoon-feeding of control errors,
// instead provides body-frame target info and energy state for learning pursuit
void compute_obs_pursuit(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Own Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Own energy state: (potential + kinetic) / 2, normalized to [0,1]
    float speed = norm3(p->vel);
    float alt = p->pos.z;
    float potential = alt * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // Target in body frame
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Closure rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Target Euler angles
    float target_pitch = asinf(clampf(2.0f * (o->ori.w * o->ori.y - o->ori.z * o->ori.x), -1.0f, 1.0f));
    float target_roll = atan2f(2.0f * (o->ori.w * o->ori.x + o->ori.y * o->ori.z),
                               1.0f - 2.0f * (o->ori.x * o->ori.x + o->ori.y * o->ori.y));

    // Target aspect (head-on vs tail)
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    // Target energy
    float opp_speed = norm3(o->vel);
    float opp_alt = o->pos.z;
    float opp_potential = opp_alt * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;

    // Energy advantage
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (5 obs)
    env->observations[i++] = clampf(speed * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = potential;
    env->observations[i++] = pitch * INV_HALF_PI;
    env->observations[i++] = roll * INV_PI;
    env->observations[i++] = own_energy;

    // Target position in body frame (4 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = clampf(dist * INV_GUN_RANGE, 0.0f, 2.0f) - 1.0f;
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Target state (3 obs)
    env->observations[i++] = target_roll * INV_PI;
    env->observations[i++] = target_pitch * INV_HALF_PI;
    env->observations[i++] = target_aspect;

    // Energy comparison (1 obs)
    env->observations[i++] = energy_advantage;
    // OBS_SIZE = 13
}

// Scheme 2: Realistic cockpit instruments only
void compute_obs_realistic(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Target in body frame for gunsight
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Target apparent size (larger when closer)
    float target_size = 20.0f / fmaxf(dist, 10.0f);  // ~wingspan/distance

    // Opponent aspect (are they facing toward/away from us?)
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);  // 1 = head-on, -1 = tail

    // Horizon visible (is up vector pointing up?)
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;  // 1 = level, 0 = knife-edge, -1 = inverted

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = clampf(norm3(p->vel) * INV_MAX_SPEED, 0.0f, 1.0f);  // Airspeed
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;     // Altitude
    env->observations[i++] = pitch * INV_HALF_PI;            // Pitch indicator
    env->observations[i++] = roll * INV_PI;                       // Bank indicator

    // Gunsight (3 obs)
    env->observations[i++] = target_az * INV_PI;                  // Target azimuth in sight
    env->observations[i++] = target_el * INV_HALF_PI;         // Target elevation in sight
    env->observations[i++] = clampf(target_size, 0.0f, 2.0f) - 1.0f;  // Target size

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;                   // -1 to 1
    env->observations[i++] = horizon_visible;                 // -1 to 1
    env->observations[i++] = clampf(dist * INV_GUN_RANGE, 0.0f, 2.0f) - 1.0f;  // Distance estimate
    // OBS_SIZE = 10
}

// Scheme 3: REALISTIC with explicit range (10 obs)
// Like REALISTIC but with km range + closure rate instead of target_size + distance_estimate
void compute_obs_realistic_range(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Target in body frame for gunsight
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Range in km (0 = point blank, 0.5 = 1km, 1.0 = 2km+)
    float range_km = clampf(dist / 2000.0f, 0.0f, 1.0f);

    // Opponent aspect (are they facing toward/away from us?)
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);  // 1 = head-on, -1 = tail

    // Horizon visible (is up vector pointing up?)
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;  // 1 = level, 0 = knife-edge, -1 = inverted

    // Closure rate (positive = closing)
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure_rate = dot3(rel_vel, normalize3(rel_pos));

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = clampf(norm3(p->vel) * INV_MAX_SPEED, 0.0f, 1.0f);  // Airspeed
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;     // Altitude
    env->observations[i++] = pitch * INV_HALF_PI;            // Pitch indicator
    env->observations[i++] = roll * INV_PI;                       // Bank indicator

    // Gunsight (3 obs)
    env->observations[i++] = target_az * INV_PI;                  // Target azimuth in sight
    env->observations[i++] = target_el * INV_HALF_PI;         // Target elevation in sight
    env->observations[i++] = range_km;                        // Range: 0=close, 1=2km+

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;                   // -1 to 1
    env->observations[i++] = horizon_visible;                 // -1 to 1
    env->observations[i++] = clampf(closure_rate * INV_MAX_SPEED, -1.0f, 1.0f);  // Closure rate
    // OBS_SIZE = 10
}

// Scheme 4: REALISTIC_ENEMY_STATE (13 obs)
// REALISTIC_RANGE + enemy pitch/roll/heading
void compute_obs_realistic_enemy_state(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Target in body frame for gunsight
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Range in km
    float range_km = clampf(dist / 2000.0f, 0.0f, 1.0f);

    // Opponent aspect
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    // Horizon visible
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;

    // Closure rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure_rate = dot3(rel_vel, normalize3(rel_pos));

    // Enemy Euler angles (relative to horizon)
    float enemy_pitch = asinf(clampf(2.0f * (o->ori.w * o->ori.y - o->ori.z * o->ori.x), -1.0f, 1.0f));
    float enemy_roll = atan2f(2.0f * (o->ori.w * o->ori.x + o->ori.y * o->ori.z),
                              1.0f - 2.0f * (o->ori.x * o->ori.x + o->ori.y * o->ori.y));

    // Enemy heading relative to player (+1 = pointing at player, -1 = pointing away)
    float enemy_heading_rel = target_aspect;  // Already computed as dot(opp_fwd, to_player)

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = clampf(norm3(p->vel) * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = pitch * INV_HALF_PI;
    env->observations[i++] = roll * INV_PI;

    // Gunsight (3 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = range_km;

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;
    env->observations[i++] = horizon_visible;
    env->observations[i++] = clampf(closure_rate * INV_MAX_SPEED, -1.0f, 1.0f);

    // Enemy state (3 obs) - NEW
    env->observations[i++] = enemy_pitch * INV_HALF_PI;  // Enemy nose angle vs horizon
    env->observations[i++] = enemy_roll * INV_PI;             // Enemy bank angle vs horizon
    env->observations[i++] = enemy_heading_rel;           // Pointing toward/away
    // OBS_SIZE = 13
}

// Scheme 5: REALISTIC_FULL (15 obs)
// REALISTIC_ENEMY_STATE + turn rate + G-loading
void compute_obs_realistic_full(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Target in body frame for gunsight
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Range in km
    float range_km = clampf(dist / 2000.0f, 0.0f, 1.0f);

    // Opponent aspect
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    // Horizon visible
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;

    // Closure rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure_rate = dot3(rel_vel, normalize3(rel_pos));

    // Enemy Euler angles
    float enemy_pitch = asinf(clampf(2.0f * (o->ori.w * o->ori.y - o->ori.z * o->ori.x), -1.0f, 1.0f));
    float enemy_roll = atan2f(2.0f * (o->ori.w * o->ori.x + o->ori.y * o->ori.z),
                              1.0f - 2.0f * (o->ori.x * o->ori.x + o->ori.y * o->ori.y));
    float enemy_heading_rel = target_aspect;

    // Turn rate from velocity change
    float speed = norm3(p->vel);
    float turn_rate_actual = 0.0f;
    if (speed > 10.0f) {
        Vec3 accel = mul3(sub3(p->vel, p->prev_vel), 1.0f / DT);
        Vec3 vel_dir = mul3(p->vel, 1.0f / speed);
        float accel_forward = dot3(accel, vel_dir);
        Vec3 accel_centripetal = sub3(accel, mul3(vel_dir, accel_forward));
        float centripetal_mag = norm3(accel_centripetal);
        turn_rate_actual = centripetal_mag / speed;  // omega = a/v
    }
    // Normalize turn rate: max ~0.5 rad/s (29 deg/s) for sustained turn
    float turn_rate_norm = clampf(turn_rate_actual / 0.5f, -1.0f, 1.0f);

    // G-loading: use physics-accurate p->g_force (aerodynamic forces)
    // Range: -1.5 to +6.0 G, normalize so 1G = 0, 6G = 1, -1.5G = -0.5
    float g_loading_norm = clampf((p->g_force - 1.0f) / 5.0f, -0.5f, 1.0f);

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = clampf(speed * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = pitch * INV_HALF_PI;
    env->observations[i++] = roll * INV_PI;

    // Gunsight (3 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = range_km;

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;
    env->observations[i++] = horizon_visible;
    env->observations[i++] = clampf(closure_rate * INV_MAX_SPEED, -1.0f, 1.0f);

    // Enemy state (3 obs)
    env->observations[i++] = enemy_pitch * INV_HALF_PI;
    env->observations[i++] = enemy_roll * INV_PI;
    env->observations[i++] = enemy_heading_rel;

    // Own state (2 obs) - NEW
    env->observations[i++] = turn_rate_norm;    // How fast am I turning?
    env->observations[i++] = g_loading_norm;    // How hard am I pulling?
    // OBS_SIZE = 15
}

// Dispatcher function
void compute_observations(Dogfight *env) {
    switch (env->obs_scheme) {
        case OBS_ANGLES:               compute_obs_angles(env); break;
        case OBS_PURSUIT:              compute_obs_pursuit(env); break;
        case OBS_REALISTIC:            compute_obs_realistic(env); break;
        case OBS_REALISTIC_RANGE:      compute_obs_realistic_range(env); break;
        case OBS_REALISTIC_ENEMY_STATE: compute_obs_realistic_enemy_state(env); break;
        case OBS_REALISTIC_FULL:       compute_obs_realistic_full(env); break;
        default:                       compute_obs_angles(env); break;
    }
}

#endif // DOGFIGHT_OBSERVATIONS_H
