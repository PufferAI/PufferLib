#ifndef DOGFIGHT_SPAWN_H
#define DOGFIGHT_SPAWN_H

// Spawn randomization parameters - stage-dependent ranges for variety
typedef struct SpawnRandomization {
    float speed_min, speed_max;       // Initial airspeed range (m/s)
    float pitch_max_deg;              // Max pitch deviation (±degrees)
    float bank_max_deg;               // Max bank deviation (±degrees)
    float throttle_min, throttle_max; // Initial throttle range
} SpawnRandomization;

// Get spawn randomization parameters for a given stage
// Earlier stages = tighter ranges (easier), later stages = wider ranges (harder)
// Updated 2026-01-27: Stage boundaries adjusted for 20-stage curriculum (added DIVE_ATTACK, ZOOM_ATTACK)
static inline SpawnRandomization get_spawn_randomization(int stage) {
    if (stage <= 3)  return (SpawnRandomization){75, 85,  5, 10, 0.45f, 0.55f};
    if (stage <= 7)  return (SpawnRandomization){70, 95, 10, 20, 0.35f, 0.65f};
    if (stage <= 13) return (SpawnRandomization){65, 105, 15, 30, 0.30f, 0.70f};
    return (SpawnRandomization){60, 110, 15, 45, 0.25f, 0.80f};
}

// Stage 0: TAIL_CHASE - Opponent ahead, same heading (easiest)
static void spawn_tail_chase(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 200-400m ahead with guaranteed minimum offset
    // At 300m, 5° gun cone = ~26m radius for hits
    // Minimum 26m y-offset guarantees ~5° at 300m (more at closer range)
    // Signed offset with minimum magnitude: either [-50, -26] or [26, 50]
    float y_sign = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float y_offset = y_sign * rndf(26, 50);

    // 20% chance: spawn player LOW (400m) with opponent ABOVE
    // Teaches altitude awareness early - don't descend with target
    if (rndf(0, 1) < 0.2f) {
        env->player.pos.z = 400.0f;  // Just below 500m recovery threshold
        Vec3 opp_pos = vec3(
            player_pos.x + rndf(200, 400),
            player_pos.y + y_offset,
            700.0f + rndf(0, 200)  // Opponent 300-500m above player
        );
        reset_plane(&env->opponent, opp_pos, player_vel);
        env->opponent_ap.mode = AP_STRAIGHT;
        // More time for climb + pursuit in altitude-disadvantage variant
        env->max_steps = 2000;
        return;
    }

    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + y_offset,        // Min 26m = ~5° at 300m
        player_pos.z + rndf(-38, 38)    // z can still vary
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = AP_STRAIGHT;
}

// Stage 1: HEAD_ON - Opponent coming toward us
static void spawn_head_on(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // 20% chance: spawn player LOW with opponent coming from ABOVE
    // Teaches: don't dive into head-on, maintain altitude
    if (rndf(0, 1) < 0.2f) {
        env->player.pos.z = 400.0f;  // Just below 500m recovery threshold
        Vec3 opp_pos = vec3(
            player_pos.x + rndf(400, 600),
            player_pos.y + rndf(-50, 50),
            700.0f + rndf(0, 200)  // Opponent 300-500m above
        );
        Vec3 opp_vel = vec3(-player_vel.x, -player_vel.y, player_vel.z);
        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent_ap.mode = AP_STRAIGHT;
        // More time for climb + pursuit in altitude-disadvantage variant
        env->max_steps = 2000;
        return;
    }

    // Opponent 400-600m ahead, facing us (opposite velocity)
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(400, 600),
        player_pos.y + rndf(-50, 50),
        player_pos.z + rndf(-30, 30)
    );
    Vec3 opp_vel = vec3(-player_vel.x, -player_vel.y, player_vel.z);
    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent_ap.mode = AP_STRAIGHT;
}

// Stage 18: CROSSING - 45 degree deflection shots (reduced from 90° - see CURRICULUM_PLANS.md)
// 90° deflection is historically nearly impossible; 45° is achievable with proper lead
static void spawn_crossing(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 300-500m to the side, flying at 45° angle (not perpendicular)
    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(100, 200),
        player_pos.y + side * rndf(300, 500),
        player_pos.z + rndf(-50, 50)
    );
    // 45° crossing velocity: opponent flies at 45° angle across player's path
    // cos(45°) ≈ 0.707, sin(45°) ≈ 0.707
    float speed = norm3(player_vel);
    float cos45 = 0.7071f;
    float sin45 = 0.7071f;
    // side=+1 (right): fly toward (-45°) = (cos, -sin) to cross leftward
    // side=-1 (left): fly toward (+45°) = (cos, +sin) to cross rightward
    Vec3 opp_vel = vec3(speed * cos45, -side * speed * sin45, 0);
    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent_ap.mode = AP_STRAIGHT;
}

// Stage 2: VERTICAL - Above or below player
static void spawn_vertical(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 200-400m ahead, 200-400m above OR below
    float vert = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float alt_offset = vert * rndf(200, 400);
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-50, 50),
        clampf(player_pos.z + alt_offset, 300, 4700)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = AP_LEVEL;  // Maintain altitude

    // Speed boost only when opponent is ABOVE us (climbing needs energy, diving doesn't)
    if (opp_pos.z > player_pos.z) {
        env->player.vel = mul3(env->player.vel, 1.15f);
    }
}

// Stage 3: GENTLE_TURNS - Opponent does gentle turns (30°)
static void spawn_gentle_turns(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // 20% chance: spawn player LOW with opponent turning ABOVE
    // Teaches: climb while pursuing turning target
    if (rndf(0, 1) < 0.2f) {
        env->player.pos.z = 400.0f;  // Just below 500m recovery threshold
        Vec3 opp_pos = vec3(
            player_pos.x + rndf(200, 500),
            player_pos.y + rndf(-100, 100),
            700.0f + rndf(0, 200)  // Opponent 300-500m above
        );
        reset_plane(&env->opponent, opp_pos, player_vel);
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
        // More time for climb + pursuit in altitude-disadvantage variant
        env->max_steps = 2000;
        return;
    }

    // Random spawn position (similar to original)
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 500),
        player_pos.y + rndf(-100, 100),
        player_pos.z + rndf(-50, 50)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    // Randomly choose turn direction - gentle 30° bank
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stage 4: OFFSET - Large lateral/vertical offset, same heading
// Teaches: Finding and tracking targets not directly in front
static void spawn_offset(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 150-300m ahead with LARGE lateral/vertical offset
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(150, 300),
        player_pos.y + rndf(-200, 200),   // Large lateral - can be way to the side
        clampf(player_pos.z + rndf(-150, 150), 300, 4700)  // Large vertical
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stage 5: ANGLED - Offset + different heading (±22°)
// Teaches: Pursuit geometry when target isn't flying your direction (small angle)
static void spawn_angled(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-150, 150),
        clampf(player_pos.z + rndf(-100, 100), 300, 4700)
    );

    // Heading offset: ±22° from player (reduced from ±45° for smoother progression)
    float heading_offset = rndf(-0.385f, 0.385f);  // ~22° in radians
    float player_heading = atan2f(player_vel.y, player_vel.x);
    float opp_heading = player_heading + heading_offset;

    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stages 6-9: Unified side spawn - uses angle_min_deg, angle_max_deg, bank from STAGES
// Stages 6-8: Target off axis, flying away (no turns)
// Stage 9: Same geometry + 30° turns
static void spawn_side(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    const StageConfig* cfg = &STAGES[env->stage];

    // 20% chance: ENERGY BUILDING scenario
    // Opponent VERY HIGH and SLOW - player MUST build energy over time to reach them
    // Can't just zoom climb - need sustained full throttle climbing for many seconds
    // Teaches: long-term energy planning, not just immediate pursuit
    if (rndf(0, 1) < 0.2f) {
        // Player at normal altitude, opponent 800-1200m ABOVE
        // This is too high to zoom climb - requires sustained energy building
        float opp_alt = player_pos.z + rndf(800, 1200);
        opp_alt = clampf(opp_alt, 1500, 4500);  // Keep in bounds

        // Opponent ahead and above, flying gentle turns at LOW throttle
        float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
        Vec3 opp_pos = vec3(
            player_pos.x + rndf(400, 700),
            player_pos.y + side * rndf(50, 200),
            opp_alt
        );

        // Opponent flying VERY SLOW (50% of player speed) - easy target IF you can reach them
        float player_speed = norm3(player_vel);
        float opp_speed = player_speed * 0.5f;
        float opp_heading = atan2f(player_vel.y, player_vel.x) + side * rndf(0.1f, 0.3f);
        Vec3 opp_vel = vec3(opp_speed * cosf(opp_heading), opp_speed * sinf(opp_heading), 0);

        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

        // Opponent: very gentle turns (15° bank), LOW throttle, bleeding energy
        // They're a sitting duck - the challenge is GETTING UP THERE
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = 15.0f * DEG_TO_RAD;  // Gentle 15° bank - won't go OOB
        env->opponent.throttle = 0.25f;  // Very low throttle - bleeding energy fast

        return;
    }

    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float az_min = cfg->angle_min_deg * DEG_TO_RAD;
    float az_max = cfg->angle_max_deg * DEG_TO_RAD;
    float azimuth = side * rndf(az_min, az_max);

    float dist = rndf(300, 500);
    float phi = rndf(-0.2f, 0.2f);  // ±11° elevation

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(azimuth),
        player_pos.y + dist * sinf(azimuth),
        clampf(player_pos.z + dist * sinf(phi), 300, 4700)
    );

    float away_heading = azimuth;
    float opp_heading = away_heading + rndf(-0.35f, 0.35f);  // ±20° variance

    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

    // AP mode based on bank field: 0 = straight, >0 = turning
    if (cfg->bank > 0) {
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = (float)cfg->bank * DEG_TO_RAD;
    } else {
        env->opponent_ap.mode = AP_STRAIGHT;
    }

    // Stages 8-9: Boost player speed 15% for pursuit advantage (wide angle chase)
    if (env->stage >= CURRICULUM_SIDE_FAR) {
        env->player.vel = mul3(env->player.vel, 1.15f);
    }

    // Speed boost when opponent is above (climbing needs energy)
    if (opp_pos.z > player_pos.z) {
        env->player.vel = mul3(env->player.vel, 1.15f);
    }
}

// Stage 10: DIVE_ATTACK - Player starts 500m above, 75° nose-down for fast catch-up
// Same spawn geometry as spawn_rear (90-150° off axis), but player has massive altitude/energy advantage
static void spawn_dive_attack(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    const StageConfig* cfg = &STAGES[env->stage];

    // Same azimuth geometry as spawn_rear (90-150° off axis)
    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float az_min = cfg->angle_min_deg * DEG_TO_RAD;
    float az_max = cfg->angle_max_deg * DEG_TO_RAD;
    float azimuth = side * rndf(az_min, az_max);

    float dist = rndf(300, 500);
    // Opponent spawns 500m BELOW player (big altitude advantage)
    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(azimuth),
        player_pos.y + dist * sinf(azimuth),
        clampf(player_pos.z - 500 + rndf(-50, 50), 300, 4700)
    );

    float opp_heading = azimuth + rndf(-0.35f, 0.35f);  // ±20° variance
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_STRAIGHT : AP_LEVEL;

    // Player starts 75° nose down (same heading, just pitched)
    // Pitch rotation is around body Y-axis (right wing)
    // Positive pitch around Y = nose down in this coordinate system
    float pitch = 75.0f * DEG_TO_RAD;
    Quat pitch_quat = quat_from_axis_angle(vec3(0, 1, 0), pitch);
    env->player.ori = pitch_quat;
    // Velocity matches pitch direction (diving toward target area)
    env->player.vel = quat_rotate(pitch_quat, player_vel);
    env->player.prev_vel = env->player.vel;
}

// Stage 11: ZOOM_ATTACK - Player starts 500m below, 75° nose-up, near max speed
// Opposite of dive_attack: player zooms up toward target with high energy
static void spawn_zoom_attack(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    const StageConfig* cfg = &STAGES[env->stage];

    // Same azimuth geometry as spawn_rear (90-150° off axis)
    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float az_min = cfg->angle_min_deg * DEG_TO_RAD;
    float az_max = cfg->angle_max_deg * DEG_TO_RAD;
    float azimuth = side * rndf(az_min, az_max);

    float dist = rndf(300, 500);
    // Opponent spawns 300 ABOVE player (player zooms up)
    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(azimuth),
        player_pos.y + dist * sinf(azimuth),
        clampf(player_pos.z + 300 + rndf(-50, 50), 300, 4700)
    );

    float opp_heading = azimuth + rndf(-0.35f, 0.35f);  // ±20° variance
    float opp_speed = norm3(player_vel);
    Vec3 opp_vel = vec3(opp_speed * cosf(opp_heading), opp_speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_STRAIGHT : AP_LEVEL;

    // Player starts 75° nose UP with near-max speed (~145 m/s)
    // Pitch rotation is around body Y-axis (right wing)
    // Negative pitch around Y = nose up in this coordinate system
    float pitch = -75.0f * DEG_TO_RAD;
    Quat pitch_quat = quat_from_axis_angle(vec3(0, 1, 0), pitch);
    env->player.ori = pitch_quat;

    // Set player to high speed (reduced from 140-150 due to instability at extreme pitch)
    float zoom_speed = rndf(110, 120);
    Vec3 base_vel = vec3(zoom_speed, 0, 0);
    env->player.vel = quat_rotate(pitch_quat, base_vel);
    env->player.prev_vel = env->player.vel;

    // ZOOM_ATTACK always has altitude disadvantage - needs more time for climb + pursuit
    env->max_steps = 4500;
}

// Stages 12-13: Unified rear spawn - uses angle_min_deg, angle_max_deg, bank from STAGES
// Stage 12: Target 90-150° off axis (rear quarters), 50/50 straight/level
// Stage 13: Same geometry + 30° turns (unchanged, zoom_attack inserted before these)
static void spawn_rear(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    const StageConfig* cfg = &STAGES[env->stage];

    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float az_min = cfg->angle_min_deg * DEG_TO_RAD;
    float az_max = cfg->angle_max_deg * DEG_TO_RAD;
    float azimuth = side * rndf(az_min, az_max);

    float dist = rndf(300, 500);
    // Opponent spawns ~500m below player (large altitude advantage for rear chase)
    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(azimuth),
        player_pos.y + dist * sinf(azimuth),
        clampf(player_pos.z - 500 + rndf(-50, 50), 300, 4700)
    );

    float opp_heading = azimuth + rndf(-0.35f, 0.35f);  // ±20° variance
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

    // AP mode based on bank field: 0 = 50/50 straight/level, >0 = turning
    if (cfg->bank > 0) {
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = (float)cfg->bank * DEG_TO_RAD;
    } else {
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_STRAIGHT : AP_LEVEL;
    }

    // Speed boost for rear chase - player starts faster to close the gap
    env->player.vel = mul3(env->player.vel, 1.25f);
    env->player.prev_vel = env->player.vel;
}

// Stage 14: FULL_PREDICTABLE - 360° spawn, heading correlated (flying away)
// Teaches: Full sphere awareness with predictable heading
static void spawn_full_predictable(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Full 360° spawn
    float azimuth = rndf(-M_PI, M_PI);
    float dist = rndf(300, 600);
    float phi = rndf(-0.3f, 0.3f);  // ±17° elevation

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(azimuth) * cosf(phi),
        player_pos.y + dist * sinf(azimuth) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 300, 4700)
    );

    // KEY: Heading is CORRELATED - flying away from player
    float away_heading = azimuth;  // Same direction as spawn angle = flying away
    float opp_heading = away_heading + rndf(-0.52f, 0.52f);  // ±30° variance

    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stage 15: FULL_RANDOM - 360° spawn, random heading, 30° turns
// Teaches: Random heading (key difficulty!) - must read observation to determine velocity
static void spawn_full_random(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Random direction in 3D sphere (300-600m from player)
    float dist = rndf(300, 600);
    float theta = rndf(0, 2.0f * M_PI);      // Azimuth: 0-360°
    float phi = rndf(-0.3f, 0.3f);           // Elevation: ±17° (keep near level)

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 300, 4700)
    );

    // Random velocity direction (not necessarily toward/away from player)
    float vel_theta = rndf(0, 2.0f * M_PI);
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(vel_theta), speed * sinf(vel_theta), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);

    // Set orientation to match velocity direction (yaw rotation around Z)
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), vel_theta);

    // 3 modes: straight, level, turns (still 30° - steeper turns come in stage 16)
    float r = rndf(0, 1);
    if (r < 0.2f) env->opponent_ap.mode = AP_STRAIGHT;
    else if (r < 0.4f) env->opponent_ap.mode = AP_LEVEL;
    else env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;

    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stage 16: MEDIUM_TURNS - 360° spawn, random heading, 45° turns
// Teaches: Steeper 45° turns (first introduction of harder turns)
static void spawn_medium_turns(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Same geometry as FULL_RANDOM
    float dist = rndf(300, 600);
    float theta = rndf(0, 2.0f * M_PI);      // Azimuth: 0-360°
    float phi = rndf(-0.3f, 0.3f);           // Elevation: ±17° (keep near level)

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 300, 4700)
    );

    // Random velocity direction (uncorrelated with position)
    float vel_theta = rndf(0, 2.0f * M_PI);
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(vel_theta), speed * sinf(vel_theta), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), vel_theta);

    // 5 modes with 45° turns
    float r = rndf(0, 1);
    if (r < 0.2f) env->opponent_ap.mode = AP_STRAIGHT;
    else if (r < 0.4f) env->opponent_ap.mode = AP_LEVEL;
    else if (r < 0.6f) env->opponent_ap.mode = AP_TURN_LEFT;
    else if (r < 0.8f) env->opponent_ap.mode = AP_TURN_RIGHT;
    else env->opponent_ap.mode = AP_CLIMB;

    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stage 17: HARD_MANEUVERING - Hard turns (60°) and weave patterns
static void spawn_hard_maneuvering(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-100, 100),
        player_pos.z + rndf(-50, 50)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);

    // Pick from hard maneuver modes
    float r = rndf(0, 1);
    if (r < 0.3f) {
        env->opponent_ap.mode = AP_HARD_TURN_LEFT;
    } else if (r < 0.6f) {
        env->opponent_ap.mode = AP_HARD_TURN_RIGHT;
    } else {
        env->opponent_ap.mode = AP_WEAVE;
        env->opponent_ap.phase = rndf(0, 2.0f * M_PI);  // Random start phase
    }
}

// Stage 19: EVASIVE - Opponent reacts to player position (hardest)
static void spawn_evasive(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Override player altitude to near max (3500-4500m) for high-altitude combat
    env->player.pos.z = rndf(3500, 4500);
    player_pos.z = env->player.pos.z;  // Update local copy for opponent spawn

    // Spawn in various positions (like FULL_RANDOM)
    float dist = rndf(300, 500);
    float theta = rndf(0, 2.0f * M_PI);
    float phi = rndf(-0.3f, 0.3f);

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 2500, 4800)
    );

    float vel_theta = rndf(0, 2.0f * M_PI);
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(vel_theta), speed * sinf(vel_theta), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), vel_theta);

    // Mix of hard modes with AP_EVASIVE dominant
    float r = rndf(0, 1);
    if (r < 0.4f) {
        env->opponent_ap.mode = AP_EVASIVE;
    } else if (r < 0.55f) {
        env->opponent_ap.mode = AP_HARD_TURN_LEFT;
    } else if (r < 0.7f) {
        env->opponent_ap.mode = AP_HARD_TURN_RIGHT;
    } else if (r < 0.85f) {
        env->opponent_ap.mode = AP_WEAVE;
        env->opponent_ap.phase = rndf(0, 2.0f * M_PI);
    } else {
        // 15% chance of regular turn modes (still steep 60°)
        env->opponent_ap.mode = rndf(0,1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
    }
}

// Stage 20: AUTOACE - Intelligent adversarial opponent (two-way combat)
static void spawn_autoace(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Override player altitude to mid-high (2500-4000m)
    env->player.pos.z = rndf(2500, 4000);
    player_pos.z = env->player.pos.z;

    // Spawn opponent in various positions (360 degree, varied distance)
    float dist = rndf(400, 700);
    float theta = rndf(0, 2.0f * M_PI);
    float phi = rndf(-0.25f, 0.25f);

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 2000, 4500)
    );

    float vel_theta = rndf(0, 2.0f * M_PI);
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(vel_theta), speed * sinf(vel_theta), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), vel_theta);

    env->opponent_ap.mode = AP_PURSUIT_LAG;
    autoace_init(&env->opponent_ace);
}

// ============================================================================
// Forced Vertical Merge Spawns (self-play curriculum)
// Teach vertical fighting by spawning scenarios along the timeline of a vertical merge.
// Level 0 (apex) is easiest — agent just needs to roll and dive.
// Level 4 (pre-merge) is hardest — agent must choose to pull vertical from far out.
// ============================================================================

// Level 0: "Apex Inverted"
// Player at top of climb, inverted (belly up), ~60 m/s, 750m above opponent.
// Opponent below in a flat turn at combat speed. Agent rolls over and dives to attack.
static void spawn_vertical_apex(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    float merge_alt = rndf(2000, 2500);
    float heading = rndf(0, 2.0f * (float)M_PI);

    // Player: inverted at apex, 750m above, slow
    float player_speed = rndf(55, 65);
    float player_alt = merge_alt + 750.0f;
    env->player.pos = vec3(player_pos.x, player_pos.y, player_alt);

    // Orientation: heading + slight nose-down (5-15°) + inverted (180° roll)
    float nose_down = rndf(5, 15) * DEG_TO_RAD;
    Quat p_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), heading),
                 quat_mul(quat_from_axis_angle(vec3(0, 1, 0), nose_down),
                          quat_from_axis_angle(vec3(1, 0, 0), (float)M_PI)));
    env->player.ori = p_ori;
    env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), player_speed);
    env->player.prev_vel = env->player.vel;

    // Opponent: at merge alt, in flat turn, 90-100 m/s
    // 270° into their turn — they've been turning for ~15s, bled energy
    float opp_speed = rndf(90, 100);
    float opp_heading = heading + rndf(3.5f, 5.5f);  // opponent flew past in heading dir, turned ~200-315°
    float opp_bank = rndf(45, 60) * DEG_TO_RAD;
    float horiz_offset = rndf(200, 400);

    Vec3 opp_pos = vec3(
        player_pos.x + horiz_offset * cosf(heading),
        player_pos.y + horiz_offset * sinf(heading),
        merge_alt
    );

    Quat o_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), opp_heading),
                          quat_from_axis_angle(vec3(1, 0, 0), opp_bank));
    reset_plane(&env->opponent, opp_pos, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), opp_speed));
    env->opponent.ori = o_ori;

    env->opponent_ap.mode = AP_PURSUIT_LAG;
    autoace_init(&env->opponent_ace);
    env->max_steps = 3000;
}

// Level 1: "Past Vertical"
// Player past 90° pitch (100-120° from level), ~70 m/s decelerating, 400-500m above.
// Opponent at merge alt, 120-150° into flat turn, bleeding energy.
static void spawn_vertical_past(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    float merge_alt = rndf(2000, 2500);
    float heading = rndf(0, 2.0f * (float)M_PI);

    // Player: past vertical (100-120° pitch from level = 10-30° past straight up)
    float player_speed = rndf(65, 75);
    float alt_above = rndf(400, 500);
    float player_alt = merge_alt + alt_above;
    env->player.pos = vec3(player_pos.x, player_pos.y, player_alt);

    // Pitch: -100 to -120° (negative = nose up, past vertical)
    // This means the plane is 10-30° past straight up, going over the top
    float pitch_deg = -rndf(100, 120);
    float pitch = pitch_deg * DEG_TO_RAD;
    Quat p_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), heading),
                          quat_from_axis_angle(vec3(0, 1, 0), pitch));
    env->player.ori = p_ori;
    env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), player_speed);
    env->player.prev_vel = env->player.vel;

    // Opponent: at merge alt, 700m ahead, flying away turned 45° left or right
    float opp_speed = rndf(85, 95);
    float turn_sign = (rand() % 2) ? 1.0f : -1.0f;
    float opp_heading = heading + (float)M_PI + turn_sign * 45.0f * DEG_TO_RAD;
    float opp_bank = turn_sign * 45.0f * DEG_TO_RAD;
    float horiz_offset = 700.0f;

    Vec3 opp_pos = vec3(
        player_pos.x + horiz_offset * cosf(heading + (float)M_PI),
        player_pos.y + horiz_offset * sinf(heading + (float)M_PI),
        merge_alt
    );

    Quat o_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), opp_heading),
                          quat_from_axis_angle(vec3(1, 0, 0), opp_bank));
    reset_plane(&env->opponent, opp_pos, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), opp_speed));
    env->opponent.ori = o_ori;

    env->opponent_ap.mode = AP_PURSUIT_LAG;
    autoace_init(&env->opponent_ace);
    env->max_steps = 3500;
}

// Level 2: "Mid-Climb"
// Player at 55-65° nose-up, 85-95 m/s, 100-200m above merge alt.
// Opponent at merge alt, just 30-60° into flat turn, starting to bleed speed.
static void spawn_vertical_midclimb(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    float merge_alt = rndf(2000, 2500);
    float heading = rndf(0, 2.0f * (float)M_PI);

    // Player: mid-climb, 55-65° nose-up
    float player_speed = rndf(85, 95);
    float alt_above = rndf(100, 200);
    float player_alt = merge_alt + alt_above;
    env->player.pos = vec3(player_pos.x, player_pos.y, player_alt);

    // Pitch: -55 to -65° (negative = nose up)
    float pitch_deg = -rndf(55, 65);
    float pitch = pitch_deg * DEG_TO_RAD;
    Quat p_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), heading),
                          quat_from_axis_angle(vec3(0, 1, 0), pitch));
    env->player.ori = p_ori;
    env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), player_speed);
    env->player.prev_vel = env->player.vel;

    // Opponent: at merge alt, 700m ahead, flying away turned 45° left or right
    float opp_speed = rndf(95, 105);
    float turn_sign = (rand() % 2) ? 1.0f : -1.0f;
    float opp_heading = heading + (float)M_PI + turn_sign * 45.0f * DEG_TO_RAD;
    float opp_bank = turn_sign * 45.0f * DEG_TO_RAD;
    float horiz_offset = 700.0f;

    Vec3 opp_pos = vec3(
        player_pos.x + horiz_offset * cosf(heading + (float)M_PI),
        player_pos.y + horiz_offset * sinf(heading + (float)M_PI),
        merge_alt
    );

    Quat o_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), opp_heading),
                          quat_from_axis_angle(vec3(1, 0, 0), opp_bank));
    reset_plane(&env->opponent, opp_pos, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), opp_speed));
    env->opponent.ori = o_ori;

    env->opponent_ap.mode = AP_PURSUIT_LAG;
    autoace_init(&env->opponent_ace);
    env->max_steps = 4000;
}

// Level 3: "Merge"
// Both nose-on, co-altitude, 400-600m apart, closing fast.
// Player has 5-15 m/s speed advantage. Agent must discover vertical pull beats flat turn.
static void spawn_vertical_merge(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    float merge_alt = rndf(2500, 3500);
    float heading = rndf(0, 2.0f * (float)M_PI);
    float dist = rndf(400, 600);

    // Player: heading toward opponent, speed advantage
    float player_speed = rndf(100, 110);
    env->player.pos = vec3(player_pos.x, player_pos.y, merge_alt);
    Quat p_ori = quat_from_axis_angle(vec3(0, 0, 1), heading);
    env->player.ori = p_ori;
    env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), player_speed);
    env->player.prev_vel = env->player.vel;

    // Opponent: heading toward player (opposite heading), slightly slower
    float opp_speed = player_speed - rndf(5, 15);
    float opp_heading = heading + (float)M_PI;
    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(heading),
        player_pos.y + dist * sinf(heading),
        merge_alt + rndf(-20, 20)  // Near co-altitude
    );

    Quat o_ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
    reset_plane(&env->opponent, opp_pos, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), opp_speed));
    env->opponent.ori = o_ori;

    env->head_on_lockout = 1;  // Disable guns until planes pass
    Vec3 rel_pos = sub3(env->opponent.pos, env->player.pos);
    Vec3 rel_vel = sub3(env->opponent.vel, env->player.vel);
    env->prev_rel_dot = dot3(rel_pos, rel_vel);

    env->opponent_ap.mode = AP_PURSUIT_LAG;
    autoace_init(&env->opponent_ace);
    env->max_steps = 5000;
}

// Level 4: "Pre-Merge"
// 800-1200m apart, approaching. Player has 100-200m altitude advantage.
// Agent must plan the vertical pull from further out.
static void spawn_vertical_premerge(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    float base_alt = rndf(2500, 3500);
    float heading = rndf(0, 2.0f * (float)M_PI);
    float dist = rndf(800, 1200);
    float alt_adv = rndf(100, 200);

    // Player: heading toward opponent, slight altitude advantage
    float player_speed = rndf(95, 110);
    env->player.pos = vec3(player_pos.x, player_pos.y, base_alt + alt_adv);
    Quat p_ori = quat_from_axis_angle(vec3(0, 0, 1), heading);
    env->player.ori = p_ori;
    env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), player_speed);
    env->player.prev_vel = env->player.vel;

    // Opponent: heading toward player, at base altitude
    float opp_speed = rndf(90, 105);
    float opp_heading = heading + (float)M_PI;
    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(heading),
        player_pos.y + dist * sinf(heading),
        base_alt
    );

    Quat o_ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
    reset_plane(&env->opponent, opp_pos, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), opp_speed));
    env->opponent.ori = o_ori;

    env->head_on_lockout = 1;
    Vec3 rel_pos = sub3(env->opponent.pos, env->player.pos);
    Vec3 rel_vel = sub3(env->opponent.vel, env->player.vel);
    env->prev_rel_dot = dot3(rel_pos, rel_vel);

    env->opponent_ap.mode = AP_PURSUIT_LAG;
    autoace_init(&env->opponent_ace);
    env->max_steps = 6000;
}

// EVAL spawn: True randomization with alternating advantages
// Used when curriculum_randomize=1 - creates varied, fair combat scenarios
static void spawn_eval_random(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Always clear head-on lockout first (only set if we choose head-on spawn)
    env->head_on_lockout = 0;
    env->prev_rel_dot = 0.0f;

    // Alternate who gets advantage based on episode count
    int player_advantage = (env->total_episodes % 2 == 0);

    // Random spawn type distribution:
    // 40% - tactical (one behind/side of other)
    // 30% - neutral (both at angles, neither clearly advantaged)
    // 20% - energy (altitude/speed difference)
    // 10% - head-on (with gun lockout until pass)
    float spawn_roll = rndf(0, 1);

    // Base altitude for combat (mid-altitude)
    float base_alt = rndf(2000, 3500);
    env->player.pos.z = base_alt;
    player_pos.z = base_alt;
    float speed = norm3(player_vel);

    if (spawn_roll < 0.40f) {
        // TACTICAL: One plane behind/side of other (clear advantage)
        float dist = rndf(300, 600);
        float angle_off = rndf(120, 180) * DEG_TO_RAD;  // Behind (120-180° off nose)
        float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;

        if (player_advantage) {
            // Player behind opponent - player has advantage
            float opp_heading = rndf(0, 2.0f * M_PI);
            Vec3 opp_pos = vec3(
                player_pos.x + rndf(300, 500),
                player_pos.y + side * rndf(50, 150),
                clampf(player_pos.z + rndf(-100, 100), 500, 4500)
            );
            Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
            reset_plane(&env->opponent, opp_pos, opp_vel);
            env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
            // Player heading toward opponent
            Vec3 to_opp = sub3(opp_pos, player_pos);
            float player_heading = atan2f(to_opp.y, to_opp.x);
            env->player.vel = vec3(speed * cosf(player_heading), speed * sinf(player_heading), 0);
            env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), player_heading);
        } else {
            // Opponent behind player - opponent has advantage
            float player_heading = rndf(0, 2.0f * M_PI);
            env->player.vel = vec3(speed * cosf(player_heading), speed * sinf(player_heading), 0);
            env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), player_heading);
            // Opponent behind
            Vec3 opp_pos = vec3(
                player_pos.x - cosf(player_heading) * dist + side * sinf(player_heading) * rndf(50, 150),
                player_pos.y - sinf(player_heading) * dist - side * cosf(player_heading) * rndf(50, 150),
                clampf(player_pos.z + rndf(-100, 100), 500, 4500)
            );
            Vec3 to_player = sub3(player_pos, opp_pos);
            float opp_heading = atan2f(to_player.y, to_player.x);
            Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
            reset_plane(&env->opponent, opp_pos, opp_vel);
            env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
        }
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = rndf(30, 60) * DEG_TO_RAD;

    } else if (spawn_roll < 0.70f) {
        // NEUTRAL: Both at angles, converging - fair fight
        float dist = rndf(400, 700);
        float theta = rndf(0, 2.0f * M_PI);
        Vec3 opp_pos = vec3(
            player_pos.x + dist * cosf(theta),
            player_pos.y + dist * sinf(theta),
            clampf(player_pos.z + rndf(-200, 200), 500, 4500)
        );
        // Both heading toward a point between them (converging)
        Vec3 midpoint = mul3(add3(player_pos, opp_pos), 0.5f);
        Vec3 player_to_mid = sub3(midpoint, player_pos);
        Vec3 opp_to_mid = sub3(midpoint, opp_pos);
        // Add some angle offset so they're not perfectly converging
        float player_heading = atan2f(player_to_mid.y, player_to_mid.x) + rndf(-0.5f, 0.5f);
        float opp_heading = atan2f(opp_to_mid.y, opp_to_mid.x) + rndf(-0.5f, 0.5f);

        env->player.vel = vec3(speed * cosf(player_heading), speed * sinf(player_heading), 0);
        env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), player_heading);
        Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = rndf(30, 45) * DEG_TO_RAD;

    } else if (spawn_roll < 0.90f) {
        // ENERGY: Altitude or speed advantage
        float dist = rndf(400, 600);
        float theta = rndf(0, 2.0f * M_PI);
        float alt_diff = rndf(300, 600);  // Significant altitude difference

        Vec3 opp_pos;
        if (player_advantage) {
            // Player higher (energy advantage)
            env->player.pos.z = base_alt + alt_diff;
            player_pos.z = env->player.pos.z;
            opp_pos = vec3(
                player_pos.x + dist * cosf(theta),
                player_pos.y + dist * sinf(theta),
                base_alt
            );
        } else {
            // Opponent higher (energy advantage)
            opp_pos = vec3(
                player_pos.x + dist * cosf(theta),
                player_pos.y + dist * sinf(theta),
                base_alt + alt_diff
            );
        }
        // Random headings
        float player_heading = rndf(0, 2.0f * M_PI);
        float opp_heading = rndf(0, 2.0f * M_PI);
        env->player.vel = vec3(speed * cosf(player_heading), speed * sinf(player_heading), 0);
        env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), player_heading);
        Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
        env->opponent_ap.mode = AP_PURSUIT_LEAD;  // Aggressive pursuit for energy fights

    } else {
        // HEAD-ON: Facing each other (rare, 10%) - guns locked until they pass
        float dist = rndf(600, 900);  // Start further apart
        float theta = rndf(0, 2.0f * M_PI);

        Vec3 opp_pos = vec3(
            player_pos.x + dist * cosf(theta),
            player_pos.y + dist * sinf(theta),
            clampf(player_pos.z + rndf(-100, 100), 500, 4500)
        );
        // Player faces opponent
        Vec3 to_opp = sub3(opp_pos, player_pos);
        float player_heading = atan2f(to_opp.y, to_opp.x);
        // Opponent faces player (opposite direction)
        float opp_heading = player_heading + M_PI;

        env->player.vel = vec3(speed * cosf(player_heading), speed * sinf(player_heading), 0);
        env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), player_heading);
        Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

        // HEAD-ON LOCKOUT: Disable guns until they pass each other
        env->head_on_lockout = 1;
        // Initialize tracking for pass detection
        Vec3 rel_pos = sub3(opp_pos, player_pos);
        Vec3 rel_vel = sub3(opp_vel, env->player.vel);
        env->prev_rel_dot = dot3(rel_pos, rel_vel);

        env->opponent_ap.mode = AP_STRAIGHT;  // Fly straight initially
        if (DEBUG >= 1) {
            fprintf(stderr, "[EVAL-SPAWN] Head-on spawn - guns locked until pass\n");
        }
    }

    // Reset autopilot PID state
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;

    if (DEBUG >= 1) {
        fprintf(stderr, "[EVAL-SPAWN] ep=%d advantage=%s spawn_type=%.0f%% dist=%.0fm\n",
                env->total_episodes, player_advantage ? "PLAYER" : "OPPONENT",
                spawn_roll * 100, norm3(sub3(env->opponent.pos, env->player.pos)));
    }
}

// Test spawn: Opponent behind player with advantage but not instant kill
// Player is 30° off opponent's nose - opponent must maneuver to get the shot
// Opponent is 400m behind, clear positional advantage
static void spawn_eval_opponent_advantage(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    env->head_on_lockout = 0;
    env->prev_rel_dot = 0.0f;

    // Player at base altitude, flying straight along +X
    float base_alt = 2500.0f;
    float speed = norm3(player_vel);
    if (speed < 70.0f) speed = 80.0f;

    env->player.pos = vec3(0, 0, base_alt);
    env->player.vel = vec3(speed, 0, 0);
    env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), 0.0f);  // Flying +X
    env->player.throttle = 0.5f;

    // Opponent 400m behind player
    float dist = 400.0f;
    Vec3 opp_pos = vec3(-dist, 0, base_alt);  // Directly behind player

    // Opponent heading: 30° off from pointing at player
    // Player is at (0,0), opponent at (-400,0)
    // Direct heading to player would be 0° (pointing +X)
    // We offset 30° so player is 30° off opponent's nose
    float angle_off_nose = 30.0f * DEG_TO_RAD;
    float opp_heading = angle_off_nose;  // Pointing 30° left of player

    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
    env->opponent.throttle = 0.6f;

    // Autopilot: pursuit mode to track player
    env->opponent_ap.mode = AP_PURSUIT_LEAD;
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;

    if (DEBUG >= 1) {
        Vec3 to_player = sub3(env->player.pos, opp_pos);
        float actual_dist = norm3(to_player);
        Vec3 opp_fwd = quat_rotate(env->opponent.ori, vec3(1, 0, 0));
        Vec3 to_player_norm = normalize3(to_player);
        float aim_dot = dot3(opp_fwd, to_player_norm);
        float aim_angle = acosf(clampf(aim_dot, -1.0f, 1.0f)) * RAD_TO_DEG;
        fprintf(stderr, "[EVAL-OPP-ADV] dist=%.0fm aim_angle=%.1f° (cone=5°)\n",
                actual_dist, aim_angle);
    }
}

// EVAL spawn mode 2: Symmetric scenario pool for fair Elo evaluation
// Randomly selects from 3 scenarios: head-on merge, post-merge zoom, turning fight
// All scenarios are symmetric with slight perturbations to break identical observations
static void spawn_eval_merge(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    float speed = norm3(player_vel);
    if (speed < 70.0f) speed = 80.0f;

    // Shared: random center altitude and merge axis
    float base_alt = rndf(2500, 3500);
    float theta = rndf(0, 2.0f * M_PI);  // merge axis heading

    // Tiny asymmetric perturbations (break identical obs, no real advantage)
    float pos_jitter = rndf(-5, 5);
    float alt_jitter = rndf(-5, 5);
    float speed_jitter = rndf(-3, 3);
    float angle_jitter = rndf(-0.035f, 0.035f);  // ~±2°

    int scenario = (int)(rndf(0, 2.999f));  // 0, 1, or 2

    if (scenario == 0) {
        // === Scenario 1: Head-On Merge ===
        // Classic merge. Both approaching, guns locked until pass.
        float half_dist = rndf(300, 450);
        float p_speed = speed + speed_jitter;
        float o_speed = speed - speed_jitter;

        Vec3 p_pos = vec3(
            player_pos.x - half_dist * cosf(theta) + pos_jitter,
            player_pos.y - half_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Vec3 opp_pos = vec3(
            player_pos.x + half_dist * cosf(theta) - pos_jitter,
            player_pos.y + half_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );

        float p_heading = theta + angle_jitter;
        float o_heading = theta + (float)M_PI - angle_jitter;

        // Player
        env->player.pos = p_pos;
        env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), p_heading);
        env->player.vel = vec3(p_speed * cosf(p_heading), p_speed * sinf(p_heading), 0);

        // Opponent
        Vec3 opp_vel = vec3(o_speed * cosf(o_heading), o_speed * sinf(o_heading), 0);
        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), o_heading);

        env->head_on_lockout = 1;

        // Initialize pass detection tracking
        Vec3 rel_pos = sub3(opp_pos, p_pos);
        Vec3 rel_vel = sub3(opp_vel, env->player.vel);
        env->prev_rel_dot = dot3(rel_pos, rel_vel);

        if (DEBUG >= 1) {
            fprintf(stderr, "[EVAL-MERGE] scenario=HEAD_ON dist=%.0fm alt=%.0fm heading=%.1f°\n",
                    half_dist * 2, base_alt, theta * RAD_TO_DEG);
        }

    } else if (scenario == 1) {
        // === Scenario 2: Post-Merge Zoom ===
        // Both just passed and pulled up. Who manages energy better?
        // Flying AWAY from each other, both climbing nose-up.
        float half_dist = rndf(100, 200);
        float pitch_angle = rndf(30, 50) * DEG_TO_RAD;
        float zoom_speed = rndf(70, 90);
        float p_speed = zoom_speed + speed_jitter;
        float o_speed = zoom_speed - speed_jitter;

        // Positions: separated, backs to each other
        // Player flies along +theta, opponent flies along +theta+PI (away from each other)
        Vec3 p_pos = vec3(
            player_pos.x - half_dist * cosf(theta) + pos_jitter,
            player_pos.y - half_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Vec3 opp_pos = vec3(
            player_pos.x + half_dist * cosf(theta) - pos_jitter,
            player_pos.y + half_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );

        // Player heading: away from opponent (along +theta direction)
        float p_heading = theta + angle_jitter;
        // Opponent heading: away from player (along +theta+PI direction)
        float o_heading = theta + (float)M_PI - angle_jitter;

        // Orientation: heading rotation, then pitch up
        // Compose: pitch around body Y, then heading around world Z
        Quat p_heading_q = quat_from_axis_angle(vec3(0, 0, 1), p_heading);
        Quat p_pitch_q = quat_from_axis_angle(vec3(0, 1, 0), -pitch_angle);  // negative = nose up (Z up convention)
        Quat p_ori = quat_mul(p_heading_q, p_pitch_q);

        Quat o_heading_q = quat_from_axis_angle(vec3(0, 0, 1), o_heading);
        Quat o_pitch_q = quat_from_axis_angle(vec3(0, 1, 0), -pitch_angle);
        Quat o_ori = quat_mul(o_heading_q, o_pitch_q);

        // Velocity aligned with nose direction
        Vec3 p_vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), p_speed);
        Vec3 o_vel = mul3(quat_rotate(o_ori, vec3(1, 0, 0)), o_speed);

        env->player.pos = p_pos;
        env->player.ori = p_ori;
        env->player.vel = p_vel;

        reset_plane(&env->opponent, opp_pos, o_vel);
        env->opponent.ori = o_ori;

        env->head_on_lockout = 0;
        env->prev_rel_dot = 0.0f;

        if (DEBUG >= 1) {
            fprintf(stderr, "[EVAL-MERGE] scenario=POST_MERGE_ZOOM dist=%.0fm alt=%.0fm pitch=%.0f° heading=%.1f°\n",
                    half_dist * 2, base_alt, pitch_angle * RAD_TO_DEG, theta * RAD_TO_DEG);
        }

    } else {
        // === Scenario 3: Turning Fight ===
        // Engaged in a turning fight. Both banked, pulling toward each other.
        float half_dist = rndf(150, 250);
        float bank_angle = rndf(45, 60) * DEG_TO_RAD;
        float pitch_angle = rndf(5, 10) * DEG_TO_RAD;
        float turn_speed = rndf(70, 85);
        float p_speed = turn_speed + speed_jitter;
        float o_speed = turn_speed - speed_jitter;

        Vec3 p_pos = vec3(
            player_pos.x - half_dist * cosf(theta) + pos_jitter,
            player_pos.y - half_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Vec3 opp_pos = vec3(
            player_pos.x + half_dist * cosf(theta) - pos_jitter,
            player_pos.y + half_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );

        // Both heading roughly toward each other, but offset ~45° to simulate a turn
        float turn_offset = rndf(30, 60) * DEG_TO_RAD;
        float p_heading = theta + turn_offset + angle_jitter;
        float o_heading = theta + (float)M_PI - turn_offset - angle_jitter;

        // Player banks left (toward opponent), opponent banks right (toward player)
        // Since they face each other, mirrored bank = same direction of turn
        Quat p_heading_q = quat_from_axis_angle(vec3(0, 0, 1), p_heading);
        Quat p_pitch_q = quat_from_axis_angle(vec3(0, 1, 0), -pitch_angle);
        Quat p_bank_q = quat_from_axis_angle(vec3(1, 0, 0), -bank_angle);  // bank left
        Quat p_ori = quat_mul(p_heading_q, quat_mul(p_pitch_q, p_bank_q));

        Quat o_heading_q = quat_from_axis_angle(vec3(0, 0, 1), o_heading);
        Quat o_pitch_q = quat_from_axis_angle(vec3(0, 1, 0), -pitch_angle);
        Quat o_bank_q = quat_from_axis_angle(vec3(1, 0, 0), bank_angle);   // bank right (mirrored)
        Quat o_ori = quat_mul(o_heading_q, quat_mul(o_pitch_q, o_bank_q));

        Vec3 p_vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), p_speed);
        Vec3 o_vel = mul3(quat_rotate(o_ori, vec3(1, 0, 0)), o_speed);

        env->player.pos = p_pos;
        env->player.ori = p_ori;
        env->player.vel = p_vel;

        reset_plane(&env->opponent, opp_pos, o_vel);
        env->opponent.ori = o_ori;

        env->head_on_lockout = 0;
        env->prev_rel_dot = 0.0f;

        if (DEBUG >= 1) {
            fprintf(stderr, "[EVAL-MERGE] scenario=TURNING_FIGHT dist=%.0fm alt=%.0fm bank=%.0f° heading=%.1f°\n",
                    half_dist * 2, base_alt, bank_angle * RAD_TO_DEG, theta * RAD_TO_DEG);
        }
    }
}

static void spawn_eval_midfight(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    float speed = norm3(player_vel);
    if (speed < 70.0f) speed = 80.0f;

    float base_alt = rndf(2500, 3500);
    float theta = rndf(0, 2.0f * M_PI);  // merge axis heading

    // Tiny asymmetric perturbations
    float pos_jitter = rndf(-5, 5);
    float alt_jitter = rndf(-5, 5);
    float speed_jitter = rndf(-3, 3);
    float angle_jitter = rndf(-0.035f, 0.035f);  // ~±2°

    // Alternate who gets which role
    int swap_roles = (env->total_episodes % 2);

    int scenario = (int)(rndf(0, 4.999f));  // 0-4

    if (scenario == 0) {
        // === Rolling Scissors ===
        // Crossing paths, hard banks opposite directions, both pulling up
        float half_dist = rndf(75, 125);
        float bank = rndf(60, 80) * DEG_TO_RAD;
        float pitch = rndf(15, 25) * DEG_TO_RAD;
        float scr_speed = rndf(65, 75);
        float p_speed = scr_speed + speed_jitter;
        float o_speed = scr_speed - speed_jitter;

        // Crossing angle: ~60-90° off from head-on
        float cross_offset = rndf(30, 45) * DEG_TO_RAD;
        float p_heading = theta + cross_offset + angle_jitter;
        float o_heading = theta + (float)M_PI - cross_offset - angle_jitter;

        Vec3 p_pos = vec3(
            player_pos.x - half_dist * cosf(theta) + pos_jitter,
            player_pos.y - half_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Vec3 opp_pos = vec3(
            player_pos.x + half_dist * cosf(theta) - pos_jitter,
            player_pos.y + half_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );

        // Player banks left, opponent banks right (crossing)
        float p_bank = swap_roles ? bank : -bank;
        float o_bank = swap_roles ? -bank : bank;

        Quat p_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), p_heading),
                     quat_mul(quat_from_axis_angle(vec3(0, 1, 0), -pitch),
                              quat_from_axis_angle(vec3(1, 0, 0), p_bank)));
        Quat o_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), o_heading),
                     quat_mul(quat_from_axis_angle(vec3(0, 1, 0), -pitch),
                              quat_from_axis_angle(vec3(1, 0, 0), o_bank)));

        env->player.pos = p_pos;
        env->player.ori = p_ori;
        env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), p_speed);

        reset_plane(&env->opponent, opp_pos, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), o_speed));
        env->opponent.ori = o_ori;

        env->head_on_lockout = 1;
        Vec3 rel_pos_sc0 = sub3(env->opponent.pos, env->player.pos);
        Vec3 rel_vel_sc0 = sub3(env->opponent.vel, env->player.vel);
        env->prev_rel_dot = dot3(rel_pos_sc0, rel_vel_sc0);

        if (DEBUG >= 1)
            fprintf(stderr, "[EVAL-MIDFIGHT] scenario=ROLLING_SCISSORS dist=%.0fm alt=%.0fm lockout=1\n",
                    half_dist * 2, base_alt);

    } else if (scenario == 1) {
        // === High Yo-Yo ===
        // Attacker above pulling down, defender turning hard below
        float alt_sep = rndf(300, 500);
        float horiz_dist = rndf(200, 400);
        float atk_pitch = -rndf(25, 35) * DEG_TO_RAD;  // nose down
        float atk_bank = rndf(30, 50) * DEG_TO_RAD;
        float def_bank = rndf(50, 65) * DEG_TO_RAD;
        float atk_speed = rndf(85, 95);
        float def_speed = rndf(70, 80);

        Vec3 hi_pos = vec3(
            player_pos.x - horiz_dist * cosf(theta) + pos_jitter,
            player_pos.y - horiz_dist * sinf(theta),
            clampf(base_alt + alt_sep / 2 - alt_jitter, 500, 4500)
        );
        Vec3 lo_pos = vec3(
            player_pos.x + horiz_dist * cosf(theta) - pos_jitter,
            player_pos.y + horiz_dist * sinf(theta),
            clampf(base_alt - alt_sep / 2 + alt_jitter, 500, 4500)
        );

        // Attacker: nose down + banked, heading toward defender
        float atk_heading = theta + angle_jitter;
        Quat atk_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), atk_heading),
                       quat_mul(quat_from_axis_angle(vec3(0, 1, 0), -atk_pitch),
                                quat_from_axis_angle(vec3(1, 0, 0), -atk_bank)));

        // Defender: level, hard bank turn (perpendicular to merge axis)
        float def_heading = theta + (float)M_PI / 2 + angle_jitter;
        Quat def_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), def_heading),
                                quat_from_axis_angle(vec3(1, 0, 0), -def_bank));

        Vec3 *p_pos_ptr, *o_pos_ptr;
        Quat p_ori, o_ori;
        float p_speed, o_speed;
        if (swap_roles) {
            p_pos_ptr = &lo_pos; o_pos_ptr = &hi_pos;
            p_ori = def_ori; o_ori = atk_ori;
            p_speed = def_speed + speed_jitter; o_speed = atk_speed - speed_jitter;
        } else {
            p_pos_ptr = &hi_pos; o_pos_ptr = &lo_pos;
            p_ori = atk_ori; o_ori = def_ori;
            p_speed = atk_speed + speed_jitter; o_speed = def_speed - speed_jitter;
        }

        env->player.pos = *p_pos_ptr;
        env->player.ori = p_ori;
        env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), p_speed);

        reset_plane(&env->opponent, *o_pos_ptr, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), o_speed));
        env->opponent.ori = o_ori;

        env->head_on_lockout = 0;
        env->prev_rel_dot = 0.0f;

        if (DEBUG >= 1)
            fprintf(stderr, "[EVAL-MIDFIGHT] scenario=HIGH_YOYO alt_sep=%.0fm horiz=%.0fm\n",
                    alt_sep, horiz_dist);

    } else if (scenario == 2) {
        // === Overshoot ===
        // One just overshot, scrambling to re-engage. Other reversing behind.
        float along_dist = rndf(150, 250);  // how far ahead the overshooting plane is
        float behind_dist = rndf(100, 200);
        float overshoot_speed = rndf(95, 110);
        float reversal_speed = rndf(70, 80);
        float reversal_bank = rndf(55, 70) * DEG_TO_RAD;

        // Overshooting plane: flying straight past, wings level
        float fwd_heading = theta + angle_jitter;
        Vec3 fwd_pos = vec3(
            player_pos.x + along_dist * cosf(theta) + pos_jitter,
            player_pos.y + along_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Quat fwd_ori = quat_from_axis_angle(vec3(0, 0, 1), fwd_heading);
        Vec3 fwd_vel = vec3(overshoot_speed * cosf(fwd_heading), overshoot_speed * sinf(fwd_heading), 0);

        // Reversing plane: behind, in hard bank reversal turn
        float rev_heading = theta + rndf(0.35f, 0.70f);  // ~20-40° off from straight chase
        Vec3 rev_pos = vec3(
            player_pos.x - behind_dist * cosf(theta) - pos_jitter,
            player_pos.y - behind_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );
        Quat rev_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), rev_heading),
                                quat_from_axis_angle(vec3(1, 0, 0), -reversal_bank));
        Vec3 rev_vel = mul3(quat_rotate(rev_ori, vec3(1, 0, 0)), reversal_speed);

        if (swap_roles) {
            env->player.pos = fwd_pos;
            env->player.ori = fwd_ori;
            env->player.vel = fwd_vel;
            reset_plane(&env->opponent, rev_pos, rev_vel);
            env->opponent.ori = rev_ori;
        } else {
            env->player.pos = rev_pos;
            env->player.ori = rev_ori;
            env->player.vel = rev_vel;
            reset_plane(&env->opponent, fwd_pos, fwd_vel);
            env->opponent.ori = fwd_ori;
        }

        env->head_on_lockout = 0;
        env->prev_rel_dot = 0.0f;

        if (DEBUG >= 1)
            fprintf(stderr, "[EVAL-MIDFIGHT] scenario=OVERSHOOT fwd=%.0fm behind=%.0fm\n",
                    along_dist, behind_dist);

    } else if (scenario == 3) {
        // === Vertical Fight ===
        // Both climbing in a vertical rolling engagement
        float half_dist = rndf(100, 150);
        float pitch = rndf(50, 70) * DEG_TO_RAD;
        float bank = rndf(25, 35) * DEG_TO_RAD;
        float climb_speed = rndf(75, 85);
        float p_speed = climb_speed + speed_jitter;
        float o_speed = climb_speed - speed_jitter;

        Vec3 p_pos = vec3(
            player_pos.x - half_dist * cosf(theta) + pos_jitter,
            player_pos.y - half_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Vec3 opp_pos = vec3(
            player_pos.x + half_dist * cosf(theta) - pos_jitter,
            player_pos.y + half_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );

        // Both climbing, banked opposite directions
        float p_heading = theta + rndf(-0.17f, 0.17f) + angle_jitter;  // ~±10° heading spread
        float o_heading = theta + (float)M_PI + rndf(-0.17f, 0.17f) - angle_jitter;

        float p_bank = swap_roles ? bank : -bank;
        float o_bank = swap_roles ? -bank : bank;

        Quat p_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), p_heading),
                     quat_mul(quat_from_axis_angle(vec3(0, 1, 0), -pitch),
                              quat_from_axis_angle(vec3(1, 0, 0), p_bank)));
        Quat o_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), o_heading),
                     quat_mul(quat_from_axis_angle(vec3(0, 1, 0), -pitch),
                              quat_from_axis_angle(vec3(1, 0, 0), o_bank)));

        env->player.pos = p_pos;
        env->player.ori = p_ori;
        env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), p_speed);

        reset_plane(&env->opponent, opp_pos, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), o_speed));
        env->opponent.ori = o_ori;

        env->head_on_lockout = 1;
        Vec3 rel_pos_sc3 = sub3(env->opponent.pos, env->player.pos);
        Vec3 rel_vel_sc3 = sub3(env->opponent.vel, env->player.vel);
        env->prev_rel_dot = dot3(rel_pos_sc3, rel_vel_sc3);

        if (DEBUG >= 1)
            fprintf(stderr, "[EVAL-MIDFIGHT] scenario=VERTICAL pitch=%.0f° bank=%.0f° dist=%.0fm lockout=1\n",
                    pitch * RAD_TO_DEG, bank * RAD_TO_DEG, half_dist * 2);

    } else {
        // === Split-S Entry ===
        // One inverted pulling through, other pursuing
        float sep_dist = rndf(300, 400);
        float inv_speed = rndf(80, 90);
        float pursue_speed = rndf(75, 85);
        float inv_pitch = rndf(5, 15) * DEG_TO_RAD;  // slightly nose-down
        float pursue_bank = rndf(25, 35) * DEG_TO_RAD;

        // Inverted plane: ahead, upside down, slightly nose-down
        float inv_heading = theta + angle_jitter;
        Vec3 inv_pos = vec3(
            player_pos.x + pos_jitter,
            player_pos.y,
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Quat inv_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), inv_heading),
                       quat_mul(quat_from_axis_angle(vec3(1, 0, 0), (float)M_PI),  // inverted
                                quat_from_axis_angle(vec3(0, 1, 0), inv_pitch)));   // nose-down when inverted

        // Pursuer: behind, banked, chasing
        float pursue_heading = theta - angle_jitter;
        Vec3 pursue_pos = vec3(
            player_pos.x - sep_dist * cosf(theta) - pos_jitter,
            player_pos.y - sep_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );
        Quat pursue_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), pursue_heading),
                                   quat_from_axis_angle(vec3(1, 0, 0), -pursue_bank));

        if (swap_roles) {
            env->player.pos = pursue_pos;
            env->player.ori = pursue_ori;
            env->player.vel = mul3(quat_rotate(pursue_ori, vec3(1, 0, 0)), pursue_speed + speed_jitter);
            reset_plane(&env->opponent, inv_pos, mul3(quat_rotate(inv_ori, vec3(1, 0, 0)), inv_speed - speed_jitter));
            env->opponent.ori = inv_ori;
        } else {
            env->player.pos = inv_pos;
            env->player.ori = inv_ori;
            env->player.vel = mul3(quat_rotate(inv_ori, vec3(1, 0, 0)), inv_speed + speed_jitter);
            reset_plane(&env->opponent, pursue_pos, mul3(quat_rotate(pursue_ori, vec3(1, 0, 0)), pursue_speed - speed_jitter));
            env->opponent.ori = pursue_ori;
        }

        env->head_on_lockout = 0;
        env->prev_rel_dot = 0.0f;

        if (DEBUG >= 1)
            fprintf(stderr, "[EVAL-MIDFIGHT] scenario=SPLIT_S sep=%.0fm alt=%.0fm\n",
                    sep_dist, base_alt);
    }
}

// Master spawn function: dispatches to stage-specific spawner
void spawn_by_curriculum(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    env->vertical_spawn_used = 0;  // Clear flag (set by vertical spawn intercept)

    // For eval mode (curriculum_randomize=1), use spawn based on eval_spawn_mode
    if (env->curriculum_randomize) {
        if (env->eval_spawn_mode == 1) {
            // Mode 1: opponent advantage - for testing if opponent can kill player
            spawn_eval_opponent_advantage(env, player_pos, player_vel);
        } else if (env->eval_spawn_mode == 2) {
            // Mode 2: symmetric merge - fair Elo evaluation
            spawn_eval_merge(env, player_pos, player_vel);
        } else if (env->eval_spawn_mode == 3) {
            // Mode 3: mid-fight scenarios - banked/pitched engaged orientations
            spawn_eval_midfight(env, player_pos, player_vel);
        } else {
            // Mode 0 (default): random spawn
            spawn_eval_random(env, player_pos, player_vel);
        }
        // Eval mode uses stage 20 (AutoAce) max_steps for fair combat duration
        env->max_steps = STAGES[CURRICULUM_AUTOACE].max_steps;  // 6000
        return;
    }

    CurriculumStage new_stage = get_curriculum_stage(env);

    // Log stage transitions
    if (new_stage != env->stage) {
        if (DEBUG >= 1) {
            fprintf(stderr, "[STAGE_CHANGE] ptr=%p env=%d eps=%d: stage %d -> %d\n",
                   (void*)env, env->env_num, env->total_episodes, env->stage, new_stage);
            fflush(stderr);
        }
        env->stage = new_stage;
    }

    // Forced vertical merge: during self-play, chance to override spawn geometry
    if (env->selfplay_active && env->vertical_spawn_prob > 0.0f
        && env->stage == CURRICULUM_AUTOACE) {
        if (rndf(0, 1) < env->vertical_spawn_prob) {
            switch (env->vertical_level) {
                case 0: spawn_vertical_apex(env, player_pos, player_vel); break;
                case 1: spawn_vertical_past(env, player_pos, player_vel); break;
                case 2: spawn_vertical_midclimb(env, player_pos, player_vel); break;
                case 3: spawn_vertical_merge(env, player_pos, player_vel); break;
                case 4: spawn_vertical_premerge(env, player_pos, player_vel); break;
                default: spawn_vertical_apex(env, player_pos, player_vel); break;
            }
            env->vertical_spawn_used = 1;  // Signal c_reset to skip speed randomization
            env->opponent_ap.prev_vz = 0.0f;
            env->opponent_ap.prev_bank_error = 0.0f;
            return;
        }
    }

    // Use function pointer from STAGES table (replaces 18-case switch)
    if (env->stage < CURRICULUM_COUNT) {
        STAGES[env->stage].spawn(env, player_pos, player_vel);

        // Use per-stage max_steps for advanced stages (8+) where episode length matters
        // Earlier stages use global max_steps from Python config for fast iteration
        // The original "training regression" was from variable episode lengths during early training
        // By stage 8+, agents are stable enough to handle longer episodes
        if (env->stage >= CURRICULUM_SIDE_FAR) {  // Stage 8+
            env->max_steps = STAGES[env->stage].max_steps;
        }
        // else: keep env->max_steps from Python init (already set)
    } else {
        spawn_evasive(env, player_pos, player_vel);  // Fallback for invalid stage
    }

    // Reset autopilot PID state after spawning
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;
}

// Legacy spawn (for curriculum_enabled=0)
void spawn_legacy(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 500),
        player_pos.y + rndf(-100, 100),
        player_pos.z + rndf(-50, 50)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);

    // Handle autopilot: randomize if configured, reset PID state
    if (env->opponent_ap.randomize_on_reset) {
        autopilot_randomize(&env->opponent_ap);
    }
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;
}

#endif // DOGFIGHT_SPAWN_H
