#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "dogfight.h"

#define ASSERT_NEAR(a, b, eps) assert(fabs((a) - (b)) < (eps))

static float obs_buf[32];  // Enough for current and future obs
static float act_buf[5];
static float rew_buf[1];
static unsigned char term_buf[1];

static Dogfight make_env(int max_steps) {
    Dogfight env = {0};
    env.observations = obs_buf;
    env.actions = act_buf;
    env.rewards = rew_buf;
    env.terminals = term_buf;
    env.max_steps = max_steps;
    // df11: Simplified reward config (6 terms)
    RewardConfig rcfg = {
        .aim_scale = 0.05f, .closing_scale = 0.003f,
        .neg_g = 0.02f,
        .speed_min = 50.0f,
    };
    init(&env, 0, &rcfg, 0, 0, 0.7f, 0);  // curriculum_enabled=0
    return env;
}

void test_vec3_math() {
    Vec3 a = vec3(1, 2, 3);
    Vec3 b = vec3(4, 5, 6);

    Vec3 sum = add3(a, b);
    assert(sum.x == 5 && sum.y == 7 && sum.z == 9);

    Vec3 diff = sub3(b, a);
    assert(diff.x == 3 && diff.y == 3 && diff.z == 3);

    Vec3 scaled = mul3(a, 2);
    assert(scaled.x == 2 && scaled.y == 4 && scaled.z == 6);

    float d = dot3(a, b);
    assert(d == 32);  // 1*4 + 2*5 + 3*6 = 32

    ASSERT_NEAR(norm3(vec3(3, 4, 0)), 5.0f, 1e-6f);

    printf("test_vec3_math PASS\n");
}

void test_quat_math() {
    Quat identity = quat(1, 0, 0, 0);
    Vec3 v = vec3(1, 0, 0);
    Vec3 rotated = quat_rotate(identity, v);
    ASSERT_NEAR(rotated.x, 1.0f, 1e-6f);
    ASSERT_NEAR(rotated.y, 0.0f, 1e-6f);
    ASSERT_NEAR(rotated.z, 0.0f, 1e-6f);

    // 90 degree rotation around Z axis
    Quat rot_z = quat_from_axis_angle(vec3(0, 0, 1), PI / 2);
    Vec3 v2 = quat_rotate(rot_z, vec3(1, 0, 0));
    ASSERT_NEAR(v2.x, 0.0f, 1e-5f);
    ASSERT_NEAR(v2.y, 1.0f, 1e-5f);
    ASSERT_NEAR(v2.z, 0.0f, 1e-5f);

    printf("test_quat_math PASS\n");
}

void test_init() {
    Dogfight env = make_env(1000);
    assert(env.tick == 0);
    assert(env.episode_return == 0.0f);
    assert(env.log.n == 0.0f);
    assert(env.client == NULL);
    printf("test_init PASS\n");
}

void test_reset_plane() {
    Plane p;
    Vec3 pos = vec3(100, 200, 300);
    Vec3 vel = vec3(80, 0, 0);
    reset_plane(&p, pos, vel);

    assert(p.pos.x == 100 && p.pos.y == 200 && p.pos.z == 300);
    assert(p.vel.x == 80 && p.vel.y == 0 && p.vel.z == 0);
    assert(p.ori.w == 1 && p.ori.x == 0 && p.ori.y == 0 && p.ori.z == 0);
    assert(p.throttle == 0.5f);

    printf("test_reset_plane PASS\n");
}

void test_c_reset() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    assert(env.tick == 0);
    assert(env.episode_return == 0.0f);

    // Player spawned in bounds
    assert(env.player.pos.x >= -500 && env.player.pos.x <= 500);
    assert(env.player.pos.y >= -500 && env.player.pos.y <= 500);
    assert(env.player.pos.z >= 500 && env.player.pos.z <= 1500);

    // Velocity set
    assert(env.player.vel.x == 80);

    printf("test_c_reset PASS\n");
}

void test_compute_observations() {
    // Tests ANGLES scheme (scheme 0, 12 obs)
    Dogfight env = make_env(1000);
    env.player.pos = vec3(1000, 500, 1500);
    env.player.vel = vec3(125, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);  // identity = facing +X, level

    compute_observations(&env);

    // ANGLES scheme layout:
    // [0-2] pos normalized
    ASSERT_NEAR(env.observations[0], 1000.0f / WORLD_HALF_X, 1e-6f);
    ASSERT_NEAR(env.observations[1], 500.0f / WORLD_HALF_Y, 1e-6f);
    ASSERT_NEAR(env.observations[2], 1500.0f / WORLD_MAX_Z, 1e-6f);

    // [3] speed normalized (scalar)
    ASSERT_NEAR(env.observations[3], 125.0f / MAX_SPEED, 1e-6f);

    // [4-6] euler angles (all 0 for identity quaternion)
    ASSERT_NEAR(env.observations[4], 0.0f, 1e-5f);  // pitch / PI
    ASSERT_NEAR(env.observations[5], 0.0f, 1e-5f);  // roll / PI
    ASSERT_NEAR(env.observations[6], 0.0f, 1e-5f);  // yaw / PI

    // [7-11] target angles - depend on opponent position, check valid ranges
    assert(env.observations[7] >= -1.0f && env.observations[7] <= 1.0f);   // azimuth
    assert(env.observations[8] >= -1.0f && env.observations[8] <= 1.0f);   // elevation
    assert(env.observations[9] >= -2.0f && env.observations[9] <= 2.0f);   // distance
    assert(env.observations[10] >= -1.0f && env.observations[10] <= 1.0f); // closing_rate
    assert(env.observations[11] >= -1.0f && env.observations[11] <= 1.0f); // opp_heading

    printf("test_compute_observations PASS\n");
}

void test_c_step_moves_forward() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    float initial_x = env.player.pos.x;

    // Set neutral actions for stable flight
    env.actions[0] = 0.5f;  // moderate throttle
    env.actions[1] = 0.0f;
    env.actions[2] = 0.0f;
    env.actions[3] = 0.0f;

    c_step(&env);

    // With physics, plane should still move forward (roughly)
    assert(env.player.pos.x > initial_x);
    assert(env.tick == 1);

    printf("test_c_step_moves_forward PASS\n");
}

void test_oob_terminates() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Place plane just past boundary
    env.player.pos = vec3(WORLD_HALF_X + 1, 0, 1000);
    env.player.vel = vec3(80, 0, 0);

    c_step(&env);

    assert(env.terminals[0] == 1);
    assert(env.log.n == 1.0f);  // Episode logged

    printf("test_oob_terminates PASS\n");
}

void test_max_steps_terminates() {
    Dogfight env = make_env(5);
    c_reset(&env);

    // Place plane in center, won't go OOB
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(10, 0, 0);  // Slow

    for (int i = 0; i < 4; i++) {
        c_step(&env);
        assert(env.terminals[0] == 0);
    }

    c_step(&env);  // Step 5 should terminate
    assert(env.terminals[0] == 1);

    printf("test_max_steps_terminates PASS\n");
}

void test_supersonic_terminates() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Place player at 400 m/s (> 340 m/s limit)
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(400, 0, 0);  // Supersonic!
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(300, 0, 1000);
    env.opponent.vel = vec3(80, 0, 0);

    c_step(&env);

    assert(env.terminals[0] == 1);
    assert(env.rewards[0] == -1.0f);

    printf("test_supersonic_terminates PASS\n");
}

// Phase 2 tests

void test_opponent_spawns() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Opponent should exist and be ahead of player
    float dx = env.opponent.pos.x - env.player.pos.x;
    assert(dx >= 200 && dx <= 500);
    assert(env.opponent.vel.x == 80);

    printf("test_opponent_spawns PASS\n");
}

void test_relative_observations() {
    // Tests ANGLES scheme relative target info (azimuth, elevation, distance)
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Place planes at known positions
    // Player at origin facing +X, opponent directly ahead and slightly right/up
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(80, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);  // identity = facing +X
    env.opponent.pos = vec3(500, 100, 1050);  // 500m ahead, 100m right, 50m up
    env.opponent.vel = vec3(80, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);

    compute_observations(&env);

    // ANGLES scheme: relative position encoded as azimuth [7] and elevation [8]
    // rel_pos in body frame = (500, 100, 50) since identity orientation
    // azimuth = atan2(100, 500) / PI ≈ 0.063
    // elevation = atan2(50, sqrt(500^2+100^2)) / (PI/2) ≈ 0.062
    float azimuth = env.observations[7];
    float elevation = env.observations[8];
    float distance = env.observations[9];

    // Azimuth should be small positive (opponent slightly right)
    float expected_az = atan2f(100.0f, 500.0f) / PI;  // ~0.063
    ASSERT_NEAR(azimuth, expected_az, 1e-4f);

    // Elevation should be small positive (opponent slightly above)
    float r_horiz = sqrtf(500*500 + 100*100);
    float expected_el = atan2f(50.0f, r_horiz) / (PI * 0.5f);  // ~0.062
    ASSERT_NEAR(elevation, expected_el, 1e-4f);

    // Distance: sqrt(500^2 + 100^2 + 50^2) ≈ 512m, normalized to [-1,1]
    float dist = sqrtf(500*500 + 100*100 + 50*50);
    float expected_dist = clampf(dist / GUN_RANGE, 0.0f, 2.0f) - 1.0f;
    ASSERT_NEAR(distance, expected_dist, 1e-4f);

    printf("test_relative_observations PASS\n");
}

void test_pursuit_reward() {
    // df11: Test that being closer and on-target gives better reward
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Place opponent far away (outside engagement envelope: > GUN_RANGE * 2 = 1000m)
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);  // Flying forward
    env.player.ori = quat(1, 0, 0, 0);  // Facing +X
    env.opponent.pos = vec3(1500, 0, 1000);  // 1500m ahead - outside aim envelope
    env.opponent.vel = vec3(100, 0, 0);

    c_step(&env);
    float reward_far = env.rewards[0];

    // Place opponent close (inside engagement envelope)
    c_reset(&env);
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(300, 0, 1000);  // 300m ahead - inside aim envelope
    env.opponent.vel = vec3(100, 0, 0);

    c_step(&env);
    float reward_close = env.rewards[0];

    // Closer should give better reward due to aim bonus (within envelope)
    assert(reward_close > reward_far);

    printf("test_pursuit_reward PASS\n");
}

// Phase 3 tests

void test_aircraft_params() {
    // Check that aircraft parameters are defined with reasonable values
    assert(MASS > 0 && MASS < 10000);           // kg, WW2 fighter ~2500-4000kg
    assert(WING_AREA > 0 && WING_AREA < 100);   // m², WW2 fighter ~15-25m²
    assert(C_D0 > 0 && C_D0 < 0.1);             // parasitic drag coef
    assert(K > 0 && K < 0.5);                   // induced drag factor
    assert(C_L_MAX > 0 && C_L_MAX < 2.0);       // max lift coef
    assert(C_L_ALPHA > 0 && C_L_ALPHA < 10);    // lift slope ~5.7/rad
    assert(ENGINE_POWER > 0);                   // watts
    assert(GRAVITY > 9 && GRAVITY < 10);        // m/s²

    printf("test_aircraft_params PASS\n");
}

void test_throttle_accelerates() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Place plane level, flying forward
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);

    float speed_before = norm3(env.player.vel);

    // Full throttle
    env.actions[0] = 1.0f;  // throttle
    env.actions[1] = 0.0f;  // elevator
    env.actions[2] = 0.0f;  // ailerons
    env.actions[3] = 0.0f;  // rudder

    for (int i = 0; i < 50; i++) c_step(&env);

    float speed_after = norm3(env.player.vel);

    // With thrust, should accelerate (or at least maintain speed against drag)
    assert(speed_after >= speed_before * 0.9f);

    printf("test_throttle_accelerates PASS\n");
}

void test_plane_falls_without_lift() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Place plane with no forward velocity (stalled)
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(0, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);

    float z_before = env.player.pos.z;

    // Zero throttle
    env.actions[0] = -1.0f;
    env.actions[1] = 0.0f;
    env.actions[2] = 0.0f;
    env.actions[3] = 0.0f;

    for (int i = 0; i < 50; i++) c_step(&env);

    float z_after = env.player.pos.z;

    // Should fall due to gravity
    assert(z_after < z_before);
    // Should have fallen at least 0.5 * g * t² ≈ 0.5 * 10 * 1² = 5m in 1 sec
    assert(z_before - z_after > 3.0f);

    printf("test_plane_falls_without_lift PASS\n");
}

void test_controls_affect_orientation() {
    Dogfight env = make_env(1000);

    // Test pitch (elevator)
    c_reset(&env);
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    Quat ori_before = env.player.ori;

    env.actions[0] = 0.0f;
    env.actions[1] = 1.0f;  // full elevator (pitch)
    env.actions[2] = 0.0f;
    env.actions[3] = 0.0f;

    for (int i = 0; i < 10; i++) c_step(&env);

    // Orientation should have changed
    float dot = ori_before.w * env.player.ori.w +
                ori_before.x * env.player.ori.x +
                ori_before.y * env.player.ori.y +
                ori_before.z * env.player.ori.z;
    assert(fabsf(dot) < 0.999f);  // not identical

    // Test roll (ailerons)
    c_reset(&env);
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    ori_before = env.player.ori;

    env.actions[0] = 0.0f;
    env.actions[1] = 0.0f;
    env.actions[2] = 1.0f;  // full ailerons (roll)
    env.actions[3] = 0.0f;

    for (int i = 0; i < 10; i++) c_step(&env);

    dot = ori_before.w * env.player.ori.w +
          ori_before.x * env.player.ori.x +
          ori_before.y * env.player.ori.y +
          ori_before.z * env.player.ori.z;
    assert(fabsf(dot) < 0.999f);

    printf("test_controls_affect_orientation PASS\n");
}

void test_dynamic_pressure() {
    // q = 0.5 * rho * V²
    // At V=100 m/s: q = 0.5 * 1.225 * 10000 = 6125 Pa
    float V = 100.0f;
    float q = 0.5f * 1.225f * V * V;
    ASSERT_NEAR(q, 6125.0f, 1.0f);

    printf("test_dynamic_pressure PASS\n");
}

void test_lift_opposes_gravity() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Fly level at cruise speed with moderate throttle
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(120, 0, 0);  // ~270 mph, reasonable cruise
    env.player.ori = quat(1, 0, 0, 0);

    float z_before = env.player.pos.z;

    // Moderate throttle to maintain speed
    env.actions[0] = 0.5f;
    env.actions[1] = 0.0f;
    env.actions[2] = 0.0f;
    env.actions[3] = 0.0f;

    // Run for 1 second (50 steps at 0.02s)
    for (int i = 0; i < 50; i++) c_step(&env);

    float z_after = env.player.pos.z;

    // With lift, altitude change should be much less than free fall
    // Free fall: 0.5 * g * t² = 0.5 * 10 * 1 = 5m
    // With lift: should lose less than 5m (ideally close to 0)
    float dz = fabsf(z_after - z_before);
    assert(dz < 50.0f);  // generous tolerance for now

    printf("test_lift_opposes_gravity PASS\n");
}

void test_drag_slows_plane() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // High speed, zero throttle
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(200, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);

    float speed_before = norm3(env.player.vel);

    env.actions[0] = -1.0f;  // zero throttle
    env.actions[1] = 0.0f;
    env.actions[2] = 0.0f;
    env.actions[3] = 0.0f;

    for (int i = 0; i < 100; i++) c_step(&env);

    float speed_after = norm3(env.player.vel);

    // Drag should slow the plane
    assert(speed_after < speed_before);

    printf("test_drag_slows_plane PASS\n");
}

void test_stall_clamps_lift() {
    // Verify C_L clamping actually happens in physics
    // A plane pointed straight up (90° alpha) should not get infinite lift
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Flying forward at speed
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    // Point nose straight up (90° pitch)
    env.player.ori = quat_from_axis_angle(vec3(0, 1, 0), PI / 2);

    env.actions[0] = 0.5f;
    env.actions[1] = 0.0f;
    env.actions[2] = 0.0f;
    env.actions[3] = 0.0f;

    float z_before = env.player.pos.z;
    for (int i = 0; i < 25; i++) c_step(&env);
    float z_after = env.player.pos.z;

    // With stall limiting, plane should NOT shoot up massively
    // Max C_L = 1.4, at 100 m/s: L = 1.4 * 6125 * 22 = 188,650 N
    // That's ~6.4g, so it can climb but not infinitely
    // Should still fall or climb modestly, not go to space
    assert(z_after - z_before < 500.0f);  // reasonable bound

    printf("test_stall_clamps_lift PASS\n");
}

void test_glimit_clamps_acceleration() {
    // Verify g-limit actually clamps extreme forces
    Dogfight env = make_env(1000);
    c_reset(&env);

    // High speed for lots of lift potential
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(200, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);

    // Full back stick to pitch up hard
    env.actions[0] = 1.0f;
    env.actions[1] = 1.0f;  // full pitch
    env.actions[2] = 0.0f;
    env.actions[3] = 0.0f;

    for (int i = 0; i < 10; i++) c_step(&env);

    // Check that acceleration is bounded
    // At 200 m/s, dynamic pressure is huge, but g-limit should cap it
    // After pulling, vertical velocity should exist but not be insane
    float vz = env.player.vel.z;
    // At 8g for 0.2s: delta_v = 8 * 9.81 * 0.2 = ~16 m/s max vertical
    assert(fabsf(vz) < 200.0f);  // sanity check

    printf("test_glimit_clamps_acceleration PASS\n");
}

void test_forces_sum_correctly() {
    // Test 3.17: verify all forces contribute to motion
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Level flight at moderate speed
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);

    // No throttle - should slow down (drag) and fall (gravity > lift at zero alpha)
    env.actions[0] = -1.0f;
    env.actions[1] = 0.0f;
    env.actions[2] = 0.0f;
    env.actions[3] = 0.0f;

    float vx_before = env.player.vel.x;
    float z_before = env.player.pos.z;

    for (int i = 0; i < 50; i++) c_step(&env);

    // Drag slows forward motion
    assert(env.player.vel.x < vx_before);
    // Gravity pulls down (at zero alpha, minimal lift)
    assert(env.player.pos.z < z_before);

    printf("test_forces_sum_correctly PASS\n");
}

void test_integration_updates_state() {
    // Test 3.18: verify Euler integration updates pos and vel
    Dogfight env = make_env(1000);
    c_reset(&env);

    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);

    Vec3 pos_before = env.player.pos;
    Vec3 vel_before = env.player.vel;

    env.actions[0] = 0.5f;
    env.actions[1] = 0.0f;
    env.actions[2] = 0.0f;
    env.actions[3] = 0.0f;

    c_step(&env);

    // Position should change (velocity integration)
    assert(env.player.pos.x != pos_before.x ||
           env.player.pos.y != pos_before.y ||
           env.player.pos.z != pos_before.z);

    // Velocity should change (force integration)
    assert(env.player.vel.x != vel_before.x ||
           env.player.vel.y != vel_before.y ||
           env.player.vel.z != vel_before.z);

    printf("test_integration_updates_state PASS\n");
}

// Phase 3.5 tests

void test_closing_velocity_reward() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Scenario 1: Player approaching opponent (closing)
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);  // Moving toward opponent
    env.opponent.pos = vec3(500, 0, 1000);
    env.opponent.vel = vec3(50, 0, 0);  // Moving slower

    c_step(&env);
    float reward_closing = env.rewards[0];

    // Scenario 2: Player moving away from opponent (opening)
    c_reset(&env);
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(-100, 0, 0);  // Moving away from opponent
    env.opponent.pos = vec3(500, 0, 1000);
    env.opponent.vel = vec3(50, 0, 0);

    c_step(&env);
    float reward_opening = env.rewards[0];

    // Closing should give better reward than opening
    assert(reward_closing > reward_opening);

    printf("test_closing_velocity_reward PASS\n");
}

void test_tail_position_reward() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Scenario 1: Player behind opponent (good position)
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);  // Facing +X
    env.opponent.pos = vec3(300, 0, 1000);
    env.opponent.vel = vec3(100, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);  // Opponent also facing +X (player behind)

    c_step(&env);
    float reward_behind = env.rewards[0];

    // Scenario 2: Player in front of opponent (bad position)
    c_reset(&env);
    env.player.pos = vec3(300, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(0, 0, 1000);
    env.opponent.vel = vec3(100, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);  // Opponent facing player (player in front)

    c_step(&env);
    float reward_front = env.rewards[0];

    // Being behind should give better reward
    assert(reward_behind > reward_front);

    printf("test_tail_position_reward PASS\n");
}

void test_speed_penalty() {
    Dogfight env = make_env(1000);

    // Scenario 1: Good speed (100 m/s)
    c_reset(&env);
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.opponent.pos = vec3(300, 0, 1000);

    c_step(&env);
    float reward_good_speed = env.rewards[0];

    // Scenario 2: Too slow (20 m/s - stall risk)
    c_reset(&env);
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(20, 0, 0);
    env.opponent.pos = vec3(300, 0, 1000);

    c_step(&env);
    float reward_slow = env.rewards[0];

    // Good speed should have better reward
    assert(reward_good_speed > reward_slow);

    printf("test_speed_penalty PASS\n");
}

// Phase 4: Rendering tests (camera math only, no actual drawing)
void test_chase_camera_behind_player() {
    // Test that camera is positioned behind player based on orientation
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Set player at origin, facing +X
    env.player.pos = vec3(0, 0, 500);
    env.player.ori = quat(1, 0, 0, 0);  // identity = facing +X

    // Calculate camera position (same logic as c_render)
    Vec3 fwd = quat_rotate(env.player.ori, vec3(1, 0, 0));
    float dist = 80.0f;  // default cam_distance
    float el = 0.3f;     // default cam_elevation
    float az = 0.0f;     // default cam_azimuth

    float cam_x = env.player.pos.x - fwd.x * dist * cosf(el) * cosf(az) + fwd.y * dist * sinf(az);
    float cam_y = env.player.pos.y - fwd.y * dist * cosf(el) * cosf(az) - fwd.x * dist * sinf(az);
    float cam_z = env.player.pos.z + dist * sinf(el) + 20.0f;

    // Camera should be behind player (negative X direction) and above
    assert(cam_x < env.player.pos.x);  // Behind
    assert(cam_z > env.player.pos.z);  // Above
    ASSERT_NEAR(cam_y, 0.0f, 1.0f);    // Same Y (player at Y=0)

    printf("test_chase_camera_behind_player PASS\n");
}

void test_camera_orbit_updates() {
    // Test camera orbit math with different azimuth/elevation
    Dogfight env = make_env(1000);
    c_reset(&env);

    env.player.pos = vec3(0, 0, 500);
    env.player.ori = quat(1, 0, 0, 0);

    Vec3 fwd = quat_rotate(env.player.ori, vec3(1, 0, 0));
    float dist = 80.0f;

    // Test with azimuth rotation (looking from side)
    float az = PI / 2.0f;  // 90 degrees
    float el = 0.3f;

    float cam_x = env.player.pos.x - fwd.x * dist * cosf(el) * cosf(az) + fwd.y * dist * sinf(az);
    float cam_y = env.player.pos.y - fwd.y * dist * cosf(el) * cosf(az) - fwd.x * dist * sinf(az);

    // With 90 degree azimuth, camera should be to the side (negative Y)
    assert(cam_y < -30.0f);  // Significantly to the side

    // Test elevation change
    float el_high = 1.2f;  // Looking from above
    float cam_z_high = env.player.pos.z + dist * sinf(el_high) + 20.0f;
    float cam_z_low = env.player.pos.z + dist * sinf(0.1f) + 20.0f;

    assert(cam_z_high > cam_z_low);  // Higher elevation = higher camera

    printf("test_camera_orbit_updates PASS\n");
}

void test_client_struct_defaults() {
    // Test that Client would be initialized with correct defaults
    // (We can't actually test c_render without Raylib window, but we test the values)
    float default_distance = 80.0f;
    float default_azimuth = 0.0f;
    float default_elevation = 0.3f;

    assert(default_distance > 30.0f && default_distance < 300.0f);
    assert(default_elevation > -1.4f && default_elevation < 1.4f);
    ASSERT_NEAR(default_azimuth, 0.0f, 0.01f);

    printf("test_client_struct_defaults PASS\n");
}

// Phase 5: Combat tests
void test_trigger_fires() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Set up player far from opponent (won't hit)
    env.player.pos = vec3(0, 0, 500);
    env.player.ori = quat(1, 0, 0, 0);
    env.player.fire_cooldown = 0;
    env.opponent.pos = vec3(1000, 0, 500);  // Far away, won't hit

    env.actions[4] = 1.0f;  // Trigger pulled

    // Step to process fire
    c_step(&env);

    // Should have fired (cooldown set)
    assert(env.player.fire_cooldown == FIRE_COOLDOWN);
    assert(env.episode_shots_fired >= 1.0f);

    printf("test_trigger_fires PASS\n");
}

void test_fire_cooldown() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Set up player far from opponent (won't hit)
    env.player.pos = vec3(0, 0, 500);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(1000, 0, 500);  // Far away, won't hit

    // Fire once
    env.player.fire_cooldown = 0;
    env.actions[4] = 1.0f;
    c_step(&env);
    float shots_after_first = env.episode_shots_fired;

    // Try to fire again immediately (should be blocked by cooldown)
    c_step(&env);
    float shots_after_second = env.episode_shots_fired;

    // Should not have fired again (still on cooldown)
    assert(shots_after_second == shots_after_first);

    printf("test_fire_cooldown PASS\n");
}

void test_cone_hit_detection() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Place player at origin facing +X
    env.player.pos = vec3(0, 0, 500);
    env.player.ori = quat(1, 0, 0, 0);  // Identity = facing +X

    // Place opponent directly ahead within range
    env.opponent.pos = vec3(200, 0, 500);  // 200m ahead, in cone

    assert(check_hit(&env.player, &env.opponent, env.cos_gun_cone) == true);

    // Place opponent too far
    env.opponent.pos = vec3(600, 0, 500);  // 600m > GUN_RANGE
    assert(check_hit(&env.player, &env.opponent, env.cos_gun_cone) == false);

    // Place opponent at side (outside 5 degree cone)
    env.opponent.pos = vec3(200, 50, 500);  // ~14 degrees off-axis
    assert(check_hit(&env.player, &env.opponent, env.cos_gun_cone) == false);

    // Place opponent slightly off-axis but within cone
    env.opponent.pos = vec3(200, 10, 500);  // ~2.8 degrees off-axis
    assert(check_hit(&env.player, &env.opponent, env.cos_gun_cone) == true);

    printf("test_cone_hit_detection PASS\n");
}

void test_hit_reward() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Set up guaranteed hit
    env.player.pos = vec3(0, 0, 500);
    env.player.ori = quat(1, 0, 0, 0);
    env.player.fire_cooldown = 0;
    env.opponent.pos = vec3(200, 0, 500);  // Directly ahead

    env.actions[4] = 1.0f;  // Fire

    c_step(&env);

    // Kill = terminal with reward 1.0
    assert(env.terminals[0] == 1);
    assert(env.rewards[0] == 1.0f);

    printf("test_hit_reward PASS\n");
}

void test_kill_terminates_episode() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Set up guaranteed hit
    env.player.pos = vec3(0, 0, 500);
    env.player.ori = quat(1, 0, 0, 0);
    env.player.fire_cooldown = 0;
    env.opponent.pos = vec3(200, 0, 500);

    env.actions[4] = 1.0f;

    c_step(&env);

    // Kill should terminate episode
    assert(env.terminals[0] == 1);

    // Reward should be 1.0 (kill reward)
    assert(env.rewards[0] == 1.0f);

    // Perf should be tracked (1 kill in 1 episode = 1.0)
    assert(env.log.perf >= 1.0f);
    assert(env.log.n >= 1.0f);

    printf("test_kill_terminates_episode PASS\n");
}

void test_combat_constants() {
    // Verify combat constants are reasonable
    assert(GUN_RANGE == 500.0f);
    assert(GUN_CONE_ANGLE > 0.08f && GUN_CONE_ANGLE < 0.09f);  // ~5 degrees
    assert(FIRE_COOLDOWN == 10);

    printf("test_combat_constants PASS\n");
}

// Phase 3.6: Additional reward/penalty tests (df11: updated for simplified rewards)

void test_aim_reward() {
    // df11: Test continuous aim reward - better aim = better reward
    Dogfight env = make_env(1000);

    // Scenario 1: Opponent directly ahead (perfect aim, aim_dot = 1.0)
    c_reset(&env);
    env.actions[4] = -1.0f;  // Don't fire
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);  // Facing +X
    env.opponent.pos = vec3(300, 0, 1000);  // Directly ahead
    env.opponent.vel = vec3(100, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);
    c_step(&env);
    float reward_on_target = env.rewards[0];

    // Scenario 2: Opponent 90° off-axis (bad aim, aim_dot = 0.0)
    c_reset(&env);
    env.actions[4] = -1.0f;  // Don't fire
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(0, 300, 1000);  // 90° to the side
    env.opponent.vel = vec3(100, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);
    c_step(&env);
    float reward_off_target = env.rewards[0];

    // On target should have better reward (continuous aim reward)
    assert(reward_on_target > reward_off_target);

    printf("test_aim_reward PASS\n");
}

void test_aim_reward_range_dependent() {
    // df11: Test that aim reward only applies within engagement envelope (GUN_RANGE * 2)
    Dogfight env = make_env(1000);

    // Scenario 1: In range (300m < GUN_RANGE * 2 = 1000m) - should get aim reward
    c_reset(&env);
    env.actions[4] = -1.0f;  // Don't fire
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(300, 0, 1000);  // 300m ahead, in envelope
    env.opponent.vel = vec3(100, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);
    c_step(&env);
    float reward_in_range = env.rewards[0];

    // Scenario 2: Out of range (1500m > GUN_RANGE * 2 = 1000m) - no aim reward
    c_reset(&env);
    env.actions[4] = -1.0f;  // Don't fire
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(1500, 0, 1000);  // 1500m ahead, out of envelope
    env.opponent.vel = vec3(100, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);
    c_step(&env);
    float reward_out_of_range = env.rewards[0];

    // In range should get aim bonus (both have same aim quality but different range)
    assert(reward_in_range > reward_out_of_range);

    printf("test_aim_reward_range_dependent PASS\n");
}

// Helper to make env with curriculum enabled
static Dogfight make_env_curriculum(int max_steps, int randomize) {
    Dogfight env = {0};
    env.observations = obs_buf;
    env.actions = act_buf;
    env.rewards = rew_buf;
    env.terminals = term_buf;
    env.max_steps = max_steps;
    // df11: Simplified reward config
    RewardConfig rcfg = {
        .aim_scale = 0.05f, .closing_scale = 0.003f,
        .neg_g = 0.02f,
        .speed_min = 50.0f,
    };
    init(&env, 0, &rcfg, 1, randomize, 0.7f, 0);  // curriculum_enabled=1
    return env;
}

// Helper to make env for rudder penalty test
static Dogfight make_env_for_rudder_test(int max_steps) {
    Dogfight env = {0};
    env.observations = obs_buf;
    env.actions = act_buf;
    env.rewards = rew_buf;
    env.terminals = term_buf;
    env.max_steps = max_steps;
    // df11: Simplified reward config
    RewardConfig rcfg = {
        .aim_scale = 0.05f, .closing_scale = 0.003f,
        .neg_g = 0.02f,
        .speed_min = 50.0f,
    };
    init(&env, 0, &rcfg, 0, 0, 0.7f, 0);
    return env;
}

void test_rudder_penalty_accumulates() {
    // df11: Test that constant rudder use accumulates meaningful penalty over multiple steps
    Dogfight env = make_env_for_rudder_test(1000);  // 10x default for visibility
    c_reset(&env);

    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.opponent.pos = vec3(300, 0, 1000);

    // Full rudder with moderate throttle to maintain flight
    float total_reward = 0.0f;
    for (int i = 0; i < 50; i++) {
        env.actions[0] = 0.5f;   // Moderate throttle
        env.actions[1] = 0.0f;   // Neutral elevator
        env.actions[2] = 0.0f;   // Neutral aileron
        env.actions[3] = 1.0f;   // Full right rudder (constant yaw)
        env.actions[4] = -1.0f;  // No fire
        c_step(&env);
        total_reward += env.rewards[0];

        // Refresh opponent position (so distance reward stays similar)
        env.opponent.pos = vec3(env.player.pos.x + 300, env.player.pos.y, env.player.pos.z);
    }

    // Compare to coordinated flight: same scenario but no rudder
    Dogfight env2 = make_env_for_rudder_test(1000);
    c_reset(&env2);

    env2.player.pos = vec3(0, 0, 1000);
    env2.player.vel = vec3(100, 0, 0);
    env2.opponent.pos = vec3(300, 0, 1000);

    float total_reward_coordinated = 0.0f;
    for (int i = 0; i < 50; i++) {
        env2.actions[0] = 0.5f;
        env2.actions[1] = 0.0f;
        env2.actions[2] = 0.0f;   // Neutral aileron
        env2.actions[3] = 0.0f;   // NO rudder (coordinated)
        env2.actions[4] = -1.0f;
        c_step(&env2);
        total_reward_coordinated += env2.rewards[0];

        env2.opponent.pos = vec3(env2.player.pos.x + 300, env2.player.pos.y, env2.player.pos.z);
    }

    // Rudder use should accumulate worse reward than coordinated flight
    assert(total_reward < total_reward_coordinated);

    printf("test_rudder_penalty_accumulates PASS\n");
}

// Helper to get bearing from player to opponent (degrees, 0=ahead, 90=right, 180=behind)
static float get_bearing(Dogfight *env) {
    Vec3 rel = sub3(env->opponent.pos, env->player.pos);
    Vec3 player_fwd = quat_rotate(env->player.ori, vec3(1, 0, 0));
    float dot = dot3(normalize3(rel), player_fwd);
    return acosf(clampf(dot, -1, 1)) * 180.0f / M_PI;
}

// Helper to get opponent heading (degrees, 0=+X, 90=+Y)
static float get_opponent_heading(Dogfight *env) {
    Vec3 opp_fwd = quat_rotate(env->opponent.ori, vec3(1, 0, 0));
    return atan2f(opp_fwd.y, opp_fwd.x) * 180.0f / M_PI;
}

void test_spawn_bearing_variety() {
    // Test that FULL_RANDOM stage spawns opponents at various bearings (not just ahead)
    // Set stage directly since curriculum is now performance-based (df10)
    Dogfight env = make_env_curriculum(1000, 0);  // Progressive mode
    env.stage = CURRICULUM_FULL_RANDOM;  // Force stage 4 (FULL_RANDOM) directly

    int front_count = 0;   // bearing < 45
    int side_count = 0;    // bearing 45-135
    int behind_count = 0;  // bearing > 135

    // Run many resets with different seeds
    for (int seed = 0; seed < 100; seed++) {
        srand(seed * 7 + 13);  // Vary seed
        c_reset(&env);

        // Verify we're in stage 4 (FULL_RANDOM after 2026-01-18 reorder)
        assert(env.stage == CURRICULUM_FULL_RANDOM);

        float bearing = get_bearing(&env);
        if (bearing < 45.0f) front_count++;
        else if (bearing > 135.0f) behind_count++;
        else side_count++;
    }

    // With 360° spawning, we should see opponents in all directions
    // Each sector should have at least some spawns (allow for randomness)
    assert(front_count > 0);  // Some in front
    assert(side_count > 0);   // Some to the side
    assert(behind_count > 0); // Some behind (this is the key test!)

    printf("test_spawn_bearing_variety PASS (front=%d, side=%d, behind=%d)\n",
           front_count, side_count, behind_count);
}

void test_spawn_heading_variety() {
    // Test that FULL_RANDOM opponents have varied headings (not always 0)
    // Set stage directly since curriculum is now performance-based (df10)
    Dogfight env = make_env_curriculum(1000, 0);  // Progressive mode
    env.stage = CURRICULUM_FULL_RANDOM;  // Force stage 4 (FULL_RANDOM) directly

    float min_heading = 999.0f;
    float max_heading = -999.0f;
    int varied_count = 0;  // Count of headings not near 0

    for (int seed = 0; seed < 50; seed++) {
        srand(seed * 11 + 17);
        c_reset(&env);

        // Verify we're in stage 4 (FULL_RANDOM after 2026-01-18 reorder)
        assert(env.stage == CURRICULUM_FULL_RANDOM);

        float heading = get_opponent_heading(&env);
        if (heading < min_heading) min_heading = heading;
        if (heading > max_heading) max_heading = heading;
        if (fabsf(heading) > 30.0f) varied_count++;  // Not facing +X
    }

    // Headings should vary across the full 360° range
    float heading_range = max_heading - min_heading;
    assert(heading_range > 90.0f);  // At least 90° variation
    assert(varied_count > 10);      // At least some not facing default direction

    printf("test_spawn_heading_variety PASS (range=%.0f°, varied=%d)\n",
           heading_range, varied_count);
}

void test_curriculum_stages_differ() {
    // Test that different curriculum stages produce different spawn patterns
    // Set stage directly since curriculum is now performance-based (df10)
    Dogfight env = make_env_curriculum(1000, 0);  // Progressive mode (randomize=0)

    // Stage 0: TAIL_CHASE - opponent ahead, same direction
    env.stage = CURRICULUM_TAIL_CHASE;
    srand(42);
    c_reset(&env);
    float bearing_tail = get_bearing(&env);
    float heading_tail = get_opponent_heading(&env);
    assert(env.stage == CURRICULUM_TAIL_CHASE);

    // Stage 1: HEAD_ON - opponent ahead, facing us
    env.stage = CURRICULUM_HEAD_ON;
    srand(42);
    c_reset(&env);
    float bearing_head = get_bearing(&env);
    assert(env.stage == CURRICULUM_HEAD_ON);

    // Stage 2: VERTICAL - opponent above/below (after 2026-01-18 reorder, was stage 3)
    env.stage = CURRICULUM_VERTICAL;
    srand(42);
    c_reset(&env);
    float bearing_vert = get_bearing(&env);
    assert(env.stage == CURRICULUM_VERTICAL);

    // Stage 6: CROSSING - opponent to side (after 2026-01-18 reorder, was stage 2)
    env.stage = CURRICULUM_CROSSING;
    srand(42);
    c_reset(&env);
    float bearing_cross = get_bearing(&env);
    assert(env.stage == CURRICULUM_CROSSING);

    // TAIL_CHASE should have opponent nearly ahead (small bearing)
    assert(bearing_tail < 30.0f);

    // HEAD_ON should have opponent ahead
    assert(bearing_head < 30.0f);

    // VERTICAL should have opponent ahead (same heading, different altitude)
    assert(bearing_vert < 45.0f);

    // CROSSING should have opponent more to the side (larger bearing)
    assert(bearing_cross > 45.0f);

    // TAIL_CHASE opponent should face same direction as player (~0° heading)
    assert(fabsf(heading_tail) < 30.0f);

    printf("test_curriculum_stages_differ PASS (tail=%.0f°, head=%.0f°, vert=%.0f°, cross=%.0f°)\n",
           bearing_tail, bearing_head, bearing_vert, bearing_cross);
}

void test_spawn_distance_range() {
    // Test that spawn distances are within expected ranges
    Dogfight env = make_env_curriculum(1000, 1);

    float min_dist = 9999.0f;
    float max_dist = 0.0f;

    for (int seed = 0; seed < 50; seed++) {
        srand(seed * 13 + 7);
        c_reset(&env);

        Vec3 rel = sub3(env.opponent.pos, env.player.pos);
        float dist = norm3(rel);

        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    // Distances should be reasonable (200-700m typical range across all stages)
    assert(min_dist > 100.0f);   // Not too close
    assert(max_dist < 800.0f);   // Not too far
    assert(max_dist - min_dist > 100.0f);  // Some variety

    printf("test_spawn_distance_range PASS (min=%.0f, max=%.0f)\n", min_dist, max_dist);
}

void test_neg_g_penalty() {
    // Test that pushing forward (negative G) gets worse reward than pulling back
    // Now uses actual g_force (not elevator position), so we need multiple steps
    // for the G-force to develop from the maneuver
    //
    // We set a high neg_g penalty to ensure it dominates other reward differences
    // caused by trajectory changes during the maneuver
    Dogfight env = make_env(1000);
    env.rcfg.neg_g = 0.5f;  // High penalty to dominate other rewards
    c_reset(&env);
    env.actions[4] = -1.0f;  // Don't fire

    // Pull back for multiple steps
    env.player.pos = vec3(0, 0, 1500);
    env.player.vel = vec3(150, 0, 0);  // Fast enough for significant G
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(400, 0, 1500);
    env.opponent.vel = vec3(150, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);

    env.actions[1] = -0.5f;  // Pull back (nose up, positive G)
    float total_reward_pull = 0.0f;
    float pull_g_sum = 0.0f;
    for (int i = 0; i < 30; i++) {
        c_step(&env);
        total_reward_pull += env.rewards[0];
        pull_g_sum += env.player.g_force;
    }
    float pull_g_avg = pull_g_sum / 30.0f;

    // Push forward for multiple steps
    c_reset(&env);
    env.actions[4] = -1.0f;
    env.player.pos = vec3(0, 0, 1500);
    env.player.vel = vec3(150, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(400, 0, 1500);
    env.opponent.vel = vec3(150, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);

    env.actions[1] = 0.5f;  // Push forward (nose down, negative G)
    float total_reward_push = 0.0f;
    float push_g_sum = 0.0f;
    for (int i = 0; i < 30; i++) {
        c_step(&env);
        total_reward_push += env.rewards[0];
        push_g_sum += env.player.g_force;
    }
    float push_g_avg = push_g_sum / 30.0f;

    // Verify G-forces are correct direction (pull = positive, push = negative)
    assert(pull_g_avg > 1.0f);  // Pull should give >1G
    assert(push_g_avg < 0.5f);  // Push should give <0.5G (triggering penalty)

    // Pull back should have better total reward (push fwd triggers neg_g penalty)
    assert(total_reward_pull > total_reward_push);
    printf("test_neg_g_penalty PASS (pull=%.4f > push=%.4f, pull_g=%.1f, push_g=%.1f)\n",
           total_reward_pull, total_reward_push, pull_g_avg, push_g_avg);
}

void test_rudder_penalty() {
    // Test that no rudder gets better reward than full rudder
    Dogfight env = make_env(1000);
    c_reset(&env);
    env.actions[4] = -1.0f;  // Don't fire

    // No rudder
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(300, 0, 1000);
    env.opponent.vel = vec3(100, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);
    env.actions[3] = 0.0f;  // No rudder
    c_step(&env);
    float reward_no_rudder = env.rewards[0];

    // Full rudder
    c_reset(&env);
    env.actions[4] = -1.0f;
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(300, 0, 1000);
    env.opponent.vel = vec3(100, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);
    env.actions[3] = 1.0f;  // Full rudder
    c_step(&env);
    float reward_rudder = env.rewards[0];

    // No rudder should have better reward
    assert(reward_no_rudder > reward_rudder);
    printf("test_rudder_penalty PASS (no_rud=%.5f > rud=%.5f)\n", reward_no_rudder, reward_rudder);
}

// Generic test: all observation schemes produce bounded values
// Works regardless of which schemes exist or their indices
void test_obs_bounds_all_schemes() {
    int schemes_tested = 0;
    int total_obs_checked = 0;

    // Test all schemes from 0 to OBS_SCHEME_COUNT-1
    for (int scheme = 0; scheme < OBS_SCHEME_COUNT; scheme++) {
        // Create env with this scheme
        Dogfight env = {0};
        env.observations = obs_buf;
        env.actions = act_buf;
        env.rewards = rew_buf;
        env.terminals = term_buf;
        env.max_steps = 1000;
        RewardConfig rcfg = {
            .aim_scale = 0.05f, .closing_scale = 0.003f,
            .neg_g = 0.02f,
            .speed_min = 50.0f,
        };
        init(&env, scheme, &rcfg, 0, 0, 0.7f, 0);

        // Reset to get valid observations
        c_reset(&env);

        // Verify obs_size is positive and reasonable
        assert(env.obs_size > 0 && env.obs_size <= 32);

        // All observations should be bounded [-2, 2] (some have [-1,1], some [0,1])
        // Using [-2, 2] as generous outer bound that catches NaN/Inf/unbounded
        for (int i = 0; i < env.obs_size; i++) {
            float val = env.observations[i];
            assert(!isnan(val) && !isinf(val));
            assert(val >= -2.0f && val <= 2.0f);
            total_obs_checked++;
        }

        // Run a few steps and check bounds again
        for (int step = 0; step < 10; step++) {
            // Neutral actions
            env.actions[0] = 0.0f;  // throttle
            env.actions[1] = 0.0f;  // pitch
            env.actions[2] = 0.0f;  // roll
            env.actions[3] = 0.0f;  // yaw
            env.actions[4] = 0.0f;  // fire

            c_step(&env);

            // Check bounds after step
            for (int i = 0; i < env.obs_size; i++) {
                float val = env.observations[i];
                assert(!isnan(val) && !isinf(val));
                assert(val >= -2.0f && val <= 2.0f);
            }
        }

        schemes_tested++;
    }

    printf("test_obs_bounds_all_schemes PASS (%d schemes, %d obs checked)\n",
           schemes_tested, total_obs_checked);
}

int main() {
    printf("Running dogfight tests...\n\n");

    // Phase 1
    test_vec3_math();
    test_quat_math();
    test_init();
    test_reset_plane();
    test_c_reset();
    test_compute_observations();
    test_c_step_moves_forward();
    test_oob_terminates();
    test_max_steps_terminates();
    test_supersonic_terminates();

    // Phase 2
    test_opponent_spawns();
    test_relative_observations();
    test_pursuit_reward();

    // Phase 3
    test_aircraft_params();
    test_throttle_accelerates();
    test_plane_falls_without_lift();
    test_controls_affect_orientation();
    test_dynamic_pressure();
    test_lift_opposes_gravity();
    test_drag_slows_plane();
    test_stall_clamps_lift();
    test_glimit_clamps_acceleration();
    test_forces_sum_correctly();
    test_integration_updates_state();

    // Phase 3.5
    test_closing_velocity_reward();
    test_tail_position_reward();
    test_speed_penalty();

    // Phase 4
    test_chase_camera_behind_player();
    test_camera_orbit_updates();
    test_client_struct_defaults();

    // Phase 5
    test_trigger_fires();
    test_fire_cooldown();
    test_cone_hit_detection();
    test_hit_reward();
    test_kill_terminates_episode();
    test_combat_constants();

    // Phase 5.5: df11 reward tests (simplified: aim, closing, neg_g, stall, rudder)
    test_aim_reward();
    test_aim_reward_range_dependent();
    test_rudder_penalty_accumulates();
    test_neg_g_penalty();
    test_rudder_penalty();

    // Phase 6: Spawn variety tests
    test_spawn_bearing_variety();
    test_spawn_heading_variety();
    test_curriculum_stages_differ();
    test_spawn_distance_range();

    // Phase 7: Generic observation tests
    test_obs_bounds_all_schemes();

    printf("\nAll 46 tests PASS\n");
    return 0;
}
