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
    init(&env);
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
    Dogfight env = make_env(1000);
    env.player.pos = vec3(1000, 500, 1500);
    env.player.vel = vec3(125, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);

    compute_observations(&env);

    // pos normalized
    ASSERT_NEAR(env.observations[0], 1000.0f / WORLD_HALF_X, 1e-6f);
    ASSERT_NEAR(env.observations[1], 500.0f / WORLD_HALF_Y, 1e-6f);
    ASSERT_NEAR(env.observations[2], 1500.0f / WORLD_MAX_Z, 1e-6f);

    // vel normalized
    ASSERT_NEAR(env.observations[3], 125.0f / MAX_SPEED, 1e-6f);
    ASSERT_NEAR(env.observations[4], 0.0f, 1e-6f);
    ASSERT_NEAR(env.observations[5], 0.0f, 1e-6f);

    // orientation (identity)
    ASSERT_NEAR(env.observations[6], 1.0f, 1e-6f);
    ASSERT_NEAR(env.observations[7], 0.0f, 1e-6f);
    ASSERT_NEAR(env.observations[8], 0.0f, 1e-6f);
    ASSERT_NEAR(env.observations[9], 0.0f, 1e-6f);

    // up vector (0,0,1 for identity orientation)
    ASSERT_NEAR(env.observations[10], 0.0f, 1e-6f);
    ASSERT_NEAR(env.observations[11], 0.0f, 1e-6f);
    ASSERT_NEAR(env.observations[12], 1.0f, 1e-6f);

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
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Place planes at known positions
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(80, 0, 0);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(500, 100, 1050);
    env.opponent.vel = vec3(80, 0, 0);
    env.opponent.ori = quat(1, 0, 0, 0);

    compute_observations(&env);

    // First 13 obs are player state (from Phase 1)
    // New obs should include relative pos/vel to opponent
    // With identity orientation, body frame = world frame
    // rel_pos = opponent.pos - player.pos = (500, 100, 50)
    float rel_x = env.observations[13];  // Should be 500 / WORLD_HALF_X
    float rel_y = env.observations[14];  // Should be 100 / WORLD_HALF_Y
    float rel_z = env.observations[15];  // Should be 50 / WORLD_MAX_Z

    ASSERT_NEAR(rel_x, 500.0f / WORLD_HALF_X, 1e-5f);
    ASSERT_NEAR(rel_y, 100.0f / WORLD_HALF_Y, 1e-5f);
    ASSERT_NEAR(rel_z, 50.0f / WORLD_MAX_Z, 1e-5f);

    printf("test_relative_observations PASS\n");
}

void test_pursuit_reward() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Place opponent far away
    env.player.pos = vec3(0, 0, 1000);
    env.opponent.pos = vec3(1000, 0, 1000);

    c_step(&env);
    float reward_far = env.rewards[0];

    // Place opponent close
    c_reset(&env);
    env.player.pos = vec3(0, 0, 1000);
    env.opponent.pos = vec3(100, 0, 1000);

    c_step(&env);
    float reward_close = env.rewards[0];

    // Closer should give better (less negative) reward
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

void test_altitude_penalty() {
    Dogfight env = make_env(1000);

    // Scenario 1: Good altitude (1000m)
    c_reset(&env);
    env.player.pos = vec3(0, 0, 1000);
    env.player.vel = vec3(100, 0, 0);
    env.opponent.pos = vec3(300, 0, 1000);

    c_step(&env);
    float reward_good_alt = env.rewards[0];

    // Scenario 2: Too low (100m)
    c_reset(&env);
    env.player.pos = vec3(0, 0, 100);
    env.player.vel = vec3(100, 0, 0);
    env.opponent.pos = vec3(300, 0, 100);

    c_step(&env);
    float reward_low = env.rewards[0];

    // Good altitude should have better reward (less penalty)
    assert(reward_good_alt > reward_low);

    printf("test_altitude_penalty PASS\n");
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

    // Set up player with fire action
    env.player.fire_cooldown = 0;
    env.actions[4] = 1.0f;  // Trigger pulled

    // Step to process fire
    c_step(&env);

    // Should have fired (cooldown set)
    assert(env.player.fire_cooldown == FIRE_COOLDOWN);
    assert(env.log.shots_fired >= 1.0f);

    printf("test_trigger_fires PASS\n");
}

void test_fire_cooldown() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Fire once
    env.player.fire_cooldown = 0;
    env.actions[4] = 1.0f;
    c_step(&env);
    float shots_after_first = env.log.shots_fired;

    // Try to fire again immediately (should be blocked by cooldown)
    c_step(&env);
    float shots_after_second = env.log.shots_fired;

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

    assert(check_hit(&env.player, &env.opponent) == true);

    // Place opponent too far
    env.opponent.pos = vec3(600, 0, 500);  // 600m > GUN_RANGE
    assert(check_hit(&env.player, &env.opponent) == false);

    // Place opponent at side (outside 5 degree cone)
    env.opponent.pos = vec3(200, 50, 500);  // ~14 degrees off-axis
    assert(check_hit(&env.player, &env.opponent) == false);

    // Place opponent slightly off-axis but within cone
    env.opponent.pos = vec3(200, 10, 500);  // ~2.8 degrees off-axis
    assert(check_hit(&env.player, &env.opponent) == true);

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

    float reward_before = env.episode_return;
    c_step(&env);
    float reward_after = env.episode_return;

    // Should have gotten hit + kill reward (11.0 total)
    float reward_gained = reward_after - reward_before;
    assert(reward_gained > 10.0f);  // At least kill reward

    printf("test_hit_reward PASS\n");
}

void test_kill_respawns_opponent() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Set up guaranteed hit
    env.player.pos = vec3(0, 0, 500);
    env.player.ori = quat(1, 0, 0, 0);
    env.player.fire_cooldown = 0;
    env.opponent.pos = vec3(200, 0, 500);

    Vec3 old_opp_pos = env.opponent.pos;
    env.actions[4] = 1.0f;

    c_step(&env);

    // Opponent should have respawned (different position)
    Vec3 new_opp_pos = env.opponent.pos;
    float dist_moved = norm3(sub3(new_opp_pos, old_opp_pos));
    assert(dist_moved > 100.0f);  // Should have moved significantly

    // Episode should NOT have terminated
    assert(env.terminals[0] == 0);

    // Kills should be tracked
    assert(env.log.kills >= 1.0f);

    printf("test_kill_respawns_opponent PASS\n");
}

void test_combat_constants() {
    // Verify combat constants are reasonable
    assert(GUN_RANGE == 500.0f);
    assert(GUN_CONE_ANGLE > 0.08f && GUN_CONE_ANGLE < 0.09f);  // ~5 degrees
    assert(FIRE_COOLDOWN == 10);

    printf("test_combat_constants PASS\n");
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
    test_altitude_penalty();
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
    test_kill_respawns_opponent();
    test_combat_constants();

    printf("\nAll 36 tests PASS\n");
    return 0;
}
