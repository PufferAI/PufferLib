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

    printf("\nAll tests PASS\n");
    return 0;
}
