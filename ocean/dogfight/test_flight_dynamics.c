/**
 * test_flight_dynamics.c - Flight dynamics test with parameter sweep
 *
 * Tests high-speed oscillation fix and sweeps control/damping parameters
 * to find optimal values automatically.
 *
 * Usage:
 *   ./test_flight_dynamics                    # Run oscillation test (current params)
 *   ./test_flight_dynamics --sweep            # Full parameter sweep (CSV output)
 *   ./test_flight_dynamics --sweep-fine       # Fine sweep around best params
 *   ./test_flight_dynamics --analyze          # Analyze and report best params
 *
 * Compile: gcc -O2 -I. test_flight_dynamics.c -o test_flight_dynamics -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "flightlib.h"

#define TEST_DURATION 3.0f
#define DT 0.01667f           // 60 Hz
#define SETTLE_TIME 0.5f
#define MAX_PITCH_RATE_STD 0.2f  // Target: < 0.2 rad/s at ALL speeds

typedef struct {
    float speed;
    float elevator;
    float pitch_rate_std;
    float pitch_std;
    float speed_final;
} TestResult;

typedef struct {
    FlightParams params;
    float total_score;      // Sum of pitch_rate_std across all speeds
    float max_pitch_rate;   // Worst case pitch rate std
    int all_passed;         // 1 if all speeds pass threshold
} SweepResult;

// Result for per-speed optimal scale discovery
typedef struct {
    float speed;
    float optimal_scale;
    float pitch_rate_std;
    float responsiveness;   // How much pitch rate is achieved (higher = more responsive)
} OptimalScaleResult;

// Compute statistics
static void compute_stats(float* data, int n, float* out_mean, float* out_std) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += data[i];
    float mean = sum / n;
    if (out_mean) *out_mean = mean;
    if (out_std) {
        float var_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = data[i] - mean;
            var_sum += diff * diff;
        }
        *out_std = sqrtf(var_sum / n);
    }
}

// Run oscillation test with runtime parameters
static TestResult run_test_with_params(float speed, float elevator, FlightParams* params) {
    TestResult result = {0};
    result.speed = speed;
    result.elevator = elevator;

    Plane plane;
    Vec3 pos = vec3(0, 0, 1000);
    Vec3 vel = vec3(speed, 0, 0);
    reset_plane(&plane, pos, vel);

    int max_samples = (int)((TEST_DURATION - SETTLE_TIME) / DT) + 1;
    float* pitch_rates = malloc(max_samples * sizeof(float));
    float* pitches = malloc(max_samples * sizeof(float));
    int sample_count = 0;

    int steps = (int)(TEST_DURATION / DT);
    int settle_steps = (int)(SETTLE_TIME / DT);

    for (int step = 0; step < steps; step++) {
        float actions[5] = {1.0f, elevator, 0.0f, 0.0f, -1.0f};
        step_plane_with_params(&plane, actions, DT, params);

        if (step >= settle_steps && sample_count < max_samples) {
            pitch_rates[sample_count] = plane.omega.y;
            Vec3 fwd = quat_rotate(plane.ori, vec3(1, 0, 0));
            pitches[sample_count] = asinf(fminf(fmaxf(fwd.z, -1.0f), 1.0f));
            sample_count++;
        }
    }

    compute_stats(pitch_rates, sample_count, NULL, &result.pitch_rate_std);
    compute_stats(pitches, sample_count, NULL, &result.pitch_std);
    result.speed_final = norm3(plane.vel);

    free(pitch_rates);
    free(pitches);
    return result;
}

// Run test with compile-time defaults (for backward compatibility)
static TestResult run_test(float speed, float elevator) {
    FlightParams params = default_flight_params();
    return run_test_with_params(speed, elevator, &params);
}

// Run oscillation test with a fixed control_scale (bypasses the V_ref/slope formula)
// This is used for per-speed optimal scale discovery
static TestResult run_test_with_fixed_scale(float speed, float elevator, float fixed_scale) {
    TestResult result = {0};
    result.speed = speed;
    result.elevator = elevator;

    Plane plane;
    Vec3 pos = vec3(0, 0, 1000);
    Vec3 vel = vec3(speed, 0, 0);
    reset_plane(&plane, pos, vel);

    int max_samples = (int)((TEST_DURATION - SETTLE_TIME) / DT) + 1;
    float* pitch_rates = malloc(max_samples * sizeof(float));
    float* pitches = malloc(max_samples * sizeof(float));
    int sample_count = 0;

    int steps = (int)(TEST_DURATION / DT);
    int settle_steps = (int)(SETTLE_TIME / DT);

    for (int step = 0; step < steps; step++) {
        float actions[5] = {1.0f, elevator, 0.0f, 0.0f, -1.0f};

        // Custom physics step with fixed control_scale
        // We manually compute the physics here to bypass the V_ref formula
        plane.prev_vel = plane.vel;

        float clamped_actions[4];
        for (int i = 0; i < 4; i++) {
            clamped_actions[i] = clampf(actions[i], -1.0f, 1.0f);
        }

        // Inline RK4 step with fixed control scale
        // We'll use a simplified approach: modify the actions to achieve the fixed scale
        // The control_scale affects delta_e = action * MAX_DEFLECTION * control_scale
        // So we scale the elevator action to achieve the same effect
        float scaled_actions[4];
        scaled_actions[0] = clamped_actions[0];  // throttle unchanged
        scaled_actions[1] = clamped_actions[1] * fixed_scale;  // elevator scaled
        scaled_actions[2] = clamped_actions[2] * fixed_scale;  // aileron scaled
        scaled_actions[3] = clamped_actions[3] * fixed_scale;  // rudder scaled

        // Use a params struct with control_scale_min = 1.0 to disable the formula,
        // but this won't work cleanly. Instead, set v_ref very high so scale = 1.0
        // and pre-scale the actions.
        FlightParams fixed_params = {
            .control_v_ref = 10000.0f,  // Very high, so scale = 1.0 always
            .control_scale_slope = 0.0f,
            .control_scale_min = 1.0f,
            .damping_scale_slope = 0.0f
        };
        rk4_step_with_params(&plane, scaled_actions, DT, &fixed_params);

        plane.throttle = (clamped_actions[0] + 1.0f) * 0.5f;
        plane.omega.x = clampf(plane.omega.x, -5.0f, 5.0f);
        plane.omega.y = clampf(plane.omega.y, -5.0f, 5.0f);
        plane.omega.z = clampf(plane.omega.z, -2.0f, 2.0f);

        // G-force calculation
        Vec3 dv = sub3(plane.vel, plane.prev_vel);
        Vec3 accel = mul3(dv, 1.0f / DT);
        Vec3 body_up = quat_rotate(plane.ori, vec3(0, 0, 1));
        float accel_up = dot3(accel, body_up);
        plane.g_force = accel_up * INV_GRAVITY + 1.0f;

        // G-limit enforcement
        if (plane.g_force > G_LIMIT_POS) {
            float excess_g = plane.g_force - G_LIMIT_POS;
            float excess_accel = excess_g * GRAVITY;
            Vec3 correction = mul3(body_up, excess_accel * DT);
            Vec3 vel_norm = normalize3(plane.vel);
            float correction_along_vel = dot3(correction, vel_norm);
            Vec3 correction_perp = sub3(correction, mul3(vel_norm, correction_along_vel));
            plane.vel = sub3(plane.vel, correction_perp);
            plane.g_force = G_LIMIT_POS;
        } else if (plane.g_force < -G_LIMIT_NEG) {
            float deficit_g = -G_LIMIT_NEG - plane.g_force;
            float deficit_accel = deficit_g * GRAVITY;
            Vec3 correction = mul3(body_up, deficit_accel * DT);
            Vec3 vel_norm = normalize3(plane.vel);
            float correction_along_vel = dot3(correction, vel_norm);
            Vec3 correction_perp = sub3(correction, mul3(vel_norm, correction_along_vel));
            plane.vel = add3(plane.vel, correction_perp);
            plane.g_force = -G_LIMIT_NEG;
        }

        if (step >= settle_steps && sample_count < max_samples) {
            pitch_rates[sample_count] = plane.omega.y;
            Vec3 fwd = quat_rotate(plane.ori, vec3(1, 0, 0));
            pitches[sample_count] = asinf(fminf(fmaxf(fwd.z, -1.0f), 1.0f));
            sample_count++;
        }
    }

    compute_stats(pitch_rates, sample_count, NULL, &result.pitch_rate_std);
    compute_stats(pitches, sample_count, NULL, &result.pitch_std);
    result.speed_final = norm3(plane.vel);

    free(pitch_rates);
    free(pitches);
    return result;
}

// Score a parameter set across all test conditions
static SweepResult score_params(FlightParams* params) {
    SweepResult result = {0};
    result.params = *params;
    result.all_passed = 1;
    result.max_pitch_rate = 0.0f;

    float speeds[] = {80, 100, 120, 140, 160, 180};
    float elevators[] = {-0.3f, -0.5f, -0.7f};
    int num_speeds = 6;
    int num_elevs = 3;

    for (int e = 0; e < num_elevs; e++) {
        for (int s = 0; s < num_speeds; s++) {
            TestResult r = run_test_with_params(speeds[s], elevators[e], params);
            result.total_score += r.pitch_rate_std;
            if (r.pitch_rate_std > result.max_pitch_rate) {
                result.max_pitch_rate = r.pitch_rate_std;
            }
            if (r.pitch_rate_std >= MAX_PITCH_RATE_STD) {
                result.all_passed = 0;
            }
        }
    }

    return result;
}

// Run the main oscillation test suite (uses compile-time defaults)
static int run_oscillation_test(void) {
    printf("=============================================================\n");
    printf("  HIGH-SPEED OSCILLATION TEST\n");
    printf("=============================================================\n\n");

    printf("Control scaling: V_ref=%.0f, slope=%.3f, min=%.2f\n",
           CONTROL_V_REF, CONTROL_SCALE_SLOPE, CONTROL_SCALE_MIN);
    printf("Pass threshold: pitch_rate_std < %.2f rad/s\n\n", MAX_PITCH_RATE_STD);

    float speeds[] = {80, 100, 120, 140, 160, 180};
    float elevators[] = {-0.3f, -0.5f, -0.7f};
    int num_speeds = 6;
    int num_elevs = 3;

    int all_passed = 1;

    for (int e = 0; e < num_elevs; e++) {
        printf("--- Elevator: %.1f ---\n", elevators[e]);
        printf("%-8s  %-10s  %-8s  %s\n", "Speed", "RateStd", "Scale", "Status");
        printf("--------  ----------  --------  ------\n");

        for (int s = 0; s < num_speeds; s++) {
            TestResult r = run_test(speeds[s], elevators[e]);

            float scale = 1.0f - fmaxf(0.0f, speeds[s] - CONTROL_V_REF) * CONTROL_SCALE_SLOPE;
            scale = fmaxf(scale, CONTROL_SCALE_MIN);

            int passed = (r.pitch_rate_std < MAX_PITCH_RATE_STD);
            if (!passed) all_passed = 0;

            printf("%6.0f    %8.4f    %6.2f    %s\n",
                   speeds[s], r.pitch_rate_std, scale, passed ? "PASS" : "FAIL");
        }
        printf("\n");
    }

    printf("RESULT: %s\n", all_passed ? "ALL PASSED" : "SOME FAILED");
    return all_passed ? 0 : 1;
}

// Full coarse parameter sweep
static void run_full_sweep(void) {
    // Output CSV header
    printf("v_ref,ctrl_slope,ctrl_min,damp_slope,speed,elev,pitch_rate_std,ctrl_scale\n");

    float v_refs[] = {70, 80, 90};
    float ctrl_slopes[] = {0.004, 0.006, 0.008, 0.010, 0.012};
    float ctrl_mins[] = {0.25, 0.35, 0.45};
    float damp_slopes[] = {0.0, 0.005, 0.010};
    float speeds[] = {80, 120, 160, 180};
    float elevators[] = {-0.5f};

    int nv = sizeof(v_refs) / sizeof(v_refs[0]);
    int nc = sizeof(ctrl_slopes) / sizeof(ctrl_slopes[0]);
    int nm = sizeof(ctrl_mins) / sizeof(ctrl_mins[0]);
    int nd = sizeof(damp_slopes) / sizeof(damp_slopes[0]);
    int ns = sizeof(speeds) / sizeof(speeds[0]);
    int ne = sizeof(elevators) / sizeof(elevators[0]);

    for (int iv = 0; iv < nv; iv++) {
        for (int ic = 0; ic < nc; ic++) {
            for (int im = 0; im < nm; im++) {
                for (int id = 0; id < nd; id++) {
                    FlightParams params = {
                        .control_v_ref = v_refs[iv],
                        .control_scale_slope = ctrl_slopes[ic],
                        .control_scale_min = ctrl_mins[im],
                        .damping_scale_slope = damp_slopes[id]
                    };

                    for (int ie = 0; ie < ne; ie++) {
                        for (int is = 0; is < ns; is++) {
                            TestResult r = run_test_with_params(speeds[is], elevators[ie], &params);

                            float ctrl_scale = 1.0f - fmaxf(0.0f, speeds[is] - params.control_v_ref) * params.control_scale_slope;
                            ctrl_scale = fmaxf(ctrl_scale, params.control_scale_min);

                            printf("%.0f,%.4f,%.2f,%.4f,%.0f,%.1f,%.4f,%.2f\n",
                                   params.control_v_ref,
                                   params.control_scale_slope,
                                   params.control_scale_min,
                                   params.damping_scale_slope,
                                   speeds[is],
                                   elevators[ie],
                                   r.pitch_rate_std,
                                   ctrl_scale);
                        }
                    }
                }
            }
        }
    }
}

// Fine sweep around known good values
static void run_fine_sweep(void) {
    printf("v_ref,ctrl_slope,ctrl_min,damp_slope,total_score,max_rate,all_passed\n");

    // Fine grid around current values
    float v_refs[] = {75, 78, 80, 82, 85};
    float ctrl_slopes[] = {0.005, 0.006, 0.007, 0.008, 0.009};
    float ctrl_mins[] = {0.30, 0.33, 0.35, 0.37, 0.40};
    float damp_slopes[] = {0.0, 0.002, 0.004, 0.006};

    int nv = sizeof(v_refs) / sizeof(v_refs[0]);
    int nc = sizeof(ctrl_slopes) / sizeof(ctrl_slopes[0]);
    int nm = sizeof(ctrl_mins) / sizeof(ctrl_mins[0]);
    int nd = sizeof(damp_slopes) / sizeof(damp_slopes[0]);

    for (int iv = 0; iv < nv; iv++) {
        for (int ic = 0; ic < nc; ic++) {
            for (int im = 0; im < nm; im++) {
                for (int id = 0; id < nd; id++) {
                    FlightParams params = {
                        .control_v_ref = v_refs[iv],
                        .control_scale_slope = ctrl_slopes[ic],
                        .control_scale_min = ctrl_mins[im],
                        .damping_scale_slope = damp_slopes[id]
                    };

                    SweepResult sr = score_params(&params);

                    printf("%.0f,%.4f,%.2f,%.4f,%.4f,%.4f,%d\n",
                           params.control_v_ref,
                           params.control_scale_slope,
                           params.control_scale_min,
                           params.damping_scale_slope,
                           sr.total_score,
                           sr.max_pitch_rate,
                           sr.all_passed);
                }
            }
        }
    }
}

// Analyze and find best parameters
static void analyze_sweep(void) {
    printf("=============================================================\n");
    printf("  PARAMETER SWEEP ANALYSIS\n");
    printf("=============================================================\n\n");

    // Test ranges
    float v_refs[] = {70, 75, 80, 85, 90};
    float ctrl_slopes[] = {0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012};
    float ctrl_mins[] = {0.25, 0.30, 0.35, 0.40, 0.45, 0.50};
    float damp_slopes[] = {0.0, 0.003, 0.006, 0.009, 0.012};

    int nv = sizeof(v_refs) / sizeof(v_refs[0]);
    int nc = sizeof(ctrl_slopes) / sizeof(ctrl_slopes[0]);
    int nm = sizeof(ctrl_mins) / sizeof(ctrl_mins[0]);
    int nd = sizeof(damp_slopes) / sizeof(damp_slopes[0]);

    int total_combos = nv * nc * nm * nd;
    printf("Testing %d parameter combinations...\n\n", total_combos);

    SweepResult best_score = {.total_score = FLT_MAX};
    SweepResult best_maxrate = {.max_pitch_rate = FLT_MAX};
    int passing_count = 0;
    int tested = 0;

    for (int iv = 0; iv < nv; iv++) {
        for (int ic = 0; ic < nc; ic++) {
            for (int im = 0; im < nm; im++) {
                for (int id = 0; id < nd; id++) {
                    FlightParams params = {
                        .control_v_ref = v_refs[iv],
                        .control_scale_slope = ctrl_slopes[ic],
                        .control_scale_min = ctrl_mins[im],
                        .damping_scale_slope = damp_slopes[id]
                    };

                    SweepResult sr = score_params(&params);
                    tested++;

                    if (sr.all_passed) passing_count++;

                    if (sr.total_score < best_score.total_score) {
                        best_score = sr;
                    }
                    if (sr.max_pitch_rate < best_maxrate.max_pitch_rate) {
                        best_maxrate = sr;
                    }
                }
            }
        }
        // Progress
        fprintf(stderr, "\rProgress: %d/%d (%d%%)...", tested, total_combos, 100 * tested / total_combos);
    }
    fprintf(stderr, "\n\n");

    printf("Results:\n");
    printf("  Total combinations tested: %d\n", total_combos);
    printf("  Combinations passing all tests: %d (%.1f%%)\n\n",
           passing_count, 100.0f * passing_count / total_combos);

    printf("Best by total score (lowest sum of pitch_rate_std):\n");
    printf("  V_ref=%.0f, slope=%.4f, min=%.2f, damp_slope=%.4f\n",
           best_score.params.control_v_ref,
           best_score.params.control_scale_slope,
           best_score.params.control_scale_min,
           best_score.params.damping_scale_slope);
    printf("  Total score: %.4f, Max rate: %.4f, All passed: %s\n\n",
           best_score.total_score, best_score.max_pitch_rate,
           best_score.all_passed ? "YES" : "NO");

    printf("Best by worst-case (lowest max pitch rate):\n");
    printf("  V_ref=%.0f, slope=%.4f, min=%.2f, damp_slope=%.4f\n",
           best_maxrate.params.control_v_ref,
           best_maxrate.params.control_scale_slope,
           best_maxrate.params.control_scale_min,
           best_maxrate.params.damping_scale_slope);
    printf("  Total score: %.4f, Max rate: %.4f, All passed: %s\n\n",
           best_maxrate.total_score, best_maxrate.max_pitch_rate,
           best_maxrate.all_passed ? "YES" : "NO");

    // Test current defaults
    printf("Current defaults (compile-time #defines):\n");
    FlightParams defaults = default_flight_params();
    SweepResult sr_default = score_params(&defaults);
    printf("  V_ref=%.0f, slope=%.4f, min=%.2f, damp_slope=%.4f\n",
           defaults.control_v_ref, defaults.control_scale_slope,
           defaults.control_scale_min, defaults.damping_scale_slope);
    printf("  Total score: %.4f, Max rate: %.4f, All passed: %s\n\n",
           sr_default.total_score, sr_default.max_pitch_rate,
           sr_default.all_passed ? "YES" : "NO");

    // Show detailed breakdown of best params
    printf("=============================================================\n");
    printf("  DETAILED BREAKDOWN OF BEST PARAMS\n");
    printf("=============================================================\n\n");

    FlightParams* best = &best_score.params;
    printf("Parameters: V_ref=%.0f, slope=%.4f, min=%.2f, damp_slope=%.4f\n\n",
           best->control_v_ref, best->control_scale_slope,
           best->control_scale_min, best->damping_scale_slope);

    float speeds[] = {80, 100, 120, 140, 160, 180};
    float elevators[] = {-0.3f, -0.5f, -0.7f};

    for (int e = 0; e < 3; e++) {
        printf("--- Elevator: %.1f ---\n", elevators[e]);
        printf("%-8s  %-10s  %-10s  %-10s  %s\n",
               "Speed", "RateStd", "CtrlScale", "DampScale", "Status");
        printf("--------  ----------  ----------  ----------  ------\n");

        for (int s = 0; s < 6; s++) {
            TestResult r = run_test_with_params(speeds[s], elevators[e], best);

            float ctrl_scale = 1.0f - fmaxf(0.0f, speeds[s] - best->control_v_ref) * best->control_scale_slope;
            ctrl_scale = fmaxf(ctrl_scale, best->control_scale_min);
            float damp_scale = 1.0f + fmaxf(0.0f, speeds[s] - best->control_v_ref) * best->damping_scale_slope;

            int passed = (r.pitch_rate_std < MAX_PITCH_RATE_STD);
            printf("%6.0f    %8.4f    %8.2f    %8.2f    %s\n",
                   speeds[s], r.pitch_rate_std, ctrl_scale, damp_scale,
                   passed ? "PASS" : "FAIL");
        }
        printf("\n");
    }

    // Print recommended #define updates
    printf("=============================================================\n");
    printf("  RECOMMENDED #DEFINE UPDATES\n");
    printf("=============================================================\n\n");

    printf("If the best params differ from current, update flightlib.h:\n\n");
    printf("#define CONTROL_V_REF %.1ff\n", best_score.params.control_v_ref);
    printf("#define CONTROL_SCALE_SLOPE %.4ff\n", best_score.params.control_scale_slope);
    printf("#define CONTROL_SCALE_MIN %.2ff\n", best_score.params.control_scale_min);
    if (best_score.params.damping_scale_slope > 0.0f) {
        printf("#define DAMPING_SCALE_SLOPE %.4ff  // NEW\n", best_score.params.damping_scale_slope);
    }
}

// Find optimal control_scale for each speed independently
// This is data-driven: we let the physics tell us what control authority each speed needs
static void find_optimal_per_speed(void) {
    float speeds[] = {80, 100, 120, 140, 160, 180, 200};
    float elevators[] = {-0.3f, -0.5f, -0.7f};
    int num_speeds = sizeof(speeds) / sizeof(speeds[0]);
    int num_elevs = sizeof(elevators) / sizeof(elevators[0]);

    printf("=============================================================\n");
    printf("  PER-SPEED OPTIMAL CONTROL SCALE DISCOVERY\n");
    printf("=============================================================\n\n");
    printf("For each speed, sweep control_scale from 0.10 to 1.00 and find\n");
    printf("the value that minimizes pitch_rate_std while maintaining response.\n\n");

    // First, output detailed per-elevator results
    for (int e = 0; e < num_elevs; e++) {
        float elevator = elevators[e];
        printf("--- Elevator: %.1f ---\n", elevator);
        printf("speed,optimal_scale,pitch_rate_std,mean_pitch_rate\n");

        for (int s = 0; s < num_speeds; s++) {
            float speed = speeds[s];
            float best_scale = 1.0f;
            float best_std = FLT_MAX;
            float best_mean_rate = 0.0f;

            // Sweep control_scale from 0.05 to 1.00
            for (float scale = 0.05f; scale <= 1.01f; scale += 0.02f) {
                TestResult r = run_test_with_fixed_scale(speed, elevator, scale);

                // We want low pitch_rate_std (stability)
                // but also some responsiveness (mean pitch rate should be reasonable)
                if (r.pitch_rate_std < best_std) {
                    best_std = r.pitch_rate_std;
                    best_scale = scale;
                }
            }

            // Run one more time with the best scale to get all metrics
            TestResult final = run_test_with_fixed_scale(speed, elevator, best_scale);

            printf("%.0f,%.2f,%.4f,%.4f\n",
                   speed, best_scale, best_std, final.pitch_std);
        }
        printf("\n");
    }

    // Now run with all elevators combined and find average optimal per speed
    printf("=============================================================\n");
    printf("  COMBINED ANALYSIS (averaged across elevator inputs)\n");
    printf("=============================================================\n\n");

    printf("speed,optimal_scale,avg_pitch_rate_std\n");

    float optimal_scales[7];  // Store for formula fitting
    float optimal_stds[7];

    for (int s = 0; s < num_speeds; s++) {
        float speed = speeds[s];
        float best_scale = 1.0f;
        float best_total_std = FLT_MAX;

        // Sweep control_scale
        for (float scale = 0.05f; scale <= 1.01f; scale += 0.02f) {
            float total_std = 0.0f;

            // Test across all elevator inputs
            for (int e = 0; e < num_elevs; e++) {
                TestResult r = run_test_with_fixed_scale(speed, elevators[e], scale);
                total_std += r.pitch_rate_std;
            }

            float avg_std = total_std / num_elevs;

            if (avg_std < best_total_std) {
                best_total_std = avg_std;
                best_scale = scale;
            }
        }

        optimal_scales[s] = best_scale;
        optimal_stds[s] = best_total_std;

        printf("%.0f,%.2f,%.4f\n", speed, best_scale, best_total_std);
    }

    // Derive formula from the data
    printf("\n=============================================================\n");
    printf("  FORMULA DERIVATION\n");
    printf("=============================================================\n\n");

    // The flightlib.h formula is: scale = 1.0 - slope * (speed - v_ref)
    // We need to find v_ref and slope to match the optimal values.
    //
    // Two approaches:
    // 1. Linear regression on the data: scale = a + b * speed
    // 2. Find the speed where scale=1.0 (v_ref) and the slope

    // Linear regression: scale = a + b * speed
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int n = num_speeds;

    for (int i = 0; i < n; i++) {
        float x = speeds[i];
        float y = optimal_scales[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    float intercept = (sum_y - slope * sum_x) / n;

    // Find min scale (floor)
    float min_scale = optimal_scales[0];
    for (int i = 1; i < n; i++) {
        if (optimal_scales[i] < min_scale) min_scale = optimal_scales[i];
    }

    // For flightlib.h formula: scale = 1.0 - ctrl_slope * (speed - v_ref)
    // This is: scale = 1.0 + ctrl_slope * v_ref - ctrl_slope * speed
    // Comparing with: scale = intercept + slope * speed
    // We get: ctrl_slope = -slope, and 1.0 + ctrl_slope * v_ref = intercept
    // So: v_ref = (intercept - 1.0) / ctrl_slope = (intercept - 1.0) / (-slope)
    float ctrl_slope = -slope;
    float v_ref = (intercept - 1.0f) / ctrl_slope;

    printf("Linear regression: scale = %.4f + %.6f * speed\n", intercept, slope);
    printf("  (At speed=0: scale=%.3f, slope=%.6f per m/s)\n\n", intercept, slope);

    printf("Mapping to flightlib.h formula: scale = 1.0 - slope * (speed - v_ref)\n");
    printf("  v_ref = %.1f m/s (speed where scale would be 1.0)\n", v_ref);
    printf("  ctrl_slope = %.6f per m/s\n\n", ctrl_slope);

    // Check if v_ref is reasonable (should be below our lowest test speed)
    if (v_ref > speeds[0] || v_ref < 0) {
        printf("NOTE: v_ref=%.1f is outside typical range. The optimal scales\n", v_ref);
        printf("      suggest that control authority is already too high at all\n");
        printf("      test speeds. Consider reducing base control derivatives.\n\n");
    }

    printf("Recommended #define updates for flightlib.h:\n");
    if (v_ref > 0 && v_ref < speeds[0]) {
        printf("  #define CONTROL_V_REF %.1ff\n", v_ref);
    } else {
        printf("  #define CONTROL_V_REF 80.0f  // (keeping current, see note above)\n");
    }
    printf("  #define CONTROL_SCALE_SLOPE %.6ff\n", ctrl_slope);
    printf("  #define CONTROL_SCALE_MIN %.2ff\n\n", min_scale);

    // Verify with the linear fit formula (not the flightlib formula)
    printf("Verification: comparing optimal vs linear fit:\n");
    printf("speed,optimal,linear_fit,diff\n");

    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float fit_scale = intercept + slope * speeds[i];
        fit_scale = fmaxf(fit_scale, min_scale);
        float diff = fabsf(optimal_scales[i] - fit_scale);
        if (diff > max_diff) max_diff = diff;

        printf("%.0f,%.2f,%.2f,%.3f\n",
               speeds[i], optimal_scales[i], fit_scale, diff);
    }

    printf("\nMax deviation from linear fit: %.3f\n", max_diff);

    // Also show what scale the current flightlib.h formula gives
    printf("\n--- Comparison with current flightlib.h formula ---\n");
    printf("Current: V_ref=%.0f, slope=%.4f, min=%.2f\n",
           CONTROL_V_REF, CONTROL_SCALE_SLOPE, CONTROL_SCALE_MIN);
    printf("speed,optimal,current_formula,diff\n");
    for (int i = 0; i < n; i++) {
        float current_scale = 1.0f - fmaxf(0.0f, speeds[i] - CONTROL_V_REF) * CONTROL_SCALE_SLOPE;
        current_scale = fmaxf(current_scale, CONTROL_SCALE_MIN);
        float diff = fabsf(optimal_scales[i] - current_scale);
        printf("%.0f,%.2f,%.2f,%.3f\n",
               speeds[i], optimal_scales[i], current_scale, diff);
    }

    // Alternative approach: scale the base control derivatives
    printf("\n=============================================================\n");
    printf("  ALTERNATIVE APPROACH: Scale Base Control Derivatives\n");
    printf("=============================================================\n\n");

    float scale_at_80 = optimal_scales[0];  // Optimal at lowest test speed
    float scale_at_200 = optimal_scales[num_speeds - 1];  // Optimal at highest
    float reduction_factor = scale_at_80;  // How much to reduce base derivatives

    printf("Key insight: optimal scale at 80 m/s is %.2f, not 1.0\n", scale_at_80);
    printf("This means the base control derivatives are ~%.1fx too strong.\n\n", 1.0f / reduction_factor);

    printf("Option 1: Reduce base derivatives, adjust formula\n");
    printf("  Scale CM_DELTA_E by %.2f: -0.5 * %.2f = %.3f\n",
           reduction_factor, reduction_factor, -0.5f * reduction_factor);
    printf("  Then use formula with V_ref=80, new slope, min=%.2f/%.2f=%.2f\n\n",
           scale_at_200, scale_at_80, scale_at_200 / scale_at_80);

    printf("Option 2: Modify formula to match optimal directly\n");
    // New formula: scale = base_scale * (1.0 - rel_slope * (speed - 80))
    // where base_scale is the optimal at 80 m/s
    float rel_slope = (scale_at_80 - scale_at_200) / (200.0f - 80.0f);
    printf("  Use: scale = %.2f * (1.0 - %.5f * (speed - 80))\n", scale_at_80, rel_slope / scale_at_80);
    printf("  Or equivalently: scale = %.2f - %.6f * (speed - 80)\n\n", scale_at_80, rel_slope);

    // This maps to flightlib.h as:
    // scale = 1.0 - slope * (speed - v_ref)
    // We want scale(80) = scale_at_80, scale(200) = scale_at_200
    // From scale = a - b * (speed - 80):
    //   a = scale_at_80, b = rel_slope
    // From scale = 1.0 - slope * (speed - v_ref):
    //   At speed=80: scale_at_80 = 1.0 - slope * (80 - v_ref)
    //   We need v_ref such that scale = 1.0 at v_ref
    //   slope = rel_slope, v_ref = 80 - (1.0 - scale_at_80) / rel_slope
    float new_v_ref = 80.0f - (1.0f - scale_at_80) / rel_slope;
    printf("  In flightlib.h terms:\n");
    printf("    #define CONTROL_V_REF %.1ff\n", new_v_ref);
    printf("    #define CONTROL_SCALE_SLOPE %.6ff\n", rel_slope);
    printf("    #define CONTROL_SCALE_MIN %.2ff\n\n", scale_at_200);

    // Final validation: run the oscillation test with derived params
    printf("\n=============================================================\n");
    printf("  VALIDATION WITH DERIVED PARAMETERS (Option 2)\n");
    printf("=============================================================\n\n");

    // Use the new formula parameters from Option 2
    FlightParams derived = {
        .control_v_ref = new_v_ref,
        .control_scale_slope = rel_slope,
        .control_scale_min = scale_at_200,
        .damping_scale_slope = 0.0f
    };

    printf("Testing with: V_ref=%.1f, slope=%.6f, min=%.2f\n\n",
           derived.control_v_ref, derived.control_scale_slope, derived.control_scale_min);

    int all_passed = 1;
    for (int e = 0; e < num_elevs; e++) {
        printf("--- Elevator: %.1f ---\n", elevators[e]);
        printf("%-8s  %-10s  %-8s  %s\n", "Speed", "RateStd", "Scale", "Status");
        printf("--------  ----------  --------  ------\n");

        for (int s = 0; s < num_speeds; s++) {
            TestResult r = run_test_with_params(speeds[s], elevators[e], &derived);

            float scale = 1.0f - fmaxf(0.0f, speeds[s] - derived.control_v_ref) * derived.control_scale_slope;
            scale = fmaxf(scale, derived.control_scale_min);

            int passed = (r.pitch_rate_std < MAX_PITCH_RATE_STD);
            if (!passed) all_passed = 0;

            printf("%6.0f    %8.4f    %6.2f    %s\n",
                   speeds[s], r.pitch_rate_std, scale, passed ? "PASS" : "FAIL");
        }
        printf("\n");
    }

    printf("VALIDATION RESULT: %s\n", all_passed ? "ALL PASSED" : "SOME FAILED");
}

// Print usage information
static void print_usage(const char* prog) {
    printf("Usage: %s [option]\n\n", prog);
    printf("Options:\n");
    printf("  (none)         Run oscillation test with current parameters\n");
    printf("  --sweep        Full coarse parameter sweep (CSV output)\n");
    printf("  --sweep-fine   Fine sweep around good values (CSV output)\n");
    printf("  --analyze      Find and report optimal parameters\n");
    printf("  --find-optimal Find optimal control_scale for each speed independently\n");
    printf("  --help         Show this help message\n");
}

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        if (strcmp(argv[1], "--sweep") == 0) {
            run_full_sweep();
            return 0;
        }
        if (strcmp(argv[1], "--sweep-fine") == 0) {
            run_fine_sweep();
            return 0;
        }
        if (strcmp(argv[1], "--analyze") == 0) {
            analyze_sweep();
            return 0;
        }
        if (strcmp(argv[1], "--find-optimal") == 0) {
            find_optimal_per_speed();
            return 0;
        }
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        // Legacy options for backward compatibility
        if (strcmp(argv[1], "--sweep-control") == 0) {
            run_full_sweep();
            return 0;
        }
        if (strcmp(argv[1], "--analyze-damping") == 0) {
            analyze_sweep();
            return 0;
        }
        printf("Unknown option: %s\n", argv[1]);
        print_usage(argv[0]);
        return 1;
    }

    return run_oscillation_test();
}
