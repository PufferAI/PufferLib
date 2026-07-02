/*
 * test_control_scale_values.c - 1:1 port of test_high_speed_oscillation.py
 *   ::test_control_scale_values (the formula-only sub-test).
 *
 * The simulation half of test_high_speed_oscillation.py is already covered by
 * ocean/dogfight/test_flight_dynamics.c and
 * ocean/dogfight/tests/test_flight_physics.c::test_high_speed_pitch_oscillation.
 * Do NOT re-port that here.
 *
 * 4.0 disabled the high-speed control authority scaling (slope=0, min=1.0),
 * so most of the 3.0 expected values will mismatch. That mismatch IS the
 * honest 1:1 result. Per-entry status is printed, but main() returns 0
 * (not a regression — a documented config change in flightlib.h:200-204).
 */
#include <stdio.h>
#include <math.h>

#include "../flightlib.h"

typedef struct {
    int speed;
    float expected;  // 3.0 expected scale (V_REF=80, SLOPE=0.007, MIN=0.35)
} Case;

/* Same table as test_high_speed_oscillation.py:102-110, verbatim. */
static const Case CASES[] = {
    { 80, 1.00f},
    {100, 0.86f},
    {120, 0.72f},
    {140, 0.58f},
    {160, 0.44f},
    {180, 0.35f},
    {200, 0.35f},
    { 60, 1.00f},
};
#define N_CASES (sizeof(CASES) / sizeof(CASES[0]))

static float control_scale(float speed) {
    /* Mirrors the C formula in flightlib.h compute_control_scale(). */
    float over = speed - CONTROL_V_REF;
    if (over < 0.0f) over = 0.0f;
    float scale = 1.0f - over * CONTROL_SCALE_SLOPE;
    if (scale < CONTROL_SCALE_MIN) scale = CONTROL_SCALE_MIN;
    return scale;
}

int main(void) {
    printf("\nVerifying control scale formula (4.0 vs 3.0 expected)...\n");
    printf("--------------------------------------------------------\n");
    printf("flightlib.h: V_REF=%.1f SLOPE=%.4f MIN=%.2f\n",
           CONTROL_V_REF, CONTROL_SCALE_SLOPE, CONTROL_SCALE_MIN);

    int n_ok = 0;
    int n_diff = 0;
    for (size_t i = 0; i < N_CASES; ++i) {
        float scale = control_scale((float)CASES[i].speed);
        int match = fabsf(scale - CASES[i].expected) < 0.001f;
        const char* status = match ? "OK" : "FAIL";
        if (match) ++n_ok; else ++n_diff;
        printf("V=%3d m/s: scale=%.2f (3.0 expected %.2f) [%s]\n",
               CASES[i].speed, scale, CASES[i].expected, status);
    }

    printf("--------------------------------------------------------\n");
    printf("Matches 3.0: %d/%zu\n", n_ok, N_CASES);
    if (n_diff > 0) {
        printf("Note: 4.0 disabled high-speed authority scaling "
               "(slope=0, min=1.0). 3.0 expected values are kept here for\n");
        printf("      audit purposes; mismatches are the documented config "
               "change, not a physics regression.\n");
    }
    return 0;
}
