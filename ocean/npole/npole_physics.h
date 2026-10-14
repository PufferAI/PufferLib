/* MIT, Copyright (c) 2026 Marc / Rage Industries.
 * Ten passive serial links, point masses at their tips, one cart actuator.
 * Coordinates are x, ten absolute angles, x velocity, ten angular velocities.
 * This file has no rendering, controller, CUDA, or external dependencies.
 */
#ifndef PUFFER_NPOLE_PHYSICS_H
#define PUFFER_NPOLE_PHYSICS_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#define NP_POLES 10
#define NP_DOF 11
#define NP_STATE 22
#define NP_DT 0.02
#define NP_SUBSTEPS 20
#define NP_HORIZON 1000
#define NP_FINAL_TICKS 250
#define NP_RAIL 2.4
#define NP_ANGLE 0.22
#define NP_OMEGA 1.0
#define NP_PI 3.14159265358979323846264338327950288

typedef struct {
    double y[NP_STATE];
    double last_applied;
    double pending;
    int tick;
} NpState;

typedef struct {
    NpState state;
    int hold;
    int rail_ok;
    int numeric_ok;
    int final_window_ok;
} NpEpisode;

static double np_clip(double value) {
    return fmax(-1.0, fmin(1.0, value));
}

static double np_wrap(double angle) {
    double value = fmod(angle + NP_PI, 2.0 * NP_PI);
    if (value < 0.0) {
        value += 2.0 * NP_PI;
    }
    return value - NP_PI;
}

/* The 1e-6 diagonal is part of the declared analytic plant, not a solver
 * fallback. Keep it fixed and disclose it when comparing physical engines.
 */
static int np_acceleration(const double *y, double action, double *out) {
    double mass[NP_DOF][NP_DOF] = {{0.0}};
    double rhs[NP_DOF] = {0.0};
    const double *theta = y + 1;
    const double *velocity = y + NP_DOF;
    const double *omega = velocity + 1;
    if (!isfinite(action)) {
        return 1;
    }
    for (int i = 0; i < NP_STATE; i++) {
        if (!isfinite(y[i])) {
            return 1;
        }
    }
    mass[0][0] = 1.0 + 0.1 * NP_POLES + 1e-6;
    rhs[0] = 10.0 * np_clip(action) - 0.05 * velocity[0];
    for (int i = 0; i < NP_POLES; i++) {
        double remaining = 0.1 * (NP_POLES - i);
        double sine = sin(theta[i]);
        double cosine = cos(theta[i]);
        mass[0][i + 1] = 0.5 * remaining * cosine;
        mass[i + 1][0] = mass[0][i + 1];
        rhs[0] += 0.5 * remaining * sine * omega[i] * omega[i];
        rhs[i + 1] = 9.81 * 0.5 * remaining * sine - 0.05 * omega[i];
        for (int j = 0; j < NP_POLES; j++) {
            int last = i > j ? i : j;
            double coupling = 0.25 * (0.1 * (NP_POLES - last));
            double difference = theta[i] - theta[j];
            mass[i + 1][j + 1] = coupling * cos(difference);
            if (i == j) {
                mass[i + 1][j + 1] += 1e-6;
            }
            rhs[i + 1] -= coupling * sin(difference) * omega[j] * omega[j];
        }
    }
    /* Partial-pivot LU, without clipping coordinates or accelerations. */
    for (int k = 0; k < NP_DOF; k++) {
        int pivot = k;
        for (int i = k + 1; i < NP_DOF; i++) {
            if (fabs(mass[i][k]) > fabs(mass[pivot][k])) {
                pivot = i;
            }
        }
        if (mass[pivot][k] == 0.0 || !isfinite(mass[pivot][k])) {
            return 2;
        }
        if (pivot != k) {
            for (int j = k; j < NP_DOF; j++) {
                double temp = mass[k][j];
                mass[k][j] = mass[pivot][j];
                mass[pivot][j] = temp;
            }
            double temp = rhs[k];
            rhs[k] = rhs[pivot];
            rhs[pivot] = temp;
        }
        for (int i = k + 1; i < NP_DOF; i++) {
            double factor = mass[i][k] / mass[k][k];
            for (int j = k + 1; j < NP_DOF; j++) {
                mass[i][j] -= factor * mass[k][j];
            }
            rhs[i] -= factor * rhs[k];
        }
    }
    for (int i = NP_DOF - 1; i >= 0; i--) {
        double value = rhs[i];
        for (int j = i + 1; j < NP_DOF; j++) {
            value -= mass[i][j] * out[j];
        }
        out[i] = value / mass[i][i];
        if (!isfinite(out[i])) {
            return 3;
        }
    }
    return 0;
}

static int np_derivative(const double *y, double action, double *out) {
    memcpy(out, y + NP_DOF, NP_DOF * sizeof(double));
    return np_acceleration(y, action, out + NP_DOF);
}

static int np_integrate(double *y, double action, int substeps, double dt) {
    double k[4][NP_STATE];
    double trial[NP_STATE];
    if (substeps < 1 || !isfinite(dt) || dt <= 0.0) {
        return 1;
    }
    double h = dt / substeps;
    for (int step = 0; step < substeps; step++) {
        if (np_derivative(y, action, k[0])) {
            return 1;
        }
        for (int stage = 1; stage < 4; stage++) {
            double factor = stage == 3 ? h : h * 0.5;
            for (int i = 0; i < NP_STATE; i++) {
                trial[i] = y[i] + factor * k[stage - 1][i];
            }
            if (np_derivative(trial, action, k[stage])) {
                return 1;
            }
        }
        for (int i = 0; i < NP_STATE; i++) {
            y[i] += h / 6.0 * (k[0][i] + 2.0 * k[1][i]
                + 2.0 * k[2][i] + k[3][i]);
            if (!isfinite(y[i])) {
                return 1;
            }
        }
    }
    return 0;
}

static int np_tick(NpState *state, double command) {
    if (!isfinite(command)) {
        return 1;
    }
    double applied = state->pending;
    state->pending = np_clip(command);
    if (np_integrate(state->y, applied, NP_SUBSTEPS, NP_DT)) {
        return 1;
    }
    state->last_applied = applied;
    state->tick++;
    return 0;
}

static int np_good(const double *y) {
    for (int i = 0; i < NP_STATE; i++) {
        if (!isfinite(y[i])) {
            return 0;
        }
    }
    if (fabs(y[0]) >= NP_RAIL) {
        return 0;
    }
    for (int i = 0; i < NP_POLES; i++) {
        if (!isfinite(y[1 + i]) || !isfinite(y[NP_DOF + 1 + i])
            || fabs(np_wrap(y[1 + i])) >= NP_ANGLE
            || fabs(y[NP_DOF + 1 + i]) >= NP_OMEGA) {
            return 0;
        }
    }
    return 1;
}

/* SplitMix64 produces identical reset samples in native and WASM builds.
 * Golden Python fixtures carry explicit initial states, not matching RNG IDs.
 */
static double np_uniform(uint64_t *rng, double span) {
    *rng += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = *rng;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    double unit = (double)(z >> 11) * 0x1.0p-53;
    return span * (2.0 * unit - 1.0);
}

static void np_reset(NpEpisode *episode, const double *initial) {
    memset(episode, 0, sizeof(*episode));
    memcpy(episode->state.y, initial, NP_STATE * sizeof(double));
    episode->rail_ok = isfinite(initial[0]) && fabs(initial[0]) < NP_RAIL;
    episode->numeric_ok = 1;
    for (int i = 0; i < NP_STATE; i++) {
        episode->numeric_ok &= isfinite(initial[i]);
    }
    episode->final_window_ok = 1;
}

static void np_reset_seed(NpEpisode *episode, uint64_t seed) {
    double initial[NP_STATE];
    initial[0] = np_uniform(&seed, 0.08);
    for (int i = 0; i < NP_POLES; i++) {
        initial[1 + i] = NP_PI + np_uniform(&seed, 0.04);
    }
    for (int i = NP_DOF; i < NP_STATE; i++) {
        initial[i] = np_uniform(&seed, 0.02);
    }
    np_reset(episode, initial);
}

static int np_episode_tick(NpEpisode *episode, double command) {
    if (!episode->numeric_ok || !episode->rail_ok || episode->state.tick >= NP_HORIZON) {
        return 1;
    }
    if (np_tick(&episode->state, command)) {
        episode->numeric_ok = 0;
        episode->hold = 0;
        return 1;
    }
    episode->rail_ok &= fabs(episode->state.y[0]) < NP_RAIL;
    int good = episode->rail_ok && np_good(episode->state.y);
    episode->hold = good ? episode->hold + 1 : 0;
    if (episode->state.tick >= NP_HORIZON - NP_FINAL_TICKS) {
        episode->final_window_ok &= good;
    }
    return !episode->rail_ok;
}

static int np_success(const NpEpisode *episode) {
    return episode->state.tick == NP_HORIZON && episode->numeric_ok
        && episode->rail_ok && episode->final_window_ok
        && episode->hold >= NP_FINAL_TICKS;
}

#endif
