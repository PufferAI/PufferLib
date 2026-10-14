/* MIT, Copyright (c) 2026 Marc / Rage Industries.
 * Frozen target encoder -> latent-only state inversion -> TVLQR/LQR.
 * The JEPA predictor is not an action planner. No raw-state fallback exists.
 */
#ifndef PUFFER_NPOLE_CONTROLLER_H
#define PUFFER_NPOLE_CONTROLLER_H

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "npole_physics.h"

#define NP_OBS 33
#define NP_HIDDEN 128
#define NP_LATENT 64
#define NP_KNOTS 900
#define NP_INVERSE_STEPS 20
#define NP_INVERSE_EPS 1e-5
#define NP_INVERSE_TOL 1e-12

/* Serialized as little-endian doubles in this exact order. */
typedef struct {
    double mean[NP_OBS];
    double std[NP_OBS];
    double w1[NP_HIDDEN][NP_OBS];
    double b1[NP_HIDDEN];
    double w2[NP_HIDDEN][NP_HIDDEN];
    double b2[NP_HIDDEN];
    double w3[NP_LATENT][NP_HIDDEN];
    double b3[NP_LATENT];
    double reference[NP_KNOTS + 1][NP_STATE];
    double actions[NP_KNOTS];
    double gains[NP_KNOTS][NP_STATE];
    double balance[NP_STATE + 1];
} NpModel;

typedef struct {
    int tick;
    double residual;
    double condition;
    double estimate[NP_STATE];
    double latent[NP_LATENT];
} NpPolicy;

static NpModel *np_load_model(const char *path) {
    uint32_t endian = 1;
    if (sizeof(double) != 8 || *((unsigned char *)&endian) != 1) {
        return NULL;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        return NULL;
    }
    char magic[8];
    uint32_t dims[6];
    uint32_t expected[6] = {NP_STATE, NP_OBS, NP_HIDDEN, NP_LATENT,
        NP_KNOTS, (uint32_t)(sizeof(NpModel) / sizeof(double))};
    if (fread(magic, 1, 8, file) != 8 || memcmp(magic, "NP10v001", 8)
        || fread(dims, sizeof(uint32_t), 6, file) != 6
        || memcmp(dims, expected, sizeof(dims))) {
        fclose(file);
        return NULL;
    }
    NpModel *model = (NpModel *)malloc(sizeof(NpModel));
    if (!model) {
        fclose(file);
        return NULL;
    }
    int valid = fread(model, 1, sizeof(*model), file) == sizeof(*model)
        && fgetc(file) == EOF;
    fclose(file);
    const double *values = (const double *)model;
    for (size_t i = 0; valid && i < sizeof(*model) / sizeof(double); i++) {
        valid = isfinite(values[i]);
    }
    for (int i = 0; valid && i < NP_OBS; i++) {
        valid = model->std[i] > 0.0;
    }
    if (!valid) {
        free(model);
        return NULL;
    }
    return model;
}

static double np_silu(double x) {
    if (x >= 0.0) {
        return x / (1.0 + exp(-x));
    }
    double exponential = exp(x);
    return x * exponential / (1.0 + exponential);
}

static int np_encode(const NpModel *model, const double *y, double last,
        double *latent) {
    double obs[NP_OBS];
    double hidden[2][NP_HIDDEN];
    obs[0] = y[0];
    obs[1] = y[NP_DOF];
    for (int i = 0; i < NP_POLES; i++) {
        obs[2 + 3 * i] = sin(y[1 + i]);
        obs[3 + 3 * i] = cos(y[1 + i]);
        obs[4 + 3 * i] = y[NP_DOF + 1 + i] / 8.0;
    }
    obs[NP_OBS - 1] = last;
    for (int i = 0; i < NP_OBS; i++) {
        obs[i] = (obs[i] - model->mean[i]) / model->std[i];
        if (!isfinite(obs[i])) {
            return 1;
        }
    }
    for (int i = 0; i < NP_HIDDEN; i++) {
        double value = model->b1[i];
        for (int j = 0; j < NP_OBS; j++) {
            value += model->w1[i][j] * obs[j];
        }
        hidden[0][i] = np_silu(value);
    }
    for (int i = 0; i < NP_HIDDEN; i++) {
        double value = model->b2[i];
        for (int j = 0; j < NP_HIDDEN; j++) {
            value += model->w2[i][j] * hidden[0][j];
        }
        hidden[1][i] = np_silu(value);
    }
    for (int i = 0; i < NP_LATENT; i++) {
        double value = model->b3[i];
        for (int j = 0; j < NP_HIDDEN; j++) {
            value += model->w3[i][j] * hidden[1][j];
        }
        latent[i] = value;
        if (!isfinite(value)) {
            return 1;
        }
    }
    return 0;
}

static int np_jacobian(const NpModel *model, const double *reference,
        double last, double jacobian[NP_LATENT][NP_STATE]) {
    double candidate[NP_STATE];
    double plus[NP_LATENT];
    double minus[NP_LATENT];
    for (int col = 0; col < NP_STATE; col++) {
        memcpy(candidate, reference, sizeof(candidate));
        candidate[col] += NP_INVERSE_EPS;
        if (np_encode(model, candidate, last, plus)) {
            return 1;
        }
        candidate[col] -= 2.0 * NP_INVERSE_EPS;
        if (np_encode(model, candidate, last, minus)) {
            return 1;
        }
        for (int row = 0; row < NP_LATENT; row++) {
            jacobian[row][col] = (plus[row] - minus[row]) / (2.0 * NP_INVERSE_EPS);
        }
    }
    return 0;
}

/* One-sided Jacobi SVD orthogonalizes the rectangular Jacobian directly.
 * Do not form J^T J: that squares the inverse problem's condition number.
 * B = J V, J^+ = V diag(1 / sigma^2) B^T. Full rank is required.
 */
static int np_pinv(double jacobian[NP_LATENT][NP_STATE],
        double inverse[NP_STATE][NP_LATENT], double *condition) {
    double vectors[NP_STATE][NP_STATE] = {{0.0}};
    for (int i = 0; i < NP_STATE; i++) {
        vectors[i][i] = 1.0;
    }
    int converged = 0;
    for (int sweep = 0; sweep < 80; sweep++) {
        int changed = 0;
        for (int p = 0; p < NP_STATE - 1; p++) {
            for (int q = p + 1; q < NP_STATE; q++) {
                double alpha = 0.0;
                double beta = 0.0;
                double gamma = 0.0;
                for (int row = 0; row < NP_LATENT; row++) {
                    double a = jacobian[row][p];
                    double b = jacobian[row][q];
                    alpha += a * a;
                    beta += b * b;
                    gamma += a * b;
                }
                if (!isfinite(alpha + beta + gamma) || alpha <= 0.0 || beta <= 0.0) {
                    return 1;
                }
                if (fabs(gamma) <= 8.0 * DBL_EPSILON * sqrt(alpha * beta)) {
                    continue;
                }
                double tau = (beta - alpha) / (2.0 * gamma);
                double t = copysign(1.0, tau) / (fabs(tau) + hypot(1.0, tau));
                double cosine = 1.0 / sqrt(1.0 + t * t);
                double sine = cosine * t;
                for (int row = 0; row < NP_LATENT; row++) {
                    double a = jacobian[row][p];
                    double b = jacobian[row][q];
                    jacobian[row][p] = cosine * a - sine * b;
                    jacobian[row][q] = sine * a + cosine * b;
                }
                for (int row = 0; row < NP_STATE; row++) {
                    double a = vectors[row][p];
                    double b = vectors[row][q];
                    vectors[row][p] = cosine * a - sine * b;
                    vectors[row][q] = sine * a + cosine * b;
                }
                changed = 1;
            }
        }
        if (!changed) {
            converged = 1;
            break;
        }
    }
    if (!converged) {
        return 1;
    }
    double squares[NP_STATE];
    double largest = 0.0;
    double smallest = HUGE_VAL;
    for (int col = 0; col < NP_STATE; col++) {
        squares[col] = 0.0;
        for (int row = 0; row < NP_LATENT; row++) {
            squares[col] += jacobian[row][col] * jacobian[row][col];
        }
        largest = fmax(largest, squares[col]);
        smallest = fmin(smallest, squares[col]);
    }
    if (!isfinite(largest) || smallest <= 1e-24 || smallest <= largest * 1e-24) {
        return 1;
    }
    *condition = sqrt(largest / smallest);
    for (int row = 0; row < NP_STATE; row++) {
        for (int col = 0; col < NP_LATENT; col++) {
            double value = 0.0;
            for (int k = 0; k < NP_STATE; k++) {
                value += vectors[row][k] * jacobian[col][k] / squares[k];
            }
            inverse[row][col] = value;
        }
    }
    return 0;
}

static double np_latent_residual(const double *a, const double *b) {
    double sum = 0.0;
    for (int i = 0; i < NP_LATENT; i++) {
        double difference = a[i] - b[i];
        sum += difference * difference;
    }
    return sqrt(sum);
}

/* This function only receives a latent, the known applied force and a fixed
 * reference. The state used to encode the observation is unavailable here.
 */
static int np_estimate(const NpModel *model, NpPolicy *policy, double last) {
    int index = policy->tick > 0 ? policy->tick - 1 : 0;
    if (index > NP_KNOTS) {
        index = NP_KNOTS;
    }
    const double *reference = model->reference[index];
    double jacobian[NP_LATENT][NP_STATE];
    double inverse[NP_STATE][NP_LATENT];
    double encoded[NP_LATENT];
    memcpy(policy->estimate, reference, sizeof(policy->estimate));
    for (int refinement = 0; refinement < 2; refinement++) {
        const double *linearization = refinement ? policy->estimate : reference;
        if (np_jacobian(model, linearization, last, jacobian)
            || np_pinv(jacobian, inverse, &policy->condition)) {
            return 1;
        }
        for (int iteration = 0; iteration < NP_INVERSE_STEPS; iteration++) {
            if (np_encode(model, policy->estimate, last, encoded)) {
                return 1;
            }
            for (int row = 0; row < NP_STATE; row++) {
                double correction = 0.0;
                for (int col = 0; col < NP_LATENT; col++) {
                    correction += inverse[row][col] * (policy->latent[col] - encoded[col]);
                }
                policy->estimate[row] += correction;
                if (!isfinite(policy->estimate[row])) {
                    return 1;
                }
            }
            for (int i = 1; i <= NP_POLES; i++) {
                policy->estimate[i] = reference[i] + np_wrap(policy->estimate[i] - reference[i]);
            }
        }
        if (np_encode(model, policy->estimate, last, encoded)) {
            return 1;
        }
        policy->residual = np_latent_residual(encoded, policy->latent);
        if (isfinite(policy->residual) && policy->residual <= NP_INVERSE_TOL) {
            return 0;
        }
    }
    return 1;
}

static int np_feedback(const NpModel *model, int tick, const double *estimate,
        double pending, double *command) {
    double error[NP_STATE];
    memcpy(error, estimate, sizeof(error));
    double value = 0.0;
    if (tick < NP_KNOTS) {
        if (np_integrate(error, pending, 4, NP_DT)) {
            return 1;
        }
        value = model->actions[tick];
        for (int i = 0; i < NP_STATE; i++) {
            error[i] -= model->reference[tick][i];
        }
        for (int i = 1; i <= NP_POLES; i++) {
            error[i] = np_wrap(error[i]);
        }
        for (int i = 0; i < NP_STATE; i++) {
            value -= model->gains[tick][i] * error[i];
        }
    } else {
        for (int i = 1; i <= NP_POLES; i++) {
            error[i] = np_wrap(error[i]);
        }
        for (int i = 0; i < NP_STATE; i++) {
            value -= model->balance[i] * error[i];
        }
        value -= model->balance[NP_STATE] * pending;
    }
    if (!isfinite(value)) {
        return 1;
    }
    *command = np_clip(value);
    return 0;
}

static int np_policy_act(const NpModel *model, NpPolicy *policy,
        const NpState *state, double *command) {
    if (np_encode(model, state->y, state->last_applied, policy->latent)
        || np_estimate(model, policy, state->last_applied)
        || np_feedback(model, policy->tick, policy->estimate, state->pending, command)) {
        return 1;
    }
    policy->tick++;
    return 0;
}

#endif
