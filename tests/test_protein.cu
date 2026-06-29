// test_protein.cu -- Standalone tests for protein.cu
//
// Build: nvcc -o test_protein tests/test_protein.cu -I src/ -lcublas -lcusolver -lcurand

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "gp_cuda.cu"
#include "protein.cu"

#define T_PASS(...)  do { printf("  PASS: "); printf(__VA_ARGS__); printf("\n"); } while(0)
#define T_FAIL(name, ...) do { printf("  FAIL: %s -- ", name); printf(__VA_ARGS__); printf("\n"); return 1; } while(0)

// -- test: Pareto front -------------------------------------------------------

static int test_pareto()
{
    printf("[test_pareto]\n");

    float scores[] = {1.0f, 3.0f, 2.0f, 4.0f, 2.5f};
    float costs[]  = {10.0f, 30.0f, 20.0f, 50.0f, 40.0f};
    int idx[5];
    // Sorted by cost: (10,1) (20,2) (30,3) (40,2.5) (50,4)
    // Pareto: 1 < 2 < 3 > 2.5 < 4  →  indices 0,2,1,3  (original positions)
    int n = protein_pareto_front(scores, costs, 5, idx);
    if (n != 4) T_FAIL("count", "expected 4, got %d", n);
    // Verify monotonic scores along the front
    for (int i = 1; i < n; i++)
        if (scores[idx[i]] <= scores[idx[i-1]])
            T_FAIL("monotonic", "scores not strictly increasing on front");
    T_PASS("pareto_front");

    // All dominated except best
    float scores2[] = {5.0f, 1.0f, 2.0f, 3.0f};
    float costs2[]  = {10.0f, 20.0f, 30.0f, 40.0f};
    n = protein_pareto_front(scores2, costs2, 4, idx);
    if (n != 1) T_FAIL("single", "expected 1 pareto point, got %d", n);
    T_PASS("pareto_single_dominant");

    return 0;
}

// -- test: Pareto pruning -----------------------------------------------------

static int test_prune()
{
    printf("[test_prune]\n");

    // Create a front with an inefficient tail:
    // (10, 1.0) (20, 2.0) (30, 2.5) (100, 2.51)  ← last one is inefficient
    float scores[] = {1.0f, 2.0f, 2.5f, 2.51f};
    float costs[]  = {10.0f, 20.0f, 30.0f, 100.0f};
    int idx[] = {0, 1, 2, 3};
    int n = protein_prune_pareto(scores, costs, idx, 4, 0.5f, 0.98f);
    if (n >= 4) T_FAIL("prune", "expected pruning, got n=%d", n);
    T_PASS("prune_removes_tail");

    // Front where all steps are efficient → no pruning
    float scores3[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float costs3[]  = {10.0f, 20.0f, 30.0f, 40.0f};
    int idx3[] = {0, 1, 2, 3};
    n = protein_prune_pareto(scores3, costs3, idx3, 4, 0.5f, 0.98f);
    if (n != 4) T_FAIL("no_prune", "expected 4, got %d", n);
    T_PASS("prune_keeps_efficient");

    return 0;
}

// -- test: build search centers -----------------------------------------------

static int test_search_centers()
{
    printf("[test_search_centers]\n");

    // 3 observations, dim=2, cost_dim=1
    float params[] = {0.1f, 0.5f,   0.3f, 0.7f,   0.5f, -0.2f};
    int pareto[] = {0, 1};
    int top[]    = {2};
    float centers[12]; // max: 2 pareto + 2*1 top = 4 rows
    int nc = protein_build_search_centers(params, 2, pareto, 2, top, 1, 1, centers);
    // 2 pareto + 2 cost-shifted top = 4
    if (nc != 4) T_FAIL("count", "expected 4 centers, got %d", nc);

    // Verify pareto centers are unchanged
    if (fabsf(centers[0] - 0.1f) > 1e-5f) T_FAIL("pareto0", "wrong value");
    if (fabsf(centers[2] - 0.3f) > 1e-5f) T_FAIL("pareto1", "wrong value");

    // Verify cost-shifted top obs have lower cost than original (-0.2)
    float shifted1 = centers[4*1 + 1]; // 3rd row, cost dim
    float shifted2 = centers[4*1 + 3]; // 4th row, cost dim (wait, layout is row-major dim=2)
    // row 2 (idx=2): centers[4], centers[5]  → cost = centers[5]
    // row 3 (idx=3): centers[6], centers[7]  → cost = centers[7]
    shifted1 = centers[5];
    shifted2 = centers[7];
    if (shifted1 >= -0.2f) T_FAIL("shift1", "expected < -0.2, got %.3f", shifted1);
    if (shifted2 >= -0.2f) T_FAIL("shift2", "expected < -0.2, got %.3f", shifted2);
    // Half-shift moves further toward -1 than third-shift
    if (shifted1 >= shifted2) T_FAIL("shift_order", "half-shift should be more negative than third-shift");
    T_PASS("search_centers");

    return 0;
}

// -- test: cost model ---------------------------------------------------------

static int test_cost_model()
{
    printf("[test_cost_model]\n");

    int n = 50;
    float scores[50], costs[50];
    srand(42);
    for (int i = 0; i < n; i++) {
        costs[i]  = 10.0f + 90.0f * (float)i / (float)n;
        scores[i] = 0.5f + 0.3f * logf(costs[i]);
    }

    ProteinCostModel model;
    protein_cost_model_init(&model, 0.3f, 10);
    protein_cost_model_fit(&model, scores, costs, n, 100.0f);
    if (!model.is_fitted) T_FAIL("fit", "model not fitted");

    float t50 = protein_cost_model_threshold(&model, 50.0f, 0.3f, 10.0f);
    float expected = 0.5f + 0.3f * logf(50.0f);
    if (fabsf(t50 - expected) > 0.5f) T_FAIL("thresh", "threshold %.3f far from expected %.3f", t50, expected);

    // Below min cost → -FLT_MAX
    float t_low = protein_cost_model_threshold(&model, 1.0f, 0.3f, 10.0f);
    if (t_low > -1e30f) T_FAIL("low_cost", "expected -inf for low cost");

    // Above upper threshold → 0.9 * max_score
    float t_high = protein_cost_model_threshold(&model, 200.0f, 0.3f, 10.0f);
    if (fabsf(t_high - 0.9f * model.max_score) > 0.01f)
        T_FAIL("high_cost", "expected 0.9*max_score, got %.3f", t_high);

    T_PASS("cost_model");
    return 0;
}

// -- test: classifier ---------------------------------------------------------

static int test_classifier()
{
    printf("[test_classifier]\n");

    int dim = 2, n = 40;
    float *X = (float *)malloc((size_t)n * dim * sizeof(float));
    int   *y = (int *)malloc((size_t)n * sizeof(int));

    srand(123);
    for (int i = 0; i < n; i++) {
        X[i * 2]     = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        X[i * 2 + 1] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        y[i] = (X[i * 2] + X[i * 2 + 1] > 0.0f) ? 1 : 0;
    }

    ProteinClassifier *clf = protein_classifier_create(dim);
    protein_classifier_fit(clf, X, y, n, 1.0f, 200);
    if (!clf->is_fitted) T_FAIL("fit", "classifier not fitted");

    // Predict on device
    float *d_X, *d_probs;
    cudaMalloc(&d_X, (size_t)n * dim * sizeof(float));
    cudaMalloc(&d_probs, (size_t)n * sizeof(float));
    cudaMemcpy(d_X, X, (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice);
    protein_classifier_predict_d(clf, d_X, d_probs, n, 0);

    float *h_probs = (float *)malloc((size_t)n * sizeof(float));
    cudaMemcpy(h_probs, d_probs, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);

    // Check: positive examples should generally have p > 0.5
    int correct = 0;
    for (int i = 0; i < n; i++) {
        int pred = h_probs[i] > 0.5f ? 1 : 0;
        if (pred == y[i]) correct++;
    }
    float accuracy = (float)correct / (float)n;
    if (accuracy < 0.7f)
        T_FAIL("accuracy", "%.0f%% accuracy on linearly separable data", accuracy * 100);

    cudaFree(d_X); cudaFree(d_probs);
    free(X); free(y); free(h_probs);
    protein_classifier_destroy(clf);
    T_PASS("classifier (accuracy=%.0f%%)", accuracy * 100);
    return 0;
}

// -- test: sampling -----------------------------------------------------------

static int test_sampling()
{
    printf("[test_sampling]\n");

    int dim = 3, cap = 512;
    float bmin[] = {-1.0f, -1.0f, -1.0f};
    float bmax[] = { 1.0f,  1.0f,  1.0f};
    float scl[]  = { 0.5f,  0.5f,  0.5f};
    ProteinAcq *acq = protein_acq_create(dim, cap, bmin, bmax, scl, 42ULL);

    float centers[] = {0.0f, 0.0f, 0.0f};
    int m = protein_acq_sample(acq, centers, 1, cap,
                                1.0f, -1, NAN, 0.0f, 0);
    if (m != cap) T_FAIL("count", "expected %d, got %d", cap, m);

    float *h_cands = (float *)malloc((size_t)m * dim * sizeof(float));
    cudaMemcpy(h_cands, acq->d_candidates, (size_t)m * dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++)
        for (int d = 0; d < dim; d++) {
            float v = h_cands[i * dim + d];
            if (v < bmin[d] - 1e-5f || v > bmax[d] + 1e-5f)
                T_FAIL("bounds", "candidate[%d][%d]=%.3f out of bounds", i, d, v);
        }

    free(h_cands);
    protein_acq_destroy(acq);
    T_PASS("sampling_bounds");
    return 0;
}

// -- test: scoring kernel -----------------------------------------------------

static int test_scoring()
{
    printf("[test_scoring]\n");

    int dim = 2, cap = 64;
    float bmin[] = {-1.0f, -1.0f}, bmax[] = {1.0f, 1.0f}, scl[] = {0.5f, 0.5f};
    ProteinAcq *acq = protein_acq_create(dim, cap, bmin, bmax, scl, 99ULL);

    // Manually fill prediction buffers with known values
    int m = 4;
    float h_pred_y[] = {0.2f, 0.8f, 0.5f, 0.1f};
    float h_pred_c[] = {0.3f, 0.5f, 0.7f, 0.9f};
    cudaMemcpy(acq->d_pred_y, h_pred_y, (size_t)m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(acq->d_pred_c, h_pred_c, (size_t)m * sizeof(float), cudaMemcpyHostToDevice);

    // Fixed cost mode → pure score maximization
    ProteinAcqResult r = protein_acq_score(acq, m,
        0.0f, 10.0f, 0.0f, 5.0f,
        3600.0f, 0.5f,
        1, 1, NULL, 0);
    if (r.best_idx != 1)
        T_FAIL("fixed_cost", "expected best_idx=1 (highest y_norm), got %d", r.best_idx);
    T_PASS("scoring_fixed_cost");

    // Cost-weighted mode
    // target_cost=0.5, candidate 1 has c_norm=0.5 (exact match) and highest score
    r = protein_acq_score(acq, m,
        0.0f, 10.0f, 0.0f, 5.0f,
        3600.0f, 0.5f,
        1, 0, NULL, 0);
    if (r.best_idx != 1)
        T_FAIL("cost_weighted", "expected best_idx=1, got %d", r.best_idx);
    T_PASS("scoring_cost_weighted");

    // With success_prob that zeros out candidate 1
    float h_sprob[] = {1.0f, 0.0f, 1.0f, 1.0f};
    float *d_sprob;
    cudaMalloc(&d_sprob, (size_t)m * sizeof(float));
    cudaMemcpy(d_sprob, h_sprob, (size_t)m * sizeof(float), cudaMemcpyHostToDevice);
    r = protein_acq_score(acq, m,
        0.0f, 10.0f, 0.0f, 5.0f,
        3600.0f, 0.5f,
        1, 1, d_sprob, 0);
    if (r.best_idx == 1)
        T_FAIL("success_prob", "candidate 1 should be zeroed out");
    cudaFree(d_sprob);
    T_PASS("scoring_success_prob");

    protein_acq_destroy(acq);
    return 0;
}

// -- test: full GP pipeline ---------------------------------------------------

static int test_full_pipeline()
{
    printf("[test_full_pipeline]\n");

    int dim = 2, n_obs = 40, cap = 256;

    // Generate observations: score = x0 + 0.5*x1, cost = exp(2 + x0)
    float *obs_params = (float *)malloc((size_t)n_obs * dim * sizeof(float));
    float *obs_scores = (float *)malloc((size_t)n_obs * sizeof(float));
    float *obs_costs  = (float *)malloc((size_t)n_obs * sizeof(float));
    float *gp_y_train = (float *)malloc((size_t)n_obs * sizeof(float));
    float *gp_c_train = (float *)malloc((size_t)n_obs * sizeof(float));

    srand(7);
    float min_s = FLT_MAX, max_s = -FLT_MAX;
    float log_c_min = FLT_MAX, log_c_max = -FLT_MAX;

    for (int i = 0; i < n_obs; i++) {
        float x0 = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
        float x1 = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
        obs_params[i * 2]     = x0;
        obs_params[i * 2 + 1] = x1;
        obs_scores[i] = x0 + 0.5f * x1;
        obs_costs[i]  = expf(2.0f + x0);

        if (obs_scores[i] < min_s) min_s = obs_scores[i];
        if (obs_scores[i] > max_s) max_s = obs_scores[i];
        float lc = logf(obs_costs[i]);
        if (lc < log_c_min) log_c_min = lc;
        if (lc > log_c_max) log_c_max = lc;
    }

    // Normalize for GP training
    for (int i = 0; i < n_obs; i++) {
        gp_y_train[i] = (obs_scores[i] - min_s) / (max_s - min_s + PROTEIN_EPSILON);
        float lc = logf(obs_costs[i]);
        gp_c_train[i] = (lc - log_c_min) / (log_c_max - log_c_min + PROTEIN_EPSILON);
    }

    // Create and fit GPs
    GPKernel *k_score = gp_kernel_matern32_linear(dim, 1.0f, 1.0f, 1.0f);
    GPKernel *k_cost  = gp_kernel_matern32_linear(dim, 1.0f, 1.0f, 1.0f);
    GaussianProcess *gp_score = gp_create(dim, n_obs + 10, k_score, 1e-2f);
    GaussianProcess *gp_cost  = gp_create(dim, n_obs + 10, k_cost,  1e-2f);

    int rc = gp_fit(gp_score, obs_params, gp_y_train, n_obs, 0);
    if (rc != 0) T_FAIL("gp_score_fit", "rc=%d", rc);
    rc = gp_fit(gp_cost, obs_params, gp_c_train, n_obs, 0);
    if (rc != 0) T_FAIL("gp_cost_fit", "rc=%d", rc);

    // Pareto front
    int *pareto_idx = (int *)malloc((size_t)n_obs * sizeof(int));
    int n_pareto = protein_pareto_front(obs_scores, obs_costs, n_obs, pareto_idx);
    if (n_pareto == 0) T_FAIL("pareto", "empty pareto front");
    printf("  pareto points: %d\n", n_pareto);

    // Build search centers (no top obs for simplicity)
    float *centers = (float *)malloc((size_t)n_pareto * dim * sizeof(float));
    int nc = protein_build_search_centers(obs_params, dim,
                                           pareto_idx, n_pareto,
                                           NULL, 0, -1, centers);

    // Sample + predict + score
    float bmin[] = {-1.0f, -1.0f}, bmax[] = {1.0f, 1.0f}, scl[] = {0.5f, 0.5f};
    ProteinAcq *acq = protein_acq_create(dim, cap, bmin, bmax, scl, 1234ULL);

    int m = protein_acq_sample(acq, centers, nc, nc * 64,
                                1.0f, -1, NAN, PROTEIN_EPSILON, 0);
    if (m == 0) T_FAIL("sample", "no candidates after dedup");
    printf("  candidates after dedup: %d\n", m);

    float target = 0.5f;
    ProteinAcqResult result = protein_acq_suggest(
        acq, m, gp_score, gp_cost,
        min_s, max_s, log_c_min, log_c_max,
        3600.0f, target, 1, 0, NULL, 0);

    printf("  best_idx=%d  score=%.3f  cost=%.3f  rating=%.4f\n",
           result.best_idx, result.predicted_score,
           result.predicted_cost, result.rating);

    float best_params[2];
    protein_acq_get_candidate(acq, result.best_idx, best_params, 0);
    printf("  best params: [%.3f, %.3f]\n", best_params[0], best_params[1]);

    if (result.predicted_score < min_s || result.predicted_score > max_s * 1.5f)
        T_FAIL("score_range", "predicted score %.3f out of reasonable range", result.predicted_score);

    // Cost model
    ProteinCostModel cm;
    protein_cost_model_init(&cm, 0.3f, 10);
    protein_cost_model_fit(&cm, obs_scores, obs_costs, n_obs, obs_costs[0]);
    printf("  cost_model: fitted=%d A=%.3f B=%.3f\n", cm.is_fitted, cm.A, cm.B);

    // Cleanup
    protein_acq_destroy(acq);
    gp_destroy(gp_score);
    gp_destroy(gp_cost);
    free(obs_params); free(obs_scores); free(obs_costs);
    free(gp_y_train); free(gp_c_train);
    free(pareto_idx); free(centers);

    T_PASS("full_pipeline");
    return 0;
}

// -- main ---------------------------------------------------------------------

int main()
{
    srand((unsigned)time(NULL));
    printf("=== Protein CUDA tests ===\n\n");

    int failures = 0;
    failures += test_pareto();
    failures += test_prune();
    failures += test_search_centers();
    failures += test_cost_model();
    failures += test_classifier();
    failures += test_sampling();
    failures += test_scoring();
    failures += test_full_pipeline();

    printf("\n=== %s: %d test(s) failed ===\n",
           failures ? "FAILED" : "ALL PASSED", failures);
    return failures;
}
