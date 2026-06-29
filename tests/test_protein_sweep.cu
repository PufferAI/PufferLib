// test_protein_sweep.cu -- End-to-end sweep test for protein.cu
// Replicates tests/test_sweep.py.
//
// Build:
//   nvcc -o test_protein_sweep tests/test_protein_sweep.cu -I src/ -lcublas -lcusolver -lcurand

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gp_cuda.cu"
#include "protein.cu"

// -- default search space (21 dims, matches Python ordering) ------------------

#define DIM 21
#define COST_IDX 0 // train/total_timesteps
#define LR_IDX 2 // train/learning_rate

static Hyperparameters* build_default_search_space(void)
{
    Space spaces[DIM];
    //  0: train/total_timesteps  Log    3e7..1e11  scale=1/(log2(1e11)-log2(3e7))
    float ts_scale = 1.0f / (log2f(1e11f) - log2f(3e7f));
    space_init(&spaces[0], SPACE_LOG, 3e7f, 1e11f, ts_scale, 0);
    //  1: train/horizon          Pow2   8..1024    scale=0.5
    space_init(&spaces[1], SPACE_POW2, 8.0f, 1024.0f, 0.5f, 1);
    //  2: train/learning_rate    Log    1e-5..0.1  scale=0.5
    space_init(&spaces[2], SPACE_LOG, 1e-5f, 0.1f, 0.5f, 0);
    //  3: train/ent_coef         Log    1e-5..0.2  scale=0.5
    space_init(&spaces[3], SPACE_LOG, 1e-5f, 0.2f, 0.5f, 0);
    //  4: train/gamma            Logit  0.8..0.9999  scale=0.5
    space_init(&spaces[4], SPACE_LOGIT, 0.8f, 0.9999f, 0.5f, 0);
    //  5: train/gae_lambda       Logit  0.2..0.995   scale=0.5
    space_init(&spaces[5], SPACE_LOGIT, 0.2f, 0.995f, 0.5f, 0);
    //  6: train/vtrace_rho_clip  Linear 0.1..5.0   scale=0.5
    space_init(&spaces[6], SPACE_LINEAR, 0.1f, 5.0f, 0.5f, 0);
    //  7: train/vtrace_c_clip    Linear 0.1..5.0   scale=0.5
    space_init(&spaces[7], SPACE_LINEAR, 0.1f, 5.0f, 0.5f, 0);
    //  8: train/replay_ratio     Linear 0.25..4.0  scale=0.5
    space_init(&spaces[8], SPACE_LINEAR, 0.25f, 4.0f, 0.5f, 0);
    //  9: train/clip_coef        Linear 0.01..1.0  scale=0.5
    space_init(&spaces[9], SPACE_LINEAR, 0.01f, 1.0f, 0.5f, 0);
    // 10: train/vf_clip_coef     Linear 0.01..5.0  scale=0.5
    space_init(&spaces[10], SPACE_LINEAR, 0.01f, 5.0f, 0.5f, 0);
    // 11: train/vf_coef          Linear 0.1..5.0   scale=0.5
    space_init(&spaces[11], SPACE_LINEAR, 0.1f, 5.0f, 0.5f, 0);
    // 12: train/max_grad_norm    Linear 0.1..5.0   scale=0.5
    space_init(&spaces[12], SPACE_LINEAR, 0.1f, 5.0f, 0.5f, 0);
    // 13: train/beta1            Logit  0.5..0.999   scale=0.5
    space_init(&spaces[13], SPACE_LOGIT, 0.5f, 0.999f, 0.5f, 0);
    // 14: train/beta2            Logit  0.9..0.99999 scale=0.5
    space_init(&spaces[14], SPACE_LOGIT, 0.9f, 0.99999f, 0.5f, 0);
    // 15: train/eps              Log    1e-14..1e-4  scale=0.5
    space_init(&spaces[15], SPACE_LOG, 1e-14f, 1e-4f, 0.5f, 0);
    // 16: train/prio_alpha       Linear 0.0..1.0   scale=0.5
    space_init(&spaces[16], SPACE_LINEAR, 0.0f, 1.0f, 0.5f, 0);
    // 17: train/prio_beta0       Linear 0.0..1.0   scale=0.5
    space_init(&spaces[17], SPACE_LINEAR, 0.0f, 1.0f, 0.5f, 0);
    // 18: policy/hidden_size     Pow2   32..1024   scale=0.5
    space_init(&spaces[18], SPACE_POW2, 32.0f, 1024.0f, 0.5f, 1);
    // 19: policy/num_layers      Linear 1..8       scale=0.5
    space_init(&spaces[19], SPACE_LINEAR, 1.0f, 8.0f, 0.5f, 0);
    // 20: vec/num_buffers        Linear 1..8       scale=0.5
    space_init(&spaces[20], SPACE_LINEAR, 1.0f, 8.0f, 0.5f, 0);

    return hyperparameters_create(spaces, DIM, COST_IDX, 1);
}

static void synthetic_linear(float learning_rate, float total_timesteps,
    float* out_score, float* out_cost)
{
    float basic = expf(-powf(log10f(learning_rate) + 3.0f, 2.0f));
    *out_cost = total_timesteps / 5e7f;
    *out_score = basic * (*out_cost);
}

// -- helpers ------------------------------------------------------------------

static float randn(void)
{
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// -- plot output --------------------------------------------------------------

static int cmp_float_pair(const void* a, const void* b)
{
    float va = ((const float*)a)[0], vb = ((const float*)b)[0];
    return (va > vb) - (va < vb);
}

static void write_plot(const char* path, const float* scores,
    const float* costs, int n)
{
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return;
    }

    static const char* turbo[] = { "#30123b", "#4662d7", "#36aaf9", "#1ae4b6",
        "#72fe5e", "#c8ef34", "#faba39", "#f6511d" };

    float max_s = scores[0], sum_c = 0;
    for (int i = 0; i < n; i++) {
        if (scores[i] > max_s)
            max_s = scores[i];
        sum_c += costs[i];
    }

    float(*sorted)[2] = (float(*)[2])malloc((size_t)n * 2 * sizeof(float));
    for (int i = 0; i < n; i++) {
        sorted[i][0] = costs[i];
        sorted[i][1] = (float)i;
    }
    qsort(sorted, n, 2 * sizeof(float), cmp_float_pair);

    float* aoc_x = (float*)malloc((size_t)n * sizeof(float));
    float* aoc_y = (float*)malloc((size_t)n * sizeof(float));
    float cumsum = 0;
    for (int i = 0; i < n; i++) {
        cumsum += sorted[i][0];
        aoc_x[i] = sorted[i][0];
        aoc_y[i] = max_s * cumsum / sum_c;
    }

    float x_min = costs[0], x_max = costs[0];
    float y_min = scores[0], y_max = scores[0];
    for (int i = 1; i < n; i++) {
        if (costs[i] < x_min)
            x_min = costs[i];
        if (costs[i] > x_max)
            x_max = costs[i];
        if (scores[i] < y_min)
            y_min = scores[i];
        if (scores[i] > y_max)
            y_max = scores[i];
    }
    for (int i = 0; i < n; i++)
        if (aoc_y[i] > y_max)
            y_max = aoc_y[i];

    int W = 800, H = 500, M = 60;
    int pw = W - 2 * M, ph = H - 2 * M;

#define PX(v) (M + (int)(((v)-x_min) / (x_max - x_min) * pw))
#define PY(v) (M + ph - (int)(((v)-y_min) / (y_max - y_min) * ph))

    fprintf(f,
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Protein Sweep</title></head><body style='background:#222'>\n"
        "<svg width='%d' height='%d' xmlns='http://www.w3.org/2000/svg'>\n"
        "<rect width='100%%' height='100%%' fill='#FFF'/>\n",
        W, H);

    fprintf(f, "<g stroke='#444' stroke-width='1'>\n");
    int n_ticks = 5;
    for (int i = 0; i <= n_ticks; i++) {
        float xv = x_min + (x_max - x_min) * i / n_ticks;
        float yv = y_min + (y_max - y_min) * i / n_ticks;
        fprintf(f, "<line x1='%d' y1='%d' x2='%d' y2='%d'/>\n", PX(xv), M, PX(xv),
            M + ph);
        fprintf(f, "<line x1='%d' y1='%d' x2='%d' y2='%d'/>\n", M, PY(yv), M + pw,
            PY(yv));
        fprintf(f,
            "<text x='%d' y='%d' fill='#aaa' font-size='11' "
            "text-anchor='middle'>%.0f</text>\n",
            PX(xv), M + ph + 16, xv);
        fprintf(f,
            "<text x='%d' y='%d' fill='#aaa' font-size='11' "
            "text-anchor='end' dominant-baseline='middle'>%.0f</text>\n",
            M - 6, PY(yv), yv);
    }
    fprintf(f, "</g>\n");

    fprintf(f,
        "<text x='%d' y='%d' fill='#ccc' font-size='13' "
        "text-anchor='middle'>Cost</text>\n",
        W / 2, H - 8);
    fprintf(f,
        "<text x='14' y='%d' fill='#ccc' font-size='13' "
        "text-anchor='middle' transform='rotate(-90,14,%d)'>Score</text>\n",
        H / 2, H / 2);
    fprintf(f,
        "<text x='%d' y='20' fill='#eee' font-size='15' "
        "text-anchor='middle'>Protein Sweep (synthetic linear, 21D)</text>\n",
        W / 2);

    fprintf(f,
        "<polyline fill='none' stroke='#b06cff' stroke-width='2' points='");
    for (int i = 0; i < n; i++)
        fprintf(f, "%d,%d ", PX(aoc_x[i]), PY(aoc_y[i]));
    fprintf(f, "'/>\n");

    for (int i = 0; i < n; i++) {
        int ci = (int)((float)i / fmaxf((float)(n - 1), 1.0f) * 7.0f);
        if (ci > 7)
            ci = 7;
        fprintf(f, "<circle cx='%d' cy='%d' r='3' fill='%s' opacity='0.75'/>\n",
            PX(costs[i]), PY(scores[i]), turbo[ci]);
    }

    fprintf(f, "</svg></body></html>\n");

#undef PX
#undef PY

    free(sorted);
    free(aoc_x);
    free(aoc_y);
    fclose(f);
    printf("  Plot written to %s\n", path);
}

// -- main sweep loop ----------------------------------------------------------

int main(int argc, char** argv)
{
    setbuf(stdout, NULL);
    srand((unsigned)time(NULL));
    const char* plot_base = (argc > 1) ? argv[1] : "sweep_plot";
    char plot_path[512], csv_path[512];
    snprintf(plot_path, sizeof(plot_path), "%s.html", plot_base);
    snprintf(csv_path, sizeof(csv_path), "%s.csv", plot_base);
    printf("=== Protein Sweep Test (synthetic linear, 21D) ===\n");
    printf("  Plot output: %s\n\n", plot_path);

    Hyperparameters* hp = build_default_search_space();

    int num_runs = 200;
    int num_random_samples = 10;
    int downsample = 5;
    int max_obs = num_runs * downsample + 10;
    int gp_max_obs = 750;
    int n_candidates = 4096;
    int suggestions_per_pareto = 256;
    int gp_train_iter = 50;
    float gp_lr = 0.001f;
    float expansion_rate = 1.0f;
    int top_k = 5;
    float max_suggestion_cost = 3600.0f;
    int optimizer_reset_freq = 50;
    float cost_random_suggestion = -0.8f;
    float early_stop_quantile = 0.3f;

    ProteinObs* obs = protein_obs_create(DIM, max_obs, max_obs / 10, top_k,
        COST_IDX);

    GPKernel* ks = gp_kernel_matern32_linear(DIM, 1.0f, 1.0f, 1.0f);
    GPKernel* kc = gp_kernel_matern32_linear(DIM, 1.0f, 1.0f, 1.0f);
    GaussianProcess* gp_s = gp_create(DIM, gp_max_obs, ks, 1e-2f);
    GaussianProcess* gp_c = gp_create(DIM, gp_max_obs, kc, 1e-2f);
    Adam* opt_s = adam_create(ks->n_params, gp_lr);
    Adam* opt_c = adam_create(kc->n_params, gp_lr);
    ProteinAcq* acq = protein_acq_create(DIM, n_candidates, hp->bounds_min,
        hp->bounds_max, hp->scales, 42ULL);
    Sobol* sobol = sobol_create(DIM, 73);

    ProteinCostModel cost_model;
    protein_cost_model_init(&cost_model, early_stop_quantile, 30);
    float upper_cost_threshold = -FLT_MAX;

    float ratio_pool[] = { 0.16f, 0.32f, 0.48f, 0.64f, 0.80f, 1.0f };
    int ratio_remaining = 0;

    float* gp_params = (float*)malloc((size_t)gp_max_obs * DIM * sizeof(float));
    float* gp_scores = (float*)malloc((size_t)gp_max_obs * sizeof(float));
    float* gp_costs = (float*)malloc((size_t)gp_max_obs * sizeof(float));
    float* y_norm = (float*)malloc((size_t)gp_max_obs * sizeof(float));
    float* c_norm = (float*)malloc((size_t)gp_max_obs * sizeof(float));

    float* run_scores = (float*)malloc((size_t)num_runs * sizeof(float));
    float* run_costs = (float*)malloc((size_t)num_runs * sizeof(float));
    float suggestion[DIM];
    float sobol_buf[DIM];
    int suggestion_idx = 0;

    float random_best = -FLT_MAX, guided_best = -FLT_MAX;
    float random_sum = 0.0f, guided_sum = 0.0f;
    int random_count = 0, guided_count = 0;

    for (int run = 0; run < num_runs; run++) {
        suggestion_idx++;
        int is_random = (suggestion_idx <= num_random_samples);

        if (is_random) {
            sobol_next(sobol, sobol_buf);
            for (int d = 0; d < DIM; d++)
                suggestion[d] = 2.0f * sobol_buf[d] - 1.0f;
            float cost_s = cost_random_suggestion + 0.1f * randn();
            suggestion[COST_IDX] = fmaxf(-1.0f, fminf(1.0f, cost_s));
        } else {
            int n_gp = protein_obs_sample_for_gp(obs, gp_max_obs, 0.5f,
                gp_params, gp_scores, gp_costs);
            float min_s = obs->min_score, max_s = obs->max_score;
            float lc_min = obs->log_c_min, lc_max = obs->log_c_max;

            float s_range = fabsf(max_s - min_s) + PROTEIN_EPSILON;
            float lc_range = fabsf(lc_max - lc_min) + PROTEIN_EPSILON;
            for (int i = 0; i < n_gp; i++) {
                y_norm[i] = (gp_scores[i] - min_s) / s_range;
                float lc = logf(fmaxf(gp_costs[i], PROTEIN_EPSILON));
                c_norm[i] = (lc - lc_min) / lc_range;
            }

            float loss_s = protein_train_gp(gp_s, opt_s, gp_params, y_norm, n_gp,
                gp_train_iter, 0);
            float loss_c = protein_train_gp(gp_c, opt_c, gp_params, c_norm, n_gp,
                gp_train_iter, 0);

            if (optimizer_reset_freq > 0 && suggestion_idx % optimizer_reset_freq == 0) {
                adam_reset(opt_s);
                adam_reset(opt_c);
                printf("  [%4d] reset Adam optimizers\n", run);
            }

            ProteinObsList* suc = &obs->success;
            int* pidx = (int*)malloc((size_t)suc->n * sizeof(int));
            int np = protein_pareto_front(suc->scores, suc->costs, suc->n, pidx);
            np = protein_prune_pareto(suc->scores, suc->costs, pidx, np,
                0.5f, 0.98f);

            if (np > 0) {
                float pruned_max_cost = suc->costs[pidx[np - 1]];
                if (upper_cost_threshold < 0)
                    upper_cost_threshold = pruned_max_cost;
                else if (upper_cost_threshold < pruned_max_cost)
                    upper_cost_threshold *= 1.01f;
            }
            protein_cost_model_fit(&cost_model, suc->scores, suc->costs, suc->n,
                upper_cost_threshold);

            int max_centers = np + 2 * obs->n_top;
            float* centers = (float*)malloc((size_t)max_centers * DIM * sizeof(float));
            int nc = protein_build_search_centers(suc->params, DIM, pidx, np,
                obs->top_idx, obs->n_top, COST_IDX, centers);

            float target = protein_sample_target_cost(ratio_pool, &ratio_remaining,
                PROTEIN_NUM_COST_RATIOS, expansion_rate);

            int m = protein_acq_sample(acq, centers, nc, nc * suggestions_per_pareto,
                1.0f, -1, NAN, PROTEIN_EPSILON, 0);
            if (m == 0) {
                sobol_next(sobol, sobol_buf);
                for (int d = 0; d < DIM; d++)
                    suggestion[d] = 2.0f * sobol_buf[d] - 1.0f;
            } else {
                ProteinAcqResult r = protein_acq_suggest(
                    acq, m, gp_s, gp_c, min_s, max_s, lc_min, lc_max,
                    max_suggestion_cost, target, 1, 0, NULL, 0);

                protein_acq_get_candidate(acq, r.best_idx, suggestion, 0);

                if (run % 50 == 0 || run == num_runs - 1) {
                    printf("  [%4d] loss_s=%.3f loss_c=%.3f pareto=%d "
                           "gp_obs=%d cands=%d rating=%.4f\n",
                        run, loss_s, loss_c, np, n_gp, m, r.rating);
                }
            }

            free(pidx);
            free(centers);
        }

        float real_lr = space_unnormalize(&hp->spaces[LR_IDX], suggestion[LR_IDX]);
        float real_ts = space_unnormalize(&hp->spaces[COST_IDX], suggestion[COST_IDX]);

        float final_score = 0.0f, final_cost = 0.0f;

        for (int ds = 1; ds <= downsample; ds++) {
            float frac_ts = real_ts * (float)ds / (float)downsample;
            float score, cost;
            synthetic_linear(real_lr, frac_ts, &score, &cost);

            float obs_p[DIM];
            memcpy(obs_p, suggestion, (size_t)DIM * sizeof(float));
            obs_p[COST_IDX] = space_normalize(&hp->spaces[COST_IDX], frac_ts);
            protein_obs_add(obs, obs_p, score, cost, 0);

            if (ds == downsample) {
                final_score = score;
                final_cost = cost;
            }
        }

        run_scores[run] = final_score;
        run_costs[run] = final_cost;

        if (run < num_random_samples || run % 50 == 0 || run == num_runs - 1)
            printf("  Run %4d (%s): lr=%.2e ts=%.2e "
                   "score=%.4f cost=%.2f  [obs=%d]\n",
                run, is_random ? "rand" : "  GP", real_lr, real_ts, final_score,
                final_cost, obs->success.n);

        if (is_random) {
            if (final_score > random_best)
                random_best = final_score;
            random_sum += final_score;
            random_count++;
        } else {
            if (final_score > guided_best)
                guided_best = final_score;
            guided_sum += final_score;
            guided_count++;
        }
    }

    float random_avg = random_sum / fmaxf((float)random_count, 1.0f);
    float guided_avg = guided_sum / fmaxf((float)guided_count, 1.0f);

    printf("\n=== Results ===\n");
    printf("  Random: best=%.4f avg=%.4f (%d runs)\n", random_best, random_avg,
        random_count);
    printf("  GP-guided: best=%.4f avg=%.4f (%d runs)\n", guided_best, guided_avg,
        guided_count);
    printf("  Total observations: %d (success=%d failure=%d)\n",
        obs->success.n + obs->failure.n, obs->success.n, obs->failure.n);

    {
        FILE* csv = fopen(csv_path, "w");
        if (csv) {
            fprintf(csv, "run,score,cost\n");
            for (int i = 0; i < num_runs; i++)
                fprintf(csv, "%d,%.6f,%.6f\n", i, run_scores[i], run_costs[i]);
            fclose(csv);
            printf("  Data written to %s\n", csv_path);
        }
    }

    write_plot(plot_path, run_scores, run_costs, num_runs);

    int pass = (guided_best >= random_best * 0.9f);
    printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");

    free(run_scores);
    free(run_costs);
    free(gp_params);
    free(gp_scores);
    free(gp_costs);
    free(y_norm);
    free(c_norm);
    sobol_destroy(sobol);
    protein_acq_destroy(acq);
    adam_destroy(opt_s);
    adam_destroy(opt_c);
    gp_destroy(gp_s);
    gp_destroy(gp_c);
    protein_obs_destroy(obs);
    hyperparameters_destroy(hp);

    return pass ? 0 : 1;
}
