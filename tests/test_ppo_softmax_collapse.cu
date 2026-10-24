// Softmax Collapse (arXiv 2610.31642) in the PPO discrete head: the real
// cache_imp_and_v + ppo_loss_compute on rows whose taken action holds nearly all
// the probability, against fp64. That action's logit grad is d_logp * (1 - p),
// ~exp(-margin); forming 1 - p after p rounded to 1 zeroed it while the other
// actions kept theirs, so the row stopped summing to 0. From the repo root
// (raylib fetched by ./build.sh):
//   nvcc -O2 -std=c++17 -arch=native -I. -Isrc -Ivendor -Iraylib-5.5_linux_amd64/include \
//     -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing --diag-suppress=2361,111,128 \
//     tests/test_ppo_softmax_collapse.cu raylib-5.5_linux_amd64/lib/libraylib.a \
//     -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -lGL -lm -lpthread -ldl \
//     -lgomp -o /tmp/test_ppo_softmax_collapse && /tmp/test_ppo_softmax_collapse
#define ENV_HEADER "../ocean/checkers/checkers.h"  // one 512-way head
#define PUFFER_ENV_NAME "checkers"
#include "../src/pufferl.cu"

#include <vector>

__global__ void to_prec(precision_t* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = from_float(src[i]);
    }
}

template <class T> static T* dev(int n) {
    T* p;
    cudaMalloc(&p, n * sizeof(T));
    cudaMemset(p, 0, n * sizeof(T));
    return p;
}

static precision_t* dev_prec(const std::vector<float>& v) {
    float* f = dev<float>((int)v.size());
    cudaMemcpy(f, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    precision_t* p = dev<precision_t>((int)v.size());
    to_prec<<<1, 1024>>>(p, f, (int)v.size());
    return p;
}

// One row with adv = -1 and ent_coef = 0, so the logit grads are
// (1[j == act] - p_j) * ratio; returned divided by ratio.
static std::vector<float> logit_grads(const std::vector<float>& z, int act, float* logp_act) {
    int A = (int)z.size();
    std::vector<float> row(z);
    row.push_back(0.0f);  // fused value column
    precision_t* dec = dev_prec(row);
    precision_t* mask = dev_prec(std::vector<float>(A, 1.0f));
    precision_t* adv = dev_prec({-1.0f});
    precision_t* zero = dev_prec({0.0f});
    precision_t* old_lp = dev<precision_t>(1);
    precision_t* imp = dev<precision_t>(1);
    precision_t* value = dev<precision_t>(1);
    float* actions = dev<float>(1);
    float fact = (float)act;
    cudaMemcpy(actions, &fact, sizeof(float), cudaMemcpyHostToDevice);
    int* act_sizes = dev<int>(1);
    cudaMemcpy(act_sizes, &A, sizeof(int), cudaMemcpyHostToDevice);
    float* grads = dev<float>(A);  // logps from cache_imp_and_v, then logit grads
    float* new_lp = dev<float>(1);
    float* partials = dev<float>(LOSS_N);
    float* ent_coef = dev<float>(1);

    Prec dec_out = {.data = dec, .shape = {1, 1, A + 1}};
    Prec logstd = {};
    cache_imp_and_v<<<1, 32>>>(dec_out, actions, old_lp, mask, logstd, act_sizes,
        imp, value, grads, new_lp);
    cudaMemcpy(logp_act, grads + act, sizeof(float), cudaMemcpyDeviceToHost);
    // Same weights as the rollout: old logprob = new one at storage precision.
    to_prec<<<1, 1>>>(old_lp, new_lp, 1);
    PPOGraphArgs g = {.imp = imp, .actions = actions, .old_logprobs = old_lp,
        .advantages = adv, .values = zero, .returns = zero};
    PPOKernelArgs k = {.grad_logits = grads, .grad_logstd = NULL,
        .grad_values_pred = new_lp, .logits = dec, .logstd = NULL,
        .values_pred = dec + A, .act_sizes = act_sizes, .action_mask = mask,
        .num_atns = 1, .clip_coef = 0.2f, .vf_clip_coef = 0.2f, .vf_coef = 0.5f,
        .ent_coef = ent_coef, .T_seq = 1, .A_total = A, .N = 1, .is_continuous = false};
    ppo_loss_compute<<<1, PPO_THREADS>>>(partials, k, g);
    std::vector<float> out(A);
    cudaMemcpy(out.data(), grads, A * sizeof(float), cudaMemcpyDeviceToHost);
    float losses[LOSS_N];
    cudaMemcpy(losses, partials, sizeof(losses), cudaMemcpyDeviceToHost);
    for (float& x : out) {
        x /= losses[LOSS_IMP];  // ratio (N * T = 1)
    }
    return out;
}

static int check(const char* name, const std::vector<float>& z, int act) {
    int A = (int)z.size(), top = 0;
    for (int j = 1; j < A; j++) {
        if (z[j] > z[top]) {
            top = j;
        }
    }
    double rest = 0.0;
    for (int j = 0; j < A; j++) {
        if (j != top) {
            rest += exp((double)z[j] - z[top]);
        }
    }
    double want_lp = ((double)z[act] - z[top]) - log1p(rest);
    double want_g = -expm1(want_lp);  // 1 - p_act
    float lp;
    std::vector<float> g = logit_grads(z, act, &lp);
    double sum = 0.0;
    for (float x : g) {
        sum += x;
    }
    int ok = fabs(g[act] - want_g) <= 1e-3 * want_g && fabs(sum) <= 1e-3 * want_g
        && fabs(lp - want_lp) <= 1e-3 * fabs(want_lp);
    printf("%-26s grad %.4e (want %.4e)  row sum %+.1e  logp %.4e (want %.4e)  %s\n",
        name, g[act], want_g, sum, lp, want_lp, ok ? "ok" : "FAIL");
    return ok;
}

int main() {
    // Margins: 17.3 (paper's example), 13.3 (below fp32's 16.6, but at |logit| ~ 110),
    // 10.8 (each other p ~ 4e-8 < 2^-24: lost one at a time if summed onto 1).
    std::vector<float> wide(512, 0.0f);
    wide[0] = 17.0f;
    int ok = check("paper [9, -9, -9]", {9, -9, -9}, 0)
        & check("large logits [110, 96, 96]", {110, 96, 96}, 0)
        & check("512 actions, top first", wide, 0)
        & check("non-top action [3, 1, 0]", {3, 1, 0}, 1);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "FAIL %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (!ok) {
        fprintf(stderr, "FAIL softmax collapse\n");
        return 1;
    }
    printf("PPO SOFTMAX COLLAPSE OK\n");
    return 0;
}
