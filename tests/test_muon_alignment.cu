// Check that padding between tensors does not change Muon updates or lose weights.
// Build from the repo root (Linux/GCC):
//   mkdir -p build
//   nvcc -O2 -arch=native -std=c++17 -DPRECISION_FLOAT -DTEST_ACTIONS=5 \
//       -I. -Isrc -Ivendor -Iraylib-5.5_linux_amd64/include \
//       -DPLATFORM_DESKTOP -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing \
//       --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 \
//       tests/test_muon_alignment.cu raylib-5.5_linux_amd64/lib/libraylib.a \
//       -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
//       -lGL -lm -lpthread -ldl -lgomp -o build/test_muon_alignment
//   ./build/test_muon_alignment 0
//   ./build/test_muon_alignment 1
// Add the CUDA/NCCL include/library paths used by build.sh if not in default paths.
// Also test without PRECISION_FLOAT and with TEST_ACTIONS=8 (no padding).
// Argument 1 checks the separate actor buffer. Keep assertions enabled.
typedef float obs_t;
#include "pufferenv.h"

#ifndef TEST_ACTIONS
#define TEST_ACTIONS 5
#endif
#if TEST_ACTIONS == 5
#define ACT_SIZES {1, 1, 1, 1, 1}
#elif TEST_ACTIONS == 8
#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1, 1}
#else
#error "Build with TEST_ACTIONS=5 (padded) or 8 (aligned control)"
#endif
#define NUM_ATNS TEST_ACTIONS
#define OBS_SIZE 14
#define ENV_HEADER "pufferenv.h"
#define PUFFER_ENV_NAME "muon_alignment"

struct Log { float perf, n; };
struct Env {
    Log log;
    Agent agents[1];
    int tag, boundary_reached, num_agents;
    unsigned int rng;
};
void puf_init(Env* env, Dict* kwargs) { env->num_agents = 1; }
void puf_reset(Env* env) {
    memset(env->agents[0].observations, 0, OBS_SIZE * sizeof(obs_t));
}
void puf_step(Env* env) {}
void puf_render(Env* env) {}
void puf_close(Env* env) {}
void puf_log(Log* log, Dict* out) { dict_set(out, "perf", log->perf); }

#include "../src/pufferl.cu"

struct Reference {
    Allocator params, scratch;
    Prec param, grad;
    Float weights;
    Muon muon;
};

int main(int argc, char** argv) {
    assert(argc == 2 && "usage: test_muon_alignment 0|1 (sync|async allocation)");
    assert(strcmp(argv[1], "0") == 0 || strcmp(argv[1], "1") == 0);
    Ini ini = {};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_put(&ini, "base.async", argv[1]);
    puf_ini_put(&ini, "base.cudagraphs", "-1");
    puf_ini_put(&ini, "vec.total_agents", "4");
    puf_ini_put(&ini, "vec.num_buffers", "1");
    puf_ini_put(&ini, "vec.num_threads", "1");
    puf_ini_put(&ini, "policy.hidden_size", "32");
    puf_ini_put(&ini, "policy.num_layers", "1");
    puf_ini_put(&ini, "train.horizon", "4");
    puf_ini_put(&ini, "train.minibatch_size", "16");
    TrainContext ctx = {.rank = 0, .world_size = 1, .gpu_id = 0, .artifact_owner = 1};
    PuffeRL* p = create_pufferl(&ini, &ctx);
    Policy* pol = &p->policies[0];
    Allocator* params = &pol->params_alloc;
    long n = params->total_bytes / sizeof(precision_t);
    assert((n > params->total_elems) == (TEST_ACTIONS == 5));
    assert(numel(pol->param.shape) == n && "parameter view omits alignment padding");
    assert(numel(pol->master_weights.shape) == n && "master weights omit padding");
    assert(numel(p->grad.shape) == n && "gradient view omits alignment padding");
    assert(numel(p->muon.mb.shape) == n && "momentum buffer omits alignment padding");
    assert(params->total_bytes == p->grads_alloc.total_bytes);
    if (p->hypers.async) {
        assert(numel(p->actor_param.shape) == n && "actor view omits padding");
    }

    // Compare with separate tensors, each starting at offset zero.
    // Disable gradient clipping so tensors can be tested independently.
    Reference* refs = (Reference*)calloc(params->num_regs, sizeof(Reference));
    float* initial = (float*)malloc(n * sizeof(float));
    assert(cudaMemcpy(initial, pol->master_weights.data, n * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess);
    float lr = 0.003f;
    cudaMemcpy(p->muon.lr, &lr, sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < params->num_regs; i++) {
        AllocEntry& e = params->regs[i];
        long offset = ((char*)*e.data_ptr - (char*)params->mem) / sizeof(precision_t);
        long goffset = ((char*)*p->grads_alloc.regs[i].data_ptr
            - (char*)p->grads_alloc.mem) / sizeof(precision_t);
        assert(offset == goffset);
        Reference* r = &refs[i];
        memcpy(r->param.shape, e.shape, sizeof(r->param.shape));
        alloc_register(&r->params, &r->param);
        muon_init(&r->muon, &r->params, p->muon.momentum, &r->scratch);
        alloc_create(&r->params);
        alloc_create(&r->scratch);
        long count = numel(e.shape);
        r->grad.shape[0] = count;
        r->weights.shape[0] = count;
        cudaMalloc(&r->grad.data, count * sizeof(precision_t));
        cudaMalloc(&r->weights.data, count * sizeof(float));
        cudaMemcpy(r->weights.data, initial + offset,
            count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(r->muon.lr, &lr, sizeof(float), cudaMemcpyHostToDevice);
    }

    precision_t* gradients = (precision_t*)calloc(n, sizeof(precision_t));
    float* actual = (float*)malloc(n * sizeof(float));
    float* expected = (float*)malloc(n * sizeof(float));
    float max_error = 0;
    for (int step = 0; step < 3; step++) {
        // Leave padding at zero.
        for (int i = 0; i < params->num_regs; i++) {
            AllocEntry& e = params->regs[i];
            long offset = ((char*)*e.data_ptr - (char*)params->mem) / sizeof(precision_t);
            long count = numel(e.shape);
            for (long j = 0; j < count; j++) {
                gradients[offset + j] = from_float(
                    0.1f * sinf((j + 1) * (i + 1) * 0.37f + step * 0.2f));
            }
            cudaMemcpy(refs[i].grad.data, gradients + offset,
                count * sizeof(precision_t), cudaMemcpyHostToDevice);
            muon_step(&refs[i].muon, refs[i].weights, refs[i].grad, 1e6f);
        }
        cudaMemcpy(p->grad.data, gradients, n * sizeof(precision_t), cudaMemcpyHostToDevice);
        muon_step(&p->muon, pol->master_weights, p->grad, 1e6f);
        assert(cudaDeviceSynchronize() == cudaSuccess);
        cudaMemcpy(actual, pol->master_weights.data, n * sizeof(float), cudaMemcpyDeviceToHost);
        memcpy(expected, initial, n * sizeof(float));
        for (int i = 0; i < params->num_regs; i++) {
            AllocEntry& e = params->regs[i];
            long offset = ((char*)*e.data_ptr - (char*)params->mem) / sizeof(precision_t);
            cudaMemcpy(expected + offset, refs[i].weights.data,
                numel(e.shape) * sizeof(float), cudaMemcpyDeviceToHost);
        }
        for (long j = 0; j < n; j++) {
            assert(isfinite(actual[j]));
            max_error = fmaxf(max_error, fabsf(actual[j] - expected[j]));
        }
        printf("update %d: max error %.9g\n", step + 1, max_error);
        assert(max_error < 1e-6f && "packed Muon differs from independent tensor updates");
    }

    memcpy(expected, actual, n * sizeof(float));

    // Saving and loading must preserve all weights, including the last ones.
    char path[] = "/tmp/puffer-muon-alignment-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    puf_save_weights(p, path);
    struct stat st;
    assert(stat(path, &st) == 0 && st.st_size == n * (long)sizeof(float));
    cudaMemset(pol->master_weights.data, 0, n * sizeof(float));
    cudaMemset(pol->param.data, 0, params->total_bytes);
    pufferl_load_policy(p, 0, path);
    cudaMemcpy(actual, pol->master_weights.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    assert(memcmp(actual, expected, n * sizeof(float)) == 0);
    precision_t* restored = (precision_t*)malloc(params->total_bytes);
    cudaMemcpy(restored, pol->param.data, params->total_bytes, cudaMemcpyDeviceToHost);
    for (long j = 0; j < n; j++) {
        assert(to_float(restored[j]) == to_float(from_float(expected[j])));
    }
    if (p->hypers.async) {
        cudaMemset(p->actor_param.data, 0, p->weight_alloc.total_bytes);
        puf_copy(&p->actor_param, &pol->param, p->default_stream);
        assert(cudaDeviceSynchronize() == cudaSuccess);
        cudaMemcpy(gradients, p->actor_param.data, p->weight_alloc.total_bytes,
            cudaMemcpyDeviceToHost);
        assert(memcmp(gradients, restored, params->total_bytes) == 0);
    }
    assert(unlink(path) == 0);
    close_pufferl(p);
    printf("PASS: %s, %d actions, async=%s, %ld logical / %ld storage elements\n",
        USE_BF16 ? "bf16" : "float32", TEST_ACTIONS, argv[1], params->total_elems, n);
    return 0;
}
