#ifndef PUFFERLIB_PUF_TYPES_H
#define PUFFERLIB_PUF_TYPES_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tensor.h"

inline int puf_ndim(const int64_t* shape) {
    int n = 0;
    while (n < PUF_MAX_DIMS && shape[n] != 0) n++;
    return n;
}

inline int64_t puf_numel(const int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) n *= shape[i];
    return n;
}

inline int64_t puf_batch_size(const int64_t* shape) {
    int n = puf_ndim(shape);
    int64_t b = 1;
    for (int i = 0; i < n - 2; i++) b *= shape[i];
    return b;
}

using std::vector;

#define PUF_HD
typedef void *cudaStream_t;
#define CUDA_STREAM_T_DEFINED

constexpr bool USE_BF16 = false;

struct PufTensor {
  char *bytes = nullptr;
  int64_t shape[PUF_MAX_DIMS] = {};
  int dtype_size = 0;

  PUF_HD int ndim() const {
    return puf_ndim(shape);
  }

  PUF_HD int64_t numel() const {
    return puf_numel(shape);
  }

  PufTensor squeeze(int dim) {
    int n = ndim();
    shape[dim + 1] *= shape[dim];
    for (int i = dim; i < n - 1; i++)
      shape[i] = shape[i + 1];
    shape[n - 1] = 0;
    return *this;
  }

  PufTensor unsqueeze(int dim, int64_t d0, int64_t d1) {
    assert(d0 * d1 == shape[dim] && "unsqueeze: d0 * d1 must equal shape[dim]");
    int n = ndim();
    for (int i = n; i > dim; i--)
      shape[i] = shape[i - 1];
    shape[dim] = d0;
    shape[dim + 1] = d1;
    return *this;
  }

  int64_t batch_size() const {
    return puf_batch_size(shape);
  }

  const char *dtype_name() const {
    switch (dtype_size) {
    case 1:
      return "i8";
    case 2:
      return "f16";
    case 4:
      return "f32";
    case 8:
      return "f64";
    default:
      return "?";
    }
  }

  const char *repr() const {
    static char buf[256];
    if (!bytes) {
      snprintf(buf, sizeof(buf), "PufTensor(empty)");
      return buf;
    }
    int pos = snprintf(buf, sizeof(buf), "PufTensor(%s, [", dtype_name());
    for (int i = 0; i < ndim() && pos < (int)sizeof(buf) - 32; i++) {
      pos += snprintf(buf + pos, sizeof(buf) - pos, "%s%lld", i ? ", " : "",
                      (long long)shape[i]);
    }
    snprintf(buf + pos, sizeof(buf) - pos, "], %lld elems)",
             (long long)puf_numel(shape));
    return buf;
  }
};

enum LossIdx {
  LOSS_PG = 0,
  LOSS_VF = 1,
  LOSS_ENT = 2,
  LOSS_TOTAL = 3,
  LOSS_OLD_APPROX_KL = 4,
  LOSS_APPROX_KL = 5,
  LOSS_CLIPFRAC = 6,
  LOSS_N = 7,
  NUM_LOSSES = 8,
};

struct PrefixScan {
  void *combined_ptr = nullptr;
  void *state_ptr = nullptr;
  void *input_ptr = nullptr;
  int B = 0, T = 0, H = 0;
  FloatTensor a_star, s_vals, log_values_buf;
  PrecisionTensor out, next_state;
  PrecisionTensor grad_combined, grad_state, grad_input;
};

struct AllocEntry {
  void **data_ptr;
  int64_t *shape;
  int elem_size;
};

struct Allocator {
  std::vector<AllocEntry> regs;
  void *mem = nullptr;
  int64_t total_elems = 0;

  void create() {
    int64_t total_bytes = 0;
    total_elems = 0;

    for (auto &e : regs) {
      total_bytes = (total_bytes + 15) & ~15;
      int64_t n = puf_numel(e.shape);
      total_bytes += n * e.elem_size;
      total_elems += n;
    }

    if (total_bytes > 0) {
#ifdef __CUDACC__
      cudaMalloc(&mem, total_bytes);
      cudaMemset(mem, 0, total_bytes);
#elif defined(WITH_METAL)
      posix_memalign(&mem, 16384, total_bytes);
      memset(mem, 0, total_bytes);
#else
      mem = calloc(1, total_bytes);
#endif
      int64_t offset = 0;
      for (auto &e : regs) {
        offset = (offset + 15) & ~15;
        *e.data_ptr = (char *)mem + offset;
        offset += puf_numel(e.shape) * e.elem_size;
      }
    }
  }

  void destroy() {
#ifdef __CUDACC__
    if (mem) {
      cudaFree(mem);
      mem = nullptr;
    }
#else
    if (mem) {
      free(mem);
      mem = nullptr;
    }
#endif
  }
};

inline void alloc_register(Allocator *a, FloatTensor *t) {
  a->regs.push_back({(void **)&t->data, t->shape, (int)sizeof(float)});
}
inline void alloc_register(Allocator *a, PrecisionTensor *t) {
  assert((t->dtype_size == 2 || t->dtype_size == 4) &&
         "alloc_register: unsupported precision tensor dtype");
  a->regs.push_back({(void **)&t->data, t->shape, t->dtype_size});
}
inline void alloc_register(Allocator *a, IntTensor *t) {
  a->regs.push_back({(void **)&t->data, t->shape, (int)sizeof(int)});
}
inline void alloc_register(Allocator *a, LongTensor *t) {
  a->regs.push_back({(void **)&t->data, t->shape, (int)sizeof(long)});
}
inline void alloc_register(Allocator *a, PufTensor *t) {
  a->regs.push_back({(void **)&t->bytes, t->shape, t->dtype_size});
}

struct AllocSet {
  Allocator params, grads, acts;
  int esz = 0;
  void create() {
    params.create();
    grads.create();
    acts.create();
  }
  void destroy() {
    params.destroy();
    grads.destroy();
    acts.destroy();
  }
};

struct PrioBuffers {
  FloatTensor prio_probs, cdf, mb_prio;
  LongTensor idx;
};

inline void register_prio_buffers(PrioBuffers &bufs, Allocator &alloc, int S,
                                  int minibatch_segments) {
  bufs = (PrioBuffers){
      .prio_probs = {.shape = {S}},
      .cdf = {.shape = {S}},
      .mb_prio = {.shape = {minibatch_segments, 1}},
      .idx = {.shape = {minibatch_segments}},
  };
  alloc_register(&alloc, &bufs.prio_probs);
  alloc_register(&alloc, &bufs.cdf);
  alloc_register(&alloc, &bufs.mb_prio);
  alloc_register(&alloc, &bufs.idx);
}

struct PPOBuffersPuf {
  FloatTensor loss_output;
  FloatTensor grad_logits, grad_values, grad_logstd, adv_scratch;
};

inline void register_ppo_buffers(PPOBuffersPuf &bufs, Allocator &alloc, int N,
                                 int T, int A_total, bool is_continuous) {
  bufs = (PPOBuffersPuf){
      .loss_output = {.shape = {1}},
      .grad_logits = {.shape = {N, T, A_total}},
      .grad_values = {.shape = {N, T, 1}},
      .grad_logstd = {.shape = {N, T, A_total}},
      .adv_scratch = {.shape = {2}},
  };
  alloc_register(&alloc, &bufs.loss_output);
  alloc_register(&alloc, &bufs.grad_logits);
  alloc_register(&alloc, &bufs.grad_values);
  if (is_continuous)
    alloc_register(&alloc, &bufs.grad_logstd);
  alloc_register(&alloc, &bufs.adv_scratch);
}

struct RolloutBuf {
  FloatTensor observations;
  FloatTensor actions;
  FloatTensor values;
  FloatTensor logprobs;
  FloatTensor rewards;
  FloatTensor terminals;
  FloatTensor ratio;
  FloatTensor importance;
};

inline void register_rollout_buffers(RolloutBuf &bufs, Allocator &alloc, int H,
                                     int S, int input_size, int num_atns) {
  bufs.observations = {.shape = {H, S, input_size}};
  bufs.actions = {.shape = {H, S, num_atns}};
  bufs.values = {.shape = {H, S}};
  bufs.logprobs = {.shape = {H, S}};
  bufs.rewards = {.shape = {H, S}};
  bufs.terminals = {.shape = {H, S}};
  bufs.ratio = {.shape = {H, S}};
  bufs.importance = {.shape = {H, S}};
  alloc_register(&alloc, &bufs.observations);
  alloc_register(&alloc, &bufs.actions);
  alloc_register(&alloc, &bufs.values);
  alloc_register(&alloc, &bufs.logprobs);
  alloc_register(&alloc, &bufs.rewards);
  alloc_register(&alloc, &bufs.terminals);
  alloc_register(&alloc, &bufs.ratio);
  alloc_register(&alloc, &bufs.importance);
}

struct TrainGraph {
  FloatTensor mb_obs;
  FloatTensor mb_state;
  FloatTensor mb_actions;
  FloatTensor mb_logprobs;
  FloatTensor mb_advantages;
  FloatTensor mb_prio;
  FloatTensor mb_values;
  FloatTensor mb_returns;
  FloatTensor mb_ratio;
  FloatTensor mb_newvalue;
};

inline void register_train_buffers(TrainGraph &bufs, Allocator &alloc, int S,
                                   int H, int input_size, int hidden_size,
                                   int num_atns, int num_layers) {
  bufs.mb_obs = {.shape = {S, H, input_size}};
  bufs.mb_state = {.shape = {num_layers, S, 1, hidden_size}};
  bufs.mb_actions = {.shape = {S, H, num_atns}};
  bufs.mb_logprobs = {.shape = {S, H}};
  bufs.mb_advantages = {.shape = {S, H}};
  bufs.mb_prio = {.shape = {S, 1}};
  bufs.mb_values = {.shape = {S, H}};
  bufs.mb_returns = {.shape = {S, H}};
  bufs.mb_ratio = {.shape = {S, H}};
  bufs.mb_newvalue = {.shape = {S, H, 1}};
  alloc_register(&alloc, &bufs.mb_obs);
  alloc_register(&alloc, &bufs.mb_state);
  alloc_register(&alloc, &bufs.mb_actions);
  alloc_register(&alloc, &bufs.mb_logprobs);
  alloc_register(&alloc, &bufs.mb_advantages);
  alloc_register(&alloc, &bufs.mb_prio);
  alloc_register(&alloc, &bufs.mb_values);
  alloc_register(&alloc, &bufs.mb_returns);
  alloc_register(&alloc, &bufs.mb_ratio);
  alloc_register(&alloc, &bufs.mb_newvalue);
}

typedef void (*init_weights_fn)(void *weights, uint64_t *seed,
                                cudaStream_t stream);
typedef void (*reg_params_fn)(void *weights, Allocator *alloc, int esz);
typedef void (*reg_train_fn)(void *weights, void *buf, Allocator *acts,
                             Allocator *grads, int B_TT, int precision);
typedef void (*reg_rollout_fn)(void *weights, void *buf, Allocator *alloc,
                               int B);
typedef PrecisionTensor (*forward_fn)(void *weights, void *activations,
                                      PrecisionTensor input, cudaStream_t stream);
typedef void (*encoder_backward_fn)(void *weights, void *activations,
                                    PrecisionTensor grad, cudaStream_t stream);
typedef PrecisionTensor (*decoder_backward_fn)(void *weights, void *activations,
                                               FloatTensor grad_logits,
                                               FloatTensor grad_logstd,
                                               FloatTensor grad_value,
                                               cudaStream_t stream);
typedef PrecisionTensor (*network_forward_fn)(void *weights, PrecisionTensor x,
                                              PrecisionTensor state, void *activations,
                                              cudaStream_t stream);
typedef PrecisionTensor (*network_forward_train_fn)(void *weights, PrecisionTensor x,
                                                    PrecisionTensor state,
                                                    void *activations,
                                                    cudaStream_t stream);
typedef PrecisionTensor (*network_backward_fn)(void *weights, PrecisionTensor grad,
                                               void *activations,
                                               cudaStream_t stream);

struct Encoder {
  forward_fn forward;
  encoder_backward_fn backward;
  init_weights_fn init_weights;
  reg_params_fn reg_params;
  reg_train_fn reg_train;
  reg_rollout_fn reg_rollout;
};

struct Decoder {
  forward_fn forward;
  decoder_backward_fn backward;
  init_weights_fn init_weights;
  reg_params_fn reg_params;
  reg_train_fn reg_train;
  reg_rollout_fn reg_rollout;
};

struct Network {
  network_forward_fn forward;
  network_forward_train_fn forward_train;
  network_backward_fn backward;
  init_weights_fn init_weights;
  reg_params_fn reg_params;
  reg_train_fn reg_train;
  reg_rollout_fn reg_rollout;
};

struct EncoderWeights {
  PrecisionTensor weight;
  int in_dim, out_dim;
};
struct EncoderActivations {
  PrecisionTensor out;
  PrecisionTensor saved_input;
  PrecisionTensor wgrad;
};

struct DecoderWeights {
  PrecisionTensor weight;
  PrecisionTensor logstd;
  int hidden_dim, output_dim;
  bool continuous;
};
struct DecoderActivations {
  PrecisionTensor out;
  PrecisionTensor grad_out;
  PrecisionTensor saved_input;
  PrecisionTensor grad_input;
  PrecisionTensor wgrad;
  PrecisionTensor logstd_scratch;
};

struct MinGRUActivations {
  int num_layers;
  vector<PrecisionTensor> combined;
  PrecisionTensor out;
  PrecisionTensor next_state;
  vector<PrecisionTensor> saved_inputs;
  vector<PrefixScan> scan_bufs;
  vector<PrecisionTensor> combined_bufs;
  vector<PrecisionTensor> wgrad_scratch;
  PrecisionTensor grad_input_buf;
  PrecisionTensor grad_next_state;
};

struct MinGRUWeights {
  int hidden, num_layers, horizon;
  vector<PrecisionTensor> weights;
};

struct Policy {
  Encoder encoder;
  Decoder decoder;
  Network network;
  int input_dim, hidden_dim, output_dim;
  int num_atns;
};

struct PolicyActivations {
  void *encoder;
  void *decoder;
  void *network;
};
struct PolicyWeights {
  void *encoder;
  void *decoder;
  void *network;
};

inline PrecisionTensor *puf_squeeze(PrecisionTensor *t, int dim) {
  int n = puf_ndim(t->shape);
  t->shape[dim + 1] *= t->shape[dim];
  for (int i = dim; i < n - 1; i++) t->shape[i] = t->shape[i + 1];
  t->shape[n - 1] = 0;
  return t;
}

inline FloatTensor *puf_squeeze(FloatTensor *t, int dim) {
  int n = puf_ndim(t->shape);
  t->shape[dim + 1] *= t->shape[dim];
  for (int i = dim; i < n - 1; i++) t->shape[i] = t->shape[i + 1];
  t->shape[n - 1] = 0;
  return t;
}

inline PrecisionTensor *puf_unsqueeze(PrecisionTensor *t, int dim, int64_t d0, int64_t d1) {
  int n = puf_ndim(t->shape);
  for (int i = n; i > dim; i--) t->shape[i] = t->shape[i - 1];
  t->shape[dim] = d0;
  t->shape[dim + 1] = d1;
  return t;
}

inline PrecisionTensor policy_forward(Policy *p, PolicyWeights &w,
                                      PolicyActivations &activations,
                                      PrecisionTensor obs,
                                      PrecisionTensor state,
                                      cudaStream_t stream) {
  PrecisionTensor enc_out =
      p->encoder.forward(w.encoder, activations.encoder, obs, stream);
  PrecisionTensor h = p->network.forward(w.network, enc_out, state,
                                         activations.network, stream);
  return p->decoder.forward(w.decoder, activations.decoder, h, stream);
}

inline PrecisionTensor policy_forward_train(Policy *p, PolicyWeights &w,
                                            PolicyActivations &activations,
                                            PrecisionTensor x,
                                            PrecisionTensor state,
                                            cudaStream_t stream) {
  int B = x.shape[0], TT = x.shape[1];
  PrecisionTensor h =
      p->encoder.forward(w.encoder, activations.encoder, *puf_squeeze(&x, 0), stream);
  h = p->network.forward_train(w.network, *puf_unsqueeze(&h, 0, B, TT), state,
                               activations.network, stream);
  PrecisionTensor dec_out =
      p->decoder.forward(w.decoder, activations.decoder, *puf_squeeze(&h, 0), stream);
  return *puf_unsqueeze(&dec_out, 0, B, TT);
}

inline void policy_backward(Policy *p, PolicyWeights &w,
                            PolicyActivations &activations,
                            FloatTensor grad_logits, FloatTensor grad_logstd,
                            FloatTensor grad_value, cudaStream_t stream) {
  int B = grad_logits.shape[0], TT = grad_logits.shape[1];
  PrecisionTensor grad_h = p->decoder.backward(w.decoder, activations.decoder,
                                               *puf_squeeze(&grad_logits, 0), grad_logstd,
                                               *puf_squeeze(&grad_value, 0), stream);
  grad_h = p->network.backward(w.network, *puf_unsqueeze(&grad_h, 0, B, TT),
                               activations.network, stream);
  p->encoder.backward(w.encoder, activations.encoder, grad_h, stream);
}

inline float cosine_annealing(float lr_base, float lr_min, int t, int T) {
  if (T == 0)
    return lr_base;
  float ratio = (float)t / (float)T;
  ratio = std::max(0.0f, std::min(1.0f, ratio));
  return lr_min + 0.5f * (lr_base - lr_min) * (1.0f + std::cos(M_PI * ratio));
}

static constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270}, {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037}, {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

struct NSScratch {
  PufTensor x, A, gram, tmp;
  PufTensor result_f32;
  float *norm_ptr = nullptr;
  int64_t max_M = 0;
  int64_t max_N = 0;
};

inline PufTensor ns_slice(PufTensor &buf, int64_t rows, int64_t cols) {
  return {
      .bytes = buf.bytes, .shape = {rows, cols}, .dtype_size = buf.dtype_size};
}

struct Muon {
  double momentum;
  float lr_val_init;
  float *lr_ptr;
  float *lr_derived_ptr;
  FloatTensor lr_puf, lr_derived_puf;
  FloatTensor ns_norm_puf;
  FloatTensor wb_puf, mb_puf, gc_puf, up_puf;
  NSScratch ns;
  Allocator *param_alloc;
};

inline PufTensor puf_slice(PufTensor &p, int t, int start, int count) {
  if (p.ndim() == 3) {
    int64_t S = p.shape[1], F = p.shape[2];
    return {.bytes = p.bytes + (t * S + start) * F * p.dtype_size,
            .shape = {count, F},
            .dtype_size = p.dtype_size};
  } else {
    int64_t S = p.shape[1];
    return {.bytes = p.bytes + (t * S + start) * p.dtype_size,
            .shape = {count},
            .dtype_size = p.dtype_size};
  }
}

inline FloatTensor puf_slice(FloatTensor &p, int t, int start, int count) {
  if (puf_ndim(p.shape) == 3) {
    int64_t S = p.shape[1], F = p.shape[2];
    return {.data = p.data + (t * S + start) * F, .shape = {count, F}};
  } else {
    int64_t S = p.shape[1];
    return {.data = p.data + (t * S + start), .shape = {count}};
  }
}

inline PrecisionTensor mingru_state_layer(PrecisionTensor &state, int i) {
  int64_t B = state.shape[1], H = state.shape[2];
  PrecisionTensor layer = {};
  layer.data = (decltype(state.data))((char *)state.data +
                                      i * B * H * state.dtype_size);
  layer.shape[0] = B;
  layer.shape[1] = H;
  layer.dtype_size = state.dtype_size;
  return layer;
}

struct EnvBuf {
  PufTensor obs;
  int obs_raw_dtype;
  PufTensor actions;
  FloatTensor rewards;
  FloatTensor terminals;
};

#endif // PUFFERLIB_PUF_TYPES_H
