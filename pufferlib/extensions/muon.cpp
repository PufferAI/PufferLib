//#include <torch/optim/muon.h>
#include "muon.h"

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/irange.h>

#include <cmath>
#include <functional>
#include <iostream>

#include "muon.h"

namespace torch::optim {

const double coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

MuonOptions::MuonOptions(double initial_lr) : initial_lr_(initial_lr) {}

bool operator==(const MuonOptions& lhs, const MuonOptions& rhs) {
  return (lhs.initial_lr() == rhs.initial_lr()) &&
      (lhs.eps() == rhs.eps()) &&
      (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.momentum() == rhs.momentum());
}

/*
void MuonOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
}

void MuonOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, momentum);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
}
*/

double MuonOptions::get_lr() const {
  return initial_lr();
}

void MuonOptions::set_lr(const double initial_lr) {
  this->initial_lr(initial_lr);
}

bool operator==(const MuonParamState& lhs, const MuonParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
      torch::equal(lhs.momentum_buffer(), rhs.momentum_buffer());
}

/*
void MuonParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum_buffer);
}

void MuonParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, momentum_buffer);
}
*/

//TODO: You actually want this in bfloat16. Still seems slow
Tensor _zeropower_via_newtonschulz(Tensor G) {
    auto x = G.to(torch::kBFloat16);
    //auto x = G.clone();
    if (G.size(-2) > G.size(-1)) {
        x = x.mT();
    }

    // Heavyball hardcodes 1e-7
    x.div_(x.norm().clamp(1e-7));

    for (int i = 0; i < 5; ++i) {
        auto a = coeffs[i][0];
        auto b = coeffs[i][1];
        auto c = coeffs[i][2];
        auto A = x.mm(x.mT());
        auto gram_update = at::addmm(A, A, A, b, c);  // beta=b, alpha=c
        x = at::addmm(x, gram_update, x, a, 1.0);
    }

    if (G.size(-2) > G.size(-1)) {
        x = x.mT();
    }

    return x.to(G.dtype());
}

Tensor Muon::step(LossClosure closure) {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  for (auto& group : param_groups_) {
    auto& options = static_cast<MuonOptions&>(group.options());
    auto momentum_coef = options.momentum();
    auto weight_decay = options.weight_decay();

    // Fast path: use contiguous buffers
    if (weight_buffer.defined()) {
      // Initialize momentum buffer lazily to match weight_buffer
      if (!momentum_buffer.defined()) {
        momentum_buffer = torch::zeros_like(weight_buffer);
      }

      // Build full-size grad tensor (zeros for unused params)
      Tensor all_grads = torch::zeros_like(weight_buffer);
      int64_t offset = 0;
      for (auto& p : group.params()) {
        int64_t size = p.numel();
        if (p.grad().defined()) {
          all_grads.narrow(0, offset, size).copy_(p.grad().flatten());
        }
        offset += size;
      }

      // Multi-GPU gradient sync: average gradients across all ranks
      if (nccl_comm != nullptr && world_size > 1) {
        ncclAllReduce(all_grads.data_ptr(), all_grads.data_ptr(),
                      all_grads.numel(), ncclFloat, ncclAvg,
                      nccl_comm, at::cuda::getCurrentCUDAStream());
      }

      // Batched Nesterov momentum (one mul_, one add_ each)
      momentum_buffer.mul_(momentum_coef);
      momentum_buffer.add_(all_grads);
      all_grads.add_(momentum_buffer, momentum_coef);

      // Newton-Schulz per-param and build full-size update tensor
      Tensor all_updates = torch::zeros_like(weight_buffer);
      offset = 0;
      for (auto& p : group.params()) {
        int64_t size = p.numel();
        if (p.grad().defined()) {
          Tensor update = all_grads.narrow(0, offset, size).view(p.sizes());

          if (p.dim() >= 2) {
            auto G = update.view({update.size(0), -1});
            update = _zeropower_via_newtonschulz(G);
            double ratio = (double)update.size(-2) / (double)update.size(-1);
            double scale = std::sqrt(std::max(1.0, ratio));
            update.mul_(scale);
          }

          all_updates.narrow(0, offset, size).copy_(update.flatten());
        }
        offset += size;
      }

      // Single batched param update (one mul_, one sub_)
      if (weight_decay != 0) {
        weight_buffer.mul_(1 - lr * weight_decay);
      }
      weight_buffer.sub_(all_updates * lr);
    }
  }
  return loss;
}

/*
void Muon::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Muon::load(serialize::InputArchive& archive) {
  IValue pytorch_version;
  if (archive.try_read("pytorch_version", pytorch_version)) {
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    TORCH_WARN(
        "Your serialized Muon optimizer is still using the old serialization format. "
        "You should re-save your Muon optimizer to use the new serialization format.");
    std::vector<int64_t> step_buffers;
    std::vector<at::Tensor> momentum_buffers;
    torch::optim::serialize(archive, "step_buffers", step_buffers);
    torch::optim::serialize(archive, "momentum_buffers", momentum_buffers);
    // since there were no param_groups prior to version 1.5.0, assuming all
    // tensors are now in one param_group
    std::vector<Tensor> params = param_groups_.at(0).params();
    for (const auto idx : c10::irange(step_buffers.size())) {
      auto state = std::make_unique<MuonParamState>();
      state->step(step_buffers.at(idx));
      state->momentum_buffer(momentum_buffers.at(idx));
      state_[params.at(idx).unsafeGetTensorImpl()] = std::move(state);
    }
  }
}
*/
} // namespace torch::optim
