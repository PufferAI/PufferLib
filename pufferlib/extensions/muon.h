#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <nccl.h>

#include <utility>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::optim {

struct TORCH_API MuonOptions : public OptimizerCloneableOptions<MuonOptions> {
  MuonOptions(double initial_lr = 0.0025);
  TORCH_ARG(double, initial_lr) = 0.0025;
  TORCH_ARG(double, weight_decay) = 0.0;
  TORCH_ARG(double, momentum) = 0.9;
  TORCH_ARG(double, eps) = 1e-8;

 public:
  //void serialize(torch::serialize::InputArchive& archive) override;
  //void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const MuonOptions& lhs,
      const MuonOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API MuonParamState
    : public OptimizerCloneableParamState<MuonParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, momentum_buffer);

 public:
  //void serialize(torch::serialize::InputArchive& archive) override;
  //void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const MuonParamState& lhs,
      const MuonParamState& rhs);
};

class TORCH_API Muon : public Optimizer {
 public:
  torch::Tensor lr;
  torch::Tensor weight_buffer;  // Contiguous weight buffer for batched updates
  torch::Tensor momentum_buffer;  // Contiguous momentum buffer
  ncclComm_t nccl_comm = nullptr;  // NCCL communicator for multi-GPU gradient sync
  int world_size = 1;

  explicit Muon(
      const std::vector<OptimizerParamGroup>& param_groups,
      MuonOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<MuonOptions>(defaults)) {
    TORCH_CHECK(defaults.initial_lr() >= 0, "Invalid initial learning rate: ", defaults.initial_lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());

    lr = torch::tensor(defaults.initial_lr(), torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
  }
  explicit Muon(std::vector<Tensor> params, MuonOptions defaults = {})
      : Muon({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  // Create contiguous weight buffer from params for batched updates
  void init_contiguous_weights() {
    torch::NoGradGuard no_grad;
    auto& params = param_groups_[0].params();

    // Count total size
    int64_t total_size = 0;
    for (auto& p : params) {
      total_size += p.numel();
    }

    // Allocate single contiguous buffer
    auto device = params[0].device();
    weight_buffer = torch::zeros({total_size},
        torch::dtype(torch::kFloat32).device(device));
    weight_buffer.set_requires_grad(true);

    // Copy params into buffer and replace with views
    int64_t offset = 0;
    for (auto& p : params) {
      int64_t size = p.numel();
      auto shape = p.sizes().vec();

      // Copy current values into buffer
      weight_buffer.narrow(0, offset, size).copy_(p.flatten());

      // Replace param data with view into buffer
      torch::Tensor view = weight_buffer.narrow(0, offset, size).view(shape);
      p.set_data(view);

      offset += size;
    }
  }

  torch::Tensor step(LossClosure closure = nullptr) override;
  //void save(serialize::OutputArchive& archive) const override;
  //void load(serialize::InputArchive& archive) override;

 //private:
 // template <typename Self, typename Archive>
 // static void serialize(Self& self, Archive& archive) {
 //   _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Muon);
 // }
};
} // namespace torch::optim
