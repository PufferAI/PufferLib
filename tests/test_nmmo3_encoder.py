"""Test NMMO3 CUDA encoder forward and backward against PyTorch reference.

Builds a shared library from test_nmmo3_cuda.cu (thin wrapper around ocean.cu),
then compares numerics for forward output and all weight gradients.
"""

import subprocess
import ctypes
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = os.path.join(os.path.dirname(__file__), "test_nmmo3_cuda.cu")
SO = os.path.join(os.path.dirname(__file__), "ocean_test.so")


def build():
    cmd = [
        "nvcc", "-shared", "-o", SO, SRC,
        "-I", os.path.join(os.path.dirname(__file__), "..", "pufferlib", "src"),
        "-lcublas", "-lcudnn", "-lcurand",
        "--compiler-options", "-fPIC", "-Xcompiler", "-O2",
    ]
    print(f"Building: {' '.join(cmd)}")
    subprocess.check_call(cmd)


# ============================================================================
# PyTorch reference
# ============================================================================

FACTORS = [4, 4, 17, 5, 3, 5, 5, 5, 7, 4]
OFFSETS_NP = [0] + list(np.cumsum(FACTORS)[:-1])


class NMMO3EncoderRef(nn.Module):
    def __init__(self):
        super().__init__()
        offsets = torch.tensor(OFFSETS_NP).view(1, -1, 1, 1)
        self.register_buffer("offsets", offsets)
        self.conv1 = nn.Conv2d(59, 128, 5, stride=3)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1)
        self.embed = nn.Embedding(128, 32)
        self.proj = nn.Linear(1817, 512)

    def forward(self, observations):
        B = observations.shape[0]
        ob_map = observations[:, :1650].view(B, 11, 15, 10)
        mh = torch.zeros(B, 59, 11, 15, dtype=torch.float32, device=observations.device)
        codes = ob_map.long().permute(0, 3, 1, 2) + self.offsets
        mh.scatter_(1, codes, 1)
        x = F.relu(self.conv1(mh))
        x = self.conv2(x).flatten(1)
        ob_player = observations[:, 1650:-10]
        player_embed = self.embed(ob_player.int()).flatten(1)
        ob_reward = observations[:, -10:]
        cat = torch.cat([x, player_embed, ob_player.float(), ob_reward.float()], dim=1)
        return F.relu(self.proj(cat))


# ============================================================================
# ctypes wrapper
# ============================================================================

def load_lib():
    lib = ctypes.CDLL(SO)
    VP = ctypes.c_void_p
    lib.nmmo3_test_init.argtypes = [ctypes.c_int]
    lib.nmmo3_test_set_weights.argtypes = [VP] * 7
    lib.nmmo3_test_forward.argtypes = [VP, VP, ctypes.c_int]
    lib.nmmo3_test_backward.argtypes = [VP, ctypes.c_int]
    for name in ["conv1_wgrad", "conv1_bgrad", "conv2_wgrad", "conv2_bgrad",
                 "proj_wgrad", "proj_bgrad", "embed_wgrad"]:
        getattr(lib, f"nmmo3_test_get_{name}").argtypes = [VP]
    lib.nmmo3_test_get_conv1_out.argtypes = [VP, ctypes.c_int]
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def extract_weights(model):
    return [t.data.contiguous() for t in [
        model.conv1.weight, model.conv1.bias,
        model.conv2.weight, model.conv2.bias,
        model.embed.weight, model.proj.weight, model.proj.bias,
    ]]


def generate_valid_obs(B, device):
    obs = torch.zeros(B, 1707, dtype=torch.float32, device=device)
    for h in range(11):
        for w in range(15):
            for f in range(10):
                idx = (h * 15 + w) * 10 + f
                obs[:, idx] = torch.randint(0, FACTORS[f], (B,)).float()
    obs[:, 1650:1697] = torch.randint(0, 128, (B, 47)).float()
    obs[:, 1697:1707] = torch.randint(0, 256, (B, 10)).float()
    return obs


def check_match(name, got, ref, atol=1e-4, rtol=1e-4):
    max_diff = (got - ref).abs().max().item()
    mean_diff = (got - ref).abs().mean().item()
    ref_norm = ref.abs().mean().item()
    print(f"  [{name}] max={max_diff:.6e} mean={mean_diff:.6e} "
          f"rel={mean_diff / (ref_norm + 1e-8):.6e}")
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    if not ok:
        diff = (got - ref).abs()
        idx = np.unravel_index(diff.argmax().item(), got.shape)
        print(f"    Worst at {idx}: got={got[idx].item():.6f}, ref={ref[idx].item():.6f}")
    assert ok, f"{name} FAILED"


# ============================================================================
# Tests
# ============================================================================

def test_forward(lib, B):
    """Compare CUDA encoder forward against PyTorch reference."""
    print(f"\n--- Forward B={B} ---")
    device = torch.device("cuda")
    torch.manual_seed(42)
    model = NMMO3EncoderRef().to(device).float().eval()
    obs = generate_valid_obs(B, device)

    lib.nmmo3_test_init(B)
    lib.nmmo3_test_set_weights(*[ptr(w) for w in extract_weights(model)])

    cuda_out = torch.zeros(B, 512, device=device)
    lib.nmmo3_test_forward(ptr(cuda_out), ptr(obs), B)
    torch.cuda.synchronize()

    with torch.no_grad():
        ref_out = model(obs)

    check_match("forward", cuda_out, ref_out)
    print("  PASSED")


def test_backward(lib, B):
    """Compare CUDA encoder backward (all weight grads) against PyTorch autograd.

    Uses CUDA forward relu masks on the PyTorch path so both sides agree on
    which activations are zeroed — avoids cuDNN fused-relu edge cases near zero.
    """
    print(f"\n--- Backward B={B} ---")
    device = torch.device("cuda")
    torch.manual_seed(42)
    model = NMMO3EncoderRef().to(device).float()
    obs = generate_valid_obs(B, device)

    lib.nmmo3_test_init(B)
    weights = extract_weights(model)
    lib.nmmo3_test_set_weights(*[ptr(w) for w in weights])

    # ---- CUDA forward ----
    cuda_out = torch.zeros(B, 512, device=device)
    lib.nmmo3_test_forward(ptr(cuda_out), ptr(obs), B)

    conv1_out_cuda = torch.zeros(B, 128, 3, 4, device=device)
    lib.nmmo3_test_get_conv1_out(ptr(conv1_out_cuda), B)
    torch.cuda.synchronize()

    # ---- PyTorch forward with CUDA relu masks ----
    c1w, c1b, c2w, c2b, ew, pw, pb = [w.detach().requires_grad_(True) for w in weights]

    # Multihot (discrete, no grad)
    ob_map = obs[:, :1650].view(B, 11, 15, 10)
    mh = torch.zeros(B, 59, 11, 15, device=device)
    offsets = torch.tensor(OFFSETS_NP).view(1, -1, 1, 1).to(device)
    codes = ob_map.long().permute(0, 3, 1, 2) + offsets
    mh.scatter_(1, codes, 1)

    # Conv1: raw conv, apply CUDA's relu mask
    conv1_raw = F.conv2d(mh, c1w, c1b, stride=3)
    conv1_masked = conv1_raw * (conv1_out_cuda > 0).float()

    # Conv2: no relu
    conv2_out = F.conv2d(conv1_masked, c2w, c2b, stride=1)

    # Player/reward branches
    ob_player = obs[:, 1650:-10]
    player_embed = F.embedding(ob_player.int(), ew).flatten(1)
    ob_reward = obs[:, -10:]

    # Concat + projection with CUDA's relu mask
    cat = torch.cat([conv2_out.flatten(1), player_embed,
                     ob_player.detach().float(), ob_reward.detach().float()], dim=1)
    proj_raw = F.linear(cat, pw, pb)
    proj_masked = proj_raw * (cuda_out > 0).float()

    # ---- Backward ----
    grad_output = torch.randn(B, 512, device=device)
    proj_masked.backward(grad_output)

    grad_cuda = grad_output.clone()
    lib.nmmo3_test_backward(ptr(grad_cuda), B)
    torch.cuda.synchronize()

    # ---- Compare all weight grads ----
    tol = dict(atol=1e-2, rtol=1e-2)

    cuda_proj_wgrad = torch.zeros(512, 1817, device=device)
    lib.nmmo3_test_get_proj_wgrad(ptr(cuda_proj_wgrad))
    check_match("proj_wgrad", cuda_proj_wgrad, pw.grad, **tol)

    cuda_proj_bgrad = torch.zeros(512, device=device)
    lib.nmmo3_test_get_proj_bgrad(ptr(cuda_proj_bgrad))
    check_match("proj_bgrad", cuda_proj_bgrad, pb.grad, **tol)

    cuda_c2_wgrad = torch.zeros(128, 128 * 3 * 3, device=device)
    lib.nmmo3_test_get_conv2_wgrad(ptr(cuda_c2_wgrad))
    check_match("conv2_wgrad", cuda_c2_wgrad, c2w.grad.view(128, -1), **tol)

    cuda_c2_bgrad = torch.zeros(128, device=device)
    lib.nmmo3_test_get_conv2_bgrad(ptr(cuda_c2_bgrad))
    check_match("conv2_bgrad", cuda_c2_bgrad, c2b.grad, **tol)

    cuda_c1_wgrad = torch.zeros(128, 59 * 5 * 5, device=device)
    lib.nmmo3_test_get_conv1_wgrad(ptr(cuda_c1_wgrad))
    check_match("conv1_wgrad", cuda_c1_wgrad, c1w.grad.view(128, -1), **tol)

    cuda_c1_bgrad = torch.zeros(128, device=device)
    lib.nmmo3_test_get_conv1_bgrad(ptr(cuda_c1_bgrad))
    check_match("conv1_bgrad", cuda_c1_bgrad, c1b.grad, **tol)

    cuda_embed_wgrad = torch.zeros(128, 32, device=device)
    lib.nmmo3_test_get_embed_wgrad(ptr(cuda_embed_wgrad))
    check_match("embed_wgrad", cuda_embed_wgrad, ew.grad, **tol)

    print("  PASSED")


# ============================================================================
# Benchmarks
# ============================================================================

def bench_encoder(lib, B, warmup=10, iters=100):
    print(f"\n--- Benchmark B={B} ({iters} iters) ---")
    device = torch.device("cuda")
    torch.manual_seed(42)
    model = NMMO3EncoderRef().to(device).float().eval()
    obs = generate_valid_obs(B, device)

    lib.nmmo3_test_init(B)
    lib.nmmo3_test_set_weights(*[ptr(w) for w in extract_weights(model)])

    output = torch.zeros(B, 512, device=device)

    def run_cuda():
        lib.nmmo3_test_forward(ptr(output), ptr(obs), B)

    def run_torch():
        with torch.no_grad():
            model(obs)

    for fn in [run_cuda, run_torch]:
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = {}
    for name, fn in [("cuda", run_cuda), ("torch", run_torch)]:
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times[name] = start.elapsed_time(end) / iters

    for name, ms in times.items():
        print(f"  {name:8s} {ms:.3f} ms")
    print(f"  cuda vs torch: {times['torch'] / times['cuda']:.2f}x")


if __name__ == "__main__":
    build()
    lib = load_lib()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("test", "all"):
        for B in [1, 8, 64]:
            test_forward(lib, B)
            test_backward(lib, B)
        print("\nAll tests passed!")

    if mode in ("bench", "all"):
        for B in [1, 8, 64, 256]:
            bench_encoder(lib, B)
