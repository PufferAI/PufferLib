import argparse
import ctypes
import os
import subprocess

import torch

from point_linear_max_reference import FlatLinearEncoder, PointLinearMaxEncoder, input_dim


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "point_linear_max_kernel.cu")
SO = os.path.join(ROOT, "point_linear_max_kernel.so")


def build(force=False):
    if not force and os.path.exists(SO) and os.path.getmtime(SO) >= os.path.getmtime(SRC):
        return

    cmd = ["nvcc", "-shared", "-o", SO, SRC, "-Xcompiler", "-fPIC", "-O2"]
    print(f"Building: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def load_lib():
    lib = ctypes.CDLL(SO)
    vp = ctypes.c_void_p
    lib.point_linear_max_forward.argtypes = [
        vp, vp, vp, vp,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.point_linear_max_forward.restype = ctypes.c_int
    lib.point_linear_max_synchronize.restype = ctypes.c_int
    lib.point_linear_max_error_string.argtypes = [ctypes.c_int]
    lib.point_linear_max_error_string.restype = ctypes.c_char_p
    return lib


def check_cuda(code, lib, where):
    if code != 0:
        msg = lib.point_linear_max_error_string(code).decode("utf-8")
        raise RuntimeError(f"{where} failed: {msg}")


def ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def run_kernel(lib, output, observations, encoder):
    code = lib.point_linear_max_forward(
        ptr(output),
        ptr(observations),
        ptr(encoder.linear.weight),
        ptr(encoder.linear.bias),
        observations.shape[0],
        encoder.self_obs_size,
        encoder.point_obs_size,
        encoder.num_points,
        encoder.hidden_size,
    )
    check_cuda(code, lib, "kernel launch")


def synchronize(lib):
    check_cuda(lib.point_linear_max_synchronize(), lib, "device synchronize")


def generate_observations(batch_size, obs_dim, device):
    return torch.randn(batch_size, obs_dim, device=device, dtype=torch.float32)


def check_close(name, got, ref, atol=1e-6, rtol=1e-5):
    diff = (got - ref).abs()
    print(f"  [{name}] max={diff.max().item():.6e} mean={diff.mean().item():.6e}")
    if not torch.allclose(got, ref, atol=atol, rtol=rtol):
        idx = diff.argmax().item()
        row = idx // got.shape[1]
        col = idx % got.shape[1]
        raise AssertionError(
            f"{name} mismatch at ({row}, {col}): got={got[row, col].item():.6f} ref={ref[row, col].item():.6f}"
        )


def make_encoders(args, device):
    obs_dim = input_dim(args.self_dim, args.point_dim, args.num_points)
    point_encoder = PointLinearMaxEncoder(
        self_dim=args.self_dim,
        point_dim=args.point_dim,
        num_points=args.num_points,
        hidden_size=args.hidden_size,
    ).to(device).float().eval()
    flat_encoder = FlatLinearEncoder(obs_dim, hidden_size=args.hidden_size).to(device).float().eval()
    return obs_dim, point_encoder, flat_encoder


def test_correctness(lib, args, batches):
    print("Correctness")
    device = torch.device("cuda")
    torch.manual_seed(0)
    obs_dim, _, _ = make_encoders(args, device)

    for batch_size in batches:
        _, encoder, _ = make_encoders(args, device)
        observations = generate_observations(batch_size, obs_dim, device)
        output = torch.empty(batch_size, args.hidden_size, device=device, dtype=torch.float32)
        run_kernel(lib, output, observations, encoder)
        synchronize(lib)
        with torch.no_grad():
            reference = encoder(observations)
        check_close(f"B={batch_size}", output, reference)


def benchmark_one(name, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"  {name:<20} {ms:8.3f} ms")
    return ms


def benchmark(lib, args):
    print(
        f"Benchmark B={args.benchmark_batch} H={args.hidden_size} "
        f"self={args.self_dim} point={args.point_dim} npoints={args.num_points}"
    )
    device = torch.device("cuda")
    torch.manual_seed(0)
    obs_dim, encoder, flat_encoder = make_encoders(args, device)
    observations = generate_observations(args.benchmark_batch, obs_dim, device)
    kernel_output = torch.empty(args.benchmark_batch, args.hidden_size, device=device, dtype=torch.float32)

    def fused_kernel():
        run_kernel(lib, kernel_output, observations, encoder)

    def torch_pointwise():
        encoder(observations)

    def torch_flat():
        flat_encoder(observations)

    kernel_ms = benchmark_one("fused kernel", fused_kernel, args.warmup, args.iters)
    torch_ms = benchmark_one("torch pointwise", torch_pointwise, args.warmup, args.iters)
    flat_ms = benchmark_one("torch flat", torch_flat, args.warmup, args.iters)

    print(f"  speedup vs torch: {torch_ms / kernel_ms:.2f}x")
    print(f"  relative to flat: {kernel_ms / flat_ms:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-dim", type=int, default=2)
    parser.add_argument("--point-dim", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--correctness-batches", type=int, nargs="+", default=[1, 17, 257, 4096])
    parser.add_argument("--benchmark-batch", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    build(force=args.force_build)
    lib = load_lib()
    test_correctness(lib, args, args.correctness_batches)
    benchmark(lib, args)


if __name__ == "__main__":
    main()
