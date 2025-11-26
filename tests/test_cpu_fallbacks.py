import torch

# These tests are CPU-only sanity checks for the pure-Torch fallbacks
# that are compiled when CUDA is unavailable. They run quickly.

def test_rmsnorm_cpu_fallback():
    from pufferlib import _C

    B, T, H = 2, 3, 4
    x = torch.randn(B, T, H)
    w = torch.randn(H)
    eps = 1e-5

    out_ref = torch.nn.functional.rms_norm(x, (H,), w, eps)
    out_ext = _C.rmsnorm(x, w, eps)[0]
    assert torch.allclose(out_ref, out_ext, rtol=1e-4, atol=1e-5)


def test_logcumsumexp_cpu_fallback():
    from pufferlib import _C

    B, T, H = 2, 5, 3
    x = torch.randn(B, T, H)
    out_ref = x.logcumsumexp(dim=1)
    out_ext = _C.logcumsumexp_cuda(x)
    assert torch.allclose(out_ref, out_ext, rtol=1e-5, atol=1e-6)


def test_fused_scan_cpu_fallback():
    from pufferlib import _C

    B, T, H = 1, 4, 2
    gate = torch.randn(B, T, H)
    hidden = torch.randn(B, T, H)

    # Reference: log_coeffs/log_values + associative scan in Python
    log_coeffs = -torch.nn.functional.softplus(gate)
    log_z = -torch.nn.functional.softplus(-gate)
    relu_h = torch.relu(hidden)
    log_tilde_h = torch.where(hidden >= 0, (relu_h + 0.5).log(), -torch.nn.functional.softplus(-hidden))
    log_values = log_z + log_tilde_h

    a_star = log_coeffs.cumsum(1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1)
    out_ref = (a_star + log_h0_plus_b_star).exp()

    out_ext = _C.fused_scan(log_coeffs, log_values)[0]
    assert torch.allclose(out_ref, out_ext, rtol=1e-4, atol=1e-5)


def test_mingru_gate_cpu_fallback():
    from pufferlib import _C

    state = torch.randn(8, 3, 5)
    gate = torch.randn(8, 3, 5)
    hidden = torch.randn(8, 3, 5)

    hidden_pos = torch.where(hidden >= 0, hidden + 0.5, torch.sigmoid(hidden))
    gate_sig = torch.sigmoid(gate)
    out_ref = torch.lerp(state, hidden_pos, gate_sig)

    out_ext = _C.mingru_gate(state, gate, hidden)
    assert torch.allclose(out_ref, out_ext, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_rmsnorm_cpu_fallback()
    test_logcumsumexp_cpu_fallback()
    test_fused_scan_cpu_fallback()
    test_mingru_gate_cpu_fallback()
    print("CPU fallback tests passed.")
