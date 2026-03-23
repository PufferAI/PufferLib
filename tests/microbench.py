import time
import rich

import torch
import torch.utils.cpp_extension

import pufferlib
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')

BR = 4096  # Rollout batch (no T dim)
BT = 512   # Train batch (with T dim)
T = 64
H = 128
A = 4
TIMEOUT = 1

def check_close(a, b, rtol=1e-3, atol=1e-4):
    output = []
    assert len(a) == len(b)
    for a, b in zip(a, b):
        a = a.float()
        b = b.float()
        max_diff = (a - b).abs().max()
        passed = torch.allclose(a, b, rtol=rtol, atol=atol)
        color = 'green' if passed else 'red'
        output.append(f'[{color}]{max_diff:.2e}[/{color}]')

    return ' '.join(output)

def parse_args(args):
    py_args = []
    cpp_args = []
    backward = False
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if arg.requires_grad:
                backward = True

            # VERY IMPORTANT: You have to set requires_grad AFTER moving to GPU
            # Otherwise torch moves grads back to CPU and crushes perf
            #dtype = torch.float64 if arg.dtype == torch.float32 else arg.dtype
            dtype = torch.float32 if arg.dtype == torch.float32 else arg.dtype
            py_args.append(arg.clone().detach().to(dtype).cuda().requires_grad_(arg.requires_grad))
            cpp_args.append(arg.clone().detach().cuda().requires_grad_(arg.requires_grad))
        else:
            py_args.append(arg)
            cpp_args.append(arg)

    return py_args, cpp_args, backward

def test_loss(outputs):
    if type(outputs) == torch.Tensor:
        return outputs.sum()

    return sum([o.sum() for o in outputs])/len(outputs)

def test_kernel(py_func, cpp_func, *args, benchmark=True):
    py_args, cpp_args, backward = parse_args(args)

    py_out = py_func(*py_args)
    cpp_out = cpp_func(*cpp_args)

    if not isinstance(py_out, (tuple, list)):
        py_out = [py_out]
    if not isinstance(cpp_out, (tuple, list)):
        cpp_out = [cpp_out]

    output = check_close(py_out, cpp_out)
    rich.print('\tForward check:', output)

    if backward:
        py_loss = test_loss(py_out)
        cpp_loss = test_loss(cpp_out)

        py_loss.backward()
        cpp_loss.backward()

        has_grad = lambda v: [e.grad for e in v if isinstance(e, torch.Tensor) and e.grad is not None]
        py_grad = has_grad(py_args)
        cpp_grad = has_grad(cpp_args)


        output = check_close(py_grad, cpp_grad)
        rich.print('\tBackward check:', output)

    if benchmark:
        py_sps = time_sps(py_func, *py_args)
        cpp_sps = time_sps(cpp_func, *cpp_args)
        print(f'\tForward sps: {py_sps} (naive) {cpp_sps} (C++)')

        if backward:
            py_sps = time_sps(py_func, *py_args, backward=True)
            cpp_sps = time_sps(cpp_func, *cpp_args, backward=True)
            print(f'\tBackward sps: {py_sps} (naive) {cpp_sps} (C++)')

def time_sps(func, *args, backward=False):
    assert isinstance(args[0], torch.Tensor)
    N = args[0].shape[:-1].numel()

    if backward:
        outputs = func(*args)
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]
        grad_outputs = [torch.randn_like(o) for o in outputs]

    # Warm up
    for i in range(3):
        if backward:
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.requires_grad:
                    arg.grad = None
            torch.autograd.backward(outputs, grad_outputs, retain_graph=True)
        else:
            with torch.no_grad():
                func(*args)

    torch.cuda.synchronize()
    start = time.time()
    steps = 0
    while time.time() - start < TIMEOUT:
        steps += 1
        if backward:
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.requires_grad:
                    arg.grad = None
            torch.autograd.backward(outputs, grad_outputs, retain_graph=True)
        else:
            with torch.no_grad():
                func(*args)

    torch.cuda.synchronize()
    sps = N*steps/(time.time() - start)
    if sps < 1e3:
        return f'{sps:.2f}'
    if sps < 1e6:
        return f'{sps/1e3:.2f} K'
    if sps < 1e9:
        return f'{sps/1e6:.2f} M'

    return f'{sps/1e9:.2f} B'

def mingru_gate(state, gate, hidden):
    hidden = torch.where(hidden >= 0, hidden + 0.5, hidden.sigmoid())
    gate = gate.sigmoid()
    out = torch.lerp(state, hidden, gate)
    return out

def test_mingru_gate():
    state = torch.randn(BR, H)
    gate = torch.randn(BR, H)
    hidden = torch.randn(BR, H)
    print('mingru_gate')
    test_kernel(mingru_gate, _C.mingru_gate, state, gate, hidden)

def log_coeffs_and_values(gate, hidden):
    log_coeffs = -torch.nn.functional.softplus(gate)
    log_z = -torch.nn.functional.softplus(-gate)
    log_tilde_h = torch.where(hidden >= 0,
        (torch.nn.functional.relu(hidden) + 0.5).log(),
        -torch.nn.functional.softplus(-hidden))
    log_values = log_z + log_tilde_h
    return log_coeffs, log_values

def log_coeffs_and_values_loss(outputs):
    log_coeffs, log_values = outputs
    return torch.sum(log_coeffs) + torch.sum(log_values)

def test_log_coeffs_and_values():
    gate = torch.randn(BT, T, H, requires_grad=True)
    hidden = torch.randn(BT, T, H, requires_grad=True)
    print('log_coeffs_and_values')
    test_kernel(log_coeffs_and_values, _C.log_coeffs_and_values, gate, hidden)

def fused_scan(log_coeffs, log_values, state):
    # Fuse cat+pad+narrow into the scan (matches kernel behavior)
    log_values = torch.cat([state.log(), log_values], dim=1)
    log_coeffs = torch.nn.functional.pad(log_coeffs, (0, 0, 1, 0))
    a_star = log_coeffs.cumsum(1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1)
    log_h = a_star + log_h0_plus_b_star
    full_out = log_h.exp()
    # Narrow to get last T timesteps (kernel returns this directly)
    T = log_values.size(1) - 1  # original T before cat
    out = full_out.narrow(1, 1, T)  # skip first timestep
    next_state = full_out.narrow(1, T, 1)  # last timestep
    return [out, next_state]

def fused_scan_loss(outputs):
    return torch.sum(outputs[0]) + torch.sum(outputs[1])

def test_fused_scan():
    # Numerically unstable function. Must be called with the distribution
    # that is used in the full network.
    log_coeffs = -torch.nn.functional.softplus(torch.randn(BT, T, H)).requires_grad_(True)
    log_values = -torch.nn.functional.softplus(torch.randn(BT, T, H)).requires_grad_(True)
    state = torch.rand(BT, 1, H).requires_grad_(True)  # state must be positive for log

    print('fused_scan')
    test_kernel(fused_scan, _C.fused_scan, log_coeffs, log_values, state)

def logcumsumexp(x):
    return [torch.log(torch.exp(x).cumsum(1))]

def logcumsumexp_loss(outputs):
    return torch.sum(outputs[0])

def test_logcumsumexp():
    x = torch.randn(BT, T, H, requires_grad=True)
    print('logcumsumexp')
    test_kernel(logcumsumexp, _C.logcumsumexp_cuda, x)

def fused_ppo_loss(logits, newvalue, actions, old_logprobs,
        advantages, prio, values, returns, adv_mean, adv_std,
        clip_coef, vf_clip_coef, vf_coef, ent_coef):

    segments, horizon, _ = logits.shape

    flat_logits = logits.reshape(-1, logits.size(-1));
    flat_actions = actions.reshape(-1);
    logprobs_new = torch.log_softmax(flat_logits, 1);

    probs_new = logprobs_new.exp();
    entropy = - (probs_new * logprobs_new).sum(1).mean();

    newlogprob_flat = logprobs_new.gather(1, flat_actions.unsqueeze(1)).squeeze(1);
    newlogprob = newlogprob_flat.reshape(segments, horizon);
    logratio = newlogprob - old_logprobs;
    ratio_new = logratio.exp();

    adv_normalized = prio.unsqueeze(1) * (advantages - adv_mean) / (adv_std + 1e-8);
    pg_loss1 = -adv_normalized * ratio_new;
    pg_loss2 = -adv_normalized * torch.clamp(ratio_new, 1.0 - clip_coef, 1.0 + clip_coef);
    pg_loss = torch.max(pg_loss1, pg_loss2).mean();

    newvalue = newvalue.view(returns.shape)
    v_clipped = values + torch.clamp(newvalue - values, -vf_clip_coef, vf_clip_coef);
    v_loss_unclipped = (newvalue - returns).pow(2);
    v_loss_clipped = (v_clipped - returns).pow(2);
    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean();

    # Entrop is a little off (1e-6)
    loss = pg_loss + vf_coef*v_loss - ent_coef*entropy
    return loss

def test_fused_ppo_loss():
    logits = torch.randn(BT, T, A, requires_grad=True)
    values_pred = torch.randn(BT, T, requires_grad=True).contiguous()
    actions = torch.randint(0, A, (BT, T))
    old_logprobs = torch.randn(BT, T)
    advantages = torch.randn(BT, T)
    prio = torch.rand(BT)
    values = torch.randn(BT, T)
    returns = torch.randn(BT, T)

    adv_mean = advantages.mean()
    adv_std = advantages.std()

    # TODO: These should be tensors, but have to adjust the test kernel too.
    # This makes it much slower... but needed for graphing? More perf checks required.
    clip_coef = 0.1
    vf_clip_coef = 0.1
    vf_coef = 0.1
    ent_coef = 0.1

    args = (fused_ppo_loss, _C.fused_ppo_loss, logits, values_pred, actions,
        old_logprobs, advantages, prio, values, returns, advantages.mean(), advantages.std(),
        clip_coef, vf_clip_coef, vf_coef, ent_coef)
    print('fused_ppo_loss')
    test_kernel(*args)

def rmsnorm(x, weight, eps):
    shape = (x.shape[-1],)
    return torch.nn.functional.rms_norm(x, shape, weight, eps)

def rmsnorm_loss(outputs):
    return torch.sum(outputs[0])

def test_rmsnorm():
    x = torch.randn(BT, T, H, requires_grad=True)
    weight = torch.randn(H, requires_grad=True)
    eps = 1e-5

    print('rmsnorm correctness')
    test_kernel(rmsnorm, _C.rmsnorm, x, weight, eps)

def sample_logits_py(logits):
    """Reference implementation: nan_to_num + log_softmax + multinomial + gather."""
    # nan_to_num
    clean_logits = torch.nan_to_num(logits)
    # log_softmax
    log_probs = torch.log_softmax(clean_logits, 1)
    # multinomial sampling
    probs = log_probs.exp()
    actions = torch.multinomial(probs, 1).squeeze(1)
    # gather logprobs
    sampled_logprobs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    return [actions, sampled_logprobs]

def test_sample_logits():
    """Test sample_logits kernel.

    Verifies that:
    1. Actions are valid indices
    2. Logprobs are correct for the sampled actions (match log_softmax gather)
    3. Value is correctly copied (handles strided input)
    """
    logits = torch.randn(BR, A).cuda()
    value = torch.randn(BR, 1).cuda()  # (B, 1) like fused decoder output
    seed = 42
    offset = torch.zeros(1, dtype=torch.int64, device='cuda')  # Tensor for CUDA graph support

    # Pre-allocate output tensors (kernel writes directly to these)
    actions = torch.empty(BR, dtype=torch.float64, device='cuda')
    logprobs = torch.empty(BR, dtype=logits.dtype, device='cuda')
    value_out = torch.empty(BR, dtype=logits.dtype, device='cuda')

    print('sample_logits')

    # Run kernel (writes to actions, logprobs, value_out in-place)
    _C.sample_logits(logits, value, actions, logprobs, value_out, seed, offset)

    # Verify actions are valid indices (float64 but should be integer values)
    valid_actions = actions.min() >= 0 and actions.max() < A
    action_color = 'green' if valid_actions else 'red'
    rich.print(f'\tActions valid: [{action_color}]{valid_actions}[/{action_color}]')
    assert valid_actions, "Actions contain invalid values"

    # Verify logprobs match log_softmax gather
    log_probs = torch.log_softmax(torch.nan_to_num(logits), 1)
    # Convert float64 actions to int64 for indexing
    actions_int = actions.long()
    expected_logprobs = log_probs.gather(1, actions_int.unsqueeze(1)).squeeze(1)
    logprob_max_diff = (expected_logprobs - logprobs.float()).abs().max()
    logprob_match = torch.allclose(expected_logprobs, logprobs.float(), rtol=1e-3, atol=1e-4)
    match_color = 'green' if logprob_match else 'red'
    rich.print(f'\tLogprobs = log_softmax[action]: [{match_color}]{logprob_match} (max diff: {logprob_max_diff:.2e})[/{match_color}]')

    # Verify value copy
    expected_value = value.flatten()
    value_match = torch.allclose(expected_value, value_out, rtol=1e-5, atol=1e-6)
    value_color = 'green' if value_match else 'red'
    rich.print(f'\tValue copy: [{value_color}]{value_match}[/{value_color}]')

    # Benchmark
    py_sps = time_sps(sample_logits_py, logits)
    # Wrapper for benchmarking with in-place signature
    def cpp_sample(logits):
        _C.sample_logits(logits, value, actions, logprobs, value_out, seed, offset)
        # Offset increment is now fused into kernel
        return [actions, logprobs]
    cpp_sps = time_sps(cpp_sample, logits)
    print(f'\tForward sps: {py_sps} (naive) {cpp_sps} (C++)')

if __name__ == '__main__':
    #test_mingru_gate()
    #test_log_coeffs_and_values()
    #test_logcumsumexp()
    #test_fused_scan()
    #test_fused_ppo_loss()
    test_sample_logits()
    #test_rmsnorm()
