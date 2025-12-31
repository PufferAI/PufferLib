import time
import rich

import torch

import pufferlib
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')


B = 512
T = 64
H = 128
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
    # Warm up
    for i in range(3):
        outputs = func(*args)
        if backward:
            test_loss(outputs).backward()

    torch.cuda.synchronize()
    start = time.time()
    steps = 0
    while time.time() - start < TIMEOUT:
        steps += 1
        outputs = func(*args)
        if backward:
            test_loss(outputs).backward()

    torch.cuda.synchronize()
    sps = B*T*steps/(time.time() - start)
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
    state = torch.randn(B, T, H)
    gate = torch.randn(B, T, H)
    hidden = torch.randn(B, T, H)
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
    gate = torch.randn(B, T, H, requires_grad=True)
    hidden = torch.randn(B, T, H, requires_grad=True)
    print('log_coeffs_and_values')
    test_kernel(log_coeffs_and_values, _C.log_coeffs_and_values, gate, hidden)

def fused_scan(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1)
    log_h = a_star + log_h0_plus_b_star
    out = log_h.exp()
    return [out]

def fused_scan_loss(outputs):
    return torch.sum(outputs[0])

def test_fused_scan():
    # Numerically unstable function. Must be called with the distribution
    # that is used in the full network.
    log_coeffs = -torch.nn.functional.softplus(torch.randn(B, T+1, H, requires_grad=True))
    log_values = -torch.nn.functional.softplus(torch.randn(B, T+1, H, requires_grad=True))

    print('fused_scan')
    test_kernel(fused_scan, _C.fused_scan, log_coeffs, log_values)

def logcumsumexp(x):
    return [torch.log(torch.exp(x).cumsum(1))]

def logcumsumexp_loss(outputs):
    return torch.sum(outputs[0])

def test_logcumsumexp():
    x = torch.randn(B, T, H, requires_grad=True)
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
    A = 4
    logits = torch.randn(B, T, A, requires_grad=True)
    values_pred = torch.randn(B, T, requires_grad=True).contiguous()
    actions = torch.randint(0, A, (B, T))
    old_logprobs = torch.randn(B, T)
    advantages = torch.randn(B, T)
    prio = torch.rand(B)
    values = torch.randn(B, T)
    returns = torch.randn(B, T)

    adv_mean = advantages.mean()
    adv_std = advantages.std()
    clip_coef = torch.tensor(0.1)
    vf_clip_coef = torch.tensor(0.1)
    vf_coef = torch.tensor(0.1)
    ent_coef = torch.tensor(0.1)

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
    x = torch.randn(B, T, H, requires_grad=True)
    weight = torch.randn(H, requires_grad=True)
    eps = 1e-5

    print('rmsnorm correctness')
    test_kernel(rmsnorm, _C.rmsnorm, x, weight, eps)

if __name__ == '__main__':
    test_mingru_gate()
    test_log_coeffs_and_values()
    test_logcumsumexp()
    test_fused_scan()
    test_fused_ppo_loss()
    #test_rmsnorm()
