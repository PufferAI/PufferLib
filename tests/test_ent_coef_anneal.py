"""
Regression test for entropy-coefficient annealing in the CUDA backend.

We can't read the kernel's ent_coef directly, but the kernel computes

    total = policy + vf_coef*value - ent_coef*entropy

and logs all four terms, so the *effective* ent_coef it used each epoch is

    ent_eff = (policy + vf_coef*value - total) / entropy

With the bug ent_eff is frozen (std == 0); once fixed it tracks the cosine.
"""
import math
import sys

import pytest

torch = pytest.importorskip("torch")

try:
    from pufferlib import _C
except Exception as exc:  # backend not built
    pytest.skip(f"pufferlib._C not built: {exc}", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("no CUDA device available", allow_module_level=True)

ENV_NAME = getattr(_C, "env_name", None)
if not ENV_NAME:
    pytest.skip("pufferlib._C has no compiled env_name", allow_module_level=True)

BASE_ENT = 0.5      # large base so the sweep is easy to see
MIN_RATIO = 0.02    # decay down to 2% of base
TARGET_EPOCHS = 20  # enough real epochs to span most of the cosine


def _load_args():
    # load_config() parses sys.argv via argparse; keep it clean under pytest.
    from pufferlib.pufferl import load_config
    saved = sys.argv
    sys.argv = ["pytest"]
    try:
        return load_config(ENV_NAME)
    finally:
        sys.argv = saved


def _cosine(t, T, base, mn):
    if T == 0:
        return base
    r = min(1.0, max(0.0, t / T))
    return mn + 0.5 * (base - mn) * (1.0 + math.cos(math.pi * r))


def _std(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _run_and_recover(anneal):
    """Train for a few epochs and return (effective_ent_coef, schedule) per epoch."""
    from pufferlib.pufferl import _resolve_backend

    args = _load_args()
    args["train"]["anneal_ent_coef"] = int(anneal)
    args["train"]["ent_coef"] = BASE_ENT
    args["train"]["min_ent_coef_ratio"] = MIN_RATIO
    total_agents = int(args["vec"]["total_agents"])
    horizon = int(args["train"]["horizon"])
    # Keep the env's own total_agents (buffer/worker math depends on it); just
    # run few epochs by shrinking total_timesteps.
    args["train"]["total_timesteps"] = TARGET_EPOCHS * total_agents * horizon

    vf_coef = float(args["train"]["vf_coef"])
    total_epochs = args["train"]["total_timesteps"] // (total_agents * horizon)

    backend = _resolve_backend(args)
    assert backend is _C, "expected the CUDA backend for this test"

    pufferl = backend.create_pufferl(args)
    eff, sched = [], []
    try:
        for epoch in range(total_epochs):
            backend.rollouts(pufferl)
            backend.train(pufferl)
            loss = backend.log(pufferl)["loss"]
            entropy = loss["entropy"]
            if abs(entropy) < 1e-6:
                continue
            eff.append((loss["policy"] + vf_coef * loss["value"] - loss["total"]) / entropy)
            sched.append(_cosine(epoch, total_epochs, BASE_ENT, BASE_ENT * MIN_RATIO))
    finally:
        backend.close(pufferl)

    # Drop the first couple recorded epochs: the loss accumulator still carries
    # warmup residue right after the captured graph takes over.
    return eff[2:], sched[2:]


def test_ent_coef_anneals_through_cuda_graph():
    eff, sched = _run_and_recover(anneal=True)
    assert len(eff) >= 8, "not enough training epochs recorded"

    # (1) The core regression: the buggy build freezes ent_coef inside the
    #     captured CUDA graph, giving std == 0 across epochs.
    assert _std(eff) > 0.05, (
        f"effective ent_coef is frozen (std={_std(eff):.4f}); the annealed value "
        "is not reaching the loss kernel -- likely baked into the captured CUDA graph"
    )

    # (2) It should actually follow the cosine schedule the host computes.
    mae = sum(abs(e - s) for e, s in zip(eff, sched)) / len(eff)
    assert mae < 0.03, f"effective ent_coef does not track the cosine schedule (MAE={mae:.4f})"

    # (3) And it should decay well below the base value by the end.
    assert min(eff) < 0.5 * BASE_ENT, f"ent_coef never decayed (min={min(eff):.4f}, base={BASE_ENT})"


def test_ent_coef_constant_when_not_annealing():
    # Control: with annealing off, the same recovered quantity should sit flat at
    # the configured base. This also validates that the recovery method itself is
    # sound (a frozen reading is only meaningful if a non-frozen one is possible).
    eff, _sched = _run_and_recover(anneal=False)
    assert len(eff) >= 8, "not enough training epochs recorded"
    assert _std(eff) < 0.02, f"ent_coef drifted with annealing off (std={_std(eff):.4f})"
    mean = sum(eff) / len(eff)
    assert abs(mean - BASE_ENT) < 0.03, f"constant ent_coef {mean:.4f} does not match base {BASE_ENT}"


if __name__ == "__main__":
    test_ent_coef_anneals_through_cuda_graph()
    print("test_ent_coef_anneals_through_cuda_graph PASS")
    test_ent_coef_constant_when_not_annealing()
    print("test_ent_coef_constant_when_not_annealing PASS")
