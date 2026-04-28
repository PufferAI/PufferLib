from types import SimpleNamespace

from tests import craftax_parity


def test_craftax_full_native_step_parity():
    args = SimpleNamespace(
        seeds=16,
        seed_start=0,
        steps=2000,
        action_seed=0,
        atol=1e-5,
    )
    assert craftax_parity.run(args) == 0
