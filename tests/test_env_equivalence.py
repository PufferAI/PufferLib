import numpy as np
import pytest

# /* UNVERIFIED: these class names are guesses. If your original env filename differs,
#    update the import path accordingly. */
try:
    from envs.grid_flappy import GridFlappy as Original
except Exception:
    Original = None

try:
    from envs.grid_flappy_puffer import GridEnv as PufferEnv
except Exception:
    PufferEnv = None

@pytest.mark.skipif(Original is None or PufferEnv is None, reason="envs not present")
def test_step_equivalence():
    orig = Original(seed=42)
    puf = PufferEnv(seed=42)
    o1 = orig.reset()
    o2 = puf.reset()
    assert np.array(o1).shape == np.array(o2).shape
    # run deterministic steps
    for _ in range(100):
        a = 0
        r1_out = orig.step(a)
        r2_out = puf.step(a)
        # normalize outputs to (obs, reward, done, info)
        def norm(x):
            if len(x) == 4:
                return x
            elif len(x) == 5:
                obs, term, trunc, info = x[0], x[2], x[3], x[4] if len(x) > 4 else {}
                done = term or trunc
                return (obs, x[1], done, info)
            else:
                raise AssertionError("unexpected return shape")
        o1, r1, d1, _ = norm(r1_out)
        o2, r2, d2, _ = norm(r2_out)
        # if numeric arrays
        try:
            assert np.allclose(np.array(o1), np.array(o2))
        except Exception:
            assert type(o1) == type(o2)
        assert r1 == r2
        assert d1 == d2
