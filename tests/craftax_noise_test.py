import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from craftax.craftax.util.noise import generate_fractal_noise_2d


ROOT = Path(__file__).resolve().parents[1]

# C libm and XLA may differ by a few ulps for sin/cos. The generator is still
# soft-parity close enough for thresholded worldgen, which is tested separately.
NOISE_ATOL = 2e-6
NOISE_RTOL = 2e-6


def build_noise_lib():
    source = r"""
    #include <stdint.h>
    #include "ocean/craftax/noise.h"

    void fractal_noise(
        uint32_t key0,
        uint32_t key1,
        int rows,
        int cols,
        int res_rows,
        int res_cols,
        int octaves,
        float persistence,
        int lacunarity,
        float* out
    ) {
        CraftaxThreefryKey key = {{key0, key1}};
        craftax_generate_fractal_noise_2d(
            key,
            rows,
            cols,
            res_rows,
            res_cols,
            octaves,
            persistence,
            lacunarity,
            NULL,
            out
        );
    }
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    src = tmp_path / "noise_test.c"
    so = tmp_path / "noise_test.so"
    src.write_text(source)
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(ROOT),
            str(src),
            "-lm",
            "-o",
            str(so),
        ],
        check=True,
        cwd=ROOT,
    )
    lib = ctypes.CDLL(str(so))
    lib._tmpdir = tmp
    lib.fractal_noise.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    return lib


def c_fractal_noise(lib, key, shape, res, octaves=1, persistence=0.5, lacunarity=2):
    out = np.empty(shape, dtype=np.float32)
    key = np.asarray(key, dtype=np.uint32)
    lib.fractal_noise(
        int(key[0]),
        int(key[1]),
        shape[0],
        shape[1],
        res[0],
        res[1],
        octaves,
        ctypes.c_float(persistence),
        lacunarity,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


def test_fractal_noise_matches_jax_soft_parity():
    lib = build_noise_lib()
    cases = [
        ((48, 48), (3, 3), 1),
        ((48, 48), (12, 12), 1),
        ((48, 48), (6, 24), 1),
        ((32, 32), (4, 4), 2),
    ]
    seeds = [0, 1, 17, 123, 2**32 - 1]

    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        for shape, res, octaves in cases:
            expected = np.asarray(
                generate_fractal_noise_2d(key, shape, res, octaves=octaves),
                dtype=np.float32,
            )
            got = c_fractal_noise(lib, key, shape, res, octaves=octaves)
            np.testing.assert_allclose(
                got,
                expected,
                atol=NOISE_ATOL,
                rtol=NOISE_RTOL,
                err_msg=f"seed={seed} shape={shape} res={res} octaves={octaves}",
            )
