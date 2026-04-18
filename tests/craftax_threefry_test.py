import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def build_threefry_lib():
    source = r"""
    #include <stdint.h>
    #include <stddef.h>
    #include "ocean/craftax/threefry.h"

    void key_from_seed(uint32_t seed, uint32_t* out) {
        CraftaxThreefryKey key = craftax_prng_key(seed);
        out[0] = key.word[0];
        out[1] = key.word[1];
    }

    void split_n(uint32_t key0, uint32_t key1, size_t count, uint32_t* out) {
        CraftaxThreefryKey key = {{key0, key1}};
        CraftaxThreefryKey keys[64];
        craftax_threefry_split_n(key, keys, count);
        for (size_t i = 0; i < count; i++) {
            out[2 * i + 0] = keys[i].word[0];
            out[2 * i + 1] = keys[i].word[1];
        }
    }

    void fold_in(uint32_t key0, uint32_t key1, uint32_t data, uint32_t* out) {
        CraftaxThreefryKey key = {{key0, key1}};
        CraftaxThreefryKey folded = craftax_threefry_fold_in(key, data);
        out[0] = folded.word[0];
        out[1] = folded.word[1];
    }

    uint32_t uniform_u32(uint32_t key0, uint32_t key1) {
        CraftaxThreefryKey key = {{key0, key1}};
        return craftax_threefry_uniform_u32(key);
    }
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    src = tmp_path / "threefry_test.c"
    so = tmp_path / "threefry_test.so"
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
            "-o",
            str(so),
        ],
        check=True,
        cwd=ROOT,
    )
    lib = ctypes.CDLL(str(so))
    lib._tmpdir = tmp
    lib.key_from_seed.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    lib.split_n.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.fold_in.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.uniform_u32.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    lib.uniform_u32.restype = ctypes.c_uint32
    return lib


def test_threefry_matches_jax_prng_key_split_fold_in_and_bits():
    lib = build_threefry_lib()
    seeds = [
        0,
        1,
        2,
        3,
        7,
        17,
        123,
        999,
        65535,
        65536,
        2**31 - 1,
        2**32 - 1,
    ]
    fold_data = [0, 1, 2, 31, 12345, 2**31, 2**32 - 1]

    for seed in seeds:
        expected_key = np.asarray(jax.random.PRNGKey(seed), dtype=np.uint32)
        key_out = (ctypes.c_uint32 * 2)()
        lib.key_from_seed(seed, key_out)
        c_key = np.frombuffer(key_out, dtype=np.uint32).copy()
        np.testing.assert_array_equal(c_key, expected_key, err_msg=f"PRNGKey({seed})")

        expected_bits = np.asarray(
            jax.random.bits(expected_key, (), dtype=np.uint32),
            dtype=np.uint32,
        ).reshape(())
        c_bits = np.uint32(lib.uniform_u32(int(c_key[0]), int(c_key[1])))
        assert c_bits == expected_bits, f"uniform_u32 seed={seed}"

        for count in [2, 3, 7, 16]:
            split_out = (ctypes.c_uint32 * (count * 2))()
            lib.split_n(int(c_key[0]), int(c_key[1]), count, split_out)
            c_split = np.frombuffer(split_out, dtype=np.uint32).copy().reshape(count, 2)
            expected_split = np.asarray(
                jax.random.split(expected_key, count),
                dtype=np.uint32,
            )
            np.testing.assert_array_equal(
                c_split,
                expected_split,
                err_msg=f"split seed={seed} count={count}",
            )

        for data in fold_data:
            fold_out = (ctypes.c_uint32 * 2)()
            lib.fold_in(int(c_key[0]), int(c_key[1]), data, fold_out)
            c_fold = np.frombuffer(fold_out, dtype=np.uint32).copy()
            expected_fold = np.asarray(
                jax.random.fold_in(expected_key, data),
                dtype=np.uint32,
            )
            np.testing.assert_array_equal(
                c_fold,
                expected_fold,
                err_msg=f"fold_in seed={seed} data={data}",
            )
