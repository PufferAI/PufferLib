import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.world_gen.world_gen import generate_world


ROOT = Path(__file__).resolve().parents[1]
MAP_SIZE = 48
CELLS = MAP_SIZE * MAP_SIZE


def build_worldgen_lib():
    source = r"""
    #include <stdint.h>
    #include <string.h>
    #include "ocean/craftax/worldgen.h"

    void overworld_from_seed(
        uint32_t seed,
        int32_t* map,
        int32_t* item_map,
        float* light_map,
        int32_t* ladder_down,
        int32_t* ladder_up
    ) {
        CraftaxOverworldFloor floor;
        craftax_generate_overworld_from_seed(seed, &floor);
        memcpy(map, floor.map, sizeof(floor.map));
        memcpy(item_map, floor.item_map, sizeof(floor.item_map));
        memcpy(light_map, floor.light_map, sizeof(floor.light_map));
        ladder_down[0] = floor.ladder_down[0];
        ladder_down[1] = floor.ladder_down[1];
        ladder_up[0] = floor.ladder_up[0];
        ladder_up[1] = floor.ladder_up[1];
    }
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    src = tmp_path / "worldgen_test.c"
    so = tmp_path / "worldgen_test.so"
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
    lib.overworld_from_seed.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    return lib


def jax_floor0(seed):
    rng = jax.random.PRNGKey(seed)
    _rng, reset_key = jax.random.split(rng)
    _rng, world_key = jax.random.split(reset_key)
    state = generate_world(world_key, EnvParams(), StaticEnvParams())
    return (
        np.asarray(state.map[0], dtype=np.int32),
        np.asarray(state.item_map[0], dtype=np.int32),
        np.asarray(state.light_map[0], dtype=np.float32),
        np.asarray(state.down_ladders[0], dtype=np.int32),
        np.asarray(state.up_ladders[0], dtype=np.int32),
    )


def c_floor0(lib, seed):
    map_out = np.empty((MAP_SIZE, MAP_SIZE), dtype=np.int32)
    item_out = np.empty((MAP_SIZE, MAP_SIZE), dtype=np.int32)
    light_out = np.empty((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    down = np.empty((2,), dtype=np.int32)
    up = np.empty((2,), dtype=np.int32)
    lib.overworld_from_seed(
        seed,
        map_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        item_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        light_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        down.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        up.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    return map_out, item_out, light_out, down, up


def test_native_floor0_overworld_matches_jax_default_worldgen():
    lib = build_worldgen_lib()
    for seed in range(16):
        expected = jax_floor0(seed)
        got = c_floor0(lib, seed)
        np.testing.assert_array_equal(got[0], expected[0], err_msg=f"map seed={seed}")
        np.testing.assert_array_equal(
            got[1],
            expected[1],
            err_msg=f"item_map seed={seed}",
        )
        np.testing.assert_allclose(
            got[2],
            expected[2],
            atol=1e-6,
            rtol=0.0,
            err_msg=f"light_map seed={seed}",
        )
        np.testing.assert_array_equal(
            got[3],
            expected[3],
            err_msg=f"ladder_down seed={seed}",
        )
        np.testing.assert_array_equal(
            got[4],
            expected[4],
            err_msg=f"ladder_up seed={seed}",
        )
