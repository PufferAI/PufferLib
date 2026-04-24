import argparse
import ctypes
import subprocess
import tempfile
from pathlib import Path

import numpy as np


FULL_BLOCK_CHANNELS = 37
FULL_ITEM_CHANNELS = 5
FULL_MOB_TYPES = 8
NUM_MOB_CLASSES = 5
OBS_ROWS = 9
OBS_COLS = 11
INVENTORY_OBS_SIZE = 51
NUM_TILE_CHANNELS = (
    FULL_BLOCK_CHANNELS
    + FULL_ITEM_CHANNELS
    + NUM_MOB_CLASSES * FULL_MOB_TYPES
    + 1
)
MAP_OBS_SIZE = OBS_ROWS * OBS_COLS * NUM_TILE_CHANNELS
PACKED_TILE_CHANNELS = 3 + NUM_MOB_CLASSES
PACKED_MAP_OBS_SIZE = OBS_ROWS * OBS_COLS * PACKED_TILE_CHANNELS
PACKED_OBS_SIZE = PACKED_MAP_OBS_SIZE + INVENTORY_OBS_SIZE


WRAPPER_SOURCE = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "raylib.h"
#include "vecenv.h"

typedef int cudaError_t;
typedef int cudaMemcpyKind;
cudaError_t cudaHostAlloc(void**, size_t, unsigned int) { return 0; }
cudaError_t cudaMalloc(void**, size_t) { return 0; }
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) { return 0; }
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) { return 0; }
cudaError_t cudaMemset(void*, int, size_t) { return 0; }
cudaError_t cudaFree(void*) { return 0; }
cudaError_t cudaFreeHost(void*) { return 0; }
cudaError_t cudaSetDevice(int) { return 0; }
cudaError_t cudaDeviceSynchronize(void) { return 0; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return 0; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned int) { return 0; }
cudaError_t cudaStreamQuery(cudaStream_t) { return 0; }
const char* cudaGetErrorString(cudaError_t) { return "stub"; }

Image LoadImageFromMemory(const char* fileType, const unsigned char* fileData, int dataSize) {
    Image img = {0};
    return img;
}
Texture2D LoadTextureFromImage(Image image) {
    Texture2D tex = {0};
    return tex;
}
void SetTextureFilter(Texture2D texture, int filter) { (void)texture; (void)filter; }
void DrawTexturePro(Texture2D texture, Rectangle source, Rectangle dest, Vector2 origin, float rotation, Color tint) {
    (void)texture; (void)source; (void)dest; (void)origin; (void)rotation; (void)tint;
}
void InitWindow(int width, int height, const char* title) {
    (void)width; (void)height; (void)title;
}
void SetTargetFPS(int fps) { (void)fps; }
void BeginDrawing(void) {}
void ClearBackground(Color color) { (void)color; }
void DrawText(const char* text, int posX, int posY, int fontSize, Color color) {
    (void)text; (void)posX; (void)posY; (void)fontSize; (void)color;
}
void DrawRectangle(int posX, int posY, int width, int height, Color color) {
    (void)posX; (void)posY; (void)width; (void)height; (void)color;
}
void EndDrawing(void) {}
bool IsWindowReady(void) { return false; }
bool IsKeyDown(int key) { (void)key; return false; }
const char* TextFormat(const char* text, ...) { return text; }

StaticVec* craftax_diff_create_vec(
    int total_agents,
    int num_buffers,
    uint64_t seed_offset,
    int reset_pool_size
) {
    Dict* vec_kwargs = create_dict(3);
    dict_set(vec_kwargs, "total_agents", (double)total_agents);
    dict_set(vec_kwargs, "num_buffers", (double)num_buffers);
    dict_set(vec_kwargs, "num_threads", 1.0);

    Dict* env_kwargs = create_dict(2);
    dict_set(env_kwargs, "seed_offset", (double)seed_offset);
    dict_set(env_kwargs, "reset_pool_size", (double)reset_pool_size);

    StaticVec* vec = create_static_vec(
        total_agents,
        num_buffers,
        0,
        vec_kwargs,
        env_kwargs
    );
    static_vec_reset(vec);

    free(vec_kwargs->items);
    free(vec_kwargs);
    free(env_kwargs->items);
    free(env_kwargs);
    return vec;
}

void craftax_diff_step_vec(StaticVec* vec, const int32_t* actions) {
    for (int i = 0; i < vec->total_agents; i++) {
        vec->actions[i] = (float)actions[i];
    }
    cpu_vec_step(vec);
}

float* craftax_diff_obs_ptr(StaticVec* vec) {
    return (float*)vec->observations;
}

float* craftax_diff_rewards_ptr(StaticVec* vec) {
    return vec->rewards;
}

float* craftax_diff_terminals_ptr(StaticVec* vec) {
    return vec->terminals;
}

int craftax_diff_obs_size(void) {
    return get_obs_size();
}

int craftax_diff_num_actions(void) {
    int* act_sizes = get_act_sizes();
    int n = get_num_act_sizes();
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += act_sizes[i];
    }
    return total;
}

void craftax_diff_close_vec(StaticVec* vec) {
    static_vec_close(vec);
}
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_root(path: str | None) -> Path:
    if path is None:
        return repo_root()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (repo_root() / resolved).resolve()
    return resolved


def _resolve_craftax_dir(root: Path, craftax_dir: str) -> Path:
    path = Path(craftax_dir)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _describe_obs_index(index: int) -> str:
    if index < MAP_OBS_SIZE:
        tile = index // NUM_TILE_CHANNELS
        channel = index % NUM_TILE_CHANNELS
        row = tile // OBS_COLS
        col = tile % OBS_COLS
        if channel < FULL_BLOCK_CHANNELS:
            return f"map.block_onehot[{row},{col},{channel}]"
        channel -= FULL_BLOCK_CHANNELS
        if channel < FULL_ITEM_CHANNELS:
            return f"map.item_onehot[{row},{col},{channel}]"
        channel -= FULL_ITEM_CHANNELS
        if channel < NUM_MOB_CLASSES * FULL_MOB_TYPES:
            mob_class = channel // FULL_MOB_TYPES
            mob_type = channel % FULL_MOB_TYPES
            return f"map.mob_onehot[{row},{col},class={mob_class},type={mob_type}]"
        return f"map.visible[{row},{col}]"

    inventory_index = index - MAP_OBS_SIZE
    if 0 <= inventory_index < INVENTORY_OBS_SIZE:
        return f"inventory[{inventory_index}]"
    return f"obs[{index}]"


def _first_bitwise_float_diff(ref: np.ndarray, got: np.ndarray):
    ref_u32 = np.ascontiguousarray(ref).view(np.uint32).reshape(-1)
    got_u32 = np.ascontiguousarray(got).view(np.uint32).reshape(-1)
    mismatch = np.flatnonzero(ref_u32 != got_u32)
    if mismatch.size == 0:
        return None
    idx = int(mismatch[0])
    return idx, ref_u32[idx], got_u32[idx], float(ref.reshape(-1)[idx]), float(got.reshape(-1)[idx])


def _expand_packed_craftax_obs(packed: np.ndarray) -> np.ndarray:
    if packed.shape[1] != PACKED_OBS_SIZE:
        raise ValueError(f"packed obs has size {packed.shape[1]}, expected {PACKED_OBS_SIZE}")

    full = np.zeros((packed.shape[0], MAP_OBS_SIZE + INVENTORY_OBS_SIZE), dtype=np.float32)
    for env_i in range(packed.shape[0]):
        for cell in range(OBS_ROWS * OBS_COLS):
            packed_base = cell * PACKED_TILE_CHANNELS
            full_base = cell * NUM_TILE_CHANNELS
            block = int(packed[env_i, packed_base + 0])
            item = int(packed[env_i, packed_base + 1])
            visible = int(packed[env_i, packed_base + 2])
            if visible:
                if 0 <= block < FULL_BLOCK_CHANNELS:
                    full[env_i, full_base + block] = 1.0
                item_id = item - 1
                if 0 <= item_id < FULL_ITEM_CHANNELS:
                    full[env_i, full_base + FULL_BLOCK_CHANNELS + item_id] = 1.0
                full[env_i, full_base + NUM_TILE_CHANNELS - 1] = 1.0

            mob_offset = full_base + FULL_BLOCK_CHANNELS + FULL_ITEM_CHANNELS
            for mob_class in range(NUM_MOB_CLASSES):
                mob_value = int(packed[env_i, packed_base + 3 + mob_class])
                if mob_value:
                    mob_type = mob_value - 1
                    if 0 <= mob_type < FULL_MOB_TYPES:
                        full[
                            env_i,
                            mob_offset + mob_class * FULL_MOB_TYPES + mob_type,
                        ] = 1.0

    full[:, MAP_OBS_SIZE:] = packed[:, PACKED_MAP_OBS_SIZE:]
    return full


def _candidate_obs_for_compare(candidate: "DiffVec", candidate_obs_format: str) -> np.ndarray:
    if candidate_obs_format == "full":
        return candidate.obs
    if candidate_obs_format == "packed_float":
        return _expand_packed_craftax_obs(candidate.obs)
    raise ValueError(f"unknown candidate obs format {candidate_obs_format!r}")


class DiffLib:
    def __init__(self, root: Path, craftax_dir: Path, label: str):
        self.root = root
        self.craftax_dir = craftax_dir
        self.label = label
        self._tmp = tempfile.TemporaryDirectory(prefix=f"craftax_diff_{label}_")
        tmp = Path(self._tmp.name)
        src = tmp / "craftax_diff_wrapper.c"
        so = tmp / f"craftax_diff_{label}.so"
        src.write_text(WRAPPER_SOURCE)

        raylib_include = root / "raylib-5.5_linux_amd64/include"
        compile_cmd = [
            "cc",
            "-std=c99",
            "-O2",
            "-shared",
            "-fPIC",
            "-ffunction-sections",
            "-fdata-sections",
            "-fopenmp",
            str(src),
            str(craftax_dir / "binding.c"),
            "-I",
            str(root / "src"),
            "-I",
            str(root),
            "-I",
            str(root / "vendor"),
            "-I",
            str(craftax_dir),
            "-I",
            str(raylib_include),
            "-lm",
            "-lpthread",
            "-Wl,--gc-sections",
            "-o",
            str(so),
        ]
        subprocess.run(compile_cmd, check=True, cwd=root)
        self.lib = ctypes.CDLL(str(so))

        self.lib.craftax_diff_create_vec.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        self.lib.craftax_diff_create_vec.restype = ctypes.c_void_p
        self.lib.craftax_diff_step_vec.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        self.lib.craftax_diff_step_vec.restype = None
        self.lib.craftax_diff_obs_ptr.argtypes = [ctypes.c_void_p]
        self.lib.craftax_diff_obs_ptr.restype = ctypes.c_void_p
        self.lib.craftax_diff_rewards_ptr.argtypes = [ctypes.c_void_p]
        self.lib.craftax_diff_rewards_ptr.restype = ctypes.c_void_p
        self.lib.craftax_diff_terminals_ptr.argtypes = [ctypes.c_void_p]
        self.lib.craftax_diff_terminals_ptr.restype = ctypes.c_void_p
        self.lib.craftax_diff_obs_size.argtypes = []
        self.lib.craftax_diff_obs_size.restype = ctypes.c_int
        self.lib.craftax_diff_num_actions.argtypes = []
        self.lib.craftax_diff_num_actions.restype = ctypes.c_int
        self.lib.craftax_diff_close_vec.argtypes = [ctypes.c_void_p]
        self.lib.craftax_diff_close_vec.restype = None


class DiffVec:
    def __init__(
        self,
        lib: DiffLib,
        total_agents: int,
        num_buffers: int,
        seed_offset: int,
        reset_pool_size: int,
    ):
        self.lib = lib
        self.total_agents = total_agents
        self.ptr = lib.lib.craftax_diff_create_vec(
            total_agents,
            num_buffers,
            ctypes.c_uint64(seed_offset),
            reset_pool_size,
        )
        if not self.ptr:
            raise RuntimeError(f"{lib.label}: failed to create vec")

        self.obs_size = int(lib.lib.craftax_diff_obs_size())
        self.num_actions = int(lib.lib.craftax_diff_num_actions())

        obs_ptr = int(lib.lib.craftax_diff_obs_ptr(self.ptr))
        rewards_ptr = int(lib.lib.craftax_diff_rewards_ptr(self.ptr))
        terminals_ptr = int(lib.lib.craftax_diff_terminals_ptr(self.ptr))

        obs_t = ctypes.c_float * (total_agents * self.obs_size)
        rewards_t = ctypes.c_float * total_agents
        terminals_t = ctypes.c_float * total_agents

        self.obs = np.ctypeslib.as_array(obs_t.from_address(obs_ptr)).reshape(total_agents, self.obs_size)
        self.rewards = np.ctypeslib.as_array(rewards_t.from_address(rewards_ptr))
        self.terminals = np.ctypeslib.as_array(terminals_t.from_address(terminals_ptr))

    def step(self, actions: np.ndarray) -> None:
        if actions.dtype != np.int32:
            actions = actions.astype(np.int32, copy=False)
        self.lib.lib.craftax_diff_step_vec(
            self.ptr,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )

    def close(self) -> None:
        if self.ptr:
            self.lib.lib.craftax_diff_close_vec(self.ptr)
            self.ptr = None


def _save_trace(path: Path, actions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, actions)


def _load_or_generate_actions(
    num_steps: int,
    num_envs: int,
    num_actions: int,
    action_seed: int,
    trace_in: str | None,
) -> np.ndarray:
    if trace_in is not None:
        trace = np.load(trace_in)
        trace = np.asarray(trace, dtype=np.int32)
        expected = (num_steps, num_envs)
        if trace.shape != expected:
            raise ValueError(f"--trace-in has shape {trace.shape}, expected {expected}")
        return trace
    rng = np.random.default_rng(action_seed)
    return rng.integers(0, num_actions, size=(num_steps, num_envs), dtype=np.int32)


def run_diff(args) -> int:
    baseline_root = _resolve_root(args.baseline_root)
    candidate_root = _resolve_root(args.candidate_root) if args.candidate_root else baseline_root
    baseline_dir = _resolve_craftax_dir(baseline_root, args.baseline_craftax_dir)
    candidate_dir = _resolve_craftax_dir(candidate_root, args.candidate_craftax_dir)

    baseline_lib = DiffLib(baseline_root, baseline_dir, "baseline")
    candidate_lib = DiffLib(candidate_root, candidate_dir, "candidate")

    baseline = DiffVec(
        baseline_lib,
        args.seeds,
        args.num_buffers,
        args.seed_start,
        args.reset_pool_size,
    )
    candidate = DiffVec(
        candidate_lib,
        args.seeds,
        args.num_buffers,
        args.seed_start,
        args.reset_pool_size,
    )

    try:
        candidate_obs_format = getattr(args, "candidate_obs_format", "full")
        if candidate_obs_format == "full" and baseline.obs_size != candidate.obs_size:
            raise RuntimeError(
                f"obs_size mismatch: baseline={baseline.obs_size} candidate={candidate.obs_size}"
            )
        if baseline.num_actions != candidate.num_actions:
            raise RuntimeError(
                f"num_actions mismatch: baseline={baseline.num_actions} candidate={candidate.num_actions}"
            )

        actions = _load_or_generate_actions(
            num_steps=args.steps,
            num_envs=args.seeds,
            num_actions=baseline.num_actions,
            action_seed=args.action_seed,
            trace_in=args.trace_in,
        )
        if args.trace_out is not None:
            _save_trace(Path(args.trace_out), actions)

        candidate_obs = _candidate_obs_for_compare(candidate, candidate_obs_format)
        if baseline.obs_size != candidate_obs.shape[1]:
            raise RuntimeError(
                "expanded obs_size mismatch: "
                f"baseline={baseline.obs_size} candidate={candidate_obs.shape[1]}"
            )

        init_obs_diff = _first_bitwise_float_diff(baseline.obs, candidate_obs)
        if init_obs_diff is not None:
            idx, ref_bits, got_bits, ref_value, got_value = init_obs_diff
            env_i = idx // baseline.obs_size
            obs_i = idx % baseline.obs_size
            print(
                "RESET DIVERGENCE "
                f"env={env_i} seed={args.seed_start + env_i} "
                f"obs_index={obs_i} section={_describe_obs_index(obs_i)} "
                f"baseline_bits=0x{ref_bits:08x} candidate_bits=0x{got_bits:08x} "
                f"baseline={ref_value:.8g} candidate={got_value:.8g}"
            )
            return 1

        for step in range(args.steps):
            step_actions = actions[step]
            baseline.step(step_actions)
            candidate.step(step_actions)

            reward_diff = _first_bitwise_float_diff(baseline.rewards, candidate.rewards)
            if reward_diff is not None:
                env_i, ref_bits, got_bits, ref_value, got_value = reward_diff
                print(
                    "REWARD DIVERGENCE "
                    f"step={step} env={env_i} seed={args.seed_start + env_i} "
                    f"action={int(step_actions[env_i])} "
                    f"baseline_bits=0x{ref_bits:08x} candidate_bits=0x{got_bits:08x} "
                    f"baseline={ref_value:.8g} candidate={got_value:.8g}"
                )
                if args.divergence_trace is not None:
                    _save_trace(Path(args.divergence_trace), actions[: step + 1])
                return 1

            done_diff = _first_bitwise_float_diff(baseline.terminals, candidate.terminals)
            if done_diff is not None:
                env_i, ref_bits, got_bits, ref_value, got_value = done_diff
                print(
                    "TERMINAL DIVERGENCE "
                    f"step={step} env={env_i} seed={args.seed_start + env_i} "
                    f"action={int(step_actions[env_i])} "
                    f"baseline_bits=0x{ref_bits:08x} candidate_bits=0x{got_bits:08x} "
                    f"baseline={ref_value:.8g} candidate={got_value:.8g}"
                )
                if args.divergence_trace is not None:
                    _save_trace(Path(args.divergence_trace), actions[: step + 1])
                return 1

            candidate_obs = _candidate_obs_for_compare(candidate, candidate_obs_format)
            obs_diff = _first_bitwise_float_diff(baseline.obs, candidate_obs)
            if obs_diff is not None:
                idx, ref_bits, got_bits, ref_value, got_value = obs_diff
                env_i = idx // baseline.obs_size
                obs_i = idx % baseline.obs_size
                print(
                    "OBS DIVERGENCE "
                    f"step={step} env={env_i} seed={args.seed_start + env_i} "
                    f"action={int(step_actions[env_i])} "
                    f"obs_index={obs_i} section={_describe_obs_index(obs_i)} "
                    f"baseline_bits=0x{ref_bits:08x} candidate_bits=0x{got_bits:08x} "
                    f"baseline={ref_value:.8g} candidate={got_value:.8g}"
                )
                if args.divergence_trace is not None:
                    _save_trace(Path(args.divergence_trace), actions[: step + 1])
                return 1

        print(
            "PASS craftax diff "
            f"seeds={args.seeds} steps={args.steps} action_seed={args.action_seed} "
            f"reset_pool_size={args.reset_pool_size} "
            f"baseline={baseline_dir} candidate={candidate_dir}"
        )
        return 0
    finally:
        baseline.close()
        candidate.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--reset-pool-size", type=int, default=1024)
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument("--baseline-root", type=str, default=None)
    parser.add_argument("--candidate-root", type=str, default=None)
    parser.add_argument("--baseline-craftax-dir", type=str, default="ocean/craftax")
    parser.add_argument("--candidate-craftax-dir", type=str, default="ocean/craftax")
    parser.add_argument(
        "--candidate-obs-format",
        choices=["full", "packed_float"],
        default="full",
    )
    parser.add_argument("--trace-in", type=str, default=None)
    parser.add_argument("--trace-out", type=str, default=None)
    parser.add_argument(
        "--divergence-trace",
        type=str,
        default="build/craftax_diff_divergence.npy",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.num_buffers <= 0:
        raise ValueError("--num-buffers must be positive")
    return run_diff(args)


def test_craftax_diff_self() -> None:
    args = argparse.Namespace(
        seeds=4,
        seed_start=0,
        steps=64,
        action_seed=7,
        reset_pool_size=1024,
        num_buffers=1,
        baseline_root=None,
        candidate_root=None,
        baseline_craftax_dir="ocean/craftax",
        candidate_craftax_dir="ocean/craftax",
        candidate_obs_format="full",
        trace_in=None,
        trace_out=None,
        divergence_trace="build/test_craftax_diff_divergence.npy",
    )
    assert run_diff(args) == 0


def test_craftax_diff_moonshot_self() -> None:
    args = argparse.Namespace(
        seeds=4,
        seed_start=0,
        steps=64,
        action_seed=13,
        reset_pool_size=1024,
        num_buffers=1,
        baseline_root=None,
        candidate_root=None,
        baseline_craftax_dir="ocean/craftax_moonshot",
        candidate_craftax_dir="ocean/craftax_moonshot",
        candidate_obs_format="full",
        trace_in=None,
        trace_out=None,
        divergence_trace="build/test_craftax_moonshot_diff_divergence.npy",
    )
    assert run_diff(args) == 0


if __name__ == "__main__":
    raise SystemExit(main())
