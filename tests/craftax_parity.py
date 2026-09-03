#!/usr/bin/env python3
"""Compare ocean/craftax against original JAX Craftax-Symbolic-v1.

https://github.com/MichaelTMatthews/Craftax

The reference is the symbolic env (8268-d one-hot obs from
render_craftax_symbolic), not the pixels env. Those observations are packed
into the 9x11x8+51 layout used by ocean/craftax.

Checks:
1. Reset worldgen (maps, items, lights, ladders, starter stats)
2. Packed symbolic observations after reset and after a shared action sequence

JAX sleep/rest is collapsed into one agent step to match puf_step. Rewards are
not compared: clean dropped health shaping, adds armour delta, and uses -1 on
death.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnvNoAutoReset
from craftax.craftax_env import make_craftax_env_from_name


MAP_SIZE = 48
NUM_LEVELS = 9
NUM_POTIONS = 6
MAP_CELLS = NUM_LEVELS * MAP_SIZE * MAP_SIZE
OBS_ROWS = 9
OBS_COLS = 11
NUM_MOB_CLASSES = 5
NUM_MOB_TYPES = 8
NUM_BLOCK_TYPES = 37
NUM_ITEM_TYPES = 5
INVENTORY_OBS_SIZE = 51
OBS_TILE_CHANNELS = 3 + NUM_MOB_CLASSES
OBS_SIZE = OBS_ROWS * OBS_COLS * OBS_TILE_CHANNELS + INVENTORY_OBS_SIZE
JAX_TILE_CHANNELS = (
    NUM_BLOCK_TYPES + NUM_ITEM_TYPES + NUM_MOB_CLASSES * NUM_MOB_TYPES + 1
)
JAX_MAP_OBS = OBS_ROWS * OBS_COLS * JAX_TILE_CHANNELS
JAX_OBS_SIZE = JAX_MAP_OBS + INVENTORY_OBS_SIZE
NUM_ACTIONS = 43

DUNGEON_FLOORS = (1, 3, 4)
SMOOTH_FLOORS = (0, 2, 5, 6, 7, 8)
BOSS_FLOOR = 8
FIRE_FLOOR = 6

BLOCK_WATER = 3
BLOCK_STONE = 4
BLOCK_TREE = 5
BLOCK_PATH = 7
BLOCK_LAVA = 14
BLOCK_DARKNESS = 18
BLOCK_CHEST = 23
BLOCK_FOUNTAIN = 24
BLOCK_FIRE_GRASS = 25
BLOCK_FIRE_TREE = 28
BLOCK_ENCHANTMENT_TABLE_FIRE = 30
BLOCK_ENCHANTMENT_TABLE_ICE = 31
BLOCK_NECROMANCER = 32
BLOCK_GRAVE = 33

ITEM_TORCH = 1
ITEM_LADDER_DOWN = 2
ITEM_LADDER_UP = 3


class WorldDump(ctypes.Structure):
    _fields_ = [
        ("map", ctypes.c_int32 * MAP_CELLS),
        ("item_map", ctypes.c_int32 * MAP_CELLS),
        ("light_map", ctypes.c_uint8 * MAP_CELLS),
        ("down_ladders", ctypes.c_int32 * (NUM_LEVELS * 2)),
        ("up_ladders", ctypes.c_int32 * (NUM_LEVELS * 2)),
        ("monsters_killed", ctypes.c_int32 * NUM_LEVELS),
        ("potion_mapping", ctypes.c_int32 * NUM_POTIONS),
        ("player_position", ctypes.c_int32 * 2),
        ("player_level", ctypes.c_int32),
        ("player_direction", ctypes.c_int32),
        ("player_health", ctypes.c_float),
        ("player_food", ctypes.c_int32),
        ("player_drink", ctypes.c_int32),
        ("player_energy", ctypes.c_int32),
        ("player_mana", ctypes.c_int32),
        ("player_dexterity", ctypes.c_int32),
        ("player_strength", ctypes.c_int32),
        ("player_intelligence", ctypes.c_int32),
        ("light_level", ctypes.c_float),
        ("boss_timesteps", ctypes.c_int32),
    ]


SOURCE = r"""
#include <stdint.h>
#include <string.h>
#include "ocean/craftax/craftax.h"

static void flatten_clean(const State* state, WorldDump* out) {
    int cell = 0;
    for (int level = 0; level < NUM_LEVELS; level++) {
        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                out->map[cell] = state->map[level][row][col];
                out->item_map[cell] = state->item_map[level][row][col];
                out->light_map[cell] = state->light_map[level][row][col];
                cell++;
            }
        }
        out->down_ladders[level * 2 + 0] = state->down_ladders[level][0];
        out->down_ladders[level * 2 + 1] = state->down_ladders[level][1];
        out->up_ladders[level * 2 + 0] = state->up_ladders[level][0];
        out->up_ladders[level * 2 + 1] = state->up_ladders[level][1];
        out->monsters_killed[level] = state->monsters_killed[level];
    }
    memcpy(out->potion_mapping, state->potion_mapping, sizeof(out->potion_mapping));
    out->player_position[0] = state->player_position[0];
    out->player_position[1] = state->player_position[1];
    out->player_level = state->player_level;
    out->player_direction = state->player_direction;
    out->player_health = state->player_health;
    out->player_food = state->player_food;
    out->player_drink = state->player_drink;
    out->player_energy = state->player_energy;
    out->player_mana = state->player_mana;
    out->player_dexterity = state->player_dexterity;
    out->player_strength = state->player_strength;
    out->player_intelligence = state->player_intelligence;
    out->light_level = state->light_level;
    out->boss_timesteps = state->boss_timestep_to_spawn_this_round;
}

void generate_clean_world(int32_t seed, WorldDump* out) {
    State state;
    Rng initial = rng_seed((uint32_t)seed);
    Rng env_rng;
    Rng reset_key;
    rng_split(initial, &env_rng, &reset_key);
    Rng unused;
    Rng world_key;
    rng_split(reset_key, &unused, &world_key);
    generate_world_from_key(&state, world_key);
    flatten_clean(&state, out);
}
"""

CLEAN_REPLAY = r"""
#include <stdint.h>
#include <string.h>
#include "ocean/craftax/craftax.h"

void replay_clean(
    int32_t seed,
    const int32_t* actions,
    int32_t num_actions,
    float* obs_out,
    float* rewards_out,
    int32_t* terminal_step
) {
    Craftax env;
    float action_value = 0.0f;
    float reward_value = 0.0f;
    float terminal_value = 0.0f;
    float live_obs[OBS_SIZE];
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.rng = (unsigned int)seed;
    env.seed = (uint64_t)(uint32_t)seed;
    env.agents[0].actions = &action_value;
    env.agents[0].rewards = &reward_value;
    env.agents[0].terminals = &terminal_value;
    env.agents[0].observations = live_obs;

    puf_reset(&env);
    memcpy(obs_out, live_obs, OBS_SIZE * sizeof(float));
    *terminal_step = -1;
    for (int32_t i = 0; i < num_actions; i++) {
        action_value = (float)actions[i];
        puf_step(&env);
        memcpy(obs_out + (size_t)(i + 1) * OBS_SIZE, live_obs, OBS_SIZE * sizeof(float));
        rewards_out[i] = env.agents[0].rewards[0];
        if (env.agents[0].terminals[0] > 0.5f) {
            *terminal_step = i;
            break;
        }
    }
}
"""


def _fill_c_array(c_arr, values, dtype):
    flat = np.ascontiguousarray(values, dtype=dtype).reshape(-1)
    if flat.size != len(c_arr):
        raise ValueError(f"size mismatch: {flat.size} vs {len(c_arr)}")
    ctypes.memmove(ctypes.addressof(c_arr), flat.ctypes.data, flat.nbytes)


def jax_state_to_dump(state) -> WorldDump:
    dump = WorldDump()
    _fill_c_array(dump.map, state.map, np.int32)
    _fill_c_array(dump.item_map, state.item_map, np.int32)
    lights = np.clip(np.asarray(state.light_map) * 255.0, 0, 255).astype(np.uint8)
    _fill_c_array(dump.light_map, lights, np.uint8)
    _fill_c_array(dump.down_ladders, state.down_ladders, np.int32)
    _fill_c_array(dump.up_ladders, state.up_ladders, np.int32)
    _fill_c_array(dump.monsters_killed, state.monsters_killed, np.int32)
    _fill_c_array(dump.potion_mapping, state.potion_mapping, np.int32)
    pos = np.asarray(state.player_position, dtype=np.int32).reshape(-1)
    dump.player_position[0] = int(pos[0])
    dump.player_position[1] = int(pos[1])
    dump.player_level = int(np.asarray(state.player_level))
    dump.player_direction = int(np.asarray(state.player_direction))
    dump.player_health = float(np.asarray(state.player_health))
    dump.player_food = int(np.asarray(state.player_food))
    dump.player_drink = int(np.asarray(state.player_drink))
    dump.player_energy = int(np.asarray(state.player_energy))
    dump.player_mana = int(np.asarray(state.player_mana))
    dump.player_dexterity = int(np.asarray(state.player_dexterity))
    dump.player_strength = int(np.asarray(state.player_strength))
    dump.player_intelligence = int(np.asarray(state.player_intelligence))
    dump.light_level = float(np.asarray(state.light_level))
    dump.boss_timesteps = int(np.asarray(state.boss_timesteps_to_spawn_this_round))
    return dump


def pack_symbolic_obs(obs) -> np.ndarray:
    """Convert JAX 8268-d one-hot symbolic obs to packed 9x11x8+51 obs."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.size != JAX_OBS_SIZE:
        raise ValueError(f"JAX obs size {obs.size}, expected {JAX_OBS_SIZE}")
    tiled = obs[:JAX_MAP_OBS].reshape(OBS_ROWS, OBS_COLS, JAX_TILE_CHANNELS)
    blocks = tiled[..., :NUM_BLOCK_TYPES]
    items = tiled[..., NUM_BLOCK_TYPES : NUM_BLOCK_TYPES + NUM_ITEM_TYPES]
    mobs = tiled[
        ...,
        NUM_BLOCK_TYPES + NUM_ITEM_TYPES : NUM_BLOCK_TYPES
        + NUM_ITEM_TYPES
        + NUM_MOB_CLASSES * NUM_MOB_TYPES,
    ].reshape(OBS_ROWS, OBS_COLS, NUM_MOB_CLASSES, NUM_MOB_TYPES)
    visible = tiled[..., -1]
    packed = np.zeros((OBS_ROWS, OBS_COLS, OBS_TILE_CHANNELS), dtype=np.float32)
    packed[..., 0] = np.argmax(blocks, axis=-1).astype(np.float32) * visible
    packed[..., 1] = (np.argmax(items, axis=-1).astype(np.float32) + 1.0) * visible
    packed[..., 2] = visible
    present = mobs.max(axis=-1)
    packed[..., 3:] = (np.argmax(mobs, axis=-1).astype(np.float32) + 1.0) * present
    return np.concatenate([packed.reshape(-1), obs[JAX_MAP_OBS:]], axis=0)


class JaxCraftax:
    def __init__(self):
        self.env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
        if not isinstance(self.env, CraftaxSymbolicEnvNoAutoReset):
            raise TypeError(
                f"expected Craftax-Symbolic-v1, got {type(self.env).__name__}"
            )
        self.params = self.env.default_params
        self._step = self.env.step

    def reset(self, seed: int):
        rng = jax.random.PRNGKey(int(seed))
        rng, reset_key = jax.random.split(rng)
        obs, state = self.env.reset(reset_key, self.params)
        return rng, obs, state

    def replay(self, seed: int, actions):
        rng, obs, state = self.reset(seed)
        packed = [pack_symbolic_obs(obs)]
        terminal_step = -1
        for i, action in enumerate(actions):
            current = int(action)
            while True:
                rng, step_key = jax.random.split(rng)
                step_rng, reset_key = jax.random.split(step_key)
                obs, state, _reward, done, _info = self._step(
                    step_rng, state, jnp.int32(current), self.params
                )
                done = bool(np.asarray(done))
                sleeping = bool(np.asarray(state.is_sleeping)) or bool(
                    np.asarray(state.is_resting)
                )
                if done or not sleeping:
                    break
                current = 0
            if done:
                # puf_step generates a new world from this tick's reset_key.
                obs, state = self.env.reset(reset_key, self.params)
                packed.append(pack_symbolic_obs(obs))
                terminal_step = i
                break
            packed.append(pack_symbolic_obs(obs))
        return packed, terminal_step


def compile_lib(root: Path) -> ctypes.CDLL:
    tmp = tempfile.TemporaryDirectory(prefix="craftax_parity_")
    src = Path(tmp.name) / "parity.c"
    so = Path(tmp.name) / "parity.so"
    compile_lib._tmp = tmp  # type: ignore[attr-defined]
    c_struct = f"""
#include <stdint.h>
typedef struct WorldDump {{
    int32_t map[{MAP_CELLS}];
    int32_t item_map[{MAP_CELLS}];
    uint8_t light_map[{MAP_CELLS}];
    int32_t down_ladders[{NUM_LEVELS * 2}];
    int32_t up_ladders[{NUM_LEVELS * 2}];
    int32_t monsters_killed[{NUM_LEVELS}];
    int32_t potion_mapping[{NUM_POTIONS}];
    int32_t player_position[2];
    int32_t player_level;
    int32_t player_direction;
    float player_health;
    int32_t player_food;
    int32_t player_drink;
    int32_t player_energy;
    int32_t player_mana;
    int32_t player_dexterity;
    int32_t player_strength;
    int32_t player_intelligence;
    float light_level;
    int32_t boss_timesteps;
}} WorldDump;
"""
    src.write_text(c_struct + SOURCE)
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(root),
            "-I",
            str(root / "src"),
            "-I",
            str(root / "ocean" / "craftax"),
            "-I",
            str(root / "raylib-5.5_linux_amd64/include"),
            str(src),
            str(root / "raylib-5.5_linux_amd64/lib/libraylib.a"),
            "-lm",
            "-lpthread",
            "-lGL",
            "-ldl",
            "-o",
            str(so),
        ],
        check=True,
        cwd=root,
    )
    lib = ctypes.CDLL(str(so))
    lib.generate_clean_world.argtypes = [ctypes.c_int32, ctypes.POINTER(WorldDump)]
    lib.generate_clean_world.restype = None
    return lib


def compile_replay_lib(root: Path) -> ctypes.CDLL:
    tmp = tempfile.TemporaryDirectory(prefix="craftax_replay_")
    compile_replay_lib._tmp = tmp  # type: ignore[attr-defined]
    src = Path(tmp.name) / "replay_clean.c"
    so = Path(tmp.name) / "replay_clean.so"
    src.write_text(CLEAN_REPLAY)
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(root),
            "-I",
            str(root / "src"),
            "-I",
            str(root / "ocean" / "craftax"),
            "-I",
            str(root / "raylib-5.5_linux_amd64/include"),
            str(src),
            str(root / "raylib-5.5_linux_amd64/lib/libraylib.a"),
            "-lm",
            "-lpthread",
            "-lGL",
            "-ldl",
            "-o",
            str(so),
        ],
        check=True,
        cwd=root,
    )
    clean = ctypes.CDLL(str(so))
    clean.replay_clean.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
    ]
    clean.replay_clean.restype = None
    return clean


def obs_section(index: int) -> str:
    map_size = OBS_ROWS * OBS_COLS * OBS_TILE_CHANNELS
    if index < map_size:
        cell = index // OBS_TILE_CHANNELS
        channel = index % OBS_TILE_CHANNELS
        names = [
            "block",
            "item",
            "visible",
            "melee",
            "passive",
            "ranged",
            "mob_proj",
            "player_proj",
        ]
        name = names[channel] if channel < len(names) else str(channel)
        return f"map cell={cell} channel={name}"
    return f"scalar[{index - map_size}]"


def first_obs_mismatch(clean, jax_obs, atol: float):
    clean = np.asarray(clean, dtype=np.float32).reshape(-1)
    jax_obs = np.asarray(jax_obs, dtype=np.float32).reshape(-1)
    diff = np.abs(clean - jax_obs)
    index = int(np.argmax(diff))
    max_diff = float(diff[index])
    if max_diff <= atol:
        return None
    return index, max_diff, float(clean[index]), float(jax_obs[index])


def first_mismatch(left, right, name: str):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return f"{name}[{index}] clean={a} jax={b}"
    return None


def compare_dumps(clean: WorldDump, jax_dump: WorldDump) -> list[str]:
    diffs = []
    for name, _ctype in WorldDump._fields_:
        left = getattr(clean, name)
        right = getattr(jax_dump, name)
        if hasattr(left, "__len__"):
            mismatch = first_mismatch(left, right, name)
            if mismatch:
                diffs.append(mismatch)
        elif left != right:
            diffs.append(f"{name} clean={left} jax={right}")
    return diffs


def count_block(dump: WorldDump, level: int, block: int) -> int:
    start = level * MAP_SIZE * MAP_SIZE
    end = start + MAP_SIZE * MAP_SIZE
    return sum(1 for value in dump.map[start:end] if value == block)


def count_item(dump: WorldDump, level: int, item: int) -> int:
    start = level * MAP_SIZE * MAP_SIZE
    end = start + MAP_SIZE * MAP_SIZE
    return sum(1 for value in dump.item_map[start:end] if value == item)


def min_light(dump: WorldDump, level: int) -> int:
    start = level * MAP_SIZE * MAP_SIZE
    end = start + MAP_SIZE * MAP_SIZE
    return min(dump.light_map[start:end])


def mean_light(dump: WorldDump, level: int) -> float:
    start = level * MAP_SIZE * MAP_SIZE
    end = start + MAP_SIZE * MAP_SIZE
    values = dump.light_map[start:end]
    return sum(values) / float(len(values))


def check_structure(dump: WorldDump) -> list[str]:
    errors = []
    if dump.player_position[0] != MAP_SIZE // 2 or dump.player_position[1] != MAP_SIZE // 2:
        errors.append("player does not spawn at map center")
    if dump.monsters_killed[0] != 10:
        errors.append("overworld is not pre-cleared")
    if dump.player_health != 9.0 or dump.player_food != 9:
        errors.append("starter vitals are not 9")

    for level in DUNGEON_FLOORS:
        if count_block(dump, level, BLOCK_DARKNESS) == 0:
            errors.append(f"floor {level} is missing BLOCK_DARKNESS")
        if count_block(dump, level, BLOCK_CHEST) < 6:
            errors.append(
                f"floor {level} has {count_block(dump, level, BLOCK_CHEST)} chests, expected at least 6"
            )
        if count_item(dump, level, ITEM_TORCH) < 16:
            errors.append(f"floor {level} is missing room-corner torches")
        if min_light(dump, level) != 255:
            errors.append(f"floor {level} is not fully lit")
        if count_item(dump, level, ITEM_LADDER_UP) != 1:
            errors.append(f"floor {level} is missing an up ladder")
        if count_item(dump, level, ITEM_LADDER_DOWN) != 1:
            errors.append(f"floor {level} is missing a down ladder")

    if count_block(dump, 1, BLOCK_FOUNTAIN) == 0:
        errors.append("dungeon is missing a fountain")
    if count_block(dump, 3, BLOCK_ENCHANTMENT_TABLE_ICE) == 0:
        errors.append("sewers are missing the ice enchantment table")
    if count_block(dump, 4, BLOCK_ENCHANTMENT_TABLE_FIRE) == 0:
        errors.append("vault is missing the fire enchantment table")

    if mean_light(dump, 0) < 250:
        errors.append("overworld is not fully lit")
    if mean_light(dump, FIRE_FLOOR) < 250:
        errors.append("fire realm is not bright")
    if mean_light(dump, BOSS_FLOOR) > 80:
        errors.append("boss floor is not dark")
    if count_block(dump, BOSS_FLOOR, BLOCK_NECROMANCER) == 0:
        errors.append("boss floor is missing the necromancer")
    if count_item(dump, BOSS_FLOOR, ITEM_LADDER_UP) != 0:
        errors.append("boss floor should not have an up ladder")
    if count_item(dump, BOSS_FLOOR, ITEM_LADDER_DOWN) != 0:
        errors.append("boss floor should not have a down ladder")
    if count_block(dump, FIRE_FLOOR, BLOCK_FIRE_GRASS) == 0:
        errors.append("fire realm is missing fire grass")
    return errors


def replay_seed(clean_lib, jax_env: JaxCraftax, seed: int, actions, atol: float):
    num_actions = len(actions)
    action_array = (ctypes.c_int32 * num_actions)(*actions)
    obs_count = (num_actions + 1) * OBS_SIZE
    clean_obs = (ctypes.c_float * obs_count)()
    clean_rewards = (ctypes.c_float * num_actions)()
    clean_done = ctypes.c_int32(-1)

    clean_lib.replay_clean(
        seed,
        action_array,
        num_actions,
        clean_obs,
        clean_rewards,
        ctypes.byref(clean_done),
    )
    jax_obs, jax_done = jax_env.replay(seed, actions)

    compare_steps = min(len(jax_obs), num_actions + 1)
    if clean_done.value >= 0 or jax_done >= 0:
        compare_steps = min(
            clean_done.value if clean_done.value >= 0 else num_actions,
            jax_done if jax_done >= 0 else num_actions,
        ) + 2
        compare_steps = min(compare_steps, len(jax_obs), num_actions + 1)

    for step in range(compare_steps):
        start = step * OBS_SIZE
        end = start + OBS_SIZE
        mismatch = first_obs_mismatch(clean_obs[start:end], jax_obs[step], atol)
        if mismatch is not None:
            index, abs_diff, clean_value, jax_value = mismatch
            return (
                f"obs mismatch seed={seed} step={step} "
                f"index={index} section={obs_section(index)} "
                f"abs_diff={abs_diff:.8g} clean={clean_value:.8g} jax={jax_value:.8g}"
            )
    if clean_done.value != jax_done:
        return (
            f"terminal mismatch seed={seed} "
            f"clean_done={clean_done.value} jax_done={jax_done}"
        )
    return None


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    print(f"Compiling craftax worldgen harness in {root}")
    lib = compile_lib(root)
    print("Compiling craftax replay harness")
    clean_lib = compile_replay_lib(root)
    print("Loading JAX Craftax-Symbolic-v1")
    jax_env = JaxCraftax()
    print(f"Reference env: {type(jax_env.env).__name__}")

    failures = 0
    for seed in range(args.seeds):
        clean = WorldDump()
        lib.generate_clean_world(seed, ctypes.byref(clean))
        _rng, _obs, jax_state = jax_env.reset(seed)
        jax_dump = jax_state_to_dump(jax_state)

        diffs = compare_dumps(clean, jax_dump)
        struct_errors = check_structure(clean)
        jax_struct_errors = [f"jax {error}" for error in check_structure(jax_dump)]
        rng = random.Random(args.action_seed + seed)
        actions = [rng.randrange(NUM_ACTIONS) for _ in range(args.steps)]
        obs_error = replay_seed(clean_lib, jax_env, seed, actions, args.atol)

        if diffs or struct_errors or jax_struct_errors or obs_error:
            failures += 1
            print(f"FAIL seed={seed}")
            for diff in diffs[:8]:
                print(f"  {diff}")
            for error in struct_errors:
                print(f"  {error}")
            for error in jax_struct_errors:
                print(f"  {error}")
            if obs_error:
                print(f"  {obs_error}")
            if not args.keep_going:
                return 1
        else:
            print(
                f"PASS seed={seed} "
                f"dungeon_dark={count_block(clean, 1, BLOCK_DARKNESS)} "
                f"dungeon_chests={count_block(clean, 1, BLOCK_CHEST)} "
                f"boss_light={mean_light(clean, BOSS_FLOOR):.1f} "
                f"steps={args.steps}"
            )

    passed = args.seeds - failures
    if failures:
        print(
            f"FAIL craftax vs JAX parity: "
            f"{passed}/{args.seeds} passed, {failures}/{args.seeds} diverged"
        )
        return 1

    sample = WorldDump()
    lib.generate_clean_world(0, ctypes.byref(sample))
    print(
        "PASS craftax vs JAX Craftax: "
        f"seeds={args.seeds} steps={args.steps} atol={args.atol} "
        f"dungeon_chests={count_block(sample, 1, BLOCK_CHEST)} "
        f"dungeon_darkness={count_block(sample, 1, BLOCK_DARKNESS)} "
        f"sewer_ice_table={count_block(sample, 3, BLOCK_ENCHANTMENT_TABLE_ICE)} "
        f"vault_fire_table={count_block(sample, 4, BLOCK_ENCHANTMENT_TABLE_FIRE)} "
        f"fire_light={mean_light(sample, FIRE_FLOOR):.1f} "
        f"boss_light={mean_light(sample, BOSS_FLOOR):.1f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
