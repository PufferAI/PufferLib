import argparse
import ctypes
import os
import subprocess
import tempfile
from collections import deque
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
try:
    from craftax_state_fixtures import (
        CraftaxState,
        craftax_state_to_jax,
        flatten_env_state,
    )
except ModuleNotFoundError:
    from tests.craftax_state_fixtures import (
        CraftaxState,
        craftax_state_to_jax,
        flatten_env_state,
    )


OBS_SIZE = 8268
NUM_ACTIONS = 43

OBS_ROWS = 9
OBS_COLS = 11
NUM_BLOCK_TYPES = 37
NUM_ITEM_TYPES = 5
NUM_MOB_CLASSES = 5
NUM_MOB_TYPES = 8
NUM_TILE_CHANNELS = NUM_BLOCK_TYPES + NUM_ITEM_TYPES + NUM_MOB_CLASSES * NUM_MOB_TYPES + 1
MAP_OBS_SIZE = OBS_ROWS * OBS_COLS * NUM_TILE_CHANNELS
MAP_SIZE = 48
NUM_LEVELS = 9
MONSTERS_KILLED_TO_CLEAR_LEVEL = 8

NOOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
DO = 5
PLACE_STONE = 7
PLACE_TABLE = 8
PLACE_FURNACE = 9
MAKE_WOOD_PICKAXE = 11
MAKE_STONE_PICKAXE = 12
MAKE_IRON_PICKAXE = 13
MAKE_WOOD_SWORD = 14
MAKE_STONE_SWORD = 15
MAKE_IRON_SWORD = 16
DESCEND = 18
MAKE_DIAMOND_PICKAXE = 20
MAKE_DIAMOND_SWORD = 21
MAKE_IRON_ARMOUR = 22
MAKE_DIAMOND_ARMOUR = 23
SHOOT_ARROW = 24
MAKE_ARROW = 25
CAST_FIREBALL = 26
CAST_ICEBALL = 27
PLACE_TORCH = 28
MAKE_TORCH = 38

BLOCK_WATER = 3
BLOCK_LAVA = 14
ITEM_LADDER_DOWN = 2

MOVE_ACTIONS = np.asarray([LEFT, RIGHT, UP, DOWN], dtype=np.int32)
DIRS = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP: (-1, 0),
    DOWN: (1, 0),
}

SOLID_BLOCKS = frozenset(
    [
        4,
        5,
        8,
        9,
        10,
        11,
        12,
        15,
        16,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        28,
        30,
        31,
        32,
        33,
        34,
        35,
    ]
)

INVENTORY_OBS_NAMES = [
    "inventory.wood",
    "inventory.stone",
    "inventory.coal",
    "inventory.iron",
    "inventory.diamond",
    "inventory.sapphire",
    "inventory.ruby",
    "inventory.sapling",
    "inventory.torches",
    "inventory.arrows",
    "inventory.books",
    "inventory.pickaxe",
    "inventory.sword",
    "sword_enchantment",
    "bow_enchantment",
    "inventory.bow",
    "inventory.potions.red",
    "inventory.potions.green",
    "inventory.potions.blue",
    "inventory.potions.pink",
    "inventory.potions.cyan",
    "inventory.potions.yellow",
    "player_health",
    "player_food",
    "player_drink",
    "player_energy",
    "player_mana",
    "player_xp",
    "player_dexterity",
    "player_strength",
    "player_intelligence",
    "direction.left",
    "direction.right",
    "direction.up",
    "direction.down",
    "inventory.armour.0",
    "inventory.armour.1",
    "inventory.armour.2",
    "inventory.armour.3",
    "armour_enchantments.0",
    "armour_enchantments.1",
    "armour_enchantments.2",
    "armour_enchantments.3",
    "light_level",
    "is_sleeping",
    "is_resting",
    "learned_spells.fireball",
    "learned_spells.iceball",
    "player_level",
    "ladder_down_open",
    "boss_vulnerable",
]

MOB_CLASS_NAMES = [
    "melee_mobs",
    "passive_mobs",
    "ranged_mobs",
    "mob_projectiles",
    "player_projectiles",
]

POLICIES = ("uniform", "combat", "descend", "suicide", "boss", "mixed")
MIXED_ORDER = ("uniform", "combat", "descend", "suicide", "boss")


def _preload_nccl():
    root = Path(__file__).resolve().parents[1]
    nccl = root / ".venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
    if nccl.exists():
        ctypes.CDLL(str(nccl), mode=ctypes.RTLD_GLOBAL)


def import_c_env():
    _preload_nccl()
    import pufferlib._C as cmod

    env_name = getattr(cmod, "env_name", None)
    if env_name != "craftax":
        raise RuntimeError(
            f"pufferlib._C is compiled for {env_name!r}, expected 'craftax'. "
            "Run: uv run --with pybind11 --with rich_argparse ./build.sh craftax"
        )
    return cmod


def float_view(ptr, count):
    array_t = ctypes.c_float * count
    return np.ctypeslib.as_array(array_t.from_address(ptr))


def _stack_states(states):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)


class JaxCraftaxBatch:
    def __init__(self, seeds, resetter=None):
        self.env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
        self.params = self.env.default_params
        self.num_envs = len(seeds)
        self.resetter = resetter
        self.reset_keys = []
        rngs = []
        states = []
        obs = []
        for seed in seeds:
            rng = jax.random.PRNGKey(int(seed))
            rng, reset_key = jax.random.split(rng)
            env_obs, state = self.env.reset(reset_key, self.params)
            rngs.append(rng)
            self.reset_keys.append(np.asarray(reset_key, dtype=np.uint32))
            states.append(state)
            obs.append(np.asarray(env_obs, dtype=np.float32).reshape(-1))

        self.rngs = jnp.stack(rngs)
        self.states = _stack_states(states)
        self.obs = np.stack(obs, axis=0)
        self._step_batch = self._make_step_batch()

    def _make_step_batch(self):
        env = self.env
        params = self.params

        def step_one(key, state, action):
            step_rng, reset_key = jax.random.split(key, 2)
            obs, next_state, reward, done, _info = env.step(
                step_rng,
                state,
                action,
                params,
            )
            return obs, next_state, reward, done, reset_key

        def step_batch(rngs, states, actions):
            split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(rngs)
            next_rngs = split_keys[:, 0]
            step_keys = split_keys[:, 1]
            obs, next_states, rewards, dones, reset_keys = jax.vmap(step_one)(
                step_keys, states, actions
            )
            return next_rngs, next_states, obs, rewards, dones, reset_keys

        return jax.jit(step_batch)

    def step(self, actions):
        actions = jnp.asarray(actions, dtype=jnp.int32)
        (
            self.rngs,
            self.states,
            obs,
            rewards,
            dones,
            reset_keys,
        ) = self._step_batch(self.rngs, self.states, actions)
        self.obs = np.asarray(obs, dtype=np.float32).reshape(self.num_envs, -1).copy()
        dones_np = np.asarray(dones, dtype=np.bool_)
        reset_keys_np = np.asarray(reset_keys, dtype=np.uint32)
        if self.resetter is not None and np.any(dones_np):
            for env_i, done in enumerate(dones_np):
                if not bool(done):
                    continue
                reset_state, reset_obs = self.resetter.reset(
                    reset_keys_np[env_i],
                    self.state_at(env_i),
                )
                self.states = jax.tree_util.tree_map(
                    lambda batched, value: batched.at[env_i].set(value),
                    self.states,
                    reset_state,
                )
                self.obs[env_i] = reset_obs
        return (
            self.obs,
            np.asarray(rewards, dtype=np.float32),
            dones_np,
            reset_keys_np,
        )

    def state_at(self, env_i):
        return jax.tree_util.tree_map(lambda leaf: leaf[env_i], self.states)


class PolicySnapshot:
    def __init__(self, states):
        self.level = np.asarray(states.player_level, dtype=np.int32)
        self.position = np.asarray(states.player_position, dtype=np.int32)
        self.direction = np.asarray(states.player_direction, dtype=np.int32)
        self.health = np.asarray(states.player_health, dtype=np.float32)
        self.mana = np.asarray(states.player_mana, dtype=np.int32)
        self.learned_spells = np.asarray(states.learned_spells, dtype=np.bool_)

        self.inventory = states.inventory
        self.wood = np.asarray(self.inventory.wood, dtype=np.int32)
        self.stone = np.asarray(self.inventory.stone, dtype=np.int32)
        self.coal = np.asarray(self.inventory.coal, dtype=np.int32)
        self.iron = np.asarray(self.inventory.iron, dtype=np.int32)
        self.diamond = np.asarray(self.inventory.diamond, dtype=np.int32)
        self.bow = np.asarray(self.inventory.bow, dtype=np.int32)
        self.arrows = np.asarray(self.inventory.arrows, dtype=np.int32)
        self.torches = np.asarray(self.inventory.torches, dtype=np.int32)

        num_envs = int(self.level.shape[0])
        env_idx = np.arange(num_envs)

        full_map = np.asarray(states.map, dtype=np.int32)
        full_item_map = np.asarray(states.item_map, dtype=np.int32)
        full_mob_map = np.asarray(states.mob_map, dtype=np.bool_)
        full_monsters_killed = np.asarray(states.monsters_killed, dtype=np.int32)
        full_down_ladders = np.asarray(states.down_ladders, dtype=np.int32)

        self.map = full_map[env_idx, self.level]
        self.item_map = full_item_map[env_idx, self.level]
        self.mob_map = full_mob_map[env_idx, self.level]
        self.monsters_killed = full_monsters_killed[env_idx, self.level]
        self.down_ladders = full_down_ladders[env_idx, self.level]

        self.melee_pos, self.melee_mask, self.melee_type = self._take_mobs(
            states.melee_mobs, env_idx
        )
        self.passive_pos, self.passive_mask, self.passive_type = self._take_mobs(
            states.passive_mobs, env_idx
        )
        self.ranged_pos, self.ranged_mask, self.ranged_type = self._take_mobs(
            states.ranged_mobs, env_idx
        )
        (
            self.mob_projectile_pos,
            self.mob_projectile_mask,
            self.mob_projectile_type,
        ) = self._take_mobs(states.mob_projectiles, env_idx)
        (
            self.player_projectile_pos,
            self.player_projectile_mask,
            self.player_projectile_type,
        ) = self._take_mobs(states.player_projectiles, env_idx)

    def _take_mobs(self, mobs, env_idx):
        pos = np.asarray(mobs.position, dtype=np.int32)[env_idx, self.level]
        mask = np.asarray(mobs.mask, dtype=np.bool_)[env_idx, self.level]
        type_id = np.asarray(mobs.type_id, dtype=np.int32)[env_idx, self.level]
        return pos, mask, type_id


class ResetVerifier:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        source = r"""
        #include <stdbool.h>
        #include <stdint.h>
        #define CRAFTAX_ENABLE_ENV_IMPL
        #include "ocean/craftax/craftax.h"
        #include "ocean/craftax/step_crafting.h"
        #include "ocean/craftax/step_update_mobs.h"
        #include "ocean/craftax/step_spawn_mobs.h"

        void reset_from_key(
            uint32_t key0,
            uint32_t key1,
            CraftaxState* out,
            float* obs
        ) {
            CraftaxThreefryKey reset_key = {{key0, key1}};
            craftax_reset_state_from_reset_key(out, reset_key);
            craftax_encode_native_observation(out, obs);
        }
        """
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        src = tmp_path / "craftax_reset_verify.c"
        so = tmp_path / "craftax_reset_verify.so"
        src.write_text(source)
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
                str(root / "raylib-5.5_linux_amd64/include"),
                str(src),
                "-lm",
                "-o",
                str(so),
            ],
            check=True,
            cwd=root,
        )
        self.lib = ctypes.CDLL(str(so))
        self.lib.reset_from_key.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(CraftaxState),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.reset_from_key.restype = None

    def reset(self, reset_key, template):
        c_state = CraftaxState()
        c_obs = np.empty(OBS_SIZE, dtype=np.float32)
        key = np.asarray(reset_key, dtype=np.uint32)
        self.lib.reset_from_key(
            ctypes.c_uint32(int(key[0])),
            ctypes.c_uint32(int(key[1])),
            ctypes.byref(c_state),
            c_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return craftax_state_to_jax(c_state, template=template), c_obs

    def compare(self, jax_state, jax_obs, reset_key, seed, step, policy, atol):
        c_jax_state, c_obs = self.reset(reset_key, jax_state)

        obs_diff = first_obs_diff(jax_obs, c_obs, atol)
        state_diff = first_state_diff(jax_state, c_jax_state, atol)
        if obs_diff is not None:
            idx, max_diff, jax_value, c_value = obs_diff
            key = np.asarray(reset_key, dtype=np.uint32)
            print(
                "RESET DIVERGENCE "
                f"seed={seed} step={step} policy={policy} "
                f"reset_key=[{int(key[0])},{int(key[1])}] "
                f"obs_index={idx} section={section_for_index(idx)} "
                f"subsystem={subsystem_for_section(section_for_index(idx))} "
                f"abs_diff={max_diff:.8g} jax={jax_value:.8g} c={c_value:.8g}"
            )
            if state_diff is not None:
                name, index, state_max_diff, state_jax_value, state_c_value = state_diff
                print(
                    "reset_state_first_diff: "
                    f"field={name} index={index} "
                    f"abs_diff={state_max_diff:.8g} "
                    f"jax={state_jax_value} c={state_c_value}"
                )
            return False

        if state_diff is not None:
            name, index, max_diff, jax_value, c_value = state_diff
            key = np.asarray(reset_key, dtype=np.uint32)
            print(
                "RESET STATE DIVERGENCE "
                f"seed={seed} step={step} policy={policy} "
                f"reset_key=[{int(key[0])},{int(key[1])}] "
                f"field={name} index={index} abs_diff={max_diff:.8g} "
                f"jax={jax_value} c={c_value}"
            )
            return False
        return True


_RESET_VERIFIER = None


def get_reset_verifier(enabled):
    global _RESET_VERIFIER
    if not enabled:
        return None
    if _RESET_VERIFIER is None:
        _RESET_VERIFIER = ResetVerifier()
    return _RESET_VERIFIER


def make_c_vec(cmod, num_envs, seed_offset, num_threads=1):
    args = {
        "vec": {
            "total_agents": num_envs,
            "num_buffers": 1,
            "num_threads": num_threads,
        },
        "env": {
            "seed_offset": seed_offset,
        },
    }
    vec = cmod.create_vec(args, 0)
    if vec.obs_size != OBS_SIZE:
        raise RuntimeError(f"C obs_size={vec.obs_size}, expected {OBS_SIZE}")
    if vec.num_atns != 1:
        raise RuntimeError(f"C num_atns={vec.num_atns}, expected 1")
    if list(vec.act_sizes) != [NUM_ACTIONS]:
        raise RuntimeError(f"C act_sizes={vec.act_sizes}, expected [{NUM_ACTIONS}]")
    vec.reset()
    obs = float_view(vec.obs_ptr, num_envs * OBS_SIZE).reshape(num_envs, OBS_SIZE)
    rewards = float_view(vec.rewards_ptr, num_envs)
    terminals = float_view(vec.terminals_ptr, num_envs)
    return vec, obs, rewards, terminals


def action_plan(seeds, steps, action_seed):
    rng = np.random.default_rng(action_seed)
    return rng.integers(0, NUM_ACTIONS, size=(steps, len(seeds)), dtype=np.int32)


def first_obs_diff(ref, got, atol):
    diff = np.abs(ref - got)
    idx = int(np.argmax(diff))
    max_diff = float(diff[idx])
    if max_diff <= atol:
        return None
    return idx, max_diff, float(ref[idx]), float(got[idx])


def _format_index(index):
    index = np.asarray(index)
    if index.ndim == 0:
        return "scalar"
    return ",".join(str(int(i)) for i in index)


def first_state_diff(jax_state, c_state, atol):
    jax_flat = flatten_env_state(jax_state)
    c_flat = flatten_env_state(c_state)
    if jax_flat.keys() != c_flat.keys():
        missing = sorted(jax_flat.keys() - c_flat.keys())
        extra = sorted(c_flat.keys() - jax_flat.keys())
        return "state_keys", "scalar", 1.0, f"missing_c={missing}", f"extra_c={extra}"

    for name, jax_value in jax_flat.items():
        c_value = c_flat[name]
        if np.asarray(jax_value).dtype.kind == "f":
            diff = np.abs(np.asarray(jax_value) - np.asarray(c_value))
            if diff.size == 0:
                continue
            idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
            max_diff = float(diff[idx])
            if max_diff > atol:
                return (
                    name,
                    _format_index(np.asarray(idx)),
                    max_diff,
                    float(np.asarray(jax_value)[idx]),
                    float(np.asarray(c_value)[idx]),
                )
        else:
            neq = np.asarray(jax_value) != np.asarray(c_value)
            if np.any(neq):
                idx = np.argwhere(neq)[0] if np.asarray(neq).ndim else np.asarray(())
                idx_tuple = tuple(int(i) for i in np.asarray(idx).reshape(-1))
                return (
                    name,
                    _format_index(idx),
                    1.0,
                    np.asarray(jax_value)[idx_tuple].item()
                    if idx_tuple
                    else np.asarray(jax_value).item(),
                    np.asarray(c_value)[idx_tuple].item()
                    if idx_tuple
                    else np.asarray(c_value).item(),
                )
    return None


def section_for_index(idx):
    if idx < MAP_OBS_SIZE:
        tile = idx // NUM_TILE_CHANNELS
        channel = idx % NUM_TILE_CHANNELS
        row = tile // OBS_COLS
        col = tile % OBS_COLS
        if channel < NUM_BLOCK_TYPES:
            return f"map_one_hot[row={row},col={col},block={channel}]"
        channel -= NUM_BLOCK_TYPES
        if channel < NUM_ITEM_TYPES:
            return f"item_one_hot[row={row},col={col},item={channel}]"
        channel -= NUM_ITEM_TYPES
        if channel < NUM_MOB_CLASSES * NUM_MOB_TYPES:
            mob_class = channel // NUM_MOB_TYPES
            mob_type = channel % NUM_MOB_TYPES
            return (
                f"{MOB_CLASS_NAMES[mob_class]}_type_{mob_type}"
                f"[row={row},col={col}]"
            )
        return f"light[row={row},col={col}]"

    inv_idx = idx - MAP_OBS_SIZE
    if 0 <= inv_idx < len(INVENTORY_OBS_NAMES):
        return INVENTORY_OBS_NAMES[inv_idx]
    return f"inventory_or_special[{inv_idx}]"


def subsystem_for_section(section):
    if section.startswith("map_one_hot"):
        return "symbolic_observation.map"
    if section.startswith("item_one_hot"):
        return "symbolic_observation.item_or_ladder"
    if section.startswith("melee_mobs") or section.startswith("passive_mobs"):
        return "mobs.update_or_observation"
    if section.startswith("ranged_mobs") or section.startswith("mob_projectiles"):
        return "projectiles_or_ranged_mobs"
    if section.startswith("player_projectiles"):
        return "player_projectiles"
    if section.startswith("light[") or section == "light_level":
        return "light"
    if section.startswith("inventory."):
        return "inventory"
    if section.startswith("player_"):
        return "player_intrinsics"
    if section.startswith("direction."):
        return "movement"
    if section in {"ladder_down_open", "player_level"}:
        return "floor_change"
    if section == "boss_vulnerable":
        return "boss_logic"
    return "state_or_observation"


def compare_reset(ref_obs, c_obs, seeds, atol):
    for env_i, seed in enumerate(seeds):
        diff = first_obs_diff(ref_obs[env_i], c_obs[env_i], atol)
        if diff is not None:
            idx, max_diff, ref_value, c_value = diff
            section = section_for_index(idx)
            print(
                "RESET DIVERGENCE "
                f"seed={seed} obs_index={idx} section={section} "
                f"subsystem={subsystem_for_section(section)} "
                f"abs_diff={max_diff:.8g} jax={ref_value:.8g} c={c_value:.8g}"
            )
            return False
    return True


def _in_bounds(pos):
    return 0 <= int(pos[0]) < MAP_SIZE and 0 <= int(pos[1]) < MAP_SIZE


def _action_toward_delta(delta):
    dr, dc = int(delta[0]), int(delta[1])
    if abs(dr) > abs(dc):
        return DOWN if dr > 0 else UP
    if dc != 0:
        return RIGHT if dc > 0 else LEFT
    if dr != 0:
        return DOWN if dr > 0 else UP
    return NOOP


def _action_to_neighbor(start, target):
    delta = np.asarray(target, dtype=np.int32) - np.asarray(start, dtype=np.int32)
    if abs(int(delta[0])) + abs(int(delta[1])) != 1:
        return None
    return _action_toward_delta(delta)


def _passable_map(snapshot, env_i, allow_danger=False, allow_mobs=False):
    level_map = snapshot.map[env_i]
    passable = np.ones((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
    for block in SOLID_BLOCKS:
        passable &= level_map != block
    if not allow_danger:
        passable &= level_map != BLOCK_WATER
        passable &= level_map != BLOCK_LAVA
    if not allow_mobs:
        passable &= ~snapshot.mob_map[env_i]
    return passable


def _valid_move_actions(snapshot, env_i, allow_danger=False):
    pos = snapshot.position[env_i]
    passable = _passable_map(snapshot, env_i, allow_danger=allow_danger)
    actions = []
    for action, delta in DIRS.items():
        target = pos + np.asarray(delta, dtype=np.int32)
        if _in_bounds(target) and passable[int(target[0]), int(target[1])]:
            actions.append(action)
    return actions


def _random_move(snapshot, env_i, rng, allow_danger=False):
    actions = _valid_move_actions(snapshot, env_i, allow_danger=allow_danger)
    if actions:
        return int(rng.choice(actions))
    return int(rng.choice(MOVE_ACTIONS))


def _bfs_first_action(snapshot, env_i, target, rng, allow_danger=False):
    start = tuple(int(x) for x in snapshot.position[env_i])
    target = tuple(int(x) for x in np.asarray(target, dtype=np.int32))
    if start == target:
        return NOOP

    passable = _passable_map(snapshot, env_i, allow_danger=allow_danger)
    passable[start] = True
    if not _in_bounds(target) or not passable[target]:
        return _greedy_action(snapshot, env_i, np.asarray(target), rng, allow_danger)

    visited = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
    visited[start] = True
    queue = deque()
    for action in rng.permutation(MOVE_ACTIONS):
        delta = DIRS[int(action)]
        row = start[0] + delta[0]
        col = start[1] + delta[1]
        if not (0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE):
            continue
        if visited[row, col] or not passable[row, col]:
            continue
        if (row, col) == target:
            return int(action)
        visited[row, col] = True
        queue.append((row, col, int(action)))

    while queue:
        row, col, first_action = queue.popleft()
        for action in MOVE_ACTIONS:
            delta = DIRS[int(action)]
            next_row = row + delta[0]
            next_col = col + delta[1]
            if not (0 <= next_row < MAP_SIZE and 0 <= next_col < MAP_SIZE):
                continue
            if visited[next_row, next_col] or not passable[next_row, next_col]:
                continue
            if (next_row, next_col) == target:
                return int(first_action)
            visited[next_row, next_col] = True
            queue.append((next_row, next_col, first_action))

    return _greedy_action(snapshot, env_i, np.asarray(target), rng, allow_danger)


def _greedy_action(snapshot, env_i, target, rng, allow_danger=False):
    pos = snapshot.position[env_i]
    actions = _valid_move_actions(snapshot, env_i, allow_danger=allow_danger)
    if not actions:
        return int(rng.choice(MOVE_ACTIONS))
    scored = []
    for action in actions:
        delta = np.asarray(DIRS[action], dtype=np.int32)
        next_pos = pos + delta
        dist = int(np.abs(next_pos - target).sum())
        scored.append((dist, action))
    best_dist = min(dist for dist, _action in scored)
    best = [action for dist, action in scored if dist == best_dist]
    return int(rng.choice(best))


def _nearest_target(snapshot, env_i, positions):
    if len(positions) == 0:
        return None
    pos = snapshot.position[env_i]
    positions = np.asarray(positions, dtype=np.int32)
    distances = np.abs(positions - pos).sum(axis=1)
    return positions[int(np.argmin(distances))]


def _live_mobs(snapshot, env_i, include_passive=True, include_projectiles=False):
    groups = [
        (0, snapshot.melee_pos[env_i], snapshot.melee_mask[env_i], snapshot.melee_type[env_i]),
        (2, snapshot.ranged_pos[env_i], snapshot.ranged_mask[env_i], snapshot.ranged_type[env_i]),
    ]
    if include_passive:
        groups.append(
            (
                1,
                snapshot.passive_pos[env_i],
                snapshot.passive_mask[env_i],
                snapshot.passive_type[env_i],
            )
        )
    if include_projectiles:
        groups.append(
            (
                3,
                snapshot.mob_projectile_pos[env_i],
                snapshot.mob_projectile_mask[env_i],
                snapshot.mob_projectile_type[env_i],
            )
        )

    mobs = []
    for mob_class, positions, masks, type_ids in groups:
        for index, mask in enumerate(masks):
            if bool(mask):
                mobs.append((mob_class, index, positions[index], int(type_ids[index])))
    return mobs


def _mob_positions(snapshot, env_i, include_passive=True, include_projectiles=False):
    return [
        np.asarray(position, dtype=np.int32)
        for _cls, _idx, position, _type_id in _live_mobs(
            snapshot,
            env_i,
            include_passive=include_passive,
            include_projectiles=include_projectiles,
        )
    ]


def _projectile_slot_available(snapshot, env_i):
    return int(np.count_nonzero(snapshot.player_projectile_mask[env_i])) < 3


def _target_in_current_line(snapshot, env_i, target):
    pos = snapshot.position[env_i]
    direction = int(snapshot.direction[env_i])
    delta = np.asarray(target, dtype=np.int32) - pos
    if direction == LEFT:
        return int(delta[0]) == 0 and int(delta[1]) < 0
    if direction == RIGHT:
        return int(delta[0]) == 0 and int(delta[1]) > 0
    if direction == UP:
        return int(delta[1]) == 0 and int(delta[0]) < 0
    if direction == DOWN:
        return int(delta[1]) == 0 and int(delta[0]) > 0
    return False


def _combat_action(snapshot, env_i, rng):
    pos = snapshot.position[env_i]
    mobs = _live_mobs(snapshot, env_i, include_passive=True)
    mob_positions = [mob[2] for mob in mobs]
    adjacent = [
        np.asarray(position, dtype=np.int32)
        for position in mob_positions
        if int(np.abs(np.asarray(position) - pos).sum()) == 1
    ]

    for target in adjacent:
        action = _action_to_neighbor(pos, target)
        if action == int(snapshot.direction[env_i]) and rng.random() < 0.75:
            return DO
    if adjacent:
        target = adjacent[int(rng.integers(0, len(adjacent)))]
        return int(_action_to_neighbor(pos, target))

    has_projectile_slot = _projectile_slot_available(snapshot, env_i)
    projectile_actions = []
    if has_projectile_slot and int(snapshot.bow[env_i]) >= 1 and int(snapshot.arrows[env_i]) >= 1:
        projectile_actions.append(SHOOT_ARROW)
    if has_projectile_slot and int(snapshot.mana[env_i]) >= 2:
        if bool(snapshot.learned_spells[env_i, 0]):
            projectile_actions.append(CAST_FIREBALL)
        if bool(snapshot.learned_spells[env_i, 1]):
            projectile_actions.append(CAST_ICEBALL)

    if projectile_actions and mob_positions:
        line_targets = [
            target
            for target in mob_positions
            if _target_in_current_line(snapshot, env_i, target)
        ]
        if line_targets and rng.random() < 0.8:
            return int(rng.choice(projectile_actions))

        axis_targets = [
            target
            for target in mob_positions
            if int(target[0]) == int(pos[0]) or int(target[1]) == int(pos[1])
        ]
        if axis_targets:
            target = _nearest_target(snapshot, env_i, axis_targets)
            return _action_toward_delta(target - pos)

    if mob_positions:
        target = _nearest_target(snapshot, env_i, mob_positions)
        return _bfs_first_action(snapshot, env_i, target, rng)

    return _random_move(snapshot, env_i, rng)


def _craft_or_place_action(snapshot, env_i, rng):
    options = []
    if int(snapshot.wood[env_i]) > 0:
        options.extend([PLACE_TABLE, MAKE_WOOD_PICKAXE, MAKE_WOOD_SWORD])
    if int(snapshot.stone[env_i]) > 0:
        options.append(PLACE_STONE)
    if int(snapshot.stone[env_i]) >= 4:
        options.append(PLACE_FURNACE)
    if int(snapshot.stone[env_i]) > 0 and int(snapshot.wood[env_i]) > 0:
        options.extend([MAKE_STONE_PICKAXE, MAKE_STONE_SWORD])
    if int(snapshot.iron[env_i]) > 0 and int(snapshot.wood[env_i]) > 0:
        options.extend([MAKE_IRON_PICKAXE, MAKE_IRON_SWORD, MAKE_IRON_ARMOUR])
    if int(snapshot.diamond[env_i]) > 0 and int(snapshot.wood[env_i]) > 0:
        options.extend([MAKE_DIAMOND_PICKAXE, MAKE_DIAMOND_SWORD, MAKE_DIAMOND_ARMOUR])
    if int(snapshot.wood[env_i]) > 0 and int(snapshot.stone[env_i]) > 0:
        options.append(MAKE_ARROW)
    if int(snapshot.coal[env_i]) > 0 and int(snapshot.wood[env_i]) > 0:
        options.append(MAKE_TORCH)
    if int(snapshot.torches[env_i]) > 0:
        options.append(PLACE_TORCH)
    if not options:
        return None
    return int(rng.choice(options))


def _descend_action(snapshot, env_i, rng):
    level = int(snapshot.level[env_i])
    pos = snapshot.position[env_i]
    if level >= NUM_LEVELS - 1:
        return _combat_action(snapshot, env_i, rng)

    row, col = int(pos[0]), int(pos[1])
    on_down_ladder = int(snapshot.item_map[env_i, row, col]) == ITEM_LADDER_DOWN
    ladder_open = int(snapshot.monsters_killed[env_i]) >= MONSTERS_KILLED_TO_CLEAR_LEVEL
    if on_down_ladder and ladder_open:
        return DESCEND

    mobs = _mob_positions(snapshot, env_i, include_passive=False)
    if not ladder_open and mobs:
        return _combat_action(snapshot, env_i, rng)

    if rng.random() < 0.12:
        craft_action = _craft_or_place_action(snapshot, env_i, rng)
        if craft_action is not None:
            return craft_action

    ladder = snapshot.down_ladders[env_i]
    if ladder_open:
        return _bfs_first_action(snapshot, env_i, ladder, rng)

    if mobs:
        return _combat_action(snapshot, env_i, rng)
    return _random_move(snapshot, env_i, rng)


def _danger_adjacent_action(snapshot, env_i, rng):
    pos = snapshot.position[env_i]
    level_map = snapshot.map[env_i]
    dangerous_actions = []
    for action, delta in DIRS.items():
        target = pos + np.asarray(delta, dtype=np.int32)
        if not _in_bounds(target):
            continue
        block = int(level_map[int(target[0]), int(target[1])])
        if block in (BLOCK_WATER, BLOCK_LAVA) or bool(
            snapshot.mob_map[env_i, int(target[0]), int(target[1])]
        ):
            dangerous_actions.append(action)
    if dangerous_actions:
        return int(rng.choice(dangerous_actions))
    return None


def _suicide_action(snapshot, env_i, rng):
    adjacent = _danger_adjacent_action(snapshot, env_i, rng)
    if adjacent is not None:
        return adjacent

    hostile_positions = _mob_positions(
        snapshot, env_i, include_passive=False, include_projectiles=True
    )
    danger_blocks = np.argwhere(
        (snapshot.map[env_i] == BLOCK_LAVA) | (snapshot.map[env_i] == BLOCK_WATER)
    )

    targets = []
    targets.extend(hostile_positions)
    if danger_blocks.size:
        targets.extend([danger_blocks[i] for i in range(danger_blocks.shape[0])])

    target = _nearest_target(snapshot, env_i, targets)
    if target is None:
        return _random_move(snapshot, env_i, rng, allow_danger=True)

    if int(np.abs(target - snapshot.position[env_i]).sum()) == 1:
        return _action_toward_delta(target - snapshot.position[env_i])

    passable = _passable_map(snapshot, env_i, allow_danger=False)
    adjacent_cells = []
    for delta in DIRS.values():
        cell = target + np.asarray(delta, dtype=np.int32)
        if _in_bounds(cell) and passable[int(cell[0]), int(cell[1])]:
            adjacent_cells.append(cell)
    adjacent_target = _nearest_target(snapshot, env_i, adjacent_cells)
    if adjacent_target is not None:
        return _bfs_first_action(snapshot, env_i, adjacent_target, rng)
    return _greedy_action(snapshot, env_i, target, rng, allow_danger=True)


def _boss_action(snapshot, env_i, rng, step):
    if step < 1000:
        return _descend_action(snapshot, env_i, rng)
    level = int(snapshot.level[env_i])
    if level >= NUM_LEVELS - 1:
        return _combat_action(snapshot, env_i, rng)
    pos = snapshot.position[env_i]
    on_down_ladder = int(snapshot.item_map[env_i, int(pos[0]), int(pos[1])]) == ITEM_LADDER_DOWN
    ladder_open = int(snapshot.monsters_killed[env_i]) >= MONSTERS_KILLED_TO_CLEAR_LEVEL
    if on_down_ladder and ladder_open:
        return DESCEND
    if rng.random() < 0.25:
        return DESCEND
    return _descend_action(snapshot, env_i, rng)


class ActionPolicy:
    def __init__(self, policy, action_seed, num_envs):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        self.policy = policy
        self.rng = np.random.default_rng(action_seed)
        self.num_envs = num_envs

    def effective_policy(self, step):
        if self.policy != "mixed":
            return self.policy
        return MIXED_ORDER[(step // 500) % len(MIXED_ORDER)]

    def actions(self, step, ref):
        policy = self.effective_policy(step)
        if policy == "uniform":
            return (
                self.rng.integers(0, NUM_ACTIONS, size=self.num_envs, dtype=np.int32),
                policy,
            )

        snapshot = PolicySnapshot(ref.states)
        out = np.empty(self.num_envs, dtype=np.int32)
        for env_i in range(self.num_envs):
            if policy == "combat":
                out[env_i] = _combat_action(snapshot, env_i, self.rng)
            elif policy == "descend":
                out[env_i] = _descend_action(snapshot, env_i, self.rng)
            elif policy == "suicide":
                out[env_i] = _suicide_action(snapshot, env_i, self.rng)
            elif policy == "boss":
                out[env_i] = _boss_action(snapshot, env_i, self.rng, step)
            else:
                raise AssertionError(policy)
        return out, policy


def _print_step_divergence(
    seed,
    step,
    action,
    policy_name,
    reward_diff,
    ref_reward,
    c_reward,
    ref_done,
    c_done,
    obs_diff,
    history,
):
    terminal_delta = int(bool(c_done)) - int(bool(ref_done))
    print(
        "STEP DIVERGENCE "
        f"seed={seed} step={step} action={int(action)} policy={policy_name}"
    )
    print(
        f"reward_delta={reward_diff:.8g} "
        f"reward: jax={float(ref_reward):.8g} c={float(c_reward):.8g}"
    )
    print(
        f"terminal_delta={terminal_delta} "
        f"done: jax={bool(ref_done)} c={bool(c_done)}"
    )
    if obs_diff is None:
        print("obs: ok")
    else:
        idx, max_diff, ref_value, c_value = obs_diff
        section = section_for_index(idx)
        print(
            "obs: "
            f"index={idx} section={section} "
            f"subsystem={subsystem_for_section(section)} "
            f"abs_diff={max_diff:.8g} "
            f"jax={ref_value:.8g} c={c_value:.8g}"
        )
    print(f"last_10_actions={list(history)}")


def _print_terminal_reset_check(
    reset_verifier,
    ref,
    ref_obs,
    reset_key,
    env_i,
    seed,
    step,
    policy_name,
    atol,
):
    if reset_verifier is None:
        return True
    key = np.asarray(reset_key, dtype=np.uint32)
    ok = reset_verifier.compare(
        ref.state_at(env_i),
        ref_obs[env_i],
        reset_key,
        int(seed),
        step,
        policy_name,
        atol,
    )
    if ok:
        print(
            "terminal_reset_reference: ok "
            f"reset_key=[{int(key[0])},{int(key[1])}]"
        )
    return ok


def _terminal_summary(seeds, terminal_counts, episode_length_sums):
    total_terminals = int(np.sum(terminal_counts))
    per_seed = []
    for seed, count, length_sum in zip(seeds, terminal_counts, episode_length_sums):
        if int(count) > 0:
            mean_len = float(length_sum) / float(count)
            per_seed.append(f"{int(seed)}:{int(count)}@{mean_len:.1f}")
        else:
            per_seed.append(f"{int(seed)}:0")
    return total_terminals, " ".join(per_seed)


def _diagnose_isolated_replay(cmod, seed, actions, atol, num_threads, reset_verifier):
    print(
        "isolated_replay: start "
        f"seed={int(seed)} steps={len(actions)}"
    )
    trace_path = Path("build") / f"craftax_repro_seed_{int(seed)}_steps_{len(actions)}.txt"
    trace_path.parent.mkdir(exist_ok=True)
    trace_path.write_text("\n".join(str(int(action)) for action in actions) + "\n")
    print(f"isolated_replay_actions={trace_path}")
    ref = JaxCraftaxBatch(np.asarray([seed], dtype=np.int64), resetter=reset_verifier)
    vec, c_obs, c_rewards, c_terminals = make_c_vec(
        cmod,
        1,
        int(seed),
        num_threads=num_threads,
    )
    try:
        if not compare_reset(ref.obs, c_obs.copy(), np.asarray([seed]), atol):
            print("isolated_replay: initial reset diverged")
            return
        action_buf = np.zeros((1, 1), dtype=np.float32)
        for step, action in enumerate(actions):
            action_buf[0, 0] = float(action)
            ref_obs, ref_rewards, ref_dones, reset_keys = ref.step(
                np.asarray([action], dtype=np.int32)
            )
            vec.cpu_step(action_buf.ctypes.data)
            c_obs_snapshot = c_obs.copy()
            c_rewards_snapshot = c_rewards.copy()
            c_dones_snapshot = c_terminals.copy().astype(bool)
            reward_diff = abs(float(ref_rewards[0]) - float(c_rewards_snapshot[0]))
            done_match = bool(ref_dones[0]) == bool(c_dones_snapshot[0])
            obs_diff = first_obs_diff(ref_obs[0], c_obs_snapshot[0], atol)
            if reward_diff > atol or not done_match or obs_diff is not None:
                print(
                    "isolated_replay: divergence "
                    f"step={step} action={int(action)} "
                    f"reward_delta={reward_diff:.8g} "
                    f"done_jax={bool(ref_dones[0])} "
                    f"done_c={bool(c_dones_snapshot[0])}"
                )
                if obs_diff is not None:
                    idx, max_diff, ref_value, c_value = obs_diff
                    section = section_for_index(idx)
                    print(
                        "isolated_replay_obs: "
                        f"index={idx} section={section} "
                        f"subsystem={subsystem_for_section(section)} "
                        f"abs_diff={max_diff:.8g} "
                        f"jax={ref_value:.8g} c={c_value:.8g}"
                    )
                if bool(ref_dones[0]) and bool(c_dones_snapshot[0]):
                    _print_terminal_reset_check(
                        reset_verifier,
                        ref,
                        ref_obs,
                        reset_keys[0],
                        0,
                        seed,
                        step,
                        "isolated_replay",
                        atol,
                    )
                return
        print("isolated_replay: no divergence")
    finally:
        vec.close()


def run(args):
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")

    policy_name = getattr(args, "policy", "uniform")
    if policy_name not in POLICIES:
        raise ValueError(f"--policy must be one of {POLICIES}")

    num_threads = int(getattr(args, "num_threads", 1))
    if num_threads <= 0:
        raise ValueError("--num-threads must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))

    reset_on_done = bool(getattr(args, "reset_on_done", True))
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int64)

    cmod = import_c_env()
    reset_verifier = get_reset_verifier(True)
    ref = JaxCraftaxBatch(seeds, resetter=reset_verifier)
    ref_obs = ref.obs

    vec, c_obs, c_rewards, c_terminals = make_c_vec(
        cmod, len(seeds), int(seeds[0]), num_threads=num_threads
    )
    try:
        if not compare_reset(ref_obs, c_obs.copy(), seeds, args.atol):
            return 1

        if reset_verifier is not None:
            for env_i, seed in enumerate(seeds):
                if not reset_verifier.compare(
                    ref.state_at(env_i),
                    ref_obs[env_i],
                    ref.reset_keys[env_i],
                    int(seed),
                    "initial",
                    policy_name,
                    args.atol,
                ):
                    return 1

        policy = ActionPolicy(policy_name, args.action_seed, len(seeds))
        action_buf = np.zeros((len(seeds), 1), dtype=np.float32)
        histories = [deque(maxlen=10) for _seed in seeds]
        full_histories = [[] for _seed in seeds]
        terminal_counts = np.zeros(len(seeds), dtype=np.int64)
        episode_lengths = np.zeros(len(seeds), dtype=np.int64)
        episode_length_sums = np.zeros(len(seeds), dtype=np.int64)

        for step in range(args.steps):
            step_actions, effective_policy = policy.actions(step, ref)
            action_buf[:, 0] = step_actions.astype(np.float32)
            for env_i, action in enumerate(step_actions):
                histories[env_i].append(int(action))
                full_histories[env_i].append(int(action))

            ref_obs, ref_rewards, ref_dones, reset_keys = ref.step(step_actions)
            vec.cpu_step(action_buf.ctypes.data)

            c_obs_snapshot = c_obs.copy()
            c_rewards_snapshot = c_rewards.copy()
            c_dones_snapshot = c_terminals.copy().astype(bool)

            for env_i, seed in enumerate(seeds):
                reward_diff = abs(float(ref_rewards[env_i]) - float(c_rewards_snapshot[env_i]))
                done_match = bool(ref_dones[env_i]) == bool(c_dones_snapshot[env_i])
                obs_diff = first_obs_diff(ref_obs[env_i], c_obs_snapshot[env_i], args.atol)
                if reward_diff > args.atol or not done_match or obs_diff is not None:
                    _print_step_divergence(
                        seed=seed,
                        step=step,
                        action=step_actions[env_i],
                        policy_name=effective_policy,
                        reward_diff=reward_diff,
                        ref_reward=ref_rewards[env_i],
                        c_reward=c_rewards_snapshot[env_i],
                        ref_done=ref_dones[env_i],
                        c_done=c_dones_snapshot[env_i],
                        obs_diff=obs_diff,
                        history=histories[env_i],
                    )
                    if bool(ref_dones[env_i]) and bool(c_dones_snapshot[env_i]):
                        _print_terminal_reset_check(
                            reset_verifier,
                            ref,
                            ref_obs,
                            reset_keys[env_i],
                            env_i,
                            seed,
                            step,
                            effective_policy,
                            args.atol,
                        )
                    _diagnose_isolated_replay(
                        cmod,
                        int(seed),
                        full_histories[env_i],
                        args.atol,
                        num_threads,
                        reset_verifier,
                    )
                    return 1

            episode_lengths += 1
            done_any = np.logical_or(ref_dones, c_dones_snapshot)
            if reset_on_done and np.any(done_any):
                for env_i, is_done in enumerate(done_any):
                    if not bool(is_done):
                        continue
                    terminal_counts[env_i] += 1
                    episode_length_sums[env_i] += episode_lengths[env_i]
                    if reset_verifier is not None:
                        if not reset_verifier.compare(
                            ref.state_at(env_i),
                            ref_obs[env_i],
                            reset_keys[env_i],
                            int(seeds[env_i]),
                            step,
                            effective_policy,
                            args.atol,
                        ):
                            return 1
                    episode_lengths[env_i] = 0

        total_terminals, per_seed_summary = _terminal_summary(
            seeds, terminal_counts, episode_length_sums
        )
        print(
            f"PASS craftax parity: seeds={args.seeds} steps={args.steps} "
            f"atol={args.atol:g} action_seed={args.action_seed}"
        )
        print(
            f"policy={policy_name} reset_on_done={reset_on_done} "
            f"terminal_count={total_terminals} "
            f"mean_episode_length_by_seed={per_seed_summary}"
        )
        return 0
    finally:
        vec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--policy", choices=POLICIES, default="uniform")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.set_defaults(reset_on_done=True)
    parser.add_argument("--reset-on-done", dest="reset_on_done", action="store_true")
    parser.add_argument("--no-reset-on-done", dest="reset_on_done", action="store_false")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
