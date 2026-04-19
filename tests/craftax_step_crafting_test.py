import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from craftax.craftax.constants import (
    CAN_PLACE_ITEM_BLOCKS,
    SOLID_BLOCKS,
    Action,
    BlockType,
    ItemType,
)
from craftax.craftax.game_logic import (
    add_new_growing_plant,
    do_crafting,
    place_block,
)
from craftax.craftax_env import make_craftax_env_from_name

from tests.craftax_state_fixtures import (
    CraftaxState,
    assert_env_states_equal,
    craftax_state_to_jax,
    jax_state_to_c_state,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(16))
DIRECTION_VECTORS = {
    Action.LEFT.value: jnp.array([0, -1], dtype=jnp.int32),
    Action.RIGHT.value: jnp.array([0, 1], dtype=jnp.int32),
    Action.UP.value: jnp.array([-1, 0], dtype=jnp.int32),
    Action.DOWN.value: jnp.array([1, 0], dtype=jnp.int32),
}
CLOSE_OFFSETS = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
PLACE_ACTIONS = (
    Action.PLACE_STONE.value,
    Action.PLACE_TABLE.value,
    Action.PLACE_FURNACE.value,
    Action.PLACE_PLANT.value,
    Action.PLACE_TORCH.value,
)


@pytest.fixture(scope="session")
def crafting_lib():
    source = r"""
    #include <stdbool.h>
    #include <stdint.h>
    #include <stddef.h>
    #include "ocean/craftax/step_crafting.h"

    size_t craftax_test_state_size(void) {
        return sizeof(CraftaxState);
    }

    void run_do_crafting(CraftaxState* state, int32_t action) {
        craftax_do_crafting_native(state, action);
    }

    void run_place_block(CraftaxState* state, int32_t action) {
        craftax_place_block_native(state, action);
    }

    void run_add_new_growing_plant(
        CraftaxState* state,
        int32_t row,
        int32_t col,
        bool is_placing_sapling
    ) {
        int32_t position[2] = {row, col};
        craftax_add_new_growing_plant_native(
            state,
            position,
            is_placing_sapling
        );
    }
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    src = tmp_path / "craftax_step_crafting_test.c"
    so = tmp_path / "craftax_step_crafting_test.so"
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
            "-ldl",
            "-o",
            str(so),
        ],
        check=True,
        cwd=ROOT,
    )

    lib = ctypes.CDLL(str(so))
    lib._tmpdir = tmp
    state_ptr = ctypes.POINTER(CraftaxState)

    lib.craftax_test_state_size.argtypes = []
    lib.craftax_test_state_size.restype = ctypes.c_size_t
    assert ctypes.sizeof(CraftaxState) == lib.craftax_test_state_size()

    lib.run_do_crafting.argtypes = [state_ptr, ctypes.c_int32]
    lib.run_place_block.argtypes = [state_ptr, ctypes.c_int32]
    lib.run_add_new_growing_plant.argtypes = [
        state_ptr,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_bool,
    ]
    return lib


@pytest.fixture(scope="session")
def jax_context():
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    return env, env.default_params, env.static_env_params


@pytest.fixture(scope="session")
def stepped_states(jax_context):
    env, params, _static_params = jax_context
    action_trace = [
        Action.NOOP.value,
        Action.RIGHT.value,
        Action.DOWN.value,
        Action.LEFT.value,
        Action.UP.value,
        Action.REST.value,
        Action.SLEEP.value,
    ]
    states = {}
    for seed in SEEDS:
        rng = jax.random.PRNGKey(seed)
        rng, reset_key = jax.random.split(rng)
        _obs, state = env.reset(reset_key, params)
        for step in range(3 + seed % 4):
            rng, step_key = jax.random.split(rng)
            action = action_trace[(seed + step) % len(action_trace)]
            _obs, state, _reward, _done, _info = env.step(
                step_key,
                state,
                int(action),
                params,
            )
        states[seed] = state
    return states


def _assert_native_matches(state, expected, run_native, context):
    c_state = jax_state_to_c_state(state)
    run_native(c_state)
    actual = craftax_state_to_jax(c_state, template=state)
    assert_env_states_equal(actual, expected, context)


def _with_inventory(state, **kwargs):
    return state.replace(inventory=state.inventory.replace(**kwargs))


def _empty_inventory_state(state):
    return _with_inventory(
        state,
        wood=0,
        stone=0,
        coal=0,
        iron=0,
        diamond=0,
        sapling=0,
        pickaxe=0,
        sword=0,
        bow=0,
        arrows=0,
        armour=jnp.zeros((4,), dtype=jnp.int32),
        torches=0,
        ruby=0,
        sapphire=0,
        potions=jnp.zeros((6,), dtype=jnp.int32),
        books=0,
    )


def _base_action_state(state):
    return state.replace(
        player_level=0,
        player_position=jnp.array([24, 24], dtype=jnp.int32),
        player_direction=Action.RIGHT.value,
        mob_map=jnp.zeros_like(state.mob_map),
        achievements=jnp.zeros_like(state.achievements),
    )


def _base_crafting_state(state, table=True, furnace=True):
    state = _empty_inventory_state(_base_action_state(state))
    level = int(state.player_level)
    state_map = state.map
    for row_delta, col_delta in CLOSE_OFFSETS:
        row = int(state.player_position[0]) + row_delta
        col = int(state.player_position[1]) + col_delta
        state_map = state_map.at[level, row, col].set(BlockType.GRASS.value)

    if table:
        state_map = state_map.at[level, 24, 23].set(
            BlockType.CRAFTING_TABLE.value
        )
    if furnace:
        state_map = state_map.at[level, 24, 25].set(BlockType.FURNACE.value)

    return state.replace(map=state_map)


def _base_place_state(state):
    return _with_inventory(
        _empty_inventory_state(
            _base_action_state(state).replace(light_map=jnp.zeros_like(state.light_map))
        ),
        wood=8,
        stone=8,
        sapling=8,
        torches=8,
    )


def _target_position(position, direction):
    return np.asarray(
        jnp.asarray(position, dtype=jnp.int32) + DIRECTION_VECTORS[direction],
        dtype=np.int32,
    )


def _set_place_target(
    state,
    block,
    item=ItemType.NONE.value,
    mob=False,
    position=(24, 24),
    direction=Action.RIGHT.value,
):
    state = state.replace(
        player_level=0,
        player_position=jnp.array(position, dtype=jnp.int32),
        player_direction=direction,
    )
    target = _target_position(position, direction)
    if 0 <= target[0] < 48 and 0 <= target[1] < 48:
        level = int(state.player_level)
        state = state.replace(
            map=state.map.at[level, int(target[0]), int(target[1])].set(block),
            item_map=state.item_map.at[level, int(target[0]), int(target[1])].set(item),
            mob_map=state.mob_map.at[level, int(target[0]), int(target[1])].set(mob),
        )
    return state


def _block_name(block):
    return BlockType(block).name.lower()


def _crafting_expected(state, action):
    return do_crafting(state, action)


CRAFT_RECIPES = (
    (
        "wood_pickaxe",
        Action.MAKE_WOOD_PICKAXE.value,
        {"wood": 1, "pickaxe": 0},
        {"wood": 0},
        {"pickaxe": 1},
        False,
    ),
    (
        "stone_pickaxe",
        Action.MAKE_STONE_PICKAXE.value,
        {"wood": 1, "stone": 1, "pickaxe": 0},
        {"stone": 0},
        {"pickaxe": 2},
        False,
    ),
    (
        "iron_pickaxe",
        Action.MAKE_IRON_PICKAXE.value,
        {"wood": 1, "stone": 1, "iron": 1, "coal": 1, "pickaxe": 0},
        {"coal": 0},
        {"pickaxe": 3},
        True,
    ),
    (
        "diamond_pickaxe",
        Action.MAKE_DIAMOND_PICKAXE.value,
        {"wood": 1, "diamond": 3, "pickaxe": 0},
        {"diamond": 2},
        {"pickaxe": 4},
        False,
    ),
    (
        "wood_sword",
        Action.MAKE_WOOD_SWORD.value,
        {"wood": 1, "sword": 0},
        {"wood": 0},
        {"sword": 1},
        False,
    ),
    (
        "stone_sword",
        Action.MAKE_STONE_SWORD.value,
        {"wood": 1, "stone": 1, "sword": 0},
        {"stone": 0},
        {"sword": 2},
        False,
    ),
    (
        "iron_sword",
        Action.MAKE_IRON_SWORD.value,
        {"wood": 1, "stone": 1, "iron": 1, "coal": 1, "sword": 0},
        {"iron": 0},
        {"sword": 3},
        True,
    ),
    (
        "diamond_sword",
        Action.MAKE_DIAMOND_SWORD.value,
        {"wood": 1, "diamond": 2, "sword": 0},
        {"diamond": 1},
        {"sword": 4},
        False,
    ),
    (
        "iron_armour",
        Action.MAKE_IRON_ARMOUR.value,
        {
            "iron": 3,
            "coal": 3,
            "armour": jnp.array([1, 0, 1, 1], dtype=jnp.int32),
        },
        {"iron": 2},
        {"armour": jnp.ones((4,), dtype=jnp.int32)},
        True,
    ),
    (
        "diamond_armour",
        Action.MAKE_DIAMOND_ARMOUR.value,
        {
            "diamond": 3,
            "armour": jnp.array([2, 2, 1, 2], dtype=jnp.int32),
        },
        {"diamond": 2},
        {"armour": jnp.full((4,), 2, dtype=jnp.int32)},
        False,
    ),
    (
        "arrow",
        Action.MAKE_ARROW.value,
        {"wood": 1, "stone": 1, "arrows": 98},
        {"wood": 0},
        {"arrows": 99},
        False,
    ),
    (
        "torch",
        Action.MAKE_TORCH.value,
        {"wood": 1, "coal": 1, "torches": 98},
        {"coal": 0},
        {"torches": 99},
        False,
    ),
)


def test_do_crafting_native_parity(crafting_lib, stepped_states):
    for seed, base_state in stepped_states.items():
        for (
            name,
            action,
            success_inventory,
            missing_inventory,
            blocked_inventory,
            needs_furnace,
        ) in CRAFT_RECIPES:
            success = _with_inventory(
                _base_crafting_state(base_state, table=True, furnace=needs_furnace),
                **success_inventory,
            )
            cases = [
                ("success", success),
                (
                    "missing_resource",
                    _with_inventory(success, **missing_inventory),
                ),
                (
                    "blocked_existing_tool_or_full_stack",
                    _with_inventory(success, **blocked_inventory),
                ),
                (
                    "not_near_table",
                    _with_inventory(
                        _base_crafting_state(
                            base_state,
                            table=False,
                            furnace=needs_furnace,
                        ),
                        **success_inventory,
                    ),
                ),
            ]
            if needs_furnace:
                cases.append(
                    (
                        "not_near_furnace",
                        _with_inventory(
                            _base_crafting_state(
                                base_state,
                                table=True,
                                furnace=False,
                            ),
                            **success_inventory,
                        ),
                    )
                )

            for case_name, state in cases:
                expected = _crafting_expected(state, action)
                _assert_native_matches(
                    state,
                    expected,
                    lambda c_state, action=action: crafting_lib.run_do_crafting(
                        ctypes.byref(c_state),
                        int(action),
                    ),
                    f"do_crafting seed={seed} recipe={name} case={case_name}",
                )


def _legal_place_blocks(action):
    non_solid_blocks = tuple(
        block.value for block in BlockType if block.value not in set(SOLID_BLOCKS)
    )
    if action in {
        Action.PLACE_STONE.value,
        Action.PLACE_TABLE.value,
        Action.PLACE_FURNACE.value,
    }:
        return non_solid_blocks
    if action == Action.PLACE_PLANT.value:
        return (BlockType.GRASS.value,)
    if action == Action.PLACE_TORCH.value:
        return tuple(CAN_PLACE_ITEM_BLOCKS)
    raise ValueError(action)


def _place_missing_inventory(state, action):
    if action == Action.PLACE_TABLE.value:
        return _with_inventory(state, wood=1)
    if action == Action.PLACE_FURNACE.value:
        return _with_inventory(state, stone=0)
    if action == Action.PLACE_STONE.value:
        return _with_inventory(state, stone=0)
    if action == Action.PLACE_PLANT.value:
        return _with_inventory(state, sapling=0)
    if action == Action.PLACE_TORCH.value:
        return _with_inventory(state, torches=0)
    raise ValueError(action)


def test_place_block_native_parity(crafting_lib, jax_context, stepped_states):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        base_state = _base_place_state(base_state)
        cases = []

        for action in PLACE_ACTIONS:
            for block in _legal_place_blocks(action):
                cases.append(
                    (
                        f"action_{action}_legal_{_block_name(block)}",
                        _set_place_target(base_state, block),
                        action,
                    )
                )

            illegal_cases = [
                ("wall", BlockType.WALL.value, ItemType.NONE.value, False),
                ("existing_item", BlockType.GRASS.value, ItemType.TORCH.value, False),
                ("target_mob", BlockType.GRASS.value, ItemType.NONE.value, True),
            ]
            if BlockType.WATER.value not in _legal_place_blocks(action):
                illegal_cases.append(
                    ("water", BlockType.WATER.value, ItemType.NONE.value, False)
                )

            for illegal_name, block, item, mob in illegal_cases:
                cases.append(
                    (
                        f"action_{action}_illegal_{illegal_name}",
                        _set_place_target(base_state, block, item=item, mob=mob),
                        action,
                    )
                )

            cases.append(
                (
                    f"action_{action}_missing_inventory",
                    _place_missing_inventory(
                        _set_place_target(base_state, BlockType.GRASS.value),
                        action,
                    ),
                    action,
                )
            )

            boundary_cases = [
                ("upper_left_left", (0, 0), Action.LEFT.value),
                ("upper_left_up", (0, 0), Action.UP.value),
                ("lower_right_right", (47, 47), Action.RIGHT.value),
                ("lower_right_down", (47, 47), Action.DOWN.value),
            ]
            for boundary_name, position, direction in boundary_cases:
                cases.append(
                    (
                        f"action_{action}_boundary_{boundary_name}",
                        _set_place_target(
                            base_state,
                            BlockType.GRASS.value,
                            position=position,
                            direction=direction,
                        ),
                        action,
                    )
                )

        for name, state, action in cases:
            expected = place_block(state, action, static_params)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, action=action: crafting_lib.run_place_block(
                    ctypes.byref(c_state),
                    int(action),
                ),
                f"place_block seed={seed} case={name}",
            )


def _with_growing_plants(state, mask):
    positions = jnp.arange(20, dtype=jnp.int32).reshape((10, 2))
    ages = jnp.arange(10, dtype=jnp.int32) * 11 + 3
    return state.replace(
        growing_plants_positions=positions,
        growing_plants_age=ages,
        growing_plants_mask=jnp.array(mask, dtype=bool),
    )


def _expected_add_new_growing_plant(state, position, is_placing_sapling, static_params):
    positions, ages, mask = add_new_growing_plant(
        state,
        jnp.array(position, dtype=jnp.int32),
        is_placing_sapling,
        static_params,
    )
    return state.replace(
        growing_plants_positions=positions,
        growing_plants_age=ages,
        growing_plants_mask=mask,
    )


def test_add_new_growing_plant_native_parity(
    crafting_lib,
    jax_context,
    stepped_states,
):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        base_state = _base_action_state(base_state)
        cases = [
            (
                "first_empty_middle",
                _with_growing_plants(
                    base_state,
                    [True, True, False, True, False, True, True, True, True, True],
                ),
                (31, 32),
                True,
            ),
            (
                "first_empty_zero",
                _with_growing_plants(
                    base_state,
                    [False, True, True, True, True, True, True, True, True, True],
                ),
                (7, 8),
                True,
            ),
            (
                "no_empty_slot",
                _with_growing_plants(base_state, [True] * 10),
                (9, 10),
                True,
            ),
            (
                "not_placing",
                _with_growing_plants(
                    base_state,
                    [True, False, True, True, True, True, True, True, True, True],
                ),
                (11, 12),
                False,
            ),
        ]

        for name, state, position, is_placing_sapling in cases:
            expected = _expected_add_new_growing_plant(
                state,
                position,
                is_placing_sapling,
                static_params,
            )
            _assert_native_matches(
                state,
                expected,
                lambda c_state, position=position, is_placing_sapling=is_placing_sapling: (
                    crafting_lib.run_add_new_growing_plant(
                        ctypes.byref(c_state),
                        int(position[0]),
                        int(position[1]),
                        bool(is_placing_sapling),
                    )
                ),
                f"add_new_growing_plant seed={seed} case={name}",
            )
