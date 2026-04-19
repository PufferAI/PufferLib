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

from craftax.craftax.constants import Action, Achievement, BlockType
from craftax.craftax.game_logic import (
    boss_logic,
    calculate_inventory_achievements,
    drink_potion,
    level_up_attributes,
    move_player,
    read_book,
    update_plants,
    update_player_intrinsics,
)
from craftax.craftax.util.game_logic_utils import clip_inventory_and_intrinsics
from craftax.craftax_env import make_craftax_env_from_name

from tests.craftax_state_fixtures import (
    CraftaxState,
    assert_env_states_equal,
    craftax_state_to_jax,
    jax_state_to_c_state,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(16))


@pytest.fixture(scope="session")
def step_lib():
    source = r"""
    #include <stdbool.h>
    #include <stdint.h>
    #include <stddef.h>
    #include "ocean/craftax/step_simple.h"

    size_t craftax_test_state_size(void) {
        return sizeof(CraftaxState);
    }

    void run_move_player(CraftaxState* state, int32_t action, bool god_mode) {
        craftax_move_player_native(state, action, god_mode);
    }

    void run_update_plants(CraftaxState* state) {
        craftax_update_plants_native(state);
    }

    void run_boss_logic(CraftaxState* state) {
        craftax_boss_logic_native(state);
    }

    void run_level_up_attributes(
        CraftaxState* state,
        int32_t action,
        int32_t max_attribute
    ) {
        craftax_level_up_attributes_native(state, action, max_attribute);
    }

    void run_clip_inventory_and_intrinsics(CraftaxState* state, bool god_mode) {
        craftax_clip_inventory_and_intrinsics_native(state, god_mode);
    }

    void run_calculate_inventory_achievements(CraftaxState* state) {
        craftax_calculate_inventory_achievements_native(state);
    }

    void run_update_player_intrinsics(CraftaxState* state, int32_t action) {
        craftax_update_player_intrinsics_native(state, action);
    }

    void run_drink_potion(CraftaxState* state, int32_t action) {
        craftax_drink_potion_native(state, action);
    }

    void run_read_book(
        CraftaxState* state,
        uint32_t rng0,
        uint32_t rng1,
        int32_t action
    ) {
        uint32_t rng[2] = {rng0, rng1};
        craftax_read_book_native(state, rng, action);
    }
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    src = tmp_path / "craftax_step_simple_test.c"
    so = tmp_path / "craftax_step_simple_test.so"
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

    lib.run_move_player.argtypes = [state_ptr, ctypes.c_int32, ctypes.c_bool]
    lib.run_update_plants.argtypes = [state_ptr]
    lib.run_boss_logic.argtypes = [state_ptr]
    lib.run_level_up_attributes.argtypes = [
        state_ptr,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.run_clip_inventory_and_intrinsics.argtypes = [state_ptr, ctypes.c_bool]
    lib.run_calculate_inventory_achievements.argtypes = [state_ptr]
    lib.run_update_player_intrinsics.argtypes = [state_ptr, ctypes.c_int32]
    lib.run_drink_potion.argtypes = [state_ptr, ctypes.c_int32]
    lib.run_read_book.argtypes = [
        state_ptr,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int32,
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
                step_key, state, int(action), params
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


def _clear_local_mobs(state):
    return state.replace(mob_map=jnp.zeros_like(state.mob_map))


def _set_neighbour_block(state, action, block):
    directions = {
        Action.LEFT.value: jnp.array([0, -1], dtype=jnp.int32),
        Action.RIGHT.value: jnp.array([0, 1], dtype=jnp.int32),
        Action.UP.value: jnp.array([-1, 0], dtype=jnp.int32),
        Action.DOWN.value: jnp.array([1, 0], dtype=jnp.int32),
    }
    level = int(state.player_level)
    position = np.asarray(state.player_position + directions[action], dtype=np.int32)
    return state.replace(
        map=state.map.at[level, int(position[0]), int(position[1])].set(block),
        mob_map=state.mob_map.at[level, int(position[0]), int(position[1])].set(False),
    )


def _set_neighbour_mob(state, action):
    directions = {
        Action.LEFT.value: jnp.array([0, -1], dtype=jnp.int32),
        Action.RIGHT.value: jnp.array([0, 1], dtype=jnp.int32),
        Action.UP.value: jnp.array([-1, 0], dtype=jnp.int32),
        Action.DOWN.value: jnp.array([1, 0], dtype=jnp.int32),
    }
    level = int(state.player_level)
    position = np.asarray(state.player_position + directions[action], dtype=np.int32)
    return state.replace(
        map=state.map.at[level, int(position[0]), int(position[1])].set(
            BlockType.GRASS.value
        ),
        mob_map=state.mob_map.at[level, int(position[0]), int(position[1])].set(True),
    )


def test_state_fixture_roundtrip(stepped_states):
    for seed, state in stepped_states.items():
        c_state = jax_state_to_c_state(state)
        roundtrip = craftax_state_to_jax(c_state, template=state)
        assert_env_states_equal(roundtrip, state, f"fixture roundtrip seed={seed}")


def test_move_player_native_parity(step_lib, jax_context, stepped_states):
    _env, params, _static_params = jax_context
    for seed, base_state in stepped_states.items():
        base_state = _clear_local_mobs(base_state)
        cases = [
            ("noop", base_state, Action.NOOP.value, params),
            ("left", base_state, Action.LEFT.value, params),
            ("right", base_state, Action.RIGHT.value, params),
            ("up", base_state, Action.UP.value, params),
            ("down", base_state, Action.DOWN.value, params),
            ("zero_direction_high_action", base_state, Action.READ_BOOK.value, params),
            ("negative_index_direction", base_state, -12, params),
            (
                "solid_block",
                _set_neighbour_block(base_state, Action.LEFT.value, BlockType.STONE.value),
                Action.LEFT.value,
                params,
            ),
            (
                "water_block",
                _set_neighbour_block(base_state, Action.RIGHT.value, BlockType.WATER.value),
                Action.RIGHT.value,
                params,
            ),
            (
                "lava_block",
                _set_neighbour_block(base_state, Action.DOWN.value, BlockType.LAVA.value),
                Action.DOWN.value,
                params,
            ),
            ("mob_block", _set_neighbour_mob(base_state, Action.UP.value), Action.UP.value, params),
            (
                "god_oob",
                base_state.replace(
                    player_position=jnp.array([0, 0], dtype=jnp.int32),
                    mob_map=jnp.zeros_like(base_state.mob_map),
                ),
                Action.LEFT.value,
                params.replace(god_mode=True),
            ),
        ]
        for name, state, action, case_params in cases:
            expected = move_player(state, action, case_params)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, action=action, case_params=case_params: (
                    step_lib.run_move_player(
                        ctypes.byref(c_state),
                        int(action),
                        bool(case_params.god_mode),
                    )
                ),
                f"move_player seed={seed} case={name}",
            )


def test_update_plants_native_parity(step_lib, jax_context, stepped_states):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        positions = jnp.array(
            [[5, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17],
             [18, 19], [20, 21], [22, 23]],
            dtype=jnp.int32,
        )
        empty = base_state.replace(
            growing_plants_positions=positions,
            growing_plants_age=jnp.zeros((10,), dtype=jnp.int32),
            growing_plants_mask=jnp.zeros((10,), dtype=bool),
        )
        mixed = empty.replace(
            map=empty.map.at[0, 5, 5].set(BlockType.PLANT.value)
            .at[0, 6, 7].set(BlockType.PLANT.value)
            .at[0, 8, 9].set(BlockType.GRASS.value),
            growing_plants_age=jnp.array(
                [0, 598, 599, 600, 12, 0, 0, 0, 0, 0], dtype=jnp.int32
            ),
            growing_plants_mask=jnp.array(
                [True, True, True, False, True, False, False, False, False, False],
                dtype=bool,
            ),
        )
        cases = [("empty", empty), ("mixed_growth", mixed)]
        for name, state in cases:
            expected = update_plants(state, static_params)
            _assert_native_matches(
                state,
                expected,
                lambda c_state: step_lib.run_update_plants(ctypes.byref(c_state)),
                f"update_plants seed={seed} case={name}",
            )


def test_boss_logic_native_parity(step_lib, jax_context, stepped_states):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        cases = [
            ("nonboss", base_state.replace(player_level=0, boss_progress=0)),
            ("boss_waiting", base_state.replace(player_level=8, boss_progress=0)),
            ("boss_beaten", base_state.replace(player_level=8, boss_progress=8)),
            (
                "already_achieved",
                base_state.replace(
                    player_level=0,
                    boss_progress=0,
                    achievements=base_state.achievements.at[
                        Achievement.DEFEAT_NECROMANCER.value
                    ].set(True),
                ),
            ),
        ]
        for name, state in cases:
            expected = boss_logic(state, static_params)
            _assert_native_matches(
                state,
                expected,
                lambda c_state: step_lib.run_boss_logic(ctypes.byref(c_state)),
                f"boss_logic seed={seed} case={name}",
            )


def test_level_up_attributes_native_parity(step_lib, jax_context, stepped_states):
    _env, params, _static_params = jax_context
    for seed, base_state in stepped_states.items():
        cases = [
            (
                "dex",
                base_state.replace(
                    player_xp=2,
                    player_dexterity=1,
                    player_strength=1,
                    player_intelligence=1,
                ),
                Action.LEVEL_UP_DEXTERITY.value,
            ),
            ("str", base_state.replace(player_xp=1, player_strength=2), Action.LEVEL_UP_STRENGTH.value),
            (
                "int",
                base_state.replace(player_xp=1, player_intelligence=3),
                Action.LEVEL_UP_INTELLIGENCE.value,
            ),
            (
                "at_cap",
                base_state.replace(player_xp=1, player_dexterity=params.max_attribute),
                Action.LEVEL_UP_DEXTERITY.value,
            ),
            ("no_xp", base_state.replace(player_xp=0), Action.LEVEL_UP_STRENGTH.value),
            ("noop", base_state.replace(player_xp=1), Action.NOOP.value),
        ]
        for name, state, action in cases:
            expected = level_up_attributes(state, action, params)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, action=action: step_lib.run_level_up_attributes(
                    ctypes.byref(c_state), int(action), int(params.max_attribute)
                ),
                f"level_up_attributes seed={seed} case={name}",
            )


def test_clip_inventory_and_intrinsics_native_parity(step_lib, jax_context, stepped_states):
    _env, params, _static_params = jax_context
    for seed, base_state in stepped_states.items():
        overfull = _with_inventory(
            base_state.replace(
                player_health=-5.0,
                player_food=-2,
                player_drink=500,
                player_energy=500,
                player_mana=500,
                player_dexterity=2,
                player_strength=3,
                player_intelligence=4,
            ),
            wood=120,
            stone=101,
            coal=100,
            iron=99,
            diamond=-4,
            sapling=150,
            pickaxe=104,
            sword=105,
            bow=106,
            arrows=107,
            armour=jnp.array([120, 99, -3, 140], dtype=jnp.int32),
            torches=108,
            ruby=109,
            sapphire=110,
            potions=jnp.array([111, 98, -2, 130, 99, 100], dtype=jnp.int32),
            books=112,
        )
        low_max = base_state.replace(
            player_health=50.0,
            player_food=50,
            player_drink=50,
            player_energy=50,
            player_mana=50,
            player_dexterity=0,
            player_strength=0,
            player_intelligence=0,
        )
        cases = [
            ("overfull", overfull, params),
            ("god_mode", overfull, params.replace(god_mode=True)),
            ("low_attribute_max", low_max, params.replace(god_mode=True)),
        ]
        for name, state, case_params in cases:
            expected = clip_inventory_and_intrinsics(state, case_params)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, case_params=case_params: (
                    step_lib.run_clip_inventory_and_intrinsics(
                        ctypes.byref(c_state), bool(case_params.god_mode)
                    )
                ),
                f"clip_inventory_and_intrinsics seed={seed} case={name}",
            )


def test_calculate_inventory_achievements_native_parity(step_lib, stepped_states):
    for seed, base_state in stepped_states.items():
        empty = _with_inventory(
            base_state.replace(achievements=jnp.zeros_like(base_state.achievements)),
            wood=0,
            stone=0,
            coal=0,
            iron=0,
            diamond=0,
            sapling=0,
            bow=0,
            arrows=0,
            torches=0,
            ruby=0,
            sapphire=0,
            pickaxe=0,
            sword=0,
        )
        full = _with_inventory(
            empty,
            wood=1,
            stone=1,
            coal=1,
            iron=1,
            diamond=1,
            sapling=1,
            bow=1,
            arrows=1,
            torches=1,
            ruby=1,
            sapphire=1,
            pickaxe=4,
            sword=4,
        )
        partial = _with_inventory(empty, pickaxe=2, sword=3, arrows=4)
        preexisting = empty.replace(
            achievements=empty.achievements.at[
                Achievement.MAKE_DIAMOND_PICKAXE.value
            ].set(True)
        )
        cases = [("empty", empty), ("full", full), ("partial", partial), ("preexisting", preexisting)]
        for name, state in cases:
            expected = calculate_inventory_achievements(state)
            _assert_native_matches(
                state,
                expected,
                lambda c_state: step_lib.run_calculate_inventory_achievements(
                    ctypes.byref(c_state)
                ),
                f"calculate_inventory_achievements seed={seed} case={name}",
            )


def test_update_player_intrinsics_native_parity(step_lib, jax_context, stepped_states):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        base_state = base_state.replace(
            player_level=0,
            player_dexterity=1,
            player_strength=1,
            player_intelligence=1,
            is_sleeping=False,
            is_resting=False,
        )
        cases = [
            (
                "sleep_start",
                base_state.replace(player_energy=3, player_hunger=0.0, player_thirst=0.0),
                Action.SLEEP.value,
            ),
            (
                "sleep_wake",
                base_state.replace(
                    is_sleeping=True,
                    player_energy=9,
                    achievements=base_state.achievements.at[
                        Achievement.WAKE_UP.value
                    ].set(False),
                ),
                Action.NOOP.value,
            ),
            (
                "rest_start",
                base_state.replace(player_health=4.0, player_food=5, player_drink=5),
                Action.REST.value,
            ),
            (
                "rest_wake_no_food",
                base_state.replace(is_resting=True, player_health=4.0, player_food=0),
                Action.NOOP.value,
            ),
            (
                "positive_thresholds",
                base_state.replace(
                    player_hunger=25.0,
                    player_thirst=20.0,
                    player_fatigue=30.0,
                    player_recover=25.0,
                    player_recover_mana=30.0,
                    player_food=5,
                    player_drink=5,
                    player_energy=5,
                    player_mana=5,
                    player_health=5.0,
                ),
                Action.NOOP.value,
            ),
            (
                "health_decay",
                base_state.replace(
                    player_recover=-15.0,
                    player_food=0,
                    player_drink=5,
                    player_energy=5,
                    player_health=5.0,
                ),
                Action.NOOP.value,
            ),
            (
                "sleep_energy_recovery",
                base_state.replace(
                    is_sleeping=True,
                    player_energy=3,
                    player_fatigue=-10.5,
                ),
                Action.NOOP.value,
            ),
            (
                "boss_floor_decay_gated",
                base_state.replace(
                    player_level=8,
                    player_hunger=25.0,
                    player_thirst=20.0,
                    player_fatigue=30.0,
                    player_food=5,
                    player_drink=5,
                    player_energy=5,
                ),
                Action.NOOP.value,
            ),
        ]
        for name, state, action in cases:
            expected = update_player_intrinsics(state, action, static_params)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, action=action: step_lib.run_update_player_intrinsics(
                    ctypes.byref(c_state), int(action)
                ),
                f"update_player_intrinsics seed={seed} case={name}",
            )


def test_drink_potion_native_parity(step_lib, stepped_states):
    actions = [
        Action.DRINK_POTION_RED.value,
        Action.DRINK_POTION_GREEN.value,
        Action.DRINK_POTION_BLUE.value,
        Action.DRINK_POTION_PINK.value,
        Action.DRINK_POTION_CYAN.value,
        Action.DRINK_POTION_YELLOW.value,
    ]
    for seed, base_state in stepped_states.items():
        potion_state = _with_inventory(
            base_state.replace(
                potion_mapping=jnp.arange(6, dtype=jnp.int32),
                player_health=5.0,
                player_mana=5,
                player_energy=5,
                achievements=base_state.achievements.at[
                    Achievement.DRINK_POTION.value
                ].set(False),
            ),
            potions=jnp.array([2, 2, 2, 2, 2, 2], dtype=jnp.int32),
        )
        cases = [(f"effect_{idx}", potion_state, action) for idx, action in enumerate(actions)]
        cases.extend(
            [
                (
                    "empty_red",
                    _with_inventory(potion_state, potions=jnp.zeros((6,), dtype=jnp.int32)),
                    Action.DRINK_POTION_RED.value,
                ),
                ("noop", potion_state, Action.NOOP.value),
            ]
        )
        for name, state, action in cases:
            expected = drink_potion(state, action)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, action=action: step_lib.run_drink_potion(
                    ctypes.byref(c_state), int(action)
                ),
                f"drink_potion seed={seed} case={name}",
            )


def test_read_book_native_parity(step_lib, stepped_states):
    for seed, base_state in stepped_states.items():
        clean_achievements = (
            base_state.achievements.at[Achievement.LEARN_FIREBALL.value].set(False)
            .at[Achievement.LEARN_ICEBALL.value].set(False)
        )
        cases = [
            (
                "none_learned",
                _with_inventory(
                    base_state.replace(
                        learned_spells=jnp.array([False, False], dtype=bool),
                        achievements=clean_achievements,
                    ),
                    books=1,
                ),
                Action.READ_BOOK.value,
            ),
            (
                "fire_known",
                _with_inventory(
                    base_state.replace(
                        learned_spells=jnp.array([True, False], dtype=bool),
                        achievements=clean_achievements,
                    ),
                    books=2,
                ),
                Action.READ_BOOK.value,
            ),
            (
                "ice_known",
                _with_inventory(
                    base_state.replace(
                        learned_spells=jnp.array([False, True], dtype=bool),
                        achievements=clean_achievements,
                    ),
                    books=2,
                ),
                Action.READ_BOOK.value,
            ),
            (
                "both_known",
                _with_inventory(
                    base_state.replace(
                        learned_spells=jnp.array([True, True], dtype=bool),
                        achievements=clean_achievements,
                    ),
                    books=1,
                ),
                Action.READ_BOOK.value,
            ),
            (
                "no_books",
                _with_inventory(
                    base_state.replace(
                        learned_spells=jnp.array([False, False], dtype=bool),
                        achievements=clean_achievements,
                    ),
                    books=0,
                ),
                Action.READ_BOOK.value,
            ),
            (
                "noop",
                _with_inventory(
                    base_state.replace(
                        learned_spells=jnp.array([False, False], dtype=bool),
                        achievements=clean_achievements,
                    ),
                    books=1,
                ),
                Action.NOOP.value,
            ),
        ]
        for case_index, (name, state, action) in enumerate(cases):
            rng = jax.random.PRNGKey(seed * 101 + case_index)
            rng_words = np.asarray(rng, dtype=np.uint32)
            expected = read_book(rng, state, action)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, action=action, rng_words=rng_words: (
                    step_lib.run_read_book(
                        ctypes.byref(c_state),
                        int(rng_words[0]),
                        int(rng_words[1]),
                        int(action),
                    )
                ),
                f"read_book seed={seed} case={name}",
            )
