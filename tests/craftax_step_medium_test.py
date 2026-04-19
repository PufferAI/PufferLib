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

from craftax.craftax.constants import Action, Achievement, BlockType, ItemType
from craftax.craftax.game_logic import (
    add_items_from_chest,
    cast_spell,
    change_floor,
    enchant,
    shoot_projectile,
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
DIRECTION_ACTIONS = (
    Action.LEFT.value,
    Action.RIGHT.value,
    Action.UP.value,
    Action.DOWN.value,
)
LEVEL_ACHIEVEMENTS = (
    0,
    Achievement.ENTER_DUNGEON.value,
    Achievement.ENTER_GNOMISH_MINES.value,
    Achievement.ENTER_SEWERS.value,
    Achievement.ENTER_VAULT.value,
    Achievement.ENTER_TROLL_MINES.value,
    Achievement.ENTER_FIRE_REALM.value,
    Achievement.ENTER_ICE_REALM.value,
    Achievement.ENTER_GRAVEYARD.value,
)


@pytest.fixture(scope="session")
def medium_lib():
    source = r"""
    #include <stdbool.h>
    #include <stdint.h>
    #include <stddef.h>
    #include "ocean/craftax/step_medium.h"

    size_t craftax_test_state_size(void) {
        return sizeof(CraftaxState);
    }

    void run_shoot_projectile(CraftaxState* state, int32_t action) {
        craftax_shoot_projectile_native(state, action);
    }

    void run_cast_spell(CraftaxState* state, int32_t action) {
        craftax_cast_spell_native(state, action);
    }

    void run_enchant(
        CraftaxState* state,
        int32_t action,
        uint32_t rng0,
        uint32_t rng1
    ) {
        CraftaxThreefryKey rng = {{rng0, rng1}};
        craftax_enchant_native(state, action, rng);
    }

    void run_change_floor(CraftaxState* state, int32_t action) {
        craftax_change_floor_native(state, action);
    }

    void run_add_items_from_chest(
        CraftaxState* state,
        bool is_opening_chest,
        uint32_t rng0,
        uint32_t rng1
    ) {
        CraftaxThreefryKey rng = {{rng0, rng1}};
        craftax_add_items_from_chest_native(
            state,
            &state->inventory,
            is_opening_chest,
            rng
        );
    }
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    src = tmp_path / "craftax_step_medium_test.c"
    so = tmp_path / "craftax_step_medium_test.so"
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

    lib.run_shoot_projectile.argtypes = [state_ptr, ctypes.c_int32]
    lib.run_cast_spell.argtypes = [state_ptr, ctypes.c_int32]
    lib.run_enchant.argtypes = [
        state_ptr,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.run_change_floor.argtypes = [state_ptr, ctypes.c_int32]
    lib.run_add_items_from_chest.argtypes = [
        state_ptr,
        ctypes.c_bool,
        ctypes.c_uint32,
        ctypes.c_uint32,
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


def _rng_words(seed):
    return np.asarray(jax.random.PRNGKey(seed), dtype=np.uint32)


def _with_inventory(state, **kwargs):
    return state.replace(inventory=state.inventory.replace(**kwargs))


def _base_action_state(state):
    return state.replace(
        player_level=0,
        player_position=jnp.array([24, 24], dtype=jnp.int32),
        player_direction=Action.RIGHT.value,
    )


def _set_player_projectile_masks(state, masks):
    level = int(state.player_level)
    return state.replace(
        player_projectiles=state.player_projectiles.replace(
            mask=state.player_projectiles.mask.at[level].set(
                jnp.array(masks, dtype=bool)
            )
        )
    )


def _empty_projectiles(state):
    return _set_player_projectile_masks(state, [False, False, False])


def _full_projectiles(state):
    return _set_player_projectile_masks(state, [True, True, True])


def _set_target_block(state, block):
    directions = {
        Action.LEFT.value: jnp.array([0, -1], dtype=jnp.int32),
        Action.RIGHT.value: jnp.array([0, 1], dtype=jnp.int32),
        Action.UP.value: jnp.array([-1, 0], dtype=jnp.int32),
        Action.DOWN.value: jnp.array([1, 0], dtype=jnp.int32),
    }
    level = int(state.player_level)
    target = np.asarray(
        state.player_position + directions[int(state.player_direction)],
        dtype=np.int32,
    )
    return state.replace(
        map=state.map.at[level, int(target[0]), int(target[1])].set(block)
    )


def _with_clean_enchant_achievements(state):
    return state.replace(
        achievements=state.achievements.at[Achievement.ENCHANT_SWORD.value]
        .set(False)
        .at[Achievement.ENCHANT_ARMOUR.value]
        .set(False)
    )


def _base_enchant_state(state, block, enchantment_type):
    ruby = 1 if enchantment_type == 1 else 0
    sapphire = 1 if enchantment_type == 2 else 0
    state = _base_action_state(_with_clean_enchant_achievements(state)).replace(
        player_mana=9,
        sword_enchantment=0,
        bow_enchantment=0,
        armour_enchantments=jnp.zeros((4,), dtype=jnp.int32),
    )
    state = _with_inventory(
        state,
        ruby=ruby,
        sapphire=sapphire,
        sword=1,
        bow=1,
        armour=jnp.ones((4,), dtype=jnp.int32),
    )
    return _set_target_block(state, block)


def _set_floor_achievement(state, level, value):
    achievement = LEVEL_ACHIEVEMENTS[level]
    if achievement == 0:
        return state
    return state.replace(achievements=state.achievements.at[achievement].set(value))


def _floor_state(state, level, item, position, monsters_killed=8):
    state = state.replace(
        player_level=level,
        player_position=jnp.array(position, dtype=jnp.int32),
        monsters_killed=state.monsters_killed.at[level].set(monsters_killed),
    )
    return state.replace(
        item_map=state.item_map.at[level, int(position[0]), int(position[1])].set(item)
    )


def _chest_expected_state(rng, state, is_opening_chest):
    return state.replace(
        inventory=add_items_from_chest(
            rng,
            state,
            state.inventory,
            is_opening_chest,
        )
    )


def _find_chest_rng(state, predicate, start_seed=0, limit=10000):
    for seed in range(start_seed, start_seed + limit):
        rng = jax.random.PRNGKey(seed)
        inventory = add_items_from_chest(rng, state, state.inventory, True)
        if predicate(inventory):
            return np.asarray(rng, dtype=np.uint32)
    raise AssertionError("could not find targeted chest rng")


@pytest.fixture(scope="session")
def chest_target_keys(stepped_states):
    base = _with_inventory(
        _base_action_state(stepped_states[0]).replace(
            chests_opened=jnp.ones((9,), dtype=bool),
            player_level=0,
        ),
        coal=0,
        iron=0,
        diamond=0,
        sapphire=0,
        ruby=0,
        torches=0,
        arrows=0,
        pickaxe=0,
        sword=0,
        bow=0,
        potions=jnp.zeros((6,), dtype=jnp.int32),
        books=0,
    )
    keys = {
        f"potion_{idx}": _find_chest_rng(
            base,
            lambda inventory, idx=idx: int(np.asarray(inventory.potions)[idx]) > 0,
            start_seed=idx * 1000,
        )
        for idx in range(6)
    }
    keys["sapphire"] = _find_chest_rng(
        base,
        lambda inventory: int(inventory.sapphire) > 0,
        start_seed=7000,
    )
    keys["ruby"] = _find_chest_rng(
        base,
        lambda inventory: int(inventory.ruby) > 0,
        start_seed=8000,
    )
    return keys


def test_shoot_projectile_native_parity(medium_lib, jax_context, stepped_states):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        base_state = _empty_projectiles(_base_action_state(base_state))
        shoot_ready = _with_inventory(base_state, bow=1, arrows=3)
        cases = [
            (
                f"direction_{direction}",
                shoot_ready.replace(player_direction=direction),
                Action.SHOOT_ARROW.value,
            )
            for direction in DIRECTION_ACTIONS
        ]
        cases.extend(
            [
                (
                    "no_bow",
                    _with_inventory(base_state, bow=0, arrows=3),
                    Action.SHOOT_ARROW.value,
                ),
                (
                    "no_arrows",
                    _with_inventory(base_state, bow=1, arrows=0),
                    Action.SHOOT_ARROW.value,
                ),
                (
                    "mana_irrelevant",
                    _with_inventory(shoot_ready.replace(player_mana=0), bow=1, arrows=3),
                    Action.SHOOT_ARROW.value,
                ),
                (
                    "full_projectiles",
                    _full_projectiles(shoot_ready),
                    Action.SHOOT_ARROW.value,
                ),
                ("noop", shoot_ready, Action.NOOP.value),
            ]
        )
        for name, state, action in cases:
            expected = shoot_projectile(state, action, static_params)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, action=action: medium_lib.run_shoot_projectile(
                    ctypes.byref(c_state), int(action)
                ),
                f"shoot_projectile seed={seed} case={name}",
            )


def test_cast_spell_native_parity(medium_lib, jax_context, stepped_states):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        base_state = _empty_projectiles(_base_action_state(base_state)).replace(
            player_mana=6,
            learned_spells=jnp.array([True, True], dtype=bool),
            achievements=base_state.achievements.at[Achievement.CAST_FIREBALL.value]
            .set(False)
            .at[Achievement.CAST_ICEBALL.value]
            .set(False),
        )
        cases = [
            (
                f"fire_direction_{direction}",
                base_state.replace(player_direction=direction),
                Action.CAST_FIREBALL.value,
            )
            for direction in DIRECTION_ACTIONS
        ]
        cases.extend(
            [
                ("ice_learned", base_state, Action.CAST_ICEBALL.value),
                (
                    "fire_unlearned",
                    base_state.replace(learned_spells=jnp.array([False, True], dtype=bool)),
                    Action.CAST_FIREBALL.value,
                ),
                (
                    "ice_unlearned",
                    base_state.replace(learned_spells=jnp.array([True, False], dtype=bool)),
                    Action.CAST_ICEBALL.value,
                ),
                (
                    "fire_no_mana",
                    base_state.replace(player_mana=1),
                    Action.CAST_FIREBALL.value,
                ),
                (
                    "ice_no_mana",
                    base_state.replace(player_mana=1),
                    Action.CAST_ICEBALL.value,
                ),
                (
                    "full_projectiles",
                    _full_projectiles(base_state),
                    Action.CAST_FIREBALL.value,
                ),
                ("noop", base_state, Action.NOOP.value),
            ]
        )
        for name, state, action in cases:
            expected = cast_spell(state, action, static_params)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, action=action: medium_lib.run_cast_spell(
                    ctypes.byref(c_state), int(action)
                ),
                f"cast_spell seed={seed} case={name}",
            )


def test_enchant_native_parity(medium_lib, stepped_states):
    element_cases = [
        ("fire", BlockType.ENCHANTMENT_TABLE_FIRE.value, 1),
        ("ice", BlockType.ENCHANTMENT_TABLE_ICE.value, 2),
    ]
    for seed, base_state in stepped_states.items():
        cases = []
        for element_name, block, enchantment_type in element_cases:
            element_state = _base_enchant_state(base_state, block, enchantment_type)
            cases.extend(
                [
                    (
                        f"{element_name}_sword",
                        element_state,
                        Action.ENCHANT_SWORD.value,
                    ),
                    (
                        f"{element_name}_bow",
                        element_state,
                        Action.ENCHANT_BOW.value,
                    ),
                ]
            )
            for slot in range(4):
                enchantments = jnp.full((4,), enchantment_type, dtype=jnp.int32)
                enchantments = enchantments.at[slot].set(0)
                cases.append(
                    (
                        f"{element_name}_armour_slot_{slot}",
                        element_state.replace(armour_enchantments=enchantments),
                        Action.ENCHANT_ARMOUR.value,
                    )
                )

            opposite_type = 2 if enchantment_type == 1 else 1
            opposite_state = element_state.replace(
                armour_enchantments=jnp.array(
                    [enchantment_type, opposite_type, enchantment_type, enchantment_type],
                    dtype=jnp.int32,
                )
            )
            cases.append(
                (
                    f"{element_name}_armour_opposite_fallback",
                    opposite_state,
                    Action.ENCHANT_ARMOUR.value,
                )
            )

        fire_state = _base_enchant_state(
            base_state,
            BlockType.ENCHANTMENT_TABLE_FIRE.value,
            1,
        )
        cases.extend(
            [
                ("no_mana", fire_state.replace(player_mana=8), Action.ENCHANT_SWORD.value),
                (
                    "no_gem",
                    _with_inventory(fire_state, ruby=0),
                    Action.ENCHANT_SWORD.value,
                ),
                (
                    "not_table",
                    _set_target_block(fire_state, BlockType.GRASS.value),
                    Action.ENCHANT_SWORD.value,
                ),
                (
                    "no_sword",
                    _with_inventory(fire_state, sword=0),
                    Action.ENCHANT_SWORD.value,
                ),
                (
                    "no_armour",
                    _with_inventory(
                        fire_state,
                        armour=jnp.zeros((4,), dtype=jnp.int32),
                    ),
                    Action.ENCHANT_ARMOUR.value,
                ),
                ("noop", fire_state, Action.NOOP.value),
            ]
        )

        for case_index, (name, state, action) in enumerate(cases):
            rng_words = _rng_words(seed * 1000 + case_index)
            rng = jnp.asarray(rng_words, dtype=jnp.uint32)
            expected = enchant(rng, state, action)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, action=action, rng_words=rng_words: medium_lib.run_enchant(
                    ctypes.byref(c_state),
                    int(action),
                    int(rng_words[0]),
                    int(rng_words[1]),
                ),
                f"enchant seed={seed} case={name}",
            )


def test_change_floor_native_parity(medium_lib, jax_context, stepped_states):
    _env, params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        base_state = base_state.replace(player_xp=0)
        cases = []

        for level in range(8):
            position = np.asarray(base_state.down_ladders[level], dtype=np.int32)
            state = _floor_state(
                base_state,
                level,
                ItemType.LADDER_DOWN.value,
                position,
                monsters_killed=8,
            )
            state = _set_floor_achievement(state, level + 1, False)
            cases.append((f"descend_level_{level}", state, Action.DESCEND.value))

            cleared = _set_floor_achievement(state, level + 1, True)
            cases.append(
                (f"descend_level_{level}_already_cleared", cleared, Action.DESCEND.value)
            )

        for level in range(1, 9):
            position = np.asarray(base_state.up_ladders[level], dtype=np.int32)
            state = _floor_state(
                base_state,
                level,
                ItemType.LADDER_UP.value,
                position,
                monsters_killed=8,
            )
            state = _set_floor_achievement(state, level - 1, False)
            cases.append((f"ascend_level_{level}", state, Action.ASCEND.value))

        blocked_position = np.asarray(base_state.down_ladders[2], dtype=np.int32)
        blocked = _floor_state(
            base_state,
            2,
            ItemType.LADDER_DOWN.value,
            blocked_position,
            monsters_killed=7,
        )
        blocked = _set_floor_achievement(blocked, 2, True)
        cases.extend(
            [
                ("insufficient_monsters_killed", blocked, Action.DESCEND.value),
                (
                    "not_on_ladder",
                    blocked.replace(
                        item_map=blocked.item_map.at[
                            2,
                            int(blocked_position[0]),
                            int(blocked_position[1]),
                        ].set(ItemType.NONE.value)
                    ),
                    Action.DESCEND.value,
                ),
                ("noop", blocked, Action.NOOP.value),
            ]
        )

        for name, state, action in cases:
            expected = change_floor(state, action, params, static_params)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, action=action: medium_lib.run_change_floor(
                    ctypes.byref(c_state), int(action)
                ),
                f"change_floor seed={seed} case={name}",
            )


def test_add_items_from_chest_native_parity(
    medium_lib,
    stepped_states,
    chest_target_keys,
):
    for seed, base_state in stepped_states.items():
        random_base = _with_inventory(
            _base_action_state(base_state).replace(
                chests_opened=jnp.ones((9,), dtype=bool),
                player_level=0,
            ),
            coal=0,
            iron=0,
            diamond=0,
            sapphire=0,
            ruby=0,
            torches=0,
            arrows=0,
            pickaxe=0,
            sword=0,
            bow=0,
            potions=jnp.zeros((6,), dtype=jnp.int32),
            books=0,
        )
        random_cases = [
            (
                f"seeded_random_{case_index}",
                random_base,
                True,
                _rng_words(seed * 100 + case_index),
            )
            for case_index in range(2)
        ]

        targeted_cases = [
            (
                f"potion_{idx}",
                random_base,
                True,
                chest_target_keys[f"potion_{idx}"],
            )
            for idx in range(6)
        ]
        targeted_cases.extend(
            [
                ("sapphire_roll", random_base, True, chest_target_keys["sapphire"]),
                ("ruby_roll", random_base, True, chest_target_keys["ruby"]),
                (
                    "not_opening",
                    random_base,
                    False,
                    _rng_words(seed * 100 + 50),
                ),
                (
                    "special_book_level_3",
                    random_base.replace(
                        player_level=3,
                        chests_opened=random_base.chests_opened.at[3].set(False),
                    ),
                    True,
                    _rng_words(seed * 100 + 51),
                ),
                (
                    "special_book_already_opened",
                    random_base.replace(player_level=3),
                    True,
                    _rng_words(seed * 100 + 52),
                ),
                (
                    "special_book_level_4",
                    random_base.replace(
                        player_level=4,
                        chests_opened=random_base.chests_opened.at[4].set(False),
                    ),
                    True,
                    _rng_words(seed * 100 + 53),
                ),
                (
                    "special_bow_level_1",
                    random_base.replace(
                        player_level=1,
                        chests_opened=random_base.chests_opened.at[1].set(False),
                    ),
                    True,
                    _rng_words(seed * 100 + 54),
                ),
                (
                    "special_bow_already_opened",
                    random_base.replace(player_level=1),
                    True,
                    _rng_words(seed * 100 + 55),
                ),
            ]
        )

        for name, state, is_opening_chest, rng_words in random_cases + targeted_cases:
            rng = jnp.asarray(rng_words, dtype=jnp.uint32)
            expected = _chest_expected_state(rng, state, is_opening_chest)
            _assert_native_matches(
                state,
                expected,
                lambda c_state, is_opening_chest=is_opening_chest, rng_words=rng_words: (
                    medium_lib.run_add_items_from_chest(
                        ctypes.byref(c_state),
                        bool(is_opening_chest),
                        int(rng_words[0]),
                        int(rng_words[1]),
                    )
                ),
                f"add_items_from_chest seed={seed} case={name}",
            )
