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

from craftax.craftax.constants import Action, BlockType
from craftax.craftax.game_logic import do_action
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

MINING_CASES = (
    ("tree", BlockType.TREE.value, 0),
    ("fire_tree", BlockType.FIRE_TREE.value, 0),
    ("ice_shrub", BlockType.ICE_SHRUB.value, 0),
    ("stone", BlockType.STONE.value, 1),
    ("coal", BlockType.COAL.value, 1),
    ("iron", BlockType.IRON.value, 2),
    ("diamond", BlockType.DIAMOND.value, 3),
    ("sapphire", BlockType.SAPPHIRE.value, 4),
    ("ruby", BlockType.RUBY.value, 4),
    ("stalagmite", BlockType.STALAGMITE.value, 1),
    ("furnace", BlockType.FURNACE.value, 0),
    ("crafting_table", BlockType.CRAFTING_TABLE.value, 0),
    ("wood_block_jax_noop", BlockType.WOOD.value, 0),
)

MOB_KILL_CASES = (
    ("passive_cow", "passive", 0),
    ("passive_bat", "passive", 1),
    ("passive_snail", "passive", 2),
    ("melee_zombie", "melee", 0),
    ("melee_gnome", "melee", 1),
    ("melee_orc", "melee", 2),
    ("melee_lizard", "melee", 3),
    ("melee_knight", "melee", 4),
    ("melee_troll", "melee", 5),
    ("melee_pigman", "melee", 6),
    ("melee_frost_troll", "melee", 7),
    ("ranged_skeleton", "ranged", 0),
    ("ranged_gnome_archer", "ranged", 1),
    ("ranged_orc_mage", "ranged", 2),
    ("ranged_kobold", "ranged", 3),
    ("ranged_archer", "ranged", 4),
    ("ranged_deep_thing", "ranged", 5),
    ("ranged_fire_elemental", "ranged", 6),
    ("ranged_ice_elemental", "ranged", 7),
)


@pytest.fixture(scope="session")
def do_action_lib():
    source = r"""
    #include <stdbool.h>
    #include <stdint.h>
    #include <stddef.h>
    #include "ocean/craftax/step_do_action.h"

    size_t craftax_test_state_size(void) {
        return sizeof(CraftaxState);
    }

    void run_do_action(
        CraftaxState* state,
        int32_t action,
        uint32_t rng0,
        uint32_t rng1
    ) {
        CraftaxThreefryKey rng = {{rng0, rng1}};
        craftax_do_action_native(state, action, rng);
    }
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    src = tmp_path / "craftax_step_do_action_test.c"
    so = tmp_path / "craftax_step_do_action_test.so"
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

    lib.run_do_action.argtypes = [
        state_ptr,
        ctypes.c_int32,
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


def _assert_do_action_matches(do_action_lib, state, rng_words, action, static_params, context):
    rng = jnp.asarray(rng_words, dtype=jnp.uint32)
    expected = do_action(rng, state, int(action), static_params)
    _assert_native_matches(
        state,
        expected,
        lambda c_state: do_action_lib.run_do_action(
            ctypes.byref(c_state),
            int(action),
            int(rng_words[0]),
            int(rng_words[1]),
        ),
        context,
    )


def _assert_sequence_matches(do_action_lib, state, actions, rng_words_seq, static_params, context):
    expected = state
    c_state = jax_state_to_c_state(state)
    for action, rng_words in zip(actions, rng_words_seq, strict=True):
        rng = jnp.asarray(rng_words, dtype=jnp.uint32)
        expected = do_action(rng, expected, int(action), static_params)
        do_action_lib.run_do_action(
            ctypes.byref(c_state),
            int(action),
            int(rng_words[0]),
            int(rng_words[1]),
        )
    actual = craftax_state_to_jax(c_state, template=state)
    assert_env_states_equal(actual, expected, context)


def _rng_words(seed):
    return np.asarray(jax.random.PRNGKey(seed), dtype=np.uint32)


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


def _clear_mobs(state):
    return state.replace(
        mob_map=jnp.zeros_like(state.mob_map),
        melee_mobs=state.melee_mobs.replace(mask=jnp.zeros_like(state.melee_mobs.mask)),
        passive_mobs=state.passive_mobs.replace(mask=jnp.zeros_like(state.passive_mobs.mask)),
        ranged_mobs=state.ranged_mobs.replace(mask=jnp.zeros_like(state.ranged_mobs.mask)),
        mob_projectiles=state.mob_projectiles.replace(
            mask=jnp.zeros_like(state.mob_projectiles.mask)
        ),
        player_projectiles=state.player_projectiles.replace(
            mask=jnp.zeros_like(state.player_projectiles.mask)
        ),
    )


def _base_action_state(
    state,
    level=0,
    position=(24, 24),
    direction=Action.RIGHT.value,
):
    state = _clear_mobs(_empty_inventory_state(state))
    return state.replace(
        player_level=level,
        player_position=jnp.array(position, dtype=jnp.int32),
        player_direction=direction,
        player_food=1,
        player_drink=1,
        player_hunger=5.0,
        player_thirst=5.0,
        player_mana=1,
        player_dexterity=1,
        player_strength=1,
        player_intelligence=1,
        achievements=jnp.zeros_like(state.achievements),
        monsters_killed=jnp.zeros_like(state.monsters_killed),
    )


def _target_position(state):
    return np.asarray(
        state.player_position + DIRECTION_VECTORS[int(state.player_direction)],
        dtype=np.int32,
    )


def _set_target_block(state, block):
    level = int(state.player_level)
    target = _target_position(state)
    if 0 <= target[0] < 48 and 0 <= target[1] < 48:
        return state.replace(
            map=state.map.at[level, int(target[0]), int(target[1])].set(block),
            mob_map=state.mob_map.at[level, int(target[0]), int(target[1])].set(False),
        )
    return state


def _set_cell_block(state, row, col, block, level=None):
    level = int(state.player_level) if level is None else int(level)
    return state.replace(map=state.map.at[level, int(row), int(col)].set(block))


def _with_growing_plant_at_target(state, index=3):
    target = _target_position(state)
    positions = jnp.arange(20, dtype=jnp.int32).reshape((10, 2))
    ages = jnp.arange(10, dtype=jnp.int32) * 13 + 7
    positions = positions.at[index].set(jnp.array(target, dtype=jnp.int32))
    return state.replace(
        growing_plants_positions=positions,
        growing_plants_age=ages,
        growing_plants_mask=jnp.ones((10,), dtype=bool),
    )


def _set_target_mob(state, mob_class, type_id, health, slot=0):
    level = int(state.player_level)
    target = _target_position(state)
    target_value = jnp.array(target, dtype=jnp.int32)

    if mob_class == "passive":
        mobs = state.passive_mobs.replace(
            position=state.passive_mobs.position.at[level, slot].set(target_value),
            health=state.passive_mobs.health.at[level, slot].set(float(health)),
            mask=state.passive_mobs.mask.at[level, slot].set(True),
            type_id=state.passive_mobs.type_id.at[level, slot].set(type_id),
        )
        state = state.replace(passive_mobs=mobs)
    elif mob_class == "melee":
        mobs = state.melee_mobs.replace(
            position=state.melee_mobs.position.at[level, slot].set(target_value),
            health=state.melee_mobs.health.at[level, slot].set(float(health)),
            mask=state.melee_mobs.mask.at[level, slot].set(True),
            type_id=state.melee_mobs.type_id.at[level, slot].set(type_id),
        )
        state = state.replace(melee_mobs=mobs)
    elif mob_class == "ranged":
        mobs = state.ranged_mobs.replace(
            position=state.ranged_mobs.position.at[level, slot].set(target_value),
            health=state.ranged_mobs.health.at[level, slot].set(float(health)),
            mask=state.ranged_mobs.mask.at[level, slot].set(True),
            type_id=state.ranged_mobs.type_id.at[level, slot].set(type_id),
        )
        state = state.replace(ranged_mobs=mobs)
    else:
        raise ValueError(mob_class)

    return state.replace(
        mob_map=state.mob_map.at[level, int(target[0]), int(target[1])].set(True)
    )


def _set_mob_projectile_at_target(state):
    level = int(state.player_level)
    target = _target_position(state)
    projectiles = state.mob_projectiles.replace(
        position=state.mob_projectiles.position.at[level, 0].set(
            jnp.array(target, dtype=jnp.int32)
        ),
        health=state.mob_projectiles.health.at[level, 0].set(1.0),
        mask=state.mob_projectiles.mask.at[level, 0].set(True),
        type_id=state.mob_projectiles.type_id.at[level, 0].set(0),
    )
    return state.replace(mob_projectiles=projectiles)


def _chest_state(base_state, level, already_opened=False):
    chests_opened = jnp.ones((9,), dtype=bool).at[level].set(bool(already_opened))
    state = _base_action_state(base_state, level=level).replace(
        chests_opened=chests_opened
    )
    state = _with_inventory(
        state,
        pickaxe=0,
        sword=0,
        bow=0,
        arrows=0,
        torches=0,
        coal=0,
        iron=0,
        diamond=0,
        sapphire=0,
        ruby=0,
        potions=jnp.zeros((6,), dtype=jnp.int32),
        books=0,
    )
    return _set_target_block(state, BlockType.CHEST.value)


def _sapling_rng(want_sapling):
    for seed in range(10000):
        rng = jax.random.PRNGKey(seed)
        _carry, draw = jax.random.split(rng)
        has_sapling = bool(jax.random.uniform(draw) < 0.1)
        if has_sapling == want_sapling:
            return _rng_words(seed)
    raise AssertionError("could not find sapling rng")


def _sequence_case(seed, base_state):
    state = _with_inventory(
        _base_action_state(base_state),
        pickaxe=4,
        sword=4,
    )
    case_index = seed % 16
    if case_index == 0:
        return _set_target_block(state, BlockType.TREE.value)
    if case_index == 1:
        return _set_target_block(state, BlockType.STONE.value)
    if case_index == 2:
        return _set_target_block(state, BlockType.COAL.value)
    if case_index == 3:
        return _set_target_block(state, BlockType.IRON.value)
    if case_index == 4:
        return _set_target_block(state, BlockType.DIAMOND.value)
    if case_index == 5:
        return _set_target_block(state, BlockType.SAPPHIRE.value)
    if case_index == 6:
        return _set_target_block(state, BlockType.RUBY.value)
    if case_index == 7:
        return _with_growing_plant_at_target(
            _set_target_block(state, BlockType.RIPE_PLANT.value)
        )
    if case_index == 8:
        return _set_target_block(state, BlockType.WATER.value)
    if case_index == 9:
        return _set_target_block(state, BlockType.FOUNTAIN.value)
    if case_index == 10:
        return _chest_state(base_state, level=0)
    if case_index == 11:
        return _chest_state(base_state, level=1)
    if case_index == 12:
        return _set_target_mob(
            _set_target_block(state, BlockType.PATH.value),
            "passive",
            0,
            1.0,
        )
    if case_index == 13:
        return _set_target_mob(
            _set_target_block(state, BlockType.PATH.value),
            "melee",
            0,
            1.0,
        )
    if case_index == 14:
        return _set_target_mob(
            _set_target_block(state, BlockType.PATH.value),
            "ranged",
            0,
            1.0,
        )
    return _set_target_block(state, BlockType.PATH.value)


def test_do_action_seeded_sequence_native_parity(
    do_action_lib,
    jax_context,
    stepped_states,
):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        state = _sequence_case(seed, base_state)
        _assert_sequence_matches(
            do_action_lib,
            state,
            [Action.NOOP.value, Action.DO.value],
            [_rng_words(seed * 1000), _rng_words(seed * 1000 + 1)],
            static_params,
            f"seeded_sequence seed={seed}",
        )


def test_do_action_mining_native_parity(do_action_lib, jax_context, stepped_states):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        for name, block, required_pickaxe in MINING_CASES:
            success = _with_inventory(
                _base_action_state(base_state),
                pickaxe=max(required_pickaxe, 0),
            )
            cases = [("success", _set_target_block(success, block))]
            if required_pickaxe > 0:
                blocked = _with_inventory(success, pickaxe=required_pickaxe - 1)
                cases.append(("missing_pickaxe", _set_target_block(blocked, block)))

            for case_name, state in cases:
                _assert_do_action_matches(
                    do_action_lib,
                    state,
                    _rng_words(seed * 2000 + block),
                    Action.DO.value,
                    static_params,
                    f"mining seed={seed} block={name} case={case_name}",
                )


def test_do_action_sapling_roll_native_parity(do_action_lib, jax_context, stepped_states):
    _env, _params, static_params = jax_context
    rng_cases = [
        ("sapling", _sapling_rng(True)),
        ("no_sapling", _sapling_rng(False)),
    ]
    for seed, base_state in stepped_states.items():
        state = _set_target_block(_base_action_state(base_state), BlockType.GRASS.value)
        for name, rng_words in rng_cases:
            _assert_do_action_matches(
                do_action_lib,
                state,
                rng_words,
                Action.DO.value,
                static_params,
                f"sapling seed={seed} case={name}",
            )


def test_do_action_food_and_drink_native_parity(
    do_action_lib,
    jax_context,
    stepped_states,
):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        plant = _with_growing_plant_at_target(
            _set_target_block(
                _base_action_state(base_state).replace(player_food=1, player_hunger=9.0),
                BlockType.RIPE_PLANT.value,
            )
        )
        plant_cap = _with_growing_plant_at_target(
            _set_target_block(
                _base_action_state(base_state).replace(
                    player_dexterity=5,
                    player_food=16,
                    player_hunger=9.0,
                ),
                BlockType.RIPE_PLANT.value,
            )
        )
        water = _set_target_block(
            _base_action_state(base_state).replace(player_drink=1, player_thirst=9.0),
            BlockType.WATER.value,
        )
        water_cap = _set_target_block(
            _base_action_state(base_state).replace(
                player_dexterity=5,
                player_drink=16,
                player_thirst=9.0,
            ),
            BlockType.WATER.value,
        )
        fountain = _set_target_block(
            _base_action_state(base_state).replace(
                player_drink=1,
                player_thirst=9.0,
                player_mana=0,
            ),
            BlockType.FOUNTAIN.value,
        )
        cases = [
            ("ripe_plant", plant),
            ("ripe_plant_cap", plant_cap),
            ("water", water),
            ("water_cap", water_cap),
            ("fountain", fountain),
        ]

        for passive_type in range(3):
            for dexterity in (1, 5):
                passive = _with_inventory(
                    _base_action_state(base_state).replace(
                        player_dexterity=dexterity,
                        player_food=1 if dexterity == 1 else 16,
                        player_hunger=9.0,
                    ),
                    sword=4,
                )
                passive = _set_target_mob(
                    _set_target_block(passive, BlockType.PATH.value),
                    "passive",
                    passive_type,
                    1.0,
                )
                cases.append((f"passive_{passive_type}_dex_{dexterity}", passive))

        for case_index, (name, state) in enumerate(cases):
            _assert_do_action_matches(
                do_action_lib,
                state,
                _rng_words(seed * 3000 + case_index),
                Action.DO.value,
                static_params,
                f"food_drink seed={seed} case={name}",
            )


def test_do_action_chest_level_variants_native_parity(
    do_action_lib,
    jax_context,
    stepped_states,
):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        cases = [(f"level_{level}", _chest_state(base_state, level)) for level in range(9)]
        cases.extend(
            [
                ("level_1_already_opened", _chest_state(base_state, 1, True)),
                ("level_3_already_opened", _chest_state(base_state, 3, True)),
                ("level_4_already_opened", _chest_state(base_state, 4, True)),
            ]
        )
        for case_index, (name, state) in enumerate(cases):
            _assert_do_action_matches(
                do_action_lib,
                state,
                _rng_words(seed * 4000 + case_index),
                Action.DO.value,
                static_params,
                f"chest seed={seed} case={name}",
            )


def test_do_action_attack_kill_achievements_native_parity(
    do_action_lib,
    jax_context,
    stepped_states,
):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        for case_index, (name, mob_class, type_id) in enumerate(MOB_KILL_CASES):
            state = _with_inventory(
                _base_action_state(base_state).replace(
                    player_food=1,
                    player_hunger=9.0,
                    player_strength=5,
                    player_intelligence=5,
                ),
                sword=4,
            )
            state = _set_target_mob(
                _set_target_block(state, BlockType.PATH.value),
                mob_class,
                type_id,
                0.5,
            )
            _assert_do_action_matches(
                do_action_lib,
                state,
                _rng_words(seed * 5000 + case_index),
                Action.DO.value,
                static_params,
                f"attack_kill seed={seed} case={name}",
            )


def test_do_action_attack_damage_modifiers_native_parity(
    do_action_lib,
    jax_context,
    stepped_states,
):
    _env, _params, static_params = jax_context
    damage_cases = [
        ("passive_no_sword", "passive", 0, 0, 0, 1, 1),
        ("melee_no_sword", "melee", 4, 0, 0, 1, 1),
        ("ranged_no_sword", "ranged", 4, 0, 0, 1, 1),
        ("melee_no_enchant", "melee", 6, 4, 0, 5, 5),
        ("melee_fire_enchant", "melee", 6, 4, 1, 5, 5),
        ("melee_ice_enchant", "melee", 7, 4, 2, 5, 5),
        ("ranged_strength_1", "ranged", 5, 4, 0, 1, 1),
        ("ranged_strength_5", "ranged", 5, 4, 0, 5, 1),
    ]

    for seed, base_state in stepped_states.items():
        for case_index, (
            name,
            mob_class,
            type_id,
            sword,
            enchantment,
            strength,
            intelligence,
        ) in enumerate(damage_cases):
            state = _with_inventory(
                _base_action_state(base_state).replace(
                    player_strength=strength,
                    player_intelligence=intelligence,
                ),
                sword=sword,
            ).replace(sword_enchantment=enchantment)
            state = _set_target_mob(
                _set_target_block(state, BlockType.PATH.value),
                mob_class,
                type_id,
                50.0,
            )
            _assert_do_action_matches(
                do_action_lib,
                state,
                _rng_words(seed * 6000 + case_index),
                Action.DO.value,
                static_params,
                f"attack_damage seed={seed} case={name}",
            )


def test_do_action_edge_cases_native_parity(do_action_lib, jax_context, stepped_states):
    _env, _params, static_params = jax_context
    for seed, base_state in stepped_states.items():
        no_block = _set_target_block(
            _base_action_state(base_state),
            BlockType.PATH.value,
        )

        out_up = _base_action_state(
            base_state,
            position=(0, 0),
            direction=Action.UP.value,
        )
        out_up = _set_cell_block(out_up, 47, 0, BlockType.PATH.value)

        out_left = _base_action_state(
            base_state,
            position=(0, 0),
            direction=Action.LEFT.value,
        )
        out_left = _set_cell_block(out_left, 0, 47, BlockType.PATH.value)

        out_down = _base_action_state(
            base_state,
            position=(47, 47),
            direction=Action.DOWN.value,
        )
        out_down = _set_cell_block(out_down, 47, 47, BlockType.PATH.value)

        out_right = _base_action_state(
            base_state,
            position=(47, 47),
            direction=Action.RIGHT.value,
        )
        out_right = _set_cell_block(out_right, 47, 47, BlockType.PATH.value)

        projectile = _with_inventory(_base_action_state(base_state), pickaxe=1)
        projectile = _set_mob_projectile_at_target(
            _set_target_block(projectile, BlockType.STONE.value)
        )

        mob_on_chest = _with_inventory(_base_action_state(base_state), sword=4)
        mob_on_chest = _set_target_mob(
            _set_target_block(mob_on_chest, BlockType.CHEST.value),
            "melee",
            0,
            10.0,
        )

        cases = [
            ("path_noop", no_block),
            ("out_of_bounds_up", out_up),
            ("out_of_bounds_left", out_left),
            ("out_of_bounds_down", out_down),
            ("out_of_bounds_right", out_right),
            ("occupied_by_projectile", projectile),
            ("mob_on_chest_blocks_block_effects", mob_on_chest),
        ]

        for case_index, (name, state) in enumerate(cases):
            _assert_do_action_matches(
                do_action_lib,
                state,
                _rng_words(seed * 7000 + case_index),
                Action.DO.value,
                static_params,
                f"edge seed={seed} case={name}",
            )
