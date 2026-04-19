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

from craftax.craftax.constants import Achievement, Action, BlockType, ProjectileType
from craftax.craftax.game_logic import update_mobs
from craftax.craftax_env import make_craftax_env_from_name

from tests.craftax_state_fixtures import (
    CraftaxState,
    assert_env_states_equal,
    craftax_state_to_jax,
    jax_state_to_c_state,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(16))
MAP_SIZE = 48
FLOOR_MOB_TYPES = (0, 2, 1, 3, 4, 5, 6, 7, 0)
CLASS_SPECS = (
    ("passive", "passive_mobs", 3),
    ("melee", "melee_mobs", 3),
    ("ranged", "ranged_mobs", 2),
    ("mob_projectile", "mob_projectiles", 3),
    ("player_projectile", "player_projectiles", 3),
)


@pytest.fixture(scope="session")
def update_mobs_lib():
    source = r"""
    #include <stdbool.h>
    #include <stdint.h>
    #include <stddef.h>
    #include "ocean/craftax/step_update_mobs.h"

    size_t craftax_test_state_size(void) {
        return sizeof(CraftaxState);
    }

    void run_update_mobs(CraftaxState* state, uint32_t rng0, uint32_t rng1) {
        CraftaxThreefryKey rng = {{rng0, rng1}};
        craftax_update_mobs_native(state, rng);
    }
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    src = tmp_path / "craftax_step_update_mobs_test.c"
    so = tmp_path / "craftax_step_update_mobs_test.so"
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

    lib.run_update_mobs.argtypes = [
        state_ptr,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    return lib


@pytest.fixture(scope="session")
def jax_context():
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    return env, env.default_params, env.static_env_params


@pytest.fixture(scope="session")
def rng_stepped_states(jax_context):
    env, params, _static_params = jax_context
    states = {}
    for seed in SEEDS:
        rng = jax.random.PRNGKey(seed)
        rng, reset_key = jax.random.split(rng)
        _obs, state = env.reset(reset_key, params)
        for step in range(8 + seed % 5):
            rng, action_key = jax.random.split(rng)
            action = int(jax.random.randint(action_key, (), 0, 43))
            rng, step_key = jax.random.split(rng)
            _obs, state, _reward, _done, _info = env.step(
                step_key,
                state,
                action,
                params,
            )
        states[seed] = state
    return states


def _rng_words(seed):
    return np.asarray(jax.random.PRNGKey(seed), dtype=np.uint32)


def _assert_native_matches(update_mobs_lib, state, expected, rng_words, context):
    c_state = jax_state_to_c_state(state)
    update_mobs_lib.run_update_mobs(
        ctypes.byref(c_state),
        int(rng_words[0]),
        int(rng_words[1]),
    )
    actual = craftax_state_to_jax(c_state, template=state)
    assert_env_states_equal(actual, expected, context)
    return actual


def _assert_update_mobs_matches(
    update_mobs_lib,
    state,
    rng_words,
    params,
    static_params,
    context,
):
    rng = jnp.asarray(rng_words, dtype=jnp.uint32)
    expected = update_mobs(rng, state, params, static_params)
    actual = _assert_native_matches(update_mobs_lib, state, expected, rng_words, context)
    return expected, actual


def _empty_mobs(mobs):
    return mobs.replace(
        position=jnp.zeros_like(mobs.position),
        health=jnp.zeros_like(mobs.health),
        mask=jnp.zeros_like(mobs.mask),
        attack_cooldown=jnp.zeros_like(mobs.attack_cooldown),
        type_id=jnp.zeros_like(mobs.type_id),
    )


def _clear_mobs(state):
    return state.replace(
        mob_map=jnp.zeros_like(state.mob_map),
        melee_mobs=_empty_mobs(state.melee_mobs),
        passive_mobs=_empty_mobs(state.passive_mobs),
        ranged_mobs=_empty_mobs(state.ranged_mobs),
        mob_projectiles=_empty_mobs(state.mob_projectiles),
        mob_projectile_directions=jnp.zeros_like(state.mob_projectile_directions),
        player_projectiles=_empty_mobs(state.player_projectiles),
        player_projectile_directions=jnp.zeros_like(state.player_projectile_directions),
    )


def _with_inventory(state, **kwargs):
    return state.replace(inventory=state.inventory.replace(**kwargs))


def _base_state(
    state,
    level=0,
    player_position=(24, 24),
    fill_block=BlockType.PATH.value,
):
    floor = jnp.full((MAP_SIZE, MAP_SIZE), int(fill_block), dtype=jnp.int32)
    state = _clear_mobs(state)
    state = _with_inventory(
        state,
        sword=0,
        bow=1,
        armour=jnp.zeros((4,), dtype=jnp.int32),
    )
    return state.replace(
        map=state.map.at[level].set(floor),
        player_level=int(level),
        player_position=jnp.asarray(player_position, dtype=jnp.int32),
        player_direction=Action.RIGHT.value,
        player_health=np.float32(12.0),
        player_food=3,
        player_hunger=np.float32(7.0),
        player_dexterity=1,
        player_strength=1,
        player_intelligence=1,
        is_sleeping=False,
        is_resting=False,
        achievements=jnp.zeros_like(state.achievements),
        monsters_killed=jnp.zeros_like(state.monsters_killed),
        armour_enchantments=jnp.zeros_like(state.armour_enchantments),
        sword_enchantment=0,
        bow_enchantment=0,
        boss_progress=0,
        boss_timesteps_to_spawn_this_round=0,
    )


def _set_cell(state, level, position, block):
    row, col = position
    return state.replace(
        map=state.map.at[int(level), int(row), int(col)].set(int(block))
    )


def _open_cell(state, level, position):
    return _set_cell(state, level, position, BlockType.PATH.value)


def _set_mob(
    state,
    mob_class,
    level,
    position,
    type_id,
    health=10.0,
    cooldown=0,
    slot=0,
    mask=True,
):
    value = jnp.asarray(position, dtype=jnp.int32)
    if mob_class == "passive":
        mobs = state.passive_mobs.replace(
            position=state.passive_mobs.position.at[level, slot].set(value),
            health=state.passive_mobs.health.at[level, slot].set(float(health)),
            mask=state.passive_mobs.mask.at[level, slot].set(bool(mask)),
            attack_cooldown=state.passive_mobs.attack_cooldown.at[level, slot].set(
                int(cooldown)
            ),
            type_id=state.passive_mobs.type_id.at[level, slot].set(int(type_id)),
        )
        state = state.replace(passive_mobs=mobs)
    elif mob_class == "melee":
        mobs = state.melee_mobs.replace(
            position=state.melee_mobs.position.at[level, slot].set(value),
            health=state.melee_mobs.health.at[level, slot].set(float(health)),
            mask=state.melee_mobs.mask.at[level, slot].set(bool(mask)),
            attack_cooldown=state.melee_mobs.attack_cooldown.at[level, slot].set(
                int(cooldown)
            ),
            type_id=state.melee_mobs.type_id.at[level, slot].set(int(type_id)),
        )
        state = state.replace(melee_mobs=mobs)
    elif mob_class == "ranged":
        mobs = state.ranged_mobs.replace(
            position=state.ranged_mobs.position.at[level, slot].set(value),
            health=state.ranged_mobs.health.at[level, slot].set(float(health)),
            mask=state.ranged_mobs.mask.at[level, slot].set(bool(mask)),
            attack_cooldown=state.ranged_mobs.attack_cooldown.at[level, slot].set(
                int(cooldown)
            ),
            type_id=state.ranged_mobs.type_id.at[level, slot].set(int(type_id)),
        )
        state = state.replace(ranged_mobs=mobs)
    else:
        raise ValueError(mob_class)

    if mask:
        row, col = position
        state = state.replace(
            mob_map=state.mob_map.at[level, int(row), int(col)].set(True)
        )
    return _open_cell(state, level, position)


def _set_mob_projectile(
    state,
    level,
    position,
    direction,
    projectile_type=ProjectileType.ARROW.value,
    slot=0,
    mask=True,
):
    projectiles = state.mob_projectiles.replace(
        position=state.mob_projectiles.position.at[level, slot].set(
            jnp.asarray(position, dtype=jnp.int32)
        ),
        health=state.mob_projectiles.health.at[level, slot].set(1.0),
        mask=state.mob_projectiles.mask.at[level, slot].set(bool(mask)),
        type_id=state.mob_projectiles.type_id.at[level, slot].set(
            int(projectile_type)
        ),
    )
    directions = state.mob_projectile_directions.at[level, slot].set(
        jnp.asarray(direction, dtype=jnp.int32)
    )
    state = state.replace(
        mob_projectiles=projectiles,
        mob_projectile_directions=directions,
    )
    return _open_cell(state, level, position)


def _set_player_projectile(
    state,
    level,
    position,
    direction,
    projectile_type=ProjectileType.ARROW2.value,
    slot=0,
    mask=True,
):
    projectiles = state.player_projectiles.replace(
        position=state.player_projectiles.position.at[level, slot].set(
            jnp.asarray(position, dtype=jnp.int32)
        ),
        health=state.player_projectiles.health.at[level, slot].set(1.0),
        mask=state.player_projectiles.mask.at[level, slot].set(bool(mask)),
        type_id=state.player_projectiles.type_id.at[level, slot].set(
            int(projectile_type)
        ),
    )
    directions = state.player_projectile_directions.at[level, slot].set(
        jnp.asarray(direction, dtype=jnp.int32)
    )
    state = state.replace(
        player_projectiles=projectiles,
        player_projectile_directions=directions,
    )
    return _open_cell(state, level, position)


def _floor_class_state(base_state, level, mob_class):
    fill = BlockType.WATER.value if mob_class == "ranged" and level == 5 else BlockType.PATH.value
    state = _base_state(base_state, level=level, fill_block=fill)
    type_id = FLOOR_MOB_TYPES[level]
    if mob_class == "passive":
        return _set_mob(state, "passive", level, (24, 27), type_id, health=8.0)
    if mob_class == "melee":
        return _set_mob(state, "melee", level, (24, 28), type_id, health=8.0)
    if mob_class == "ranged":
        return _set_mob(state, "ranged", level, (24, 29), type_id, health=8.0)
    if mob_class == "mob_projectile":
        return _set_mob_projectile(
            state,
            level,
            (24, 27),
            (0, -1),
            projectile_type=type_id,
        )
    if mob_class == "player_projectile":
        return _set_player_projectile(
            state,
            level,
            (24, 24),
            (0, 1),
            projectile_type=ProjectileType.ARROW2.value,
        )
    raise ValueError(mob_class)


def _stone_box_state(base_state, level=0):
    state = _base_state(base_state, level=level, fill_block=BlockType.STONE.value)
    state = _open_cell(state, level, tuple(np.asarray(state.player_position)))
    return state


def test_update_mobs_native_parity_on_rng_stepped_states(
    update_mobs_lib,
    jax_context,
    rng_stepped_states,
):
    _env, params, static_params = jax_context
    for seed, state in rng_stepped_states.items():
        rng_words = _rng_words(10000 + seed)
        _assert_update_mobs_matches(
            update_mobs_lib,
            state,
            rng_words,
            params,
            static_params,
            f"rng stepped seed={seed}",
        )


@pytest.mark.parametrize("level", range(9))
@pytest.mark.parametrize("mob_class", [spec[0] for spec in CLASS_SPECS])
def test_update_mobs_each_mob_class_each_floor_native_parity(
    update_mobs_lib,
    jax_context,
    rng_stepped_states,
    level,
    mob_class,
):
    _env, params, static_params = jax_context
    base = rng_stepped_states[level % len(SEEDS)]
    state = _floor_class_state(base, level, mob_class)
    _assert_update_mobs_matches(
        update_mobs_lib,
        state,
        _rng_words(20000 + level * 31 + len(mob_class)),
        params,
        static_params,
        f"floor={level} mob_class={mob_class}",
    )


def test_update_mobs_melee_attacks_player_and_wakes_sleeping_player(
    update_mobs_lib,
    jax_context,
    rng_stepped_states,
):
    _env, params, static_params = jax_context
    level = 0
    state = _stone_box_state(rng_stepped_states[0], level).replace(
        is_sleeping=True,
        is_resting=True,
        player_health=np.float32(20.0),
    )
    state = _set_mob(
        state,
        "melee",
        level,
        (24, 25),
        0,
        health=10.0,
        cooldown=0,
    )
    expected, _actual = _assert_update_mobs_matches(
        update_mobs_lib,
        state,
        _rng_words(31001),
        params,
        static_params,
        "melee attacks sleeping player",
    )
    assert float(expected.player_health) < float(state.player_health)
    assert bool(expected.achievements[Achievement.WAKE_UP.value])
    assert not bool(expected.is_sleeping)
    assert not bool(expected.is_resting)
    assert int(expected.melee_mobs.attack_cooldown[level, 0]) == 5


def test_update_mobs_ranged_mob_fires_projectile(
    update_mobs_lib,
    jax_context,
    rng_stepped_states,
):
    _env, params, static_params = jax_context
    level = 0
    state = _base_state(rng_stepped_states[1], level=level)
    state = _set_mob(
        state,
        "ranged",
        level,
        (24, 28),
        0,
        health=8.0,
        cooldown=0,
    )
    expected, _actual = _assert_update_mobs_matches(
        update_mobs_lib,
        state,
        _rng_words(32001),
        params,
        static_params,
        "ranged fires projectile",
    )
    assert int(np.asarray(expected.mob_projectiles.mask[level]).sum()) == 1
    assert int(expected.ranged_mobs.attack_cooldown[level, 0]) == 4


def test_update_mobs_mob_projectile_hits_player(
    update_mobs_lib,
    jax_context,
    rng_stepped_states,
):
    _env, params, static_params = jax_context
    level = 0
    state = _stone_box_state(rng_stepped_states[2], level).replace(
        is_sleeping=True,
        is_resting=True,
        player_health=np.float32(20.0),
    )
    state = _set_mob_projectile(
        state,
        level,
        (24, 25),
        (0, -1),
        projectile_type=ProjectileType.ARROW.value,
    )
    expected, _actual = _assert_update_mobs_matches(
        update_mobs_lib,
        state,
        _rng_words(33001),
        params,
        static_params,
        "mob projectile hits player",
    )
    assert not bool(expected.mob_projectiles.mask[level, 0])
    assert float(expected.player_health) < float(state.player_health)
    assert not bool(expected.is_sleeping)
    assert not bool(expected.is_resting)


@pytest.mark.parametrize(
    ("name", "position", "direction", "wall_position"),
    [
        ("wall", (24, 25), (0, 1), (24, 26)),
        ("oob", (0, 0), (-1, 0), None),
    ],
)
def test_update_mobs_mob_projectile_expires_on_wall_or_oob(
    update_mobs_lib,
    jax_context,
    rng_stepped_states,
    name,
    position,
    direction,
    wall_position,
):
    _env, params, static_params = jax_context
    level = 0
    state = _stone_box_state(rng_stepped_states[3], level)
    state = _open_cell(state, level, position)
    if wall_position is not None:
        state = _set_cell(state, level, wall_position, BlockType.STONE.value)
    state = _set_mob_projectile(
        state,
        level,
        position,
        direction,
        projectile_type=ProjectileType.ARROW.value,
    )
    expected, _actual = _assert_update_mobs_matches(
        update_mobs_lib,
        state,
        _rng_words(34001 + (name == "oob")),
        params,
        static_params,
        f"mob projectile {name}",
    )
    assert not bool(expected.mob_projectiles.mask[level, 0])


def test_update_mobs_player_projectile_kills_mob_and_updates_kill_bookkeeping(
    update_mobs_lib,
    jax_context,
    rng_stepped_states,
):
    _env, params, static_params = jax_context
    level = 0
    state = _stone_box_state(rng_stepped_states[4], level)
    state = _set_mob(
        state,
        "melee",
        level,
        (24, 25),
        0,
        health=1.0,
        cooldown=99,
    )
    state = _set_player_projectile(
        state,
        level,
        (24, 24),
        (0, 1),
        projectile_type=ProjectileType.ARROW2.value,
    )
    expected, _actual = _assert_update_mobs_matches(
        update_mobs_lib,
        state,
        _rng_words(35001),
        params,
        static_params,
        "player projectile kills melee mob",
    )
    assert not bool(expected.melee_mobs.mask[level, 0])
    assert not bool(expected.player_projectiles.mask[level, 0])
    assert int(expected.monsters_killed[level]) == int(state.monsters_killed[level]) + 1
    assert bool(expected.achievements[Achievement.DEFEAT_ZOMBIE.value])
    assert int(expected.player_xp) == int(state.player_xp)


def test_update_mobs_despawns_far_mob(
    update_mobs_lib,
    jax_context,
    rng_stepped_states,
):
    _env, params, static_params = jax_context
    level = 0
    state = _stone_box_state(rng_stepped_states[5], level)
    state = _set_mob(
        state,
        "melee",
        level,
        (24, 39),
        0,
        health=8.0,
        cooldown=3,
    )
    expected, _actual = _assert_update_mobs_matches(
        update_mobs_lib,
        state,
        _rng_words(36001),
        params,
        static_params,
        "far melee despawn",
    )
    assert not bool(expected.melee_mobs.mask[level, 0])
    assert not bool(expected.mob_map[level, 24, 39])


def test_update_mobs_cooldown_decrements_when_not_attacking(
    update_mobs_lib,
    jax_context,
    rng_stepped_states,
):
    _env, params, static_params = jax_context
    level = 0
    state = _stone_box_state(rng_stepped_states[6], level)
    state = _set_mob(
        state,
        "melee",
        level,
        (24, 30),
        0,
        health=8.0,
        cooldown=3,
    )
    expected, _actual = _assert_update_mobs_matches(
        update_mobs_lib,
        state,
        _rng_words(37001),
        params,
        static_params,
        "cooldown decrement",
    )
    assert int(expected.melee_mobs.attack_cooldown[level, 0]) == 2


def test_update_mobs_empty_masks_have_no_live_side_effects(
    update_mobs_lib,
    jax_context,
    rng_stepped_states,
):
    _env, params, static_params = jax_context
    level = 0
    state = _stone_box_state(rng_stepped_states[7], level)
    before_health = float(state.player_health)
    expected, _actual = _assert_update_mobs_matches(
        update_mobs_lib,
        state,
        _rng_words(38001),
        params,
        static_params,
        "empty masks",
    )
    assert float(expected.player_health) == before_health
    assert not bool(np.asarray(expected.mob_map[level]).any())
    assert not bool(np.asarray(expected.melee_mobs.mask[level]).any())
    assert not bool(np.asarray(expected.passive_mobs.mask[level]).any())
    assert not bool(np.asarray(expected.ranged_mobs.mask[level]).any())
    assert not bool(np.asarray(expected.mob_projectiles.mask[level]).any())
    assert not bool(np.asarray(expected.player_projectiles.mask[level]).any())
