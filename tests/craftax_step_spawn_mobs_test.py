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
from craftax.craftax.game_logic import spawn_mobs
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
PASSIVE_CANDIDATE = (24, 28)
MONSTER_CANDIDATE = (24, 34)
BOSS_CANDIDATE = (24, 30)


@pytest.fixture(scope="session")
def spawn_mobs_lib():
    source = r"""
    #include <stdbool.h>
    #include <stdint.h>
    #include <stddef.h>
    #include "ocean/craftax/step_spawn_mobs.h"

    size_t craftax_test_state_size(void) {
        return sizeof(CraftaxState);
    }

    void run_spawn_mobs(CraftaxState* state, uint32_t rng0, uint32_t rng1) {
        CraftaxThreefryKey rng = {{rng0, rng1}};
        craftax_spawn_mobs_native(state, rng);
    }
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    src = tmp_path / "craftax_step_spawn_mobs_test.c"
    so = tmp_path / "craftax_step_spawn_mobs_test.so"
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

    lib.run_spawn_mobs.argtypes = [
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
def noop_stepped_states(jax_context):
    env, params, _static_params = jax_context
    states = {}
    for seed in SEEDS:
        rng = jax.random.PRNGKey(seed)
        rng, reset_key = jax.random.split(rng)
        _obs, state = env.reset(reset_key, params)
        snapshots = [state]
        for _step in range(3):
            rng, step_key = jax.random.split(rng)
            _obs, state, _reward, _done, _info = env.step(
                step_key,
                state,
                Action.NOOP.value,
                params,
            )
            snapshots.append(state)
        states[seed] = snapshots
    return states


def _assert_native_matches(state, expected, run_native, context):
    c_state = jax_state_to_c_state(state)
    run_native(c_state)
    actual = craftax_state_to_jax(c_state, template=state)
    assert_env_states_equal(actual, expected, context)


def _assert_spawn_matches(
    spawn_mobs_lib,
    state,
    rng_words,
    params,
    static_params,
    context,
):
    rng = jnp.asarray(rng_words, dtype=jnp.uint32)
    expected = spawn_mobs(state, rng, params, static_params)
    _assert_native_matches(
        state,
        expected,
        lambda c_state: spawn_mobs_lib.run_spawn_mobs(
            ctypes.byref(c_state),
            int(rng_words[0]),
            int(rng_words[1]),
        ),
        context,
    )
    return expected


def _rng_words(seed):
    return np.asarray(jax.random.PRNGKey(seed), dtype=np.uint32)


def _folded_state_rng_words(state):
    return np.asarray(
        jax.random.fold_in(state.state_rng, int(state.timestep)),
        dtype=np.uint32,
    )


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
        player_projectiles=_empty_mobs(state.player_projectiles),
    )


def _base_spawn_state(
    state,
    level,
    position=(24, 24),
    fill_block=BlockType.STONE.value,
    light_level=1.0,
):
    floor = jnp.full((MAP_SIZE, MAP_SIZE), fill_block, dtype=jnp.int32)
    state = _clear_mobs(state)
    return state.replace(
        map=state.map.at[level].set(floor),
        player_level=int(level),
        player_position=jnp.asarray(position, dtype=jnp.int32),
        player_direction=Action.UP.value,
        monsters_killed=state.monsters_killed.at[level].set(0),
        boss_timesteps_to_spawn_this_round=0,
        boss_progress=0,
        light_level=np.float32(light_level),
    )


def _set_cell(state, level, position, block):
    row, col = position
    return state.replace(
        map=state.map.at[level, int(row), int(col)].set(int(block))
    )


def _set_cells(state, level, positions, block):
    for position in positions:
        state = _set_cell(state, level, position, block)
    return state


def _cap_positions(mob_class):
    positions = {
        "passive": ((5, 5), (5, 6), (5, 7)),
        "melee": ((6, 5), (6, 6), (6, 7)),
        "ranged": ((7, 5), (7, 6)),
    }[mob_class]
    return jnp.asarray(positions, dtype=jnp.int32)


def _set_cap_for_class(state, level, mob_class):
    positions = _cap_positions(mob_class)
    if mob_class == "passive":
        mobs = state.passive_mobs.replace(
            position=state.passive_mobs.position.at[level].set(positions),
            health=state.passive_mobs.health.at[level].set(
                jnp.asarray([3.0, 4.0, 5.0], dtype=jnp.float32)
            ),
            mask=state.passive_mobs.mask.at[level].set(
                jnp.ones((3,), dtype=bool)
            ),
            type_id=state.passive_mobs.type_id.at[level].set(
                jnp.full((3,), 7, dtype=jnp.int32)
            ),
        )
        state = state.replace(passive_mobs=mobs)
    elif mob_class == "melee":
        mobs = state.melee_mobs.replace(
            position=state.melee_mobs.position.at[level].set(positions),
            health=state.melee_mobs.health.at[level].set(
                jnp.asarray([6.0, 7.0, 8.0], dtype=jnp.float32)
            ),
            mask=state.melee_mobs.mask.at[level].set(
                jnp.ones((3,), dtype=bool)
            ),
            type_id=state.melee_mobs.type_id.at[level].set(
                jnp.full((3,), 7, dtype=jnp.int32)
            ),
        )
        state = state.replace(melee_mobs=mobs)
    elif mob_class == "ranged":
        mobs = state.ranged_mobs.replace(
            position=state.ranged_mobs.position.at[level].set(positions),
            health=state.ranged_mobs.health.at[level].set(
                jnp.asarray([9.0, 10.0], dtype=jnp.float32)
            ),
            mask=state.ranged_mobs.mask.at[level].set(
                jnp.ones((2,), dtype=bool)
            ),
            type_id=state.ranged_mobs.type_id.at[level].set(
                jnp.full((2,), 7, dtype=jnp.int32)
            ),
        )
        state = state.replace(ranged_mobs=mobs)
    else:
        raise ValueError(mob_class)

    for row, col in np.asarray(positions):
        state = state.replace(
            mob_map=state.mob_map.at[level, int(row), int(col)].set(True)
        )
    return state


def _with_caps(state, level, passive=False, melee=False, ranged=False):
    if passive:
        state = _set_cap_for_class(state, level, "passive")
    if melee:
        state = _set_cap_for_class(state, level, "melee")
    if ranged:
        state = _set_cap_for_class(state, level, "ranged")
    return state


def _mob_count(state, mob_class, level):
    mobs = getattr(state, f"{mob_class}_mobs")
    return int(np.asarray(mobs.mask[level]).sum())


def _mob_position(state, mob_class, level, slot=0):
    mobs = getattr(state, f"{mob_class}_mobs")
    return tuple(np.asarray(mobs.position[level, slot], dtype=np.int32))


def _find_rng(
    state,
    params,
    static_params,
    predicate,
    start_seed=0,
    limit=5000,
):
    for seed in range(start_seed, start_seed + limit):
        rng_words = _rng_words(seed)
        expected = spawn_mobs(
            state,
            jnp.asarray(rng_words, dtype=jnp.uint32),
            params,
            static_params,
        )
        if predicate(expected):
            return rng_words
    raise AssertionError("could not find spawn_mobs rng for targeted case")


def _single_candidate_state(state, level, mob_class, block, position):
    candidate = tuple(position)
    state = _base_spawn_state(state, level)
    state = _set_cell(state, level, candidate, block)
    if mob_class == "passive":
        return _with_caps(state, level, melee=True, ranged=True)
    if mob_class == "melee":
        return _with_caps(state, level, passive=True, ranged=True)
    if mob_class == "ranged":
        return _with_caps(state, level, passive=True, melee=True)
    raise ValueError(mob_class)


def test_spawn_mobs_native_parity_on_noop_stepped_states(
    spawn_mobs_lib,
    jax_context,
    noop_stepped_states,
):
    _env, params, static_params = jax_context
    for seed, states in noop_stepped_states.items():
        for step, state in enumerate(states):
            rng_words = _folded_state_rng_words(state)
            _assert_spawn_matches(
                spawn_mobs_lib,
                state,
                rng_words,
                params,
                static_params,
                f"noop stepped seed={seed} step={step}",
            )


@pytest.mark.parametrize(
    ("mob_class", "level", "block", "candidate"),
    [
        ("passive", 0, BlockType.GRASS.value, PASSIVE_CANDIDATE),
        ("melee", 0, BlockType.GRASS.value, MONSTER_CANDIDATE),
        ("ranged", 5, BlockType.WATER.value, MONSTER_CANDIDATE),
    ],
)
def test_spawn_mobs_empty_slots_spawn_at_single_candidate(
    spawn_mobs_lib,
    jax_context,
    noop_stepped_states,
    mob_class,
    level,
    block,
    candidate,
):
    _env, params, static_params = jax_context
    base = noop_stepped_states[0][0]
    state = _single_candidate_state(base, level, mob_class, block, candidate)
    rng_words = _find_rng(
        state,
        params,
        static_params,
        lambda expected: _mob_count(expected, mob_class, level) == 1,
        start_seed=1000 + level * 100,
    )

    expected = _assert_spawn_matches(
        spawn_mobs_lib,
        state,
        rng_words,
        params,
        static_params,
        f"single candidate {mob_class} level={level}",
    )
    assert _mob_count(expected, mob_class, level) == 1
    assert _mob_position(expected, mob_class, level) == tuple(candidate)


def test_spawn_mobs_full_caps_do_not_add_mobs(
    spawn_mobs_lib,
    jax_context,
    noop_stepped_states,
):
    _env, params, static_params = jax_context
    level = 0
    state = _base_spawn_state(noop_stepped_states[1][0], level)
    state = _set_cells(
        state,
        level,
        [PASSIVE_CANDIDATE, MONSTER_CANDIDATE, (24, 35)],
        BlockType.GRASS.value,
    )
    state = _with_caps(state, level, passive=True, melee=True, ranged=True)

    expected = _assert_spawn_matches(
        spawn_mobs_lib,
        state,
        _rng_words(231),
        params,
        static_params,
        "full mob caps",
    )
    assert _mob_count(expected, "passive", level) == 3
    assert _mob_count(expected, "melee", level) == 3
    assert _mob_count(expected, "ranged", level) == 2


@pytest.mark.parametrize(
    ("level", "block", "candidate"),
    [
        (0, BlockType.GRASS.value, MONSTER_CANDIDATE),
        (1, BlockType.PATH.value, MONSTER_CANDIDATE),
        (2, BlockType.GRASS.value, MONSTER_CANDIDATE),
        (3, BlockType.PATH.value, MONSTER_CANDIDATE),
        (4, BlockType.GRASS.value, MONSTER_CANDIDATE),
        (5, BlockType.GRASS.value, MONSTER_CANDIDATE),
        (6, BlockType.FIRE_GRASS.value, MONSTER_CANDIDATE),
        (7, BlockType.ICE_GRASS.value, MONSTER_CANDIDATE),
        (8, BlockType.GRAVE.value, BOSS_CANDIDATE),
    ],
)
def test_spawn_mobs_each_floor_melee_spawn_constraints(
    spawn_mobs_lib,
    jax_context,
    noop_stepped_states,
    level,
    block,
    candidate,
):
    _env, params, static_params = jax_context
    state = _base_spawn_state(noop_stepped_states[level % len(SEEDS)][0], level)
    state = _set_cell(state, level, candidate, block)
    state = _with_caps(state, level, passive=True, ranged=True)
    if level == 8:
        state = state.replace(
            boss_timesteps_to_spawn_this_round=3,
            boss_progress=4,
        )

    rng_words = _find_rng(
        state,
        params,
        static_params,
        lambda expected: _mob_count(expected, "melee", level) == 1,
        start_seed=2000 + level * 100,
    )
    expected = _assert_spawn_matches(
        spawn_mobs_lib,
        state,
        rng_words,
        params,
        static_params,
        f"floor melee level={level}",
    )
    assert _mob_count(expected, "melee", level) == 1
    assert _mob_position(expected, "melee", level) == tuple(candidate)


def test_spawn_mobs_night_light_adds_overworld_melee_chance(
    spawn_mobs_lib,
    jax_context,
    noop_stepped_states,
):
    _env, params, static_params = jax_context
    level = 0
    day_state = _single_candidate_state(
        noop_stepped_states[2][0],
        level,
        "melee",
        BlockType.GRASS.value,
        MONSTER_CANDIDATE,
    ).replace(light_level=np.float32(1.0))
    night_state = day_state.replace(light_level=np.float32(0.0))

    for seed in range(3000, 8000):
        candidate_rng = _rng_words(seed)
        night_expected = spawn_mobs(
            night_state,
            jnp.asarray(candidate_rng, dtype=jnp.uint32),
            params,
            static_params,
        )
        day_expected = spawn_mobs(
            day_state,
            jnp.asarray(candidate_rng, dtype=jnp.uint32),
            params,
            static_params,
        )
        if (
            _mob_count(night_expected, "melee", level) == 1
            and _mob_count(day_expected, "melee", level) == 0
        ):
            rng_words = candidate_rng
            break
    else:
        raise AssertionError("could not find day/night split rng")

    day_expected = _assert_spawn_matches(
        spawn_mobs_lib,
        day_state,
        rng_words,
        params,
        static_params,
        "overworld day melee chance",
    )
    night_expected = _assert_spawn_matches(
        spawn_mobs_lib,
        night_state,
        rng_words,
        params,
        static_params,
        "overworld night melee chance",
    )
    assert _mob_count(day_expected, "melee", level) == 0
    assert _mob_count(night_expected, "melee", level) == 1


def test_spawn_mobs_boss_floor_pacing_uses_spawn_wave(
    spawn_mobs_lib,
    jax_context,
    noop_stepped_states,
):
    _env, params, static_params = jax_context
    level = 8
    wave_state = _base_spawn_state(noop_stepped_states[3][0], level)
    wave_state = _set_cell(wave_state, level, BOSS_CANDIDATE, BlockType.GRAVE.value)
    wave_state = wave_state.replace(
        boss_progress=2,
        boss_timesteps_to_spawn_this_round=2,
    )
    cooldown_state = wave_state.replace(boss_timesteps_to_spawn_this_round=0)
    rng_words = _rng_words(41)

    wave_expected = _assert_spawn_matches(
        spawn_mobs_lib,
        wave_state,
        rng_words,
        params,
        static_params,
        "boss spawn wave",
    )
    cooldown_expected = _assert_spawn_matches(
        spawn_mobs_lib,
        cooldown_state,
        rng_words,
        params,
        static_params,
        "boss cooldown no spawn",
    )
    assert _mob_count(wave_expected, "melee", level) == 1
    assert _mob_position(wave_expected, "melee", level) == BOSS_CANDIDATE
    assert _mob_count(cooldown_expected, "melee", level) == 0
    assert _mob_count(cooldown_expected, "ranged", level) == 0


def test_spawn_mobs_rejects_only_player_adjacent_candidates(
    spawn_mobs_lib,
    jax_context,
    noop_stepped_states,
):
    _env, params, static_params = jax_context
    level = 0
    adjacent_positions = [
        (24, 23),
        (24, 25),
        (23, 24),
        (25, 24),
        (23, 23),
        (23, 25),
        (25, 23),
        (25, 25),
    ]
    state = _base_spawn_state(noop_stepped_states[4][0], level)
    state = _set_cells(state, level, adjacent_positions, BlockType.GRASS.value)

    expected = _assert_spawn_matches(
        spawn_mobs_lib,
        state,
        _rng_words(99),
        params,
        static_params,
        "adjacent candidates rejected",
    )
    assert _mob_count(expected, "passive", level) == 0
    assert _mob_count(expected, "melee", level) == 0
    assert _mob_count(expected, "ranged", level) == 0


@pytest.mark.parametrize(
    ("name", "level", "mob_class", "allowed_block", "rejected_block"),
    [
        (
            "land_rejects_water",
            0,
            "melee",
            BlockType.GRASS.value,
            BlockType.WATER.value,
        ),
        (
            "deep_thing_requires_water",
            5,
            "ranged",
            BlockType.WATER.value,
            BlockType.GRASS.value,
        ),
        (
            "boss_requires_grave",
            8,
            "melee",
            BlockType.GRAVE.value,
            BlockType.GRASS.value,
        ),
    ],
)
def test_spawn_mobs_collision_style_terrain_constraints(
    spawn_mobs_lib,
    jax_context,
    noop_stepped_states,
    name,
    level,
    mob_class,
    allowed_block,
    rejected_block,
):
    _env, params, static_params = jax_context
    candidate = BOSS_CANDIDATE if level == 8 else MONSTER_CANDIDATE
    allowed_state = _single_candidate_state(
        noop_stepped_states[5][0],
        level,
        mob_class,
        allowed_block,
        candidate,
    )
    rejected_state = _single_candidate_state(
        noop_stepped_states[5][0],
        level,
        mob_class,
        rejected_block,
        candidate,
    )
    if level == 8:
        allowed_state = allowed_state.replace(
            boss_progress=3,
            boss_timesteps_to_spawn_this_round=2,
        )
        rejected_state = rejected_state.replace(
            boss_progress=3,
            boss_timesteps_to_spawn_this_round=2,
        )

    rng_words = _find_rng(
        allowed_state,
        params,
        static_params,
        lambda expected: _mob_count(expected, mob_class, level) == 1,
        start_seed=5000 + level * 100,
    )
    allowed_expected = _assert_spawn_matches(
        spawn_mobs_lib,
        allowed_state,
        rng_words,
        params,
        static_params,
        f"{name} allowed",
    )
    rejected_expected = _assert_spawn_matches(
        spawn_mobs_lib,
        rejected_state,
        rng_words,
        params,
        static_params,
        f"{name} rejected",
    )
    assert _mob_count(allowed_expected, mob_class, level) == 1
    assert _mob_count(rejected_expected, mob_class, level) == 0
