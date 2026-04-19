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
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.craftax.world_gen.world_gen import generate_world


ROOT = Path(__file__).resolve().parents[1]
LEVELS = 9
MAP_SIZE = 48
OBS_SIZE = 8268


def build_worldgen_lib():
    source = r"""
    #include <stdbool.h>
    #include <stdint.h>
    #include <string.h>
    #include "ocean/craftax/worldgen.h"

    void world_from_seed(
        uint32_t seed,
        int32_t* map,
        int32_t* item_map,
        bool* mob_map,
        float* light_map,
        int32_t* down_ladders,
        int32_t* up_ladders,
        bool* chests_opened,
        int32_t* monsters_killed,
        int32_t* potion_mapping,
        int32_t* melee_pos,
        float* melee_health,
        bool* melee_mask,
        int32_t* melee_cooldown,
        int32_t* melee_type,
        int32_t* passive_pos,
        float* passive_health,
        bool* passive_mask,
        int32_t* passive_cooldown,
        int32_t* passive_type,
        int32_t* ranged_pos,
        float* ranged_health,
        bool* ranged_mask,
        int32_t* ranged_cooldown,
        int32_t* ranged_type,
        int32_t* mob_projectile_pos,
        float* mob_projectile_health,
        bool* mob_projectile_mask,
        int32_t* mob_projectile_cooldown,
        int32_t* mob_projectile_type,
        int32_t* mob_projectile_directions,
        int32_t* player_projectile_pos,
        float* player_projectile_health,
        bool* player_projectile_mask,
        int32_t* player_projectile_cooldown,
        int32_t* player_projectile_type,
        int32_t* player_projectile_directions,
        int32_t* growing_plants_positions,
        int32_t* growing_plants_age,
        bool* growing_plants_mask,
        int32_t* scalar_i,
        float* scalar_f,
        bool* scalar_b,
        uint32_t* state_rng,
        float* obs
    ) {
        CraftaxWorldState state;
        craftax_generate_world_from_seed(seed, &state);
        memcpy(map, state.map, sizeof(state.map));
        memcpy(item_map, state.item_map, sizeof(state.item_map));
        memcpy(mob_map, state.mob_map, sizeof(state.mob_map));
        memcpy(light_map, state.light_map, sizeof(state.light_map));
        memcpy(down_ladders, state.down_ladders, sizeof(state.down_ladders));
        memcpy(up_ladders, state.up_ladders, sizeof(state.up_ladders));
        memcpy(chests_opened, state.chests_opened, sizeof(state.chests_opened));
        memcpy(monsters_killed, state.monsters_killed, sizeof(state.monsters_killed));
        memcpy(potion_mapping, state.potion_mapping, sizeof(state.potion_mapping));

        memcpy(melee_pos, state.melee_mobs.position, sizeof(state.melee_mobs.position));
        memcpy(melee_health, state.melee_mobs.health, sizeof(state.melee_mobs.health));
        memcpy(melee_mask, state.melee_mobs.mask, sizeof(state.melee_mobs.mask));
        memcpy(melee_cooldown, state.melee_mobs.attack_cooldown, sizeof(state.melee_mobs.attack_cooldown));
        memcpy(melee_type, state.melee_mobs.type_id, sizeof(state.melee_mobs.type_id));

        memcpy(passive_pos, state.passive_mobs.position, sizeof(state.passive_mobs.position));
        memcpy(passive_health, state.passive_mobs.health, sizeof(state.passive_mobs.health));
        memcpy(passive_mask, state.passive_mobs.mask, sizeof(state.passive_mobs.mask));
        memcpy(passive_cooldown, state.passive_mobs.attack_cooldown, sizeof(state.passive_mobs.attack_cooldown));
        memcpy(passive_type, state.passive_mobs.type_id, sizeof(state.passive_mobs.type_id));

        memcpy(ranged_pos, state.ranged_mobs.position, sizeof(state.ranged_mobs.position));
        memcpy(ranged_health, state.ranged_mobs.health, sizeof(state.ranged_mobs.health));
        memcpy(ranged_mask, state.ranged_mobs.mask, sizeof(state.ranged_mobs.mask));
        memcpy(ranged_cooldown, state.ranged_mobs.attack_cooldown, sizeof(state.ranged_mobs.attack_cooldown));
        memcpy(ranged_type, state.ranged_mobs.type_id, sizeof(state.ranged_mobs.type_id));

        memcpy(mob_projectile_pos, state.mob_projectiles.position, sizeof(state.mob_projectiles.position));
        memcpy(mob_projectile_health, state.mob_projectiles.health, sizeof(state.mob_projectiles.health));
        memcpy(mob_projectile_mask, state.mob_projectiles.mask, sizeof(state.mob_projectiles.mask));
        memcpy(mob_projectile_cooldown, state.mob_projectiles.attack_cooldown, sizeof(state.mob_projectiles.attack_cooldown));
        memcpy(mob_projectile_type, state.mob_projectiles.type_id, sizeof(state.mob_projectiles.type_id));
        memcpy(mob_projectile_directions, state.mob_projectile_directions, sizeof(state.mob_projectile_directions));

        memcpy(player_projectile_pos, state.player_projectiles.position, sizeof(state.player_projectiles.position));
        memcpy(player_projectile_health, state.player_projectiles.health, sizeof(state.player_projectiles.health));
        memcpy(player_projectile_mask, state.player_projectiles.mask, sizeof(state.player_projectiles.mask));
        memcpy(player_projectile_cooldown, state.player_projectiles.attack_cooldown, sizeof(state.player_projectiles.attack_cooldown));
        memcpy(player_projectile_type, state.player_projectiles.type_id, sizeof(state.player_projectiles.type_id));
        memcpy(player_projectile_directions, state.player_projectile_directions, sizeof(state.player_projectile_directions));

        memcpy(growing_plants_positions, state.growing_plants_positions, sizeof(state.growing_plants_positions));
        memcpy(growing_plants_age, state.growing_plants_age, sizeof(state.growing_plants_age));
        memcpy(growing_plants_mask, state.growing_plants_mask, sizeof(state.growing_plants_mask));

        scalar_i[0] = state.player_position[0];
        scalar_i[1] = state.player_position[1];
        scalar_i[2] = state.player_level;
        scalar_i[3] = state.player_direction;
        scalar_i[4] = state.player_food;
        scalar_i[5] = state.player_drink;
        scalar_i[6] = state.player_energy;
        scalar_i[7] = state.player_mana;
        scalar_i[8] = state.player_xp;
        scalar_i[9] = state.player_dexterity;
        scalar_i[10] = state.player_strength;
        scalar_i[11] = state.player_intelligence;

        scalar_i[12] = state.inventory.wood;
        scalar_i[13] = state.inventory.stone;
        scalar_i[14] = state.inventory.coal;
        scalar_i[15] = state.inventory.iron;
        scalar_i[16] = state.inventory.diamond;
        scalar_i[17] = state.inventory.sapling;
        scalar_i[18] = state.inventory.pickaxe;
        scalar_i[19] = state.inventory.sword;
        scalar_i[20] = state.inventory.bow;
        scalar_i[21] = state.inventory.arrows;
        scalar_i[22] = state.inventory.armour[0];
        scalar_i[23] = state.inventory.armour[1];
        scalar_i[24] = state.inventory.armour[2];
        scalar_i[25] = state.inventory.armour[3];
        scalar_i[26] = state.inventory.torches;
        scalar_i[27] = state.inventory.ruby;
        scalar_i[28] = state.inventory.sapphire;
        for (int i = 0; i < 6; i++) {
            scalar_i[29 + i] = state.inventory.potions[i];
        }
        scalar_i[35] = state.inventory.books;

        scalar_i[36] = state.sword_enchantment;
        scalar_i[37] = state.bow_enchantment;
        for (int i = 0; i < 4; i++) {
            scalar_i[38 + i] = state.armour_enchantments[i];
        }
        scalar_i[42] = state.boss_progress;
        scalar_i[43] = state.boss_timesteps_to_spawn_this_round;
        scalar_i[44] = state.timestep;

        scalar_f[0] = state.player_health;
        scalar_f[1] = state.player_recover;
        scalar_f[2] = state.player_hunger;
        scalar_f[3] = state.player_thirst;
        scalar_f[4] = state.player_fatigue;
        scalar_f[5] = state.player_recover_mana;
        scalar_f[6] = state.light_level;

        scalar_b[0] = state.is_sleeping;
        scalar_b[1] = state.is_resting;
        scalar_b[2] = state.learned_spells[0];
        scalar_b[3] = state.learned_spells[1];

        state_rng[0] = state.state_rng[0];
        state_rng[1] = state.state_rng[1];

        craftax_encode_reset_observation(&state, obs);
    }

    void reset_key_threshold_edge(
        uint32_t key0,
        uint32_t key1,
        int32_t* map_cell,
        float* grass_obs,
        float* sand_obs
    ) {
        CraftaxThreefryKey reset_key = {{key0, key1}};
        CraftaxThreefryKey unused;
        CraftaxThreefryKey world_key;
        CraftaxWorldState state;
        float obs[CRAFTAX_WG_OBS_SIZE];
        const int channels = CRAFTAX_WG_NUM_BLOCK_TYPES
            + CRAFTAX_WG_NUM_ITEM_TYPES
            + CRAFTAX_WG_NUM_MOB_CLASSES * CRAFTAX_WG_NUM_MOB_TYPES
            + 1;
        const int obs_base = (4 * CRAFTAX_WG_OBS_COLS + 2) * channels;

        craftax_threefry_split(reset_key, &unused, &world_key);
        craftax_generate_world_from_key(world_key, &state);
        craftax_encode_reset_observation(&state, obs);

        *map_cell = state.map[0][24][21];
        *grass_obs = obs[obs_base + CRAFTAX_WG_BLOCK_GRASS];
        *sand_obs = obs[obs_base + CRAFTAX_WG_BLOCK_SAND];
    }
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    src = tmp_path / "worldgen_all_test.c"
    so = tmp_path / "worldgen_all_test.so"
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
    pointer_args = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_bool),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_bool),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    mobs3_args = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_bool),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    mobs2_args = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_bool),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.world_from_seed.argtypes = (
        [ctypes.c_uint32]
        + pointer_args
        + mobs3_args
        + mobs3_args
        + mobs2_args
        + mobs3_args
        + [ctypes.POINTER(ctypes.c_int32)]
        + mobs3_args
        + [ctypes.POINTER(ctypes.c_int32)]
        + [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float),
        ]
    )
    lib.reset_key_threshold_edge.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    return lib


def jax_world(seed):
    rng = jax.random.PRNGKey(seed)
    _rng, reset_key = jax.random.split(rng)
    _rng, world_key = jax.random.split(reset_key)
    state = generate_world(world_key, EnvParams(), StaticEnvParams())
    return state, np.asarray(render_craftax_symbolic(state), dtype=np.float32)


def as_i32(array):
    return np.asarray(array, dtype=np.int32)


def as_f32(array):
    return np.asarray(array, dtype=np.float32)


def as_bool(array):
    return np.asarray(array, dtype=np.bool_)


def scalar_i_from_jax(state):
    inv = state.inventory
    values = [
        state.player_position[0],
        state.player_position[1],
        state.player_level,
        state.player_direction,
        state.player_food,
        state.player_drink,
        state.player_energy,
        state.player_mana,
        state.player_xp,
        state.player_dexterity,
        state.player_strength,
        state.player_intelligence,
        inv.wood,
        inv.stone,
        inv.coal,
        inv.iron,
        inv.diamond,
        inv.sapling,
        inv.pickaxe,
        inv.sword,
        inv.bow,
        inv.arrows,
        inv.armour[0],
        inv.armour[1],
        inv.armour[2],
        inv.armour[3],
        inv.torches,
        inv.ruby,
        inv.sapphire,
        inv.potions[0],
        inv.potions[1],
        inv.potions[2],
        inv.potions[3],
        inv.potions[4],
        inv.potions[5],
        inv.books,
        state.sword_enchantment,
        state.bow_enchantment,
        state.armour_enchantments[0],
        state.armour_enchantments[1],
        state.armour_enchantments[2],
        state.armour_enchantments[3],
        state.boss_progress,
        state.boss_timesteps_to_spawn_this_round,
        state.timestep,
    ]
    return np.asarray([int(np.asarray(v)) for v in values], dtype=np.int32)


def scalar_f_from_jax(state):
    values = [
        state.player_health,
        state.player_recover,
        state.player_hunger,
        state.player_thirst,
        state.player_fatigue,
        state.player_recover_mana,
        state.light_level,
    ]
    return np.asarray([float(np.asarray(v)) for v in values], dtype=np.float32)


def scalar_b_from_jax(state):
    values = [
        state.is_sleeping,
        state.is_resting,
        state.learned_spells[0],
        state.learned_spells[1],
    ]
    return np.asarray([bool(np.asarray(v)) for v in values], dtype=np.bool_)


def c_world(lib, seed):
    arrays = {
        "map": np.empty((LEVELS, MAP_SIZE, MAP_SIZE), dtype=np.int32),
        "item_map": np.empty((LEVELS, MAP_SIZE, MAP_SIZE), dtype=np.int32),
        "mob_map": np.empty((LEVELS, MAP_SIZE, MAP_SIZE), dtype=np.bool_),
        "light_map": np.empty((LEVELS, MAP_SIZE, MAP_SIZE), dtype=np.float32),
        "down_ladders": np.empty((LEVELS, 2), dtype=np.int32),
        "up_ladders": np.empty((LEVELS, 2), dtype=np.int32),
        "chests_opened": np.empty((LEVELS,), dtype=np.bool_),
        "monsters_killed": np.empty((LEVELS,), dtype=np.int32),
        "potion_mapping": np.empty((6,), dtype=np.int32),
        "melee_pos": np.empty((LEVELS, 3, 2), dtype=np.int32),
        "melee_health": np.empty((LEVELS, 3), dtype=np.float32),
        "melee_mask": np.empty((LEVELS, 3), dtype=np.bool_),
        "melee_cooldown": np.empty((LEVELS, 3), dtype=np.int32),
        "melee_type": np.empty((LEVELS, 3), dtype=np.int32),
        "passive_pos": np.empty((LEVELS, 3, 2), dtype=np.int32),
        "passive_health": np.empty((LEVELS, 3), dtype=np.float32),
        "passive_mask": np.empty((LEVELS, 3), dtype=np.bool_),
        "passive_cooldown": np.empty((LEVELS, 3), dtype=np.int32),
        "passive_type": np.empty((LEVELS, 3), dtype=np.int32),
        "ranged_pos": np.empty((LEVELS, 2, 2), dtype=np.int32),
        "ranged_health": np.empty((LEVELS, 2), dtype=np.float32),
        "ranged_mask": np.empty((LEVELS, 2), dtype=np.bool_),
        "ranged_cooldown": np.empty((LEVELS, 2), dtype=np.int32),
        "ranged_type": np.empty((LEVELS, 2), dtype=np.int32),
        "mob_projectile_pos": np.empty((LEVELS, 3, 2), dtype=np.int32),
        "mob_projectile_health": np.empty((LEVELS, 3), dtype=np.float32),
        "mob_projectile_mask": np.empty((LEVELS, 3), dtype=np.bool_),
        "mob_projectile_cooldown": np.empty((LEVELS, 3), dtype=np.int32),
        "mob_projectile_type": np.empty((LEVELS, 3), dtype=np.int32),
        "mob_projectile_directions": np.empty((LEVELS, 3, 2), dtype=np.int32),
        "player_projectile_pos": np.empty((LEVELS, 3, 2), dtype=np.int32),
        "player_projectile_health": np.empty((LEVELS, 3), dtype=np.float32),
        "player_projectile_mask": np.empty((LEVELS, 3), dtype=np.bool_),
        "player_projectile_cooldown": np.empty((LEVELS, 3), dtype=np.int32),
        "player_projectile_type": np.empty((LEVELS, 3), dtype=np.int32),
        "player_projectile_directions": np.empty((LEVELS, 3, 2), dtype=np.int32),
        "growing_plants_positions": np.empty((10, 2), dtype=np.int32),
        "growing_plants_age": np.empty((10,), dtype=np.int32),
        "growing_plants_mask": np.empty((10,), dtype=np.bool_),
        "scalar_i": np.empty((45,), dtype=np.int32),
        "scalar_f": np.empty((7,), dtype=np.float32),
        "scalar_b": np.empty((4,), dtype=np.bool_),
        "state_rng": np.empty((2,), dtype=np.uint32),
        "obs": np.empty((OBS_SIZE,), dtype=np.float32),
    }

    def ptr(name, ctype):
        return arrays[name].ctypes.data_as(ctypes.POINTER(ctype))

    lib.world_from_seed(
        seed,
        ptr("map", ctypes.c_int32),
        ptr("item_map", ctypes.c_int32),
        ptr("mob_map", ctypes.c_bool),
        ptr("light_map", ctypes.c_float),
        ptr("down_ladders", ctypes.c_int32),
        ptr("up_ladders", ctypes.c_int32),
        ptr("chests_opened", ctypes.c_bool),
        ptr("monsters_killed", ctypes.c_int32),
        ptr("potion_mapping", ctypes.c_int32),
        ptr("melee_pos", ctypes.c_int32),
        ptr("melee_health", ctypes.c_float),
        ptr("melee_mask", ctypes.c_bool),
        ptr("melee_cooldown", ctypes.c_int32),
        ptr("melee_type", ctypes.c_int32),
        ptr("passive_pos", ctypes.c_int32),
        ptr("passive_health", ctypes.c_float),
        ptr("passive_mask", ctypes.c_bool),
        ptr("passive_cooldown", ctypes.c_int32),
        ptr("passive_type", ctypes.c_int32),
        ptr("ranged_pos", ctypes.c_int32),
        ptr("ranged_health", ctypes.c_float),
        ptr("ranged_mask", ctypes.c_bool),
        ptr("ranged_cooldown", ctypes.c_int32),
        ptr("ranged_type", ctypes.c_int32),
        ptr("mob_projectile_pos", ctypes.c_int32),
        ptr("mob_projectile_health", ctypes.c_float),
        ptr("mob_projectile_mask", ctypes.c_bool),
        ptr("mob_projectile_cooldown", ctypes.c_int32),
        ptr("mob_projectile_type", ctypes.c_int32),
        ptr("mob_projectile_directions", ctypes.c_int32),
        ptr("player_projectile_pos", ctypes.c_int32),
        ptr("player_projectile_health", ctypes.c_float),
        ptr("player_projectile_mask", ctypes.c_bool),
        ptr("player_projectile_cooldown", ctypes.c_int32),
        ptr("player_projectile_type", ctypes.c_int32),
        ptr("player_projectile_directions", ctypes.c_int32),
        ptr("growing_plants_positions", ctypes.c_int32),
        ptr("growing_plants_age", ctypes.c_int32),
        ptr("growing_plants_mask", ctypes.c_bool),
        ptr("scalar_i", ctypes.c_int32),
        ptr("scalar_f", ctypes.c_float),
        ptr("scalar_b", ctypes.c_bool),
        ptr("state_rng", ctypes.c_uint32),
        ptr("obs", ctypes.c_float),
    )
    return arrays


def assert_mobs_equal(got, state, prefix, mobs):
    np.testing.assert_array_equal(got[f"{prefix}_pos"], as_i32(mobs.position))
    np.testing.assert_allclose(got[f"{prefix}_health"], as_f32(mobs.health), atol=1e-6, rtol=0.0)
    np.testing.assert_array_equal(got[f"{prefix}_mask"], as_bool(mobs.mask))
    np.testing.assert_array_equal(got[f"{prefix}_cooldown"], as_i32(mobs.attack_cooldown))
    np.testing.assert_array_equal(got[f"{prefix}_type"], as_i32(mobs.type_id))


def test_native_worldgen_matches_jax_for_all_reset_state():
    lib = build_worldgen_lib()
    for seed in range(16):
        state, expected_obs = jax_world(seed)
        got = c_world(lib, seed)

        np.testing.assert_array_equal(got["map"], as_i32(state.map), err_msg=f"map seed={seed}")
        np.testing.assert_array_equal(
            got["item_map"],
            as_i32(state.item_map),
            err_msg=f"item_map seed={seed}",
        )
        np.testing.assert_array_equal(
            got["mob_map"],
            as_bool(state.mob_map),
            err_msg=f"mob_map seed={seed}",
        )
        np.testing.assert_allclose(
            got["light_map"],
            as_f32(state.light_map),
            atol=1e-6,
            rtol=0.0,
            err_msg=f"light_map seed={seed}",
        )
        np.testing.assert_array_equal(
            got["down_ladders"],
            as_i32(state.down_ladders),
            err_msg=f"down_ladders seed={seed}",
        )
        np.testing.assert_array_equal(
            got["up_ladders"],
            as_i32(state.up_ladders),
            err_msg=f"up_ladders seed={seed}",
        )
        np.testing.assert_array_equal(
            got["chests_opened"],
            as_bool(state.chests_opened),
            err_msg=f"chests_opened seed={seed}",
        )
        np.testing.assert_array_equal(
            got["monsters_killed"],
            as_i32(state.monsters_killed),
            err_msg=f"monsters_killed seed={seed}",
        )
        assert got["monsters_killed"][0] == 10
        assert not got["chests_opened"].any()

        assert_mobs_equal(got, state, "melee", state.melee_mobs)
        assert_mobs_equal(got, state, "passive", state.passive_mobs)
        assert_mobs_equal(got, state, "ranged", state.ranged_mobs)
        assert_mobs_equal(got, state, "mob_projectile", state.mob_projectiles)
        assert_mobs_equal(got, state, "player_projectile", state.player_projectiles)
        np.testing.assert_array_equal(
            got["mob_projectile_directions"],
            as_i32(state.mob_projectile_directions),
            err_msg=f"mob_projectile_directions seed={seed}",
        )
        np.testing.assert_array_equal(
            got["player_projectile_directions"],
            as_i32(state.player_projectile_directions),
            err_msg=f"player_projectile_directions seed={seed}",
        )
        np.testing.assert_array_equal(
            got["growing_plants_positions"],
            as_i32(state.growing_plants_positions),
            err_msg=f"growing_plants_positions seed={seed}",
        )
        np.testing.assert_array_equal(
            got["growing_plants_age"],
            as_i32(state.growing_plants_age),
            err_msg=f"growing_plants_age seed={seed}",
        )
        np.testing.assert_array_equal(
            got["growing_plants_mask"],
            as_bool(state.growing_plants_mask),
            err_msg=f"growing_plants_mask seed={seed}",
        )

        np.testing.assert_array_equal(
            got["potion_mapping"],
            as_i32(state.potion_mapping),
            err_msg=f"potion_mapping seed={seed}",
        )
        np.testing.assert_array_equal(
            got["scalar_i"],
            scalar_i_from_jax(state),
            err_msg=f"scalar_i seed={seed}",
        )
        np.testing.assert_allclose(
            got["scalar_f"],
            scalar_f_from_jax(state),
            atol=1e-6,
            rtol=0.0,
            err_msg=f"scalar_f seed={seed}",
        )
        np.testing.assert_array_equal(
            got["scalar_b"],
            scalar_b_from_jax(state),
            err_msg=f"scalar_b seed={seed}",
        )
        np.testing.assert_array_equal(
            got["state_rng"],
            np.asarray(state.state_rng, dtype=np.uint32),
            err_msg=f"state_rng seed={seed}",
        )
        np.testing.assert_allclose(
            got["obs"],
            expected_obs,
            atol=1e-6,
            rtol=0.0,
            err_msg=f"obs seed={seed}",
        )


def test_native_reset_key_matches_materialized_jax_at_sand_threshold_edge():
    lib = build_worldgen_lib()
    reset_keys = [
        np.asarray([616102339, 1559696082], dtype=np.uint32),
        np.asarray([934346395, 1048685838], dtype=np.uint32),
    ]

    for reset_key in reset_keys:
        _unused, world_key = jax.random.split(reset_key)
        expected_state = generate_world(world_key, EnvParams(), StaticEnvParams())
        expected_obs = np.asarray(
            render_craftax_symbolic(expected_state),
            dtype=np.float32,
        )

        map_cell = ctypes.c_int32()
        grass_obs = ctypes.c_float()
        sand_obs = ctypes.c_float()
        lib.reset_key_threshold_edge(
            int(reset_key[0]),
            int(reset_key[1]),
            ctypes.byref(map_cell),
            ctypes.byref(grass_obs),
            ctypes.byref(sand_obs),
        )

        assert int(np.asarray(expected_state.map[0, 24, 21])) == 2
        assert map_cell.value == 2
        assert grass_obs.value == expected_obs[((4 * 11 + 2) * 83) + 2] == 1.0
        assert sand_obs.value == expected_obs[((4 * 11 + 2) * 83) + 13] == 0.0
