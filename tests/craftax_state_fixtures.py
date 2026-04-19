import ctypes
import os
import pickle

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np

from craftax.craftax.craftax_state import EnvState, Inventory, Mobs


LEVELS = 9
MAP_SIZE = 48
ACHIEVEMENTS = 67
MAX_MELEE_MOBS = 3
MAX_PASSIVE_MOBS = 3
MAX_RANGED_MOBS = 2
MAX_MOB_PROJECTILES = 3
MAX_PLAYER_PROJECTILES = 3
MAX_GROWING_PLANTS = 10


def _c_array(ctype, *shape):
    array_type = ctype
    for size in reversed(shape):
        array_type = array_type * size
    return array_type


class CraftaxInventory(ctypes.Structure):
    _fields_ = [
        ("wood", ctypes.c_int32),
        ("stone", ctypes.c_int32),
        ("coal", ctypes.c_int32),
        ("iron", ctypes.c_int32),
        ("diamond", ctypes.c_int32),
        ("sapling", ctypes.c_int32),
        ("pickaxe", ctypes.c_int32),
        ("sword", ctypes.c_int32),
        ("bow", ctypes.c_int32),
        ("arrows", ctypes.c_int32),
        ("armour", _c_array(ctypes.c_int32, 4)),
        ("torches", ctypes.c_int32),
        ("ruby", ctypes.c_int32),
        ("sapphire", ctypes.c_int32),
        ("potions", _c_array(ctypes.c_int32, 6)),
        ("books", ctypes.c_int32),
    ]


class CraftaxMobs3(ctypes.Structure):
    _fields_ = [
        ("position", _c_array(ctypes.c_int32, LEVELS, 3, 2)),
        ("health", _c_array(ctypes.c_float, LEVELS, 3)),
        ("mask", _c_array(ctypes.c_bool, LEVELS, 3)),
        ("attack_cooldown", _c_array(ctypes.c_int32, LEVELS, 3)),
        ("type_id", _c_array(ctypes.c_int32, LEVELS, 3)),
    ]


class CraftaxMobs2(ctypes.Structure):
    _fields_ = [
        ("position", _c_array(ctypes.c_int32, LEVELS, 2, 2)),
        ("health", _c_array(ctypes.c_float, LEVELS, 2)),
        ("mask", _c_array(ctypes.c_bool, LEVELS, 2)),
        ("attack_cooldown", _c_array(ctypes.c_int32, LEVELS, 2)),
        ("type_id", _c_array(ctypes.c_int32, LEVELS, 2)),
    ]


class CraftaxState(ctypes.Structure):
    _fields_ = [
        ("map", _c_array(ctypes.c_int32, LEVELS, MAP_SIZE, MAP_SIZE)),
        ("item_map", _c_array(ctypes.c_int32, LEVELS, MAP_SIZE, MAP_SIZE)),
        ("mob_map", _c_array(ctypes.c_bool, LEVELS, MAP_SIZE, MAP_SIZE)),
        ("light_map", _c_array(ctypes.c_float, LEVELS, MAP_SIZE, MAP_SIZE)),
        ("down_ladders", _c_array(ctypes.c_int32, LEVELS, 2)),
        ("up_ladders", _c_array(ctypes.c_int32, LEVELS, 2)),
        ("chests_opened", _c_array(ctypes.c_bool, LEVELS)),
        ("monsters_killed", _c_array(ctypes.c_int32, LEVELS)),
        ("player_position", _c_array(ctypes.c_int32, 2)),
        ("player_level", ctypes.c_int32),
        ("player_direction", ctypes.c_int32),
        ("player_health", ctypes.c_float),
        ("player_food", ctypes.c_int32),
        ("player_drink", ctypes.c_int32),
        ("player_energy", ctypes.c_int32),
        ("player_mana", ctypes.c_int32),
        ("is_sleeping", ctypes.c_bool),
        ("is_resting", ctypes.c_bool),
        ("player_recover", ctypes.c_float),
        ("player_hunger", ctypes.c_float),
        ("player_thirst", ctypes.c_float),
        ("player_fatigue", ctypes.c_float),
        ("player_recover_mana", ctypes.c_float),
        ("player_xp", ctypes.c_int32),
        ("player_dexterity", ctypes.c_int32),
        ("player_strength", ctypes.c_int32),
        ("player_intelligence", ctypes.c_int32),
        ("inventory", CraftaxInventory),
        ("melee_mobs", CraftaxMobs3),
        ("passive_mobs", CraftaxMobs3),
        ("ranged_mobs", CraftaxMobs2),
        ("mob_projectiles", CraftaxMobs3),
        (
            "mob_projectile_directions",
            _c_array(ctypes.c_int32, LEVELS, MAX_MOB_PROJECTILES, 2),
        ),
        ("player_projectiles", CraftaxMobs3),
        (
            "player_projectile_directions",
            _c_array(ctypes.c_int32, LEVELS, MAX_PLAYER_PROJECTILES, 2),
        ),
        (
            "growing_plants_positions",
            _c_array(ctypes.c_int32, MAX_GROWING_PLANTS, 2),
        ),
        ("growing_plants_age", _c_array(ctypes.c_int32, MAX_GROWING_PLANTS)),
        ("growing_plants_mask", _c_array(ctypes.c_bool, MAX_GROWING_PLANTS)),
        ("potion_mapping", _c_array(ctypes.c_int32, 6)),
        ("learned_spells", _c_array(ctypes.c_bool, 2)),
        ("sword_enchantment", ctypes.c_int32),
        ("bow_enchantment", ctypes.c_int32),
        ("armour_enchantments", _c_array(ctypes.c_int32, 4)),
        ("boss_progress", ctypes.c_int32),
        ("boss_timesteps_to_spawn_this_round", ctypes.c_int32),
        ("light_level", ctypes.c_float),
        ("achievements", _c_array(ctypes.c_bool, ACHIEVEMENTS)),
        ("state_rng", _c_array(ctypes.c_uint32, 2)),
        ("timestep", ctypes.c_int32),
        ("fractal_noise_angles", _c_array(ctypes.c_int32, 4)),
    ]


def _np_array(value, dtype):
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _copy_to_c(c_array, value, dtype, shape):
    array = _np_array(value, dtype)
    if array.shape != shape:
        raise ValueError(f"shape mismatch: got {array.shape}, expected {shape}")
    ctypes.memmove(ctypes.addressof(c_array), array.ctypes.data, array.nbytes)


def _copy_from_c(c_array, dtype):
    return np.asarray(np.ctypeslib.as_array(c_array), dtype=dtype).copy()


def _mobs_payload(mobs):
    return {
        "position": _np_array(mobs.position, np.int32),
        "health": _np_array(mobs.health, np.float32),
        "mask": _np_array(mobs.mask, np.bool_),
        "attack_cooldown": _np_array(mobs.attack_cooldown, np.int32),
        "type_id": _np_array(mobs.type_id, np.int32),
    }


def _inventory_payload(inventory):
    return {
        "wood": int(inventory.wood),
        "stone": int(inventory.stone),
        "coal": int(inventory.coal),
        "iron": int(inventory.iron),
        "diamond": int(inventory.diamond),
        "sapling": int(inventory.sapling),
        "pickaxe": int(inventory.pickaxe),
        "sword": int(inventory.sword),
        "bow": int(inventory.bow),
        "arrows": int(inventory.arrows),
        "armour": _np_array(inventory.armour, np.int32),
        "torches": int(inventory.torches),
        "ruby": int(inventory.ruby),
        "sapphire": int(inventory.sapphire),
        "potions": _np_array(inventory.potions, np.int32),
        "books": int(inventory.books),
    }


def _fractal_payload(state):
    values = []
    for value in state.fractal_noise_angles:
        values.append(0 if value is None else int(value))
    return np.asarray(values, dtype=np.int32)


def serialize_jax_state(state: EnvState) -> bytes:
    payload = {
        "map": _np_array(state.map, np.int32),
        "item_map": _np_array(state.item_map, np.int32),
        "mob_map": _np_array(state.mob_map, np.bool_),
        "light_map": _np_array(state.light_map, np.float32),
        "down_ladders": _np_array(state.down_ladders, np.int32),
        "up_ladders": _np_array(state.up_ladders, np.int32),
        "chests_opened": _np_array(state.chests_opened, np.bool_),
        "monsters_killed": _np_array(state.monsters_killed, np.int32),
        "player_position": _np_array(state.player_position, np.int32),
        "player_level": int(state.player_level),
        "player_direction": int(state.player_direction),
        "player_health": float(state.player_health),
        "player_food": int(state.player_food),
        "player_drink": int(state.player_drink),
        "player_energy": int(state.player_energy),
        "player_mana": int(state.player_mana),
        "is_sleeping": bool(state.is_sleeping),
        "is_resting": bool(state.is_resting),
        "player_recover": float(state.player_recover),
        "player_hunger": float(state.player_hunger),
        "player_thirst": float(state.player_thirst),
        "player_fatigue": float(state.player_fatigue),
        "player_recover_mana": float(state.player_recover_mana),
        "player_xp": int(state.player_xp),
        "player_dexterity": int(state.player_dexterity),
        "player_strength": int(state.player_strength),
        "player_intelligence": int(state.player_intelligence),
        "inventory": _inventory_payload(state.inventory),
        "melee_mobs": _mobs_payload(state.melee_mobs),
        "passive_mobs": _mobs_payload(state.passive_mobs),
        "ranged_mobs": _mobs_payload(state.ranged_mobs),
        "mob_projectiles": _mobs_payload(state.mob_projectiles),
        "mob_projectile_directions": _np_array(
            state.mob_projectile_directions, np.int32
        ),
        "player_projectiles": _mobs_payload(state.player_projectiles),
        "player_projectile_directions": _np_array(
            state.player_projectile_directions, np.int32
        ),
        "growing_plants_positions": _np_array(
            state.growing_plants_positions, np.int32
        ),
        "growing_plants_age": _np_array(state.growing_plants_age, np.int32),
        "growing_plants_mask": _np_array(state.growing_plants_mask, np.bool_),
        "potion_mapping": _np_array(state.potion_mapping, np.int32),
        "learned_spells": _np_array(state.learned_spells, np.bool_),
        "sword_enchantment": int(state.sword_enchantment),
        "bow_enchantment": int(state.bow_enchantment),
        "armour_enchantments": _np_array(state.armour_enchantments, np.int32),
        "boss_progress": int(state.boss_progress),
        "boss_timesteps_to_spawn_this_round": int(
            state.boss_timesteps_to_spawn_this_round
        ),
        "light_level": float(state.light_level),
        "achievements": _np_array(state.achievements, np.bool_),
        "state_rng": _np_array(state.state_rng, np.uint32),
        "timestep": int(state.timestep),
        "fractal_noise_angles": _fractal_payload(state),
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _copy_inventory_to_c(c_inventory, payload):
    for name in [
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "sapling",
        "pickaxe",
        "sword",
        "bow",
        "arrows",
        "torches",
        "ruby",
        "sapphire",
        "books",
    ]:
        setattr(c_inventory, name, int(payload[name]))
    _copy_to_c(c_inventory.armour, payload["armour"], np.int32, (4,))
    _copy_to_c(c_inventory.potions, payload["potions"], np.int32, (6,))


def _copy_mobs_to_c(c_mobs, payload, max_mobs):
    _copy_to_c(c_mobs.position, payload["position"], np.int32, (LEVELS, max_mobs, 2))
    _copy_to_c(c_mobs.health, payload["health"], np.float32, (LEVELS, max_mobs))
    _copy_to_c(c_mobs.mask, payload["mask"], np.bool_, (LEVELS, max_mobs))
    _copy_to_c(
        c_mobs.attack_cooldown,
        payload["attack_cooldown"],
        np.int32,
        (LEVELS, max_mobs),
    )
    _copy_to_c(c_mobs.type_id, payload["type_id"], np.int32, (LEVELS, max_mobs))


def deserialize_jax_state_to_c(buffer: bytes) -> CraftaxState:
    payload = pickle.loads(buffer)
    state = CraftaxState()

    _copy_to_c(state.map, payload["map"], np.int32, (LEVELS, MAP_SIZE, MAP_SIZE))
    _copy_to_c(
        state.item_map, payload["item_map"], np.int32, (LEVELS, MAP_SIZE, MAP_SIZE)
    )
    _copy_to_c(
        state.mob_map, payload["mob_map"], np.bool_, (LEVELS, MAP_SIZE, MAP_SIZE)
    )
    _copy_to_c(
        state.light_map, payload["light_map"], np.float32, (LEVELS, MAP_SIZE, MAP_SIZE)
    )
    _copy_to_c(state.down_ladders, payload["down_ladders"], np.int32, (LEVELS, 2))
    _copy_to_c(state.up_ladders, payload["up_ladders"], np.int32, (LEVELS, 2))
    _copy_to_c(state.chests_opened, payload["chests_opened"], np.bool_, (LEVELS,))
    _copy_to_c(state.monsters_killed, payload["monsters_killed"], np.int32, (LEVELS,))

    _copy_to_c(state.player_position, payload["player_position"], np.int32, (2,))
    state.player_level = int(payload["player_level"])
    state.player_direction = int(payload["player_direction"])
    state.player_health = float(payload["player_health"])
    state.player_food = int(payload["player_food"])
    state.player_drink = int(payload["player_drink"])
    state.player_energy = int(payload["player_energy"])
    state.player_mana = int(payload["player_mana"])
    state.is_sleeping = bool(payload["is_sleeping"])
    state.is_resting = bool(payload["is_resting"])
    state.player_recover = float(payload["player_recover"])
    state.player_hunger = float(payload["player_hunger"])
    state.player_thirst = float(payload["player_thirst"])
    state.player_fatigue = float(payload["player_fatigue"])
    state.player_recover_mana = float(payload["player_recover_mana"])
    state.player_xp = int(payload["player_xp"])
    state.player_dexterity = int(payload["player_dexterity"])
    state.player_strength = int(payload["player_strength"])
    state.player_intelligence = int(payload["player_intelligence"])

    _copy_inventory_to_c(state.inventory, payload["inventory"])
    _copy_mobs_to_c(state.melee_mobs, payload["melee_mobs"], MAX_MELEE_MOBS)
    _copy_mobs_to_c(state.passive_mobs, payload["passive_mobs"], MAX_PASSIVE_MOBS)
    _copy_mobs_to_c(state.ranged_mobs, payload["ranged_mobs"], MAX_RANGED_MOBS)
    _copy_mobs_to_c(
        state.mob_projectiles, payload["mob_projectiles"], MAX_MOB_PROJECTILES
    )
    _copy_to_c(
        state.mob_projectile_directions,
        payload["mob_projectile_directions"],
        np.int32,
        (LEVELS, MAX_MOB_PROJECTILES, 2),
    )
    _copy_mobs_to_c(
        state.player_projectiles,
        payload["player_projectiles"],
        MAX_PLAYER_PROJECTILES,
    )
    _copy_to_c(
        state.player_projectile_directions,
        payload["player_projectile_directions"],
        np.int32,
        (LEVELS, MAX_PLAYER_PROJECTILES, 2),
    )
    _copy_to_c(
        state.growing_plants_positions,
        payload["growing_plants_positions"],
        np.int32,
        (MAX_GROWING_PLANTS, 2),
    )
    _copy_to_c(
        state.growing_plants_age,
        payload["growing_plants_age"],
        np.int32,
        (MAX_GROWING_PLANTS,),
    )
    _copy_to_c(
        state.growing_plants_mask,
        payload["growing_plants_mask"],
        np.bool_,
        (MAX_GROWING_PLANTS,),
    )
    _copy_to_c(state.potion_mapping, payload["potion_mapping"], np.int32, (6,))
    _copy_to_c(state.learned_spells, payload["learned_spells"], np.bool_, (2,))
    state.sword_enchantment = int(payload["sword_enchantment"])
    state.bow_enchantment = int(payload["bow_enchantment"])
    _copy_to_c(
        state.armour_enchantments, payload["armour_enchantments"], np.int32, (4,)
    )
    state.boss_progress = int(payload["boss_progress"])
    state.boss_timesteps_to_spawn_this_round = int(
        payload["boss_timesteps_to_spawn_this_round"]
    )
    state.light_level = float(payload["light_level"])
    _copy_to_c(state.achievements, payload["achievements"], np.bool_, (ACHIEVEMENTS,))
    _copy_to_c(state.state_rng, payload["state_rng"], np.uint32, (2,))
    state.timestep = int(payload["timestep"])
    _copy_to_c(
        state.fractal_noise_angles,
        payload["fractal_noise_angles"],
        np.int32,
        (4,),
    )
    return state


def jax_state_to_c_state(state: EnvState) -> CraftaxState:
    return deserialize_jax_state_to_c(serialize_jax_state(state))


def _inventory_from_c(inventory):
    return Inventory(
        wood=int(inventory.wood),
        stone=int(inventory.stone),
        coal=int(inventory.coal),
        iron=int(inventory.iron),
        diamond=int(inventory.diamond),
        sapling=int(inventory.sapling),
        pickaxe=int(inventory.pickaxe),
        sword=int(inventory.sword),
        bow=int(inventory.bow),
        arrows=int(inventory.arrows),
        armour=jnp.asarray(_copy_from_c(inventory.armour, np.int32)),
        torches=int(inventory.torches),
        ruby=int(inventory.ruby),
        sapphire=int(inventory.sapphire),
        potions=jnp.asarray(_copy_from_c(inventory.potions, np.int32)),
        books=int(inventory.books),
    )


def _mobs_from_c(mobs):
    return Mobs(
        position=jnp.asarray(_copy_from_c(mobs.position, np.int32)),
        health=jnp.asarray(_copy_from_c(mobs.health, np.float32)),
        mask=jnp.asarray(_copy_from_c(mobs.mask, np.bool_)),
        attack_cooldown=jnp.asarray(_copy_from_c(mobs.attack_cooldown, np.int32)),
        type_id=jnp.asarray(_copy_from_c(mobs.type_id, np.int32)),
    )


def _fractal_from_template(template):
    if template is None:
        return (None, None, None, None)
    return template.fractal_noise_angles


def craftax_state_to_jax(state: CraftaxState, template: EnvState | None = None) -> EnvState:
    return EnvState(
        map=jnp.asarray(_copy_from_c(state.map, np.int32)),
        item_map=jnp.asarray(_copy_from_c(state.item_map, np.int32)),
        mob_map=jnp.asarray(_copy_from_c(state.mob_map, np.bool_)),
        light_map=jnp.asarray(_copy_from_c(state.light_map, np.float32)),
        down_ladders=jnp.asarray(_copy_from_c(state.down_ladders, np.int32)),
        up_ladders=jnp.asarray(_copy_from_c(state.up_ladders, np.int32)),
        chests_opened=jnp.asarray(_copy_from_c(state.chests_opened, np.bool_)),
        monsters_killed=jnp.asarray(_copy_from_c(state.monsters_killed, np.int32)),
        player_position=jnp.asarray(_copy_from_c(state.player_position, np.int32)),
        player_level=int(state.player_level),
        player_direction=int(state.player_direction),
        player_health=float(state.player_health),
        player_food=int(state.player_food),
        player_drink=int(state.player_drink),
        player_energy=int(state.player_energy),
        player_mana=int(state.player_mana),
        is_sleeping=bool(state.is_sleeping),
        is_resting=bool(state.is_resting),
        player_recover=float(state.player_recover),
        player_hunger=float(state.player_hunger),
        player_thirst=float(state.player_thirst),
        player_fatigue=float(state.player_fatigue),
        player_recover_mana=float(state.player_recover_mana),
        player_xp=int(state.player_xp),
        player_dexterity=int(state.player_dexterity),
        player_strength=int(state.player_strength),
        player_intelligence=int(state.player_intelligence),
        inventory=_inventory_from_c(state.inventory),
        melee_mobs=_mobs_from_c(state.melee_mobs),
        passive_mobs=_mobs_from_c(state.passive_mobs),
        ranged_mobs=_mobs_from_c(state.ranged_mobs),
        mob_projectiles=_mobs_from_c(state.mob_projectiles),
        mob_projectile_directions=jnp.asarray(
            _copy_from_c(state.mob_projectile_directions, np.int32)
        ),
        player_projectiles=_mobs_from_c(state.player_projectiles),
        player_projectile_directions=jnp.asarray(
            _copy_from_c(state.player_projectile_directions, np.int32)
        ),
        growing_plants_positions=jnp.asarray(
            _copy_from_c(state.growing_plants_positions, np.int32)
        ),
        growing_plants_age=jnp.asarray(
            _copy_from_c(state.growing_plants_age, np.int32)
        ),
        growing_plants_mask=jnp.asarray(
            _copy_from_c(state.growing_plants_mask, np.bool_)
        ),
        potion_mapping=jnp.asarray(_copy_from_c(state.potion_mapping, np.int32)),
        learned_spells=jnp.asarray(_copy_from_c(state.learned_spells, np.bool_)),
        sword_enchantment=int(state.sword_enchantment),
        bow_enchantment=int(state.bow_enchantment),
        armour_enchantments=jnp.asarray(
            _copy_from_c(state.armour_enchantments, np.int32)
        ),
        boss_progress=int(state.boss_progress),
        boss_timesteps_to_spawn_this_round=int(
            state.boss_timesteps_to_spawn_this_round
        ),
        light_level=float(state.light_level),
        achievements=jnp.asarray(_copy_from_c(state.achievements, np.bool_)),
        state_rng=jnp.asarray(_copy_from_c(state.state_rng, np.uint32)),
        timestep=int(state.timestep),
        fractal_noise_angles=_fractal_from_template(template),
    )


def _flatten_mobs(prefix, mobs):
    return {
        f"{prefix}.position": np.asarray(mobs.position),
        f"{prefix}.health": np.asarray(mobs.health),
        f"{prefix}.mask": np.asarray(mobs.mask),
        f"{prefix}.attack_cooldown": np.asarray(mobs.attack_cooldown),
        f"{prefix}.type_id": np.asarray(mobs.type_id),
    }


def _flatten_inventory(inventory):
    return {
        "inventory.wood": np.asarray(inventory.wood),
        "inventory.stone": np.asarray(inventory.stone),
        "inventory.coal": np.asarray(inventory.coal),
        "inventory.iron": np.asarray(inventory.iron),
        "inventory.diamond": np.asarray(inventory.diamond),
        "inventory.sapling": np.asarray(inventory.sapling),
        "inventory.pickaxe": np.asarray(inventory.pickaxe),
        "inventory.sword": np.asarray(inventory.sword),
        "inventory.bow": np.asarray(inventory.bow),
        "inventory.arrows": np.asarray(inventory.arrows),
        "inventory.armour": np.asarray(inventory.armour),
        "inventory.torches": np.asarray(inventory.torches),
        "inventory.ruby": np.asarray(inventory.ruby),
        "inventory.sapphire": np.asarray(inventory.sapphire),
        "inventory.potions": np.asarray(inventory.potions),
        "inventory.books": np.asarray(inventory.books),
    }


def flatten_env_state(state: EnvState):
    flat = {
        "map": np.asarray(state.map),
        "item_map": np.asarray(state.item_map),
        "mob_map": np.asarray(state.mob_map),
        "light_map": np.asarray(state.light_map),
        "down_ladders": np.asarray(state.down_ladders),
        "up_ladders": np.asarray(state.up_ladders),
        "chests_opened": np.asarray(state.chests_opened),
        "monsters_killed": np.asarray(state.monsters_killed),
        "player_position": np.asarray(state.player_position),
        "player_level": np.asarray(state.player_level),
        "player_direction": np.asarray(state.player_direction),
        "player_health": np.asarray(state.player_health, dtype=np.float32),
        "player_food": np.asarray(state.player_food),
        "player_drink": np.asarray(state.player_drink),
        "player_energy": np.asarray(state.player_energy),
        "player_mana": np.asarray(state.player_mana),
        "is_sleeping": np.asarray(state.is_sleeping),
        "is_resting": np.asarray(state.is_resting),
        "player_recover": np.asarray(state.player_recover, dtype=np.float32),
        "player_hunger": np.asarray(state.player_hunger, dtype=np.float32),
        "player_thirst": np.asarray(state.player_thirst, dtype=np.float32),
        "player_fatigue": np.asarray(state.player_fatigue, dtype=np.float32),
        "player_recover_mana": np.asarray(
            state.player_recover_mana, dtype=np.float32
        ),
        "player_xp": np.asarray(state.player_xp),
        "player_dexterity": np.asarray(state.player_dexterity),
        "player_strength": np.asarray(state.player_strength),
        "player_intelligence": np.asarray(state.player_intelligence),
        "mob_projectile_directions": np.asarray(state.mob_projectile_directions),
        "player_projectile_directions": np.asarray(
            state.player_projectile_directions
        ),
        "growing_plants_positions": np.asarray(state.growing_plants_positions),
        "growing_plants_age": np.asarray(state.growing_plants_age),
        "growing_plants_mask": np.asarray(state.growing_plants_mask),
        "potion_mapping": np.asarray(state.potion_mapping),
        "learned_spells": np.asarray(state.learned_spells),
        "sword_enchantment": np.asarray(state.sword_enchantment),
        "bow_enchantment": np.asarray(state.bow_enchantment),
        "armour_enchantments": np.asarray(state.armour_enchantments),
        "boss_progress": np.asarray(state.boss_progress),
        "boss_timesteps_to_spawn_this_round": np.asarray(
            state.boss_timesteps_to_spawn_this_round
        ),
        "light_level": np.asarray(state.light_level, dtype=np.float32),
        "achievements": np.asarray(state.achievements),
        "state_rng": np.asarray(state.state_rng, dtype=np.uint32),
        "timestep": np.asarray(state.timestep),
        "fractal_noise_angles": np.asarray(
            [0 if value is None else int(value) for value in state.fractal_noise_angles],
            dtype=np.int32,
        ),
    }
    flat.update(_flatten_inventory(state.inventory))
    flat.update(_flatten_mobs("melee_mobs", state.melee_mobs))
    flat.update(_flatten_mobs("passive_mobs", state.passive_mobs))
    flat.update(_flatten_mobs("ranged_mobs", state.ranged_mobs))
    flat.update(_flatten_mobs("mob_projectiles", state.mob_projectiles))
    flat.update(_flatten_mobs("player_projectiles", state.player_projectiles))
    return flat


def assert_env_states_equal(actual: EnvState, expected: EnvState, context: str):
    actual_flat = flatten_env_state(actual)
    expected_flat = flatten_env_state(expected)
    if actual_flat.keys() != expected_flat.keys():
        missing = expected_flat.keys() - actual_flat.keys()
        extra = actual_flat.keys() - expected_flat.keys()
        raise AssertionError(f"{context}: state keys differ missing={missing} extra={extra}")

    for name, expected_value in expected_flat.items():
        actual_value = actual_flat[name]
        err_msg = f"{context}: field {name}"
        if expected_value.dtype.kind == "f":
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                atol=1e-6,
                rtol=0.0,
                err_msg=err_msg,
            )
        else:
            np.testing.assert_array_equal(actual_value, expected_value, err_msg=err_msg)
