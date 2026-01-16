#include <Python.h>

#include "env.h"

static PyObject *get_consts(PyObject *self, PyObject *args);

#define Env iwEnv
#define MY_SHARED
#define MY_METHODS {"get_consts", get_consts, METH_VARARGS, "Get constants"}

#include "../env_binding.h"

#define setDictVal(dict, key, val)                                            \
    if (PyDict_SetItemString(dict, key, PyLong_FromLong(val)) < 0) {          \
        PyErr_SetString(PyExc_RuntimeError, "Failed to set " key " in dict"); \
        return NULL;                                                          \
    }

static PyObject *get_consts(PyObject *self, PyObject *args) {
    PyObject *dronesArg = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(dronesArg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "num_drones must be an integer");
        return NULL;
    }
    const uint8_t numDrones = (uint8_t)PyLong_AsLong(dronesArg);

    PyObject *dict = PyDict_New();
    if (PyErr_Occurred()) {
        return NULL;
    }

    const uint16_t droneObsOffset = ENEMY_DRONE_OBS_OFFSET + ((numDrones - 1) * ENEMY_DRONE_OBS_SIZE);

    setDictVal(dict, "obsBytes", obsBytes(numDrones));
    setDictVal(dict, "mapObsSize", MAP_OBS_SIZE);
    setDictVal(dict, "discreteObsSize", discreteObsSize(numDrones));
    setDictVal(dict, "continuousObsSize", continuousObsSize(numDrones));
    setDictVal(dict, "continuousObsBytes", continuousObsSize(numDrones) * sizeof(float));
    setDictVal(dict, "wallTypes", NUM_WALL_TYPES);
    setDictVal(dict, "weaponTypes", NUM_WEAPONS + 1);
    setDictVal(dict, "mapObsRows", MAP_OBS_ROWS);
    setDictVal(dict, "mapObsColumns", MAP_OBS_COLUMNS);
    setDictVal(dict, "continuousObsOffset", alignedSize(MAP_OBS_SIZE, sizeof(float)));
    setDictVal(dict, "numNearWallObs", NUM_NEAR_WALL_OBS);
    setDictVal(dict, "nearWallTypesObsOffset", NEAR_WALL_TYPES_OBS_OFFSET);
    setDictVal(dict, "nearWallPosObsSize", NEAR_WALL_POS_OBS_SIZE);
    setDictVal(dict, "nearWallObsSize", NEAR_WALL_OBS_SIZE);
    setDictVal(dict, "nearWallPosObsOffset", NEAR_WALL_POS_OBS_OFFSET);
    setDictVal(dict, "numFloatingWallObs", NUM_FLOATING_WALL_OBS);
    setDictVal(dict, "floatingWallTypesObsOffset", FLOATING_WALL_TYPES_OBS_OFFSET);
    setDictVal(dict, "floatingWallInfoObsSize", FLOATING_WALL_INFO_OBS_SIZE);
    setDictVal(dict, "floatingWallObsSize", FLOATING_WALL_OBS_SIZE);
    setDictVal(dict, "floatingWallInfoObsOffset", FLOATING_WALL_INFO_OBS_OFFSET);
    setDictVal(dict, "numWeaponPickupObs", NUM_WEAPON_PICKUP_OBS);
    setDictVal(dict, "weaponPickupTypesObsOffset", WEAPON_PICKUP_WEAPONS_OBS_OFFSET);
    setDictVal(dict, "weaponPickupPosObsSize", WEAPON_PICKUP_POS_OBS_SIZE);
    setDictVal(dict, "weaponPickupObsSize", WEAPON_PICKUP_OBS_SIZE);
    setDictVal(dict, "weaponPickupPosObsOffset", WEAPON_PICKUP_POS_OBS_OFFSET);
    setDictVal(dict, "numProjectileObs", NUM_PROJECTILE_OBS);
    setDictVal(dict, "projectileDroneObsOffset", PROJECTILE_DRONE_OBS_OFFSET);
    setDictVal(dict, "projectileTypesObsOffset", PROJECTILE_WEAPONS_OBS_OFFSET);
    setDictVal(dict, "projectileInfoObsSize", PROJECTILE_INFO_OBS_SIZE);
    setDictVal(dict, "projectileObsSize", PROJECTILE_OBS_SIZE);
    setDictVal(dict, "projectileInfoObsOffset", PROJECTILE_INFO_OBS_OFFSET);
    setDictVal(dict, "enemyDroneWeaponsObsOffset", ENEMY_DRONE_WEAPONS_OBS_OFFSET);
    setDictVal(dict, "enemyDroneObsOffset", ENEMY_DRONE_OBS_OFFSET);
    setDictVal(dict, "enemyDroneObsSize", ENEMY_DRONE_OBS_SIZE);
    setDictVal(dict, "droneObsOffset", droneObsOffset);
    setDictVal(dict, "droneObsSize", DRONE_OBS_SIZE);
    setDictVal(dict, "miscObsSize", MISC_OBS_SIZE);
    setDictVal(dict, "miscObsOffset", droneObsOffset + DRONE_OBS_SIZE);

    setDictVal(dict, "maxDrones", MAX_DRONES);
    setDictVal(dict, "contActionsSize", CONTINUOUS_ACTION_SIZE);

    return dict;
}

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs) {
    VecEnv *ve = unpack_vecenv(args);
    initMaps(ve->envs[0]);

    for (uint16_t i = 0; i < ve->num_envs; i++) {
        iwEnv *e = (iwEnv *)ve->envs[i];
        setupEnv(e);
    }

    return Py_None;
}

static int my_init(iwEnv *e, PyObject *args, PyObject *kwargs) {
    initEnv(
        e,
        (uint8_t)unpack(kwargs, "num_drones"),
        (uint8_t)unpack(kwargs, "num_agents"),
        (int8_t)unpack(kwargs, "map_idx"),
        (uint64_t)unpack(kwargs, "seed"),
        (bool)unpack(kwargs, "enable_teams"),
        (bool)unpack(kwargs, "sitting_duck"),
        (bool)unpack(kwargs, "is_training"),
        (bool)unpack(kwargs, "continuous")
    );
    setRewards(
        e,
        (float)unpack(kwargs, "reward_win"),
        (float)unpack(kwargs, "reward_self_kill"),
        (float)unpack(kwargs, "reward_enemy_death"),
        (float)unpack(kwargs, "reward_enemy_kill"),
        0.0f, // teammate death punishment
        0.0f, // teammate kill punishment
        (float)unpack(kwargs, "reward_death"),
        (float)unpack(kwargs, "reward_energy_emptied"),
        (float)unpack(kwargs, "reward_weapon_pickup"),
        (float)unpack(kwargs, "reward_shield_break"),
        (float)unpack(kwargs, "reward_shot_hit_coef"),
        (float)unpack(kwargs, "reward_explosion_hit_coef")
    );
    return 0;
}

#define _LOG_BUF_SIZE 128

char *droneLog(char *buf, const uint8_t droneIdx, const char *name) {
    snprintf(buf, _LOG_BUF_SIZE, "drone_%d_%s", droneIdx, name);
    return buf;
}

char *weaponLog(char *buf, const uint8_t droneIdx, const uint8_t weaponIdx, const char *name) {
    snprintf(buf, _LOG_BUF_SIZE, "drone_%d_%s_%s", droneIdx, weaponNames[weaponIdx], name);
    return buf;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "episode_length", log->length);
    assign_to_dict(dict, "ties", log->ties);

    assign_to_dict(dict, "perf", log->stats[0].wins);
    assign_to_dict(dict, "score", log->stats[0].wins);

    char buf[_LOG_BUF_SIZE] = {0};
    for (uint8_t i = 0; i < MAX_DRONES; i++) {
        assign_to_dict(dict, droneLog(buf, i, "returns"), log->stats[i].returns);
        assign_to_dict(dict, droneLog(buf, i, "distance_traveled"), log->stats[i].distanceTraveled);
        assign_to_dict(dict, droneLog(buf, i, "abs_distance_traveled"), log->stats[i].absDistanceTraveled);
        assign_to_dict(dict, droneLog(buf, i, "brake_time"), log->stats[i].brakeTime);
        assign_to_dict(dict, droneLog(buf, i, "total_bursts"), log->stats[i].totalBursts);
        assign_to_dict(dict, droneLog(buf, i, "bursts_hit"), log->stats[i].burstsHit);
        assign_to_dict(dict, droneLog(buf, i, "energy_emptied"), log->stats[i].energyEmptied);
        assign_to_dict(dict, droneLog(buf, i, "shields_broken"), log->stats[i].shieldsBroken);
        assign_to_dict(dict, droneLog(buf, i, "own_shield_broken"), log->stats[i].ownShieldBroken);
        assign_to_dict(dict, droneLog(buf, i, "self_kills"), log->stats[i].selfKills);
        assign_to_dict(dict, droneLog(buf, i, "kills"), log->stats[i].kills);
        assign_to_dict(dict, droneLog(buf, i, "unknown_kills"), log->stats[i].unknownKills);
        assign_to_dict(dict, droneLog(buf, i, "wins"), log->stats[i].wins);

        // useful for debugging weapon balance, but really slows down
        // sweeps due to adding a ton of extra logging data
        //
        // for (uint8_t j = 0; j < _NUM_WEAPONS; j++) {
        //     assign_to_dict(dict, weaponLog(buf, i, j, "shots_fired"), log->stats[i].shotsFired[j]);
        //     assign_to_dict(dict, weaponLog(buf, i, j, "shots_hit"), log->stats[i].shotsHit[j]);
        //     assign_to_dict(dict, weaponLog(buf, i, j, "shots_taken"), log->stats[i].shotsTaken[j]);
        //     assign_to_dict(dict, weaponLog(buf, i, j, "own_shots_taken"), log->stats[i].ownShotsTaken[j]);
        //     assign_to_dict(dict, weaponLog(buf, i, j, "picked_up"), log->stats[i].weaponsPickedUp[j]);
        //     assign_to_dict(dict, weaponLog(buf, i, j, "shot_distances"), log->stats[i].shotDistances[j]);
        // }

        assign_to_dict(dict, droneLog(buf, i, "total_shots_fired"), log->stats[i].totalShotsFired);
        assign_to_dict(dict, droneLog(buf, i, "total_shots_hit"), log->stats[i].totalShotsHit);
        assign_to_dict(dict, droneLog(buf, i, "total_shots_taken"), log->stats[i].totalShotsTaken);
        assign_to_dict(dict, droneLog(buf, i, "total_own_shots_taken"), log->stats[i].totalOwnShotsTaken);
        assign_to_dict(dict, droneLog(buf, i, "total_picked_up"), log->stats[i].totalWeaponsPickedUp);
        assign_to_dict(dict, droneLog(buf, i, "total_shot_distances"), log->stats[i].totalShotDistances);
    }

    return 0;
}
