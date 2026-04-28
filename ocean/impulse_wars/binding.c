#include "env.h"

#define OBS_SIZE 998 // for 2 drones (players)
// actions:
// 9: move, noop + 8 directions
// 17: aim, noop + 16 directions
// 2: shoot or not
// 2: brake or not
// 2: burst or not
#define NUM_ATNS 5
#define ACT_SIZES {9, 17, 2, 2, 2}
#define OBS_TENSOR_T FloatTensor

#define Env iwEnv
#include "vecenv.h"

#define DICTGET(key) dict_get(kwargs, key)->value

void my_init(Env* env, Dict* kwargs) {
    initEnv(
        env,
        2,
        1,
        -1,
        0,
        (bool)DICTGET("enable_teams"),
        (bool)DICTGET("sitting_duck"),
        (bool)DICTGET("is_training"),
        (bool)DICTGET("continuous")
    );

    setRewards(
        env,
        (float)DICTGET("reward_win"),
        (float)DICTGET("reward_self_kill"),
        (float)DICTGET("reward_enemy_death"),
        (float)DICTGET("reward_enemy_kill"),
        0.0f, // teammate death punishment
        0.0f, // teammate kill punishment
        (float)DICTGET("reward_death"),
        (float)DICTGET("reward_energy_emptied"),
        (float)DICTGET("reward_weapon_pickup"),
        (float)DICTGET("reward_shield_break"),
        (float)DICTGET("reward_shot_hit_coef"),
        (float)DICTGET("reward_explosion_hit_coef")
    );

    initMaps(env);
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

void my_log(Log *log, Dict *out) {
    dict_set(out, "episode_length", log->length);
    dict_set(out, "ties", log->ties);

    dict_set(out, "perf", log->stats[0].wins);
    dict_set(out, "score", log->stats[0].wins);

    char buf[_LOG_BUF_SIZE] = {0};
    for (uint8_t i = 0; i < MAX_DRONES; i++) {
        dict_set(out, droneLog(buf, i, "returns"), log->stats[i].returns);
        dict_set(out, droneLog(buf, i, "distance_traveled"), log->stats[i].distanceTraveled);
        dict_set(out, droneLog(buf, i, "abs_distance_traveled"), log->stats[i].absDistanceTraveled);
        dict_set(out, droneLog(buf, i, "brake_time"), log->stats[i].brakeTime);
        dict_set(out, droneLog(buf, i, "total_bursts"), log->stats[i].totalBursts);
        dict_set(out, droneLog(buf, i, "bursts_hit"), log->stats[i].burstsHit);
        dict_set(out, droneLog(buf, i, "energy_emptied"), log->stats[i].energyEmptied);
        dict_set(out, droneLog(buf, i, "shields_broken"), log->stats[i].shieldsBroken);
        dict_set(out, droneLog(buf, i, "own_shield_broken"), log->stats[i].ownShieldBroken);
        dict_set(out, droneLog(buf, i, "self_kills"), log->stats[i].selfKills);
        dict_set(out, droneLog(buf, i, "kills"), log->stats[i].kills);
        dict_set(out, droneLog(buf, i, "unknown_kills"), log->stats[i].unknownKills);
        dict_set(out, droneLog(buf, i, "wins"), log->stats[i].wins);

        // useful for debugging weapon balance, but really slows down
        // sweeps due to adding a ton of extra logging data
        //
        // for (uint8_t j = 0; j < _NUM_WEAPONS; j++) {
        //     dict_set(out, weaponLog(buf, i, j, "shots_fired"), log->stats[i].shotsFired[j]);
        //     dict_set(out, weaponLog(buf, i, j, "shots_hit"), log->stats[i].shotsHit[j]);
        //     dict_set(out, weaponLog(buf, i, j, "shots_taken"), log->stats[i].shotsTaken[j]);
        //     dict_set(out, weaponLog(buf, i, j, "own_shots_taken"), log->stats[i].ownShotsTaken[j]);
        //     dict_set(out, weaponLog(buf, i, j, "picked_up"), log->stats[i].weaponsPickedUp[j]);
        //     dict_set(out, weaponLog(buf, i, j, "shot_distances"), log->stats[i].shotDistances[j]);
        // }

        dict_set(out, droneLog(buf, i, "total_shots_fired"), log->stats[i].totalShotsFired);
        dict_set(out, droneLog(buf, i, "total_shots_hit"), log->stats[i].totalShotsHit);
        dict_set(out, droneLog(buf, i, "total_shots_taken"), log->stats[i].totalShotsTaken);
        dict_set(out, droneLog(buf, i, "total_own_shots_taken"), log->stats[i].totalOwnShotsTaken);
        dict_set(out, droneLog(buf, i, "total_picked_up"), log->stats[i].totalWeaponsPickedUp);
        dict_set(out, droneLog(buf, i, "total_shot_distances"), log->stats[i].totalShotDistances);
    }
}
