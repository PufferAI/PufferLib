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

void my_init(Env *env, Dict *kwargs) {
    initEnv(
        env,
        MAX_DRONES,
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

#define LOG_DRONE_STATS(log, out, idx, idxStr)                                                    \
    dict_set(out, "drone_" idxStr "_returns", log->stats[idx].returns);                           \
    dict_set(out, "drone_" idxStr "_distance_traveled", log->stats[idx].distanceTraveled);        \
    dict_set(out, "drone_" idxStr "_abs_distance_traveled", log->stats[idx].absDistanceTraveled); \
    dict_set(out, "drone_" idxStr "_brake_time", log->stats[idx].brakeTime);                      \
    dict_set(out, "drone_" idxStr "_total_bursts", log->stats[idx].totalBursts);                  \
    dict_set(out, "drone_" idxStr "_bursts_hit", log->stats[idx].burstsHit);                      \
    dict_set(out, "drone_" idxStr "_energy_emptied", log->stats[idx].energyEmptied);              \
    dict_set(out, "drone_" idxStr "_shields_broken", log->stats[idx].shieldsBroken);              \
    dict_set(out, "drone_" idxStr "_own_shield_broken", log->stats[idx].ownShieldBroken);         \
    dict_set(out, "drone_" idxStr "_self_kills", log->stats[idx].selfKills);                      \
    dict_set(out, "drone_" idxStr "_kills", log->stats[idx].kills);                               \
    dict_set(out, "drone_" idxStr "_unknown_kills", log->stats[idx].unknownKills);                \
    dict_set(out, "drone_" idxStr "_wins", log->stats[idx].wins);                                 \
    dict_set(out, "drone_" idxStr "_total_shots_fired", log->stats[idx].totalShotsFired);         \
    dict_set(out, "drone_" idxStr "_total_shots_hit", log->stats[idx].totalShotsHit);             \
    dict_set(out, "drone_" idxStr "_total_shots_taken", log->stats[idx].totalShotsTaken);         \
    dict_set(out, "drone_" idxStr "_total_own_shots_taken", log->stats[idx].totalOwnShotsTaken);  \
    dict_set(out, "drone_" idxStr "_total_picked_up", log->stats[idx].totalWeaponsPickedUp);      \
    dict_set(out, "drone_" idxStr "_total_shot_distances", log->stats[idx].totalShotDistances)

void my_log(Log *log, Dict *out) {
    dict_set(out, "episode_length", log->length);
    dict_set(out, "ties", log->ties);

    dict_set(out, "perf", log->stats[0].wins);
    dict_set(out, "score", log->stats[0].wins);

    LOG_DRONE_STATS(log, out, 0, "0");
    LOG_DRONE_STATS(log, out, 1, "1");
}
