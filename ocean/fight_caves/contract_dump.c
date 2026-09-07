/* Cold-path machine-readable metadata exported by the compiled FC backend. */

#include <stdio.h>

#include "fc_api.h"
#include "fc_contracts.h"
#include "fc_player_init.h"

#if defined(_WIN32)
#define FC_CONTRACT_EXPORT __declspec(dllexport)
#else
#define FC_CONTRACT_EXPORT __attribute__((visibility("default")))
#endif

#define FC_STRINGIFY_INNER(value) #value
#define FC_STRINGIFY(value) FC_STRINGIFY_INNER(value)

FC_CONTRACT_EXPORT const char* fc_training_contract_json(void) {
    static char json[2048];
    static int initialized = 0;
    if (!initialized) {
        snprintf(
            json,
            sizeof(json),
            "{"
            "\"contract_dump_schema_version\":%d,"
            "\"policy_obs_size\":%d,"
            "\"puffer_obs_size\":%d,"
            "\"puffer_action_dims\":[%d,%d,%d],"
            "\"puffer_mask_size\":%d,"
            "\"core_obs_size\":%d,"
            "\"core_action_dims\":[%d,%d,%d,%d,%d,%d,%d],"
            "\"core_action_mask\":%d,"
            "\"reward_feature_count\":%d,"
            "\"observation_version\":\"%s\","
            "\"action_version\":\"%s\","
            "\"reward_version\":\"%s\","
            "\"prayer_timing_version\":\"%s\","
            "\"state_hash_version\":%u,"
            "\"active_loadout\":\"%s\""
            "}",
            FC_CONTRACT_DUMP_SCHEMA_VERSION,
            FC_POLICY_OBS_SIZE,
            FC_PUFFER_OBS_SIZE,
            FC_PUFFER_ACTION_DIMS[0],
            FC_PUFFER_ACTION_DIMS[1],
            FC_PUFFER_ACTION_DIMS[2],
            FC_PUFFER_MASK_SIZE,
            FC_OBS_SIZE,
            FC_ACTION_DIMS[0],
            FC_ACTION_DIMS[1],
            FC_ACTION_DIMS[2],
            FC_ACTION_DIMS[3],
            FC_ACTION_DIMS[4],
            FC_ACTION_DIMS[5],
            FC_ACTION_DIMS[6],
            FC_ACTION_MASK_SIZE,
            FC_REWARD_FEATURES,
            FC_OBSERVATION_VERSION,
            FC_ACTION_VERSION,
            FC_REWARD_VERSION,
            FC_PRAYER_TIMING_VERSION,
            FC_STATE_HASH_VERSION,
            FC_STRINGIFY(FC_ACTIVE_LOADOUT));
        initialized = 1;
    }
    return json;
}
