/*
 * Fight Caves model compatibility wrapper.
 *
 * The viewer historically used NpcModelSet/NpcModelEntry names for every
 * runtime model file. Keep those names while routing all NPC/player/projectile
 * files through the generalized MDL2/MDL3 model loader.
 */

#include "fc_npc_models.h"
#include "fc_models.h"

uint32_t fc_npc_type_to_model_id(int npc_type) {
    switch (npc_type) {
        case 1: return FC_B237_TZ_KIH;
        case 2: return FC_B237_TZ_KEK;
        case 3: return FC_B237_TZ_KEK_SM;
        case 4: return FC_B237_TOK_XIL;
        case 5: return FC_B237_YT_MEJKOT;
        case 6: return FC_B237_KET_ZEK;
        case 7: return FC_B237_TZTOK_JAD;
        case 8: return FC_B237_YT_HURKOT;
        default: return 0;
    }
}

NpcModelEntry* fc_npc_model_find(NpcModelSet* set, uint32_t model_id) {
    return model_find(set, model_id);
}

NpcModelSet* fc_npc_models_load(const char* path, Texture2D atlas_texture) {
    return models_load(path, atlas_texture);
}

void fc_npc_models_unload(NpcModelSet* set) {
    models_free(set);
}
