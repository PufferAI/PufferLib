#ifndef FC_NPC_MODELS_H
#define FC_NPC_MODELS_H

#include "fc_models.h"

#define FC_B237_TZ_KIH 3116u
#define FC_B237_TZ_KEK 3118u
#define FC_B237_TZ_KEK_SM 3120u
#define FC_B237_TOK_XIL 3121u
#define FC_B237_YT_MEJKOT 3123u
#define FC_B237_KET_ZEK 3125u
#define FC_B237_TZTOK_JAD 3127u
#define FC_B237_YT_HURKOT 3128u

typedef ModelEntry NpcModelEntry;
typedef ModelSet NpcModelSet;

uint32_t fc_npc_type_to_model_id(int npc_type);
NpcModelEntry *fc_npc_model_find(NpcModelSet *set, uint32_t model_id);
NpcModelSet *fc_npc_models_load(const char *path, Texture2D atlas_texture);
void fc_npc_models_unload(NpcModelSet *set);

#endif
