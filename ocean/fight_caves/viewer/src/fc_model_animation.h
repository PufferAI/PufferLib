#ifndef FC_MODEL_ANIMATION_H
#define FC_MODEL_ANIMATION_H

#include "fc_anim_loader.h"
#include "fc_npc_models.h"

void fc_model_animation_upload(NpcModelEntry *entry, AnimModelState *state);
void fc_model_animation_update(NpcModelEntry *entry,
                               AnimCache *cache,
                               AnimModelState **state,
                               uint16_t *current_sequence,
                               int *frame_index,
                               float *frame_timer,
                               int animation_id,
                               float dt,
                               float phase_ticks);

#endif
