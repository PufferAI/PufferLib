#include "fc_model_animation.h"

#include "raylib.h"

static void apply_frame(NpcModelEntry *entry,
                        AnimModelState *state,
                        const AnimFrameData *frame,
                        const AnimFrameBase *framebase) {
    if (!entry || !entry->loaded || !state || !frame || !framebase) return;
    anim_apply_frame(state, entry->base_verts, frame, framebase);
    fc_model_animation_upload(entry, state);
}

void fc_model_animation_upload(NpcModelEntry *entry, AnimModelState *state) {
    if (!entry || !entry->loaded || !state) return;
    models_recompute_texture_uvs_from_vertices(entry, state->verts);
    float *mesh_vertices = entry->model.meshes[0].vertices;
    anim_update_mesh(mesh_vertices, state, entry->face_indices,
                     entry->face_count);
    int expanded_vertices = entry->face_count * 3;
    for (int i = 0; i < expanded_vertices; i++) {
        mesh_vertices[i * 3] /= 128.0f;
        mesh_vertices[i * 3 + 1] /= 128.0f;
        mesh_vertices[i * 3 + 2] /= -128.0f;
    }
    UpdateMeshBuffer(entry->model.meshes[0], 0, mesh_vertices,
                     expanded_vertices * 3 * sizeof(float), 0);
}

void fc_model_animation_update(NpcModelEntry *entry,
                               AnimCache *cache,
                               AnimModelState **state,
                               uint16_t *current_sequence,
                               int *frame_index,
                               float *frame_timer,
                               int animation_id,
                               float dt,
                               float phase_ticks) {
    if (!entry || !entry->loaded || !cache || animation_id < 0 ||
        !entry->vertex_skins || !state || !current_sequence ||
        !frame_index || !frame_timer) return;
    AnimSequence *sequence = anim_get_sequence(cache, (uint16_t)animation_id);
    if (!sequence || sequence->frame_count == 0) return;
    if (!*state || (*state)->vert_count != entry->base_vert_count) {
        if (*state) anim_model_state_free(*state);
        *state = anim_model_state_create(entry->vertex_skins,
                                         entry->base_vert_count);
        *current_sequence = (uint16_t)animation_id;
        *frame_index = (int)phase_ticks % sequence->frame_count;
        if (*frame_index < 0) *frame_index = 0;
        *frame_timer = (float)sequence->frames[*frame_index].delay * 0.02f;
        if (*frame_timer < 0.016f) *frame_timer = 0.016f;
    }
    if (*current_sequence != (uint16_t)animation_id) {
        *current_sequence = (uint16_t)animation_id;
        *frame_index = 0;
        *frame_timer = (float)sequence->frames[0].delay * 0.02f;
        if (*frame_timer < 0.016f) *frame_timer = 0.016f;
    }
    *frame_timer -= dt;
    while (*frame_timer <= 0.0f) {
        *frame_index = (*frame_index + 1) % sequence->frame_count;
        float delay = (float)sequence->frames[*frame_index].delay * 0.02f;
        if (delay < 0.016f) delay = 0.016f;
        *frame_timer += delay;
    }
    AnimFrameData *frame = &sequence->frames[*frame_index].frame;
    AnimFrameBase *framebase = anim_get_framebase(cache, frame->framebase_id);
    if (framebase) apply_frame(entry, *state, frame, framebase);
}
