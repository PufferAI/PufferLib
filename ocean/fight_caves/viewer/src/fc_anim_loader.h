#ifndef FC_ANIM_LOADER_H
#define FC_ANIM_LOADER_H

#include <stdint.h>

#define ANIM_MAX_LABELS 256

typedef struct {
    uint16_t base_id;
    uint8_t slot_count;
    uint8_t *types;
    uint8_t *map_lengths;
    uint8_t **frame_maps;
} AnimFrameBase;

typedef struct {
    uint8_t slot_index;
    int16_t dx;
    int16_t dy;
    int16_t dz;
} AnimTransform;

typedef struct {
    uint16_t framebase_id;
    uint8_t transform_count;
    AnimTransform *transforms;
} AnimFrameData;

typedef struct {
    uint16_t delay;
    AnimFrameData frame;
} AnimSequenceFrame;

typedef struct {
    uint16_t seq_id;
    uint16_t frame_count;
    uint8_t interleave_count;
    uint8_t *interleave_order;
    int16_t frame_step;
    int8_t preanim_move;
    int8_t postanim_move;
    uint8_t forced_priority;
    uint8_t max_loops;
    int8_t reply_mode;
    uint8_t stretches;
    int8_t walk_flag;
    AnimSequenceFrame *frames;
} AnimSequence;

typedef struct {
    AnimFrameBase *bases;
    int base_count;
    uint16_t *base_ids;
    AnimSequence *sequences;
    int seq_count;
} AnimCache;

typedef struct {
    int16_t *verts;
    int vert_count;
    int **groups;
    int *group_counts;
} AnimModelState;

AnimCache *anim_cache_load(const char *path);
AnimSequence *anim_get_sequence(AnimCache *cache, uint16_t seq_id);
AnimFrameBase *anim_get_framebase(AnimCache *cache, uint16_t base_id);
AnimModelState *anim_model_state_create(const uint8_t *vertex_skins,
                                        int base_vert_count);
void anim_model_state_free(AnimModelState *state);
void anim_apply_frame(AnimModelState *state, const int16_t *base_verts_src,
                      const AnimFrameData *frame, const AnimFrameBase *fb);
int anim_mix_pose_action(AnimCache *cache, AnimModelState *state,
                         const int16_t *base_verts, AnimSequence *pose,
                         int pose_frame_index, AnimSequence *action,
                         int action_frame_index);
void anim_update_mesh(float *mesh_vertices, const AnimModelState *state,
                      const uint16_t *face_indices, int face_count);
void anim_cache_free(AnimCache *cache);

#endif
