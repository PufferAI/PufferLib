#ifndef FIGHT_CAVES_ASSETS_H
#define FIGHT_CAVES_ASSETS_H

/* Io */

#include <stdbool.h>
#include <stdio.h>

static inline bool fc_read_exact(FILE* f, void* dst, size_t elem_size,
                                 size_t elem_count, const char* path,
                                 const char* what) {
    if (fread(dst, elem_size, elem_count, f) != elem_count) {
        fprintf(stderr, "%s: short read while loading %s\n", path, what);
        return false;
    }
    return true;
}

static inline bool fc_seek(FILE* f, long offset, int origin,
                           const char* path, const char* what) {
    if (fseek(f, offset, origin) != 0) {
        fprintf(stderr, "%s: seek failed while loading %s\n", path, what);
        return false;
    }
    return true;
}


/* Assets */

#include <stddef.h>
#include <stdio.h>

#define FC_ASSET_PATH_MAX 1024

const char* fc_asset_root(void);
const char* fc_repo_root(void);

int fc_asset_resolve_path(const char* logical_path, char* out, size_t cap);
int fc_repo_resolve_path(const char* logical_path, char* out, size_t cap);
int fc_asset_exists(const char* logical_path);

FILE* fc_asset_fopen(const char* logical_path, const char* mode);
int fc_asset_close(FILE* f);
unsigned char* fc_asset_read_all(const char* logical_path, size_t* out_size);


/* Asset Raylib */

#include "raylib.h"

Texture2D fc_load_texture_asset(const char* path);
Image fc_load_image_asset(const char* path);
Font fc_load_font_asset(const char* path, int font_size);


/* Animated Atlas */

#include "raylib.h"

#include <stdint.h>

typedef struct {
    uint32_t texture_id;
    uint16_t x;
    uint16_t y;
    uint16_t w;
    uint16_t h;
    uint8_t direction;
    uint8_t speed;
    uint16_t pad;
} FcTextureAnimRow;

typedef struct {
    Texture2D texture;
    unsigned char* base_pixels;
    unsigned char* pixels;
    FcTextureAnimRow* anims;
    int width;
    int height;
    int anim_count;
    float anim_ticks;
} FcAnimatedAtlas;

int fc_animated_atlas_load(FcAnimatedAtlas* atlas, const char* atlas_path,
                           int enable_animation);
void fc_animated_atlas_update(FcAnimatedAtlas* atlas, float dt);
void fc_animated_atlas_unload(FcAnimatedAtlas* atlas);


/* Anim Loader */

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


/* Models */

#include "raylib.h"
#include <stdint.h>

typedef struct {
    uint8_t textured;
    uint16_t tex_a;
    uint16_t tex_b;
    uint16_t tex_c;
    float u_base;
    float v_base;
    float u_scale;
    float v_scale;
    float repeat_v_margin;
} ModelFaceUvInfo;

typedef struct {
    uint32_t model_id;
    Model model;
    int loaded;
    float *rest_verts;
    float *rest_texcoords;
    int16_t *base_verts;
    uint8_t *vertex_skins;
    uint16_t *face_indices;
    uint8_t *face_priorities;
    ModelFaceUvInfo *face_uvs;
    int base_vert_count;
    int face_count;
} ModelEntry;

typedef struct {
    ModelEntry *entries;
    int *index_by_id;
    int count;
    int index_limit;
    int has_textures;
    int loaded;
} ModelSet;

ModelSet *models_load(const char *path, Texture2D atlas_texture);
ModelEntry *model_find(ModelSet *set, uint32_t id);
/* Screen-space face picking at the drawn pose, with the client's 5px tolerance.
 * Returns camera depth for overlap ordering, or -1 for a miss. posed_vertices
 * uses animation/cache coordinates; NULL selects the asset's rest mesh. */
float models_pick_depth(const ModelEntry *entry, const int16_t *posed_vertices,
                         Vector3 position, float yaw_degrees, Camera3D camera,
                         Vector2 mouse, int screen_width, int screen_height);
void models_recompute_texture_uvs_from_vertices(ModelEntry *entry,
                                                const int16_t *verts);
void models_free(ModelSet *set);


/* Npc Models */

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


/* Model Animation */

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


/* Objects Loader */

#include "raylib.h"
#include <stdint.h>

#define OANM_FLAG_DYNAMIC_BASE 1u
#define OANM_FLAG_DYNAMIC_REPLACEMENT 2u

typedef struct {
    Model model;
    FcAnimatedAtlas atlas;
    int placement_count;
    int total_vertex_count;
    int min_world_x;
    int min_world_y;
    int has_textures;
    int loaded;
} ObjectMesh;

typedef struct {
    uint32_t model_id;
    uint32_t obj_id;
    int32_t animation_id;
    int32_t world_x;
    int32_t world_y;
    uint8_t plane;
    uint8_t obj_type;
    uint8_t rotation;
    uint8_t flags;
    float pos_x;
    float pos_y;
    float pos_z;
    float phase_ticks;
} ObjectAnimPlacement;

typedef struct {
    ObjectAnimPlacement *rows;
    int count;
    int loaded;
} ObjectAnimSet;

ObjectMesh *objects_load(const char *path);
ObjectAnimSet *object_anims_load(const char *path);
void object_anims_offset(ObjectAnimSet *set, int wx, int wy);
void objects_offset(ObjectMesh *om, int wx, int wy);
void objects_free(ObjectMesh *om);
void object_anims_free(ObjectAnimSet *set);


/* Terrain Loader */

#include "raylib.h"

typedef struct {
    Model model;
    int vertex_count;
    int region_count;
    int min_world_x;
    int min_world_y;
    int loaded;
    float *heightmap;
    int hm_min_x;
    int hm_min_y;
    int hm_width;
    int hm_height;
} TerrainMesh;

TerrainMesh *terrain_load(const char *path);
void terrain_offset(TerrainMesh *tm, int wx, int wy);
float terrain_height_at(TerrainMesh *tm, int world_x, int world_y);
void terrain_free(TerrainMesh *tm);


/* Spotanims */

#include <stdint.h>

typedef struct {
    uint32_t id;
    int32_t model_id;
    int32_t animation_id;
    uint32_t resize_xy;
    uint32_t resize_z;
    uint32_t rotation;
    int32_t brightness;
    int32_t shadow;
} SpotAnimDef;

typedef struct {
    SpotAnimDef *defs;
    int count;
    int loaded;
} SpotAnimSet;

SpotAnimSet *spotanims_load(const char *path);
const SpotAnimDef *spotanim_find(const SpotAnimSet *set, int id);
void spotanims_free(SpotAnimSet *set);

#include "simulation.h"

/* Player Appearance */
typedef struct {
    ModelSet *parts;
    ModelSet *model;
    struct { uint32_t item_id, hide_mask; } records[64];
    int record_count;
    int worn_ids[FC_EQUIPMENT_SLOTS];
} FcPlayerAppearance;

int fc_player_appearance_load(FcPlayerAppearance *appearance);
/* 1 = rebuilt, 0 = unchanged, -1 = missing/invalid asset or allocation failure. */
int fc_player_appearance_sync(FcPlayerAppearance *appearance,
                               const FcPlayer *player, uint32_t model_id);
void fc_player_appearance_free(FcPlayerAppearance *appearance);

/* Assets */
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int fc_has_prefix(const char* s, const char* prefix) {
    size_t n;
    if (!s || !prefix) return 0;
    n = strlen(prefix);
    return strncmp(s, prefix, n) == 0;
}

static int fc_path_is_absolute(const char* path) {
    return path && path[0] == '/';
}

static int fc_file_exists_path(const char* path) {
    struct stat st;
    return path && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int fc_dir_exists_path(const char* path) {
    struct stat st;
    return path && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int fc_join_path(char* out, size_t cap, const char* a, const char* b) {
    int n;
    if (!out || cap == 0 || !a || !b) return 0;
    if (a[0] == '\0') {
        n = snprintf(out, cap, "%s", b);
    } else if (a[strlen(a) - 1] == '/') {
        n = snprintf(out, cap, "%s%s", a, b);
    } else {
        n = snprintf(out, cap, "%s/%s", a, b);
    }
    return n > 0 && (size_t)n < cap;
}

static void fc_copy_path(char* dst, size_t cap, const char* src) {
    if (!dst || cap == 0) return;
    if (!src) src = "";
    snprintf(dst, cap, "%s", src);
}

static void fc_trim_trailing_slash(char* path) {
    size_t n;
    if (!path) return;
    n = strlen(path);
    while (n > 1 && path[n - 1] == '/') {
        path[n - 1] = '\0';
        n--;
    }
}

static const char* fc_asset_logical_path(const char* path) {
    const char* marker;
    if (!path) return "";
    while (fc_has_prefix(path, "./")) path += 2;
    marker = strstr(path, "/resources/fight_caves/viewer/");
    if (marker) return marker + strlen("/resources/fight_caves/viewer/");
    if (fc_has_prefix(path, "resources/fight_caves/viewer/"))
        return path + strlen("resources/fight_caves/viewer/");
    if (fc_has_prefix(path, "assets/"))
        return path + strlen("assets/");
    return path;
}

static int fc_derive_asset_root(char* out, size_t cap) {
    const char* marker = strstr(__FILE__, "ocean/fight_caves/");
    size_t prefix_len;
    int n;
    if (!marker) return 0;
    prefix_len = (size_t)(marker - __FILE__);
    n = snprintf(out, cap, "%.*sresources/fight_caves/viewer",
                 (int)prefix_len, __FILE__);
    return n > 0 && (size_t)n < cap;
}

static int fc_derive_repo_root(char* out, size_t cap) {
    const char* marker = strstr(__FILE__, "ocean/fight_caves/");
    size_t prefix_len;
    int n;
    if (!marker) return 0;
    prefix_len = (size_t)(marker - __FILE__);
    n = snprintf(out, cap, "%.*s", (int)prefix_len, __FILE__);
    if (!(n > 0 && (size_t)n < cap)) return 0;
    fc_trim_trailing_slash(out);
    return out[0] != '\0';
}

static int fc_derive_repo_root_from_asset_root(char* out, size_t cap,
                                               const char* asset_root) {
    const char* marker;
    size_t prefix_len;
    int n;
    if (!asset_root || !asset_root[0]) return 0;
    marker = strstr(asset_root, "/resources/fight_caves/viewer");
    if (!marker
            && fc_has_prefix(asset_root, "resources/fight_caves/viewer"))
        return snprintf(out, cap, ".") > 0;
    if (!marker) return 0;
    prefix_len = (size_t)(marker - asset_root);
    n = snprintf(out, cap, "%.*s", (int)prefix_len, asset_root);
    if (!(n > 0 && (size_t)n < cap)) return 0;
    fc_trim_trailing_slash(out);
    return out[0] != '\0';
}

static int fc_repo_candidate_valid(const char* root) {
    char path[FC_ASSET_PATH_MAX];
    if (!root || !root[0]) return 0;
    return fc_join_path(path, sizeof(path), root, "ocean/fight_caves")
        && fc_dir_exists_path(path);
}

const char* fc_asset_root(void) {
    static char root[FC_ASSET_PATH_MAX];
    static int initialized;
    const char* env;
    const char* candidates[] = {
        "resources/fight_caves/viewer",
        "../resources/fight_caves/viewer",
        "../../resources/fight_caves/viewer",
        NULL
    };

    env = getenv("FC_ASSET_ROOT");
    if (!env || !env[0]) env = getenv("FC_ASSETS_PATH");
    if (env && env[0]) return env;
    if (initialized) return root;
    initialized = 1;

    if (fc_derive_asset_root(root, sizeof(root)) && fc_dir_exists_path(root))
        return root;
    for (int i = 0; candidates[i]; i++) {
        if (fc_dir_exists_path(candidates[i])) {
            fc_copy_path(root, sizeof(root), candidates[i]);
            return root;
        }
    }
    fc_copy_path(root, sizeof(root), "resources/fight_caves/viewer");
    return root;
}

const char* fc_repo_root(void) {
    static char root[FC_ASSET_PATH_MAX];
    static int initialized;
    const char* env;
    const char* asset_root;
    const char* candidates[] = { ".", "runescape-rl", "..", "../..", NULL };

    env = getenv("FC_REPO_ROOT");
    if (env && env[0]) return env;
    if (initialized) return root;
    initialized = 1;

    if (fc_derive_repo_root(root, sizeof(root)) && fc_repo_candidate_valid(root))
        return root;
    asset_root = fc_asset_root();
    if (fc_derive_repo_root_from_asset_root(root, sizeof(root), asset_root)
            && fc_repo_candidate_valid(root))
        return root;
    for (int i = 0; candidates[i]; i++) {
        if (fc_repo_candidate_valid(candidates[i])) {
            fc_copy_path(root, sizeof(root), candidates[i]);
            return root;
        }
    }
    fc_copy_path(root, sizeof(root), ".");
    return root;
}

int fc_asset_resolve_path(const char* logical_path, char* out, size_t cap) {
    const char* logical = fc_asset_logical_path(logical_path);
    const char* root;
    char joined[FC_ASSET_PATH_MAX];
    if (!out || cap == 0 || !logical_path || !logical[0]) return 0;
    if (fc_path_is_absolute(logical_path) && fc_file_exists_path(logical_path)) {
        fc_copy_path(out, cap, logical_path);
        return 1;
    }
    root = fc_asset_root();
    if (fc_join_path(joined, sizeof(joined), root, logical)
            && fc_file_exists_path(joined)) {
        fc_copy_path(out, cap, joined);
        return 1;
    }
    if (fc_file_exists_path(logical_path)) {
        fc_copy_path(out, cap, logical_path);
        return 1;
    }
    if (fc_join_path(joined, sizeof(joined), root, logical))
        fc_copy_path(out, cap, joined);
    else
        fc_copy_path(out, cap, logical);
    return 0;
}

int fc_repo_resolve_path(const char* logical_path, char* out, size_t cap) {
    char joined[FC_ASSET_PATH_MAX];
    if (!out || cap == 0 || !logical_path || !logical_path[0]) return 0;
    if (fc_path_is_absolute(logical_path) && fc_file_exists_path(logical_path)) {
        fc_copy_path(out, cap, logical_path);
        return 1;
    }
    if (fc_join_path(joined, sizeof(joined), fc_repo_root(), logical_path)
            && fc_file_exists_path(joined)) {
        fc_copy_path(out, cap, joined);
        return 1;
    }
    if (fc_file_exists_path(logical_path)) {
        fc_copy_path(out, cap, logical_path);
        return 1;
    }
    if (fc_join_path(joined, sizeof(joined), fc_repo_root(), logical_path))
        fc_copy_path(out, cap, joined);
    else
        fc_copy_path(out, cap, logical_path);
    return 0;
}

int fc_asset_exists(const char* logical_path) {
    char resolved[FC_ASSET_PATH_MAX];
    return fc_asset_resolve_path(logical_path, resolved, sizeof(resolved));
}

FILE* fc_asset_fopen(const char* logical_path, const char* mode) {
    char resolved[FC_ASSET_PATH_MAX];
    FILE* f;
    if (!fc_asset_resolve_path(logical_path, resolved, sizeof(resolved))) {
        fprintf(stderr, "fc_asset_fopen: missing %s (looked for %s)\n",
                logical_path ? logical_path : "(null)", resolved);
        return NULL;
    }
    f = fopen(resolved, mode);
    if (!f)
        fprintf(stderr, "fc_asset_fopen: cannot open %s: %s\n",
                resolved, strerror(errno));
    return f;
}

int fc_asset_close(FILE* f) {
    return f ? fclose(f) : 0;
}

unsigned char* fc_asset_read_all(const char* logical_path, size_t* out_size) {
    FILE* f = fc_asset_fopen(logical_path, "rb");
    long size;
    unsigned char* data;
    size_t got;

    if (out_size) *out_size = 0;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "%s: seek failed while reading asset\n", logical_path);
        fc_asset_close(f);
        return NULL;
    }
    size = ftell(f);
    if (size <= 0) {
        fprintf(stderr, "%s: empty or unreadable asset\n", logical_path);
        fc_asset_close(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: seek failed while reading asset\n", logical_path);
        fc_asset_close(f);
        return NULL;
    }
    data = malloc((size_t)size);
    if (!data) {
        fprintf(stderr, "%s: out of memory while reading asset\n", logical_path);
        fc_asset_close(f);
        return NULL;
    }
    got = fread(data, 1, (size_t)size, f);
    fc_asset_close(f);
    if (got != (size_t)size) {
        fprintf(stderr, "%s: short read while reading asset\n", logical_path);
        free(data);
        return NULL;
    }
    if (out_size) *out_size = (size_t)size;
    return data;
}


/* Asset Raylib */
#include <stdlib.h>
#include <string.h>

static const char* fc_asset_extension(const char* path, const char* fallback) {
    const char* dot = path ? strrchr(path, '.') : NULL;
    return dot && dot[0] ? dot : fallback;
}

Image fc_load_image_asset(const char* path) {
    Image empty = {0};
    size_t size = 0;
    unsigned char* bytes = fc_asset_read_all(path, &size);
    Image image;

    if (!bytes || size == 0) return empty;
    image = LoadImageFromMemory(fc_asset_extension(path, ".png"), bytes,
                                (int)size);
    free(bytes);
    return image;
}

Texture2D fc_load_texture_asset(const char* path) {
    Texture2D empty = {0};
    Image image = fc_load_image_asset(path);
    Texture2D texture;

    if (!image.data) return empty;
    texture = LoadTextureFromImage(image);
    UnloadImage(image);
    return texture;
}

Font fc_load_font_asset(const char* path, int font_size) {
    Font empty = {0};
    size_t size = 0;
    unsigned char* bytes = fc_asset_read_all(path, &size);
    Font font;

    if (!bytes || size == 0) return empty;
    font = LoadFontFromMemory(fc_asset_extension(path, ".ttf"), bytes,
                              (int)size, font_size, NULL, 95);
    free(bytes);
    return font.texture.id != 0 ? font : empty;
}


/* Animated Atlas */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FC_ATLAS_MAGIC 0x41544C53u
#define FC_TEXTURE_ANIM_MAGIC 0x4D4E4154u
#define FC_TEXTURE_ANIM_VERSION 1u

_Static_assert(sizeof(FcTextureAnimRow) == 16,
               "TANM rows must retain their 16-byte file layout");

static int fc_companion_path(char* out, size_t cap, const char* path,
                             const char* extension) {
    char* dot;
    int written;
    size_t offset;
    size_t remaining;
    if (!out || cap == 0 || !path || !extension) return 0;
    written = snprintf(out, cap, "%s", path);
    if (written < 0 || (size_t)written >= cap) return 0;
    dot = strrchr(out, '.');
    if (dot) {
        offset = (size_t)(dot - out);
    } else {
        offset = strlen(out);
    }
    remaining = cap - offset;
    written = snprintf(out + offset, remaining, "%s", extension);
    return written >= 0 && (size_t)written < remaining;
}

static void fc_animated_atlas_load_anims(FcAnimatedAtlas* atlas,
                                         const char* atlas_path) {
    char path[FC_ASSET_PATH_MAX];
    FILE* file;
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    FcTextureAnimRow* rows;

    if (!atlas || !atlas->base_pixels || !atlas->pixels
            || !fc_companion_path(path, sizeof(path), atlas_path, ".tanim")
            || !fc_asset_exists(path))
        return;
    file = fc_asset_fopen(path, "rb");
    if (!file) return;
    if (!fc_read_exact(file, &magic, sizeof(magic), 1, path, "tanim magic")
            || !fc_read_exact(file, &version, sizeof(version), 1, path,
                              "tanim version")
            || !fc_read_exact(file, &count, sizeof(count), 1, path,
                              "tanim count")
            || magic != FC_TEXTURE_ANIM_MAGIC
            || version != FC_TEXTURE_ANIM_VERSION) {
        fc_asset_close(file);
        return;
    }
    rows = count > 0 ? calloc(count, sizeof(*rows)) : NULL;
    if (count > 0 && !rows) {
        fc_asset_close(file);
        return;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!fc_read_exact(file, &rows[i], sizeof(rows[i]), 1, path,
                           "tanim row")) {
            free(rows);
            fc_asset_close(file);
            return;
        }
    }
    fc_asset_close(file);
    atlas->anims = rows;
    atlas->anim_count = (int)count;
    fprintf(stderr, "animated atlas: %d cells loaded from %s\n",
            atlas->anim_count, path);
}

int fc_animated_atlas_load(FcAnimatedAtlas* atlas, const char* atlas_path,
                           int enable_animation) {
    FcAnimatedAtlas loaded = {0};
    FILE* file;
    uint32_t magic = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    size_t pixel_count;
    size_t pixel_size;
    unsigned char* source_pixels;
    Image image;

    if (!atlas || !atlas_path || atlas->texture.id > 0) return 0;
    file = fc_asset_fopen(atlas_path, "rb");
    if (!file) return 0;
    if (!fc_read_exact(file, &magic, sizeof(magic), 1, atlas_path,
                       "atlas magic")
            || !fc_read_exact(file, &width, sizeof(width), 1, atlas_path,
                              "atlas width")
            || !fc_read_exact(file, &height, sizeof(height), 1, atlas_path,
                              "atlas height")
            || magic != FC_ATLAS_MAGIC || width == 0 || height == 0
            || (size_t)width > SIZE_MAX / (size_t)height) {
        fc_asset_close(file);
        return 0;
    }
    pixel_count = (size_t)width * (size_t)height;
    if (pixel_count > SIZE_MAX / 4) {
        fc_asset_close(file);
        return 0;
    }
    pixel_size = pixel_count * 4;
    source_pixels = malloc(pixel_size);
    if (!source_pixels
            || !fc_read_exact(file, source_pixels, 1, pixel_size, atlas_path,
                              "atlas pixels")) {
        free(source_pixels);
        fc_asset_close(file);
        return 0;
    }
    fc_asset_close(file);

    image = (Image) {
        .data = source_pixels,
        .width = (int)width,
        .height = (int)height,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
    };
    loaded.texture = LoadTextureFromImage(image);
    if (loaded.texture.id == 0) {
        free(source_pixels);
        return 0;
    }
    SetTextureFilter(loaded.texture, TEXTURE_FILTER_POINT);
    loaded.width = (int)width;
    loaded.height = (int)height;
    if (enable_animation) {
        loaded.base_pixels = malloc(pixel_size);
        loaded.pixels = malloc(pixel_size);
        if (loaded.base_pixels && loaded.pixels) {
            memcpy(loaded.base_pixels, source_pixels, pixel_size);
            memcpy(loaded.pixels, source_pixels, pixel_size);
        } else {
            free(loaded.base_pixels);
            free(loaded.pixels);
            loaded.base_pixels = NULL;
            loaded.pixels = NULL;
        }
    }
    free(source_pixels);
    if (enable_animation)
        fc_animated_atlas_load_anims(&loaded, atlas_path);
    *atlas = loaded;
    fprintf(stderr, "animated atlas: %ux%u loaded from %s\n",
            width, height, atlas_path);
    return 1;
}

void fc_animated_atlas_update(FcAnimatedAtlas* atlas, float dt) {
    size_t total;
    if (!atlas || !atlas->pixels || !atlas->base_pixels
            || atlas->texture.id == 0 || atlas->anim_count <= 0)
        return;

    atlas->anim_ticks += dt * 50.0f;
    total = (size_t)atlas->width * (size_t)atlas->height * 4;
    memcpy(atlas->pixels, atlas->base_pixels, total);
    for (int r = 0; r < atlas->anim_count; r++) {
        FcTextureAnimRow* row = &atlas->anims[r];
        int shift;
        if (row->w == 0 || row->h == 0
                || row->x + row->w > atlas->width
                || row->y + row->h > atlas->height
                || row->speed == 0)
            continue;
        shift = (int)(atlas->anim_ticks * (float)row->speed);
        if (row->direction == 1 || row->direction == 3) {
            int pad = row->pad;
            int center_h;
            if (pad * 2 >= row->h) pad = 0;
            center_h = row->h - pad * 2;
            shift %= center_h;
            if (row->direction == 1) shift = -shift;
            for (int y = 0; y < row->h; y++) {
                int sy = (y - pad + shift) % center_h;
                if (sy < 0) sy += center_h;
                sy += pad;
                for (int x = 0; x < row->w; x++) {
                    size_t dst = ((size_t)(row->y + y) * atlas->width
                                  + row->x + x) * 4;
                    size_t src = ((size_t)(row->y + sy) * atlas->width
                                  + row->x + x) * 4;
                    memcpy(&atlas->pixels[dst], &atlas->base_pixels[src], 4);
                }
            }
        } else if (row->direction == 2 || row->direction == 4) {
            shift %= row->w;
            if (row->direction == 2) shift = -shift;
            for (int y = 0; y < row->h; y++) {
                for (int x = 0; x < row->w; x++) {
                    int sx = (x + shift) % row->w;
                    size_t dst;
                    size_t src;
                    if (sx < 0) sx += row->w;
                    dst = ((size_t)(row->y + y) * atlas->width
                           + row->x + x) * 4;
                    src = ((size_t)(row->y + y) * atlas->width
                           + row->x + sx) * 4;
                    memcpy(&atlas->pixels[dst], &atlas->base_pixels[src], 4);
                }
            }
        }
    }
    UpdateTexture(atlas->texture, atlas->pixels);
}

void fc_animated_atlas_unload(FcAnimatedAtlas* atlas) {
    if (!atlas) return;
    if (atlas->texture.id > 0) UnloadTexture(atlas->texture);
    free(atlas->base_pixels);
    free(atlas->pixels);
    free(atlas->anims);
    *atlas = (FcAnimatedAtlas) {0};
}

#undef FC_ATLAS_MAGIC
#undef FC_TEXTURE_ANIM_MAGIC
#undef FC_TEXTURE_ANIM_VERSION

/* Anim Loader */
/**
 * @fileoverview OSRS animation runtime — loads .anims binary, applies vertex-group
 * transforms to model base geometry, re-expands into raylib mesh for rendering.
 *
 * OSRS animations use vertex-group-based transforms (not bones). Each vertex has a
 * skin label (group index). FrameBase defines transform slots with types + label arrays.
 * Each frame provides per-slot {dx,dy,dz} values. Transform types:
 *   0 = origin (compute centroid of referenced vertex groups → set pivot)
 *   1 = translate (add dx/dy/dz to all vertices in referenced groups)
 *   2 = rotate (euler Z-X-Y around pivot, raw*8 → 2048-entry sine table)
 *   3 = scale (relative to pivot, 128 = 1.0x identity)
 *   5 = alpha (face transparency, not used in our viewer)
 *
 * Binary format (.anims) produced by tools/cache_pipeline/export_animations.py:
 *   legacy header: uint32 magic ("MINA"), uint16 framebase_count,
 *                  uint16 sequence_count
 *   current header: char[4] magic ("ANM2"), uint16 version,
 *                   uint16 header_size, uint32 framebase_count,
 *                   uint32 sequence_count, uint32 sequence_frame_count,
 *                   uint32 flags
 *   framebases section, sequences section with inlined frame data.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ANIM_MAGIC 0x414E494D  /* legacy bytes "MINA" when read little-endian */
#define ANIM2_MAGIC 0x324D4E41 /* bytes "ANM2" when read little-endian */
#define ANIM2_MIN_VERSION 2
#define ANIM2_VERSION 3
#define ANIM2_HEADER_SIZE 24
#define ANIM_MAX_SLOTS 256
#define ANIM_SINE_COUNT 2048

/* ======================================================================== */
/* sine/cosine table (matches OSRS Rasterizer3D, fixed-point scale 65536)     */
/* ======================================================================== */

static int anim_sine[ANIM_SINE_COUNT];
static int anim_cosine[ANIM_SINE_COUNT];
static int anim_trig_initialized = 0;

static void anim_init_trig(void) {
    if (anim_trig_initialized) return;
    for (int i = 0; i < ANIM_SINE_COUNT; i++) {
        double angle = (double)i * (2.0 * 3.14159265358979323846 / ANIM_SINE_COUNT);
        anim_sine[i] = (int)(65536.0 * sin(angle));
        anim_cosine[i] = (int)(65536.0 * cos(angle));
    }
    anim_trig_initialized = 1;
}

/* ======================================================================== */
/* loading                                                                    */
/* ======================================================================== */

typedef struct {
    const uint8_t* p;
    size_t remaining;
    const char* path;
} AnimCursor;

static int anim_take(AnimCursor* c, void* dst, size_t size, const char* what) {
    if (!c || c->remaining < size) {
        fprintf(stderr, "%s: short read while loading %s\n",
                c && c->path ? c->path : "(anim)", what);
        return 0;
    }
    if (dst) memcpy(dst, c->p, size);
    c->p += size;
    c->remaining -= size;
    return 1;
}

static int anim_read_u8(AnimCursor* c, uint8_t* out, const char* what) {
    return anim_take(c, out, 1, what);
}

static int anim_read_u16(AnimCursor* c, uint16_t* out, const char* what) {
    uint8_t b[2];
    if (!anim_take(c, b, sizeof(b), what)) return 0;
    *out = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
    return 1;
}

static int anim_read_i16(AnimCursor* c, int16_t* out, const char* what) {
    uint16_t u = 0;
    if (!anim_read_u16(c, &u, what)) return 0;
    *out = (int16_t)u;
    return 1;
}

static int anim_read_i8(AnimCursor* c, int8_t* out, const char* what) {
    uint8_t u = 0;
    if (!anim_read_u8(c, &u, what)) return 0;
    *out = (int8_t)u;
    return 1;
}

static int anim_read_u32(AnimCursor* c, uint32_t* out, const char* what) {
    uint8_t b[4];
    if (!anim_take(c, b, sizeof(b), what)) return 0;
    *out = (uint32_t)b[0]
         | ((uint32_t)b[1] << 8)
         | ((uint32_t)b[2] << 16)
         | ((uint32_t)b[3] << 24);
    return 1;
}

AnimCache* anim_cache_load(const char* path) {
    size_t size = 0;
    uint8_t* buf = fc_asset_read_all(path, &size);
    AnimCursor cur;
    uint32_t magic = 0;
    uint32_t base_count = 0;
    uint32_t seq_count = 0;
    uint16_t format_version = 1;
    AnimCache* cache;

    if (!buf) return NULL;
    cur.p = buf;
    cur.remaining = size;
    cur.path = path;

    if (!anim_read_u32(&cur, &magic, "anim magic")) {
        free(buf);
        return NULL;
    }
    if (magic == ANIM_MAGIC) {
        uint16_t legacy_base_count = 0;
        uint16_t legacy_seq_count = 0;
        if (!anim_read_u16(&cur, &legacy_base_count, "framebase count") ||
            !anim_read_u16(&cur, &legacy_seq_count, "sequence count")) {
            free(buf);
            return NULL;
        }
        base_count = legacy_base_count;
        seq_count = legacy_seq_count;
    } else if (magic == ANIM2_MAGIC) {
        uint16_t version = 0;
        uint16_t header_size = 0;
        uint32_t sequence_frame_count = 0;
        uint32_t flags = 0;
        (void)sequence_frame_count;
        (void)flags;
        if (!anim_read_u16(&cur, &version, "anim version") ||
            !anim_read_u16(&cur, &header_size, "anim header size") ||
            !anim_read_u32(&cur, &base_count, "framebase count") ||
            !anim_read_u32(&cur, &seq_count, "sequence count") ||
            !anim_read_u32(&cur, &sequence_frame_count, "sequence frame count") ||
            !anim_read_u32(&cur, &flags, "anim flags")) {
            free(buf);
            return NULL;
        }
        if (version < ANIM2_MIN_VERSION || version > ANIM2_VERSION ||
            header_size < ANIM2_HEADER_SIZE) {
            fprintf(stderr, "anim_cache_load: unsupported ANM2 v%u header %u\n",
                    version, header_size);
            free(buf);
            return NULL;
        }
        format_version = version;
        if (header_size > ANIM2_HEADER_SIZE &&
            !anim_take(&cur, NULL, header_size - ANIM2_HEADER_SIZE,
                       "anim header extension")) {
            free(buf);
            return NULL;
        }
    } else {
        fprintf(stderr, "anim_cache_load: bad magic 0x%08X\n", magic);
        free(buf);
        return NULL;
    }

    if (base_count > 65535u || seq_count > 65535u) {
        fprintf(stderr, "anim_cache_load: unreasonable counts %u/%u\n",
                base_count, seq_count);
        free(buf);
        return NULL;
    }

    cache = (AnimCache*)calloc(1, sizeof(AnimCache));
    if (!cache) {
        free(buf);
        return NULL;
    }
    cache->base_count = (int)base_count;
    cache->seq_count = (int)seq_count;

    /* load framebases */
    cache->bases = (AnimFrameBase*)calloc(cache->base_count, sizeof(AnimFrameBase));
    cache->base_ids = (uint16_t*)malloc(cache->base_count * sizeof(uint16_t));
    if (!cache->bases || !cache->base_ids) {
        free(buf);
        anim_cache_free(cache);
        return NULL;
    }

    for (int i = 0; i < cache->base_count; i++) {
        AnimFrameBase* fb = &cache->bases[i];
        if (!anim_read_u16(&cur, &fb->base_id, "framebase id")) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }
        cache->base_ids[i] = fb->base_id;
        if (!anim_read_u8(&cur, &fb->slot_count, "framebase slot count")) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }

        fb->types = (uint8_t*)malloc(fb->slot_count);
        for (int s = 0; s < fb->slot_count; s++) {
            if (!fb->types ||
                !anim_read_u8(&cur, &fb->types[s], "framebase slot type")) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
        }

        fb->map_lengths = (uint8_t*)malloc(fb->slot_count);
        fb->frame_maps = (uint8_t**)malloc(fb->slot_count * sizeof(uint8_t*));
        if (!fb->map_lengths || !fb->frame_maps) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }
        for (int s = 0; s < fb->slot_count; s++) {
            uint8_t ml = 0;
            if (!anim_read_u8(&cur, &ml, "framebase map length")) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
            fb->map_lengths[s] = ml;
            fb->frame_maps[s] = (uint8_t*)malloc(ml);
            if (ml > 0 && !fb->frame_maps[s]) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
            for (int j = 0; j < ml; j++) {
                if (!anim_read_u8(&cur, &fb->frame_maps[s][j],
                                  "framebase map label")) {
                    free(buf);
                    anim_cache_free(cache);
                    return NULL;
                }
            }
        }
    }

    /* load sequences */
    cache->sequences = (AnimSequence*)calloc(cache->seq_count, sizeof(AnimSequence));
    if (!cache->sequences) {
        free(buf);
        anim_cache_free(cache);
        return NULL;
    }
    for (int i = 0; i < cache->seq_count; i++) {
        AnimSequence* seq = &cache->sequences[i];
        if (!anim_read_u16(&cur, &seq->seq_id, "sequence id") ||
            !anim_read_u16(&cur, &seq->frame_count, "sequence frame count")) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }

        if (!anim_read_u8(&cur, &seq->interleave_count, "sequence interleave count")) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }
        if (seq->interleave_count > 0) {
            seq->interleave_order = (uint8_t*)malloc(seq->interleave_count);
            if (!seq->interleave_order) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
            for (int j = 0; j < seq->interleave_count; j++) {
                if (!anim_read_u8(&cur, &seq->interleave_order[j],
                                  "sequence interleave slot")) {
                    free(buf);
                    anim_cache_free(cache);
                    return NULL;
                }
            }
        }

        if (format_version >= 3) {
            if (!anim_read_i16(&cur, &seq->frame_step,
                               "sequence frame step") ||
                !anim_read_i8(&cur, &seq->preanim_move,
                              "sequence pre-animation movement") ||
                !anim_read_i8(&cur, &seq->postanim_move,
                              "sequence post-animation movement") ||
                !anim_read_u8(&cur, &seq->forced_priority,
                              "sequence forced priority") ||
                !anim_read_u8(&cur, &seq->max_loops,
                              "sequence max loops") ||
                !anim_read_i8(&cur, &seq->reply_mode,
                              "sequence reply mode") ||
                !anim_read_u8(&cur, &seq->stretches,
                              "sequence stretches")) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
            seq->walk_flag = seq->postanim_move;
        } else {
            uint8_t walk_flag = 0;
            if (!anim_read_u8(&cur, &walk_flag, "sequence walk flag")) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
            seq->walk_flag = (int8_t)walk_flag;
            seq->frame_step = -1;
            seq->preanim_move = seq->interleave_count > 0 ? 2 : 0;
            seq->postanim_move = seq->walk_flag >= 0
                ? seq->walk_flag
                : (seq->interleave_count > 0 ? 2 : 0);
            seq->forced_priority = 5;
            seq->max_loops = 99;
            seq->reply_mode = -1;
            seq->stretches = 0;
        }

        seq->frames = (AnimSequenceFrame*)calloc(seq->frame_count, sizeof(AnimSequenceFrame));
        if (!seq->frames) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }
        for (int fi = 0; fi < seq->frame_count; fi++) {
            AnimSequenceFrame* sf = &seq->frames[fi];
            if (!anim_read_u16(&cur, &sf->delay, "sequence frame delay") ||
                !anim_read_u16(&cur, &sf->frame.framebase_id, "sequence framebase id") ||
                !anim_read_u8(&cur, &sf->frame.transform_count, "sequence transform count")) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }

            if (sf->frame.transform_count > 0) {
                sf->frame.transforms = (AnimTransform*)malloc(
                    sf->frame.transform_count * sizeof(AnimTransform));
                if (!sf->frame.transforms) {
                    free(buf);
                    anim_cache_free(cache);
                    return NULL;
                }
                for (int t = 0; t < sf->frame.transform_count; t++) {
                    if (!anim_read_u8(&cur, &sf->frame.transforms[t].slot_index,
                                      "sequence transform slot") ||
                        !anim_read_i16(&cur, &sf->frame.transforms[t].dx,
                                       "sequence transform dx") ||
                        !anim_read_i16(&cur, &sf->frame.transforms[t].dy,
                                       "sequence transform dy") ||
                        !anim_read_i16(&cur, &sf->frame.transforms[t].dz,
                                       "sequence transform dz")) {
                        free(buf);
                        anim_cache_free(cache);
                        return NULL;
                    }
                }
            }
        }
    }

    free(buf);
    anim_init_trig();

    fprintf(stderr, "anim_cache_load: loaded %d framebases, %d sequences from %s\n",
            cache->base_count, cache->seq_count, path);
    return cache;
}

/* ======================================================================== */
/* lookup                                                                     */
/* ======================================================================== */

AnimSequence* anim_get_sequence(AnimCache* cache, uint16_t seq_id) {
    if (!cache) return NULL;
    for (int i = 0; i < cache->seq_count; i++) {
        if (cache->sequences[i].seq_id == seq_id) {
            return &cache->sequences[i];
        }
    }
    return NULL;
}

AnimFrameBase* anim_get_framebase(AnimCache* cache, uint16_t base_id) {
    if (!cache) return NULL;
    for (int i = 0; i < cache->base_count; i++) {
        if (cache->bases[i].base_id == base_id) {
            return &cache->bases[i];
        }
    }
    return NULL;
}

/* ======================================================================== */
/* per-model animation state                                                  */
/* ======================================================================== */

AnimModelState* anim_model_state_create(
    const uint8_t* vertex_skins,
    int base_vert_count
) {
    AnimModelState* state = (AnimModelState*)calloc(1, sizeof(AnimModelState));
    state->vert_count = base_vert_count;
    state->verts = (int16_t*)calloc(base_vert_count * 3, sizeof(int16_t));

    /* build vertex group lookup from skin labels */
    state->groups = (int**)calloc(ANIM_MAX_LABELS, sizeof(int*));
    state->group_counts = (int*)calloc(ANIM_MAX_LABELS, sizeof(int));

    /* first pass: count vertices per label */
    int label_counts[ANIM_MAX_LABELS] = {0};
    for (int v = 0; v < base_vert_count; v++) {
        uint8_t label = vertex_skins[v];
        label_counts[label]++;
    }

    /* allocate per-label arrays */
    for (int l = 0; l < ANIM_MAX_LABELS; l++) {
        if (label_counts[l] > 0) {
            state->groups[l] = (int*)malloc(label_counts[l] * sizeof(int));
            state->group_counts[l] = 0;
        }
    }

    /* second pass: fill vertex indices */
    for (int v = 0; v < base_vert_count; v++) {
        uint8_t label = vertex_skins[v];
        state->groups[label][state->group_counts[label]++] = v;
    }

    return state;
}

void anim_model_state_free(AnimModelState* state) {
    if (!state) return;
    free(state->verts);
    for (int l = 0; l < ANIM_MAX_LABELS; l++) {
        free(state->groups[l]);
    }
    free(state->groups);
    free(state->group_counts);
    free(state);
}

/* ======================================================================== */
/* transform application (mirrors OSRS Model.transform)                       */
/* ======================================================================== */

void anim_apply_frame(
    AnimModelState* state,
    const int16_t* base_verts_src,
    const AnimFrameData* frame,
    const AnimFrameBase* fb
) {
    /* reset to base pose */
    memcpy(state->verts, base_verts_src, state->vert_count * 3 * sizeof(int16_t));

    /* pivot point for rotate/scale */
    int pivot_x = 0, pivot_y = 0, pivot_z = 0;

    for (int t = 0; t < frame->transform_count; t++) {
        uint8_t slot_idx = frame->transforms[t].slot_index;
        if (slot_idx >= fb->slot_count) continue;

        int type = fb->types[slot_idx];
        int dx = frame->transforms[t].dx;
        int dy = frame->transforms[t].dy;
        int dz = frame->transforms[t].dz;

        uint8_t map_len = fb->map_lengths[slot_idx];
        const uint8_t* labels = fb->frame_maps[slot_idx];

        if (type == 0) {
            /* origin: compute centroid of referenced vertex groups */
            int count = 0;
            int sum_x = 0, sum_y = 0, sum_z = 0;
            for (int m = 0; m < map_len; m++) {
                uint8_t label = labels[m];
                /* label is uint8_t, always < 256 = ANIM_MAX_LABELS */
                for (int vi = 0; vi < state->group_counts[label]; vi++) {
                    int v = state->groups[label][vi];
                    sum_x += state->verts[v * 3];
                    sum_y += state->verts[v * 3 + 1];
                    sum_z += state->verts[v * 3 + 2];
                    count++;
                }
            }
            if (count > 0) {
                pivot_x = sum_x / count + dx;
                pivot_y = sum_y / count + dy;
                pivot_z = sum_z / count + dz;
            } else {
                pivot_x = dx;
                pivot_y = dy;
                pivot_z = dz;
            }
        } else if (type == 1) {
            /* translate: add dx/dy/dz to all vertices in referenced groups */
            for (int m = 0; m < map_len; m++) {
                uint8_t label = labels[m];
                /* label is uint8_t, always < 256 = ANIM_MAX_LABELS */
                for (int vi = 0; vi < state->group_counts[label]; vi++) {
                    int v = state->groups[label][vi];
                    state->verts[v * 3]     += (int16_t)dx;
                    state->verts[v * 3 + 1] += (int16_t)dy;
                    state->verts[v * 3 + 2] += (int16_t)dz;
                }
            }
        } else if (type == 2) {
            /* rotate: euler Z-X-Y around pivot.
             * raw value * 8 → index into 2048-entry sine table.
             * rotation order: Z first, then X, then Y. */
            int ax = (dx & 0xFF) * 8;
            int ay = (dy & 0xFF) * 8;
            int az = (dz & 0xFF) * 8;

            int sin_x = anim_sine[ax & 2047];
            int cos_x = anim_cosine[ax & 2047];
            int sin_y = anim_sine[ay & 2047];
            int cos_y = anim_cosine[ay & 2047];
            int sin_z = anim_sine[az & 2047];
            int cos_z = anim_cosine[az & 2047];

            for (int m = 0; m < map_len; m++) {
                uint8_t label = labels[m];
                /* label is uint8_t, always < 256 = ANIM_MAX_LABELS */
                for (int vi = 0; vi < state->group_counts[label]; vi++) {
                    int v = state->groups[label][vi];
                    int vx = state->verts[v * 3]     - pivot_x;
                    int vy = state->verts[v * 3 + 1] - pivot_y;
                    int vz = state->verts[v * 3 + 2] - pivot_z;

                    /* Z rotation */
                    int rx = (vx * cos_z + vy * sin_z) >> 16;
                    int ry = (vy * cos_z - vx * sin_z) >> 16;
                    vx = rx; vy = ry;

                    /* X rotation */
                    ry = (vy * cos_x - vz * sin_x) >> 16;
                    int rz = (vy * sin_x + vz * cos_x) >> 16;
                    vy = ry; vz = rz;

                    /* Y rotation */
                    rx = (vz * sin_y + vx * cos_y) >> 16;
                    rz = (vz * cos_y - vx * sin_y) >> 16;
                    vx = rx; vz = rz;

                    state->verts[v * 3]     = (int16_t)(vx + pivot_x);
                    state->verts[v * 3 + 1] = (int16_t)(vy + pivot_y);
                    state->verts[v * 3 + 2] = (int16_t)(vz + pivot_z);
                }
            }
        } else if (type == 3) {
            /* scale: relative to pivot, 128 = 1.0x identity */
            for (int m = 0; m < map_len; m++) {
                uint8_t label = labels[m];
                /* label is uint8_t, always < 256 = ANIM_MAX_LABELS */
                for (int vi = 0; vi < state->group_counts[label]; vi++) {
                    int v = state->groups[label][vi];
                    int vx = state->verts[v * 3]     - pivot_x;
                    int vy = state->verts[v * 3 + 1] - pivot_y;
                    int vz = state->verts[v * 3 + 2] - pivot_z;

                    vx = (vx * dx) / 128;
                    vy = (vy * dy) / 128;
                    vz = (vz * dz) / 128;

                    state->verts[v * 3]     = (int16_t)(vx + pivot_x);
                    state->verts[v * 3 + 1] = (int16_t)(vy + pivot_y);
                    state->verts[v * 3 + 2] = (int16_t)(vz + pivot_z);
                }
            }
        }
        /* type 5 (alpha) skipped — we don't use face transparency in the viewer */
    }
}

/* ======================================================================== */
/* two-track interleaved animation (matches OSRS Model.applyAnimationFrames)  */
/* ======================================================================== */

/**
 * Apply a single transform slot to the vertex state (extracted from anim_apply_frame
 * to allow per-slot interleave filtering).
 *
 * pivot_x/y/z are read/written through pointers — they persist across slots
 * within a pass, exactly like the reference's transformTempX/Y/Z.
 */
static void anim_apply_single_transform(
    AnimModelState* state,
    int type, const uint8_t* labels, uint8_t map_len,
    int dx, int dy, int dz,
    int* pivot_x, int* pivot_y, int* pivot_z
) {
    if (type == 0) {
        /* origin: compute centroid of referenced vertex groups */
        int count = 0, sx = 0, sy = 0, sz = 0;
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                sx += state->verts[v * 3];
                sy += state->verts[v * 3 + 1];
                sz += state->verts[v * 3 + 2];
                count++;
            }
        }
        if (count > 0) {
            *pivot_x = sx / count + dx;
            *pivot_y = sy / count + dy;
            *pivot_z = sz / count + dz;
        } else {
            *pivot_x = dx;
            *pivot_y = dy;
            *pivot_z = dz;
        }
    } else if (type == 1) {
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                state->verts[v * 3]     += (int16_t)dx;
                state->verts[v * 3 + 1] += (int16_t)dy;
                state->verts[v * 3 + 2] += (int16_t)dz;
            }
        }
    } else if (type == 2) {
        int ax = (dx & 0xFF) * 8, ay = (dy & 0xFF) * 8, az = (dz & 0xFF) * 8;
        int sin_x = anim_sine[ax & 2047], cos_x = anim_cosine[ax & 2047];
        int sin_y = anim_sine[ay & 2047], cos_y = anim_cosine[ay & 2047];
        int sin_z = anim_sine[az & 2047], cos_z = anim_cosine[az & 2047];
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                int vx = state->verts[v * 3]     - *pivot_x;
                int vy = state->verts[v * 3 + 1] - *pivot_y;
                int vz = state->verts[v * 3 + 2] - *pivot_z;
                int rx = (vx * cos_z + vy * sin_z) >> 16;
                int ry = (vy * cos_z - vx * sin_z) >> 16;
                vx = rx; vy = ry;
                ry = (vy * cos_x - vz * sin_x) >> 16;
                int rz = (vy * sin_x + vz * cos_x) >> 16;
                vy = ry; vz = rz;
                rx = (vz * sin_y + vx * cos_y) >> 16;
                rz = (vz * cos_y - vx * sin_y) >> 16;
                state->verts[v * 3]     = (int16_t)(rx + *pivot_x);
                state->verts[v * 3 + 1] = (int16_t)(vy + *pivot_y);
                state->verts[v * 3 + 2] = (int16_t)(rz + *pivot_z);
            }
        }
    } else if (type == 3) {
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                int vx = state->verts[v * 3]     - *pivot_x;
                int vy = state->verts[v * 3 + 1] - *pivot_y;
                int vz = state->verts[v * 3 + 2] - *pivot_z;
                state->verts[v * 3]     = (int16_t)((vx * dx) / 128 + *pivot_x);
                state->verts[v * 3 + 1] = (int16_t)((vy * dy) / 128 + *pivot_y);
                state->verts[v * 3 + 2] = (int16_t)((vz * dz) / 128 + *pivot_z);
            }
        }
    }
}

/**
 * Apply two animation frames with body-part interleaving.
 *
 * Mirrors OSRS Model.applyAnimationFrames():
 *   - interleave_order lists framebase SLOT INDICES owned by SECONDARY (walk)
 *   - Pass 1: apply primary transforms for slots NOT in interleave_order
 *   - Pass 2: apply secondary transforms for slots IN interleave_order
 *   - Type-0 (pivot) transforms always execute in both passes
 *
 * CRITICAL: interleave_order contains framebase SLOT INDICES, not vertex labels!
 * The reference code (Model.java:1322-1343) walks both the frame's slot list and
 * the interleave_order simultaneously, comparing slot indices directly.
 *
 * Both passes operate on the same vertex state with independent pivot tracking,
 * exactly as the reference does with transformTempX/Y/Z reset between passes.
 */
static void anim_apply_frame_interleaved(
    AnimModelState* state,
    const int16_t* base_verts_src,
    const AnimFrameData* secondary_frame, const AnimFrameBase* secondary_fb,
    const AnimFrameData* primary_frame, const AnimFrameBase* primary_fb,
    const uint8_t* interleave_order, int interleave_count
) {
    /* reset to base pose */
    memcpy(state->verts, base_verts_src, state->vert_count * 3 * sizeof(int16_t));

    /* build boolean mask: interleave_order lists SLOT INDICES the SECONDARY owns.
       index by slot index (0-244 for our 245-slot framebase), NOT vertex labels. */
    uint8_t secondary_slot[256];
    memset(secondary_slot, 0, sizeof(secondary_slot));
    for (int i = 0; i < interleave_count; i++) {
        secondary_slot[interleave_order[i]] = 1;
    }

    /* pass 1: primary frame — apply transforms for slots NOT in interleave_order.
     * type-0 (pivot) always executes regardless of ownership.
     * matches reference: if (k1 != i1 || class18.types[k1] == 0) */
    int pivot_x = 0, pivot_y = 0, pivot_z = 0;
    for (int t = 0; t < primary_frame->transform_count; t++) {
        uint8_t slot_idx = primary_frame->transforms[t].slot_index;
        if (slot_idx >= primary_fb->slot_count) continue;

        int type = primary_fb->types[slot_idx];
        int in_interleave = secondary_slot[slot_idx];

        if (!in_interleave || type == 0) {
            anim_apply_single_transform(
                state, type,
                primary_fb->frame_maps[slot_idx],
                primary_fb->map_lengths[slot_idx],
                primary_frame->transforms[t].dx,
                primary_frame->transforms[t].dy,
                primary_frame->transforms[t].dz,
                &pivot_x, &pivot_y, &pivot_z);
        }
    }

    /* pass 2: secondary frame — apply transforms for slots IN interleave_order.
     * type-0 (pivot) always executes.
     * matches reference: if (i2 == i1 || class18.types[i2] == 0) */
    pivot_x = 0; pivot_y = 0; pivot_z = 0;
    for (int t = 0; t < secondary_frame->transform_count; t++) {
        uint8_t slot_idx = secondary_frame->transforms[t].slot_index;
        if (slot_idx >= secondary_fb->slot_count) continue;

        int type = secondary_fb->types[slot_idx];
        int in_interleave = secondary_slot[slot_idx];

        if (in_interleave || type == 0) {
            anim_apply_single_transform(
                state, type,
                secondary_fb->frame_maps[slot_idx],
                secondary_fb->map_lengths[slot_idx],
                secondary_frame->transforms[t].dx,
                secondary_frame->transforms[t].dy,
                secondary_frame->transforms[t].dz,
                &pivot_x, &pivot_y, &pivot_z);
        }
    }
}

/* Apply the current pose/action pair to one model state. The action owns the
 * full model unless its sequence supplies an OSRS interleave table, in which
 * case the pose supplies the interleaved transform slots. If the action frame
 * cannot be applied, fall back to the pose exactly as a single-track actor
 * would. Mesh upload remains the caller's responsibility because player
 * models are unique while same-type NPCs share one render mesh. */
int anim_mix_pose_action(
    AnimCache* cache,
    AnimModelState* state,
    const int16_t* base_verts,
    AnimSequence* pose,
    int pose_frame_index,
    AnimSequence* action,
    int action_frame_index
) {
    if (!cache || !state || !base_verts) return 0;

    if (action && action_frame_index >= 0 &&
        action_frame_index < action->frame_count) {
        AnimFrameData* action_frame =
            &action->frames[action_frame_index].frame;
        AnimFrameBase* action_base =
            anim_get_framebase(cache, action_frame->framebase_id);
        if (action_base && pose && pose_frame_index >= 0 &&
            pose_frame_index < pose->frame_count &&
            action->interleave_count > 0 && action->interleave_order) {
            AnimFrameData* pose_frame =
                &pose->frames[pose_frame_index].frame;
            AnimFrameBase* pose_base =
                anim_get_framebase(cache, pose_frame->framebase_id);
            if (pose_base) {
                anim_apply_frame_interleaved(
                    state, base_verts,
                    pose_frame, pose_base, action_frame, action_base,
                    action->interleave_order, action->interleave_count);
                return 1;
            }
        } else if (action_base) {
            anim_apply_frame(state, base_verts, action_frame, action_base);
            return 1;
        }
    }

    if (pose && pose_frame_index >= 0 &&
        pose_frame_index < pose->frame_count) {
        AnimFrameData* pose_frame = &pose->frames[pose_frame_index].frame;
        AnimFrameBase* pose_base =
            anim_get_framebase(cache, pose_frame->framebase_id);
        if (pose_base) {
            anim_apply_frame(state, base_verts, pose_frame, pose_base);
            return 1;
        }
    }
    return 0;
}

/* ======================================================================== */
/* mesh re-expansion (apply animated base verts → expanded rendering verts)   */
/* ======================================================================== */

/**
 * Re-expand animated base vertices into the raylib mesh's expanded vertex buffer.
 * This mirrors expand_model from the Python exporter but in-place, using
 * face_indices to map from base to expanded vertices.
 *
 * The mesh has face_count*3 expanded vertices. Each triplet (i*3, i*3+1, i*3+2)
 * corresponds to face_indices[i*3], face_indices[i*3+1], face_indices[i*3+2]
 * pointing into base_vertices.
 *
 * OSRS Y is negated for rendering (negative-up → positive-up).
 */
void anim_update_mesh(
    float* mesh_vertices,
    const AnimModelState* state,
    const uint16_t* face_indices,
    int face_count
) {
    for (int fi = 0; fi < face_count; fi++) {
        int a = face_indices[fi * 3];
        int b = face_indices[fi * 3 + 1];
        int c = face_indices[fi * 3 + 2];

        int vi = fi * 9; /* 3 verts * 3 coords */
        mesh_vertices[vi]     = (float)state->verts[a * 3];
        mesh_vertices[vi + 1] = (float)(-state->verts[a * 3 + 1]); /* negate Y */
        mesh_vertices[vi + 2] = (float)state->verts[a * 3 + 2];

        mesh_vertices[vi + 3] = (float)state->verts[b * 3];
        mesh_vertices[vi + 4] = (float)(-state->verts[b * 3 + 1]);
        mesh_vertices[vi + 5] = (float)state->verts[b * 3 + 2];

        mesh_vertices[vi + 6] = (float)state->verts[c * 3];
        mesh_vertices[vi + 7] = (float)(-state->verts[c * 3 + 1]);
        mesh_vertices[vi + 8] = (float)state->verts[c * 3 + 2];
    }
}

/* ======================================================================== */
/* cleanup                                                                    */
/* ======================================================================== */

void anim_cache_free(AnimCache* cache) {
    if (!cache) return;

    for (int i = 0; i < cache->base_count; i++) {
        AnimFrameBase* fb = &cache->bases[i];
        free(fb->types);
        free(fb->map_lengths);
        for (int s = 0; s < fb->slot_count; s++) {
            free(fb->frame_maps[s]);
        }
        free(fb->frame_maps);
    }
    free(cache->bases);
    free(cache->base_ids);

    for (int i = 0; i < cache->seq_count; i++) {
        AnimSequence* seq = &cache->sequences[i];
        free(seq->interleave_order);
        for (int fi = 0; fi < seq->frame_count; fi++) {
            free(seq->frames[fi].frame.transforms);
        }
        free(seq->frames);
    }
    free(cache->sequences);
    free(cache);
}

#undef ANIM_MAGIC
#undef ANIM2_MAGIC
#undef ANIM2_MIN_VERSION
#undef ANIM2_VERSION
#undef ANIM2_HEADER_SIZE
#undef ANIM_MAX_SLOTS
#undef ANIM_SINE_COUNT

/* Models */
// Loads models from .models MDL2/MDL3 binary for Raylib rendering.
// Fight Caves raylib model loader.

#include "raylib.h"
#include "raymath.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MDL2_MAGIC 0x4D444C32
#define MDL3_MAGIC 0x4D444C33
#define MUV1_MAGIC 0x3156554D
#define MODEL_ID_INDEX_MAX 20000

static int model_id_filter_contains(const uint32_t *ids, int id_count, uint32_t id) {
    if (!ids) return 1;
    if (id_count <= 0) return 0;
    for (int i = 0; i < id_count; i++)
        if (ids[i] == id) return 1;
    return 0;
}

ModelEntry *model_find(ModelSet *set, uint32_t id) {
    if (!set) return NULL;
    if (id < (uint32_t)set->index_limit && set->index_by_id) {
        int idx = set->index_by_id[id];
        if (idx >= 0 && idx < set->count) return &set->entries[idx];
    }
    for (int i = 0; i < set->count; i++)
        if (set->entries[i].model_id == id && set->entries[i].loaded) return &set->entries[i];
    return NULL;
}

float models_pick_depth(const ModelEntry *entry, const int16_t *posed_vertices,
                         Vector3 position, float yaw_degrees, Camera3D camera,
                         Vector2 mouse, int screen_width, int screen_height) {
    if (!entry || !entry->loaded || !entry->rest_verts) return -1;
    /* Same model -> yaw -> translation order as DrawModelEx. Never read the
     * shared uploaded mesh: another NPC of this type may have a different pose. */
    Matrix transform = MatrixMultiply(entry->model.transform,
        MatrixMultiply(MatrixRotateY(yaw_degrees * DEG2RAD),
                       MatrixTranslate(position.x, position.y, position.z)));
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    float nearest = FLT_MAX;
    for (int face = 0; face < entry->face_count; face++) {
        float min_x = FLT_MAX, min_y = FLT_MAX;
        float max_x = -FLT_MAX, max_y = -FLT_MAX, depth = 0;
        int visible = 1;
        for (int corner = 0; corner < 3; corner++) {
            int offset = (face * 3 + corner) * 3;
            Vector3 local;
            if (posed_vertices && entry->face_indices) {
                int vertex = entry->face_indices[face * 3 + corner] * 3;
                local = (Vector3){posed_vertices[vertex] / 128.0f,
                    -posed_vertices[vertex + 1] / 128.0f,
                    -posed_vertices[vertex + 2] / 128.0f};
            } else {
                local = (Vector3){entry->rest_verts[offset], entry->rest_verts[offset + 1],
                                   entry->rest_verts[offset + 2]};
            }
            Vector3 world = Vector3Transform(local, transform);
            float z = Vector3DotProduct(Vector3Subtract(world, camera.position), forward);
            if (z <= 0.01f) { visible = 0; break; }
            Vector2 screen = GetWorldToScreenEx(world, camera, screen_width, screen_height);
            min_x = fminf(min_x, screen.x); max_x = fmaxf(max_x, screen.x);
            min_y = fminf(min_y, screen.y); max_y = fmaxf(max_y, screen.y);
            depth += z / 3.0f;
        }
        /* RuneLite Perspective.calculate2DBounds uses projected face rectangles
         * padded by five pixels, not a ground-tile test or pixel-perfect triangles. */
        if (visible && mouse.x >= min_x - 5 && mouse.x <= max_x + 5 &&
            mouse.y >= min_y - 5 && mouse.y <= max_y + 5 && depth < nearest)
            nearest = depth;
    }
    return nearest == FLT_MAX ? -1 : nearest;
}

static float model_clamp_uv(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static float model_repeat_uv_with_margin(float v, float margin) {
    if (margin <= 0.0f)
        return model_clamp_uv(v);
    while (v < -margin) v += 1.0f;
    while (v > 1.0f + margin) v -= 1.0f;
    if (v < -margin) return -margin;
    if (v > 1.0f + margin) return 1.0f + margin;
    return v;
}

static void model_project_uvs_for_face(const int16_t *verts,
                                       int tri_a, int tri_b, int tri_c,
                                       int tex_a, int tex_b, int tex_c,
                                       float *u, float *v) {
    u[0] = 0.0f; u[1] = 1.0f; u[2] = 0.0f;
    v[0] = 0.0f; v[1] = 0.0f; v[2] = 1.0f;

    float v1x = (float)verts[tex_a * 3];
    float v1y = (float)verts[tex_a * 3 + 1];
    float v1z = (float)verts[tex_a * 3 + 2];
    float v2x = (float)verts[tex_b * 3] - v1x;
    float v2y = (float)verts[tex_b * 3 + 1] - v1y;
    float v2z = (float)verts[tex_b * 3 + 2] - v1z;
    float v3x = (float)verts[tex_c * 3] - v1x;
    float v3y = (float)verts[tex_c * 3 + 1] - v1y;
    float v3z = (float)verts[tex_c * 3 + 2] - v1z;
    float v4x = (float)verts[tri_a * 3] - v1x;
    float v4y = (float)verts[tri_a * 3 + 1] - v1y;
    float v4z = (float)verts[tri_a * 3 + 2] - v1z;
    float v5x = (float)verts[tri_b * 3] - v1x;
    float v5y = (float)verts[tri_b * 3 + 1] - v1y;
    float v5z = (float)verts[tri_b * 3 + 2] - v1z;
    float v6x = (float)verts[tri_c * 3] - v1x;
    float v6y = (float)verts[tri_c * 3 + 1] - v1y;
    float v6z = (float)verts[tri_c * 3 + 2] - v1z;

    float v7x = v2y * v3z - v2z * v3y;
    float v7y = v2z * v3x - v2x * v3z;
    float v7z = v2x * v3y - v2y * v3x;

    float v8x = v3y * v7z - v3z * v7y;
    float v8y = v3z * v7x - v3x * v7z;
    float v8z = v3x * v7y - v3y * v7x;
    float denom = v8x * v2x + v8y * v2y + v8z * v2z;
    if (fabsf(denom) < 1.0e-6f)
        return;
    float inv = 1.0f / denom;
    u[0] = (v8x * v4x + v8y * v4y + v8z * v4z) * inv;
    u[1] = (v8x * v5x + v8y * v5y + v8z * v5z) * inv;
    u[2] = (v8x * v6x + v8y * v6y + v8z * v6z) * inv;

    v8x = v2y * v7z - v2z * v7y;
    v8y = v2z * v7x - v2x * v7z;
    v8z = v2x * v7y - v2y * v7x;
    denom = v8x * v3x + v8y * v3y + v8z * v3z;
    if (fabsf(denom) < 1.0e-6f)
        return;
    inv = 1.0f / denom;
    v[0] = (v8x * v4x + v8y * v4y + v8z * v4z) * inv;
    v[1] = (v8x * v5x + v8y * v5y + v8z * v5z) * inv;
    v[2] = (v8x * v6x + v8y * v6y + v8z * v6z) * inv;
}

void models_recompute_texture_uvs_from_vertices(ModelEntry *entry,
                                                const int16_t *verts) {
    if (!entry || !entry->loaded || !entry->face_uvs || !verts)
        return;
    Mesh *mesh = &entry->model.meshes[0];
    if (!mesh->texcoords || !entry->face_indices)
        return;
    for (int fi = 0; fi < entry->face_count; fi++) {
        ModelFaceUvInfo *info = &entry->face_uvs[fi];
        if (!info->textured)
            continue;
        int tri_a = entry->face_indices[fi * 3];
        int tri_b = entry->face_indices[fi * 3 + 1];
        int tri_c = entry->face_indices[fi * 3 + 2];
        if (tri_a >= entry->base_vert_count || tri_b >= entry->base_vert_count
                || tri_c >= entry->base_vert_count
                || info->tex_a >= entry->base_vert_count
                || info->tex_b >= entry->base_vert_count
                || info->tex_c >= entry->base_vert_count)
            continue;
        float u[3], v[3];
        model_project_uvs_for_face(verts, tri_a, tri_b, tri_c,
                                   info->tex_a, info->tex_b, info->tex_c,
                                   u, v);
        for (int j = 0; j < 3; j++) {
            float cu = model_clamp_uv(u[j]);
            float cv = model_repeat_uv_with_margin(v[j],
                                                   info->repeat_v_margin);
            int out = (fi * 3 + j) * 2;
            mesh->texcoords[out] = info->u_base + cu * info->u_scale;
            mesh->texcoords[out + 1] = info->v_base + cv * info->v_scale;
        }
    }
    UpdateMeshBuffer(*mesh, 1, mesh->texcoords,
                     mesh->vertexCount * 2 * sizeof(float), 0);
}

static ModelSet *models_load_filtered(const char *path, const uint32_t *ids,
                                      int id_count, Texture2D atlas_texture) {
    FILE *f = fc_asset_fopen(path, "rb");
    if (!f) { fprintf(stderr, "models: can't open %s\n", path); return NULL; }

    uint32_t magic, count;
    if (!fc_read_exact(f, &magic, sizeof(magic), 1, path, "model magic")
            || (magic != MDL2_MAGIC && magic != MDL3_MAGIC)) {
        fprintf(stderr, "models: bad magic\n");
        fc_asset_close(f);
        return NULL;
    }
    int has_tex = (magic == MDL3_MAGIC);
    if (!fc_read_exact(f, &count, sizeof(count), 1, path, "model count")) {
        fc_asset_close(f);
        return NULL;
    }
    uint32_t *offsets = malloc(count * 4);
    if (!offsets
            || !fc_read_exact(f, offsets, sizeof(offsets[0]), count, path, "model offsets")) {
        free(offsets);
        fc_asset_close(f);
        return NULL;
    }

    ModelSet *set = calloc(1, sizeof(ModelSet));
    if (!set) {
        free(offsets);
        fc_asset_close(f);
        return NULL;
    }
    set->entries = calloc(count, sizeof(ModelEntry));
    set->index_limit = MODEL_ID_INDEX_MAX;
    set->index_by_id = malloc(sizeof(int) * set->index_limit);
    if (!set->entries || !set->index_by_id) {
        free(offsets);
        fc_asset_close(f);
        models_free(set);
        return NULL;
    }
    for (int i = 0; i < set->index_limit; i++) set->index_by_id[i] = -1;
    set->count = (int)count;
    set->has_textures = has_tex;

    if (has_tex && atlas_texture.id == 0) {
        fprintf(stderr, "models: shared atlas unavailable for %s\n", path);
        free(offsets);
        fc_asset_close(f);
        models_free(set);
        return NULL;
    }

    long model_file_end = 0;
    long model_file_pos = ftell(f);
    if (model_file_pos >= 0 && fseek(f, 0, SEEK_END) == 0) {
        model_file_end = ftell(f);
        fseek(f, model_file_pos, SEEK_SET);
    }

    int loaded_count = 0;
    for (uint32_t m = 0; m < count; m++) {
        if (!fc_seek(f, offsets[m], SEEK_SET, path, "model offset table")) {
            free(offsets);
            fc_asset_close(f);
            models_free(set);
            return NULL;
        }
        uint32_t mid; uint16_t evc, fc, bvc;
        if (!fc_read_exact(f, &mid, sizeof(mid), 1, path, "model id")
                || !fc_read_exact(f, &evc, sizeof(evc), 1, path, "model expanded vertex count")
                || !fc_read_exact(f, &fc, sizeof(fc), 1, path, "model face count")
                || !fc_read_exact(f, &bvc, sizeof(bvc), 1, path, "model base vertex count")) {
            free(offsets);
            fc_asset_close(f);
            models_free(set);
            return NULL;
        }
        if (!model_id_filter_contains(ids, id_count, mid)) continue;

        int vc = (int)evc, tc = (int)fc;
        float *verts = malloc(vc * 3 * sizeof(float));
        if (!verts
                || !fc_read_exact(f, verts, sizeof(float), vc * 3, path, "model vertices")) {
            free(verts);
            free(offsets);
            fc_asset_close(f);
            models_free(set);
            return NULL;
        }
        unsigned char *colors = malloc(vc * 4);
        if (!colors
                || !fc_read_exact(f, colors, sizeof(unsigned char), vc * 4, path, "model colors")) {
            free(verts);
            free(colors);
            free(offsets);
            fc_asset_close(f);
            models_free(set);
            return NULL;
        }
        float *texcoords = NULL;
        if (has_tex) {
            texcoords = malloc(vc * 2 * sizeof(float));
            if (!texcoords
                    || !fc_read_exact(f, texcoords, sizeof(float), vc * 2, path, "model texcoords")) {
                free(verts);
                free(colors);
                free(texcoords);
                free(offsets);
                fc_asset_close(f);
                models_free(set);
                return NULL;
            }
        }

        // OSRS units -> tile units, flip Z for Raylib
        for (int i = 0; i < vc; i++) {
            verts[i*3]   /=  128.0f;
            verts[i*3+1] /=  128.0f;
            verts[i*3+2] /= -128.0f;
        }
        float *rest_verts = malloc(vc * 3 * sizeof(float));
        if (!rest_verts) {
            free(verts);
            free(colors);
            free(offsets);
            fc_asset_close(f);
            models_free(set);
            return NULL;
        }
        memcpy(rest_verts, verts, vc * 3 * sizeof(float));

        Mesh mesh = {0};
        mesh.vertexCount = vc;
        mesh.triangleCount = tc;
        mesh.vertices = verts;
        mesh.colors = colors;
        mesh.texcoords = texcoords;
        mesh.normals = calloc(vc * 3, sizeof(float));
        for (int i = 0; i < tc; i++) {
            int i0 = i*3, i1 = i*3+1, i2 = i*3+2;
            float ax = verts[i1*3]-verts[i0*3], ay = verts[i1*3+1]-verts[i0*3+1], az = verts[i1*3+2]-verts[i0*3+2];
            float bx = verts[i2*3]-verts[i0*3], by = verts[i2*3+1]-verts[i0*3+1], bz = verts[i2*3+2]-verts[i0*3+2];
            float nx = ay*bz-az*by, ny = az*bx-ax*bz, nz = ax*by-ay*bx;
            float len = sqrtf(nx*nx+ny*ny+nz*nz);
            if (len > 1e-4f) { nx/=len; ny/=len; nz/=len; }
            for (int j = 0; j < 3; j++) {
                mesh.normals[(i*3+j)*3] = nx; mesh.normals[(i*3+j)*3+1] = ny; mesh.normals[(i*3+j)*3+2] = nz;
            }
        }
        UploadMesh(&mesh, false);

        // Animation data
        int16_t *bv = malloc(bvc * 3 * sizeof(int16_t));
        if (!bv
                || !fc_read_exact(f, bv, sizeof(int16_t), bvc * 3, path, "model base vertices")) {
            free(offsets);
            fc_asset_close(f);
            models_free(set);
            return NULL;
        }
        uint8_t *skins = malloc(bvc);
        if (!skins
                || !fc_read_exact(f, skins, sizeof(uint8_t), bvc, path, "model vertex skins")) {
            free(offsets);
            fc_asset_close(f);
            models_free(set);
            return NULL;
        }
        uint16_t *fi = malloc(tc * 3 * sizeof(uint16_t));
        if (!fi
                || !fc_read_exact(f, fi, sizeof(uint16_t), tc * 3, path, "model face indices")) {
            free(offsets);
            fc_asset_close(f);
            models_free(set);
            return NULL;
        }
        uint8_t *pri = malloc(tc);
        if (!pri
                || !fc_read_exact(f, pri, sizeof(uint8_t), tc, path, "model priorities")) {
            free(offsets);
            fc_asset_close(f);
            models_free(set);
            return NULL;
        }

        ModelFaceUvInfo *face_uvs = NULL;
        long next_off = (m + 1 < count) ? (long)offsets[m + 1] : model_file_end;
        long opt_pos = ftell(f);
        if (has_tex && opt_pos >= 0 && next_off >= opt_pos + 8) {
            uint32_t uv_magic = 0;
            uint32_t uv_count = 0;
            if (fc_read_exact(f, &uv_magic, sizeof(uv_magic), 1, path,
                              "model uv magic")
                    && fc_read_exact(f, &uv_count, sizeof(uv_count), 1, path,
                                     "model uv count")
                    && uv_magic == MUV1_MAGIC && uv_count == (uint32_t)tc) {
                face_uvs = calloc(tc, sizeof(*face_uvs));
                if (!face_uvs) {
                    free(offsets);
                    fc_asset_close(f);
                    models_free(set);
                    return NULL;
                }
                for (int i = 0; i < tc; i++) {
                    uint8_t textured = 0, pad[3];
                    if (!fc_read_exact(f, &textured, sizeof(textured), 1,
                                       path, "model uv textured")
                            || !fc_read_exact(f, pad, sizeof(pad), 1,
                                              path, "model uv pad")
                            || !fc_read_exact(f, &face_uvs[i].tex_a,
                                              sizeof(face_uvs[i].tex_a), 1,
                                              path, "model uv tex a")
                            || !fc_read_exact(f, &face_uvs[i].tex_b,
                                              sizeof(face_uvs[i].tex_b), 1,
                                              path, "model uv tex b")
                            || !fc_read_exact(f, &face_uvs[i].tex_c,
                                              sizeof(face_uvs[i].tex_c), 1,
                                              path, "model uv tex c")
                            || !fc_read_exact(f, &face_uvs[i].u_base,
                                              sizeof(face_uvs[i].u_base), 1,
                                              path, "model uv u base")
                            || !fc_read_exact(f, &face_uvs[i].v_base,
                                              sizeof(face_uvs[i].v_base), 1,
                                              path, "model uv v base")
                            || !fc_read_exact(f, &face_uvs[i].u_scale,
                                              sizeof(face_uvs[i].u_scale), 1,
                                              path, "model uv u scale")
                            || !fc_read_exact(f, &face_uvs[i].v_scale,
                                              sizeof(face_uvs[i].v_scale), 1,
                                              path, "model uv v scale")
                            || !fc_read_exact(f,
                                              &face_uvs[i].repeat_v_margin,
                                              sizeof(face_uvs[i].repeat_v_margin),
                                              1, path, "model uv repeat v")) {
                        free(face_uvs);
                        free(offsets);
                        fc_asset_close(f);
                        models_free(set);
                        return NULL;
                    }
                    face_uvs[i].textured = textured != 0;
                }
            } else {
                fseek(f, opt_pos, SEEK_SET);
            }
        }

        Model ray_model = LoadModelFromMesh(mesh);
        if (has_tex)
            ray_model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture =
                atlas_texture;

        set->entries[m] = (ModelEntry){
            .model_id = mid, .model = ray_model, .loaded = 1,
            .rest_verts = rest_verts,
            .rest_texcoords = texcoords && vc > 0
                ? malloc((size_t)vc * 2 * sizeof(float)) : NULL,
            .base_verts = bv, .vertex_skins = skins, .face_indices = fi,
            .face_priorities = pri, .face_uvs = face_uvs,
            .base_vert_count = (int)bvc, .face_count = tc,
        };
        if (texcoords && set->entries[m].rest_texcoords)
            memcpy(set->entries[m].rest_texcoords, texcoords,
                   (size_t)vc * 2 * sizeof(float));
        if (mid < (uint32_t)set->index_limit) set->index_by_id[mid] = (int)m;
        loaded_count++;
        fprintf(stderr, "  model %u: %d tris, %d base verts\n", mid, tc, (int)bvc);
    }
    free(offsets); fc_asset_close(f);
    set->loaded = 1;
    fprintf(stderr, "models: loaded %d from %s\n", loaded_count, path);
    return set;
}

ModelSet *models_load(const char *path, Texture2D atlas_texture) {
    return models_load_filtered(path, NULL, 0, atlas_texture);
}

void models_free(ModelSet *set) {
    if (!set) return;
    for (int i = 0; i < set->count; i++) {
        if (set->entries[i].loaded) {
            UnloadModel(set->entries[i].model);
        }
        free(set->entries[i].base_verts);
        free(set->entries[i].rest_verts);
        free(set->entries[i].rest_texcoords);
        free(set->entries[i].vertex_skins);
        free(set->entries[i].face_indices);
        free(set->entries[i].face_priorities);
        free(set->entries[i].face_uvs);
    }
    free(set->entries);
    free(set->index_by_id);
    free(set);
}

#undef MDL2_MAGIC
#undef MDL3_MAGIC
#undef MUV1_MAGIC
#undef MODEL_ID_INDEX_MAX

/* Npc Models */
/*
 * Fight Caves model compatibility wrapper.
 *
 * The viewer historically used NpcModelSet/NpcModelEntry names for every
 * runtime model file. Keep those names while routing all NPC/player/projectile
 * files through the generalized MDL2/MDL3 model loader.
 */

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


/* Model Animation */
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


/* Objects Loader */
/**
 * @fileoverview Loads placed map objects from .objects binary into a single raylib Model.
 *
 * Supports two binary formats:
 *   v1 (OBJS): vertices + colors only (flat vertex coloring)
 *   v2 (OBJ2): vertices + colors + texcoords (texture atlas support)
 *
 * When v2 format is detected, also loads the companion .atlas file (raw RGBA)
 * and assigns it as the model's diffuse texture. Vertex colors are multiplied
 * by the texture sample: textured faces use white vertex color + real texture,
 * non-textured faces use HSL vertex color + white atlas pixel.
 */

#include "raylib.h"
#include "rlgl.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OBJS_MAGIC 0x4F424A53  /* "OBJS" v1 */
#define OBJ2_MAGIC 0x4F424A32  /* "OBJ2" v2 with texcoords */
#define OANM_MAGIC 0x4D4E414F  /* "OANM" animated object placements */
#define OANM_VERSION 1
ObjectMesh* objects_load(const char* path) {
    FILE* f = fc_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "objects_load: could not open %s\n", path);
        return NULL;
    }

    uint32_t magic, placement_count, total_verts;
    int32_t min_wx, min_wy;
    if (!fc_read_exact(f, &magic, sizeof(magic), 1, path, "object magic")) {
        fc_asset_close(f);
        return NULL;
    }

    int has_textures = 0;
    if (magic == OBJ2_MAGIC) {
        has_textures = 1;
    } else if (magic != OBJS_MAGIC) {
        fprintf(stderr, "objects_load: bad magic %08x\n", magic);
        fc_asset_close(f);
        return NULL;
    }

    if (!fc_read_exact(f, &placement_count, sizeof(placement_count), 1, path, "object placement count") ||
        !fc_read_exact(f, &min_wx, sizeof(min_wx), 1, path, "object min world x") ||
        !fc_read_exact(f, &min_wy, sizeof(min_wy), 1, path, "object min world y") ||
        !fc_read_exact(f, &total_verts, sizeof(total_verts), 1, path, "object vertex count")) {
        fc_asset_close(f);
        return NULL;
    }

    fprintf(stderr, "objects_load: %u placements, %u verts, format=%s\n",
            placement_count, total_verts, has_textures ? "OBJ2" : "OBJS");

    /* read vertices */
    float* raw_verts = (float*)malloc(total_verts * 3 * sizeof(float));
    if (!raw_verts ||
        !fc_read_exact(f, raw_verts, sizeof(float), total_verts * 3, path, "object vertices")) {
        free(raw_verts);
        fc_asset_close(f);
        return NULL;
    }

    /* read colors */
    unsigned char* raw_colors = (unsigned char*)malloc(total_verts * 4);
    if (!raw_colors ||
        !fc_read_exact(f, raw_colors, 1, total_verts * 4, path, "object colors")) {
        free(raw_verts);
        free(raw_colors);
        fc_asset_close(f);
        return NULL;
    }
    /* read texture coordinates (v2 only) */
    float* raw_texcoords = NULL;
    if (has_textures) {
        raw_texcoords = (float*)malloc(total_verts * 2 * sizeof(float));
        if (!raw_texcoords ||
            !fc_read_exact(f, raw_texcoords, sizeof(float), total_verts * 2,
                           path, "object texcoords")) {
            free(raw_verts);
            free(raw_colors);
            free(raw_texcoords);
            fc_asset_close(f);
            return NULL;
        }
    }
    fc_asset_close(f);

    /* build raylib mesh */
    Mesh mesh = { 0 };
    mesh.vertexCount = (int)total_verts;
    mesh.triangleCount = (int)(total_verts / 3);
    mesh.vertices = raw_verts;
    mesh.colors = raw_colors;
    mesh.texcoords = raw_texcoords;

    /* compute normals */
    mesh.normals = (float*)calloc(total_verts * 3, sizeof(float));
    if (!mesh.normals) {
        free(raw_verts);
        free(raw_colors);
        free(raw_texcoords);
        return NULL;
    }
    for (int i = 0; i < mesh.triangleCount; i++) {
        int base = i * 9;
        float ax = raw_verts[base + 0], ay = raw_verts[base + 1], az = raw_verts[base + 2];
        float bx = raw_verts[base + 3], by = raw_verts[base + 4], bz = raw_verts[base + 5];
        float cx = raw_verts[base + 6], cy = raw_verts[base + 7], cz = raw_verts[base + 8];

        float e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        float e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (len > 0.0001f) { nx /= len; ny /= len; nz /= len; }

        for (int v = 0; v < 3; v++) {
            mesh.normals[i * 9 + v * 3 + 0] = nx;
            mesh.normals[i * 9 + v * 3 + 1] = ny;
            mesh.normals[i * 9 + v * 3 + 2] = nz;
        }
    }

    UploadMesh(&mesh, false);

    ObjectMesh* om = (ObjectMesh*)calloc(1, sizeof(ObjectMesh));
    om->model = LoadModelFromMesh(mesh);
    om->placement_count = (int)placement_count;
    om->total_vertex_count = (int)total_verts;
    om->min_world_x = min_wx;
    om->min_world_y = min_wy;
    om->has_textures = has_textures;
    om->loaded = 1;

    /* load atlas texture if v2 format */
    if (has_textures) {
        /* derive atlas path from objects path: replace .objects with .atlas */
        char atlas_path[1024];
        strncpy(atlas_path, path, sizeof(atlas_path) - 1);
        atlas_path[sizeof(atlas_path) - 1] = '\0';
        char* dot = strrchr(atlas_path, '.');
        if (dot) {
            strcpy(dot, ".atlas");
        } else {
            strncat(atlas_path, ".atlas", sizeof(atlas_path) - strlen(atlas_path) - 1);
        }

        if (fc_animated_atlas_load(&om->atlas, atlas_path, 1)) {
            /* assign atlas as diffuse map for the model's material */
            om->model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture =
                om->atlas.texture;
        }
    }

    return om;
}

ObjectAnimSet* object_anims_load(const char* path) {
    FILE* f = fc_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "object_anims_load: could not open %s\n", path);
        return NULL;
    }

    uint32_t magic = 0, version = 0, count = 0;
    if (!fc_read_exact(f, &magic, sizeof(magic), 1, path, "oanim magic") ||
        !fc_read_exact(f, &version, sizeof(version), 1, path, "oanim version") ||
        !fc_read_exact(f, &count, sizeof(count), 1, path, "oanim count") ||
        magic != OANM_MAGIC || version != OANM_VERSION) {
        fc_asset_close(f);
        return NULL;
    }

    ObjectAnimSet* set = (ObjectAnimSet*)calloc(1, sizeof(*set));
    if (!set) {
        fc_asset_close(f);
        return NULL;
    }
    set->rows = (ObjectAnimPlacement*)calloc(count, sizeof(*set->rows));
    if (count > 0 && !set->rows) {
        fc_asset_close(f);
        free(set);
        return NULL;
    }
    set->count = (int)count;

    for (uint32_t i = 0; i < count; i++) {
        ObjectAnimPlacement* row = &set->rows[i];
        if (!fc_read_exact(f, &row->model_id, sizeof(row->model_id), 1, path, "oanim model id") ||
            !fc_read_exact(f, &row->obj_id, sizeof(row->obj_id), 1, path, "oanim object id") ||
            !fc_read_exact(f, &row->animation_id, sizeof(row->animation_id), 1, path, "oanim animation id") ||
            !fc_read_exact(f, &row->world_x, sizeof(row->world_x), 1, path, "oanim world x") ||
            !fc_read_exact(f, &row->world_y, sizeof(row->world_y), 1, path, "oanim world y") ||
            !fc_read_exact(f, &row->plane, sizeof(row->plane), 1, path, "oanim plane") ||
            !fc_read_exact(f, &row->obj_type, sizeof(row->obj_type), 1, path, "oanim obj type") ||
            !fc_read_exact(f, &row->rotation, sizeof(row->rotation), 1, path, "oanim rotation") ||
            !fc_read_exact(f, &row->flags, sizeof(row->flags), 1, path, "oanim flags") ||
            !fc_read_exact(f, &row->pos_x, sizeof(row->pos_x), 1, path, "oanim pos x") ||
            !fc_read_exact(f, &row->pos_y, sizeof(row->pos_y), 1, path, "oanim pos y") ||
            !fc_read_exact(f, &row->pos_z, sizeof(row->pos_z), 1, path, "oanim pos z") ||
            !fc_read_exact(f, &row->phase_ticks, sizeof(row->phase_ticks), 1, path, "oanim phase")) {
            fc_asset_close(f);
            free(set->rows);
            free(set);
            return NULL;
        }
    }
    fc_asset_close(f);
    set->loaded = 1;
    fprintf(stderr, "object_anims_load: loaded %d animated placements from %s\n",
            set->count, path);
    return set;
}

void object_anims_offset(ObjectAnimSet* set, int wx, int wy) {
    if (!set || !set->loaded) return;
    for (int i = 0; i < set->count; i++) {
        set->rows[i].pos_x -= (float)wx;
        set->rows[i].pos_z += (float)wy;
        set->rows[i].world_x -= wx;
        set->rows[i].world_y -= wy;
    }
}

/* shift object vertices so world coordinates (wx, wy) become local (0, 0).
   must match terrain_offset() values for alignment. */
void objects_offset(ObjectMesh* om, int wx, int wy) {
    if (!om || !om->loaded) return;
    float dx = (float)wx;
    float dz = (float)wy;
    float* verts = om->model.meshes[0].vertices;
    for (int i = 0; i < om->total_vertex_count; i++) {
        verts[i * 3 + 0] -= dx;        /* X */
        verts[i * 3 + 2] += dz;        /* Z (negated world Y) */
    }
    UpdateMeshBuffer(om->model.meshes[0], 0, verts,
                     om->total_vertex_count * 3 * sizeof(float), 0);
    om->min_world_x -= wx;
    om->min_world_y -= wy;
    fprintf(stderr, "objects_offset: shifted by (%d, %d)\n", wx, wy);
}

void objects_free(ObjectMesh* om) {
    if (!om) return;
    fc_animated_atlas_unload(&om->atlas);
    if (om->loaded) UnloadModel(om->model);
    free(om);
}

void object_anims_free(ObjectAnimSet* set) {
    if (!set) return;
    free(set->rows);
    free(set);
}

#undef OBJS_MAGIC
#undef OBJ2_MAGIC
#undef OANM_MAGIC
#undef OANM_VERSION

/* Terrain Loader */
/**
 * @fileoverview Loads terrain mesh from .terrain binary into raylib Model.
 *
 * Binary format:
 *   magic: uint32 "TERR" (0x54455252)
 *   vertex_count: uint32
 *   region_count: uint32
 *   min_world_x: int32
 *   min_world_y: int32
 *   vertices: float32[vertex_count * 3]
 *   colors: uint8[vertex_count * 4]
 */

#include "raylib.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TERR_MAGIC 0x54455252

TerrainMesh* terrain_load(const char* path) {
    FILE* f = fc_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "terrain_load: could not open %s\n", path);
        return NULL;
    }

    uint32_t magic, vert_count, region_count;
    int32_t min_wx, min_wy;
    if (!fc_read_exact(f, &magic, sizeof(magic), 1, path, "terrain magic")) {
        fc_asset_close(f);
        return NULL;
    }
    if (magic != TERR_MAGIC) {
        fprintf(stderr, "terrain_load: bad magic %08x\n", magic);
        fc_asset_close(f);
        return NULL;
    }
    if (!fc_read_exact(f, &vert_count, sizeof(vert_count), 1, path, "terrain vertex count") ||
        !fc_read_exact(f, &region_count, sizeof(region_count), 1, path, "terrain region count") ||
        !fc_read_exact(f, &min_wx, sizeof(min_wx), 1, path, "terrain min world x") ||
        !fc_read_exact(f, &min_wy, sizeof(min_wy), 1, path, "terrain min world y")) {
        fc_asset_close(f);
        return NULL;
    }

    fprintf(stderr, "terrain_load: %u verts, %u regions, origin (%d, %d)\n",
            vert_count, region_count, min_wx, min_wy);

    /* read vertices */
    float* raw_verts = (float*)malloc(vert_count * 3 * sizeof(float));
    if (!raw_verts ||
        !fc_read_exact(f, raw_verts, sizeof(float), vert_count * 3, path, "terrain vertices")) {
        free(raw_verts);
        fc_asset_close(f);
        return NULL;
    }

    /* read colors */
    unsigned char* raw_colors = (unsigned char*)malloc(vert_count * 4);
    if (!raw_colors ||
        !fc_read_exact(f, raw_colors, 1, vert_count * 4, path, "terrain colors")) {
        free(raw_verts);
        free(raw_colors);
        fc_asset_close(f);
        return NULL;
    }
    /* build raylib mesh */
    Mesh mesh = { 0 };
    mesh.vertexCount = (int)vert_count;
    mesh.triangleCount = (int)(vert_count / 3);
    mesh.vertices = raw_verts;
    mesh.colors = raw_colors;

    /* compute normals for proper lighting */
    mesh.normals = (float*)calloc(vert_count * 3, sizeof(float));
    if (!mesh.normals) {
        free(raw_verts);
        free(raw_colors);
        fc_asset_close(f);
        return NULL;
    }
    for (int i = 0; i < mesh.triangleCount; i++) {
        int base = i * 9;
        float ax = raw_verts[base + 0], ay = raw_verts[base + 1], az = raw_verts[base + 2];
        float bx = raw_verts[base + 3], by = raw_verts[base + 4], bz = raw_verts[base + 5];
        float cx = raw_verts[base + 6], cy = raw_verts[base + 7], cz = raw_verts[base + 8];

        float e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        float e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (len > 0.0001f) { nx /= len; ny /= len; nz /= len; }

        for (int v = 0; v < 3; v++) {
            mesh.normals[i * 9 + v * 3 + 0] = nx;
            mesh.normals[i * 9 + v * 3 + 1] = ny;
            mesh.normals[i * 9 + v * 3 + 2] = nz;
        }
    }

    UploadMesh(&mesh, false);

    TerrainMesh* tm = (TerrainMesh*)calloc(1, sizeof(TerrainMesh));
    tm->model = LoadModelFromMesh(mesh);
    tm->vertex_count = (int)vert_count;
    tm->region_count = (int)region_count;
    tm->min_world_x = min_wx;
    tm->min_world_y = min_wy;
    tm->loaded = 1;

    /* read heightmap (appended after colors in the binary) */
    int32_t hm_min_x, hm_min_y;
    uint32_t hm_w, hm_h;
    int next = fgetc(f);
    if (next != EOF) {
        ungetc(next, f);
        if (!fc_read_exact(f, &hm_min_x, sizeof(hm_min_x), 1, path, "terrain heightmap min x") ||
            !fc_read_exact(f, &hm_min_y, sizeof(hm_min_y), 1, path, "terrain heightmap min y") ||
            !fc_read_exact(f, &hm_w, sizeof(hm_w), 1, path, "terrain heightmap width") ||
            !fc_read_exact(f, &hm_h, sizeof(hm_h), 1, path, "terrain heightmap height")) {
            fc_asset_close(f);
            terrain_free(tm);
            return NULL;
        }
        if (!(hm_w > 0 && hm_h > 0 && hm_w <= 4096 && hm_h <= 4096)) {
            fprintf(stderr, "%s: invalid terrain heightmap dimensions %ux%u\n",
                    path, hm_w, hm_h);
            fc_asset_close(f);
            terrain_free(tm);
            return NULL;
        }
        tm->hm_min_x = hm_min_x;
        tm->hm_min_y = hm_min_y;
        tm->hm_width = (int)hm_w;
        tm->hm_height = (int)hm_h;
        tm->heightmap = (float*)malloc(hm_w * hm_h * sizeof(float));
        if (!tm->heightmap ||
            !fc_read_exact(f, tm->heightmap, sizeof(float), hm_w * hm_h,
                           path, "terrain heightmap values")) {
            fc_asset_close(f);
            terrain_free(tm);
            return NULL;
        }
        fprintf(stderr, "terrain heightmap: %dx%d, origin (%d, %d)\n",
                tm->hm_width, tm->hm_height, tm->hm_min_x, tm->hm_min_y);
    }

    fc_asset_close(f);
    return tm;
}

/* shift terrain so world coordinates (wx, wy) become local (0, 0).
   offsets all mesh vertices and heightmap origin. must call before rendering. */
void terrain_offset(TerrainMesh* tm, int wx, int wy) {
    if (!tm || !tm->loaded) return;
    float dx = (float)wx;
    float dz = (float)wy;  /* Z = -world_y in our coord system */
    float* verts = tm->model.meshes[0].vertices;
    for (int i = 0; i < tm->vertex_count; i++) {
        verts[i * 3 + 0] -= dx;        /* X */
        verts[i * 3 + 2] += dz;        /* Z (negated world Y) */
    }
    UpdateMeshBuffer(tm->model.meshes[0], 0, verts,
                     tm->vertex_count * 3 * sizeof(float), 0);
    tm->min_world_x -= wx;
    tm->min_world_y -= wy;
    if (tm->heightmap) {
        tm->hm_min_x -= wx;
        tm->hm_min_y -= wy;
    }
    fprintf(stderr, "terrain_offset: shifted by (%d, %d), new origin (%d, %d)\n",
            wx, wy, tm->min_world_x, tm->min_world_y);
}

/* query terrain height at a world tile position (tile corner) */
float terrain_height_at(TerrainMesh* tm, int world_x, int world_y) {
    if (!tm || !tm->heightmap) return -2.0f;
    int lx = world_x - tm->hm_min_x;
    int ly = world_y - tm->hm_min_y;
    if (lx < 0 || lx >= tm->hm_width || ly < 0 || ly >= tm->hm_height)
        return -2.0f;
    return tm->heightmap[lx + ly * tm->hm_width];
}

void terrain_free(TerrainMesh* tm) {
    if (!tm) return;
    if (tm->loaded) {
        UnloadModel(tm->model);
    }
    free(tm->heightmap);
    free(tm);
}

#undef TERR_MAGIC

/* Spotanims */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SPOTANIM_MAGIC 0x544F5053u
#define SPOTANIM_VERSION 1u

SpotAnimSet *spotanims_load(const char *path) {
    FILE *f = fc_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "spotanims: can't open %s\n", path);
        return NULL;
    }

    uint32_t magic, version, count;
    if (!fc_read_exact(f, &magic, sizeof(magic), 1, path, "spotanim magic")
            || !fc_read_exact(f, &version, sizeof(version), 1, path,
                              "spotanim version")
            || !fc_read_exact(f, &count, sizeof(count), 1, path,
                              "spotanim count")
            || magic != SPOTANIM_MAGIC || version != SPOTANIM_VERSION) {
        fc_asset_close(f);
        return NULL;
    }

    SpotAnimSet *set = (SpotAnimSet *)calloc(1, sizeof(*set));
    if (!set) {
        fc_asset_close(f);
        return NULL;
    }
    set->defs = (SpotAnimDef *)calloc(count, sizeof(*set->defs));
    if (count > 0 && !set->defs) {
        fc_asset_close(f);
        spotanims_free(set);
        return NULL;
    }
    set->count = (int)count;
    for (uint32_t i = 0; i < count; i++) {
        if (!fc_read_exact(f, &set->defs[i], sizeof(set->defs[i]), 1, path,
                           "spotanim row")) {
            fc_asset_close(f);
            spotanims_free(set);
            return NULL;
        }
    }
    fc_asset_close(f);
    set->loaded = 1;
    fprintf(stderr, "spotanims: loaded %d from %s\n", set->count, path);
    return set;
}

const SpotAnimDef *spotanim_find(const SpotAnimSet *set, int id) {
    if (!set || !set->loaded || id < 0) return NULL;
    for (int i = 0; i < set->count; i++) {
        if ((int)set->defs[i].id == id) return &set->defs[i];
    }
    return NULL;
}

void spotanims_free(SpotAnimSet *set) {
    if (!set) return;
    free(set->defs);
    free(set);
}

#undef SPOTANIM_MAGIC
#undef SPOTANIM_VERSION

/* Player Appearance */
#include <stdlib.h>
#include <string.h>

int fc_player_appearance_load(FcPlayerAppearance *appearance) {
    memset(appearance, 0, sizeof(*appearance));
    const char *path = "fc_player.parts";
    FILE *file = fc_asset_fopen(path, "rb");
    if (!file) return 0;
    uint32_t header[2];
    int ok = fc_read_exact(file, header, sizeof(uint32_t), 2, path, "header") &&
        header[0] == 0x31504346 && header[1] > 0 && header[1] <= 64;
    if (ok) {
        appearance->record_count = (int)header[1];
        for (int i = 0; i < appearance->record_count; i++) {
            uint32_t row[2];
            if (!fc_read_exact(file, row, sizeof(uint32_t), 2, path, "item mapping") ||
                !fc_item_definition((int)row[0]) || row[1] > 127) { ok = 0; break; }
            appearance->records[i].item_id = row[0];
            appearance->records[i].hide_mask = row[1];
        }
        if (fgetc(file) != EOF) ok = 0;
    }
    fc_asset_close(file);
    if (!ok) { fprintf(stderr, "Invalid player appearance map: %s\n", path); return 0; }
    appearance->parts = models_load("fc_player.models", (Texture2D){0});
    if (!appearance->parts || appearance->parts->has_textures) return 0;
    for (int i = 0; i < 7; i++)
        if (!model_find(appearance->parts, 0xFC100000u + (uint32_t)i)) return 0;
    for (int i = 0; i < appearance->record_count; i++)
        if (!model_find(appearance->parts, appearance->records[i].item_id)) return 0;
    return 1;
}

static ModelSet *compose(ModelEntry *parts[], int count, uint32_t id) {
    int vertices = 0, faces = 0;
    for (int i = 0; i < count; i++) {
        vertices += parts[i]->base_vert_count;
        faces += parts[i]->face_count;
    }
    if (vertices <= 0 || vertices > UINT16_MAX || faces <= 0) return NULL;
    ModelSet *set = calloc(1, sizeof(*set));
    if (!set) return NULL;
    set->entries = calloc(1, sizeof(*set->entries));
    if (!set->entries) { free(set); return NULL; }
    set->count = 1;
    ModelEntry *out = set->entries;
    out->model_id = id;
    out->base_vert_count = vertices;
    out->face_count = faces;
    out->base_verts = malloc((size_t)vertices * 3 * sizeof(int16_t));
    out->vertex_skins = malloc((size_t)vertices);
    out->face_indices = malloc((size_t)faces * 3 * sizeof(uint16_t));
    out->face_priorities = malloc((size_t)faces);
    out->rest_verts = malloc((size_t)faces * 9 * sizeof(float));
    Mesh mesh = {.vertexCount = faces * 3, .triangleCount = faces};
    mesh.vertices = malloc((size_t)faces * 9 * sizeof(float));
    mesh.normals = malloc((size_t)faces * 9 * sizeof(float));
    mesh.colors = malloc((size_t)faces * 12);
    if (!out->base_verts || !out->vertex_skins || !out->face_indices ||
        !out->face_priorities || !out->rest_verts || !mesh.vertices ||
        !mesh.normals || !mesh.colors) {
        free(mesh.vertices); free(mesh.normals); free(mesh.colors);
        models_free(set);
        return NULL;
    }
    int vertex_offset = 0, face_offset = 0;
    for (int i = 0; i < count; i++) {
        const ModelEntry *part = parts[i];
        const Mesh *source = &part->model.meshes[0];
        memcpy(out->base_verts + vertex_offset * 3, part->base_verts,
               (size_t)part->base_vert_count * 3 * sizeof(int16_t));
        memcpy(out->vertex_skins + vertex_offset, part->vertex_skins,
               (size_t)part->base_vert_count);
        for (int f = 0; f < part->face_count * 3; f++)
            out->face_indices[face_offset * 3 + f] =
                (uint16_t)(part->face_indices[f] + vertex_offset);
        memcpy(out->face_priorities + face_offset, part->face_priorities,
               (size_t)part->face_count);
        memcpy(mesh.vertices + face_offset * 9, part->rest_verts,
               (size_t)part->face_count * 9 * sizeof(float));
        memcpy(mesh.normals + face_offset * 9, source->normals,
               (size_t)part->face_count * 9 * sizeof(float));
        memcpy(mesh.colors + face_offset * 12, source->colors,
               (size_t)part->face_count * 12);
        vertex_offset += part->base_vert_count;
        face_offset += part->face_count;
    }
    memcpy(out->rest_verts, mesh.vertices, (size_t)faces * 9 * sizeof(float));
    UploadMesh(&mesh, true);
    out->model = LoadModelFromMesh(mesh);
    out->loaded = set->loaded = 1;
    return set;
}

int fc_player_appearance_sync(FcPlayerAppearance *appearance,
                               const FcPlayer *player, uint32_t model_id) {
    int ids[FC_EQUIPMENT_SLOTS];
    for (int i = 0; i < FC_EQUIPMENT_SLOTS; i++) ids[i] = player->equipment[i].item_id;
    if (appearance->model && appearance->model->entries[0].model_id == model_id &&
        memcmp(ids, appearance->worn_ids, sizeof(ids)) == 0) return 0;
    if (!appearance->parts) return -1;
    ModelEntry *selected[7 + FC_EQUIPMENT_SLOTS];
    int count = 0;
    uint32_t hidden = 0;
    for (int slot = 0; slot < FC_EQUIPMENT_SLOTS; slot++) {
        if (!ids[slot] || slot == FC_EQUIP_SLOT_AMMO || slot == FC_EQUIP_SLOT_RING) continue;
        int record = 0;
        while (record < appearance->record_count &&
               appearance->records[record].item_id != (uint32_t)ids[slot]) record++;
        if (record == appearance->record_count) return -1;
        hidden |= appearance->records[record].hide_mask;
    }
    /* Identity kits precede equipped models, matching client composition. */
    for (int body = 0; body < 7; body++)
        if (!(hidden & (1u << body)))
            selected[count++] = model_find(appearance->parts, 0xFC100000u + (uint32_t)body);
    for (int slot = 0; slot < FC_EQUIPMENT_SLOTS; slot++)
        if (ids[slot] && slot != FC_EQUIP_SLOT_AMMO && slot != FC_EQUIP_SLOT_RING)
            selected[count++] = model_find(appearance->parts, (uint32_t)ids[slot]);
    for (int i = 0; i < count; i++) if (!selected[i]) return -1;
    ModelSet *model = compose(selected, count, model_id);
    if (!model) return -1;
    models_free(appearance->model);
    appearance->model = model;
    memcpy(appearance->worn_ids, ids, sizeof(ids));
    return 1;
}

void fc_player_appearance_free(FcPlayerAppearance *appearance) {
    models_free(appearance->model);
    models_free(appearance->parts);
    memset(appearance, 0, sizeof(*appearance));
}


#endif
