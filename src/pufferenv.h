#pragma once

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ini.h"
#include "raylib.h"
#ifdef PLATFORM_WEB
void puf_web_vsync(void);
#else
static void puf_web_vsync(void) {
}
#endif

typedef struct Env Env;
typedef struct Log Log;

typedef struct Agent {
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int policy;
} Agent;

// Env backend identity. Trainer: if (PUF_BACKEND == PUF_GPU) — not #if.
// GPU env sources #define PUF_BACKEND PUF_GPU before including this header.
#define PUF_CPU 0
#define PUF_GPU 1
#ifndef PUF_BACKEND
#define PUF_BACKEND PUF_CPU
#endif

// Shared env API. CPU: per-env Env*. GPU: Env* is device batch base; step/reset
// run the full vector inside the env. puf_bind_stream / puf_vec_create are GPU
// create-path hooks (CPU builds get stubs in pufferl).
void puf_init(Env* env, Dict* kwargs);
void puf_reset(Env* env);
void puf_step(Env* env);
// CPU: host Env*. GPU: device batch base; implementation D2Hs what it needs and draws.
void puf_render(Env* env);
void puf_close(Env* env);
void puf_log(Log* log, Dict* out);
// Bot ladder writes this between rungs so one PuffeRL can eval a whole ladder.
// Default no-op; envs with scripted opponents #define PUF_HAS_BOT_POLICY and
// assign env->bot_policy.
#ifndef PUF_HAS_BOT_POLICY
static inline void puf_set_bot_policy(Env* env, int bot_policy) {
}
#endif

typedef uint16_t bf16;

static inline bf16 f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

static inline float bf16_to_f32(bf16 b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

#if defined(__AVX2__)
static inline void store_f32x8_as_bf16(bf16* dst, __m256 v) {
    __m256i vi = _mm256_srli_epi32(_mm256_castps_si256(v), 16);
    __m128i lo = _mm256_castsi256_si128(vi);
    __m128i hi = _mm256_extracti128_si256(vi, 1);
    _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(lo, hi));
}
#endif

/*
 * WIP mascot vertex-cache loader.
 * This is here for now to make it easy for envs to load and animate the
 * puffer mascot without adding another tiny source file.
 */

#define PVA_FLAG_HALF_FLOAT 1u
#define PVA_FLAG_NORMALS 2u
#define PVA_FLAG_COLORS 4u

typedef struct PvaMeshClip {
    int vertex_count;
    uint16_t* positions;
    uint16_t* normals;
} PvaMeshClip;

typedef struct PvaClip {
    Model model;
    int frame_count;
    int frame_start;
    int frame_end;
    int mesh_count;
    float fps;
    unsigned int flags;
    PvaMeshClip* meshes;
} PvaClip;

static inline float pva_clamp(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static inline Matrix pva_matrix_identity(void) {
    Matrix matrix = {0};
    matrix.m0 = 1.0f;
    matrix.m5 = 1.0f;
    matrix.m10 = 1.0f;
    matrix.m15 = 1.0f;
    return matrix;
}

static inline float pva_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x03ffu;
    uint32_t bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03ffu;
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }

    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline int pva_read(void* dst, size_t size, FILE* file) {
    return fread(dst, 1, size, file) == size;
}

static inline int pva_read_u32(FILE* file, uint32_t* out) {
    return pva_read(out, sizeof(*out), file);
}

static inline int pva_read_i32(FILE* file, int32_t* out) {
    return pva_read(out, sizeof(*out), file);
}

static inline int pva_read_f32(FILE* file, float* out) {
    return pva_read(out, sizeof(*out), file);
}

/* Flat face normals from current triangle verts (PVA meshes are de-indexed). */
static inline void pva_compute_flat_normals(Mesh* mesh) {
    if (mesh == NULL || mesh->vertices == NULL || mesh->normals == NULL) {
        return;
    }

    int tri_count = mesh->triangleCount;
    for (int t = 0; t < tri_count; t++) {
        int base = t * 9;
        float ax = mesh->vertices[base + 3] - mesh->vertices[base + 0];
        float ay = mesh->vertices[base + 4] - mesh->vertices[base + 1];
        float az = mesh->vertices[base + 5] - mesh->vertices[base + 2];
        float bx = mesh->vertices[base + 6] - mesh->vertices[base + 0];
        float by = mesh->vertices[base + 7] - mesh->vertices[base + 1];
        float bz = mesh->vertices[base + 8] - mesh->vertices[base + 2];
        float nx = ay * bz - az * by;
        float ny = az * bx - ax * bz;
        float nz = ax * by - ay * bx;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (len > 1e-12f) {
            nx /= len;
            ny /= len;
            nz /= len;
        } else {
            nx = 0.0f;
            ny = 1.0f;
            nz = 0.0f;
        }
        for (int k = 0; k < 3; k++) {
            mesh->normals[base + k * 3 + 0] = nx;
            mesh->normals[base + k * 3 + 1] = ny;
            mesh->normals[base + k * 3 + 2] = nz;
        }
    }
}

/*
 * Drop baked export shading so materials provide flat albedo and runtime
 * lighting can do the rest (low-poly / Unity-default style).
 */
static inline void pva_use_flat_albedo(PvaClip* clip) {
    if (clip == NULL) {
        return;
    }

    for (int i = 0; i < clip->mesh_count; i++) {
        Mesh* mesh = &clip->model.meshes[i];
        if (mesh->colors == NULL) {
            mesh->colors = (unsigned char*)calloc((size_t)mesh->vertexCount * 4,
                sizeof(unsigned char));
        }
        if (mesh->colors == NULL) {
            continue;
        }
        for (int v = 0; v < mesh->vertexCount; v++) {
            mesh->colors[v * 4 + 0] = 255;
            mesh->colors[v * 4 + 1] = 255;
            mesh->colors[v * 4 + 2] = 255;
            mesh->colors[v * 4 + 3] = 255;
        }
        if (mesh->vboId != NULL) {
            UpdateMeshBuffer(*mesh, 3, mesh->colors,
                mesh->vertexCount * 4 * (int)sizeof(unsigned char), 0);
        }
    }
}

static inline void pva_set_mesh_frame(Mesh* mesh, PvaMeshClip* clip,
        int frame, int update_gpu) {
    int count = clip->vertex_count * 3;
    uint16_t* pos = clip->positions + (size_t)frame * count;
    uint16_t* normal = clip->normals != NULL ? clip->normals + (size_t)frame * count : NULL;

    for (int i = 0; i < count; i++) {
        mesh->vertices[i] = pva_half_to_float(pos[i]);
        if (normal != NULL && mesh->normals != NULL) {
            mesh->normals[i] = pva_half_to_float(normal[i]);
        }
    }

    /* Exporters currently bake lighting into colors and omit normals. Rebuild
     * face normals from the deformed frame so runtime shaders can light it. */
    if (normal == NULL && mesh->normals != NULL) {
        pva_compute_flat_normals(mesh);
    }

    if (update_gpu) {
        UpdateMeshBuffer(*mesh, 0, mesh->vertices, count * (int)sizeof(float), 0);
        if (mesh->normals != NULL) {
            UpdateMeshBuffer(*mesh, 2, mesh->normals, count * (int)sizeof(float), 0);
        }
    }
}

static inline void pva_unload_clip(PvaClip* clip) {
    if (clip == NULL) {
        return;
    }

    if (clip->meshes != NULL) {
        for (int i = 0; i < clip->mesh_count; i++) {
            free(clip->meshes[i].positions);
            free(clip->meshes[i].normals);
        }
        free(clip->meshes);
    }

    if (clip->model.meshCount > 0) {
        UnloadModel(clip->model);
    }
    free(clip);
}

static inline PvaClip* pva_load_clip(const char* path) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        TraceLog(LOG_ERROR, "PVA: failed to open %s", path);
        return NULL;
    }

    char magic[4];
    uint32_t version = 0;
    uint32_t flags = 0;
    uint32_t frame_count = 0;
    uint32_t mesh_count = 0;
    uint32_t material_count = 0;
    float fps = 0.0f;
    int32_t frame_start = 0;
    int32_t frame_end = 0;
    uint32_t reserved = 0;

    if (!pva_read(magic, sizeof(magic), file) || memcmp(magic, "PVA1", 4) != 0 ||
            !pva_read_u32(file, &version) || !pva_read_u32(file, &flags) ||
            !pva_read_u32(file, &frame_count) || !pva_read_u32(file, &mesh_count) ||
            !pva_read_u32(file, &material_count) || !pva_read_f32(file, &fps) ||
            !pva_read_i32(file, &frame_start) || !pva_read_i32(file, &frame_end) ||
            !pva_read_u32(file, &reserved)) {
        TraceLog(LOG_ERROR, "PVA: invalid header in %s", path);
        fclose(file);
        return NULL;
    }

    if (version != 1 || (flags & PVA_FLAG_HALF_FLOAT) == 0 || frame_count == 0 ||
            mesh_count == 0 || material_count == 0) {
        TraceLog(LOG_ERROR, "PVA: unsupported file %s", path);
        fclose(file);
        return NULL;
    }

    PvaClip* clip = (PvaClip*)calloc(1, sizeof(PvaClip));
    if (clip == NULL) {
        fclose(file);
        return NULL;
    }

    clip->frame_count = (int)frame_count;
    clip->frame_start = frame_start;
    clip->frame_end = frame_end;
    clip->mesh_count = (int)mesh_count;
    clip->fps = fps;
    clip->flags = flags;
    clip->meshes = (PvaMeshClip*)calloc(mesh_count, sizeof(PvaMeshClip));

    clip->model.transform = pva_matrix_identity();
    clip->model.meshCount = (int)mesh_count;
    clip->model.materialCount = (int)material_count;
    clip->model.meshes = (Mesh*)calloc(mesh_count, sizeof(Mesh));
    clip->model.materials = (Material*)calloc(material_count, sizeof(Material));
    clip->model.meshMaterial = (int*)calloc(mesh_count, sizeof(int));

    if (clip->meshes == NULL || clip->model.meshes == NULL ||
            clip->model.materials == NULL || clip->model.meshMaterial == NULL) {
        fclose(file);
        pva_unload_clip(clip);
        return NULL;
    }

    for (uint32_t i = 0; i < material_count; i++) {
        char name[64];
        float color[4];
        if (!pva_read(name, sizeof(name), file) || !pva_read(color, sizeof(color), file)) {
            fclose(file);
            pva_unload_clip(clip);
            return NULL;
        }
        clip->model.materials[i] = LoadMaterialDefault();
        /* Keep base albedo even when vertex colors are present (baked shade).
         * Runtime lighting paths can flatten vertex colors and use this. */
        Color material_color = {
            (unsigned char)(pva_clamp(color[0], 0.0f, 1.0f) * 255.0f),
            (unsigned char)(pva_clamp(color[1], 0.0f, 1.0f) * 255.0f),
            (unsigned char)(pva_clamp(color[2], 0.0f, 1.0f) * 255.0f),
            (unsigned char)(pva_clamp(color[3], 0.0f, 1.0f) * 255.0f),
        };
        clip->model.materials[i].maps[MATERIAL_MAP_DIFFUSE].color = material_color;
    }

    for (uint32_t i = 0; i < mesh_count; i++) {
        char name[64];
        uint32_t material_index = 0;
        uint32_t vertex_count = 0;
        uint32_t triangle_count = 0;

        if (!pva_read(name, sizeof(name), file) ||
                !pva_read_u32(file, &material_index) ||
                !pva_read_u32(file, &vertex_count) ||
                !pva_read_u32(file, &triangle_count)) {
            fclose(file);
            pva_unload_clip(clip);
            return NULL;
        }

        if (material_index >= material_count || vertex_count == 0 ||
                triangle_count == 0 || vertex_count != triangle_count * 3) {
            TraceLog(LOG_ERROR, "PVA: invalid mesh in %s", path);
            fclose(file);
            pva_unload_clip(clip);
            return NULL;
        }

        Mesh* mesh = &clip->model.meshes[i];
        mesh->vertexCount = (int)vertex_count;
        mesh->triangleCount = (int)triangle_count;
        mesh->vertices = (float*)calloc((size_t)vertex_count * 3, sizeof(float));
        mesh->texcoords = (float*)calloc((size_t)vertex_count * 2, sizeof(float));
        /* Always allocate normals: either load per-frame half-floats or rebuild
         * flat face normals after each pose for runtime lighting. */
        mesh->normals = (float*)calloc((size_t)vertex_count * 3, sizeof(float));
        if ((flags & PVA_FLAG_COLORS) != 0) {
            mesh->colors = (unsigned char*)calloc((size_t)vertex_count * 4, sizeof(unsigned char));
        }

        if (mesh->vertices == NULL || mesh->texcoords == NULL || mesh->normals == NULL ||
                ((flags & PVA_FLAG_COLORS) != 0 && mesh->colors == NULL) ||
                !pva_read(mesh->texcoords, (size_t)vertex_count * 2 * sizeof(float), file)) {
            fclose(file);
            pva_unload_clip(clip);
            return NULL;
        }
        if ((flags & PVA_FLAG_COLORS) != 0 &&
                !pva_read(mesh->colors, (size_t)vertex_count * 4 * sizeof(unsigned char), file)) {
            fclose(file);
            pva_unload_clip(clip);
            return NULL;
        }

        clip->model.meshMaterial[i] = (int)material_index;
        clip->meshes[i].vertex_count = (int)vertex_count;
    }

    for (uint32_t i = 0; i < mesh_count; i++) {
        size_t values = (size_t)frame_count * (size_t)clip->meshes[i].vertex_count * 3;
        clip->meshes[i].positions = (uint16_t*)malloc(values * sizeof(uint16_t));
        if ((flags & PVA_FLAG_NORMALS) != 0) {
            clip->meshes[i].normals = (uint16_t*)malloc(values * sizeof(uint16_t));
        }
        if (clip->meshes[i].positions == NULL ||
                ((flags & PVA_FLAG_NORMALS) != 0 && clip->meshes[i].normals == NULL)) {
            fclose(file);
            pva_unload_clip(clip);
            return NULL;
        }
    }

    for (uint32_t frame = 0; frame < frame_count; frame++) {
        for (uint32_t mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
            size_t values = (size_t)clip->meshes[mesh_index].vertex_count * 3;
            uint16_t* pos = clip->meshes[mesh_index].positions + (size_t)frame * values;
            if (!pva_read(pos, values * sizeof(uint16_t), file)) {
                fclose(file);
                pva_unload_clip(clip);
                return NULL;
            }
            if ((flags & PVA_FLAG_NORMALS) != 0) {
                uint16_t* normal = clip->meshes[mesh_index].normals + (size_t)frame * values;
                if (!pva_read(normal, values * sizeof(uint16_t), file)) {
                    fclose(file);
                    pva_unload_clip(clip);
                    return NULL;
                }
            }
        }
    }

    fclose(file);

    for (int i = 0; i < clip->mesh_count; i++) {
        pva_set_mesh_frame(&clip->model.meshes[i], &clip->meshes[i], 0, 0);
        UploadMesh(&clip->model.meshes[i], true);
    }

    return clip;
}

static inline void pva_update_clip(PvaClip* clip, int frame) {
    if (clip == NULL || clip->frame_count == 0) {
        return;
    }
    frame %= clip->frame_count;
    if (frame < 0) {
        frame += clip->frame_count;
    }

    for (int i = 0; i < clip->mesh_count; i++) {
        pva_set_mesh_frame(&clip->model.meshes[i], &clip->meshes[i], frame, 1);
    }
}

static inline int pva_frame_at_time(PvaClip* clip, float seconds) {
    if (clip == NULL || clip->frame_count == 0 || clip->fps <= 0.0f) {
        return 0;
    }
    return (int)(seconds * clip->fps) % clip->frame_count;
}
