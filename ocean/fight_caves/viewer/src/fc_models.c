// Loads models from .models MDL2/MDL3 binary for Raylib rendering.
// Fight Caves raylib model loader.

#include "fc_models.h"
#include "fc_assets.h"
#include "fc_io.h"
#include "raylib.h"
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
        free(set->entries[i].base_verts);
        free(set->entries[i].rest_verts);
        free(set->entries[i].rest_texcoords);
            free(set->entries[i].vertex_skins);
            free(set->entries[i].face_indices);
            free(set->entries[i].face_priorities);
            free(set->entries[i].face_uvs);
        }
    }
    free(set->entries);
    free(set->index_by_id);
    free(set);
}
