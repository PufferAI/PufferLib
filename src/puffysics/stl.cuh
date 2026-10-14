#pragma once

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "math.cuh"
#include "shapes.cuh"
#include "types.cuh"

/* One million facets is already 96 MiB of expanded host vertex/normal data. */
#define PF_STL_MAX_TRIANGLES 1000000
#define PF_STL_MAX_FILE_BYTES (128u * 1024u * 1024u)

typedef enum PfStlFormat {
    PF_STL_BINARY = 0,
    PF_STL_ASCII = 1
} PfStlFormat;

typedef struct PfStlMesh {
    PfVec3* vertices;
    PfVec3* normals;
    int triangle_count;
    int facet_count;
    int degenerate_count;
    PfStlFormat format;
} PfStlMesh;

typedef struct PfStlRenderMesh {
    float* vertices;
    float* texcoords;
    float* normals;
    int* indices;
    int vertex_count;
    int triangle_count;
} PfStlRenderMesh;

typedef struct PfStlHullFace {
    int a;
    int b;
    int c;
} PfStlHullFace;

typedef struct PfStlMassProperties {
    float volume;
    PfVec3 centroid;
    float inertia[3][3];
    PfVec3 inertia_diagonal;
    PfVec3 inverse_inertia_diagonal;
    float off_diagonal_magnitude;
    float relative_off_diagonal_magnitude;
} PfStlMassProperties;

typedef struct PfStlPrimitive {
    PfShape shape;
    float volume;
    float volume_ratio;
} PfStlPrimitive;

typedef struct PfStlHull {
    PfVec3* vertices;
    int vertex_count;
    PfStlHullFace* faces;
    int face_count;
    PfStlMassProperties mass;
    PfStlPrimitive box;
    PfStlPrimitive sphere;
    PfStlPrimitive cylinder;
    PfShapeKind selected_primitive;
    float selected_ratio;
} PfStlHull;

typedef struct PfStlHullFaceInternal {
    int a;
    int b;
    int c;
    double normal[3];
} PfStlHullFaceInternal;

typedef struct PfStlEdge {
    int a;
    int b;
} PfStlEdge;

typedef struct PfStlPointSort {
    PfVec3 point;
    int triangle_vertex;
} PfStlPointSort;

static inline bool pf_stl_mul_size(size_t a, size_t b, size_t* result) {
    if (result == NULL || (b != 0 && a > (size_t)-1 / b)) return false;
    *result = a * b;
    return true;
}

static inline void* pf_stl_alloc_array(size_t count, size_t element_size) {
    size_t bytes = 0;
    return pf_stl_mul_size(count, element_size, &bytes)
        ? (bytes == 0 ? NULL : malloc(bytes)) : NULL;
}

static inline bool pf_stl_grow_array(void** array, size_t element_size,
        size_t* capacity, size_t needed) {
    if (array == NULL || capacity == NULL || needed <= *capacity) return true;
    size_t next = *capacity == 0 ? 16 : *capacity;
    while (next < needed) {
        if (next > (size_t)-1 / 2) {
            next = needed;
            break;
        }
        next *= 2;
    }
    size_t bytes = 0;
    if (!pf_stl_mul_size(next, element_size, &bytes)) return false;
    void* grown = realloc(*array, bytes);
    if (grown == NULL) return false;
    *array = grown;
    *capacity = next;
    return true;
}

static inline void pf_stl_free_mesh(PfStlMesh* mesh) {
    if (mesh == NULL) return;
    free(mesh->vertices);
    free(mesh->normals);
    *mesh = (PfStlMesh){0};
}

static inline void pf_stl_free_render_mesh(PfStlRenderMesh* mesh) {
    if (mesh == NULL) return;
    free(mesh->vertices);
    free(mesh->texcoords);
    free(mesh->normals);
    free(mesh->indices);
    *mesh = (PfStlRenderMesh){0};
}

static inline void pf_stl_free_hull(PfStlHull* hull) {
    if (hull == NULL) return;
    free(hull->vertices);
    free(hull->faces);
    *hull = (PfStlHull){0};
}

static inline double pf_stl_component(PfVec3 value, int component) {
    return component == 0 ? value.x : component == 1 ? value.y : value.z;
}

static inline void pf_stl_cross3(const double a[3], const double b[3],
        double result[3]) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

static inline double pf_stl_dot3(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline double pf_stl_length3(const double value[3]) {
    return sqrt(pf_stl_dot3(value, value));
}

static inline int pf_stl_token(const char** cursor, const char* end,
        char* token, size_t capacity) {
    const char* p = *cursor;
    while (p < end && isspace((unsigned char)*p)) ++p;
    if (p == end) {
        *cursor = p;
        return 0;
    }
    size_t count = 0;
    while (p < end && !isspace((unsigned char)*p)) {
        if (count + 1 >= capacity) return -1;
        token[count++] = *p++;
    }
    token[count] = '\0';
    *cursor = p;
    return 1;
}

static inline bool pf_stl_word(const char** cursor, const char* end,
        const char* expected) {
    char token[128];
    return pf_stl_token(cursor, end, token, sizeof(token)) == 1
        && strcmp(token, expected) == 0;
}

static inline bool pf_stl_number_token(const char** cursor, const char* end,
        float* out) {
    char token[128];
    if (pf_stl_token(cursor, end, token, sizeof(token)) != 1) return false;
    char* parsed = NULL;
    errno = 0;
    float value = strtof(token, &parsed);
    if (parsed == token || *parsed != '\0' || errno == ERANGE
            || !isfinite(value) || !pf_number(value)) return false;
    *out = value;
    return true;
}

static inline bool pf_stl_triangle(PfVec3 a, PfVec3 b, PfVec3 c,
        PfVec3* normal, int* degenerate) {
    if (!pf_vec_valid(a) || !pf_vec_valid(b) || !pf_vec_valid(c)) return false;
    double e0[3] = {(double)b.x - a.x, (double)b.y - a.y, (double)b.z - a.z};
    double e1[3] = {(double)c.x - a.x, (double)c.y - a.y, (double)c.z - a.z};
    double n[3];
    pf_stl_cross3(e0, e1, n);
    double area_twice = pf_stl_length3(n);
    if (!isfinite(area_twice) || area_twice <= 1.0e-20) {
        *degenerate = 1;
        *normal = pf_v3(0.0f, 0.0f, 0.0f);
        return true;
    }
    n[0] /= area_twice;
    n[1] /= area_twice;
    n[2] /= area_twice;
    if (!isfinite(n[0]) || !isfinite(n[1]) || !isfinite(n[2])) return false;
    *normal = pf_v3((float)n[0], (float)n[1], (float)n[2]);
    if (!pf_vec_valid(*normal)) return false;
    *degenerate = 0;
    return true;
}

static inline bool pf_stl_parse_ascii(const unsigned char* bytes, size_t size,
        PfVec3* vertices, PfVec3* normals, int capacity,
        int* triangle_count, int* degenerate_count) {
    if (bytes == NULL || triangle_count == NULL || degenerate_count == NULL
            || (vertices == NULL) != (normals == NULL)
            || (vertices != NULL && capacity <= 0)) return false;
    const char* cursor = (const char*)bytes;
    const char* end = cursor + size;
    if (size >= 3 && cursor[0] == (char)0xef && cursor[1] == (char)0xbb
            && cursor[2] == (char)0xbf) cursor += 3;
    if (!pf_stl_word(&cursor, end, "solid")) return false;
    char token[128];
    if (pf_stl_token(&cursor, end, token, sizeof(token)) < 0) return false;
    while (strcmp(token, "facet") != 0) {
        if (pf_stl_token(&cursor, end, token, sizeof(token)) != 1) return false;
    }
    int triangles = 0;
    int degenerates = 0;
    int ended = 0;
    int first_facet = 1;
    while (cursor < end) {
        int state = 1;
        if (!first_facet) state = pf_stl_token(&cursor, end, token, sizeof(token));
        first_facet = 0;
        if (state <= 0) return false;
        if (strcmp(token, "endsolid") == 0) { ended = 1; break; }
        if (strcmp(token, "facet") != 0) return false;
        if (triangles >= PF_STL_MAX_TRIANGLES) return false;
        if (!pf_stl_word(&cursor, end, "normal")) return false;
        float ignored_normal;
        if (!pf_stl_number_token(&cursor, end, &ignored_normal)
                || !pf_stl_number_token(&cursor, end, &ignored_normal)
                || !pf_stl_number_token(&cursor, end, &ignored_normal)
                || !pf_stl_word(&cursor, end, "outer")
                || !pf_stl_word(&cursor, end, "loop")) return false;
        PfVec3 triangle[3];
        for (int vertex = 0; vertex < 3; ++vertex) {
            if (!pf_stl_word(&cursor, end, "vertex")
                    || !pf_stl_number_token(&cursor, end, &triangle[vertex].x)
                    || !pf_stl_number_token(&cursor, end, &triangle[vertex].y)
                    || !pf_stl_number_token(&cursor, end, &triangle[vertex].z)) {
                return false;
            }
        }
        if (!pf_stl_word(&cursor, end, "endloop")
                || !pf_stl_word(&cursor, end, "endfacet")) return false;
        PfVec3 normal;
        int degenerate = 0;
        if (!pf_stl_triangle(triangle[0], triangle[1], triangle[2],
                &normal, &degenerate)) return false;
        if (!degenerate && vertices != NULL) {
            if (triangles - degenerates >= capacity) return false;
            int index = 3 * (triangles - degenerates);
            vertices[index] = triangle[0];
            vertices[index + 1] = triangle[1];
            vertices[index + 2] = triangle[2];
            normals[index] = normal;
            normals[index + 1] = normal;
            normals[index + 2] = normal;
        }
        ++triangles;
        degenerates += degenerate;
    }
    if (!ended || triangles <= degenerates
            || (vertices != NULL && capacity < triangles - degenerates)) {
        return false;
    }
    int trailing = pf_stl_token(&cursor, end, token, sizeof(token));
    if (trailing < 0 || (trailing == 1
            && pf_stl_token(&cursor, end, token, sizeof(token)) != 0)) return false;
    *triangle_count = triangles - degenerates;
    *degenerate_count = degenerates;
    return true;
}

static inline bool pf_stl_binary_shape(const unsigned char* bytes, size_t size,
        uint32_t* count) {
    if (bytes == NULL || count == NULL || size < 84) return false;
    uint32_t value = (uint32_t)bytes[80] | ((uint32_t)bytes[81] << 8)
        | ((uint32_t)bytes[82] << 16) | ((uint32_t)bytes[83] << 24);
    if (value == 0 || value > PF_STL_MAX_TRIANGLES) return false;
    size_t expected = 0;
    size_t record_bytes = 0;
    if (!pf_stl_mul_size((size_t)value, (size_t)50, &record_bytes)
            || record_bytes > (size_t)-1 - (size_t)84) return false;
    expected = record_bytes + (size_t)84;
    if (expected != size) return false;
    *count = value;
    return true;
}

static inline uint32_t pf_stl_u32(const unsigned char* bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8)
        | ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static inline float pf_stl_f32(const unsigned char* bytes) {
    uint32_t bits = pf_stl_u32(bytes);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline bool pf_stl_parse_binary(const unsigned char* bytes, size_t size,
        uint32_t declared, PfStlMesh* mesh) {
    (void)size;
    size_t expanded = 0;
    if (!pf_stl_mul_size((size_t)declared, (size_t)3, &expanded)
            || expanded > (size_t)INT_MAX) return false;
    PfVec3* vertices = (PfVec3*)pf_stl_alloc_array(expanded, sizeof(PfVec3));
    PfVec3* normals = (PfVec3*)pf_stl_alloc_array(expanded, sizeof(PfVec3));
    if (vertices == NULL || normals == NULL) {
        free(vertices);
        free(normals);
        return false;
    }
    int triangles = 0;
    int degenerates = 0;
    for (uint32_t facet = 0; facet < declared; ++facet) {
        size_t record = 84 + (size_t)facet * 50;
        PfVec3 triangle[3];
        for (int vertex = 0; vertex < 3; ++vertex) {
            const unsigned char* source = bytes + record + 12 + (size_t)vertex * 12;
            triangle[vertex].x = pf_stl_f32(source);
            triangle[vertex].y = pf_stl_f32(source + 4);
            triangle[vertex].z = pf_stl_f32(source + 8);
        }
        PfVec3 normal;
        int degenerate = 0;
        if (!pf_stl_triangle(triangle[0], triangle[1], triangle[2],
                &normal, &degenerate)) {
            free(vertices);
            free(normals);
            return false;
        }
        if (!degenerate) {
            int index = 3 * triangles;
            vertices[index] = triangle[0];
            vertices[index + 1] = triangle[1];
            vertices[index + 2] = triangle[2];
            normals[index] = normal;
            normals[index + 1] = normal;
            normals[index + 2] = normal;
            ++triangles;
        }
        degenerates += degenerate;
    }
    if (triangles == 0) {
        free(vertices);
        free(normals);
        return false;
    }
    PfStlMesh result = {0};
    result.vertices = vertices;
    result.normals = normals;
    result.triangle_count = triangles;
    result.facet_count = (int)declared;
    result.degenerate_count = degenerates;
    result.format = PF_STL_BINARY;
    *mesh = result;
    return true;
}

static inline bool pf_stl_load(const char* path, PfStlMesh* out) {
    if (path == NULL || path[0] == '\0' || out == NULL) return false;
    FILE* file = fopen(path, "rb");
    if (file == NULL) return false;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return false;
    }
    long length = ftell(file);
    if (length <= 0 || (unsigned long)length > PF_STL_MAX_FILE_BYTES
            || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return false;
    }
    size_t size = (size_t)length;
    unsigned char* bytes = (unsigned char*)pf_stl_alloc_array(size, 1);
    if (bytes == NULL || fread(bytes, 1, size, file) != size) {
        free(bytes);
        fclose(file);
        return false;
    }
    if (fclose(file) != 0) {
        free(bytes);
        return false;
    }
    uint32_t binary_count = 0;
    PfStlMesh result = {0};
    if (pf_stl_binary_shape(bytes, size, &binary_count)) {
        bool success = pf_stl_parse_binary(bytes, size, binary_count, &result);
        if (success) *out = result;
        free(bytes);
        return success;
    }
    int triangles = 0;
    int degenerates = 0;
    bool ascii_ok = pf_stl_parse_ascii(bytes, size, NULL, NULL, 0,
        &triangles, &degenerates);
    if (!ascii_ok) {
        free(bytes);
        return false;
    }
    size_t expanded = 0;
    if (!pf_stl_mul_size((size_t)triangles, (size_t)3, &expanded)
            || expanded > (size_t)INT_MAX) {
        free(bytes);
        return false;
    }
    PfVec3* vertices = (PfVec3*)pf_stl_alloc_array(expanded, sizeof(PfVec3));
    PfVec3* normals = (PfVec3*)pf_stl_alloc_array(expanded, sizeof(PfVec3));
    int parsed_triangles = 0;
    int parsed_degenerates = 0;
    if (vertices == NULL || normals == NULL
            || !pf_stl_parse_ascii(bytes, size, vertices, normals,
                triangles, &parsed_triangles, &parsed_degenerates)
            || parsed_triangles != triangles
            || parsed_degenerates != degenerates) {
        free(vertices);
        free(normals);
        free(bytes);
        return false;
    }
    free(bytes);
    result.vertices = vertices;
    result.normals = normals;
    result.triangle_count = parsed_triangles;
    result.facet_count = triangles + degenerates;
    result.degenerate_count = parsed_degenerates;
    result.format = PF_STL_ASCII;
    *out = result;
    return true;
}

static inline bool pf_stl_render_mesh(const PfStlMesh* mesh,
        PfStlRenderMesh* out) {
    if (mesh == NULL || out == NULL || mesh->vertices == NULL
            || mesh->normals == NULL || mesh->triangle_count <= 0
            || mesh->triangle_count > PF_STL_MAX_TRIANGLES
            || mesh->triangle_count > INT_MAX / 3) return false;
    int vertices = 3 * mesh->triangle_count;
    float* positions = (float*)pf_stl_alloc_array((size_t)vertices * 6,
        sizeof(float));
    float* texcoords = (float*)pf_stl_alloc_array((size_t)vertices * 2,
        sizeof(float));
    float* normals = (float*)pf_stl_alloc_array((size_t)vertices * 3,
        sizeof(float));
    int* indices = (int*)pf_stl_alloc_array((size_t)vertices, sizeof(int));
    if (positions == NULL || texcoords == NULL || normals == NULL || indices == NULL) {
        free(positions);
        free(texcoords);
        free(normals);
        free(indices);
        return false;
    }
    memset(texcoords, 0, (size_t)vertices * 2 * sizeof(float));
    for (int index = 0; index < vertices; ++index) {
        if (!pf_vec_valid(mesh->vertices[index])
                || !pf_vec_valid(mesh->normals[index])) {
            free(positions);
            free(texcoords);
            free(normals);
            free(indices);
            return false;
        }
        PfVec3 point = mesh->vertices[index];
        PfVec3 normal = mesh->normals[index];
        size_t offset = (size_t)index;
        positions[offset * 6] = point.x;
        positions[offset * 6 + 1] = point.y;
        positions[offset * 6 + 2] = point.z;
        positions[offset * 6 + 3] = point.x;
        positions[offset * 6 + 4] = point.y;
        positions[offset * 6 + 5] = point.z;
        normals[offset * 3] = normal.x;
        normals[offset * 3 + 1] = normal.y;
        normals[offset * 3 + 2] = normal.z;
        indices[index] = index;
    }
    PfStlRenderMesh result = {positions, texcoords, normals, indices,
        vertices, mesh->triangle_count};
    *out = result;
    return true;
}

static inline int pf_stl_point_compare(const void* left, const void* right) {
    const PfStlPointSort* a = (const PfStlPointSort*)left;
    const PfStlPointSort* b = (const PfStlPointSort*)right;
    float av[3] = {a->point.x, a->point.y, a->point.z};
    float bv[3] = {b->point.x, b->point.y, b->point.z};
    for (int component = 0; component < 3; ++component) {
        if (av[component] < bv[component]) return -1;

        if (av[component] > bv[component]) return 1;
    }
    return a->triangle_vertex < b->triangle_vertex ? -1
        : a->triangle_vertex > b->triangle_vertex;
}


static inline int pf_stl_edge_undirected_compare(const void* left,
        const void* right) {
    const PfStlEdge* a = (const PfStlEdge*)left;
    const PfStlEdge* b = (const PfStlEdge*)right;
    int a0 = a->a < a->b ? a->a : a->b;
    int a1 = a->a < a->b ? a->b : a->a;
    int b0 = b->a < b->b ? b->a : b->b;
    int b1 = b->a < b->b ? b->b : b->a;
    if (a0 != b0) return a0 < b0 ? -1 : 1;
    if (a1 != b1) return a1 < b1 ? -1 : 1;
    return 0;
}

static inline bool pf_stl_indexed_face(const PfVec3* vertices, int a, int b,
        int c, const double interior[3], double area_epsilon,
        PfStlHullFaceInternal* out) {
    double av[3] = {vertices[a].x, vertices[a].y, vertices[a].z};
    double bv[3] = {vertices[b].x, vertices[b].y, vertices[b].z};
    double cv[3] = {vertices[c].x, vertices[c].y, vertices[c].z};
    double e0[3], e1[3], face_normal[3], side[3];
    for (int component = 0; component < 3; ++component) {
        e0[component] = bv[component] - av[component];
        e1[component] = cv[component] - av[component];
        side[component] = interior[component] - av[component];
    }
    pf_stl_cross3(e0, e1, face_normal);

    double length = pf_stl_length3(face_normal);
    if (!isfinite(length) || length <= area_epsilon) return false;
    for (int component = 0; component < 3; ++component) face_normal[component] /= length;
    if (pf_stl_dot3(face_normal, side) > 0.0) {
        int temporary = b;
        b = c;
        c = temporary;
        for (int component = 0; component < 3; ++component) face_normal[component] *= -1.0;
    }
    if (pf_stl_dot3(face_normal, side) == 0.0) return false;
    out->a = a;
    out->b = b;
    out->c = c;
    out->normal[0] = face_normal[0];
    out->normal[1] = face_normal[1];
    out->normal[2] = face_normal[2];
    return true;
}

static inline bool pf_stl_point_in_face(PfVec3 a, PfVec3 b, PfVec3 c,
        PfVec3 point) {
    double e0[3] = {(double)b.x - a.x, (double)b.y - a.y, (double)b.z - a.z};
    double e1[3] = {(double)c.x - a.x, (double)c.y - a.y, (double)c.z - a.z};
    double normal[3];
    pf_stl_cross3(e0, e1, normal);
    int axis = fabs(normal[0]) >= fabs(normal[1])
        && fabs(normal[0]) >= fabs(normal[2]) ? 0
        : fabs(normal[1]) >= fabs(normal[2]) ? 1 : 2;
    int u = (axis + 1) % 3;
    int v = (axis + 2) % 3;
    double av[3] = {a.x, a.y, a.z};
    double bv[3] = {b.x, b.y, b.z};
    double cv[3] = {c.x, c.y, c.z};
    double pv[3] = {point.x, point.y, point.z};
    double ab[2] = {bv[u] - av[u], bv[v] - av[v]};
    double bc[2] = {cv[u] - bv[u], cv[v] - bv[v]};
    double ca[2] = {av[u] - cv[u], av[v] - cv[v]};
    double ap[2] = {pv[u] - av[u], pv[v] - av[v]};
    double bp[2] = {pv[u] - bv[u], pv[v] - bv[v]};
    double cp[2] = {pv[u] - cv[u], pv[v] - cv[v]};
    double s1 = ab[0] * ap[1] - ab[1] * ap[0];
    double s2 = bc[0] * bp[1] - bc[1] * bp[0];
    double s3 = ca[0] * cp[1] - ca[1] * cp[0];
    bool negative = s1 < -1.0e-12 || s2 < -1.0e-12 || s3 < -1.0e-12;
    bool positive = s1 > 1.0e-12 || s2 > 1.0e-12 || s3 > 1.0e-12;
    return !(negative && positive);
}

static inline bool pf_stl_hull_mass(PfStlHull* hull);
static inline bool pf_stl_primitive_fit(PfStlHull* hull, const double interior[3]);
static inline bool pf_stl_primitive_ratios(PfStlHull* hull, float reference);
static inline bool pf_stl_primitive_encloses(PfStlHull* hull);

static inline bool pf_stl_box_hull(PfStlHull* hull, const double interior[3]) {
    double minimum[3] = {DBL_MAX, DBL_MAX, DBL_MAX};
    double maximum[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    for (int i = 0; i < hull->vertex_count; ++i) {
        double p[3] = {hull->vertices[i].x, hull->vertices[i].y, hull->vertices[i].z};
        for (int k = 0; k < 3; ++k) { if (p[k] < minimum[k]) minimum[k] = p[k]; if (p[k] > maximum[k]) maximum[k] = p[k]; }
    }
    if (hull->vertex_count != 8 || maximum[0] <= minimum[0] || maximum[1] <= minimum[1] || maximum[2] <= minimum[2]) return false;
    int corner[8];
    for (int c = 0; c < 8; ++c) {
        double target[3] = {(c & 1) ? maximum[0] : minimum[0], (c & 2) ? maximum[1] : minimum[1], (c & 4) ? maximum[2] : minimum[2]};
        corner[c] = -1;
        for (int i = 0; i < hull->vertex_count; ++i) {
            double p[3] = {hull->vertices[i].x, hull->vertices[i].y, hull->vertices[i].z};
            if (fabs(p[0] - target[0]) < 1.0e-6 && fabs(p[1] - target[1]) < 1.0e-6 && fabs(p[2] - target[2]) < 1.0e-6) { corner[c] = i; break; }
        }
        if (corner[c] < 0) return false;
    }
    const int indices[12][3] = {{0,2,1},{0,3,2},{4,5,6},{4,6,7},
        {0,1,5},{0,5,4},{2,3,7},{2,7,6},{0,4,6},{0,6,2},{1,3,7},{1,7,5}};
    PfStlHullFace* faces = (PfStlHullFace*)pf_stl_alloc_array(12, sizeof(PfStlHullFace));
    if (faces == NULL) return false;
    hull->faces = faces;
    hull->face_count = 12;
    for (int i = 0; i < 12; ++i) {
        PfStlHullFaceInternal oriented;
        if (!pf_stl_indexed_face(hull->vertices, corner[indices[i][0]],
                corner[indices[i][1]], corner[indices[i][2]], interior,
                1.0e-20, &oriented)) { free(faces); return false; }
        faces[i] = (PfStlHullFace){oriented.a, oriented.b, oriented.c};
    }
    double width = maximum[0] - minimum[0];
    double height = maximum[1] - minimum[1];
    double depth = maximum[2] - minimum[2];
    PfStlMassProperties mass = {0};
    mass.volume = (float)(width * height * depth);
    mass.centroid = pf_v3((float)((minimum[0] + maximum[0]) * 0.5),
        (float)((minimum[1] + maximum[1]) * 0.5),
        (float)((minimum[2] + maximum[2]) * 0.5));
    mass.inertia[0][0] = mass.inertia[1][1] = mass.inertia[2][2] = 0.0f;
    mass.inertia[0][0] = (float)(mass.volume * (height * height + depth * depth) / 12.0);
    mass.inertia[1][1] = (float)(mass.volume * (width * width + depth * depth) / 12.0);
    mass.inertia[2][2] = (float)(mass.volume * (width * width + height * height) / 12.0);
    mass.inertia_diagonal = pf_v3(mass.inertia[0][0], mass.inertia[1][1], mass.inertia[2][2]);
    mass.inverse_inertia_diagonal = pf_v3(1.0f / mass.inertia[0][0],
        1.0f / mass.inertia[1][1], 1.0f / mass.inertia[2][2]);
    hull->mass = mass;
    if (!pf_stl_primitive_fit(hull, interior)) { pf_stl_free_hull(hull); return false; }
    return true;
}

static inline bool pf_stl_hull_mass(PfStlHull* hull) {
    double signed_volume = 0.0;
    double center[3] = {0.0, 0.0, 0.0};
    double second[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (int face_index = 0; face_index < hull->face_count; ++face_index) {
        PfStlHullFace face = hull->faces[face_index];
        if (face.a < 0 || face.a >= hull->vertex_count
                || face.b < 0 || face.b >= hull->vertex_count
                || face.c < 0 || face.c >= hull->vertex_count) return false;
        double p[3][3];
        const int indices[3] = {face.a, face.b, face.c};
        for (int vertex = 0; vertex < 3; ++vertex) {
            PfVec3 point = hull->vertices[indices[vertex]];
            p[vertex][0] = point.x;
            p[vertex][1] = point.y;
            p[vertex][2] = point.z;
        }
        double determinant = p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1])
            - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0])
            + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0]);
        signed_volume += determinant;
        for (int component = 0; component < 3; ++component) {
            center[component] += determinant * (p[0][component]
                + p[1][component] + p[2][component]);
        }
        for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column) {
            /* Unit-density tetrahedron (0, p0, p1, p2) with signed determinant
             * D: diagonal D/60 * (sum x_k^2 + sum_{k<l} x_k x_l), mixed
             * D/120 * (2 sum x_k y_k + sum_{k<l} (x_k y_l + x_l y_k)). The
             * mixed sum carries a factor of two the diagonal does not. */
            double moment = 0.0;
            for (int vertex = 0; vertex < 3; ++vertex) {
                moment += p[vertex][row] * p[vertex][column];
            }
            if (row != column) moment *= 2.0;
            for (int a = 0; a < 3; ++a) for (int b = a + 1; b < 3; ++b) {
                moment += row == column
                    ? p[a][row] * p[b][column]
                    : p[a][row] * p[b][column] + p[b][row] * p[a][column];
            }
            second[row][column] += determinant * moment / (row == column ? 60.0 : 120.0);
        }
    }
    double volume = signed_volume / 6.0;
    if (!isfinite(volume) || volume <= 0.0) return false;
    for (int component = 0; component < 3; ++component) {
        center[component] *= 1.0 / (24.0 * volume);
    }
    /* `second` is the raw integral of x_row * x_column; the central tensor
     * about the centroid subtracts volume * c c^T. Omitting the volume factor
     * turns any translated mesh negative. */
    double central[3][3];
    double trace = 0.0;
    double norm_squared = 0.0;
    double off_squared = 0.0;
    for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column) {
        central[row][column] = second[row][column]
            - volume * center[row] * center[column];
    }
    for (int diagonal = 0; diagonal < 3; ++diagonal) trace += central[diagonal][diagonal];
    PfStlMassProperties result = {0};
    result.volume = (float)volume;
    result.centroid = pf_v3((float)center[0], (float)center[1], (float)center[2]);
    for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column) {
        double value = (row == column ? trace : 0.0) - central[row][column];
        if (!isfinite(value) || value > 1.0e100 || value < -1.0e100) return false;
        result.inertia[row][column] = (float)value;
        norm_squared += value * value;
        if (row != column) off_squared += value * value;
    }
    result.inertia_diagonal = pf_v3(result.inertia[0][0], result.inertia[1][1],
        result.inertia[2][2]);
    result.off_diagonal_magnitude = (float)sqrt(off_squared);
    result.relative_off_diagonal_magnitude = norm_squared > 0.0
        ? (float)sqrt(off_squared / norm_squared) : 0.0f;
    for (int diagonal = 0; diagonal < 3; ++diagonal) {
        float value = diagonal == 0 ? result.inertia_diagonal.x
            : diagonal == 1 ? result.inertia_diagonal.y : result.inertia_diagonal.z;
        if (!pf_number(value) || value <= 0.0f) return false;
    }
    result.inverse_inertia_diagonal = pf_v3(
        1.0f / result.inertia_diagonal.x,
        1.0f / result.inertia_diagonal.y,
        1.0f / result.inertia_diagonal.z);
    if (!pf_vec_valid(result.centroid)
            || !pf_vec_valid(result.inverse_inertia_diagonal)) return false;
    hull->mass = result;
    return true;
}

static inline void pf_stl_principal_axes(const PfVec3* vertices, int count,
        const double interior[3], PfVec3 axes[3]) {
    double covariance[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (int i = 0; i < count; ++i) {
        double p[3] = {vertices[i].x - interior[0], vertices[i].y - interior[1],
            vertices[i].z - interior[2]};
        for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column) {
            covariance[row][column] += p[row] * p[column];
        }
    }
    for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column) {
        covariance[row][column] /= count;
    }
    double vectors[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    double covariance_scale = 0.0;
    for (int i = 0; i < 3; ++i) {
        covariance_scale = fmax(covariance_scale, fabs(covariance[i][i]));
    }
    for (int iteration = 0; iteration < 64; ++iteration) {
        int p = 0;
        int q = 1;
        double largest = fabs(covariance[0][1]);
        if (fabs(covariance[0][2]) > largest) { p = 0; q = 2; largest = fabs(covariance[0][2]); }
        if (fabs(covariance[1][2]) > largest) { p = 1; q = 2; largest = fabs(covariance[1][2]); }
        if (largest <= fmax(1.0e-30, covariance_scale * 1.0e-14)) break;
        /* One convention throughout: A' = J^T A J with J = [[c, -s], [s, c]]
         * on (p, q). That J annihilates A[p][q] when tan(2t) = 2 A[p][q] /
         * (A[p][p] - A[q][q]), and it is the J the diagonal, off-diagonal and
         * eigenvector updates below all apply. */
        double angle = 0.5 * atan2(2.0 * covariance[p][q],
            covariance[p][p] - covariance[q][q]);
        double c = cos(angle);
        double s = sin(angle);
        double app = covariance[p][p];
        double aqq = covariance[q][q];
        double apq = covariance[p][q];
        covariance[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        covariance[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        covariance[p][q] = covariance[q][p] = 0.0;
        for (int k = 0; k < 3; ++k) if (k != p && k != q) {
            double akp = covariance[k][p];
            double akq = covariance[k][q];
            covariance[k][p] = covariance[p][k] = c * akp + s * akq;
            covariance[k][q] = covariance[q][k] = -s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
            double vkp = vectors[k][p];
            double vkq = vectors[k][q];
            vectors[k][p] = c * vkp + s * vkq;
            vectors[k][q] = -s * vkp + c * vkq;
        }
    }
    int order[3] = {0, 1, 2};
    for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j) {
        if (covariance[order[j]][order[j]] > covariance[order[i]][order[i]]) {
            int temporary = order[i]; order[i] = order[j]; order[j] = temporary;
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        int column = order[axis];
        double x = vectors[0][column];
        double y = vectors[1][column];
        double z = vectors[2][column];
        double length = sqrt(x * x + y * y + z * z);
        if (!isfinite(length) || length <= 1.0e-20) return;
        axes[axis] = pf_v3((float)(x / length), (float)(y / length), (float)(z / length));
    }
    double determinant = axes[0].x * (axes[1].y * axes[2].z - axes[1].z * axes[2].y)
        - axes[1].x * (axes[0].y * axes[2].z - axes[0].z * axes[2].y)
        + axes[2].x * (axes[0].y * axes[1].z - axes[0].z * axes[1].y);
    if (determinant < 0.0) {
        axes[2] = pf_scale(axes[2], -1.0f);
    }
}


static inline PfQuat pf_stl_matrix_quat(PfVec3 axes[3]) {
    double m00 = axes[0].x, m10 = axes[0].y, m20 = axes[0].z;
    double m01 = axes[1].x, m11 = axes[1].y, m21 = axes[1].z;
    double m02 = axes[2].x, m12 = axes[2].y, m22 = axes[2].z;
    double trace = m00 + m11 + m22;
    PfQuat q;
    if (trace > 0.0) {
        double s = sqrt(trace + 1.0) * 2.0;
        q = pf_quat_normalize((PfQuat){(float)(0.25 * s), (float)((m21 - m12) / s),
            (float)((m02 - m20) / s), (float)((m10 - m01) / s)});
    } else if (m00 > m11 && m00 > m22) {
        double s = sqrt(1.0 + m00 - m11 - m22) * 2.0;
        q = pf_quat_normalize((PfQuat){(float)((m21 - m12) / s), (float)(0.25 * s),
            (float)((m01 + m10) / s), (float)((m02 + m20) / s)});
    } else if (m11 > m22) {
        double s = sqrt(1.0 + m11 - m00 - m22) * 2.0;
        q = pf_quat_normalize((PfQuat){(float)((m02 - m20) / s),
            (float)((m01 + m10) / s), (float)(0.25 * s), (float)((m12 + m21) / s)});
    } else {
        double s = sqrt(1.0 + m22 - m00 - m11) * 2.0;
        q = pf_quat_normalize((PfQuat){(float)((m10 - m01) / s),
            (float)((m02 + m20) / s), (float)((m12 + m21) / s), (float)(0.25 * s)});
    }
    return q;
}

static inline bool pf_stl_primitive_fit(PfStlHull* hull,
        const double interior[3]) {
    PfVec3 axes[3];
    pf_stl_principal_axes(hull->vertices, hull->vertex_count, interior, axes);
    double minimum[3], maximum[3];
    for (int axis = 0; axis < 3; ++axis) {
        minimum[axis] = DBL_MAX;
        maximum[axis] = -DBL_MAX;
    }
    double sphere_radius_squared = 0.0;
    double center[3] = {interior[0], interior[1], interior[2]};
    for (int i = 0; i < hull->vertex_count; ++i) {
        double point[3] = {hull->vertices[i].x, hull->vertices[i].y, hull->vertices[i].z};
        for (int axis = 0; axis < 3; ++axis) {
            double projection = (point[0] - center[0]) * axes[axis].x
                + (point[1] - center[1]) * axes[axis].y
                + (point[2] - center[2]) * axes[axis].z;
            if (projection < minimum[axis]) minimum[axis] = projection;
            if (projection > maximum[axis]) maximum[axis] = projection;
        }
    }
    /* The tight OBB is centred on the midpoint of each projected interval, not
     * on the vertex mean: for an asymmetric mesh those differ and the half
     * extents then hang off a centre that leaves vertices outside the box. */
    double fit[3] = {center[0], center[1], center[2]};
    for (int axis = 0; axis < 3; ++axis) {
        double midpoint = 0.5 * (minimum[axis] + maximum[axis]);
        for (int component = 0; component < 3; ++component) {
            fit[component] += midpoint * (double)pf_stl_component(axes[axis], component);
        }
    }
    PfVec3 position = pf_v3((float)fit[0], (float)fit[1], (float)fit[2]);
    PfQuat rotation = pf_stl_matrix_quat(axes);
    PfVec3 half = pf_v3((float)((maximum[0] - minimum[0]) * 0.5),
        (float)((maximum[1] - minimum[1]) * 0.5),
        (float)((maximum[2] - minimum[2]) * 0.5));
    if (!pf_vec_valid(position) || !pf_quat_valid(rotation)
            || !pf_vec_valid(half) || half.x <= 0.0f || half.y <= 0.0f
            || half.z <= 0.0f) return false;
    PfShape box_shape = {PF_BOX, half, position, rotation};
    for (int i = 0; i < hull->vertex_count; ++i) {
        double dx = hull->vertices[i].x - position.x;
        double dy = hull->vertices[i].y - position.y;
        double dz = hull->vertices[i].z - position.z;
        double radius_squared = dx * dx + dy * dy + dz * dz;
        if (radius_squared > sphere_radius_squared) sphere_radius_squared = radius_squared;
    }
    float sphere_radius = (float)sqrt(sphere_radius_squared);
    if (!pf_number(sphere_radius) || sphere_radius <= 0.0f) return false;
    PfShape sphere_shape = {PF_SPHERE, pf_v3(sphere_radius, 0.0f, 0.0f),
        position, pf_quat_identity()};
    int long_axis = half.x >= half.y && half.x >= half.z ? 0 : half.y >= half.z ? 1 : 2;
    /* A cylinder's longitudinal axis is its local Y (see pf_shape_axis), so the
     * rotation has to map Y onto the SELECTED principal axis rather than
     * always onto the second. Each order below is a cyclic relabelling of the
     * right-handed frame, so the matrix stays a proper rotation. */
    int order[3] = {long_axis == 0 ? 2 : long_axis == 1 ? 0 : 1, long_axis,
        long_axis == 0 ? 1 : long_axis == 1 ? 2 : 0};
    int transverse[2] = {order[0], order[2]};
    PfVec3 cylinder_axes[3] = {axes[order[0]], axes[order[1]], axes[order[2]]};
    PfQuat cylinder_rotation = pf_stl_matrix_quat(cylinder_axes);
    double radius_squared = 0.0;
    for (int i = 0; i < hull->vertex_count; ++i) {
        double delta[3] = {hull->vertices[i].x - position.x,
            hull->vertices[i].y - position.y, hull->vertices[i].z - position.z};
        double transverse_distance = 0.0;
        for (int axis = 0; axis < 2; ++axis) {
            int component = transverse[axis];
            double projection = delta[0] * axes[component].x
                + delta[1] * axes[component].y + delta[2] * axes[component].z;
            transverse_distance += projection * projection;
        }
        if (transverse_distance > radius_squared) radius_squared = transverse_distance;
    }
    float cylinder_radius = (float)sqrt(radius_squared);
    float cylinder_half_height = (float)pf_stl_component(half, long_axis);
    PfShape cylinder_shape = {PF_CYLINDER, pf_v3(cylinder_radius, cylinder_half_height, 0.0f),
        position, cylinder_rotation};
    if (cylinder_radius <= 0.0f || cylinder_half_height <= 0.0f
            || !pf_compound_shape_valid(&box_shape)
            || !pf_compound_shape_valid(&sphere_shape)
            || !pf_compound_shape_valid(&cylinder_shape)) return false;
    float box_volume = pf_shape_volume(&box_shape);
    float sphere_volume = pf_shape_volume(&sphere_shape);
    float cylinder_volume = pf_shape_volume(&cylinder_shape);
    if (!pf_number(box_volume) || !pf_number(sphere_volume)
            || !pf_number(cylinder_volume)) return false;
    hull->box = (PfStlPrimitive){box_shape, box_volume, 0.0f};
    hull->sphere = (PfStlPrimitive){sphere_shape, sphere_volume, 0.0f};
    hull->cylinder = (PfStlPrimitive){cylinder_shape, cylinder_volume, 0.0f};
    return true;
}

/* Ratios and primitive selection are a separate step: the reference volume
 * differs between a solid hull and a bare point cloud, and choosing it before
 * the shapes exist is what made small valid meshes fail on scale alone. */
static inline bool pf_stl_primitive_ratios(PfStlHull* hull, float reference) {
    if (hull == NULL || !pf_number(reference) || reference <= 0.0f
            || hull->box.volume <= 0.0f) return false;
    hull->box.volume_ratio = hull->box.volume / reference;
    hull->sphere.volume_ratio = hull->sphere.volume / reference;
    hull->cylinder.volume_ratio = hull->cylinder.volume / reference;
    hull->selected_primitive = PF_BOX;
    hull->selected_ratio = hull->box.volume_ratio;
    if (hull->sphere.volume_ratio < hull->selected_ratio) {
        hull->selected_primitive = PF_SPHERE;
        hull->selected_ratio = hull->sphere.volume_ratio;
    }
    if (hull->cylinder.volume_ratio < hull->selected_ratio) {
        hull->selected_primitive = PF_CYLINDER;
        hull->selected_ratio = hull->cylinder.volume_ratio;
    }
    return pf_number(hull->selected_ratio) && pf_number(hull->box.volume_ratio)
        && pf_number(hull->sphere.volume_ratio) && pf_number(hull->cylinder.volume_ratio);
}

/* A solid hull must be enclosed by the primitive that was fitted to it. */
static inline bool pf_stl_primitive_encloses(PfStlHull* hull) {
    return hull->selected_ratio >= 1.0f - 1.0e-5f;
}

static inline bool pf_stl_unique_vertices(const PfStlMesh* mesh, PfVec3** out,
        int* count) {
    if (mesh == NULL || out == NULL || count == NULL || mesh->vertices == NULL
            || mesh->triangle_count <= 0) return false;
    size_t expanded = 0;
    if (!pf_stl_mul_size((size_t)mesh->triangle_count, (size_t)3, &expanded)
            || expanded > (size_t)INT_MAX) return false;
    PfStlPointSort* sorted = (PfStlPointSort*)pf_stl_alloc_array(expanded,
        sizeof(PfStlPointSort));
    PfVec3* unique = (PfVec3*)pf_stl_alloc_array(expanded, sizeof(PfVec3));
    if (sorted == NULL || unique == NULL) {
        free(sorted);
        free(unique);
        return false;
    }
    for (size_t i = 0; i < expanded; ++i) {
        if (!pf_vec_valid(mesh->vertices[i])) {
            free(sorted);
            free(unique);
            return false;
        }
        sorted[i].point = mesh->vertices[i];
        sorted[i].triangle_vertex = (int)i;
    }
    qsort(sorted, expanded, sizeof(PfStlPointSort), pf_stl_point_compare);
    int unique_count = 0;
    for (size_t i = 0; i < expanded; ++i) {
        if (unique_count == 0 || sorted[i].point.x != unique[unique_count - 1].x
                || sorted[i].point.y != unique[unique_count - 1].y
                || sorted[i].point.z != unique[unique_count - 1].z) {
            unique[unique_count++] = sorted[i].point;
        }
    }
    free(sorted);
    *out = unique;
    *count = unique_count;
    return true;
}

static inline bool pf_stl_hull_closed(const PfStlHullFace* faces, int count) {
    if (faces == NULL || count <= 0) return false;
    size_t expanded = 0;
    if (!pf_stl_mul_size((size_t)count, (size_t)3, &expanded)) return false;
    PfStlEdge* edges = (PfStlEdge*)pf_stl_alloc_array(expanded, sizeof(PfStlEdge));
    if (edges == NULL) return false;
    size_t cursor = 0;
    for (int i = 0; i < count; ++i) {
        edges[cursor++] = (PfStlEdge){faces[i].a, faces[i].b};
        edges[cursor++] = (PfStlEdge){faces[i].b, faces[i].c};
        edges[cursor++] = (PfStlEdge){faces[i].c, faces[i].a};
    }
    qsort(edges, expanded, sizeof(PfStlEdge), pf_stl_edge_undirected_compare);
    size_t i = 0;
    while (i < expanded) {
        size_t j = i + 1;
        while (j < expanded && !pf_stl_edge_undirected_compare(&edges[i], &edges[j])) ++j;
        if (j - i != 2 || edges[i].a != edges[i + 1].b
                || edges[i].b != edges[i + 1].a) {
            free(edges);
            return false;
        }
        i = j;
    }
    free(edges);
    return true;
}

static inline bool pf_stl_hull_from_mesh(const PfStlMesh* mesh, PfStlHull* out) {
    if (mesh == NULL || out == NULL) return false;
    PfStlHull hull = {0};
    double interior[3] = {0.0, 0.0, 0.0};
    int p0 = 0;
    int p1 = 0;
    double farthest = -1.0;
    int p2 = -1;
    double best_area = 0.0;
    double normal[3] = {0.0, 0.0, 0.0};
    double cross[3] = {0.0, 0.0, 0.0};
    double normal_length = 0.0;
    int p3 = -1;
    double largest_plane = 0.0;
    double scale = 0.0;
    double seed[3] = {0.0, 0.0, 0.0};
    double plane_epsilon = 0.0;
    double area_epsilon = 0.0;
    PfStlHullFaceInternal* faces = NULL;
    size_t face_capacity = 0;
    unsigned char* visible = NULL;
    size_t visible_capacity = 0;
    PfStlEdge* edges = NULL;
    size_t edge_capacity = 0;
    size_t face_count = 0;
    if (!pf_stl_unique_vertices(mesh, &hull.vertices, &hull.vertex_count)
            || hull.vertex_count < 4) goto failure;
    for (int i = 0; i < hull.vertex_count; ++i) {
        interior[0] += hull.vertices[i].x;
        interior[1] += hull.vertices[i].y;
        interior[2] += hull.vertices[i].z;
    }
    for (int component = 0; component < 3; ++component) interior[component] /= hull.vertex_count;
    if (pf_stl_box_hull(&hull, interior)) {
        if (!pf_stl_primitive_ratios(&hull, hull.mass.volume)
                || !pf_stl_primitive_encloses(&hull)) {
            pf_stl_free_hull(&hull);
            return false;
        }
        *out = hull;
        return true;
    }
    for (int i = 1; i < hull.vertex_count; ++i) {
        double d[3] = {(double)hull.vertices[i].x - hull.vertices[p0].x,
            (double)hull.vertices[i].y - hull.vertices[p0].y,
            (double)hull.vertices[i].z - hull.vertices[p0].z};
        double distance = pf_stl_dot3(d, d);
        if (distance > farthest) { farthest = distance; p1 = i; }
    }
    for (int i = 0; i < hull.vertex_count; ++i) {
        double a[3] = {(double)hull.vertices[p1].x - hull.vertices[p0].x,
            (double)hull.vertices[p1].y - hull.vertices[p0].y,
            (double)hull.vertices[p1].z - hull.vertices[p0].z};
        double b[3] = {(double)hull.vertices[i].x - hull.vertices[p0].x,
            (double)hull.vertices[i].y - hull.vertices[p0].y,
            (double)hull.vertices[i].z - hull.vertices[p0].z};
        double n[3];
        pf_stl_cross3(a, b, n);
        double area = pf_stl_dot3(n, n);
        if (area > best_area) { best_area = area; p2 = i; }
    }
    if (p2 < 0 || best_area <= 1.0e-20) goto failure;
    normal[0] = (double)hull.vertices[p1].x - hull.vertices[p0].x;
    normal[1] = (double)hull.vertices[p1].y - hull.vertices[p0].y;
    normal[2] = (double)hull.vertices[p1].z - hull.vertices[p0].z;
    pf_stl_cross3(normal, (double[3]){(double)hull.vertices[p2].x - hull.vertices[p0].x,
        (double)hull.vertices[p2].y - hull.vertices[p0].y,
        (double)hull.vertices[p2].z - hull.vertices[p0].z}, cross);
    normal_length = pf_stl_length3(cross);
    if (!isfinite(normal_length) || normal_length <= 1.0e-20) goto failure;
    for (int i = 0; i < 3; ++i) normal[i] = cross[i] / normal_length;
    for (int i = 0; i < hull.vertex_count; ++i) {
        double d[3] = {(double)hull.vertices[i].x - hull.vertices[p0].x,
            (double)hull.vertices[i].y - hull.vertices[p0].y,
            (double)hull.vertices[i].z - hull.vertices[p0].z};
        double distance = fabs(pf_stl_dot3(normal, d));
        if (distance > largest_plane) { largest_plane = distance; p3 = i; }
    }
    for (int i = 0; i < 4; ++i) {
        const int index = i == 0 ? p0 : i == 1 ? p1 : i == 2 ? p2 : p3;
        seed[0] += hull.vertices[index].x;
        seed[1] += hull.vertices[index].y;
        seed[2] += hull.vertices[index].z;
    }
    for (int i = 0; i < 3; ++i) seed[i] *= 0.25;
    if (p3 < 0 || largest_plane <= 1.0e-20) {
        hull.mass = (PfStlMassProperties){0};
        hull.mass.centroid = pf_v3((float)interior[0], (float)interior[1], (float)interior[2]);
        *out = hull;
        return true;
    }
    for (int i = 0; i < hull.vertex_count; ++i) {
        double d[3] = {(double)hull.vertices[i].x - interior[0],
            (double)hull.vertices[i].y - interior[1],
            (double)hull.vertices[i].z - interior[2]};
        double distance = pf_stl_length3(d);
        if (distance > scale) scale = distance;
    }
    plane_epsilon = fmax(1.0e-20, scale * 1.0e-12);
    area_epsilon = fmax(1.0e-30, scale * scale * 1.0e-14);
    if (!pf_stl_grow_array((void**)&faces, sizeof(*faces), &face_capacity, 8)
            || !pf_stl_indexed_face(hull.vertices, p0, p1, p2,
                (double[3]){hull.vertices[p3].x, hull.vertices[p3].y, hull.vertices[p3].z},
                area_epsilon, &faces[0])
            || !pf_stl_indexed_face(hull.vertices, p0, p3, p1,
                (double[3]){hull.vertices[p2].x, hull.vertices[p2].y, hull.vertices[p2].z},
                area_epsilon, &faces[1])
            || !pf_stl_indexed_face(hull.vertices, p1, p3, p2,
                (double[3]){hull.vertices[p0].x, hull.vertices[p0].y, hull.vertices[p0].z},
                area_epsilon, &faces[2])
            || !pf_stl_indexed_face(hull.vertices, p2, p3, p0,
                (double[3]){hull.vertices[p1].x, hull.vertices[p1].y, hull.vertices[p1].z},
                area_epsilon, &faces[3])) goto failure;
    face_count = 4;
    for (int point_index = 0; point_index < hull.vertex_count; ++point_index) {
        if (point_index == p0 || point_index == p1 || point_index == p2 || point_index == p3) continue;
        if (!pf_stl_grow_array((void**)&visible, 1, &visible_capacity, face_count)) goto failure;
        size_t visible_count = 0;
        for (size_t face = 0; face < face_count; ++face) {
            PfVec3 a = hull.vertices[faces[face].a];
            PfVec3 b = hull.vertices[faces[face].b];
            PfVec3 c = hull.vertices[faces[face].c];
            double distance = faces[face].normal[0] * (hull.vertices[point_index].x - a.x)
                + faces[face].normal[1] * (hull.vertices[point_index].y - a.y)
                + faces[face].normal[2] * (hull.vertices[point_index].z - a.z);
            visible[face] = distance > plane_epsilon
                || (fabs(distance) <= plane_epsilon
                    && !pf_stl_point_in_face(a, b, c, hull.vertices[point_index]));
            if (visible[face]) ++visible_count;
        }
        if (visible_count == 0) continue;
        size_t edge_need = 0;
        if (!pf_stl_mul_size(visible_count, (size_t)3, &edge_need)
                || !pf_stl_grow_array((void**)&edges, sizeof(*edges), &edge_capacity, edge_need)) goto failure;
        size_t edge_count = 0;
        for (size_t face = 0; face < face_count; ++face) if (visible[face]) {
            edges[edge_count++] = (PfStlEdge){faces[face].a, faces[face].b};
            edges[edge_count++] = (PfStlEdge){faces[face].b, faces[face].c};
            edges[edge_count++] = (PfStlEdge){faces[face].c, faces[face].a};
        }
        /* Horizon edges are grouped by UNDIRECTED key. A directed sort leaves
         * (a,b) and (b,a) apart whenever other edges share either endpoint, so
         * internal edges survive cancellation and corrupt the face set. */
        qsort(edges, edge_count, sizeof(*edges), pf_stl_edge_undirected_compare);
        size_t horizon = 0;
        for (size_t edge = 0; edge < edge_count;) {
            size_t group = 1;
            while (edge + group < edge_count
                    && !pf_stl_edge_undirected_compare(&edges[edge], &edges[edge + group])) ++group;
            if (group == 1) {
                edges[horizon++] = edges[edge];
            } else if (group != 2 || edges[edge].a != edges[edge + 1].b
                    || edges[edge].b != edges[edge + 1].a) {
                goto failure;
            }
            edge += group;
        }
        edge_count = horizon;
        if (edge_count == 0) goto failure;
        if (!pf_stl_grow_array((void**)&faces, sizeof(*faces), &face_capacity,
                face_count - visible_count + edge_count)) goto failure;
        size_t write = 0;
        for (size_t face = 0; face < face_count; ++face) if (!visible[face]) {
            faces[write++] = faces[face];
        }
        size_t base = write;
        for (size_t edge = 0; edge < edge_count; ++edge) {
            PfVec3 a = hull.vertices[edges[edge].a];
            PfVec3 b = hull.vertices[edges[edge].b];
            PfVec3 c = hull.vertices[point_index];
            double points[3][3] = {{a.x, a.y, a.z}, {b.x, b.y, b.z}, {c.x, c.y, c.z}};
            double e0[3], e1[3], face_normal[3];
            for (int component = 0; component < 3; ++component) {
                e0[component] = points[1][component] - points[0][component];
                e1[component] = points[2][component] - points[0][component];
            }
            pf_stl_cross3(e0, e1, face_normal);
            double length = pf_stl_length3(face_normal);
            if (!isfinite(length) || length <= area_epsilon) { continue; }
            for (int component = 0; component < 3; ++component) face_normal[component] /= length;
            double side[3];
            for (int component = 0; component < 3; ++component) {
                side[component] = seed[component] - points[0][component];
            }
            int face_a = edges[edge].a;
            int face_b = edges[edge].b;
            int face_c = point_index;
            if (pf_stl_dot3(face_normal, side) > 0.0) {
                int temporary = face_b;
                face_b = face_c;
                face_c = temporary;
                for (int component = 0; component < 3; ++component) face_normal[component] *= -1.0;
            }
            if (!pf_stl_grow_array((void**)&faces, sizeof(*faces), &face_capacity, base + 1)) goto failure;
            faces[base++] = (PfStlHullFaceInternal){face_a, face_b, face_c,
                {face_normal[0], face_normal[1], face_normal[2]}};
        }
        if (base < 4) goto failure;
        face_count = base;
    }
    hull.faces = (PfStlHullFace*)pf_stl_alloc_array(face_count, sizeof(PfStlHullFace));
    if (hull.faces == NULL) goto failure;
    for (size_t i = 0; i < face_count; ++i) {
        hull.faces[i] = (PfStlHullFace){faces[i].a, faces[i].b, faces[i].c};
    }
    hull.face_count = (int)face_count;
    free(faces);
    free(visible);
    free(edges);
    if (hull.face_count > INT_MAX || !pf_stl_hull_closed(hull.faces, hull.face_count)
            || !pf_stl_hull_mass(&hull) || !pf_stl_primitive_fit(&hull, interior)
            || !pf_stl_primitive_ratios(&hull, hull.mass.volume)
            || !pf_stl_primitive_encloses(&hull)) {
        pf_stl_free_hull(&hull);
        return false;
    }
    *out = hull;
    return true;
failure:
    free(faces);
    free(visible);
    free(edges);
    pf_stl_free_hull(&hull);
    return false;
}

/* Hull-free collision fitting. The ratio reference is the fitted OBB volume;
 * mass tensor fields remain zero because a point cloud has no solid interior. */
static inline bool pf_stl_fit_primitives_hull_free(const PfStlMesh* mesh,
        PfStlHull* out) {
    if (mesh == NULL || out == NULL) return false;
    PfStlHull result = {0};
    if (!pf_stl_unique_vertices(mesh, &result.vertices, &result.vertex_count)) return false;
    double interior[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < result.vertex_count; ++i) {
        interior[0] += result.vertices[i].x;
        interior[1] += result.vertices[i].y;
        interior[2] += result.vertices[i].z;
    }
    for (int i = 0; i < 3; ++i) interior[i] /= result.vertex_count;
    if (!pf_stl_primitive_fit(&result, interior)) {
        pf_stl_free_hull(&result);
        return false;
    }
    /* No unit-volume stand-in: a point cloud has no solid interior, so the
     * fitted OBB is the reference the ratios are measured against. */
    float reference = result.box.volume;
    if (!pf_stl_primitive_ratios(&result, reference)) {
        pf_stl_free_hull(&result);
        return false;
    }
    result.mass.volume = reference;
    result.mass.centroid = result.box.shape.position;
    result.face_count = 0;
    *out = result;
    return true;
}

static inline PfStlPrimitive pf_stl_selected_primitive(const PfStlHull* hull) {
    if (hull == NULL) {
        PfStlPrimitive empty = {};
        return empty;
    }
    if (hull->selected_primitive == PF_SPHERE) return hull->sphere;
    if (hull->selected_primitive == PF_CYLINDER) return hull->cylinder;
    return hull->box;
}

/* Direct legacy bodies support boxes and spheres. A selected cylinder is
 * valid as PfShape but must be installed in the compound sidecar first. */
static inline bool pf_stl_hull_body(const PfStlHull* hull, float mass,
        float friction, float restitution, PfBody* out) {
    if (hull == NULL || out == NULL || !pf_number(mass) || mass <= 0.0f
            || !pf_number(friction) || friction < 0.0f
            || !pf_number(restitution) || restitution < 0.0f || restitution > 1.0f) return false;
    PfStlPrimitive primitive = pf_stl_selected_primitive(hull);
    if (primitive.shape.kind != PF_BOX && primitive.shape.kind != PF_SPHERE) return false;
    PfBody result = {0};
    bool valid = primitive.shape.kind == PF_SPHERE
        ? pf_sphere(&result, PF_DYNAMIC, primitive.shape.half_extents.x, mass,
            primitive.shape.position, friction, restitution)
        : pf_box(&result, PF_DYNAMIC, primitive.shape.half_extents, mass,
            primitive.shape.position, primitive.shape.rotation, friction, restitution);
    if (!valid) return false;
    *out = result;
    return true;
}
