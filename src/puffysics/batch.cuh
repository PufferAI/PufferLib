#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <string.h>

#include "math.cuh"
#include "types.cuh"
#include "shapes.cuh"

static inline bool pf_size_mul(size_t a, size_t b, size_t* result) {
    if (b != 0 && a > (size_t)-1 / b) {
        return false;
    }
    *result = a * b;
    return true;
}
static inline void pf_batch_destroy(PfBatch* batch);

/* Optional sidecar: shape index is env*compound_shape_stride +
 * body*PF_MAX_SHAPES_PER_BODY + slot; counts are env*body_capacity + body. */
static inline bool pf_batch_enable_compound_shapes(PfBatch* batch) {
    if (batch == NULL || batch->env_count <= 0 || batch->body_capacity <= 0
            || batch->compound_shapes != NULL || batch->compound_shape_counts != NULL) return false;
    size_t shape_stride = 0, shape_total = 0, count_total = 0;
    if (!pf_size_mul((size_t)batch->body_capacity, (size_t)PF_MAX_SHAPES_PER_BODY, &shape_stride)
            || !pf_size_mul((size_t)batch->env_count, shape_stride, &shape_total)
            || !pf_size_mul((size_t)batch->env_count, (size_t)batch->body_capacity, &count_total)
            || shape_total > (size_t)-1 / sizeof(PfShape)
            || count_total > (size_t)-1 / sizeof(int)) return false;
    batch->compound_shape_capacity = (int)shape_stride;
    batch->compound_shape_stride = shape_stride;
    if (cudaMalloc((void**)&batch->compound_shapes, shape_total * sizeof(PfShape)) != cudaSuccess
            || cudaMalloc((void**)&batch->compound_shape_counts, count_total * sizeof(int)) != cudaSuccess
            || cudaMemset(batch->compound_shapes, 0, shape_total * sizeof(PfShape)) != cudaSuccess
            || cudaMemset(batch->compound_shape_counts, 0, count_total * sizeof(int)) != cudaSuccess) {
        if (batch->compound_shape_counts != NULL) cudaFree(batch->compound_shape_counts);
        if (batch->compound_shapes != NULL) cudaFree(batch->compound_shapes);
        batch->compound_shape_counts = NULL;
        batch->compound_shapes = NULL;
        batch->compound_shape_capacity = 0;
        batch->compound_shape_stride = 0;
        return false;
    }
    return true;
}
static inline bool pf_batch_create_with_soft(PfBatch* batch, int env_count,
        int body_capacity, int particle_capacity, int distance_capacity,
        int volume_capacity) {
    if (batch == NULL) {
        return false;
    }
    memset(batch, 0, sizeof(*batch));
    if (env_count <= 0 || body_capacity <= 0 || particle_capacity < 0
            || distance_capacity < 0 || volume_capacity < 0) {
        return false;
    }
    size_t body_stride = (size_t)body_capacity;
    /* One body PAIR can emit up to PF_MAX_SHAPES_PER_BODY squared candidate
     * patches before the narrow phase merges them, so the manifold array must
     * hold that many per pair, not one. Scaling happens here, once, with the
     * same overflow-checked pf_size_mul convention as every other buffer, so
     * pf_detect_contacts bounds itself against a capacity that is actually
     * allocated. */
    size_t pair_count = 0;
    if (!pf_size_mul(body_stride, body_stride - 1, &pair_count)) {
        return false;
    }
    size_t manifold_stride = 0;
    if (!pf_size_mul(pair_count / 2, (size_t)PF_MAX_SHAPES_PER_BODY
            * (size_t)PF_MAX_SHAPES_PER_BODY, &manifold_stride)
            || manifold_stride > 0x7fffffffu) {
        return false;
    }
    size_t body_count = 0;
    size_t manifold_count = 0;
    size_t particle_count = 0;
    size_t distance_count = 0;
    size_t volume_count = 0;
    if (!pf_size_mul((size_t)env_count, body_stride, &body_count)
            || !pf_size_mul((size_t)env_count, manifold_stride,
                &manifold_count)
            || !pf_size_mul((size_t)env_count, (size_t)particle_capacity,
                &particle_count)
            || !pf_size_mul((size_t)env_count, (size_t)distance_capacity,
                &distance_count)
            || !pf_size_mul((size_t)env_count, (size_t)volume_capacity,
                &volume_count)
            || body_count > (size_t)-1 / sizeof(PfBody)
            || manifold_count > (size_t)-1 / sizeof(PfManifold)
            || particle_count > (size_t)-1 / sizeof(PfSoftParticle)
            || distance_count
                > (size_t)-1 / sizeof(PfSoftDistanceConstraint)
            || volume_count
                > (size_t)-1 / sizeof(PfSoftVolumeConstraint)) {
        return false;
    }
    batch->env_count = env_count;
    batch->body_capacity = body_capacity;
    batch->particle_capacity = particle_capacity;
    batch->distance_capacity = distance_capacity;
    batch->volume_capacity = volume_capacity;
    batch->body_stride = body_stride;
    batch->manifold_stride = manifold_stride;
    batch->particle_stride = (size_t)particle_capacity;
    batch->distance_stride = (size_t)distance_capacity;
    batch->volume_stride = (size_t)volume_capacity;
    if (cudaMalloc((void**)&batch->bodies, body_count * sizeof(PfBody))
            != cudaSuccess) {
        batch->bodies = NULL;
        return false;
    }
    if (manifold_count != 0
            && cudaMalloc((void**)&batch->manifolds,
                manifold_count * sizeof(PfManifold)) != cudaSuccess) {
        pf_batch_destroy(batch);
        return false;
    }
    if (particle_count != 0
            && cudaMalloc((void**)&batch->soft_particles,
                particle_count * sizeof(PfSoftParticle)) != cudaSuccess) {
        pf_batch_destroy(batch);
        return false;
    }
    if (distance_count != 0
            && cudaMalloc((void**)&batch->soft_distance_constraints,
                distance_count * sizeof(PfSoftDistanceConstraint))
                != cudaSuccess) {
        pf_batch_destroy(batch);
        return false;
    }
    if (volume_count != 0
            && cudaMalloc((void**)&batch->soft_volume_constraints,
                volume_count * sizeof(PfSoftVolumeConstraint)) != cudaSuccess) {
        pf_batch_destroy(batch);
        return false;
    }
    if (cudaMemset(batch->bodies, 0, body_count * sizeof(PfBody)) != cudaSuccess
            || (manifold_count != 0 && cudaMemset(batch->manifolds, 0,
                manifold_count * sizeof(PfManifold)) != cudaSuccess)
            || (particle_count != 0 && cudaMemset(batch->soft_particles, 0,
                particle_count * sizeof(PfSoftParticle)) != cudaSuccess)
            || (distance_count != 0
                && cudaMemset(batch->soft_distance_constraints, 0,
                    distance_count * sizeof(PfSoftDistanceConstraint))
                    != cudaSuccess)
            || (volume_count != 0 && cudaMemset(batch->soft_volume_constraints,
                0, volume_count * sizeof(PfSoftVolumeConstraint)) != cudaSuccess)) {
        pf_batch_destroy(batch);
        return false;
    }
    return true;
}

static inline bool pf_batch_create(
        PfBatch* batch, int env_count, int body_capacity) {
    return pf_batch_create_with_soft(batch, env_count, body_capacity, 0, 0, 0);
}

static inline void pf_batch_destroy(PfBatch* batch) {
    if (batch == NULL) {
        return;
    }
    if (batch->soft_volume_constraints != NULL) {
        cudaFree(batch->soft_volume_constraints);
    }
    if (batch->soft_distance_constraints != NULL) {
        cudaFree(batch->soft_distance_constraints);
    }
    if (batch->soft_particles != NULL) {
        cudaFree(batch->soft_particles);
    }
    if (batch->manifolds != NULL) {
        cudaFree(batch->manifolds);
    }
    if (batch->bodies != NULL) {
        cudaFree(batch->bodies);
    }
    if (batch->compound_shape_counts != NULL) {
        cudaFree(batch->compound_shape_counts);
    }
    if (batch->compound_shapes != NULL) {
        cudaFree(batch->compound_shapes);
    }
    memset(batch, 0, sizeof(*batch));
}

__device__ static inline bool pf_world(
        const PfBatch& batch, int env_index, int body_count, PfWorld* out) {
    if (out == NULL || env_index < 0 || env_index >= batch.env_count
            || body_count < 0 || body_count > batch.body_capacity) {
        return false;
    }
    out->bodies = batch.bodies == NULL ? NULL : batch.bodies
        + (size_t)env_index * batch.body_stride;
    out->compound_shapes = batch.compound_shapes == NULL ? NULL : batch.compound_shapes
        + (size_t)env_index * batch.compound_shape_stride;
    out->compound_shape_counts = batch.compound_shape_counts == NULL ? NULL
        : batch.compound_shape_counts + (size_t)env_index * batch.body_capacity;
    out->manifolds = batch.manifolds == NULL ? NULL : batch.manifolds
        + (size_t)env_index * batch.manifold_stride;
    out->body_count = body_count;
    out->manifold_capacity = (int)batch.manifold_stride;
    out->manifold_count = 0;
    out->env_index = env_index;
    return true;
}

__device__ static inline bool pf_soft_world(const PfBatch& batch,
        int env_index, int particle_count, int distance_count,
        int volume_count, PfSoftWorld* out) {
    if (out == NULL || env_index < 0 || env_index >= batch.env_count
            || particle_count < 0 || particle_count > batch.particle_capacity
            || distance_count < 0 || distance_count > batch.distance_capacity
            || volume_count < 0 || volume_count > batch.volume_capacity) {
        return false;
    }
    out->particles = batch.soft_particles == NULL ? NULL : batch.soft_particles
        + (size_t)env_index * batch.particle_stride;
    out->distance_constraints = batch.soft_distance_constraints == NULL ? NULL
        : batch.soft_distance_constraints + (size_t)env_index * batch.distance_stride;
    out->volume_constraints = batch.soft_volume_constraints == NULL ? NULL
        : batch.soft_volume_constraints + (size_t)env_index * batch.volume_stride;
    out->particle_count = particle_count;
    out->distance_count = distance_count;
    out->volume_count = volume_count;
    out->particle_capacity = batch.particle_capacity;
    out->distance_capacity = batch.distance_capacity;
    out->volume_capacity = batch.volume_capacity;
    out->env_index = env_index;
    return true;
}

__device__ static inline bool pf_soft_add_particle(PfSoftWorld* soft,
        PfVec3 position, float mass) {
    if (soft == NULL || soft->particles == NULL
            || soft->particle_count < 0
            || soft->particle_count >= soft->particle_capacity
            || !pf_vec_valid(position) || !pf_number(mass) || mass <= 0.0f) {
        return false;
    }
    PfSoftParticle particle = {position, position,
        {0.0f, 0.0f, 0.0f}, 1.0f / mass};
    soft->particles[soft->particle_count++] = particle;
    return true;
}

__device__ static inline bool pf_soft_add_constraint(PfSoftWorld* soft,
        int p0, int p1, float rest_length, float compliance) {
    // Each array is bounded on its own: a full particle array must still
    // accept constraints, and a full constraint array must still accept
    // particles. The particle count is only a validity bound for the two
    // indices, never a capacity bound for the write below.
    if (soft == NULL || soft->distance_constraints == NULL
            || soft->particles == NULL
            || soft->distance_count < 0
            || soft->distance_count >= soft->distance_capacity
            || soft->particle_count < 0
            || soft->particle_count > soft->particle_capacity
            || p0 < 0 || p1 < 0 || p0 >= soft->particle_count
            || p1 >= soft->particle_count || p0 == p1
            || !pf_number(rest_length) || rest_length <= 0.0f
            || !pf_number(compliance) || compliance < 0.0f) {
        return false;
    }
    PfSoftDistanceConstraint constraint = {p0, p1, rest_length, compliance, 0.0f};
    soft->distance_constraints[soft->distance_count++] = constraint;
    return true;
}

__device__ static inline bool pf_soft_add_volume_constraint(PfSoftWorld* soft,
        int p0, int p1, int p2, int p3, float rest_volume,
        float compliance) {
    if (soft == NULL || soft->volume_constraints == NULL
            || soft->particles == NULL
            || soft->volume_count < 0
            || soft->volume_count >= soft->volume_capacity
            || soft->particle_count < 0
            || soft->particle_count > soft->particle_capacity
            || p0 < 0 || p1 < 0 || p2 < 0 || p3 < 0
            || p0 >= soft->particle_count || p1 >= soft->particle_count
            || p2 >= soft->particle_count || p3 >= soft->particle_count
            || p0 == p1 || p0 == p2 || p0 == p3 || p1 == p2 || p1 == p3
            || p2 == p3
            || !pf_number(rest_volume) || rest_volume <= 0.0f
            || !pf_number(compliance) || compliance < 0.0f) {
        return false;
    }
    double a[3] = {soft->particles[p0].position.x, soft->particles[p0].position.y, soft->particles[p0].position.z};
    double b[3] = {soft->particles[p1].position.x - a[0], soft->particles[p1].position.y - a[1], soft->particles[p1].position.z - a[2]};
    double c[3] = {soft->particles[p2].position.x - a[0], soft->particles[p2].position.y - a[1], soft->particles[p2].position.z - a[2]};
    double d[3] = {soft->particles[p3].position.x - a[0], soft->particles[p3].position.y - a[1], soft->particles[p3].position.z - a[2]};
    double det = b[0] * (c[1] * d[2] - c[2] * d[1])
        - b[1] * (c[0] * d[2] - c[2] * d[0])
        + b[2] * (c[0] * d[1] - c[1] * d[0]);
    if (!pf_number((float)det) || fabs(det) <= 1.0e-20) {
        return false;
    }
    // The API carries a positive rest volume and the solver compares it with
    // the signed current volume, so the stored winding must be positive too.
    // Swapping two vertex indices negates the determinant; taking the
    // absolute current volume in the solver instead would hide inversion.
    int head = det < 0.0 ? p2 : p1;
    int second = det < 0.0 ? p1 : p2;
    PfSoftVolumeConstraint constraint = {p0, head, second, p3, rest_volume,
        compliance, 0.0f};
    soft->volume_constraints[soft->volume_count++] = constraint;
    return true;
}
