#pragma once

#include "shapes.cuh"
#include "collision_shapes.cuh"

__device__ static inline void pf_manifold_tangents(PfManifold* manifold) {
    PfVec3 reference = fabsf(manifold->normal.y) < 0.9f
        ? pf_v3(0.0f, 1.0f, 0.0f) : pf_v3(1.0f, 0.0f, 0.0f);
    manifold->tangent_1 = pf_normalize_or(
        pf_cross(reference, manifold->normal), pf_v3(0.0f, 0.0f, 1.0f));
    manifold->tangent_2 = pf_normalize_or(
        pf_cross(manifold->normal, manifold->tangent_1),
        pf_v3(0.0f, 1.0f, 0.0f));
}

__device__ static inline void pf_manifold_material(
        PfManifold* manifold, const PfBody* a, const PfBody* b) {
    manifold->static_friction = sqrtf(a->friction * b->friction);
    manifold->dynamic_friction = manifold->static_friction;
    manifold->restitution = a->restitution > b->restitution
        ? a->restitution : b->restitution;
    pf_manifold_tangents(manifold);
}


/* A compound pair can touch at several places at once, and a body pair owns
 * one manifold per retained patch, not one per pair. Every primitive pair is
 * tested on its own: the body-level radius test below is a conservative
 * early-out over the whole body, so it can never reject a component pair
 * that does overlap, and `pf_shape_contact` then decides each pair
 * individually.
 *
 * Retained-contact policy, since the storage is bounded:
 *  - A new patch is MERGED into an existing manifold only when it is
 *    compatible: normals equal within floating-point tolerance and
 *    contact centroids within `PF_COMPOUND_MERGE_DISTANCE` of each other.
 *    Anything else is a genuinely separate constraint and is kept as its own
 *    manifold, up to `capacity`.
 *  - When capacity is exhausted the remaining shape pairs are DROPPED, not
 *    resolved: the caller is told how many through the out-parameter so the
 *    omission can be reported rather than silently treated as contact-free.
 *  - Drop order is the ascending shape-slot loop order, so which patches
 *    survive is deterministic for a given scene and does not depend on
 *    floating-point ties. */

#define PF_COMPOUND_MERGE_ANGLE 0.2f
#define PF_COMPOUND_MERGE_NORMAL_TOLERANCE 1.0e-5f
#define PF_COMPOUND_MERGE_DISTANCE 0.05f

__device__ static inline PfVec3 pf_manifold_centroid(
        const PfManifold* manifold) {
    PfVec3 total = pf_v3(0.0f, 0.0f, 0.0f);
    for (int index = 0; index < manifold->point_count; ++index) {
        total = pf_add(total, manifold->points[index].point_a);
    }
    return pf_scale(total, 1.0f / (float)manifold->point_count);
}

__device__ static inline bool pf_manifold_merges(
        const PfManifold* target, const PfManifold* candidate) {
    // Different face normals represent different constraints, even at a seam.
    if (pf_length_squared(pf_sub(target->normal, candidate->normal))
            > PF_COMPOUND_MERGE_NORMAL_TOLERANCE * PF_COMPOUND_MERGE_NORMAL_TOLERANCE) {
        return false;
    }
    return pf_length_squared(pf_sub(pf_manifold_centroid(target),
        pf_manifold_centroid(candidate)))
        < PF_COMPOUND_MERGE_DISTANCE * PF_COMPOUND_MERGE_DISTANCE;
}

__device__ static inline void pf_manifold_merge(
        PfManifold* target, const PfManifold* candidate) {
    for (int index = 0; index < candidate->point_count; ++index) {
        bool duplicate = false;
        for (int slot = 0; slot < target->point_count; ++slot) {
            duplicate = duplicate || pf_length_squared(pf_sub(
                target->points[slot].point_a, candidate->points[index].point_a))
                < 1.0e-12f;
        }
        if (duplicate || target->point_count >= PF_MAX_MANIFOLD_POINTS) {
            continue;
        }
        PfContactPoint point = candidate->points[index];
        point.separation = pf_dot(pf_sub(point.point_a, point.point_b), target->normal);
        target->points[target->point_count++] = point;
    }
}

/* Conservative whole-body early-out. It bounds the body, not the individual
 * component pair, so it can only skip a body pair whose shapes cannot touch
 * at all; every component pair that survives is still tested on its own. */
__device__ static inline bool pf_compound_body_may_touch(
        const PfBody* body_a, const PfShape* shapes_a, int count_a,
        const PfBody* body_b, const PfShape* shapes_b, int count_b) {
    float radius_a = 0.0f, radius_b = 0.0f;
    for (int i = 0; i < count_a; ++i) {
        PfShapeWorld world;
        if (!pf_shape_world(body_a, &shapes_a[i], &world)) return false;
        radius_a = fmaxf(radius_a, pf_length(pf_sub(world.center, body_a->position))
            + pf_shape_radius(&world));
    }
    for (int i = 0; i < count_b; ++i) {
        PfShapeWorld world;
        if (!pf_shape_world(body_b, &shapes_b[i], &world)) return false;
        radius_b = fmaxf(radius_b, pf_length(pf_sub(world.center, body_b->position))
            + pf_shape_radius(&world));
    }
    return pf_length_squared(pf_sub(body_b->position, body_a->position))
        <= (radius_a + radius_b) * (radius_a + radius_b);
}

__device__ static inline int pf_collision_compound_multi(
        int body_a_index, const PfBody* body_a, int body_b_index,
        const PfBody* body_b, const PfShape* shapes_a, int count_a,
        const PfShape* shapes_b, int count_b, PfManifold* out, int capacity,
        int* dropped) {
    if (out == NULL || dropped == NULL || shapes_a == NULL || shapes_b == NULL
            || count_a <= 0 || count_b <= 0 || capacity <= 0
            || count_a > PF_MAX_SHAPES_PER_BODY
            || count_b > PF_MAX_SHAPES_PER_BODY) return 0;
    int produced = 0;
    if (!pf_compound_body_may_touch(body_a, shapes_a, count_a, body_b,
            shapes_b, count_b)) return 0;
    *dropped = 0;
    for (int ia = 0; ia < count_a; ++ia) {
        PfShapeWorld world_a;
        if (!pf_shape_world(body_a, &shapes_a[ia], &world_a)) continue;
        for (int ib = 0; ib < count_b; ++ib) {
            PfShapeWorld world_b;
            if (!pf_shape_world(body_b, &shapes_b[ib], &world_b)) continue;
            PfManifold candidate = {};
            if (!pf_shape_contact(&world_a, &world_b, &candidate)
                    || candidate.point_count <= 0) continue;
            candidate.body_a = body_a_index;
            candidate.body_b = body_b_index;
            candidate.normal = pf_normalize_or(candidate.normal,
                pf_v3(1.0f, 0.0f, 0.0f));
            pf_manifold_material(&candidate, body_a, body_b);
            int target = -1;
            for (int slot = 0; slot < produced; ++slot) {
                if (pf_manifold_merges(&out[slot], &candidate)) {
                    target = slot;
                    break;
                }
            }
            if (target >= 0) {
                pf_manifold_merge(&out[target], &candidate);
                continue;
            }
            if (produced >= capacity) {
                ++*dropped;
                continue;
            }
            out[produced++] = candidate;
        }
    }
    return produced;
}

/* Deepest shape-pair manifold only. Retained for the single-manifold
 * callers; `pf_detect_contacts` uses the multi-manifold path above. */
__device__ static inline bool pf_collision_compound(
        int body_a_index, const PfBody* body_a, int body_b_index,
        const PfBody* body_b, const PfShape* shapes_a, int count_a,
        const PfShape* shapes_b, int count_b, PfManifold* out) {
    int dropped = 0;
    return pf_collision_compound_multi(body_a_index, body_a, body_b_index,
        body_b, shapes_a, count_a, shapes_b, count_b, out, 1, &dropped) == 1;
}

__device__ static inline bool pf_collision_narrow(
        int body_a_index, const PfBody* body_a,
        int body_b_index, const PfBody* body_b, PfManifold* out) {
    *out = (PfManifold){0};
    out->body_a = body_a_index;
    out->body_b = body_b_index;
    bool hit = false;
    if (body_a->shape == PF_SPHERE && body_b->shape == PF_SPHERE) {
        hit = pf_sphere_sphere_contact(body_a, body_b, out);
    } else if (body_a->shape == PF_SPHERE && body_b->shape == PF_BOX) {
        hit = pf_sphere_box_contact(body_a, body_b, out);
    } else if (body_a->shape == PF_BOX && body_b->shape == PF_SPHERE) {
        hit = pf_sphere_box_contact(body_b, body_a, out);
        if (hit) {
            PfVec3 point = out->points[0].point_a;
            out->points[0].point_a = out->points[0].point_b;
            out->points[0].point_b = point;
            out->normal = pf_scale(out->normal, -1.0f);
        }
    } else if (body_a->shape == PF_BOX && body_b->shape == PF_BOX) {
        PfVec3 axes_a[3];
        PfVec3 axes_b[3];
        pf_quat_axes(body_a->rotation, axes_a);
        pf_quat_axes(body_b->rotation, axes_b);
        PfBoxQuery query;
        hit = pf_box_sat(body_a, body_b, axes_a, axes_b, &query)
            && pf_box_manifold(body_a, body_b, axes_a, axes_b, &query, out);
    } else {
        PfShapeWorld a = {(PfShapeKind)body_a->shape, body_a->position,
            body_a->rotation, body_a->half_extents, pf_point_velocity(body_a, body_a->position)};
        PfShapeWorld b = {(PfShapeKind)body_b->shape, body_b->position,
            body_b->rotation, body_b->half_extents, pf_point_velocity(body_b, body_b->position)};
        hit = pf_shape_contact(&a, &b, out);
    }
    if (!hit || out->point_count <= 0) {
        return false;
    }
    out->normal = pf_normalize_or(out->normal, pf_v3(1.0f, 0.0f, 0.0f));
    pf_manifold_material(out, body_a, body_b);
    return true;
}

__device__ static inline bool pf_collision(
        int body_a_index, const PfBody* body_a,
        int body_b_index, const PfBody* body_b, PfManifold* out) {
    if (out == NULL) {
        return false;
    }
    if (!pf_mode_is_dynamic(body_a->mode)
            && !pf_mode_is_dynamic(body_b->mode)) {
        return false;
    }
    float radius_sum = pf_radius(body_a) + pf_radius(body_b);
    return pf_length_squared(pf_sub(body_b->position, body_a->position))
            <= radius_sum * radius_sum
        && pf_collision_narrow(body_a_index, body_a, body_b_index, body_b, out);
}

__device__ static inline void pf_legacy_shape(const PfBody* body, PfShape* out) {
    out->kind = (PfShapeKind)body->shape;
    out->half_extents = body->half_extents;
    out->position = pf_v3(0.0f, 0.0f, 0.0f);
    out->rotation = pf_quat_identity();
}

__device__ static inline int pf_collision_world_multi(
        int body_a_index, const PfBody* body_a, int body_b_index,
        const PfBody* body_b, const PfWorld* world, PfManifold* out,
        int capacity, int* dropped) {
    if (out == NULL || dropped == NULL || world == NULL || body_a == NULL
            || body_b == NULL || capacity <= 0
            || body_a_index < 0 || body_b_index < 0
            || body_a_index >= world->body_count || body_b_index >= world->body_count
            || (!pf_mode_is_dynamic(body_a->mode)
                && !pf_mode_is_dynamic(body_b->mode))) return 0;
    *dropped = 0;
    int count_a = 0, count_b = 0;
    if (world->compound_shape_counts != NULL) {
        count_a = world->compound_shape_counts[body_a_index];
        count_b = world->compound_shape_counts[body_b_index];
        if (count_a < 0 || count_b < 0
                || count_a > PF_MAX_SHAPES_PER_BODY
                || count_b > PF_MAX_SHAPES_PER_BODY) return 0;
    }
    if (count_a <= 0 && count_b <= 0) {
        return pf_collision(body_a_index, body_a, body_b_index, body_b, out)
            ? 1 : 0;
    }
    if (world->compound_shapes == NULL) return 0;
    PfShape legacy_a, legacy_b;
    const PfShape* shapes_a;
    const PfShape* shapes_b;
    if (count_a > 0) shapes_a = world->compound_shapes
        + (size_t)body_a_index * PF_MAX_SHAPES_PER_BODY;
    else { pf_legacy_shape(body_a, &legacy_a); shapes_a = &legacy_a; count_a = 1; }
    if (count_b > 0) shapes_b = world->compound_shapes
        + (size_t)body_b_index * PF_MAX_SHAPES_PER_BODY;
    else { pf_legacy_shape(body_b, &legacy_b); shapes_b = &legacy_b; count_b = 1; }
    int body_dropped = 0;
    int produced = pf_collision_compound_multi(body_a_index, body_a,
        body_b_index, body_b, shapes_a, count_a, shapes_b, count_b, out,
        capacity, &body_dropped);
    *dropped = body_dropped;
    return produced;
}

/* Single-manifold view of one body pair: the first retained patch, which is
 * the first shape pair in ascending slot order that overlaps. */
__device__ static inline bool pf_collision_world(
        int body_a_index, const PfBody* body_a, int body_b_index,
        const PfBody* body_b, const PfWorld* world, PfManifold* out) {
    int dropped = 0;
    return pf_collision_world_multi(body_a_index, body_a, body_b_index,
        body_b, world, out, 1, &dropped) == 1;
}




/* Manifolds produced but not stored, because the body pair ran out of
 * manifold capacity. Zero on a clean pass. `pf_detect_contacts` is the
 * caller's usual entry point and discards this; `pf_detect_contacts_reported`
 * is the same detection with the truncation count reported, so a caller can
 * tell "no contact" from "contacts that did not fit". */
__device__ static inline int pf_detect_contacts_reported(
        PfWorld* world, int* truncated) {
    if (world == NULL || world->bodies == NULL || world->body_count < 0) {
        return 0;
    }
    if (truncated != NULL) {
        *truncated = 0;
    }
    world->manifold_count = 0;
    for (int body_a = 0; body_a < world->body_count; ++body_a) {
        const PfBody* body = &world->bodies[body_a];
        bool dynamic = pf_mode_is_dynamic(body->mode);
        for (int body_b = body_a + 1; body_b < world->body_count; ++body_b) {
            const PfBody* other = &world->bodies[body_b];
            if (!dynamic && !pf_mode_is_dynamic(other->mode)) continue;
            int capacity = world->manifold_capacity - world->manifold_count;
            if (capacity <= 0) {
                if (truncated != NULL) {
                    ++*truncated;
                }
                continue;
            }
            int dropped = 0;
            int produced = pf_collision_world_multi(body_a, body, body_b, other,
                world, &world->manifolds[world->manifold_count], capacity,
                &dropped);
            world->manifold_count += produced;
            if (truncated != NULL) {
                *truncated += dropped;
            }
        }
    }
    return world->manifold_count;
}

__device__ static inline int pf_detect_contacts(PfWorld* world) {
    return pf_detect_contacts_reported(world, NULL);
}
