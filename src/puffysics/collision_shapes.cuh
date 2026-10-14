#pragma once

#include "math.cuh"
#include "collision_box.cuh"
#include "collision_sphere.cuh"
#include "box_manifold.cuh"

/* Cylinders and capsules use local +Y. Cylinders are capped (a closed solid);
 * the box test samples the curved side and both cap rims because exact
 * cylinder--box contact needs a constrained nonlinear solve.
 *
 * `velocity` is the world velocity of `center`, including omega x shape offset.
 */
typedef struct PfShapeWorld {
    PfShapeKind kind;
    PfVec3 center;
    PfQuat rotation;
    PfVec3 half_extents;
    PfVec3 velocity;
} PfShapeWorld;

__device__ static inline bool pf_shape_world(
        const PfBody* body, const PfShape* shape, PfShapeWorld* out) {
    if (out == NULL || body == NULL || shape == NULL || !pf_vec_valid(body->position)
            || !pf_quat_valid(body->rotation) || !pf_quat_valid(shape->rotation)
            || !pf_vec_valid(shape->position) || !pf_vec_valid(shape->half_extents)) {
        return false;
    }
    out->kind = shape->kind;
    out->half_extents = shape->half_extents;
    out->center = pf_add(body->position,
        pf_quat_rotate(body->rotation, shape->position));
    out->rotation = pf_quat_normalize(pf_quat_multiply(body->rotation, shape->rotation));
    out->velocity = pf_point_velocity(body, out->center);
    return true;
}

__device__ static inline bool pf_shape_valid(const PfShape* shape) {
    if (shape == NULL || !pf_vec_valid(shape->position)
            || !pf_quat_valid(shape->rotation) || !pf_vec_valid(shape->half_extents)) return false;
    if (shape->kind == PF_BOX) return shape->half_extents.x > 1.0e-6f
        && shape->half_extents.y > 1.0e-6f && shape->half_extents.z > 1.0e-6f;
    if (shape->kind == PF_SPHERE) return shape->half_extents.x > 1.0e-6f;
    if (shape->kind == PF_CYLINDER || shape->kind == PF_CAPSULE) {
        return shape->half_extents.x > 1.0e-6f && shape->half_extents.y >= 0.0f;
    }
    return false;
}

__device__ static inline float pf_shape_radius(const PfShapeWorld* shape) {
    if (shape->kind == PF_SPHERE) return shape->half_extents.x;
    // sqrt(half_height^2 + radius^2) is the bound for a capped CYLINDER, whose
    // corner is offset diagonally from its centre. A capsule's extreme point
    // is the tip of a hemisphere on the end of its axis, so its bound is the
    // sum, and using the root here understated every capsule tip by
    // half_height + radius - sqrt(half_height^2 + radius^2).
    if (shape->kind == PF_CYLINDER) {
        return sqrtf(shape->half_extents.x * shape->half_extents.x
            + shape->half_extents.y * shape->half_extents.y);
    }
    if (shape->kind == PF_CAPSULE) {
        return shape->half_extents.x + shape->half_extents.y;
    }
    return pf_length(shape->half_extents);
}

__device__ static inline PfVec3 pf_shape_axis(const PfShapeWorld* shape) {
    return pf_quat_rotate(shape->rotation, pf_v3(0.0f, 1.0f, 0.0f));
}


// The point of a shape furthest along `direction`. A cylinder's radial
// contribution is the normalised component of `direction` perpendicular to its
// axis, and a purely axial direction has no such component: every point of the
// cap disc is then a support point, so the cap centre is returned rather than
// dividing by a zero radial length.
__device__ static inline PfVec3 pf_shape_support(
        const PfShapeWorld* shape, PfVec3 direction) {
    if (shape->kind == PF_SPHERE) {
        return pf_add(shape->center, pf_scale(pf_normalize_or(direction,
            pf_v3(1.0f, 0.0f, 0.0f)), shape->half_extents.x));
    }
    if (shape->kind == PF_BOX) {
        PfVec3 axes[3];
        pf_quat_axes(shape->rotation, axes);
        PfVec3 result = shape->center;
        for (int axis = 0; axis < 3; ++axis) {
            float half = axis == 0 ? shape->half_extents.x
                : axis == 1 ? shape->half_extents.y : shape->half_extents.z;
            result = pf_add(result, pf_scale(axes[axis],
                half * (pf_dot(axes[axis], direction) >= 0.0f ? 1.0f : -1.0f)));
        }
        return result;
    }
    PfVec3 axis = pf_shape_axis(shape);
    float along = pf_dot(axis, direction);
    PfVec3 result = pf_add(shape->center, pf_scale(axis,
        along >= 0.0f ? shape->half_extents.y : -shape->half_extents.y));
    if (shape->kind == PF_CAPSULE) {
        return pf_add(result, pf_scale(pf_normalize_or(direction,
            pf_v3(1.0f, 0.0f, 0.0f)), shape->half_extents.x));
    }
    PfVec3 radial = pf_sub(direction, pf_scale(axis, along));
    float radial_squared = pf_length_squared(radial);
    if (radial_squared <= 1.0e-20f) {
        return result;
    }
    return pf_add(result, pf_scale(radial, shape->half_extents.x * rsqrtf(radial_squared)));
}

__device__ static inline void pf_shape_box_body(
        const PfShapeWorld* shape, PfBody* out) {
    PfBody body = {};
    body.shape = PF_BOX;
    body.position = shape->center;
    body.rotation = shape->rotation;
    body.half_extents = shape->half_extents;
    body.linear_velocity = shape->velocity;
    *out = body;
}

__device__ static inline void pf_shape_sphere_body(
        const PfShapeWorld* shape, PfBody* out) {
    PfBody body = {};
    body.shape = PF_SPHERE;
    body.position = shape->center;
    body.half_extents.x = shape->half_extents.x;
    body.linear_velocity = shape->velocity;
    *out = body;
}

__device__ static inline void pf_segment_endpoints(
        const PfShapeWorld* shape, PfVec3* a, PfVec3* b) {
    PfVec3 axis = pf_shape_axis(shape);
    *a = pf_add(shape->center, pf_scale(axis, -shape->half_extents.y));
    *b = pf_add(shape->center, pf_scale(axis, shape->half_extents.y));
}

__device__ static inline void pf_closest_segment_segment(
        PfVec3 p1, PfVec3 p2, PfVec3 q1, PfVec3 q2,
        float* s_out, float* t_out) {
    PfVec3 d1 = pf_sub(p2, p1);
    PfVec3 d2 = pf_sub(q2, q1);
    PfVec3 r = pf_sub(p1, q1);
    float a = pf_dot(d1, d1), e = pf_dot(d2, d2);
    float f = pf_dot(d2, r);
    float s = 0.0f, t = 0.0f;
    if (a <= 1.0e-20f && e <= 1.0e-20f) {
        *s_out = 0.0f; *t_out = 0.0f; return;
    }
    if (a <= 1.0e-20f) {
        t = fmaxf(0.0f, fminf(1.0f, f / e));
    } else {
        float c = pf_dot(d1, r);
        if (e <= 1.0e-20f) {
            s = fmaxf(0.0f, fminf(1.0f, -c / a));
        } else {
            float b = pf_dot(d1, d2);
            float denominator = a * e - b * b;
            s = denominator > 1.0e-20f
                ? fmaxf(0.0f, fminf(1.0f, (b * f - c * e) / denominator)) : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f) { t = 0.0f; s = fmaxf(0.0f, fminf(1.0f, -c / a)); }
            else if (t > 1.0f) { t = 1.0f; s = fmaxf(0.0f, fminf(1.0f, (b - c) / a)); }
        }
    }
    *s_out = s; *t_out = t;
}

__device__ static inline void pf_closest_segment_point(
        PfVec3 a, PfVec3 b, PfVec3 point, PfVec3* closest) {
    PfVec3 delta = pf_sub(b, a);
    float length_squared = pf_length_squared(delta);
    float t = length_squared > 1.0e-20f
        ? fmaxf(0.0f, fminf(1.0f, pf_dot(pf_sub(point, a), delta) / length_squared)) : 0.0f;
    *closest = pf_add(a, pf_scale(delta, t));
}

__device__ static inline bool pf_capsule_capsule_contact(
        const PfShapeWorld* capsule_a, const PfShapeWorld* capsule_b,
        PfManifold* out) {
    PfVec3 a0, a1, b0, b1;
    pf_segment_endpoints(capsule_a, &a0, &a1);
    pf_segment_endpoints(capsule_b, &b0, &b1);
    float s, t;
    pf_closest_segment_segment(a0, a1, b0, b1, &s, &t);
    PfVec3 pa = pf_add(a0, pf_scale(pf_sub(a1, a0), s));
    PfVec3 pb = pf_add(b0, pf_scale(pf_sub(b1, b0), t));
    PfVec3 delta = pf_sub(pa, pb);
    float distance_squared = pf_length_squared(delta);
    float radius = capsule_a->half_extents.x + capsule_b->half_extents.x;
    float inverse_distance = distance_squared > 1.0e-20f ? rsqrtf(distance_squared) : 0.0f;
    PfVec3 normal = inverse_distance > 0.0f ? pf_scale(delta, inverse_distance)
        : pf_v3(1.0f, 0.0f, 0.0f);
    float separation = distance_squared * inverse_distance - radius;
    if (separation > 0.0f) return false;
    out->normal = normal;
    out->point_count = 1;
    out->points[0].point_a = pf_sub(pa, pf_scale(normal, capsule_a->half_extents.x));
    out->points[0].point_b = pf_add(pb, pf_scale(normal, capsule_b->half_extents.x));
    out->points[0].separation = separation;
    return true;
}

__device__ static inline bool pf_capsule_sphere_contact(
        const PfShapeWorld* capsule, const PfShapeWorld* sphere,
        PfManifold* out) {
    PfVec3 a, b, closest;
    pf_segment_endpoints(capsule, &a, &b);
    pf_closest_segment_point(a, b, sphere->center, &closest);
    PfVec3 delta = pf_sub(sphere->center, closest);
    float distance_squared = pf_length_squared(delta);
    float radius = capsule->half_extents.x + sphere->half_extents.x;
    float inverse_distance = distance_squared > 1.0e-20f ? rsqrtf(distance_squared) : 0.0f;
    PfVec3 normal = inverse_distance > 0.0f ? pf_scale(delta, inverse_distance)
        : pf_v3(1.0f, 0.0f, 0.0f);
    float separation = distance_squared * inverse_distance - radius;
    if (separation > 0.0f) return false;
    out->normal = pf_scale(normal, -1.0f); /* capsule A, sphere B: B-to-A normal */
    out->point_count = 1;
    out->points[0].point_a = pf_add(closest,
        pf_scale(normal, capsule->half_extents.x));
    out->points[0].point_b = pf_sub(sphere->center,
        pf_scale(normal, sphere->half_extents.x));
    out->points[0].separation = separation;
    return true;
}

// Closest point on a capped cylinder, by the two decompositions the solid is
// defined in. `direction` is the unit vector from the cylinder to the query
// point and `distance` the signed gap to the surface, negative inside, so
// `point - direction * distance` is the surface point in both branches.
//
// Outside, the axial coordinate and the radial length clamp INDEPENDENTLY,
// which is the closest point of the product of a segment and a disc. Inside,
// the nearest surface is the nearer of the side and the two caps, and the
// surface point keeps the coordinate that branch does not move.
__device__ static inline void pf_cylinder_closest(
        const PfShapeWorld* cylinder, PfVec3 point,
        PfVec3* surface, PfVec3* direction, float* distance) {
    PfVec3 axis = pf_shape_axis(cylinder);
    float radius = cylinder->half_extents.x;
    float height = cylinder->half_extents.y;
    PfVec3 delta = pf_sub(point, cylinder->center);
    float axial = pf_dot(delta, axis);
    PfVec3 radial_delta = pf_sub(delta, pf_scale(axis, axial));
    float radial = pf_length(radial_delta);
    // A point on the axis has no radial direction, and every radial direction
    // names the same distance to the side, so any perpendicular to the axis
    // is an equally correct answer and one is chosen deterministically.
    PfVec3 radial_direction = pf_normalize_or(radial_delta,
        pf_normalize_or(pf_cross(axis, pf_v3(1.0f, 0.0f, 0.0f)),
            pf_v3(0.0f, 0.0f, 1.0f)));
    if (axial > -height && axial < height && radial < radius) {
        float side = radius - radial;
        float cap = height - fabsf(axial);
        /* An interior point's nearest surface is OUTWARD, so `direction`
         * must point outward for `surface = point - direction*distance` to
         * land on it with distance negative. Negating the radial direction
         * resolved the contact toward the axis instead: on a cylinder of
         * radius 1 the query (0.9, 0, 0) returned (0.8, 0, 0) rather than
         * (1, 0, 0), and a body partly inside a shape was pushed sideways
         * rather than out. The cap branch already used the outward axial
         * direction and is unchanged; a tie takes it, because an exact
         * cap/side tie on the axis would otherwise fall back to a
         * fabricated radial direction. */
        if (cap <= side) {
            *direction = pf_scale(axis, axial < 0.0f ? -1.0f : 1.0f);
            *distance = -cap;
        } else {
            *direction = radial_direction;
            *distance = -side;
        }
    } else {
        float clamped_axial = axial < -height ? -height
            : axial > height ? height : axial;
        float clamped_radial = radial < radius ? radial : radius;
        PfVec3 closest = pf_add(cylinder->center,
            pf_add(pf_scale(axis, clamped_axial),
                pf_scale(radial_direction, clamped_radial)));
        PfVec3 to_point = pf_sub(point, closest);
        float distance_squared = pf_length_squared(to_point);
        *distance = distance_squared > 1.0e-20f
            ? distance_squared * rsqrtf(distance_squared) : 0.0f;
        *direction = *distance > 0.0f ? pf_scale(to_point, 1.0f / *distance)
            : radial_direction;
    }
    *surface = pf_sub(point, pf_scale(*direction, *distance));
}

__device__ static inline bool pf_cylinder_sphere_contact(
        const PfShapeWorld* cylinder, const PfShapeWorld* sphere,
        PfManifold* out) {
    PfVec3 surface, normal;
    float distance;
    pf_cylinder_closest(cylinder, sphere->center, &surface, &normal, &distance);
    float separation = distance - sphere->half_extents.x;
    if (separation > 0.0f) return false;
    out->normal = pf_scale(normal, -1.0f);
    out->point_count = 1;
    out->points[0].point_a = surface;
    out->points[0].point_b = pf_sub(sphere->center,
        pf_scale(normal, sphere->half_extents.x));
    out->points[0].separation = separation;
    return true;
}

__device__ static inline void pf_segment_box_candidate(
        PfVec3 direction, PfVec3 axis, PfVec3 center, float height,
        float radius, bool cylinder, const PfBody* proxy, const PfVec3 axes[3],
        float* best, PfVec3* best_normal, PfVec3* best_point_a,
        PfVec3* best_point_b) {
    direction = pf_normalize_or(direction, pf_v3(1.0f, 0.0f, 0.0f));
    /* Global supports, so any direction is a valid separating axis: the
     * shape reaches dot(center,d) - height|dot(axis,d)| - cap and the box
     * reaches dot(box,d) + sum half_extent |dot(axes,d)|. A cylinder's flat
     * cap contributes no radius along the axis, a capsule's does. */
    float along = pf_dot(axis, direction);
    float cap = cylinder
        ? radius * sqrtf(fmaxf(0.0f, 1.0f - along * along)) : radius;
    float low = pf_dot(pf_sub(center, proxy->position), direction)
        - height * fabsf(along) - cap;
    float high = 0.0f;
    float offset[3];
    for (int k = 0; k < 3; ++k) {
        float half = k == 0 ? proxy->half_extents.x
            : k == 1 ? proxy->half_extents.y : proxy->half_extents.z;
        float component = pf_dot(axes[k], direction);
        offset[k] = component >= 0.0f ? half : -half;
        high += half * fabsf(component);
    }
    float separation = low - high;
    if (separation <= *best) return;
    *best = separation;
    *best_normal = direction;
    /* The manifold convention is point_b on B and point_a = point_b + normal
     * separation, so point_b is the box's support along the normal and
     * point_a is the segment shape's support along the REVERSE normal: the
     * gap between them along the normal is exactly `separation`. Placing the
     * segment point with `- direction * cap` gave the right projection onto
     * the normal but a point that is not on the surface at all, carrying an
     * axial offset of -dot(axis,d)*cap and a radial length of cap instead of
     * the radius. */
    *best_point_a = pf_add(center, pf_scale(axis,
        along >= 0.0f ? -height : height));
    if (cylinder) {
        PfVec3 radial = pf_sub(pf_scale(direction, -1.0f),
            pf_scale(axis, -along));
        float radial_squared = pf_length_squared(radial);
        if (radial_squared > 1.0e-20f) {
            *best_point_a = pf_add(*best_point_a,
                pf_scale(radial, radius * rsqrtf(radial_squared)));
        }
    } else {
        *best_point_a = pf_sub(*best_point_a, pf_scale(direction, radius));
    }
    *best_point_b = proxy->position;
    for (int k = 0; k < 3; ++k)
        *best_point_b = pf_add(*best_point_b, pf_scale(axes[k], offset[k]));
}

__device__ static inline bool pf_segment_box_contact(
        const PfShapeWorld* segment_shape, const PfShapeWorld* box,
        bool cylinder, PfManifold* out) {
    PfVec3 axis = pf_shape_axis(segment_shape);
    PfVec3 center = segment_shape->center;
    float height = segment_shape->half_extents.y;
    float radius = segment_shape->half_extents.x;
    PfBody proxy;
    pf_shape_box_body(box, &proxy);
    PfVec3 box_axes[3];
    pf_quat_axes(proxy.rotation, box_axes);

    /* Separating axis. Every engine manifold satisfies
     * separation = dot(point_a - point_b, normal) with the normal pointing
     * from B to A, and pf_solve_positions consumes exactly that number, so
     * the winning candidate is emitted with its own support points and the
     * manifold is self-consistent by construction.
     *
     * The direction set is the box's six face axes, the three
     * segment-to-box-edge cross axes, a 33-point sweep along the segment and,
     * for a cylinder, eight samples on each cap rim. It is a SUBSET of the
     * separating axes, and a subset bounds the reported gap from below
     * because every candidate is scored with the two global support
     * functions: adding one can only raise it. The converse does not hold and
     * is the documented residual. A sampled search that fails to find a
     * separating axis has proved nothing - two shapes can be disjoint with a
     * large gap and the query still report a contact, which is a FALSE
     * POSITIVE, and sampling cannot prove intersection either way. The nine
     * exact axes come first, and an overlapping pose whose minimum-translation
     * axis falls between two samples is reported as slightly shallower than
     * the truth. Exact cylinder--box contact would need a continuous search
     * over the cap rims and is deliberately replaced by that fixed sweep. */
    float best = -1.0e30f;
    PfVec3 best_normal = pf_v3(1.0f, 0.0f, 0.0f);
    PfVec3 best_point_a = center, best_point_b = center;
    for (int k = 0; k < 3; ++k) {
        for (int sign = -1; sign <= 1; sign += 2)
            pf_segment_box_candidate(pf_scale(box_axes[k], (float)sign), axis,
                center, height, radius, cylinder, &proxy, box_axes, &best,
                &best_normal, &best_point_a, &best_point_b);
        pf_segment_box_candidate(pf_cross(axis, box_axes[k]), axis, center,
            height, radius, cylinder, &proxy, box_axes, &best, &best_normal,
            &best_point_a, &best_point_b);
    }
    for (int i = 0; i <= 32; ++i) {
        float t = height <= 0.0f ? 0.0f : -height + 2.0f * height * (float)i / 32.0f;
        PfVec3 candidate = pf_add(center, pf_scale(axis, t));
        PfPointQuery query;
        pf_point_box_query(candidate, &proxy, &query);
        pf_segment_box_candidate(pf_sub(candidate, query.closest_point), axis,
            center, height, radius, cylinder, &proxy, box_axes, &best,
            &best_normal, &best_point_a, &best_point_b);
    }
    if (cylinder) {
        PfVec3 radial = pf_normalize_or(pf_cross(axis, pf_v3(1.0f, 0.0f, 0.0f)),
            pf_v3(1.0f, 0.0f, 0.0f));
        PfVec3 tangent = pf_normalize_or(pf_cross(axis, radial), pf_v3(0.0f, 0.0f, 1.0f));
        for (int end = -1; end <= 1; end += 2) for (int i = 0; i < 8; ++i) {
            float angle = 6.28318530718f * (float)i / 8.0f;
            PfVec3 candidate = pf_add(pf_add(center, pf_scale(axis, height * (float)end)),
                pf_scale(pf_add(pf_scale(radial, cosf(angle)),
                    pf_scale(tangent, sinf(angle))), radius));
            PfPointQuery query;
            pf_point_box_query(candidate, &proxy, &query);
            pf_segment_box_candidate(pf_sub(candidate, query.closest_point), axis,
                center, height, radius, cylinder, &proxy, box_axes, &best,
                &best_normal, &best_point_a, &best_point_b);
        }
    }
    if (best > 0.0f) return false;
    out->normal = best_normal;
    out->point_count = 1;
    out->points[0].point_a = best_point_a;
    out->points[0].point_b = best_point_b;
    out->points[0].separation = best;
    return true;
}

// A separating-axis candidate scored with the two shapes' support functions.
// The normal points from B to A, so the reported gap is A's minimum along it
// minus B's maximum along it, and the two emitted points are exactly those
// support points, which makes separation = dot(point_a - point_b, normal) hold
// by construction rather than by agreement.
__device__ static inline void pf_support_candidate(PfVec3 direction,
        const PfShapeWorld* a, const PfShapeWorld* b, float* best,
        PfVec3* best_normal, PfVec3* best_point_a, PfVec3* best_point_b) {
    direction = pf_normalize_or(direction, pf_v3(1.0f, 0.0f, 0.0f));
    PfVec3 point_a = pf_shape_support(a, pf_scale(direction, -1.0f));
    PfVec3 point_b = pf_shape_support(b, direction);
    float separation = pf_dot(pf_sub(point_a, point_b), direction);
    if (separation <= *best) return;
    *best = separation;
    *best_normal = direction;
    *best_point_a = point_a;
    *best_point_b = point_b;
}

__device__ static inline void pf_perp_frame(PfVec3 axis, PfVec3* u, PfVec3* v) {
    *u = pf_normalize_or(pf_cross(axis, pf_v3(1.0f, 0.0f, 0.0f)),
        pf_v3(0.0f, 0.0f, 1.0f));
    *v = pf_normalize_or(pf_cross(axis, *u), pf_v3(0.0f, 0.0f, 1.0f));
}

// Cylinder and capsule against cylinder and capsule. Two capsules have a
// closed form - closest points of the two segments - and keep it. A cylinder
// is not a capsule: the capsule's hemisphere bulges past the cylinder's flat
// cap, so substituting one for the other reports a contact between two shapes
// whose caps are apart, by a margin of up to 2 * radius along the axis. This
// scores real support functions over a direction set built from the two axes
// and eight radial samples about each, so the flat caps are respected.
//
// As in the box case the set is a SUBSET of the separating axes and bounds
// the reported gap from below: a pose whose true minimum-translation axis
// falls between two samples is reported as shallower than the truth, which
// is a false positive, and the samples do not prove intersection.
__device__ static inline bool pf_segment_segment_contact(
        const PfShapeWorld* a, const PfShapeWorld* b, PfManifold* out) {
    if (a->kind == PF_CAPSULE && b->kind == PF_CAPSULE) {
        return pf_capsule_capsule_contact(a, b, out);
    }
    PfVec3 axis_a = pf_shape_axis(a);
    PfVec3 axis_b = pf_shape_axis(b);
    PfVec3 radial_a, tangent_a, radial_b, tangent_b;
    pf_perp_frame(axis_a, &radial_a, &tangent_a);
    pf_perp_frame(axis_b, &radial_b, &tangent_b);
    float best = -1.0e30f;
    PfVec3 best_normal = pf_v3(1.0f, 0.0f, 0.0f);
    PfVec3 best_point_a = a->center, best_point_b = b->center;
    // The set is closed under negation - the rings are symmetric and every
    // combination of an axis and a ring element is taken with both signs -
    // so the argmax compares both escapes on every direction and the two
    // argument orders of the same pair agree.
    for (int i = 0; i < 2; ++i) {
        PfVec3 axis = i ? axis_b : axis_a;
        pf_support_candidate(axis, a, b, &best, &best_normal, &best_point_a,
            &best_point_b);
        pf_support_candidate(pf_scale(axis, -1.0f), a, b, &best, &best_normal,
            &best_point_a, &best_point_b);
    }
    pf_support_candidate(pf_cross(axis_a, axis_b), a, b, &best, &best_normal,
        &best_point_a, &best_point_b);
    float alignment = pf_dot(axis_a, axis_b);
    for (int i = 0; i < 2; ++i) {
        PfVec3 axis = i ? axis_b : axis_a;
        PfVec3 other = i ? axis_a : axis_b;
        // The projection of one axis off the other is the separating
        // direction when the axes are near-parallel and the shapes lie side
        // by side, and no ring sample lands on it in general.
        pf_support_candidate(pf_sub(axis, pf_scale(other, alignment)), a, b,
            &best, &best_normal, &best_point_a, &best_point_b);
        pf_support_candidate(pf_sub(axis, pf_scale(other, -alignment)), a, b,
            &best, &best_normal, &best_point_a, &best_point_b);
    }
    for (int i = 0; i < 2; ++i) {
        PfVec3 radial = i ? radial_b : radial_a;
        PfVec3 tangent = i ? tangent_b : tangent_a;
        PfVec3 other_axis = i ? axis_a : axis_b;
        for (int k = 0; k < 8; ++k) {
            float angle = 6.28318530718f * (float)k / 8.0f;
            PfVec3 ring = pf_add(pf_scale(radial, cosf(angle)),
                pf_scale(tangent, sinf(angle)));
            pf_support_candidate(ring, a, b, &best, &best_normal,
                &best_point_a, &best_point_b);
            pf_support_candidate(pf_add(ring, other_axis), a, b, &best,
                &best_normal, &best_point_a, &best_point_b);
            pf_support_candidate(pf_sub(ring, other_axis), a, b, &best,
                &best_normal, &best_point_a, &best_point_b);
        }
    }
    if (best > 0.0f) return false;
    out->normal = best_normal;
    out->point_count = 1;
    out->points[0].point_a = best_point_a;
    out->points[0].point_b = best_point_b;
    out->points[0].separation = best;
    return true;
}

__device__ static inline bool pf_shape_contact(
        const PfShapeWorld* a, const PfShapeWorld* b, PfManifold* out) {
    if (a == NULL || b == NULL || out == NULL) return false;
    if (a->kind == PF_SPHERE && b->kind == PF_SPHERE) {
        PfBody ba, bb; pf_shape_sphere_body(a, &ba); pf_shape_sphere_body(b, &bb);
        return pf_sphere_sphere_contact(&ba, &bb, out);
    }
    if (a->kind == PF_SPHERE && b->kind == PF_BOX) {
        PfBody sa, ba; pf_shape_sphere_body(a, &sa); pf_shape_box_body(b, &ba);
        return pf_sphere_box_contact(&sa, &ba, out);
    }
    if (a->kind == PF_BOX && b->kind == PF_SPHERE) {
        PfBody sa, ba; pf_shape_sphere_body(b, &sa); pf_shape_box_body(a, &ba);
        if (!pf_sphere_box_contact(&sa, &ba, out)) return false;
        PfVec3 point = out->points[0].point_a; out->points[0].point_a = out->points[0].point_b;
        out->points[0].point_b = point; out->normal = pf_scale(out->normal, -1.0f); return true;
    }
    if (a->kind == PF_BOX && b->kind == PF_BOX) {
        PfBody ba, bb; pf_shape_box_body(a, &ba); pf_shape_box_body(b, &bb);
        PfVec3 aa[3], ab[3]; pf_quat_axes(ba.rotation, aa); pf_quat_axes(bb.rotation, ab);
        PfBoxQuery query; return pf_box_sat(&ba, &bb, aa, ab, &query) && pf_box_manifold(&ba, &bb, aa, ab, &query, out);
    }
    if (a->kind == PF_CAPSULE && b->kind == PF_SPHERE) return pf_capsule_sphere_contact(a, b, out);
    if (a->kind == PF_SPHERE && b->kind == PF_CAPSULE) {
        if (!pf_capsule_sphere_contact(b, a, out)) return false;
        PfVec3 point = out->points[0].point_a; out->points[0].point_a = out->points[0].point_b;
        out->points[0].point_b = point; out->normal = pf_scale(out->normal, -1.0f); return true;
    }
    if (a->kind == PF_CYLINDER && b->kind == PF_SPHERE) return pf_cylinder_sphere_contact(a, b, out);
    if (a->kind == PF_SPHERE && b->kind == PF_CYLINDER) {
        if (!pf_cylinder_sphere_contact(b, a, out)) return false;
        PfVec3 point = out->points[0].point_a; out->points[0].point_a = out->points[0].point_b;
        out->points[0].point_b = point; out->normal = pf_scale(out->normal, -1.0f); return true;
    }
    if (a->kind == PF_CYLINDER && b->kind == PF_BOX) return pf_segment_box_contact(a, b, true, out);
    if (a->kind == PF_BOX && b->kind == PF_CYLINDER) {
        if (!pf_segment_box_contact(b, a, true, out)) return false;
        PfVec3 point = out->points[0].point_a; out->points[0].point_a = out->points[0].point_b;
        out->points[0].point_b = point; out->normal = pf_scale(out->normal, -1.0f); return true;
    }
    if (a->kind == PF_CAPSULE && b->kind == PF_BOX) return pf_segment_box_contact(a, b, false, out);
    if (a->kind == PF_BOX && b->kind == PF_CAPSULE) {
        if (!pf_segment_box_contact(b, a, false, out)) return false;
        PfVec3 point = out->points[0].point_a; out->points[0].point_a = out->points[0].point_b;
        out->points[0].point_b = point; out->normal = pf_scale(out->normal, -1.0f); return true;
    }
    if ((a->kind == PF_CYLINDER || a->kind == PF_CAPSULE)
            && (b->kind == PF_CYLINDER || b->kind == PF_CAPSULE)) {
        return pf_segment_segment_contact(a, b, out);
    }
    return false;
}
