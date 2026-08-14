#pragma once

#include <assert.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "robot_arm.h"

#define PL_IMPULSE_MAX_MANIFOLDS 48
#define PL_IMPULSE_MAX_CANDIDATES 20
#define PL_SAT_PARALLEL_EPSILON 1.0e-8f
#define PL_SAT_MANIFOLD_DUPLICATE_EPSILON 1.0e-8f
#define PL_SAT_MAX_MANIFOLD_POINTS 4
#define PL_SAT_MAX_CLIP_VERTICES 8

typedef enum PlSatFeature {
    PL_SAT_FACE_A_X = 0,
    PL_SAT_FACE_A_Y = 1,
    PL_SAT_FACE_A_Z = 2,
    PL_SAT_FACE_B_X = 3,
    PL_SAT_FACE_B_Y = 4,
    PL_SAT_FACE_B_Z = 5,
    PL_SAT_EDGE_A0_B0 = 6,
} PlSatFeature;

typedef struct PlSatObb {
    RaVec3 center;
    RaVec3 axis[3];
    RaVec3 half_extents;
} PlSatObb;

typedef struct PlSatQuery {
    RaConvexContact contact;
    int feature;
} PlSatQuery;

typedef struct PlSatManifold {
    int count;
    int feature;
    RaConvexContact point[PL_SAT_MAX_MANIFOLD_POINTS];
    uint32_t point_feature[PL_SAT_MAX_MANIFOLD_POINTS];  // warm-start key
} PlSatManifold;

RA_D static RA_INLINE RaVec3 pl_sat_neg(RaVec3 value) {
    return ra_scale(value, -1.0f);
}

RA_D static RA_INLINE float pl_shalf(
        RaVec3 half_extents, int index) {
    return index == 0 ? half_extents.x
        : index == 1 ? half_extents.y : half_extents.z;
}

RA_D static RA_INLINE PlSatObb pl_sobb(
        const RaConvexShape* shape) {
    PlSatObb box;
    box.center = shape->pose.position;
    ra_caxes(shape->pose.rotation, box.axis);
    box.half_extents = shape->half_extents;
    return box;
}

RA_D static RA_INLINE RaVec3 pl_ssup(
        const PlSatObb* box, RaVec3 direction) {
    float sx = ra_dot(box->axis[0], direction) < 0.0f
        ? -box->half_extents.x : box->half_extents.x;
    float sy = ra_dot(box->axis[1], direction) < 0.0f
        ? -box->half_extents.y : box->half_extents.y;
    float sz = ra_dot(box->axis[2], direction) < 0.0f
        ? -box->half_extents.z : box->half_extents.z;
    return ra_add(box->center,
        ra_add(ra_scale(box->axis[0], sx),
        ra_add(ra_scale(box->axis[1], sy),
               ra_scale(box->axis[2], sz))));
}

RA_D static RA_INLINE RaVec3 pl_sorb(
        RaVec3 axis, float projection_a_minus_b) {
    return projection_a_minus_b < 0.0f ? pl_sat_neg(axis) : axis;  // zero -> +axis
}

RA_D static RA_INLINE void pl_scon(
        PlSatQuery* query, float separation, RaVec3 normal, int feature) {
    if (separation > query->contact.separation) {  // exact-float tie keeps first axis
        query->contact.separation = separation;
        query->contact.normal = normal;
        query->feature = feature;
    }
}

RA_D static RA_INLINE void pl_sxcon(
        PlSatQuery* query, float unnormalised_separation,
        float projection_b_minus_a, RaVec3 axis_a, RaVec3 axis_b,
        int feature) {
    RaVec3 cross_axis = ra_cross(axis_a, axis_b);
    float length_squared = ra_dot(cross_axis, cross_axis);
    float parallel_squared = PL_SAT_PARALLEL_EPSILON
        * PL_SAT_PARALLEL_EPSILON;
    if (!(length_squared > parallel_squared)) {
        return;
    }
    float inverse_length = 1.0f / sqrtf(length_squared);
    float separation = unnormalised_separation * inverse_length;
    if (separation <= query->contact.separation) {
        return;
    }
    RaVec3 normal = ra_scale(cross_axis, inverse_length);
    if (projection_b_minus_a > 0.0f) {
        normal = pl_sat_neg(normal);  // normal always B->A
    }
    query->contact.separation = separation;
    query->contact.normal = normal;
    query->feature = feature;
}

RA_D static RA_INLINE PlSatQuery pl_sqobb(
        const PlSatObb* a, const PlSatObb* b, float margin) {
    const RaVec3 a0 = a->axis[0];
    const RaVec3 a1 = a->axis[1];
    const RaVec3 a2 = a->axis[2];
    const RaVec3 b0 = b->axis[0];
    const RaVec3 b1 = b->axis[1];
    const RaVec3 b2 = b->axis[2];
    const float ax = a->half_extents.x;
    const float ay = a->half_extents.y;
    const float az = a->half_extents.z;
    const float bx = b->half_extents.x;
    const float by = b->half_extents.y;
    const float bz = b->half_extents.z;

    const RaVec3 b_minus_a = ra_sub(b->center, a->center);
    const float t0 = ra_dot(b_minus_a, a0);
    const float t1 = ra_dot(b_minus_a, a1);
    const float t2 = ra_dot(b_minus_a, a2);

    const float r00 = ra_dot(a0, b0);
    const float r01 = ra_dot(a0, b1);
    const float r02 = ra_dot(a0, b2);
    const float r10 = ra_dot(a1, b0);
    const float r11 = ra_dot(a1, b1);
    const float r12 = ra_dot(a1, b2);
    const float r20 = ra_dot(a2, b0);
    const float r21 = ra_dot(a2, b1);
    const float r22 = ra_dot(a2, b2);
    const float ar00 = fabsf(r00);
    const float ar01 = fabsf(r01);
    const float ar02 = fabsf(r02);
    const float ar10 = fabsf(r10);
    const float ar11 = fabsf(r11);
    const float ar12 = fabsf(r12);
    const float ar20 = fabsf(r20);
    const float ar21 = fabsf(r21);
    const float ar22 = fabsf(r22);

    PlSatQuery query;
    query.contact.hit = 0;
    query.contact.iterations = 15;
    query.contact.separation = -3.402823466e+38f;
    query.contact.normal = a0;
    query.contact.point_a = a->center;
    query.contact.point_b = b->center;
    query.feature = PL_SAT_FACE_A_X;

    pl_scon(&query,
        fabsf(t0) - ax - (bx*ar00 + by*ar01 + bz*ar02),
        pl_sorb(a0, -t0), PL_SAT_FACE_A_X);
    pl_scon(&query,
        fabsf(t1) - ay - (bx*ar10 + by*ar11 + bz*ar12),
        pl_sorb(a1, -t1), PL_SAT_FACE_A_Y);
    pl_scon(&query,
        fabsf(t2) - az - (bx*ar20 + by*ar21 + bz*ar22),
        pl_sorb(a2, -t2), PL_SAT_FACE_A_Z);

    const float u0 = t0*r00 + t1*r10 + t2*r20;
    const float u1 = t0*r01 + t1*r11 + t2*r21;
    const float u2 = t0*r02 + t1*r12 + t2*r22;
    pl_scon(&query,
        fabsf(u0) - bx - (ax*ar00 + ay*ar10 + az*ar20),
        pl_sorb(b0, -u0), PL_SAT_FACE_B_X);
    pl_scon(&query,
        fabsf(u1) - by - (ax*ar01 + ay*ar11 + az*ar21),
        pl_sorb(b1, -u1), PL_SAT_FACE_B_Y);
    pl_scon(&query,
        fabsf(u2) - bz - (ax*ar02 + ay*ar12 + az*ar22),
        pl_sorb(b2, -u2), PL_SAT_FACE_B_Z);

    pl_sxcon(&query,
        fabsf(t2*r10 - t1*r20) - (ay*ar20 + az*ar10)
            - (by*ar02 + bz*ar01),
        t2*r10 - t1*r20, a0, b0, PL_SAT_EDGE_A0_B0 + 0);
    pl_sxcon(&query,
        fabsf(t2*r11 - t1*r21) - (ay*ar21 + az*ar11)
            - (bz*ar00 + bx*ar02),
        t2*r11 - t1*r21, a0, b1, PL_SAT_EDGE_A0_B0 + 1);
    pl_sxcon(&query,
        fabsf(t2*r12 - t1*r22) - (ay*ar22 + az*ar12)
            - (bx*ar01 + by*ar00),
        t2*r12 - t1*r22, a0, b2, PL_SAT_EDGE_A0_B0 + 2);

    pl_sxcon(&query,
        fabsf(t0*r20 - t2*r00) - (az*ar00 + ax*ar20)
            - (by*ar12 + bz*ar11),
        t0*r20 - t2*r00, a1, b0, PL_SAT_EDGE_A0_B0 + 3);
    pl_sxcon(&query,
        fabsf(t0*r21 - t2*r01) - (az*ar01 + ax*ar21)
            - (bz*ar10 + bx*ar12),
        t0*r21 - t2*r01, a1, b1, PL_SAT_EDGE_A0_B0 + 4);
    pl_sxcon(&query,
        fabsf(t0*r22 - t2*r02) - (az*ar02 + ax*ar22)
            - (bx*ar11 + by*ar10),
        t0*r22 - t2*r02, a1, b2, PL_SAT_EDGE_A0_B0 + 5);

    pl_sxcon(&query,
        fabsf(t1*r00 - t0*r10) - (ax*ar10 + ay*ar00)
            - (by*ar22 + bz*ar21),
        t1*r00 - t0*r10, a2, b0, PL_SAT_EDGE_A0_B0 + 6);
    pl_sxcon(&query,
        fabsf(t1*r01 - t0*r11) - (ax*ar11 + ay*ar01)
            - (bz*ar20 + bx*ar22),
        t1*r01 - t0*r11, a2, b1, PL_SAT_EDGE_A0_B0 + 7);
    pl_sxcon(&query,
        fabsf(t1*r02 - t0*r12) - (ax*ar12 + ay*ar02)
            - (bx*ar21 + by*ar20),
        t1*r02 - t0*r12, a2, b2, PL_SAT_EDGE_A0_B0 + 8);

    query.contact.hit = query.contact.separation <= margin;
    RaVec3 normal = query.contact.normal;
    query.contact.point_a = pl_ssup(a, pl_sat_neg(normal));
    query.contact.point_b = pl_ssup(b, normal);
    return query;
}

RA_D static RA_INLINE int pl_sclip(
        const RaVec3* input, int count, RaVec3* output,
        RaVec3 normal, float offset) {
    if (count <= 0) {
        return 0;
    }
    int output_count = 0;
    RaVec3 previous = input[count - 1];
    float previous_distance = ra_dot(previous, normal) - offset;
    int previous_inside = previous_distance <= 0.0f;
    for (int index = 0; index < count; ++index) {
        RaVec3 current = input[index];
        float current_distance = ra_dot(current, normal) - offset;
        int current_inside = current_distance <= 0.0f;
        if (current_inside != previous_inside
                && output_count < PL_SAT_MAX_CLIP_VERTICES) {
            float fraction = previous_distance
                / (previous_distance - current_distance);
            output[output_count++] = ra_lerp(previous, current, fraction);
        }
        if (current_inside
                && output_count < PL_SAT_MAX_CLIP_VERTICES) {
            output[output_count++] = current;
        }
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }
    return output_count;
}

RA_D static RA_INLINE void pl_sface(
        const PlSatObb* box, int normal_axis, float sign,
        RaVec3 output[4]) {
    const int tangent_a = (normal_axis + 1) % 3;
    const int tangent_b = (normal_axis + 2) % 3;
    const float face_half = pl_shalf(
        box->half_extents, normal_axis);
    const float half_a = pl_shalf(
        box->half_extents, tangent_a);
    const float half_b = pl_shalf(
        box->half_extents, tangent_b);
    const RaVec3 center = ra_add(box->center,
        ra_scale(box->axis[normal_axis], sign*face_half));
    const RaVec3 along_a = ra_scale(box->axis[tangent_a], half_a);
    const RaVec3 along_b = ra_scale(box->axis[tangent_b], half_b);
    output[0] = ra_sub(ra_sub(center, along_a), along_b);
    output[1] = ra_add(ra_sub(center, along_a), along_b);
    output[2] = ra_add(ra_add(center, along_a), along_b);
    output[3] = ra_sub(ra_add(center, along_a), along_b);
}

RA_D static RA_INLINE int pl_saxis(
        const PlSatObb* box, RaVec3 target_normal) {
    float best = fabsf(ra_dot(box->axis[0], target_normal));
    int best_axis = 0;
    float candidate = fabsf(ra_dot(box->axis[1], target_normal));
    if (candidate > best) {
        best = candidate;
        best_axis = 1;
    }
    candidate = fabsf(ra_dot(box->axis[2], target_normal));
    if (candidate > best) {
        best_axis = 2;
    }
    return best_axis;
}

RA_D static RA_INLINE int pl_sused(
        const int* selected, int count, int candidate) {
    for (int index = 0; index < count; ++index) {
        if (selected[index] == candidate) {
            return 1;
        }
    }
    return 0;
}

RA_D static RA_INLINE int pl_sdup(
        const PlSatManifold* manifold, RaVec3 point_a, RaVec3 point_b) {
    const float epsilon = PL_SAT_MANIFOLD_DUPLICATE_EPSILON;
    const float epsilon_squared = epsilon * epsilon;
    for (int index = 0; index < manifold->count; ++index) {
        RaVec3 delta_a = ra_sub(point_a, manifold->point[index].point_a);
        RaVec3 delta_b = ra_sub(point_b, manifold->point[index].point_b);
        if (ra_dot(delta_a, delta_a) <= epsilon_squared
                && ra_dot(delta_b, delta_b) <= epsilon_squared) return 1;
    }
    return 0;
}

RA_D static RA_INLINE uint32_t pl_sfid(
        int manifold_feature, int point_index) {
    return ((uint32_t)(manifold_feature + 1) << 8)  // 0 is invalid
        | (uint32_t)(point_index + 1);
}

RA_D static RA_INLINE void pl_sclr(
        PlSatManifold* manifold) {
    manifold->count = 0;
    manifold->feature = PL_SAT_FACE_A_X;
    for (int index = 0; index < PL_SAT_MAX_MANIFOLD_POINTS; ++index) {
        RaConvexContact empty = {};
        manifold->point[index] = empty;
        manifold->point_feature[index] = 0;
    }
}

RA_D static RA_INLINE int pl_sman(
        const PlSatObb* a, const PlSatObb* b, float margin,
        const PlSatQuery* query, PlSatManifold* manifold) {
    pl_sclr(manifold);
    manifold->feature = query->feature;
    if (!query->contact.hit) {
        return 0;
    }

    if (query->feature >= PL_SAT_EDGE_A0_B0) {
        manifold->point[0] = query->contact;
        manifold->point_feature[0] = pl_sfid(
            query->feature, 0);
        manifold->count = 1;
        return 1;
    }

    const int reference_is_a = query->feature < PL_SAT_FACE_B_X;
    const int reference_axis = query->feature % 3;
    const PlSatObb* reference = reference_is_a ? a : b;
    const PlSatObb* incident = reference_is_a ? b : a;
    const RaVec3 contact_normal = query->contact.normal;

    RaVec3 reference_normal = reference_is_a
        ? pl_sat_neg(contact_normal) : contact_normal;
    const float reference_sign = ra_dot(
        reference->axis[reference_axis], reference_normal) < 0.0f
        ? -1.0f : 1.0f;
    reference_normal = ra_scale(
        reference->axis[reference_axis], reference_sign);
    const RaVec3 reference_center = ra_add(reference->center,
        ra_scale(reference_normal, pl_shalf(
            reference->half_extents, reference_axis)));

    const RaVec3 incident_target = pl_sat_neg(reference_normal);
    const int incident_axis = pl_saxis(
        incident, incident_target);
    const float incident_sign = ra_dot(
        incident->axis[incident_axis], incident_target) < 0.0f
        ? -1.0f : 1.0f;

    RaVec3 input[PL_SAT_MAX_CLIP_VERTICES];
    RaVec3 scratch[PL_SAT_MAX_CLIP_VERTICES];
    pl_sface(incident, incident_axis, incident_sign, input);
    int count = 4;

    const int tangent_a = (reference_axis + 1) % 3;
    const int tangent_b = (reference_axis + 2) % 3;
    const RaVec3 side_a = reference->axis[tangent_a];
    const RaVec3 side_b = reference->axis[tangent_b];
    const float half_a = pl_shalf(
        reference->half_extents, tangent_a);
    const float half_b = pl_shalf(
        reference->half_extents, tangent_b);
    const float center_a = ra_dot(reference_center, side_a);
    const float center_b = ra_dot(reference_center, side_b);
    count = pl_sclip(input, count, scratch, side_a,
        center_a + half_a);
    count = pl_sclip(scratch, count, input, pl_sat_neg(side_a),
        -center_a + half_a);
    count = pl_sclip(input, count, scratch, side_b,
        center_b + half_b);
    count = pl_sclip(scratch, count, input, pl_sat_neg(side_b),
        -center_b + half_b);

    if (count <= 0) {
        manifold->point[0] = query->contact;
        manifold->point_feature[0] = pl_sfid(
            query->feature, 0);
        manifold->count = 1;
        return 1;
    }

    const float direction_a[4] = {-1.0f, 1.0f, 1.0f, -1.0f};
    const float direction_b[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
    int selected[4];
    int selected_count = 0;
    for (int corner = 0; corner < 4 && selected_count < 4; ++corner) {
        int best = -1;
        float best_projection = -3.402823466e+38f;
        for (int index = 0; index < count; ++index) {
            if (pl_sused(
                    selected, selected_count, index)) continue;
            const float projection = direction_a[corner]
                    * ra_dot(input[index], side_a)
                + direction_b[corner] * ra_dot(input[index], side_b);
            if (projection > best_projection) {
                best_projection = projection;
                best = index;
            }
        }
        if (best >= 0) {
            selected[selected_count++] = best;
        }
    }
    for (int index = 0; index < count && selected_count < 4; ++index) {
        if (!pl_sused(
                selected, selected_count, index)) {
            selected[selected_count++] = index;
        }
    }

    for (int output = 0; output < selected_count; ++output) {
        const RaVec3 incident_point = input[selected[output]];
        const float separation = ra_dot(
            ra_sub(incident_point, reference_center), reference_normal);
        if (separation > margin) {
            continue;
        }
        const RaVec3 reference_point = ra_sub(
            incident_point, ra_scale(reference_normal, separation));
        RaConvexContact contact;
        contact.hit = 1;
        contact.iterations = query->contact.iterations;
        contact.separation = separation;
        contact.normal = contact_normal;
        if (reference_is_a) {
            contact.point_a = reference_point;
            contact.point_b = incident_point;
        } else {
            contact.point_a = incident_point;
            contact.point_b = reference_point;
        }
        if (pl_sdup(
                manifold, contact.point_a, contact.point_b)) continue;
        int output_index = manifold->count++;
        manifold->point[output_index] = contact;
        manifold->point_feature[output_index] = pl_sfid(
            query->feature, selected[output]);
        if (manifold->count == PL_SAT_MAX_MANIFOLD_POINTS) {
            break;
        }
    }

    if (manifold->count == 0) {
        manifold->point[0] = query->contact;
        manifold->point_feature[0] = pl_sfid(
            query->feature, 0);
        manifold->count = 1;
    }
    return manifold->count;
}

RA_D static RA_INLINE PlSatQuery pl_sssph(
        const RaConvexShape* a, const RaConvexShape* b, float margin) {
    PlSatQuery query;
    memset(&query, 0, sizeof(query));
    RaVec3 delta = ra_sub(a->pose.position, b->pose.position);
    float distance_squared = ra_dot(delta, delta);
    float distance = sqrtf(ra_max(distance_squared, 0.0f));
    RaVec3 normal = distance > 1.0e-10f
        ? ra_scale(delta, 1.0f / distance) : ra_v3(1, 0, 0);
    float radius_a = a->half_extents.x;
    float radius_b = b->half_extents.x;
    query.contact.iterations = 1;
    query.contact.separation = distance - radius_a - radius_b;
    query.contact.normal = normal;
    query.contact.point_a = ra_sub(
        a->pose.position, ra_scale(normal, radius_a));
    query.contact.point_b = ra_add(
        b->pose.position, ra_scale(normal, radius_b));
    query.contact.hit = query.contact.separation <= margin;
    query.feature = PL_SAT_FACE_A_X;
    return query;
}

RA_D static RA_INLINE PlSatQuery pl_ssbox(
        const RaConvexShape* sphere, const RaConvexShape* box,
        float margin) {
    PlSatQuery query;
    memset(&query, 0, sizeof(query));
    PlSatObb obb = pl_sobb(box);
    RaVec3 center_delta = ra_sub(sphere->pose.position, obb.center);
    float local[3] = {
        ra_dot(center_delta, obb.axis[0]),
        ra_dot(center_delta, obb.axis[1]),
        ra_dot(center_delta, obb.axis[2]),
    };
    float half[3] = {
        obb.half_extents.x, obb.half_extents.y, obb.half_extents.z};
    RaVec3 closest = obb.center;
    for (int axis = 0; axis < 3; ++axis) {
        closest = ra_add(closest, ra_scale(obb.axis[axis],
            ra_clamp(local[axis], -half[axis], half[axis])));
    }
    RaVec3 delta = ra_sub(sphere->pose.position, closest);
    float distance_squared = ra_dot(delta, delta);
    float radius = sphere->half_extents.x;
    int feature_axis = 0;
    if (distance_squared > 1.0e-20f) {
        float distance = sqrtf(distance_squared);
        query.contact.normal = ra_scale(delta, 1.0f / distance);
        query.contact.separation = distance - radius;
        float best_axis = fabsf(ra_dot(query.contact.normal, obb.axis[0]));
        for (int axis = 1; axis < 3; ++axis) {
            float alignment = fabsf(
                ra_dot(query.contact.normal, obb.axis[axis]));
            if (alignment > best_axis) {
                best_axis = alignment;
                feature_axis = axis;
            }
        }
        query.contact.point_b = closest;
    } else {
        float clearance = half[0] - fabsf(local[0]);
        for (int axis = 1; axis < 3; ++axis) {
            float candidate = half[axis] - fabsf(local[axis]);
            if (candidate < clearance) {
                clearance = candidate;
                feature_axis = axis;
            }
        }
        float sign = local[feature_axis] < 0.0f ? -1.0f : 1.0f;
        query.contact.normal = ra_scale(obb.axis[feature_axis], sign);
        query.contact.separation = -radius - clearance;
        query.contact.point_b = ra_add(sphere->pose.position,
            ra_scale(query.contact.normal, clearance));
    }
    query.contact.point_a = ra_sub(sphere->pose.position,
        ra_scale(query.contact.normal, radius));
    query.contact.hit = query.contact.separation <= margin;
    query.contact.iterations = 1;
    query.feature = PL_SAT_FACE_B_X + feature_axis;
    return query;
}

RA_D static RA_INLINE PlSatQuery pl_sq(
        const RaConvexShape* a, const RaConvexShape* b, float margin) {
    if (a->type == RA_CONVEX_SPHERE && b->type == RA_CONVEX_SPHERE) {
        return pl_sssph(a, b, margin);
    }
    if (a->type == RA_CONVEX_SPHERE && b->type == RA_CONVEX_BOX) {
        return pl_ssbox(a, b, margin);
    }
    if (a->type == RA_CONVEX_BOX && b->type == RA_CONVEX_SPHERE) {
        PlSatQuery query = pl_ssbox(b, a, margin);
        RaVec3 point = query.contact.point_a;
        query.contact.point_a = query.contact.point_b;
        query.contact.point_b = point;
        query.contact.normal = pl_sat_neg(query.contact.normal);
        query.feature -= PL_SAT_FACE_B_X;
        return query;
    }
    PlSatObb box_a = pl_sobb(a);
    PlSatObb box_b = pl_sobb(b);
    return pl_sqobb(&box_a, &box_b, margin);
}

RA_D static RA_INLINE int pl_smans(
        const RaConvexShape* a, const RaConvexShape* b, float margin,
        PlSatManifold* manifold) {
    pl_sclr(manifold);
    if (a->type != RA_CONVEX_BOX || b->type != RA_CONVEX_BOX) {
        PlSatQuery query = pl_sq(a, b, margin);
        if (!query.contact.hit) {
            return 0;
        }
        manifold->count = 1;
        manifold->feature = query.feature;
        manifold->point[0] = query.contact;
        manifold->point_feature[0] = pl_sfid(
            query.feature, 0);
        return 1;
    }
    PlSatObb box_a = pl_sobb(a);
    PlSatObb box_b = pl_sobb(b);
    PlSatQuery query = pl_sqobb(&box_a, &box_b, margin);
    return pl_sman(
        &box_a, &box_b, margin, &query, manifold);
}

#define PL_IMPULSE_MAX_POINTS PL_SAT_MAX_MANIFOLD_POINTS
#define PL_IMPULSE_MAX_CACHE 192
#define PL_IMPULSE_MAX_SOLVER_ITERS 64
#define PL_IMPULSE_EPSILON 1.0e-8f

typedef struct PlImpulseConfig {
    int velocity_iterations;
    int position_iterations;
    float velocity_impulse_tolerance;  // 0 disables; does not lower the hard cap
    int warm_start;
    int split_position;
    float position_beta;
    float slop;
    float speculative_margin;
    float static_friction;
    float dynamic_friction;
    float restitution;
    float restitution_threshold;
    float max_normal_impulse;
    float max_position_correction;
    float max_position_impulse;
    int cache_max_age;
} PlImpulseConfig;

typedef struct PlImpulsePatch {
    float area;  // 0 disables torsion
    RaVec3 centroid;
    float second_11;
    float second_22;
    float second_12;
} PlImpulsePatch;

typedef struct PlImpulseCandidate {
    RaConvexContact contact;
    uint32_t feature;
    uint32_t patch_group;
    PlImpulsePatch patch;
} PlImpulseCandidate;

typedef struct PlImpulseReaction {
    int active;  // mass==0 proxy; live qd/jaw, not cached body_b velocity
    float inverse_mass[3];
    float jaw_velocity_response[3];
    float robot_jacobian[3][RA_DOF];
    float jaw_jacobian[3];
    float robot_response[3][RA_DOF];
} PlImpulseReaction;

typedef struct PlImpulseAngularReaction {
    int active;
    float inverse_mass;
    float robot_jacobian[RA_DOF];
    float robot_response[RA_DOF];
} PlImpulseAngularReaction;

typedef struct PlImpulsePoint {
    uint32_t feature;
    uint32_t patch_group;
    RaVec3 point_a;
    RaVec3 point_b;
    RaVec3 local_a;
    RaVec3 local_b;
    float separation;
    float normal_mass;
    float tangent_1_mass;
    float tangent_2_mass;
    float normal_impulse;
    float tangent_1_impulse;
    float tangent_2_impulse;
    float velocity_bias;
    float pre_normal_velocity;
    float normal_erp;  // erp [1/s]; cfm = 1/(dt*(c+dt*k)); 0 = hard row
    float normal_cfm;
    float prescribed_separation_offset;  // split correction on FK proxy
    PlImpulsePatch patch;
    PlImpulseReaction reaction;
} PlImpulsePoint;

typedef struct PlImpulseManifold {
    int body_a;
    int body_b;
    uint32_t pair_key;
    RaVec3 normal;
    RaVec3 tangent_1;
    RaVec3 tangent_2;
    float static_friction;
    float dynamic_friction;
    float restitution;
    float patch_area;
    RaVec3 patch_centroid;
    float patch_second_11;
    float patch_second_22;
    float patch_second_12;
    float torsional_radius;
    float patch_second_moment;
    float torsional_impulse;
    float torsional_mass;
    PlImpulseAngularReaction angular_reaction;
    uint32_t angular_cache_feature;  // independent of point[0].feature
    int point_count;
    PlImpulsePoint points[PL_IMPULSE_MAX_POINTS];
} PlImpulseManifold;

typedef struct PlImpulseCacheEntry {
    int body_a;
    int body_b;
    uint32_t pair_key;
    uint32_t feature;
    uint32_t stamp;
    RaVec3 normal;
    RaVec3 tangent_1;
    RaVec3 tangent_2;
    float normal_impulse;
    float tangent_1_impulse;
    float tangent_2_impulse;
    float torsional_impulse;
} PlImpulseCacheEntry;

typedef struct PlImpulseCache {
    uint32_t tick;
    int count;
    PlImpulseCacheEntry entries[PL_IMPULSE_MAX_CACHE];
} PlImpulseCache;

RA_D static RA_INLINE int pl_ifin(float value) {
    return value == value && value < 1.0e30f && value > -1.0e30f;  // no isfinite()
}

RA_D static RA_INLINE float pl_ilen(RaVec3 value) {
    float length_squared = ra_dot(value, value);
    return length_squared > PL_IMPULSE_EPSILON * PL_IMPULSE_EPSILON
        ? sqrtf(length_squared) : 0.0f;
}

RA_D static RA_INLINE RaVec3 pl_inrm(
        RaVec3 value, RaVec3 fallback) {
    float length = pl_ilen(value);
    if (length > 0.0f) {
        return ra_scale(value, 1.0f / length);
    }
    length = pl_ilen(fallback);
    if (length > 0.0f) {
        return ra_scale(fallback, 1.0f / length);
    }
    return ra_v3(1.0f, 0.0f, 0.0f);
}

RA_D static RA_INLINE void pl_itan(
        RaVec3 normal, RaVec3* tangent_1, RaVec3* tangent_2) {
    RaVec3 reference = fabsf(normal.y) < 0.90f
        ? ra_v3(0.0f, 1.0f, 0.0f) : ra_v3(1.0f, 0.0f, 0.0f);
    *tangent_1 = pl_inrm(
        ra_cross(reference, normal), ra_v3(0.0f, 0.0f, 1.0f));
    *tangent_2 = pl_inrm(
        ra_cross(normal, *tangent_1), ra_v3(0.0f, 1.0f, 0.0f));
}

RA_D static RA_INLINE uint32_t pl_ihash(
        int body_a, int body_b) {
    uint32_t value = 2166136261u;
    value = (value ^ (uint32_t)(body_a + 1)) * 16777619u;
    value = (value ^ (uint32_t)(body_b + 1)) * 16777619u;
    return value;
}

RA_D static RA_INLINE RaVec3 pl_ipt(
        const PlImpulseCandidate* candidate) {
    return ra_scale(ra_add(candidate->contact.point_a,
        candidate->contact.point_b), 0.5f);
}

RA_D static RA_INLINE int pl_ibef(
        const PlImpulseCandidate* left, const PlImpulseCandidate* right) {
    if (left->feature != right->feature) {
        return left->feature < right->feature;
    }
    if (left->contact.separation != right->contact.separation) {
        return left->contact.separation < right->contact.separation;
    }
    RaVec3 left_point = pl_ipt(left);
    RaVec3 right_point = pl_ipt(right);
    if (left_point.x != right_point.x) {
        return left_point.x < right_point.x;
    }
    if (left_point.y != right_point.y) {
        return left_point.y < right_point.y;
    }
    return left_point.z < right_point.z;
}

RA_D static RA_INLINE int pl_ius(
        const PlImpulseCandidate* candidate, float margin) {
    const RaConvexContact* contact = &candidate->contact;
    if (!pl_ifin(contact->separation)
            || !pl_ifin(contact->normal.x)
            || !pl_ifin(contact->normal.y)
            || !pl_ifin(contact->normal.z)) return 0;
    if (!pl_ifin(contact->point_a.x)
            || !pl_ifin(contact->point_a.y)
            || !pl_ifin(contact->point_a.z)
            || !pl_ifin(contact->point_b.x)
            || !pl_ifin(contact->point_b.y)
            || !pl_ifin(contact->point_b.z)) return 0;
    return contact->hit || contact->separation <= margin;
}

RA_D static RA_INLINE int pl_iman(
        int body_a, int body_b, const PlImpulseCandidate* candidates,
        int candidate_count, float margin, float static_friction,
        float dynamic_friction, float restitution,
        PlImpulseManifold* manifold) {
    manifold->body_a = body_a;
    manifold->body_b = body_b;
    manifold->pair_key = pl_ihash(body_a, body_b);
    manifold->angular_cache_feature = 0xa0000000u
        | (manifold->pair_key & 0x0fffffffu);
    manifold->normal = ra_v3(1.0f, 0.0f, 0.0f);
    manifold->tangent_1 = ra_v3(0.0f, 1.0f, 0.0f);
    manifold->tangent_2 = ra_v3(0.0f, 0.0f, 1.0f);
    manifold->static_friction = ra_max(static_friction, 0.0f);
    manifold->dynamic_friction = ra_clamp(dynamic_friction, 0.0f,
        manifold->static_friction);
    manifold->restitution = ra_clamp(restitution, 0.0f, 1.0f);
    manifold->point_count = 0;

    PlImpulseCandidate work[PL_IMPULSE_MAX_CANDIDATES];
    int count = ra_min(ra_max(candidate_count, 0), PL_IMPULSE_MAX_CANDIDATES);
    int usable = 0;
    for (int index = 0; index < count; ++index) {
        if (!pl_ius(&candidates[index], margin)) {
            continue;
        }
        if (usable >= PL_IMPULSE_MAX_CANDIDATES) {
            break;
        }
        work[usable] = candidates[index];
        ++usable;
    }
    for (int index = 1; index < usable; ++index) {
        PlImpulseCandidate value = work[index];
        int cursor = index;
        while (cursor > 0
                && pl_ibef(&value, &work[cursor - 1])) {
            work[cursor] = work[cursor - 1];
            --cursor;
        }
        work[cursor] = value;
    }
    if (usable == 0) {
        return 0;
    }

    int unique = 0;
    for (int index = 0; index < usable; ++index) {
        if (unique > 0 && work[index].feature
                == work[unique - 1].feature) continue;
        work[unique++] = work[index];
    }
    usable = unique;
    int selected[PL_IMPULSE_MAX_POINTS];
    int selected_count = 0;
    int deepest = 0;
    for (int index = 1; index < usable; ++index) {
        if (work[index].contact.separation
                < work[deepest].contact.separation) deepest = index;
    }
    selected[selected_count++] = deepest;
    while (selected_count < PL_IMPULSE_MAX_POINTS
            && selected_count < usable) {
        int best = -1;
        float best_score = -1.0f;
        for (int index = 0; index < usable; ++index) {
            int already = 0;
            for (int slot = 0; slot < selected_count; ++slot) {
                if (selected[slot] == index) {
                    already = 1;
                }
            }
            if (already) {
                continue;
            }
            RaVec3 point = pl_ipt(&work[index]);
            float score = 1.0e30f;
            for (int slot = 0; slot < selected_count; ++slot) {
                RaVec3 other = pl_ipt(
                    &work[selected[slot]]);
                RaVec3 delta = ra_sub(point, other);
                score = ra_min(score, ra_dot(delta, delta));
            }
            if (best < 0 || score > best_score
                    || (score == best_score
                        && work[index].feature < work[best].feature)) {
                best = index;
                best_score = score;
            }
        }
        if (best < 0) {
            break;
        }
        selected[selected_count++] = best;
    }
    for (int index = 1; index < selected_count; ++index) {
        int value = selected[index];
        int cursor = index;
        while (cursor > 0 && work[value].feature
                < work[selected[cursor - 1]].feature) {
            selected[cursor] = selected[cursor - 1];
            --cursor;
        }
        selected[cursor] = value;
    }

    RaVec3 normal = work[deepest].contact.normal;
    RaVec3 fallback = ra_sub(work[deepest].contact.point_a,
        work[deepest].contact.point_b);
    manifold->normal = pl_inrm(normal, fallback);
    pl_itan(manifold->normal, &manifold->tangent_1,
        &manifold->tangent_2);
    manifold->torsional_radius = 0.0f;
    manifold->patch_area = 0.0f;
    manifold->patch_centroid = ra_v3(0.0f, 0.0f, 0.0f);
    manifold->patch_second_11 = 0.0f;
    manifold->patch_second_22 = 0.0f;
    manifold->patch_second_12 = 0.0f;
    manifold->patch_second_moment = 0.0f;
    manifold->torsional_impulse = 0.0f;
    manifold->torsional_mass = 0.0f;
    memset(&manifold->angular_reaction, 0,
        sizeof(manifold->angular_reaction));
    manifold->point_count = selected_count;
    for (int slot = 0; slot < selected_count; ++slot) {
        const PlImpulseCandidate* candidate = &work[selected[slot]];
        PlImpulsePoint* point = &manifold->points[slot];
        point->feature = candidate->feature;
        point->patch_group = candidate->patch_group;
        point->point_a = candidate->contact.point_a;
        point->point_b = candidate->contact.point_b;
        point->local_a = ra_v3(0.0f, 0.0f, 0.0f);
        point->local_b = ra_v3(0.0f, 0.0f, 0.0f);
        point->separation = candidate->contact.separation;
        point->normal_mass = 0.0f;
        point->tangent_1_mass = 0.0f;
        point->tangent_2_mass = 0.0f;
        point->normal_impulse = 0.0f;
        point->tangent_1_impulse = 0.0f;
        point->tangent_2_impulse = 0.0f;
        point->velocity_bias = 0.0f;
        point->pre_normal_velocity = 0.0f;
        point->normal_erp = 0.0f;
        point->normal_cfm = 0.0f;
        point->prescribed_separation_offset = 0.0f;
        point->patch = candidate->patch;
        memset(&point->reaction, 0, sizeof(point->reaction));
    }
    return selected_count;
}

RA_D static RA_INLINE void pl_isort(
        PlImpulseManifold* manifolds, int manifold_count) {
    int count = ra_min(ra_max(manifold_count, 0), PL_IMPULSE_MAX_MANIFOLDS);
    for (int index = 1; index < count; ++index) {
        PlImpulseManifold value = manifolds[index];
        int cursor = index;
        while (cursor > 0) {
            const PlImpulseManifold* left = &value;
            const PlImpulseManifold* right = &manifolds[cursor - 1];
            int before = 0;
            if (left->pair_key != right->pair_key) {
                before = left->pair_key < right->pair_key;
            } else if (left->body_a != right->body_a) {
                before = left->body_a < right->body_a;
            } else if (left->body_b != right->body_b) {
                before = left->body_b < right->body_b;
            } else if (left->point_count != right->point_count) {
                before = left->point_count < right->point_count;
            } else {
                int point_count = ra_min(ra_max(left->point_count, 0),
                    PL_IMPULSE_MAX_POINTS);
                for (int point = 0; point < point_count; ++point) {
                    if (left->points[point].feature
                            != right->points[point].feature) {
                        before = left->points[point].feature
                            < right->points[point].feature;
                        break;
                    }
                }
            }
            if (!before) {
                break;
            }
            manifolds[cursor] = manifolds[cursor - 1];
            --cursor;
        }
        manifolds[cursor] = value;
    }
}

RA_HD static RA_INLINE void pl_iclr(
        PlImpulseCache* cache) {
    cache->tick = 0;
    cache->count = 0;
}

RA_D static RA_INLINE int pl_ifind(
        const PlImpulseCache* cache, int body_a, int body_b,
        uint32_t feature) {
    uint32_t pair_key = pl_ihash(body_a, body_b);
    for (int index = 0; index < cache->count; ++index) {
        const PlImpulseCacheEntry* entry = &cache->entries[index];
        if (entry->pair_key == pair_key
                && entry->body_a == body_a && entry->body_b == body_b
                && entry->feature == feature) return index;
    }
    return -1;
}

RA_D static RA_INLINE int pl_islot(
        const PlImpulseCache* cache, int body_a, int body_b,
        uint32_t feature) {
    int existing = pl_ifind(cache, body_a, body_b, feature);
    if (existing >= 0) {
        return existing;
    }
    if (cache->count < PL_IMPULSE_MAX_CACHE) {
        return cache->count;
    }
    int best = 0;
    for (int index = 1; index < cache->count; ++index) {
        const PlImpulseCacheEntry* left = &cache->entries[index];
        const PlImpulseCacheEntry* right = &cache->entries[best];
        if (left->stamp < right->stamp
                || (left->stamp == right->stamp
                    && (left->pair_key < right->pair_key
                        || (left->pair_key == right->pair_key
                            && (left->feature < right->feature
                                || (left->feature == right->feature
                                    && index < best)))))) best = index;
    }
    return best;
}

RA_D static RA_INLINE void pl_iexp(
        PlImpulseCache* cache, int max_age) {
    int write = 0;
    for (int read = 0; read < cache->count; ++read) {
        PlImpulseCacheEntry* entry = &cache->entries[read];
        uint32_t age = cache->tick - entry->stamp;
        if (age > (uint32_t)max_age) {
            continue;
        }
        if (write != read) {
            cache->entries[write] = *entry;
        }
        ++write;
    }
    cache->count = write;
}

RA_D static RA_INLINE RaVec3 pl_ianch(
        const RaRigidBody* body, RaVec3 local) {
    return ra_add(body->pose.position,
        ra_rotate(body->pose.rotation, local));
}

RA_D static RA_INLINE float pl_imass(
        const RaRigidBody* body_a, RaVec3 point_a,
        const RaRigidBody* body_b, RaVec3 point_b, RaVec3 direction,
        const PlImpulseReaction* reaction, int direction_index) {
    float denominator = ra_impd(body_a, point_a, direction)
        + ra_impd(body_b, point_b, direction);
    if (reaction != NULL && reaction->active
            && direction_index >= 0 && direction_index < 3) {
        denominator += ra_max(reaction->inverse_mass[direction_index], 0.0f);
    }
    return denominator;
}

RA_D static RA_INLINE void pl_iap(
        RaRigidBody* body_a, RaVec3 point_a, RaRigidBody* body_b,
        RaVec3 point_b, RaVec3 impulse) {
    ra_impa(body_a, point_a, impulse);
    ra_impa(body_b, point_b, ra_scale(impulse, -1.0f));
}

RA_D static RA_INLINE void pl_iar(
        const PlImpulseReaction* reaction, RaState* reaction_state,
        int direction_index, float impulse) {
    if (reaction == NULL || !reaction->active || reaction_state == NULL
            || direction_index < 0 || direction_index >= 3) return;
    ra_applyr(reaction_state,
        reaction->robot_response[direction_index], -impulse);
    reaction_state->gripper_velocity +=
        reaction->jaw_velocity_response[direction_index] * impulse;
}

RA_D static RA_INLINE void pl_iupo(
        PlImpulseManifold* manifolds, int manifold_count,
        const float delta_q[RA_DOF], float delta_width) {
    int count = ra_min(ra_max(manifold_count, 0),
        PL_IMPULSE_MAX_MANIFOLDS);
    for (int manifold_index = 0; manifold_index < count;
            ++manifold_index) {
        PlImpulseManifold* manifold = &manifolds[manifold_index];
        for (int point_index = 0; point_index < manifold->point_count;
                ++point_index) {
            PlImpulsePoint* point = &manifold->points[point_index];
            if (!point->reaction.active) {
                continue;
            }
            float displacement = point->reaction.jaw_jacobian[0]
                * delta_width;
            for (int joint = 0; joint < RA_DOF; ++joint) {
                displacement += point->reaction.robot_jacobian[0][joint]
                    * delta_q[joint];
            }
            point->prescribed_separation_offset -= displacement;
        }
    }
}

RA_D static RA_INLINE float pl_ibvel(
        const RaRigidBody* body, RaVec3 point, RaVec3 direction) {
    RaVec3 velocity = ra_add(body->linear_velocity,
        ra_cross(body->angular_velocity,
            ra_sub(point, body->pose.position)));
    return ra_dot(velocity, direction);
}

RA_D static RA_INLINE float pl_irvel(
        const RaRigidBody* body, const PlImpulseReaction* reaction,
        const RaState* reaction_state, int direction_index,
        RaVec3 point, RaVec3 direction) {
    if (reaction != NULL && reaction->active && reaction_state != NULL
            && direction_index >= 0 && direction_index < 3) {
        float velocity = reaction->jaw_jacobian[direction_index]
            * reaction_state->gripper_velocity;
        for (int joint = 0; joint < RA_DOF; ++joint) {
            velocity += reaction->robot_jacobian[direction_index][joint]
                * reaction_state->qd[joint];
        }
        return velocity;
    }
    return pl_ibvel(body, point, direction);
}

RA_D static RA_INLINE float pl_iam(
        const RaRigidBody* body, RaVec3 axis) {
    if (body->mass <= 0.0f) {
        return 0.0f;
    }
    return ra_dot(axis, ra_invi(body, axis));
}

RA_D static RA_INLINE void pl_iaap(
        RaRigidBody* body_a, RaRigidBody* body_b, RaVec3 axis,
        float moment) {
    if (body_a->mass > 0.0f) {
        body_a->angular_velocity = ra_add(body_a->angular_velocity,
            ra_scale(ra_invi(body_a, axis), moment));
    }
    if (body_b->mass > 0.0f) {
        body_b->angular_velocity = ra_add(body_b->angular_velocity,
            ra_scale(ra_invi(body_b, axis), -moment));
    }
}

RA_D static RA_INLINE void pl_iaar(
        const PlImpulseAngularReaction* reaction, RaState* reaction_state,
        float moment) {
    if (reaction == NULL || !reaction->active || reaction_state == NULL) {
        return;
    }
    ra_applyr(reaction_state,
        reaction->robot_response, -moment);
}

RA_D static RA_INLINE void pl_iapos(
        RaRigidBody* body, RaVec3 point, RaVec3 impulse) {
    if (body->mass <= 0.0f) {
        return;
    }
    RaVec3 lever = ra_sub(point, body->pose.position);
    body->pose.position = ra_add(body->pose.position,
        ra_scale(impulse, 1.0f / ra_max(body->mass, 1.0e-8f)));
    RaVec3 angular_delta = ra_invi(body,
        ra_cross(lever, impulse));
    body->pose.rotation = ra_qint(body->pose.rotation,
        angular_delta, 1.0f);
}

RA_D static RA_INLINE void pl_iref(
        RaRigidBody* body_a, RaRigidBody* body_b,
        const PlImpulseManifold* manifold, PlImpulsePoint* point) {
    point->point_a = pl_ianch(body_a, point->local_a);
    point->point_b = pl_ianch(body_b, point->local_b);
    point->separation = ra_dot(ra_sub(point->point_a, point->point_b),
        manifold->normal) + point->prescribed_separation_offset;
}

RA_D static RA_INLINE void pl_ildc(
        const PlImpulseCache* cache, PlImpulseManifold* manifold,
        PlImpulsePoint* point, int warm_start, int max_age) {
    point->normal_impulse = 0.0f;
    point->tangent_1_impulse = 0.0f;
    point->tangent_2_impulse = 0.0f;
    if (!warm_start) {
        return;
    }
    assert(cache != NULL);
    int slot = pl_ifind(cache, manifold->body_a,
        manifold->body_b, point->feature);
    if (slot < 0) {
        return;
    }
    const PlImpulseCacheEntry* entry = &cache->entries[slot];
    if (max_age >= 0 && cache->tick - entry->stamp > (uint32_t)max_age) {
        return;
    }
    point->normal_impulse = ra_max(entry->normal_impulse, 0.0f);
    RaVec3 old_tangent = ra_add(
        ra_scale(entry->tangent_1, entry->tangent_1_impulse),
        ra_scale(entry->tangent_2, entry->tangent_2_impulse));
    point->tangent_1_impulse = ra_dot(old_tangent, manifold->tangent_1);
    point->tangent_2_impulse = ra_dot(old_tangent, manifold->tangent_2);
    float tangent_length = hypotf(point->tangent_1_impulse,
        point->tangent_2_impulse);
    float limit = ra_max(manifold->static_friction
        * point->normal_impulse, 0.0f);
    if (tangent_length > limit) {
        float scale = limit / ra_max(tangent_length, PL_IMPULSE_EPSILON);
        point->tangent_1_impulse *= scale;
        point->tangent_2_impulse *= scale;
    }
}

RA_D static RA_INLINE void pl_ilda(
        const PlImpulseCache* cache, PlImpulseManifold* manifold,
        int warm_start, int max_age) {
    manifold->torsional_impulse = 0.0f;
    if (!warm_start || cache == NULL) {
        return;
    }
    int slot = pl_ifind(cache, manifold->body_a,
        manifold->body_b, manifold->angular_cache_feature);
    if (slot < 0) {
        return;
    }
    const PlImpulseCacheEntry* entry = &cache->entries[slot];
    if (max_age >= 0 && cache->tick - entry->stamp > (uint32_t)max_age) {
        return;
    }
    RaVec3 old_moment = ra_scale(entry->normal, entry->torsional_impulse);
    manifold->torsional_impulse = ra_dot(old_moment, manifold->normal);
}

RA_D static RA_INLINE void pl_iprep(
        RaRigidBody* bodies, int body_count, PlImpulseManifold* manifolds,
        int manifold_count, const PlImpulseCache* cache,
        const PlImpulseConfig* config) {
    int count = ra_min(ra_max(manifold_count, 0), PL_IMPULSE_MAX_MANIFOLDS);
    for (int manifold_index = 0; manifold_index < count; ++manifold_index) {
        PlImpulseManifold* manifold = &manifolds[manifold_index];
        if (manifold->body_a < 0 || manifold->body_b < 0
                || manifold->body_a >= body_count
                || manifold->body_b >= body_count
                || manifold->body_a == manifold->body_b) {
            manifold->point_count = 0;
            continue;
        }
        RaRigidBody* body_a = &bodies[manifold->body_a];
        RaRigidBody* body_b = &bodies[manifold->body_b];
        manifold->normal = pl_inrm(manifold->normal,
            ra_sub(body_a->pose.position, body_b->pose.position));
        pl_itan(manifold->normal, &manifold->tangent_1,
            &manifold->tangent_2);
        manifold->static_friction = ra_max(manifold->static_friction, 0.0f);
        manifold->dynamic_friction = ra_clamp(manifold->dynamic_friction,
            0.0f, manifold->static_friction);
        manifold->restitution = ra_clamp(manifold->restitution, 0.0f, 1.0f);
        int point_count = ra_min(ra_max(manifold->point_count, 0),
            PL_IMPULSE_MAX_POINTS);
        manifold->point_count = point_count;
        for (int point_index = 0; point_index < point_count; ++point_index) {
            PlImpulsePoint* point = &manifold->points[point_index];
            point->local_a = ra_rotate(ra_qconj(body_a->pose.rotation),
                ra_sub(point->point_a, body_a->pose.position));
            point->local_b = ra_rotate(ra_qconj(body_b->pose.rotation),
                ra_sub(point->point_b, body_b->pose.position));
            pl_iref(body_a, body_b, manifold, point);
            point->normal_mass = 1.0f / ra_max(
                pl_imass(body_a, point->point_a,
                    body_b, point->point_b, manifold->normal,
                    &point->reaction, 0) + point->normal_cfm,
                PL_IMPULSE_EPSILON);
            point->tangent_1_mass = 1.0f / ra_max(
                pl_imass(body_a, point->point_a,
                    body_b, point->point_b, manifold->tangent_1,
                    &point->reaction, 1),
                PL_IMPULSE_EPSILON);
            point->tangent_2_mass = 1.0f / ra_max(
                pl_imass(body_a, point->point_a,
                    body_b, point->point_b, manifold->tangent_2,
                    &point->reaction, 2),
                PL_IMPULSE_EPSILON);
            pl_ildc(cache, manifold, point,
                config->warm_start != 0, config->cache_max_age);
        }
        pl_ilda(cache, manifold,
            config->warm_start != 0, config->cache_max_age);
        if (manifold->torsional_radius > 0.0f) {
            float denominator = pl_iam(
                body_a, manifold->normal)
                + pl_iam(body_b, manifold->normal)
                + manifold->angular_reaction.inverse_mass;
            manifold->torsional_mass = 1.0f / ra_max(
                denominator, PL_IMPULSE_EPSILON);
        }
    }
}

RA_D static RA_INLINE void pl_ispos(
        RaRigidBody* bodies, int body_count, PlImpulseManifold* manifolds,
        int manifold_count, const PlImpulseConfig* config,
        RaState* reaction_state) {
    if (!config->split_position) {
        return;
    }
    int count = ra_min(ra_max(manifold_count, 0), PL_IMPULSE_MAX_MANIFOLDS);
    int iterations = ra_min(ra_max(config->position_iterations, 0),
        PL_IMPULSE_MAX_SOLVER_ITERS);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        float maximum_penetration = 0.0f;
        for (int manifold_index = 0; manifold_index < count;
                ++manifold_index) {
            PlImpulseManifold* manifold = &manifolds[manifold_index];
            if (manifold->point_count <= 0
                    || manifold->body_a < 0 || manifold->body_b < 0
                    || manifold->body_a >= body_count
                    || manifold->body_b >= body_count) continue;
            RaRigidBody* body_a = &bodies[manifold->body_a];
            RaRigidBody* body_b = &bodies[manifold->body_b];
            for (int point_index = 0; point_index < manifold->point_count;
                    ++point_index) {
                PlImpulsePoint* point = &manifold->points[point_index];
                pl_iref(body_a, body_b, manifold, point);
                float penetration = ra_max(-point->separation
                    - config->slop, 0.0f);
                maximum_penetration = ra_max(maximum_penetration, penetration);
                if (penetration <= 0.0f) {
                    continue;
                }
                float correction = config->position_beta * penetration;
                correction = ra_min(correction,
                    ra_max(config->max_position_correction, 0.0f));
                float denominator = pl_imass(body_a,
                    point->point_a, body_b, point->point_b, manifold->normal,
                    &point->reaction, 0) + point->normal_cfm;
                if (denominator <= PL_IMPULSE_EPSILON) {
                    continue;
                }
                float magnitude = correction / denominator;
                magnitude = ra_min(magnitude,
                    ra_max(config->max_position_impulse, 0.0f));
                pl_iapos(body_a, point->point_a,
                    ra_scale(manifold->normal, magnitude));
                pl_iapos(body_b, point->point_b,
                    ra_scale(manifold->normal, -magnitude));
                float delta_q[RA_DOF] = {0.0f};
                float delta_width = 0.0f;
                if (reaction_state != NULL && point->reaction.active) {
                    for (int joint = 0; joint < RA_DOF; ++joint) {
                        delta_q[joint] = -magnitude
                            * point->reaction.robot_response[0][joint];
                    }
                    delta_width = magnitude
                        * point->reaction.jaw_velocity_response[0];
                    ra_applyr(reaction_state,
                        point->reaction.robot_response[0], -magnitude);
                    reaction_state->gripper_width +=
                        point->reaction.jaw_velocity_response[0] * magnitude;
                    reaction_state->gripper_width = ra_clamp(
                        reaction_state->gripper_width, 0.0f, 0.20f);
                }
                pl_iupo(manifolds,
                    manifold_count, delta_q, delta_width);
            }
        }
        if (maximum_penetration <= config->slop) {
            break;
        }
    }
}

RA_D static RA_INLINE void pl_ibias(
        RaRigidBody* bodies, int body_count, PlImpulseManifold* manifolds,
        int manifold_count, float dt, const PlImpulseConfig* config,
        RaState* reaction_state) {
    float safe_dt = ra_max(dt, PL_IMPULSE_EPSILON);
    int count = ra_min(ra_max(manifold_count, 0), PL_IMPULSE_MAX_MANIFOLDS);
    for (int manifold_index = 0; manifold_index < count; ++manifold_index) {
        PlImpulseManifold* manifold = &manifolds[manifold_index];
        if (manifold->body_a < 0 || manifold->body_b < 0
                || manifold->body_a >= body_count
                || manifold->body_b >= body_count) continue;
        RaRigidBody* body_a = &bodies[manifold->body_a];
        RaRigidBody* body_b = &bodies[manifold->body_b];
        for (int point_index = 0; point_index < manifold->point_count;
                ++point_index) {
            PlImpulsePoint* point = &manifold->points[point_index];
            pl_iref(body_a, body_b, manifold, point);
            point->normal_mass = 1.0f / ra_max(
                pl_imass(body_a, point->point_a,
                    body_b, point->point_b, manifold->normal,
                    &point->reaction, 0) + point->normal_cfm,
                PL_IMPULSE_EPSILON);
            point->tangent_1_mass = 1.0f / ra_max(
                pl_imass(body_a, point->point_a,
                    body_b, point->point_b, manifold->tangent_1,
                    &point->reaction, 1),
                PL_IMPULSE_EPSILON);
            point->tangent_2_mass = 1.0f / ra_max(
                pl_imass(body_a, point->point_a,
                    body_b, point->point_b, manifold->tangent_2,
                    &point->reaction, 2),
                PL_IMPULSE_EPSILON);
            float velocity_a = pl_ibvel(body_a,
                point->point_a, manifold->normal);
            float velocity_b = pl_irvel(body_b,
                &point->reaction, reaction_state, 0, point->point_b,
                manifold->normal);
            point->pre_normal_velocity = velocity_a - velocity_b;
            float restitution_target = 0.0f;
            if (point->pre_normal_velocity < -config->restitution_threshold
                    && point->separation <= config->speculative_margin) {
                restitution_target = -manifold->restitution
                    * point->pre_normal_velocity;
            }
            float speculative_target = point->separation > config->slop
                ? -point->separation / safe_dt : 0.0f;  // do not clamp vn to 0
            float compliance_target = 0.0f;
            if (point->normal_erp > 0.0f && point->separation < 0.0f) {
                compliance_target = point->normal_erp
                    * (-point->separation);
            }
            float active_bias = ra_max(compliance_target,
                restitution_target);
            if (point->separation > config->slop
                    && active_bias <= 0.0f) {
                active_bias = speculative_target;
            }
            point->velocity_bias = active_bias;
        }
    }
}

RA_D static RA_INLINE float pl_intot(
        const PlImpulseManifold* manifold) {
    float total = 0.0f;
    for (int point = 0; point < manifold->point_count; ++point) {
        total += ra_max(manifold->points[point].normal_impulse, 0.0f);
    }
    return total;
}

RA_D static RA_INLINE float pl_ialim(
        const PlImpulseManifold* manifold, float friction) {
    float limit = 0.0f;
    for (int point_index = 0; point_index < manifold->point_count;
            ++point_index) {
        const PlImpulsePoint* point = &manifold->points[point_index];
        if (point->patch_group == 0) {
            continue;
        }
        int first_group = 1;
        for (int previous = 0; previous < point_index; ++previous) {
            if (manifold->points[previous].patch_group
                    == point->patch_group) {
                first_group = 0;
                break;
            }
        }
        if (!first_group) {
            continue;
        }
        float area = point->patch.area;
        if (area <= PL_IMPULSE_EPSILON) {
            continue;
        }
        float group_normal_impulse = 0.0f;
        float group_tangent_squared = 0.0f;
        for (int member = point_index; member < manifold->point_count;
                ++member) {
            const PlImpulsePoint* group_point = &manifold->points[member];
            if (group_point->patch_group != point->patch_group) {
                continue;
            }
            group_normal_impulse += ra_max(group_point->normal_impulse, 0.0f);
            group_tangent_squared += group_point->tangent_1_impulse
                * group_point->tangent_1_impulse
                + group_point->tangent_2_impulse
                    * group_point->tangent_2_impulse;
        }
        float full_capacity = ra_max(friction, 0.0f)
            * group_normal_impulse;
        float remaining_squared = full_capacity * full_capacity
            - group_tangent_squared;
        float remaining_capacity = sqrtf(ra_max(remaining_squared, 0.0f));
        float intrinsic_moment = point->patch.second_11
            + point->patch.second_22;
        float radius = sqrtf(ra_max(intrinsic_moment / area, 0.0f));
        limit += remaining_capacity * radius;
    }
    return limit;
}

RA_D static RA_INLINE void pl_iaws(
        RaRigidBody* body_a, RaRigidBody* body_b,
        PlImpulseManifold* manifold, RaState* reaction_state) {
    if (manifold->angular_reaction.active == 0) {
        return;
    }
    float angular_limit = pl_ialim(manifold,
        manifold->static_friction);
    float angular_length = sqrtf(manifold->torsional_impulse
        * manifold->torsional_impulse);
    if (angular_length > angular_limit) {
        float scale = angular_limit
            / ra_max(angular_length, PL_IMPULSE_EPSILON);
        manifold->torsional_impulse *= scale;
    }
    pl_iaap(body_a, body_b, manifold->normal,
        manifold->torsional_impulse);
    pl_iaar(&manifold->angular_reaction,
        reaction_state, manifold->torsional_impulse);
}

RA_D static RA_INLINE void pl_iafr(
        RaRigidBody* body_a, RaRigidBody* body_b,
        PlImpulseManifold* manifold, RaState* reaction_state) {
    if (manifold->angular_reaction.active == 0
            || manifold->torsional_radius <= 0.0f) return;
    float normal = pl_intot(manifold);
    if (normal <= PL_IMPULSE_EPSILON) {
        manifold->torsional_impulse = 0.0f;
        return;
    }
    float static_friction = ra_max(manifold->static_friction, 0.0f);
    float dynamic_friction = ra_clamp(manifold->dynamic_friction,
        0.0f, static_friction);
    float prescribed_velocity = 0.0f;
    for (int joint = 0; joint < RA_DOF; ++joint) {
        prescribed_velocity += manifold->angular_reaction.robot_jacobian[joint]
            * reaction_state->qd[joint];
    }
    float relative_normal = ra_dot(
        body_a->angular_velocity, manifold->normal) - prescribed_velocity;
    float old_torsion = manifold->torsional_impulse;
    float candidate_torsion = old_torsion
        - relative_normal * manifold->torsional_mass;
    float torsion_limit = pl_ialim(manifold,
        static_friction);
    if (fabsf(candidate_torsion) > torsion_limit) {
        float dynamic_limit = pl_ialim(manifold,
            dynamic_friction);
        candidate_torsion = ra_clamp(candidate_torsion,
            -dynamic_limit, dynamic_limit);
    }
    manifold->torsional_impulse = candidate_torsion;
    float torsion_delta = candidate_torsion - old_torsion;
    pl_iaap(body_a, body_b, manifold->normal,
        torsion_delta);
    pl_iaar(&manifold->angular_reaction,
        reaction_state, torsion_delta);
}

RA_D static RA_INLINE void pl_iws(
        RaRigidBody* bodies, int body_count, PlImpulseManifold* manifolds,
        int manifold_count, const PlImpulseConfig* config,
        RaState* reaction_state) {
    if (!config->warm_start) {
        return;
    }
    int count = ra_min(ra_max(manifold_count, 0), PL_IMPULSE_MAX_MANIFOLDS);
    for (int manifold_index = 0; manifold_index < count; ++manifold_index) {
        PlImpulseManifold* manifold = &manifolds[manifold_index];
        if (manifold->body_a < 0 || manifold->body_b < 0
                || manifold->body_a >= body_count
                || manifold->body_b >= body_count) continue;
        RaRigidBody* body_a = &bodies[manifold->body_a];
        RaRigidBody* body_b = &bodies[manifold->body_b];
        for (int point_index = 0; point_index < manifold->point_count;
                ++point_index) {
            PlImpulsePoint* point = &manifold->points[point_index];
            RaVec3 impulse = ra_add(
                ra_scale(manifold->normal, point->normal_impulse),
                ra_add(ra_scale(manifold->tangent_1,
                    point->tangent_1_impulse),
                    ra_scale(manifold->tangent_2,
                        point->tangent_2_impulse)));
            pl_iap(body_a, point->point_a, body_b,
                point->point_b, impulse);
            pl_iar(&point->reaction, reaction_state,
                0, point->normal_impulse);
            pl_iar(&point->reaction, reaction_state,
                1, point->tangent_1_impulse);
            pl_iar(&point->reaction, reaction_state,
                2, point->tangent_2_impulse);
        }
        pl_iaws(body_a, body_b, manifold,
            reaction_state);
    }
}

RA_D static RA_INLINE void pl_isvel(
        RaRigidBody* bodies, int body_count, PlImpulseManifold* manifolds,
        int manifold_count, const PlImpulseConfig* config,
        RaState* reaction_state) {
    int count = ra_min(ra_max(manifold_count, 0), PL_IMPULSE_MAX_MANIFOLDS);
    int iterations = ra_min(ra_max(config->velocity_iterations, 0),
        PL_IMPULSE_MAX_SOLVER_ITERS);
    pl_iws(bodies, body_count, manifolds,
        count, config, reaction_state);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        float maximum_impulse_delta = 0.0f;
        for (int manifold_index = 0; manifold_index < count;
                ++manifold_index) {
            PlImpulseManifold* manifold = &manifolds[manifold_index];
            if (manifold->point_count <= 0
                    || manifold->body_a < 0 || manifold->body_b < 0
                    || manifold->body_a >= body_count
                    || manifold->body_b >= body_count) continue;
            RaRigidBody* body_a = &bodies[manifold->body_a];
            RaRigidBody* body_b = &bodies[manifold->body_b];
            float dynamic_friction = ra_clamp(manifold->dynamic_friction,
                0.0f, manifold->static_friction);
            for (int point_index = 0; point_index < manifold->point_count;
                    ++point_index) {
                PlImpulsePoint* point = &manifold->points[point_index];
                float velocity_a_normal = pl_ibvel(
                    body_a, point->point_a, manifold->normal);
                float velocity_b_normal =
                    pl_irvel(body_b,
                        &point->reaction, reaction_state, 0,
                        point->point_b, manifold->normal);
                float normal_velocity = velocity_a_normal
                    - velocity_b_normal;
                float old_normal = point->normal_impulse;
                float candidate_normal = old_normal
                    + (point->velocity_bias - normal_velocity
                        - point->normal_cfm * old_normal)
                        * point->normal_mass;
                candidate_normal = ra_clamp(candidate_normal, 0.0f,
                    ra_max(config->max_normal_impulse, 0.0f));
                point->normal_impulse = candidate_normal;
                maximum_impulse_delta = ra_max(maximum_impulse_delta,
                    fabsf(candidate_normal - old_normal));
                pl_iap(body_a, point->point_a, body_b,
                    point->point_b, ra_scale(manifold->normal,
                        candidate_normal - old_normal));
                pl_iar(&point->reaction, reaction_state,
                    0, candidate_normal - old_normal);

                float old_tangent_1 = point->tangent_1_impulse;
                float old_tangent_2 = point->tangent_2_impulse;
                float relative_tangent_1 =
                    pl_ibvel(body_a, point->point_a,
                        manifold->tangent_1)
                    - pl_irvel(body_b,
                        &point->reaction, reaction_state, 1,
                        point->point_b, manifold->tangent_1);
                float relative_tangent_2 =
                    pl_ibvel(body_a, point->point_a,
                        manifold->tangent_2)
                    - pl_irvel(body_b,
                        &point->reaction, reaction_state, 2,
                        point->point_b, manifold->tangent_2);
                float candidate_tangent_1 = old_tangent_1
                    - relative_tangent_1 * point->tangent_1_mass;
                float candidate_tangent_2 = old_tangent_2
                    - relative_tangent_2 * point->tangent_2_mass;
                float tangent_length = hypotf(candidate_tangent_1,
                    candidate_tangent_2);
                float static_limit = manifold->static_friction
                    * candidate_normal;
                if (tangent_length > static_limit) {
                    float dynamic_limit = dynamic_friction * candidate_normal;
                    float scale = dynamic_limit
                        / ra_max(tangent_length, PL_IMPULSE_EPSILON);
                    candidate_tangent_1 *= scale;
                    candidate_tangent_2 *= scale;
                }
                point->tangent_1_impulse = candidate_tangent_1;
                point->tangent_2_impulse = candidate_tangent_2;
                maximum_impulse_delta = ra_max(maximum_impulse_delta,
                    ra_max(fabsf(candidate_tangent_1 - old_tangent_1),
                        fabsf(candidate_tangent_2 - old_tangent_2)));
                RaVec3 friction_delta = ra_add(
                    ra_scale(manifold->tangent_1,
                        candidate_tangent_1 - old_tangent_1),
                    ra_scale(manifold->tangent_2,
                        candidate_tangent_2 - old_tangent_2));
                pl_iap(body_a, point->point_a, body_b,
                    point->point_b, friction_delta);
                pl_iar(&point->reaction, reaction_state,
                    1, candidate_tangent_1 - old_tangent_1);
                pl_iar(&point->reaction, reaction_state,
                    2, candidate_tangent_2 - old_tangent_2);
            }
            float old_torsion = manifold->torsional_impulse;
            pl_iafr(
                body_a, body_b, manifold, reaction_state);
            maximum_impulse_delta = ra_max(maximum_impulse_delta,
                fabsf(manifold->torsional_impulse - old_torsion));
        }
        if (config->velocity_impulse_tolerance > 0.0f
                && maximum_impulse_delta
                    <= config->velocity_impulse_tolerance) {
            break;
        }
    }
}

RA_D static RA_INLINE void pl_iwrc(
        PlImpulseCache* cache, const PlImpulseManifold* manifolds,
        int manifold_count) {
    int count = ra_min(ra_max(manifold_count, 0), PL_IMPULSE_MAX_MANIFOLDS);
    for (int manifold_index = 0; manifold_index < count; ++manifold_index) {
        const PlImpulseManifold* manifold = &manifolds[manifold_index];
        for (int point_index = 0; point_index < manifold->point_count;
                ++point_index) {
            const PlImpulsePoint* point = &manifold->points[point_index];
            int slot = pl_islot(cache, manifold->body_a,
                manifold->body_b, point->feature);
            PlImpulseCacheEntry* entry = &cache->entries[slot];
            if (slot == cache->count) {
                ++cache->count;
            }
            entry->body_a = manifold->body_a;
            entry->body_b = manifold->body_b;
            entry->pair_key = manifold->pair_key;
            entry->feature = point->feature;
            entry->stamp = cache->tick;
            entry->normal = manifold->normal;
            entry->tangent_1 = manifold->tangent_1;
            entry->tangent_2 = manifold->tangent_2;
            entry->normal_impulse = point->normal_impulse;
            entry->tangent_1_impulse = point->tangent_1_impulse;
            entry->tangent_2_impulse = point->tangent_2_impulse;
            entry->torsional_impulse = 0.0f;
        }
        if (manifold->patch_area > PL_IMPULSE_EPSILON
                && manifold->torsional_radius > 0.0f) {
            int slot = pl_islot(cache, manifold->body_a,
                manifold->body_b, manifold->angular_cache_feature);
            PlImpulseCacheEntry* entry = &cache->entries[slot];
            if (slot == cache->count) {
                ++cache->count;
            }
            entry->body_a = manifold->body_a;
            entry->body_b = manifold->body_b;
            entry->pair_key = manifold->pair_key;
            entry->feature = manifold->angular_cache_feature;
            entry->stamp = cache->tick;
            entry->normal = manifold->normal;
            entry->tangent_1 = manifold->tangent_1;
            entry->tangent_2 = manifold->tangent_2;
            entry->normal_impulse = 0.0f;
            entry->tangent_1_impulse = 0.0f;
            entry->tangent_2_impulse = 0.0f;
            entry->torsional_impulse = manifold->torsional_impulse;
        }
    }
}

RA_D static RA_INLINE void pl_isolve(
        RaRigidBody* bodies, int body_count, PlImpulseManifold* manifolds,
        int manifold_count, float dt, const PlImpulseConfig* config,
        PlImpulseCache* cache, RaState* reaction_state) {
    assert(config != NULL);
    assert(cache != NULL);
    cache->tick += 1u;
    if (cache->tick == 0u) {
        cache->tick = 1u;
    }
    pl_iexp(cache, config->cache_max_age);
    pl_iprep(bodies, body_count, manifolds, manifold_count, cache,
        config);
    pl_ispos(bodies, body_count, manifolds, manifold_count,
        config, reaction_state);
    pl_ibias(bodies, body_count, manifolds, manifold_count,
        dt, config, reaction_state);
    pl_isvel(bodies, body_count, manifolds,
        manifold_count, config, reaction_state);
    pl_iwrc(cache, manifolds, manifold_count);
}

#define RA_CUDA_PAD_MAX_VISIBLE_CANDIDATES 128
#define RA_CUDA_CONTACT_SLOP 1.0e-5f

#define RA_CUDA_BODY_CUBE 0
#define RA_CUDA_BODY_BASE 1
#define RA_CUDA_BODY_TABLE 2
#define RA_CUDA_BODY_SHELL_START 3
#define RA_CUDA_SHELL_BOXES 5
#define RA_CUDA_BODY_LINK_START \
    (RA_CUDA_BODY_SHELL_START + RA_CUDA_SHELL_BOXES)
#define RA_CUDA_BODY_PAD_LEFT_START \
    (RA_CUDA_BODY_LINK_START + RA_DOF)
#define RA_CUDA_BODY_PAD_RIGHT_START \
    (RA_CUDA_BODY_PAD_LEFT_START + RA_PAD_BOXES)
#define RA_CUDA_ROBOT_BODY_END \
    (RA_CUDA_BODY_PAD_RIGHT_START + RA_PAD_BOXES)
#define RA_CUDA_BODY_RIM RA_CUDA_ROBOT_BODY_END
#define RA_CUDA_BODY_BACKBOARD (RA_CUDA_BODY_RIM + 1)
#define RA_CUDA_BODIES (RA_CUDA_BODY_BACKBOARD + 1)

typedef struct RaCudaRigidWorld {
    int body_count;
    int shape_count;
    int manifold_count;
    unsigned int topology;
    uint32_t compound_pad_component_mask[2];
    RaRigidBody bodies[RA_CUDA_BODIES];
    RaConvexShape shapes[RA_CUDA_BODIES];
    PlImpulseManifold manifolds[PL_IMPULSE_MAX_MANIFOLDS];
    PlImpulseCandidate compound_candidate_scratch[
        RA_CUDA_PAD_MAX_VISIBLE_CANDIDATES];
    PlImpulseCache cache;
    PlImpulseConfig config;
} RaCudaRigidWorld;

RA_D static RA_INLINE int ra_manok(
        RaCudaRigidWorld* world) {
    return world->manifold_count < PL_IMPULSE_MAX_MANIFOLDS;
}

typedef struct RaCudaProductionStaged {
    float actions[RA_ACTIONS];
    float target_width;
    float energy;
    int first_grasp;
    int grasp_broken;
    int released;
    RaPose links[RA_LINKS];
    RaVec3 origins[RA_DOF];
    RaVec3 axes[RA_DOF];
} RaCudaProductionStaged;

typedef struct RaCudaProductionWorld {
    RaState state;
    RaCudaRigidWorld rigid;
    RaCudaProductionStaged staged;
} RaCudaProductionWorld;

RA_HD static RA_INLINE unsigned int ra_topo(
        const RaState* state) {
    return state->basketball_mode ? 3u : (state->stack_mode ? 2u : 1u);
}

RA_HD static RA_INLINE void ra_rbrst(
        RaCudaRigidWorld* world, unsigned int topology) {
    memset(world, 0, sizeof(*world));
    world->topology = topology;
    world->config = (PlImpulseConfig){
        .velocity_iterations = RA_CONTACT_VELOCITY_ITERS,
        .position_iterations = RA_CONTACT_POSITION_ITERS,
        .velocity_impulse_tolerance = 1.0e-7f,
        .warm_start = 1,
        .split_position = 1,
        .position_beta = 0.80f,
        .slop = RA_CUDA_CONTACT_SLOP,
        .speculative_margin = RA_CONTACT_MARGIN,
        .static_friction = 0.80f,
        .dynamic_friction = 0.70f,
        .restitution = 0.12f,
        .restitution_threshold = RA_RESTITUTION_THRESHOLD,
        .max_normal_impulse = 1.0e4f,
        .max_position_correction = 0.010f,
        .max_position_impulse = 1.0e4f,
        .cache_max_age = 24,
    };
    pl_iclr(&world->cache);
}

RA_D static RA_INLINE void ra_rbind(
        RaCudaRigidWorld* world, int index,
        RaRigidBody body, RaConvexShape shape) {
    assert(index >= 0 && index < RA_CUDA_BODIES);
    world->bodies[index] = body;
    world->shapes[index] = shape;
    int next = index + 1;
    if (next > world->body_count) {
        world->body_count = next;
    }
    if (next > world->shape_count) {
        world->shape_count = next;
    }
}

RA_D static RA_INLINE void ra_setbox(
        RaCudaRigidWorld* world, int index, RaPose pose, RaVec3 half_extents,
        float mass, RaVec3 linear_velocity, RaVec3 angular_velocity) {
    RaInertia3 inertia = {0, 0, 0, 0, 0, 0};
    if (mass > 0.0f) {
        float x2 = half_extents.x * half_extents.x;
        float y2 = half_extents.y * half_extents.y;
        float z2 = half_extents.z * half_extents.z;
        inertia = (RaInertia3){
            mass * (y2 + z2) / 3.0f,
            mass * (x2 + z2) / 3.0f,
            mass * (x2 + y2) / 3.0f,
            0, 0, 0,
        };
    }
    ra_rbind(world, index,
        (RaRigidBody){pose, linear_velocity, angular_velocity, mass, inertia},
        (RaConvexShape){RA_CONVEX_BOX, pose, half_extents});
}

RA_D static RA_INLINE void ra_setsph(
        RaCudaRigidWorld* world, int index, RaPose pose, float radius,
        float mass, RaVec3 linear_velocity, RaVec3 angular_velocity) {
    RaInertia3 inertia = {0, 0, 0, 0, 0, 0};
    if (mass > 0.0f) {
        float diagonal = 0.4f * mass * radius * radius;
        inertia = (RaInertia3){diagonal, diagonal, diagonal, 0, 0, 0};
    }
    ra_rbind(world, index,
        (RaRigidBody){pose, linear_velocity, angular_velocity, mass, inertia},
        (RaConvexShape){RA_CONVEX_SPHERE, pose, ra_v3(radius, radius, radius)});
}

RA_D static RA_INLINE int ra_pair(
        RaCudaRigidWorld* world, int body_a, int body_b, float margin,
        float static_friction, float dynamic_friction, float restitution) {
    assert(body_a >= 0 && body_b >= 0 && body_a != body_b);
    assert(body_a < world->shape_count && body_b < world->shape_count);
    PlSatManifold sat;
    int candidate_count = pl_smans(
        &world->shapes[body_a], &world->shapes[body_b], margin, &sat);
    if (candidate_count <= 0) {
        return 0;
    }
    if (!ra_manok(world)) {
        return 0;
    }
    candidate_count = ra_min(candidate_count, PL_SAT_MAX_MANIFOLD_POINTS);
    PlImpulseCandidate candidates[PL_SAT_MAX_MANIFOLD_POINTS];
    for (int index = 0; index < candidate_count; ++index) {
        memset(&candidates[index], 0, sizeof(candidates[index]));
        candidates[index].contact = sat.point[index];
        candidates[index].feature = sat.point_feature[index];
    }
    PlImpulseManifold* manifold =
        &world->manifolds[world->manifold_count];
    int made_count = pl_iman(body_a, body_b, candidates,
        candidate_count, margin, static_friction, dynamic_friction,
        restitution, manifold);
    world->manifold_count += made_count > 0;
    return made_count > 0;
}

RA_D static RA_INLINE RaConvexContact ra_rimq(
        RaVec3 ball_position, float margin) {
    RaConvexContact contact;
    memset(&contact, 0, sizeof(contact));
    RaVec3 hoop = ra_hoop();
    float dx = ball_position.x - hoop.x;
    float dz = ball_position.z - hoop.z;
    float radial = sqrtf(dx*dx + dz*dz);
    float inverse_radial = radial > 1.0e-10f ? 1.0f / radial : 0.0f;
    RaVec3 centerline = ra_v3(
        hoop.x + RA_RIM_MAJOR_RADIUS
            * (radial > 1.0e-10f ? dx*inverse_radial : 1.0f),
        hoop.y,
        hoop.z + RA_RIM_MAJOR_RADIUS
            * (radial > 1.0e-10f ? dz*inverse_radial : 0.0f));
    RaVec3 delta = ra_sub(ball_position, centerline);
    float distance = ra_length(delta);
    RaVec3 normal = distance > 1.0e-10f
        ? ra_scale(delta, 1.0f / distance) : ra_v3(0, 1, 0);
    contact.hit = distance - RA_BALL_RADIUS - RA_RIM_TUBE_RADIUS <= margin;
    contact.iterations = 1;
    contact.separation = distance - RA_BALL_RADIUS - RA_RIM_TUBE_RADIUS;
    contact.normal = normal;
    contact.point_a = ra_sub(
        ball_position, ra_scale(normal, RA_BALL_RADIUS));
    contact.point_b = ra_add(
        centerline, ra_scale(normal, RA_RIM_TUBE_RADIUS));
    return contact;
}

RA_D static RA_INLINE RaConvexSweep ra_boxccd(
        const RaConvexShape* initial_a, const RaConvexShape* initial_b,
        RaVec3 linear_a, RaVec3 angular_a, RaVec3 linear_b,
        RaVec3 angular_b, float maximum_time, float target_margin) {
    RaConvexSweep sweep;
    memset(&sweep, 0, sizeof(sweep));
    sweep.toi = maximum_time;
    float time = 0.0f;
    float angular_bound = initial_a->type == RA_CONVEX_SPHERE ? 0.0f
        : ra_length(angular_a) * ra_brad(initial_a);
    angular_bound += initial_b->type == RA_CONVEX_SPHERE ? 0.0f
        : ra_length(angular_b) * ra_brad(initial_b);
    for (int iteration = 0; iteration < 12; ++iteration) {
        sweep.iterations = iteration + 1;
        RaConvexShape a = *initial_a;
        RaConvexShape b = *initial_b;
        a.pose.position = ra_add(a.pose.position, ra_scale(linear_a, time));
        b.pose.position = ra_add(b.pose.position, ra_scale(linear_b, time));
        a.pose.rotation = ra_qint(
            a.pose.rotation, angular_a, time);
        b.pose.rotation = ra_qint(
            b.pose.rotation, angular_b, time);
        RaConvexContact contact =
            pl_sq(&a, &b, target_margin).contact;
        sweep.contact = contact;
        if (contact.hit) {
            sweep.hit = 1;
            sweep.toi = time;
            return sweep;
        }
        float closing_speed = -ra_dot(
            ra_sub(linear_a, linear_b), contact.normal) + angular_bound;
        if (closing_speed <= 1.0e-8f) {
            return sweep;
        }
        float advance = (contact.separation - target_margin)
            / closing_speed;
        if (advance <= 1.0e-7f) {
            sweep.hit = 1;
            sweep.toi = time;
            return sweep;
        }
        time += advance;
        if (time > maximum_time) {
            return sweep;
        }
    }
    return sweep;
}

RA_D static RA_INLINE void ra_bodies(
        RaCudaProductionWorld* world) {
    RaState* state = &world->state;
    RaCudaRigidWorld* rigid = &world->rigid;
    unsigned int topology = ra_topo(state);
    if (rigid->topology != topology) {
        pl_iclr(&rigid->cache);
        rigid->topology = topology;
    }
    rigid->body_count = 0;
    rigid->shape_count = 0;
    rigid->manifold_count = 0;
    rigid->compound_pad_component_mask[0] = 0;
    rigid->compound_pad_component_mask[1] = 0;
    const RaPose* links = world->staged.links;
    const RaVec3* origins = world->staged.origins;
    const RaVec3* axes = world->staged.axes;
    const RaPose cube_pose = {state->cube_position, state->cube_rotation};
    const float cube_mass = state->stack_mode
        ? RA_STACK_CUBE_MASS
        : (state->basketball_mode ? RA_BALL_MASS : RA_PICK_CUBE_MASS);
    if (state->basketball_mode) {
        ra_setsph(rigid, RA_CUDA_BODY_CUBE, cube_pose,
            RA_BALL_RADIUS, cube_mass, state->cube_velocity,
            state->cube_angular_velocity);
    } else {
        ra_setbox(rigid, RA_CUDA_BODY_CUBE, cube_pose,
            ra_v3(RA_CUBE_HALF, RA_CUBE_HALF, RA_CUBE_HALF), cube_mass,
            state->cube_velocity, state->cube_angular_velocity);
    }
    const RaPose base_pose = {
        state->base_cube_position, state->base_cube_rotation};
    ra_setbox(rigid, RA_CUDA_BODY_BASE, base_pose,
        ra_v3(RA_CUBE_HALF, RA_CUBE_HALF, RA_CUBE_HALF),
        state->stack_mode ? RA_STACK_CUBE_MASS : 0.0f,
        state->base_cube_velocity, state->base_cube_angular_velocity);
    const RaPose table_pose = {
        ra_v3(RA_TABLE_CENTER_X, RA_TABLE_TOP - 0.5f*RA_TABLE_THICKNESS, 0),
        ra_quat(0, 0, 0, 1)};
    ra_setbox(rigid, RA_CUDA_BODY_TABLE, table_pose,
        ra_v3(0.5f*RA_TABLE_SIZE_X, 0.5f*RA_TABLE_THICKNESS,
            0.5f*RA_TABLE_SIZE_Z), 0.0f,
        ra_v3(0, 0, 0), ra_v3(0, 0, 0));

    RaGripperCollisionFrame frame = ra_gripf(
        links, state->end_effector);
    const RaVec3 hand_angular = ra_angvel(
        state->qd, axes, RA_DOF - 1);
    const int shell_source[RA_CUDA_SHELL_BOXES] = {0, 1, 2, 3, 5};
    for (int item = 0; item < RA_CUDA_SHELL_BOXES; ++item) {
        RaCollisionBox box = ra_gripb(
            &frame, shell_source[item]);
        RaVec3 velocity = ra_ptvel(
            state->qd, origins, axes, RA_DOF - 1, box.pose.position);
        ra_setbox(rigid, RA_CUDA_BODY_SHELL_START + item,
            box.pose, box.half_extent, 0.0f, velocity, hand_angular);
    }
    for (int item = 0; item < RA_DOF; ++item) {
        RaCollisionBox box = ra_linkb(links, item);
        RaVec3 velocity = ra_ptvel(
            state->qd, origins, axes, item, box.pose.position);
        RaVec3 angular = ra_angvel(
            state->qd, axes, item);
        ra_setbox(rigid, RA_CUDA_BODY_LINK_START + item,
            box.pose, box.half_extent, 0.0f, velocity, angular);
    }
    const RaVec3 hand_axis = ra_rotate(frame.hand.rotation, ra_v3(0, 1, 0));
    for (int pad = 0; pad < RA_PAD_BOXES; ++pad) {
        for (int side = 0; side < 2; ++side) {
            RaPose finger = side == 0
                ? frame.left_finger : frame.right_finger;
            RaConvexShape shape = ra_padsh(finger, pad);
            RaVec3 velocity = ra_ptvel(
                state->qd, origins, axes, RA_DOF - 1, shape.pose.position);
            float jaw = (side == 0 ? 0.5f : -0.5f) * state->gripper_velocity;
            velocity = ra_add(velocity, ra_scale(hand_axis, jaw));
            int body = (side == 0
                ? RA_CUDA_BODY_PAD_LEFT_START
                : RA_CUDA_BODY_PAD_RIGHT_START) + pad;
            ra_setbox(rigid, body, shape.pose, shape.half_extents,
                0.0f, velocity, hand_angular);
        }
    }
    if (state->basketball_mode) {
        const RaPose rim_pose = {
            ra_hoop(), ra_quat(0, 0, 0, 1)};
        ra_setsph(rigid, RA_CUDA_BODY_RIM, rim_pose,
            RA_RIM_MAJOR_RADIUS + RA_RIM_TUBE_RADIUS, 0.0f,
            ra_v3(0, 0, 0), ra_v3(0, 0, 0));
        const RaPose backboard_pose = {
            ra_v3(RA_HOOP_CENTER_X, RA_BACKBOARD_CENTER_Y,
                RA_BACKBOARD_CENTER_Z),
            ra_quat(0, 0, 0, 1)};
        ra_setbox(rigid, RA_CUDA_BODY_BACKBOARD,
            backboard_pose, ra_v3(RA_BACKBOARD_HALF_X,
                RA_BACKBOARD_HALF_Y, RA_BACKBOARD_HALF_Z), 0.0f,
            ra_v3(0, 0, 0), ra_v3(0, 0, 0));
    }
}

RA_D static RA_INLINE void ra_react(
        RaCudaProductionWorld* world,
        const float mass_factor[RA_DOF][RA_DOF]) {
    RaState* state = &world->state;
    RaCudaRigidWorld* rigid = &world->rigid;
    const RaPose* links = world->staged.links;
    const RaVec3* origins = world->staged.origins;
    const RaVec3* axes = world->staged.axes;
    const int manifold_count = ra_min(ra_max(rigid->manifold_count, 0),
        PL_IMPULSE_MAX_MANIFOLDS);
    int robot_contact = 0;
    for (int index = 0; index < manifold_count; ++index) {
        int body = rigid->manifolds[index].body_b;
        robot_contact |= body >= RA_CUDA_BODY_SHELL_START
            && body < RA_CUDA_ROBOT_BODY_END;
    }
    if (!robot_contact) {
        return;
    }
    RaGripperCollisionFrame frame = ra_gripf(
        links, state->end_effector);
    RaVec3 hand_axis = ra_rotate(frame.hand.rotation, ra_v3(0, 1, 0));
    for (int manifold_index = 0; manifold_index < manifold_count;
            ++manifold_index) {
        PlImpulseManifold* manifold = &rigid->manifolds[manifold_index];
        int body = manifold->body_b;
        int robot_body = body >= RA_CUDA_BODY_SHELL_START
            && body < RA_CUDA_ROBOT_BODY_END;
        int last_joint = body >= RA_CUDA_BODY_LINK_START
                && body < RA_CUDA_BODY_PAD_LEFT_START
            ? body - RA_CUDA_BODY_LINK_START : RA_DOF - 1;
        int side = body >= RA_CUDA_BODY_PAD_LEFT_START
            && body < RA_CUDA_BODY_PAD_RIGHT_START ? 0
            : (body >= RA_CUDA_BODY_PAD_RIGHT_START
                && body < RA_CUDA_ROBOT_BODY_END ? 1 : -1);
        for (int point_index = 0; point_index < manifold->point_count;
                ++point_index) {
            PlImpulsePoint* point = &manifold->points[point_index];
            memset(&point->reaction, 0, sizeof(point->reaction));
            if (!robot_body) {
                continue;
            }
            PlImpulseReaction* reaction = &point->reaction;
            RaVec3 directions[3] = {
                manifold->normal, manifold->tangent_1, manifold->tangent_2};
            for (int direction_index = 0; direction_index < 3;
                    ++direction_index) {
                float inverse_mass = 0.0f;
                ra_jacfk(
                    mass_factor, origins, axes, last_joint, point->point_b,
                    directions[direction_index], ra_v3(0, 0, 0),
                    reaction->robot_jacobian[direction_index],
                    reaction->robot_response[direction_index],
                    &inverse_mass);
                reaction->inverse_mass[direction_index] = inverse_mass;
            }
            if (side >= 0) {
                RaVec3 outward = ra_scale(
                    hand_axis, side == 0 ? 1.0f : -1.0f);
                for (int direction_index = 0; direction_index < 3;
                        ++direction_index) {
                    float jaw_jacobian = 0.5f * ra_dot(outward,
                        directions[direction_index]);
                    reaction->jaw_jacobian[direction_index] = jaw_jacobian;
                    reaction->inverse_mass[direction_index] +=
                        jaw_jacobian * jaw_jacobian
                        / RA_GRIPPER_EFFECTIVE_MASS;
                    reaction->jaw_velocity_response[direction_index]
                        = -jaw_jacobian / RA_GRIPPER_EFFECTIVE_MASS;
                }
            }
            reaction->active = 1;
            if (side >= 0 && point->patch_group != 0
                    && point->patch.area > 0.0f) {
                int group_points = 0;
                for (int member = 0; member < manifold->point_count;
                        ++member) {
                    if (manifold->points[member].patch_group
                            == point->patch_group) group_points++;
                }
                float area = point->patch.area
                    / (float)ra_max(group_points, 1);
                float stiffness = RA_PAD_ELASTIC_MODULUS * area
                    / ra_max(RA_PAD_LAYER_THICKNESS, 1.0e-8f);
                float inverse_effective_mass =
                    pl_imass(
                        &rigid->bodies[manifold->body_a], point->point_a,
                        &rigid->bodies[manifold->body_b], point->point_b,
                        manifold->normal, reaction, 0);
                float effective_mass = 1.0f / ra_max(
                    inverse_effective_mass, PL_IMPULSE_EPSILON);
                float damping = 2.0f * RA_PAD_DAMPING_RATIO
                    * sqrtf(ra_max(stiffness * effective_mass, 0.0f));
                float spring_damping = damping + RA_PHYSICS_DT * stiffness;
                point->normal_cfm = 1.0f / ra_max(
                    RA_PHYSICS_DT * spring_damping,
                    PL_IMPULSE_EPSILON);
                point->normal_erp = stiffness / ra_max(spring_damping,
                    PL_IMPULSE_EPSILON);
            }
        }
        memset(&manifold->angular_reaction, 0,
            sizeof(manifold->angular_reaction));
        if (side >= 0 && manifold->point_count > 0
                && manifold->torsional_radius > 0.0f) {
            PlImpulsePoint* point = &manifold->points[0];
            ra_jacfk(
                mass_factor, origins, axes, RA_DOF - 1, point->point_b,
                ra_v3(0, 0, 0), manifold->normal,
                manifold->angular_reaction.robot_jacobian,
                manifold->angular_reaction.robot_response,
                &manifold->angular_reaction.inverse_mass);
            manifold->angular_reaction.active = 1;
        }
    }
}

RA_D static RA_INLINE void ra_patch(
        const RaVec3* polygon, int count, RaVec3 pad_plane_origin,
        RaVec3 pad_normal, RaVec3 tangent_1, RaVec3 tangent_2,
        PlImpulsePatch* patch) {
    memset(patch, 0, sizeof(*patch));
    if (count <= 0) {
        return;
    }
    float x[PL_SAT_MAX_CLIP_VERTICES];
    float z[PL_SAT_MAX_CLIP_VERTICES];
    int vertex_count = ra_min(count, PL_SAT_MAX_CLIP_VERTICES);
    for (int index = 0; index < vertex_count; ++index) {
        RaVec3 projected = polygon[index];
        float signed_distance = ra_dot(
            ra_sub(projected, pad_plane_origin), pad_normal);
        projected = ra_sub(projected,
            ra_scale(pad_normal, signed_distance));
        RaVec3 offset = ra_sub(projected, pad_plane_origin);
        x[index] = ra_dot(offset, tangent_1);
        z[index] = ra_dot(offset, tangent_2);
    }
    if (vertex_count < 3) {
        return;
    }

    float cross_sum = 0.0f;
    float centroid_x_sum = 0.0f;
    float centroid_z_sum = 0.0f;
    float second_11_sum = 0.0f;
    float second_22_sum = 0.0f;
    float second_12_sum = 0.0f;
    for (int index = 0; index < vertex_count; ++index) {
        int next = (index + 1) % vertex_count;
        float cross = x[index] * z[next] - x[next] * z[index];
        cross_sum += cross;
        centroid_x_sum += (x[index] + x[next]) * cross;
        centroid_z_sum += (z[index] + z[next]) * cross;
        second_11_sum += (x[index] * x[index]
            + x[index] * x[next] + x[next] * x[next]) * cross;
        second_22_sum += (z[index] * z[index]
            + z[index] * z[next] + z[next] * z[next]) * cross;
        second_12_sum += (2.0f * x[index] * z[index]
            + x[index] * z[next] + x[next] * z[index]
            + 2.0f * x[next] * z[next]) * cross;
    }
    float signed_area = 0.5f * cross_sum;
    float area = fabsf(signed_area);
    if (area <= 1.0e-12f) {
        return;
    }
    float centroid_x = centroid_x_sum / (6.0f * signed_area);
    float centroid_z = centroid_z_sum / (6.0f * signed_area);
    float orientation = signed_area < 0.0f ? -1.0f : 1.0f;
    float raw_second_11 = orientation * second_11_sum / 12.0f;
    float raw_second_22 = orientation * second_22_sum / 12.0f;
    float raw_second_12 = orientation * second_12_sum / 24.0f;
    patch->area = area;
    patch->centroid = ra_add(pad_plane_origin,
        ra_add(ra_scale(tangent_1, centroid_x),
            ra_scale(tangent_2, centroid_z)));
    patch->second_11 = ra_max(
        raw_second_11 - area * centroid_x * centroid_x, 0.0f);
    patch->second_22 = ra_max(
        raw_second_22 - area * centroid_z * centroid_z, 0.0f);
    patch->second_12 = raw_second_12 - area * centroid_x * centroid_z;
}

typedef struct RaCudaClipVertex {
    RaVec3 point;
    uint32_t feature;
} RaCudaClipVertex;

RA_D static RA_INLINE int ra_bpad(
        RaCudaProductionWorld* production, int side) {
    RaCudaRigidWorld* world = &production->rigid;
    int pad_start = side == 0 ? RA_CUDA_BODY_PAD_LEFT_START
        : RA_CUDA_BODY_PAD_RIGHT_START;
    RaVec3 inward = ra_scale(ra_rotate(
        world->shapes[pad_start].pose.rotation, ra_v3(0, 1, 0)), -1.0f);
    PlImpulseCandidate best;
    memset(&best, 0, sizeof(best));
    float best_separation = 3.402823466e+38f;
    int best_pad = -1;
    for (int pad = 0; pad < RA_PAD_BOXES; ++pad) {
        int body = pad_start + pad;
        PlSatQuery query = pl_sq(
            &world->shapes[RA_CUDA_BODY_CUBE], &world->shapes[body],
            RA_CONTACT_MARGIN);
        if (ra_dot(query.contact.normal, inward) < 0.45f) {
            continue;
        }
        const RaRigidBody* ball = &world->bodies[RA_CUDA_BODY_CUBE];
        const RaRigidBody* pad_body = &world->bodies[body];
        RaVec3 ball_velocity = ra_add(ball->linear_velocity,
            ra_cross(ball->angular_velocity,
                ra_sub(query.contact.point_a, ball->pose.position)));
        RaVec3 pad_velocity = ra_add(pad_body->linear_velocity,
            ra_cross(pad_body->angular_velocity,
                ra_sub(query.contact.point_b, pad_body->pose.position)));
        float projected = query.contact.separation
            + ra_dot(ra_sub(ball_velocity, pad_velocity),
                query.contact.normal) * RA_PHYSICS_DT;
        if (!query.contact.hit && projected > RA_CONTACT_MARGIN) {
            continue;
        }
        if (query.contact.separation < best_separation) {
            best_separation = query.contact.separation;
            best.contact = query.contact;
            best.contact.hit = 1;
            best.feature = 0x73000000u
                | ((uint32_t)side << 8) | (uint32_t)pad;
            best_pad = pad;
        }
    }
    if (best_pad < 0) {
        return 0;
    }
    if (!ra_manok(world)) {
        return 0;
    }
    int body = pad_start + best_pad;
    PlImpulseManifold* manifold = &world->manifolds[world->manifold_count];
    int count = pl_iman(RA_CUDA_BODY_CUBE, body,
        &best, 1, RA_CONTACT_MARGIN, RA_FINGER_FRICTION,
        RA_FINGER_FRICTION, 0.0f, manifold);
    if (count <= 0) {
        return 0;
    }
    world->compound_pad_component_mask[side] = 1u << best_pad;
    world->manifold_count += 1;
    return 1;
}

RA_D static RA_INLINE int ra_clipp(
        const RaCudaClipVertex* input, int count,
        RaCudaClipVertex* output, RaVec3 normal, float offset,
        uint32_t plane_index) {
    if (count <= 0) {
        return 0;
    }
    int output_count = 0;
    RaCudaClipVertex previous = input[count - 1];
    float previous_distance = ra_dot(previous.point, normal) - offset;
    int previous_inside = previous_distance <= 0.0f;
    for (int index = 0; index < count; ++index) {
        RaCudaClipVertex current = input[index];
        float current_distance = ra_dot(current.point, normal) - offset;
        int current_inside = current_distance <= 0.0f;
        if (current_inside != previous_inside
                && output_count < PL_SAT_MAX_CLIP_VERTICES) {
            float fraction = previous_distance
                / (previous_distance - current_distance);
            RaCudaClipVertex intersection;
            intersection.point = ra_lerp(previous.point, current.point,
                fraction);
            uint32_t lower = previous.feature < current.feature
                ? previous.feature : current.feature;
            uint32_t upper = previous.feature < current.feature
                ? current.feature : previous.feature;
            intersection.feature = 0x80000000u
                | ((plane_index & 0xffu) << 16)
                | ((lower & 0xffu) << 8) | (upper & 0xffu);
            output[output_count++] = intersection;
        }
        if (current_inside
                && output_count < PL_SAT_MAX_CLIP_VERTICES) {
            output[output_count++] = current;
        }
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }
    return output_count;
}

RA_D static RA_INLINE int ra_padc(
        RaCudaProductionWorld* production, int side, int cube_body) {
    RaCudaRigidWorld* world = &production->rigid;
    int pad_start = side == 0 ? RA_CUDA_BODY_PAD_LEFT_START
        : RA_CUDA_BODY_PAD_RIGHT_START;
    float margin = RA_CONTACT_MARGIN;
    float static_friction = RA_FINGER_FRICTION;
    float dynamic_friction = RA_FINGER_FRICTION;
    float restitution = 0.0f;
    PlImpulseCandidate* candidates = world->compound_candidate_scratch;
    int candidate_count = 0;
    RaVec3 inward = ra_scale(ra_rotate(
        world->shapes[pad_start].pose.rotation, ra_v3(0, 1, 0)), -1.0f);
    const RaConvexShape* cube_shape = &world->shapes[cube_body];
    const RaRigidBody* cube_body_state = &world->bodies[cube_body];
    PlSatObb cube_obb = pl_sobb(cube_shape);
    RaVec3 cube_axes[3];
    ra_caxes(cube_shape->pose.rotation, cube_axes);
    const RaVec3 cube_half = cube_shape->half_extents;
    const RaVec3 frame_origin = world->shapes[pad_start].pose.position;
    RaVec3 pad_axes[3];
    ra_caxes(world->shapes[pad_start].pose.rotation, pad_axes);
    float frame_axis_x = ra_dot(frame_origin, pad_axes[0]);
    float frame_axis_z = ra_dot(frame_origin, pad_axes[2]);
    float rect_min_x[RA_PAD_BOXES];
    float rect_max_x[RA_PAD_BOXES];
    float rect_min_z[RA_PAD_BOXES];
    float rect_max_z[RA_PAD_BOXES];
    float support_plane[RA_PAD_BOXES];
    RaVec3 inner_surface[RA_PAD_BOXES];
    float temporal_plane[RA_PAD_BOXES];
    float pad_normal_velocity[RA_PAD_BOXES];
    float pad_angular_bound[RA_PAD_BOXES];
    int pad_active[RA_PAD_BOXES];
    float x_bounds[RA_PAD_BOXES * 2];
    float z_bounds[RA_PAD_BOXES * 2];
    int x_bound_count = 0;
    int z_bound_count = 0;
    for (int pad = 0; pad < RA_PAD_BOXES; ++pad) {
        const RaConvexShape* pad_shape = &world->shapes[pad_start + pad];
        RaVec3 rectangle_delta = ra_sub(pad_shape->pose.position,
            frame_origin);
        float center_x = ra_dot(rectangle_delta, pad_axes[0]);
        float center_z = ra_dot(rectangle_delta, pad_axes[2]);
        rect_min_x[pad] = center_x - pad_shape->half_extents.x;
        rect_max_x[pad] = center_x + pad_shape->half_extents.x;
        rect_min_z[pad] = center_z - pad_shape->half_extents.z;
        rect_max_z[pad] = center_z + pad_shape->half_extents.z;
        x_bounds[x_bound_count++] = rect_min_x[pad];
        x_bounds[x_bound_count++] = rect_max_x[pad];
        z_bounds[z_bound_count++] = rect_min_z[pad];
        z_bounds[z_bound_count++] = rect_max_z[pad];
        inner_surface[pad] = ra_add(pad_shape->pose.position,
            ra_scale(inward, pad_shape->half_extents.y));
        support_plane[pad] = ra_dot(inner_surface[pad], inward);
        pad_active[pad] = 0;
        temporal_plane[pad] = support_plane[pad];
        pad_normal_velocity[pad] = 0.0f;
        pad_angular_bound[pad] = 0.0f;
    }
    for (int index = 1; index < x_bound_count; ++index) {
        float value = x_bounds[index];
        int cursor = index;
        while (cursor > 0 && value < x_bounds[cursor - 1]) {
            x_bounds[cursor] = x_bounds[cursor - 1];
            --cursor;
        }
        x_bounds[cursor] = value;
    }
    for (int index = 1; index < z_bound_count; ++index) {
        float value = z_bounds[index];
        int cursor = index;
        while (cursor > 0 && value < z_bounds[cursor - 1]) {
            z_bounds[cursor] = z_bounds[cursor - 1];
            --cursor;
        }
        z_bounds[cursor] = value;
    }
    int unique_x = 0;
    for (int index = 0; index < x_bound_count; ++index) {
        if (unique_x == 0 || fabsf(x_bounds[index]
                - x_bounds[unique_x - 1]) > RA_PAD_CSG_BOUNDARY_EPSILON) {
            x_bounds[unique_x++] = x_bounds[index];
        }
    }
    int unique_z = 0;
    for (int index = 0; index < z_bound_count; ++index) {
        if (unique_z == 0 || fabsf(z_bounds[index]
                - z_bounds[unique_z - 1]) > RA_PAD_CSG_BOUNDARY_EPSILON) {
            z_bounds[unique_z++] = z_bounds[index];
        }
    }
    for (int pad = 0; pad < RA_PAD_BOXES; ++pad) {
        int body = pad_start + pad;
            const RaConvexShape* pad_shape = &world->shapes[body];
        PlSatQuery sat_query = pl_sq(
            cube_shape, pad_shape, margin);

        RaVec3 delta = ra_sub(cube_shape->pose.position,
            pad_shape->pose.position);
        if (ra_dot(delta, inward) < 0.0f) {
            continue;
        }
        RaVec3 cube_surface = pl_ssup(&cube_obb,
            ra_scale(inward, -1.0f));
        float face_separation = ra_dot(
            ra_sub(cube_surface, inner_surface[pad]), inward);
        const RaRigidBody* pad_body_state = &world->bodies[body];
        float angular_bound = ra_length(cube_body_state->angular_velocity)
                * ra_brad(cube_shape)
            + ra_length(pad_body_state->angular_velocity)
                * ra_brad(pad_shape);
        RaVec3 sat_normal = pl_inrm(
            sat_query.contact.normal, ra_sub(
                sat_query.contact.point_a, sat_query.contact.point_b));
        RaVec3 sat_velocity_a = ra_add(cube_body_state->linear_velocity,
            ra_cross(cube_body_state->angular_velocity,
                ra_sub(sat_query.contact.point_a,
                    cube_body_state->pose.position)));
        RaVec3 sat_velocity_b = ra_add(pad_body_state->linear_velocity,
            ra_cross(pad_body_state->angular_velocity,
                ra_sub(sat_query.contact.point_b,
                    pad_body_state->pose.position)));
        float sat_projected_separation = sat_query.contact.separation
            + ra_dot(ra_sub(sat_velocity_a, sat_velocity_b), sat_normal)
                * RA_PHYSICS_DT
            - angular_bound * RA_PHYSICS_DT;
        if (!sat_query.contact.hit
                && sat_query.contact.separation > margin
                && sat_projected_separation > margin) continue;
        float cube_radius_x = ra_oboxr(
            cube_axes, cube_half, pad_axes[0]);
        float cube_radius_z = ra_oboxr(
            cube_axes, cube_half, pad_axes[2]);
        float local_x = ra_dot(delta, pad_axes[0]);
        float local_z = ra_dot(delta, pad_axes[2]);
        if (fabsf(local_x) > pad_shape->half_extents.x
                + cube_radius_x + margin
            || fabsf(local_z) > pad_shape->half_extents.z
                + cube_radius_z + margin) continue;

        RaVec3 cube_velocity = ra_add(cube_body_state->linear_velocity,
            ra_cross(cube_body_state->angular_velocity,
                ra_sub(cube_surface, cube_body_state->pose.position)));
        RaVec3 pad_velocity = ra_add(pad_body_state->linear_velocity,
            ra_cross(pad_body_state->angular_velocity,
                ra_sub(inner_surface[pad], pad_body_state->pose.position)));
        float normal_velocity = ra_dot(
            ra_sub(cube_velocity, pad_velocity), inward);
        float projected_separation = face_separation
            + (normal_velocity - angular_bound) * RA_PHYSICS_DT;
        if (face_separation > margin && projected_separation > margin) {
            continue;
        }
        pad_active[pad] = 1;
        pad_normal_velocity[pad] = normal_velocity;
        pad_angular_bound[pad] = angular_bound;
        temporal_plane[pad] = support_plane[pad] + margin
            - (normal_velocity - angular_bound) * RA_PHYSICS_DT;
    }
    RaVec3 patch_tangent_1, patch_tangent_2;
    pl_itan(inward, &patch_tangent_1,
        &patch_tangent_2);
    float patch_area_acc[RA_PAD_BOXES] = {0.0f};
    float patch_first_1[RA_PAD_BOXES] = {0.0f};
    float patch_first_2[RA_PAD_BOXES] = {0.0f};
    float patch_raw_11[RA_PAD_BOXES] = {0.0f};
    float patch_raw_22[RA_PAD_BOXES] = {0.0f};
    float patch_raw_12[RA_PAD_BOXES] = {0.0f};
    int candidate_overflow = 0;

    int incident_axis = pl_saxis(&cube_obb,
        ra_scale(inward, -1.0f));
    float incident_sign = ra_dot(cube_obb.axis[incident_axis],
        ra_scale(inward, -1.0f)) < 0.0f ? -1.0f : 1.0f;
    RaVec3 incident_face[4];
    pl_sface(&cube_obb, incident_axis, incident_sign,
        incident_face);
    float face_x_offset = frame_axis_x;
    float face_z_offset = frame_axis_z;
    for (int x_cell = 0; x_cell + 1 < unique_x; ++x_cell) {
        float cell_min_x = x_bounds[x_cell];
        float cell_max_x = x_bounds[x_cell + 1];
        if (cell_max_x - cell_min_x <= RA_PAD_CSG_BOUNDARY_EPSILON) {
            continue;
        }
        for (int z_cell = 0; z_cell + 1 < unique_z; ++z_cell) {
            float cell_min_z = z_bounds[z_cell];
            float cell_max_z = z_bounds[z_cell + 1];
            if (cell_max_z - cell_min_z
                    <= RA_PAD_CSG_BOUNDARY_EPSILON) continue;
            float cell_center_x = 0.5f * (cell_min_x + cell_max_x);
            float cell_center_z = 0.5f * (cell_min_z + cell_max_z);
            int owner = -1;
            float best_support = 0.0f;
            for (int pad = 0; pad < RA_PAD_BOXES; ++pad) {
                if (cell_center_x < rect_min_x[pad]
                        - RA_PAD_CSG_BOUNDARY_EPSILON
                    || cell_center_x > rect_max_x[pad]
                        + RA_PAD_CSG_BOUNDARY_EPSILON
                    || cell_center_z < rect_min_z[pad]
                        - RA_PAD_CSG_BOUNDARY_EPSILON
                    || cell_center_z > rect_max_z[pad]
                        + RA_PAD_CSG_BOUNDARY_EPSILON) continue;
                if (owner < 0
                        || support_plane[pad] > best_support
                            + RA_PAD_SUPPORT_PLANE_TOLERANCE
                        || (fabsf(support_plane[pad] - best_support)
                                <= RA_PAD_SUPPORT_PLANE_TOLERANCE
                            && pad < owner)) {
                    owner = pad;
                    best_support = support_plane[pad];
                }
            }
            if (owner < 0 || !pad_active[owner]) {
                continue;
            }
            RaCudaClipVertex clipped[PL_SAT_MAX_CLIP_VERTICES];
            RaCudaClipVertex scratch[PL_SAT_MAX_CLIP_VERTICES];
            for (int vertex = 0; vertex < 4; ++vertex) {
                clipped[vertex].point = incident_face[vertex];
                clipped[vertex].feature = (uint32_t)vertex;
            }
            int count = 4;
            int cell_id = x_cell * RA_PAD_BOXES + z_cell;
            int plane_base = 5;
            count = ra_clipp(clipped, count, scratch,
                pad_axes[0], face_x_offset + cell_max_x, plane_base);
            count = ra_clipp(scratch, count, clipped,
                ra_scale(pad_axes[0], -1.0f),
                -face_x_offset - cell_min_x, plane_base + 1);
            count = ra_clipp(clipped, count, scratch,
                pad_axes[2], face_z_offset + cell_max_z, plane_base + 2);
            count = ra_clipp(scratch, count, clipped,
                ra_scale(pad_axes[2], -1.0f),
                -face_z_offset - cell_min_z, plane_base + 3);
            count = ra_clipp(clipped, count, scratch,
                inward, temporal_plane[owner], 10 + owner);
            for (int point_index = 0; point_index < count; ++point_index) {
                clipped[point_index] = scratch[point_index];
            }
            if (count <= 0) {
                continue;
            }
            RaVec3 polygon[PL_SAT_MAX_CLIP_VERTICES];
            for (int point_index = 0; point_index < count; ++point_index) {
                polygon[point_index] = clipped[point_index].point;
            }
            PlImpulsePatch cell_patch;
            ra_patch(polygon, count, inner_surface[owner],
                inward, patch_tangent_1, patch_tangent_2, &cell_patch);
            if (cell_patch.area > 1.0e-12f) {
                float centroid_1 = ra_dot(ra_sub(cell_patch.centroid,
                    frame_origin), patch_tangent_1);
                float centroid_2 = ra_dot(ra_sub(cell_patch.centroid,
                    frame_origin), patch_tangent_2);
                patch_area_acc[owner] += cell_patch.area;
                patch_first_1[owner] += cell_patch.area * centroid_1;
                patch_first_2[owner] += cell_patch.area * centroid_2;
                patch_raw_11[owner] += cell_patch.second_11
                    + cell_patch.area * centroid_1 * centroid_1;
                patch_raw_22[owner] += cell_patch.second_22
                    + cell_patch.area * centroid_2 * centroid_2;
                patch_raw_12[owner] += cell_patch.second_12
                    + cell_patch.area * centroid_1 * centroid_2;
            }
            for (int point_index = 0; point_index < count; ++point_index) {
                RaVec3 point_a = clipped[point_index].point;
                float point_x = ra_dot(ra_sub(point_a, frame_origin),
                    pad_axes[0]);
                float point_z = ra_dot(ra_sub(point_a, frame_origin),
                    pad_axes[2]);
                int point_owner = -1;
                float point_support = 0.0f;
                for (int pad = 0; pad < RA_PAD_BOXES; ++pad) {
                    if (point_x < rect_min_x[pad]
                            - RA_PAD_CSG_BOUNDARY_EPSILON
                        || point_x > rect_max_x[pad]
                            + RA_PAD_CSG_BOUNDARY_EPSILON
                        || point_z < rect_min_z[pad]
                            - RA_PAD_CSG_BOUNDARY_EPSILON
                        || point_z > rect_max_z[pad]
                            + RA_PAD_CSG_BOUNDARY_EPSILON) continue;
                    if (point_owner < 0
                            || support_plane[pad] > point_support
                                + RA_PAD_SUPPORT_PLANE_TOLERANCE
                            || (fabsf(support_plane[pad] - point_support)
                                    <= RA_PAD_SUPPORT_PLANE_TOLERANCE
                                && pad < point_owner)) {
                        point_owner = pad;
                        point_support = support_plane[pad];
                    }
                }
                if (point_owner != owner) {
                    continue;
                }
                float separation = ra_dot(ra_sub(point_a,
                    inner_surface[owner]), inward);
                float projected = separation
                    + (pad_normal_velocity[owner]
                        - pad_angular_bound[owner]) * RA_PHYSICS_DT;
                if (separation > margin + RA_PAD_CSG_BOUNDARY_EPSILON
                        && projected > margin
                            + RA_PAD_CSG_BOUNDARY_EPSILON) continue;
                if (candidate_count >= RA_CUDA_PAD_MAX_VISIBLE_CANDIDATES) {
                    candidate_overflow = 1;
                    continue;
                }
                RaVec3 point_b = ra_sub(point_a,
                    ra_scale(inward, separation));
                PlImpulseCandidate candidate;
                memset(&candidate, 0, sizeof(candidate));
                candidate.contact.hit = 1;
                candidate.contact.iterations = 15;
                candidate.contact.normal = inward;
                candidate.contact.point_a = point_a;
                candidate.contact.point_b = point_b;
                candidate.contact.separation = separation;
                uint32_t local_feature = (clipped[point_index].feature
                    & 0x0000ffffu)
                    | ((uint32_t)(cell_id & 0xffu) << 16);
                candidate.feature = 0xd0000000u
                    | ((uint32_t)(side & 1) << 27)
                    | ((uint32_t)(owner & 7) << 24)
                    | local_feature;
                candidate.patch_group = ((uint32_t)(side & 1) << 8)
                    | (uint32_t)(owner + 1);
                candidates[candidate_count++] = candidate;
            }
        }
    }
    if (candidate_overflow) {
        return 0;
    }
    PlImpulsePatch component_patch[RA_PAD_BOXES];
    for (int pad = 0; pad < RA_PAD_BOXES; ++pad) {
        memset(&component_patch[pad], 0, sizeof(component_patch[pad]));
        if (patch_area_acc[pad] <= 1.0e-12f) {
            continue;
        }
        float centroid_1 = patch_first_1[pad] / patch_area_acc[pad];
        float centroid_2 = patch_first_2[pad] / patch_area_acc[pad];
        component_patch[pad].area = patch_area_acc[pad];
        component_patch[pad].centroid = ra_add(frame_origin,
            ra_add(ra_scale(patch_tangent_1, centroid_1),
                ra_scale(patch_tangent_2, centroid_2)));
        component_patch[pad].second_11 = ra_max(patch_raw_11[pad]
            - patch_area_acc[pad] * centroid_1 * centroid_1, 0.0f);
        component_patch[pad].second_22 = ra_max(patch_raw_22[pad]
            - patch_area_acc[pad] * centroid_2 * centroid_2, 0.0f);
        component_patch[pad].second_12 = patch_raw_12[pad]
            - patch_area_acc[pad] * centroid_1 * centroid_2;
    }
    for (int index = 0; index < candidate_count; ++index) {
        int pad = (int)((candidates[index].feature >> 24) & 7u);
        candidates[index].patch = component_patch[pad];
    }
    if (candidate_count <= 0) {
        return 0;
    }
    if (!ra_manok(world)) {
        return 0;
    }
    int unique_count = 0;
    for (int index = 0; index < candidate_count; ++index) {
        RaVec3 point = pl_ipt(&candidates[index]);
        int duplicate = 0;
        for (int previous = 0; previous < unique_count; ++previous) {
            RaVec3 delta = ra_sub(point,
                pl_ipt(&candidates[previous]));
            if (ra_dot(delta, delta) <= 1.0e-12f) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            candidates[unique_count++] = candidates[index];
        }
    }
    candidate_count = unique_count;
    if (candidate_count <= 0) {
        return 0;
    }
    if (!ra_manok(world)) {
        return 0;
    }
    int selected[PL_IMPULSE_MAX_POINTS];
    int selected_count = 0;
    int deepest = 0;
    for (int index = 1; index < candidate_count; ++index) {
        if (candidates[index].contact.separation
                < candidates[deepest].contact.separation
            || (candidates[index].contact.separation
                    == candidates[deepest].contact.separation
                && candidates[index].feature < candidates[deepest].feature)) {
            deepest = index;
        }
    }
    selected[selected_count++] = deepest;
    while (selected_count < PL_IMPULSE_MAX_POINTS
            && selected_count < candidate_count) {
        int best = -1;
        float best_score = -1.0f;
        for (int index = 0; index < candidate_count; ++index) {
            int already = 0;
            for (int slot = 0; slot < selected_count; ++slot) {
                if (selected[slot] == index) {
                    already = 1;
                }
            }
            if (already) {
                continue;
            }
            RaVec3 point = pl_ipt(&candidates[index]);
            float score = 3.402823466e+38f;
            for (int slot = 0; slot < selected_count; ++slot) {
                RaVec3 other = pl_ipt(
                    &candidates[selected[slot]]);
                RaVec3 delta = ra_sub(point, other);
                score = ra_min(score, ra_dot(delta, delta));
            }
            if (best < 0 || score > best_score
                    || (score == best_score
                        && candidates[index].feature
                            < candidates[best].feature)) {
                best = index;
                best_score = score;
            }
        }
        if (best < 0) {
            break;
        }
        selected[selected_count++] = best;
    }
    PlImpulseCandidate reduced[PL_IMPULSE_MAX_POINTS];
    for (int index = 0; index < selected_count; ++index) {
        reduced[index] = candidates[selected[index]];
    }
    PlImpulseManifold* manifold = &world->manifolds[
        world->manifold_count];
    int made_count = pl_iman(cube_body, pad_start, reduced,
        selected_count, margin, static_friction, dynamic_friction,
        restitution, manifold);
    int selected_manifold_count = made_count;
    if (selected_manifold_count <= 0) {
        return 0;
    }
    uint32_t component_mask = 0;
    float patch_area = 0.0f;
    RaVec3 patch_centroid_sum = ra_v3(0, 0, 0);
    for (int point = 0; point < selected_manifold_count; ++point) {
        component_mask |= 1u << ((manifold->points[point].feature >> 24)
            & 7u);
        const PlImpulsePatch* patch = &manifold->points[point].patch;
        int first_group = 1;
        for (int previous = 0; previous < point; ++previous) {
            if (manifold->points[previous].patch_group
                    == manifold->points[point].patch_group) {
                first_group = 0;
                break;
            }
        }
        if (!first_group) {
            continue;
        }
        patch_area += patch->area;
        patch_centroid_sum = ra_add(patch_centroid_sum,
            ra_scale(patch->centroid, patch->area));
    }
    if (patch_area > 1.0e-12f) {
        manifold->patch_area = patch_area;
        manifold->patch_centroid = ra_scale(patch_centroid_sum,
            1.0f / patch_area);
        float second_11 = 0.0f;
        float second_22 = 0.0f;
        float second_12 = 0.0f;
        for (int point = 0; point < selected_manifold_count; ++point) {
            const PlImpulsePatch* patch = &manifold->points[point].patch;
            int first_group = 1;
            for (int previous = 0; previous < point; ++previous) {
                if (manifold->points[previous].patch_group
                        == manifold->points[point].patch_group) {
                    first_group = 0;
                    break;
                }
            }
            if (!first_group) {
                continue;
            }
            if (patch->area <= 0.0f) {
                continue;
            }
            RaVec3 offset = ra_sub(patch->centroid,
                manifold->patch_centroid);
            float offset_1 = ra_dot(offset, manifold->tangent_1);
            float offset_2 = ra_dot(offset, manifold->tangent_2);
            second_11 += patch->second_11
                + patch->area * offset_1 * offset_1;
            second_22 += patch->second_22
                + patch->area * offset_2 * offset_2;
            second_12 += patch->second_12
                + patch->area * offset_1 * offset_2;
        }
        manifold->patch_second_11 = ra_max(second_11, 0.0f);
        manifold->patch_second_22 = ra_max(second_22, 0.0f);
        manifold->patch_second_12 = second_12;
        manifold->patch_second_moment = manifold->patch_second_11
            + manifold->patch_second_22;
    } else {
        manifold->patch_area = 0.0f;
        manifold->patch_centroid = ra_v3(0, 0, 0);
        manifold->patch_second_11 = 0.0f;
        manifold->patch_second_22 = 0.0f;
        manifold->patch_second_12 = 0.0f;
        manifold->patch_second_moment = 0.0f;
    }
    float effective_radius = patch_area > 1.0e-12f
        ? sqrtf(ra_max(manifold->patch_second_moment / patch_area, 0.0f))
        : 0.0f;
    manifold->torsional_radius = effective_radius;
    world->compound_pad_component_mask[side] |= component_mask;
    world->manifold_count++;
    return 1;
}

RA_D static RA_INLINE void ra_advbox(
        RaCudaProductionWorld* world, int body_index, int other_index) {
    RaCudaRigidWorld* rigid = &world->rigid;
    RaRigidBody* body = &rigid->bodies[body_index];
    const RaRigidBody* other = &rigid->bodies[other_index];
    RaConvexSweep sweep = ra_boxccd(
        &rigid->shapes[body_index], &rigid->shapes[other_index],
        body->linear_velocity, body->angular_velocity,
        other->linear_velocity, other->angular_velocity,
        RA_PHYSICS_DT, 0.0f);
    float normal_speed = ra_dot(
        ra_sub(body->linear_velocity, other->linear_velocity),
        sweep.contact.normal);
    if (sweep.hit && normal_speed < 0.0f) {  // only approaching TOI=0 blocks
        body->pose.position = ra_add(body->pose.position,
            ra_scale(body->linear_velocity, sweep.toi));
        body->pose.rotation = ra_qint(
            body->pose.rotation, body->angular_velocity, sweep.toi);
    } else {
        body->pose.position = ra_add(body->pose.position,
            ra_scale(body->linear_velocity, RA_PHYSICS_DT));
        body->pose.rotation = ra_qint(
            body->pose.rotation, body->angular_velocity, RA_PHYSICS_DT);
    }
}

RA_D static RA_INLINE RaConvexSweep ra_rimccd(
        const RaRigidBody* ball, float maximum_time) {
    RaConvexSweep sweep;
    memset(&sweep, 0, sizeof(sweep));
    sweep.toi = maximum_time;
    float speed = ra_length(ball->linear_velocity);
    float time = 0.0f;
    for (int iteration = 0; iteration < 12; ++iteration) {
        sweep.iterations = iteration + 1;
        RaVec3 position = ra_add(ball->pose.position,
            ra_scale(ball->linear_velocity, time));
        sweep.contact = ra_rimq(position, 0.0f);
        if (sweep.contact.hit) {
            sweep.hit = 1;
            sweep.toi = time;
            return sweep;
        }
        if (speed <= 1.0e-8f) {
            return sweep;
        }
        float advance = sweep.contact.separation / speed;
        if (advance <= 1.0e-7f) {
            sweep.hit = 1;
            sweep.toi = time;
            return sweep;
        }
        time += advance;
        if (time > maximum_time) {
            return sweep;
        }
    }
    return sweep;
}

RA_D static RA_INLINE float ra_toi(
        RaConvexSweep sweep, RaVec3 velocity, float advance) {
    float normal_speed = ra_dot(velocity, sweep.contact.normal);
    if (sweep.hit && normal_speed < 0.0f && sweep.toi < advance) {
        return sweep.toi;
    }
    return advance;
}

RA_D static RA_INLINE void ra_advb(
        RaCudaProductionWorld* world) {
    RaCudaRigidWorld* rigid = &world->rigid;
    RaRigidBody* ball = &rigid->bodies[RA_CUDA_BODY_CUBE];
    RaVec3 velocity = ball->linear_velocity;
    float advance = RA_PHYSICS_DT;
    advance = ra_toi(ra_boxccd(
        &rigid->shapes[RA_CUDA_BODY_CUBE], &rigid->shapes[RA_CUDA_BODY_TABLE],
        velocity, ball->angular_velocity, ra_v3(0, 0, 0), ra_v3(0, 0, 0),
        RA_PHYSICS_DT, 0.0f), velocity, advance);
    advance = ra_toi(ra_boxccd(
        &rigid->shapes[RA_CUDA_BODY_CUBE],
        &rigid->shapes[RA_CUDA_BODY_BACKBOARD],
        velocity, ball->angular_velocity, ra_v3(0, 0, 0), ra_v3(0, 0, 0),
        RA_PHYSICS_DT, 0.0f), velocity, advance);
    advance = ra_toi(
        ra_rimccd(ball, RA_PHYSICS_DT), velocity, advance);
    ball->pose.position = ra_add(ball->pose.position,
        ra_scale(velocity, advance));
    ball->pose.rotation = ra_qint(
        ball->pose.rotation, ball->angular_velocity, advance);
}

RA_D static RA_INLINE void ra_copyb(
        RaCudaProductionWorld* world) {
    RaState* state = &world->state;
    const RaRigidBody* cube = &world->rigid.bodies[RA_CUDA_BODY_CUBE];
    state->cube_position = cube->pose.position;
    state->cube_rotation = cube->pose.rotation;
    state->cube_velocity = cube->linear_velocity;
    state->cube_angular_velocity = cube->angular_velocity;
    if (state->stack_mode) {
        const RaRigidBody* base = &world->rigid.bodies[RA_CUDA_BODY_BASE];
        state->base_cube_position = base->pose.position;
        state->base_cube_rotation = base->pose.rotation;
        state->base_cube_velocity = base->linear_velocity;
        state->base_cube_angular_velocity = base->angular_velocity;
    }
}

RA_D static RA_INLINE int ra_tbllo(
        RaPose pose, RaVec3 half_extent, float margin) {
    RaVec3 axis_x = ra_rotate(pose.rotation, ra_v3(1, 0, 0));
    RaVec3 axis_y = ra_rotate(pose.rotation, ra_v3(0, 1, 0));
    RaVec3 axis_z = ra_rotate(pose.rotation, ra_v3(0, 0, 1));
    float radius_x = fabsf(axis_x.x)*half_extent.x
        + fabsf(axis_y.x)*half_extent.y
        + fabsf(axis_z.x)*half_extent.z;
    float radius_y = fabsf(axis_x.y)*half_extent.x
        + fabsf(axis_y.y)*half_extent.y
        + fabsf(axis_z.y)*half_extent.z;
    float radius_z = fabsf(axis_x.z)*half_extent.x
        + fabsf(axis_y.z)*half_extent.y
        + fabsf(axis_z.z)*half_extent.z;
    float table_min_x = RA_TABLE_CENTER_X - 0.5f*RA_TABLE_SIZE_X;
    float table_max_x = RA_TABLE_CENTER_X + 0.5f*RA_TABLE_SIZE_X;
    float table_min_z = -0.5f*RA_TABLE_SIZE_Z;
    float table_max_z = 0.5f*RA_TABLE_SIZE_Z;
    return pose.position.x + radius_x >= table_min_x - margin
        && pose.position.x - radius_x <= table_max_x + margin
        && pose.position.z + radius_z >= table_min_z - margin
        && pose.position.z - radius_z <= table_max_z + margin
        && pose.position.y - radius_y < RA_TABLE_TOP + margin;
}

RA_D static RA_INLINE int ra_tblhit(
        RaPose pose, RaVec3 half_extent, float margin) {
    if (!ra_tbllo(pose, half_extent, margin)) {
        return 0;
    }
    RaConvexShape table = {};
    table.type = RA_CONVEX_BOX;
    table.pose.position = ra_v3(RA_TABLE_CENTER_X,
        RA_TABLE_TOP - 0.5f*RA_TABLE_THICKNESS, 0);
    table.pose.rotation = ra_quat(0, 0, 0, 1);
    table.half_extents = ra_v3(0.5f*RA_TABLE_SIZE_X,
        0.5f*RA_TABLE_THICKNESS, 0.5f*RA_TABLE_SIZE_Z);
    RaConvexShape shape = {};
    shape.type = RA_CONVEX_BOX;
    shape.pose = pose;
    shape.half_extents = half_extent;
    RaConvexContact contact = pl_sq(
        &table, &shape, margin).contact;
    return contact.hit && contact.separation < margin;
}

RA_D static RA_INLINE int ra_tblpen(
        const RaState* state) {
    RaPose links[RA_LINKS];
    RaVec3 end_effector;
    ra_fk(state->q, state->gripper_width, links, NULL, NULL, &end_effector);
    for (int link = 0; link < RA_DOF; ++link) {
        RaCollisionBox box = ra_linkb(links, link);
        if (ra_tblhit(
                box.pose, box.half_extent, RA_CONTACT_MARGIN)) {
            return 1;
        }
    }
    RaGripperCollisionFrame frame = ra_gripf(
        links, end_effector);
    const int shell_source[RA_CUDA_SHELL_BOXES] = {0, 1, 2, 3, 5};
    for (int item = 0; item < RA_CUDA_SHELL_BOXES; ++item) {
        int box_index = shell_source[item];
        if (box_index <= 2) {
            continue;
        }
        RaCollisionBox box = ra_gripb(
            &frame, box_index);
        if (ra_tblhit(
                box.pose, box.half_extent, RA_CONTACT_MARGIN)) {
            return 1;
        }
    }
    for (int pad = 0; pad < RA_PAD_BOXES; ++pad) {
        RaConvexShape left = ra_padsh(
            frame.left_finger, pad);
        RaConvexShape right = ra_padsh(
            frame.right_finger, pad);
        if (ra_tblhit(
                left.pose, left.half_extents, RA_CONTACT_MARGIN)
                || ra_tblhit(
                    right.pose, right.half_extents, RA_CONTACT_MARGIN)) {
            return 1;
        }
    }
    return 0;
}

RA_D static RA_INLINE void ra_intpos(
        RaCudaProductionWorld* world) {
    RaState* state = &world->state;
    if (state->basketball_mode) {
        ra_advb(world);
    } else {
        ra_advbox(
            world, RA_CUDA_BODY_CUBE, RA_CUDA_BODY_TABLE);
    }
    if (state->stack_mode) {
        ra_advbox(
            world, RA_CUDA_BODY_BASE, RA_CUDA_BODY_TABLE);
    }
    float previous_q[RA_DOF];
    float candidate_q[RA_DOF];
    for (int joint = 0; joint < RA_DOF; ++joint) {
        previous_q[joint] = state->q[joint];
        state->q[joint] += state->qd[joint] * RA_PHYSICS_DT;
        float low = ra_jmin(joint);
        float high = ra_jmax(joint);
        if (state->q[joint] < low) {
            state->q[joint] = low;
            state->qd[joint] = ra_max(state->qd[joint], 0.0f) * 0.15f;
        } else if (state->q[joint] > high) {
            state->q[joint] = high;
            state->qd[joint] = ra_min(state->qd[joint], 0.0f) * 0.15f;
        }
        candidate_q[joint] = state->q[joint];
    }
    if (ra_tblpen(state)) {
        float valid = 0.0f;
        float invalid = 1.0f;
        for (int iteration = 0; iteration < 8; ++iteration) {
            float fraction = 0.5f * (valid + invalid);
            for (int joint = 0; joint < RA_DOF; ++joint) {
                state->q[joint] = previous_q[joint]
                    + fraction*(candidate_q[joint] - previous_q[joint]);
            }
            if (ra_tblpen(state)) {
                invalid = fraction;
            } else {
                valid = fraction;
            }
        }
        for (int joint = 0; joint < RA_DOF; ++joint) {
            state->q[joint] = previous_q[joint]
                + valid*(candidate_q[joint] - previous_q[joint]);
            state->qd[joint] *= 0.15f;
        }
    }
    state->gripper_width = ra_clamp(state->gripper_width
        + state->gripper_velocity * RA_PHYSICS_DT, 0.004f, 0.080f);
    if ((state->gripper_width <= 0.004f && state->gripper_velocity < 0.0f)
            || (state->gripper_width >= 0.080f
                && state->gripper_velocity > 0.0f)) {
        state->gripper_velocity = 0.0f;
    }
    ra_copyb(world);
}

RA_D static RA_INLINE void ra_colc(
        RaCudaProductionWorld* world) {
    RaState* state = &world->state;
    state->pad_normal_impulse[0] = 0.0f;
    state->pad_normal_impulse[1] = 0.0f;
    state->wrist_linear_impulse = ra_v3(0, 0, 0);
    state->wrist_angular_impulse = ra_v3(0, 0, 0);
    for (int manifold_index = 0;
            manifold_index < world->rigid.manifold_count; ++manifold_index) {
        PlImpulseManifold* manifold =
            &world->rigid.manifolds[manifold_index];
        int robot_contact = manifold->body_b >= RA_CUDA_BODY_SHELL_START
            && manifold->body_b < RA_CUDA_ROBOT_BODY_END;
        for (int point_index = 0; point_index < manifold->point_count;
                ++point_index) {
            PlImpulsePoint* point = &manifold->points[point_index];
            float impulse = ra_max(point->normal_impulse, 0.0f);
            if (manifold->body_b >= RA_CUDA_BODY_PAD_LEFT_START
                    && manifold->body_b < RA_CUDA_BODY_PAD_RIGHT_START) {
                state->pad_normal_impulse[0] += impulse;
            } else if (manifold->body_b >= RA_CUDA_BODY_PAD_RIGHT_START
                    && manifold->body_b < RA_CUDA_ROBOT_BODY_END) {
                state->pad_normal_impulse[1] += impulse;
            }
            if (robot_contact && impulse > 0.0f) {
                RaVec3 reaction = ra_scale(manifold->normal, -impulse);
                state->wrist_linear_impulse = ra_add(
                    state->wrist_linear_impulse, reaction);
                state->wrist_angular_impulse = ra_add(
                    state->wrist_angular_impulse,
                    ra_cross(ra_sub(point->point_b, state->end_effector),
                        reaction));
            }
        }
    }
    state->gripper_force = 0.5f
        * (state->pad_normal_impulse[0] + state->pad_normal_impulse[1])
        / RA_PHYSICS_DT;
}

RA_D static RA_INLINE void ra_prep(
        RaCudaProductionWorld* world,
        float mass_factor[RA_DOF][RA_DOF]) {
    RaState* state = &world->state;
    RaPose* links = world->staged.links;
    RaVec3* origins = world->staged.origins;
    RaVec3* axes = world->staged.axes;
    float target_width = world->staged.target_width;
    float motor = ra_clamp(
        RA_GRIPPER_FORCE_STIFFNESS * (target_width - state->gripper_width)
            - RA_GRIPPER_FORCE_DAMPING * state->gripper_velocity,
        -RA_GRIPPER_MAX_FORCE, RA_GRIPPER_MAX_FORCE);
    state->gripper_velocity += motor
        / RA_GRIPPER_EFFECTIVE_MASS * RA_PHYSICS_DT;
    float matrix[RA_DOF][RA_DOF];
    float gravity[RA_DOF];
    float rhs[RA_DOF];
    float acceleration[RA_DOF] = {0};
    ra_massm(state, matrix);
    ra_gravt(state, gravity);
    for (int joint = 0; joint < RA_DOF; ++joint) {
        float kp = joint < 2 ? 4500.0f : (joint < 4 ? 3500.0f : 2000.0f);
        float kd = joint < 2 ? 450.0f : (joint < 4 ? 350.0f : 200.0f);
        float arm_motor = ra_clamp(
            kp * (state->target_q[joint] - state->q[joint])
                - kd * state->qd[joint],
            -ra_mlim(joint), ra_mlim(joint));
        rhs[joint] = arm_motor - state->qd[joint] + gravity[joint];
        world->staged.energy += fabsf(arm_motor * state->qd[joint])
            * RA_PHYSICS_DT;
    }
    ra_massf(matrix, mass_factor);
    ra_masss(mass_factor, rhs, acceleration);
    for (int joint = 0; joint < RA_DOF; ++joint) {
        state->qd[joint] = ra_clamp(
            state->qd[joint] + acceleration[joint] * RA_PHYSICS_DT,
            -12.0f, 12.0f);
    }
    if (state->basketball_mode) {
        state->cube_velocity = ra_bvel(
            state->cube_velocity, RA_PHYSICS_DT);
    } else {
        state->cube_velocity.y -= 9.81f * RA_PHYSICS_DT;
    }
    if (state->stack_mode) {
        state->base_cube_velocity.y -= 9.81f * RA_PHYSICS_DT;
    }
    ra_fk(state->q, state->gripper_width, links, origins, axes, &state->end_effector);
    ra_bodies(world);
}

RA_D static RA_INLINE void ra_solve(
        RaCudaProductionWorld* world) {
    RaState* state = &world->state;
    RaPose* links = world->staged.links;
    float backboard_incoming_speed = 0.0f;
    RaVec3 backboard_normal = ra_v3(0, 0, 0);
    if (state->basketball_mode) {
        for (int index = 0; index < world->rigid.manifold_count; ++index) {
            const PlImpulseManifold* manifold =
                &world->rigid.manifolds[index];
            if (manifold->body_a != RA_CUDA_BODY_CUBE
                    || manifold->body_b != RA_CUDA_BODY_BACKBOARD
                    || manifold->point_count <= 0) {
                continue;
            }
            float incoming = -ra_dot(
                world->rigid.bodies[RA_CUDA_BODY_CUBE].linear_velocity,
                manifold->normal);
            if (incoming > backboard_incoming_speed) {
                backboard_incoming_speed = incoming;
                backboard_normal = manifold->normal;
            }
        }
    }
    pl_isort(world->rigid.manifolds,
        world->rigid.manifold_count);
    pl_isolve(world->rigid.bodies, world->rigid.body_count,
        world->rigid.manifolds, world->rigid.manifold_count, RA_PHYSICS_DT,
        &world->rigid.config, &world->rigid.cache, state);
    if (backboard_incoming_speed > 0.0f) {
        RaRigidBody* ball =
            &world->rigid.bodies[RA_CUDA_BODY_CUBE];
        float outgoing_speed = ra_dot(
            ball->linear_velocity, backboard_normal);
        float rebound_floor =
            RA_BACKBOARD_RESTITUTION * backboard_incoming_speed;
        if (outgoing_speed < rebound_floor) {
            ball->linear_velocity = ra_add(ball->linear_velocity,
                ra_scale(backboard_normal,
                    rebound_floor - outgoing_speed));
        }
    }
    ra_intpos(world);
    ra_fk(state->q, state->gripper_width, links, NULL, NULL, &state->end_effector);
    ra_colc(world);

    int active_pad[2] = {0, 0};
    for (int manifold_index = 0;
            manifold_index < world->rigid.manifold_count;
            ++manifold_index) {
        const PlImpulseManifold* manifold =
            &world->rigid.manifolds[manifold_index];
        int side = manifold->body_b >= RA_CUDA_BODY_PAD_LEFT_START
            && manifold->body_b < RA_CUDA_BODY_PAD_RIGHT_START ? 0
            : (manifold->body_b >= RA_CUDA_BODY_PAD_RIGHT_START
                && manifold->body_b < RA_CUDA_ROBOT_BODY_END ? 1 : -1);
        if (side < 0 || world->rigid.compound_pad_component_mask[side] == 0) {
            continue;
        }
        for (int point = 0; point < manifold->point_count; ++point) {
            const PlImpulsePoint* contact = &manifold->points[point];
            if (contact->normal_impulse > 1.0e-8f
                    && contact->separation
                        <= RA_CONTACT_MARGIN + 1.0e-6f) {
                active_pad[side] = 1;
            }
        }
    }
    float grip_action = ra_clamp(world->staged.actions[RA_DOF], -1.0f, 1.0f);
    int between_pads = 0;
    if (active_pad[0] && active_pad[1]) {
        RaGripperCollisionFrame frame = ra_gripf(
            links, state->end_effector);
        RaVec3 left_inward = ra_scale(ra_rotate(
            frame.left_finger.rotation, ra_v3(0, 1, 0)), -1.0f);
        RaVec3 right_inward = ra_scale(ra_rotate(
            frame.right_finger.rotation, ra_v3(0, 1, 0)), -1.0f);
        between_pads =
            ra_dot(ra_sub(state->cube_position, frame.left_finger.position),
                left_inward) > 0.0f
            && ra_dot(ra_sub(state->cube_position, frame.right_finger.position),
                right_inward) > 0.0f;
    }
    int pad_pinch = between_pads
        && state->pad_normal_impulse[0] > 1.0e-7f
        && state->pad_normal_impulse[1] > 1.0e-7f
        && grip_action < 0.25f;
    int grasp_loss_substeps = state->basketball_mode
            && grip_action <= 0.25f
        ? RA_BASKETBALL_GRASP_LOSS_SUBSTEPS : RA_GRASP_LOSS_SUBSTEPS;
    if (pad_pinch) {
        state->grasp_contact_misses = 0;
        state->episode_pinch_force += state->gripper_force;
        state->pinch_substeps += 1;
    }
    if (!state->grasped && !state->basketball_in_flight
            && !world->staged.grasp_broken
            && state->grasp_cooldown == 0 && pad_pinch) {
        world->staged.first_grasp |= !state->ever_grasped;
        state->grasped = 1;
        state->ever_grasped = 1;
    } else if (state->grasped && !pad_pinch
            && ++state->grasp_contact_misses >= grasp_loss_substeps) {
        state->grasped = 0;
        world->staged.grasp_broken = 1;
        if (grip_action > 0.25f) {
            state->grasp_cooldown = RA_GRASP_COOLDOWN_STEPS;
            world->staged.released = 1;
        } else {
            state->grasp_cooldown = 0;
            state->slip_events += 1;
        }
    }
    if (state->stack_mode) {
        state->target_position = ra_add(state->base_cube_position,
            ra_v3(0, ra_csup(
                state->base_cube_rotation, ra_v3(0, 1, 0))
                + RA_CUBE_HALF, 0));
    }
}

RA_D static RA_INLINE void ra_objr(
        RaCudaProductionWorld* world, int object_body, int pad_mask) {
    RaCudaRigidWorld* rigid = &world->rigid;
    for (int item = 0; item < RA_CUDA_SHELL_BOXES; ++item) {
        if (item < 3 && world->state.basketball_mode) {
            continue;
        }
        if (item == 3 && (pad_mask & 1)) {
            continue;
        }
        if (item == 4 && (pad_mask & 2)) {
            continue;
        }
        ra_pair(rigid, object_body,
            RA_CUDA_BODY_SHELL_START + item, RA_CONTACT_MARGIN,
            RA_HAND_COLLISION_FRICTION, RA_HAND_COLLISION_FRICTION, 0.0f);
    }
    const RaConvexShape* object = &rigid->shapes[object_body];
    float object_radius = ra_brad(object)
        + RA_CONTACT_MARGIN;
    for (int item = 0; item < RA_DOF; ++item) {
        int link_body = RA_CUDA_BODY_LINK_START + item;
        RaVec3 delta = ra_sub(object->pose.position,
            rigid->shapes[link_body].pose.position);
        float limit = object_radius
            + ra_brad(&rigid->shapes[link_body]);
        if (ra_dot(delta, delta) > limit * limit) {
            continue;
        }
        ra_pair(rigid, object_body, link_body, RA_CONTACT_MARGIN,
            RA_HAND_COLLISION_FRICTION, RA_HAND_COLLISION_FRICTION, 0.0f);
    }
}

RA_D static RA_INLINE void ra_botc(
        RaCudaProductionWorld* world) {
    RaCudaRigidWorld* rigid = &world->rigid;
    if (world->state.basketball_mode) {
        ra_bpad(world, 0);
        ra_bpad(world, 1);
    } else {
        ra_padc(world, 0, RA_CUDA_BODY_CUBE);
        ra_padc(world, 1, RA_CUDA_BODY_CUBE);
        if (world->state.stack_mode) {
            ra_padc(world, 0, RA_CUDA_BODY_BASE);
            ra_padc(world, 1, RA_CUDA_BODY_BASE);
        }
    }
    int pad_mask = (rigid->compound_pad_component_mask[0] != 0 ? 1 : 0)
        | (rigid->compound_pad_component_mask[1] != 0 ? 2 : 0);
    ra_objr(
        world, RA_CUDA_BODY_CUBE, pad_mask);
    if (world->state.stack_mode) {
        ra_objr(
            world, RA_CUDA_BODY_BASE, pad_mask);
    }
}

RA_D static RA_INLINE void ra_buildc(
        RaCudaProductionWorld* world) {
    RaState* state = &world->state;
    RaCudaRigidWorld* rigid = &world->rigid;
    const float friction = state->stack_mode
        ? RA_STACK_STATIC_FRICTION
        : (state->basketball_mode ? RA_BALL_FRICTION : RA_CUBE_FRICTION);
    const float dynamic_friction = state->stack_mode
        ? RA_STACK_DYNAMIC_FRICTION
        : (state->basketball_mode ? RA_BALL_FRICTION : RA_CUBE_FRICTION);
    const float restitution = state->basketball_mode
        ? RA_BALL_RESTITUTION : RA_CUBE_RESTITUTION;
    ra_pair(rigid, RA_CUDA_BODY_CUBE, RA_CUDA_BODY_TABLE,
        RA_CONTACT_MARGIN, friction, dynamic_friction, restitution);
    if (state->stack_mode) {
        ra_pair(rigid, RA_CUDA_BODY_BASE, RA_CUDA_BODY_TABLE,
            RA_CONTACT_MARGIN, RA_STACK_STATIC_FRICTION,
            RA_STACK_DYNAMIC_FRICTION, RA_CUBE_RESTITUTION);
        ra_pair(rigid, RA_CUDA_BODY_CUBE, RA_CUDA_BODY_BASE,
            RA_CONTACT_MARGIN, RA_STACK_STATIC_FRICTION,
            RA_STACK_DYNAMIC_FRICTION, RA_CUBE_RESTITUTION);
    }
    if (state->basketball_mode) {
        ra_pair(rigid, RA_CUDA_BODY_CUBE,
            RA_CUDA_BODY_BACKBOARD, RA_CONTACT_MARGIN,
            RA_BACKBOARD_STATIC_FRICTION,
            RA_BACKBOARD_DYNAMIC_FRICTION,
            RA_BACKBOARD_RESTITUTION);
        RaConvexContact rim_contact = ra_rimq(
            rigid->bodies[RA_CUDA_BODY_CUBE].pose.position,
            RA_CONTACT_MARGIN);
        if (rim_contact.hit
                && ra_manok(rigid)) {
            PlImpulseCandidate rim_candidate;
            memset(&rim_candidate, 0, sizeof(rim_candidate));
            rim_candidate.contact = rim_contact;
            rim_candidate.feature = 0x72000000u;
            PlImpulseManifold* rim_manifold =
                &rigid->manifolds[rigid->manifold_count];
            int rim_count = pl_iman(
                RA_CUDA_BODY_CUBE, RA_CUDA_BODY_RIM, &rim_candidate, 1,
                RA_CONTACT_MARGIN, 0.55f, 0.45f, RA_BALL_RESTITUTION,
                rim_manifold);
            rigid->manifold_count += rim_count > 0;
        }
    }
    ra_botc(world);
    for (int body = RA_CUDA_BODY_SHELL_START;
            body < RA_CUDA_ROBOT_BODY_END; ++body) {
        if (body < RA_CUDA_BODY_SHELL_START + 3) {
            continue;
        }
        const RaConvexShape* shape = &rigid->shapes[body];
        if (ra_tbllo(
                shape->pose, shape->half_extents,
                RA_CONTACT_MARGIN + 0.002f)) {
            ra_pair(rigid, RA_CUDA_BODY_TABLE, body,
                RA_CONTACT_MARGIN, 0.80f, 0.70f, 0.0f);
        }
    }
}

enum { RA_CUDA_BLOCK_SIZE = 128 };

typedef struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    RaCudaProductionWorld world;
} Env;

static_assert(sizeof(RaState) % sizeof(unsigned int) == 0,
    "Robot-arm CUDA state must remain naturally word aligned");

__global__ void ra_kinit(Env* envs, obs_t* observations,
        float* rewards, float* terminals, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float local_observation[OBS_SIZE];
    ra_observe(&envs[index].world.state, local_observation);
    for (int feature = 0; feature < OBS_SIZE; ++feature) {
        observations[index * OBS_SIZE + feature] =
            __float2bfloat16(local_observation[feature]);
    }
    rewards[index] = 0.0f;
    terminals[index] = 0.0f;
}

__global__ void ra_kbegin(Env* envs, int start, int count,
        const float* actions) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) {
        return;
    }
    int state_index = start + local;
    RaCudaProductionWorld* world = &envs[state_index].world;
    RaState* state = &world->state;
    for (int action = 0; action < RA_ACTIONS; ++action) {
        world->staged.actions[action] = actions[state_index * RA_ACTIONS
            + action];
    }
    for (int joint = 0; joint < RA_DOF; ++joint) {
        float action = ra_clamp(world->staged.actions[joint], -1.0f, 1.0f);
        state->target_q[joint] = ra_clamp(ra_jhome(joint)
            + action * ra_aspan(joint),
            ra_jmin(joint), ra_jmax(joint));
    }
    float grip_action = ra_clamp(world->staged.actions[RA_DOF], -1.0f, 1.0f);
    world->staged.target_width =
        0.004f + 0.076f * 0.5f * (grip_action + 1.0f);
    world->staged.energy = 0.0f;
    world->staged.first_grasp = 0;
    world->staged.grasp_broken = 0;
    world->staged.released = 0;
    if (state->grasp_cooldown > 0) {
        state->grasp_cooldown -= 1;
    }
    if (state->basketball_mode) {
        state->previous_ball_position = state->cube_position;
    }
    ra_fk(state->q, state->gripper_width, world->staged.links,
        NULL, NULL, &state->end_effector);
    state->step += 1;
}

__global__ void ra_kphys(Env* envs, int start,
        int count) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) {
        return;
    }
    RaCudaProductionWorld* world = &envs[start + local].world;
    for (int substep = 0; substep < RA_SUBSTEPS; ++substep) {
        float mass_factor[RA_DOF][RA_DOF];
        ra_prep(world, mass_factor);
        ra_buildc(world);
        ra_react(world, mass_factor);
        ra_solve(world);
    }
}

__global__ void ra_kfin(Env* envs, int start, int count,
        obs_t* observations, float* rewards, float* terminals) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) {
        return;
    }
    int state_index = start + local;
    Env* env = envs + state_index;
    RaCudaProductionWorld* world = &env->world;
    float reward = world->state.basketball_mode
        ? ra_stepb(&world->state, world->staged.actions,
            world->staged.energy, world->staged.first_grasp,
            world->staged.released, world->staged.links)
        : ra_stept(&world->state, world->staged.actions,
            world->staged.energy, world->staged.first_grasp,
            world->staged.released, world->staged.links);
    float terminal = world->state.done ? 1.0f : 0.0f;
    if (terminal != 0.0f) {
        ra_logep(&world->state, &env->log);
        unsigned int topology = ra_topo(&world->state);
        ra_reset(&world->state);
        ra_rbrst(&world->rigid, topology);
    } else if (world->state.basketball_reset) {
        ra_rbrst(&world->rigid, 3u);
        world->state.basketball_reset = 0;
    }
    float local_observation[OBS_SIZE];
    ra_observe(&world->state, local_observation);
    for (int feature = 0; feature < OBS_SIZE; ++feature) {
        observations[state_index * OBS_SIZE + feature] =
            __float2bfloat16(local_observation[feature]);
    }
    rewards[state_index] = reward;
    terminals[state_index] = terminal;
}
