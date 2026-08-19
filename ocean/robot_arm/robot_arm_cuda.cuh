#pragma once

#include <assert.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "robot_arm.h"

#define RA_DYN_BODIES 10

typedef struct RaInertia3 {
    float xx, yy, zz, xy, xz, yz;
} RaInertia3;

enum { RA_CONVEX_BOX = 0, RA_CONVEX_SPHERE = 1 };

typedef struct RaConvexShape {
    int type;
    RaPose pose;
    RaVec3 half_extents;
} RaConvexShape;

typedef struct RaRigidBody {
    RaPose pose;
    RaVec3 linear_velocity;
    RaVec3 angular_velocity;
    float mass;
    RaInertia3 local_inertia;
} RaRigidBody;

typedef struct RaConvexContact {
    int hit;
    int iterations;
    float separation;
    RaVec3 normal;
    RaVec3 point_a;
    RaVec3 point_b;
} RaConvexContact;

typedef struct RaConvexSweep {
    int hit;
    int iterations;
    float toi;
    RaConvexContact contact;
} RaConvexSweep;

RA_D static RA_INLINE void ra_caxes(RaQuat rotation, RaVec3 axes[3]) {
    axes[0] = ra_rotate(rotation, ra_v3(1, 0, 0));
    axes[1] = ra_rotate(rotation, ra_v3(0, 1, 0));
    axes[2] = ra_rotate(rotation, ra_v3(0, 0, 1));
}

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
    return projection_a_minus_b < 0.0f ? ra_scale(axis, -1.0f) : axis;
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
        normal = ra_scale(normal, -1.0f);
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
    query.contact.point_a = pl_ssup(a, ra_scale(normal, -1.0f));
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
            output[output_count++] = ra_add(previous,
                ra_scale(ra_sub(current, previous), fraction));
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
    float half[3] = {
        box->half_extents.x, box->half_extents.y, box->half_extents.z};
    float face_half = half[normal_axis];
    float half_a = half[tangent_a];
    float half_b = half[tangent_b];
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
        ? ra_scale(contact_normal, -1.0f) : contact_normal;
    const float reference_sign = ra_dot(
        reference->axis[reference_axis], reference_normal) < 0.0f
        ? -1.0f : 1.0f;
    reference_normal = ra_scale(
        reference->axis[reference_axis], reference_sign);
    float ref_half[3] = {
        reference->half_extents.x, reference->half_extents.y,
        reference->half_extents.z};
    const RaVec3 reference_center = ra_add(reference->center,
        ra_scale(reference_normal, ref_half[reference_axis]));

    const RaVec3 incident_target = ra_scale(reference_normal, -1.0f);
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
    float half_a = ref_half[tangent_a];
    float half_b = ref_half[tangent_b];
    const float center_a = ra_dot(reference_center, side_a);
    const float center_b = ra_dot(reference_center, side_b);
    count = pl_sclip(input, count, scratch, side_a,
        center_a + half_a);
    count = pl_sclip(scratch, count, input, ra_scale(side_a, -1.0f),
        -center_a + half_a);
    count = pl_sclip(input, count, scratch, side_b,
        center_b + half_b);
    count = pl_sclip(scratch, count, input, ra_scale(side_b, -1.0f),
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
            if (pl_sused(selected, selected_count, index)) {
                continue;
            }
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
        int duplicate = 0;
        float epsilon_squared = PL_SAT_MANIFOLD_DUPLICATE_EPSILON
            * PL_SAT_MANIFOLD_DUPLICATE_EPSILON;
        for (int dup = 0; dup < manifold->count; ++dup) {
            RaVec3 delta_a = ra_sub(contact.point_a,
                manifold->point[dup].point_a);
            RaVec3 delta_b = ra_sub(contact.point_b,
                manifold->point[dup].point_b);
            if (ra_dot(delta_a, delta_a) <= epsilon_squared
                    && ra_dot(delta_b, delta_b) <= epsilon_squared) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
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
    if (a->type == RA_CONVEX_SPHERE && b->type == RA_CONVEX_BOX) {
        return pl_ssbox(a, b, margin);
    }
    if (a->type == RA_CONVEX_BOX && b->type == RA_CONVEX_SPHERE) {
        PlSatQuery query = pl_ssbox(b, a, margin);
        RaVec3 point = query.contact.point_a;
        query.contact.point_a = query.contact.point_b;
        query.contact.point_b = point;
        query.contact.normal = ra_scale(query.contact.normal, -1.0f);
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

RA_D static RA_INLINE RaQuat ra_qint(
        RaQuat rotation, RaVec3 angular_velocity, float dt) {
    float speed = ra_length(angular_velocity);
    if (speed < 1.0e-8f) {
        return ra_qnorm(rotation);
    }
    RaQuat increment = ra_qaxis(
        ra_scale(angular_velocity, 1.0f / speed), speed * dt);
    return ra_qnorm(ra_qmul(increment, rotation));
}

RA_D static RA_INLINE RaVec3 ra_invi(
        const RaRigidBody* body, RaVec3 vector) {
    RaInertia3 inertia = body->local_inertia;
    RaVec3 local = ra_rotate(ra_qconj(body->pose.rotation), vector);
    float cofactor_xx = inertia.yy*inertia.zz - inertia.yz*inertia.yz;
    float cofactor_xy = inertia.xz*inertia.yz - inertia.xy*inertia.zz;
    float cofactor_xz = inertia.xy*inertia.yz - inertia.xz*inertia.yy;
    float cofactor_yy = inertia.xx*inertia.zz - inertia.xz*inertia.xz;
    float cofactor_yz = inertia.xy*inertia.xz - inertia.xx*inertia.yz;
    float cofactor_zz = inertia.xx*inertia.yy - inertia.xy*inertia.xy;
    float determinant = inertia.xx*cofactor_xx
        + inertia.xy*cofactor_xy + inertia.xz*cofactor_xz;
    float inverse = 1.0f / ra_max(fabsf(determinant), 1.0e-18f);
    if (determinant < 0.0f) {
        inverse = -inverse;
    }
    RaVec3 product = ra_v3(
        inverse*(cofactor_xx*local.x + cofactor_xy*local.y
            + cofactor_xz*local.z),
        inverse*(cofactor_xy*local.x + cofactor_yy*local.y
            + cofactor_yz*local.z),
        inverse*(cofactor_xz*local.x + cofactor_yz*local.y
            + cofactor_zz*local.z));
    return ra_rotate(body->pose.rotation, product);
}

RA_D static RA_INLINE void ra_impa(
        RaRigidBody* body, RaVec3 point, RaVec3 impulse) {
    if (body->mass <= 0.0f) {
        return;
    }
    body->linear_velocity = ra_add(body->linear_velocity,
        ra_scale(impulse, 1.0f / body->mass));
    body->angular_velocity = ra_add(body->angular_velocity,
        ra_invi(body,
            ra_cross(ra_sub(point, body->pose.position), impulse)));
}

RA_D static RA_INLINE float ra_impd(
        const RaRigidBody* body, RaVec3 point, RaVec3 direction) {
    if (body->mass <= 0.0f) {
        return 0.0f;
    }
    RaVec3 lever = ra_sub(point, body->pose.position);
    RaVec3 angular = ra_cross(lever, direction);
    return 1.0f / body->mass + ra_dot(angular,
        ra_invi(body, angular));
}

RA_D static RA_INLINE float ra_brad(
        const RaConvexShape* shape) {
    return shape->type == RA_CONVEX_SPHERE
        ? shape->half_extents.x : ra_length(shape->half_extents);
}

RA_D static RA_INLINE void ra_applyr(
        RaState* state, const float response[RA_DOF], float magnitude) {
    for (int joint = 0; joint < RA_DOF; ++joint) {
        state->qd[joint] += response[joint] * magnitude;
    }
}

#define PL_IMPULSE_MAX_CACHE 192
#define PL_IMPULSE_EPSILON 1.0e-8f

typedef struct PlImpulseConfig {
    int velocity_iterations;
    int position_iterations;
    float velocity_impulse_tolerance;  // 0 disables; does not lower the hard cap
    float position_beta;
    float slop;
    float speculative_margin;
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
    int active;  // mass==0 proxy; live qd/gripper, not cached body_b velocity
    float inverse_mass[3];
    float gripper_velocity_response[3];
    float robot_jacobian[3][RA_DOF];
    float gripper_jacobian[3];
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
    PlImpulsePoint points[PL_SAT_MAX_MANIFOLD_POINTS];
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
        const RaConvexContact* contact = &candidates[index].contact;
        if (!pl_ifin(contact->separation)
                || !pl_ifin(contact->normal.x)
                || !pl_ifin(contact->normal.y)
                || !pl_ifin(contact->normal.z)
                || !pl_ifin(contact->point_a.x)
                || !pl_ifin(contact->point_a.y)
                || !pl_ifin(contact->point_a.z)
                || !pl_ifin(contact->point_b.x)
                || !pl_ifin(contact->point_b.y)
                || !pl_ifin(contact->point_b.z)
                || !(contact->hit || contact->separation <= margin)) {
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
        while (cursor > 0) {
            const PlImpulseCandidate* left = &value;
            const PlImpulseCandidate* right = &work[cursor - 1];
            int before = 0;
            if (left->feature != right->feature) {
                before = left->feature < right->feature;
            } else if (left->contact.separation
                    != right->contact.separation) {
                before = left->contact.separation
                    < right->contact.separation;
            } else {
                RaVec3 left_point = pl_ipt(left);
                RaVec3 right_point = pl_ipt(right);
                if (left_point.x != right_point.x) {
                    before = left_point.x < right_point.x;
                } else if (left_point.y != right_point.y) {
                    before = left_point.y < right_point.y;
                } else {
                    before = left_point.z < right_point.z;
                }
            }
            if (!before) {
                break;
            }
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
                == work[unique - 1].feature) {
            continue;
        }
        work[unique++] = work[index];
    }
    usable = unique;
    int selected[PL_SAT_MAX_MANIFOLD_POINTS];
    int selected_count = 0;
    int deepest = 0;
    for (int index = 1; index < usable; ++index) {
        if (work[index].contact.separation
                < work[deepest].contact.separation) {
            deepest = index;
        }
    }
    selected[selected_count++] = deepest;
    while (selected_count < PL_SAT_MAX_MANIFOLD_POINTS
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
    for (int index = 1; index < manifold_count; ++index) {
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
                for (int point = 0; point < left->point_count; ++point) {
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
                && entry->feature == feature) {
            return index;
        }
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
        int better = 0;
        if (left->stamp != right->stamp) {
            better = left->stamp < right->stamp;
        } else if (left->pair_key != right->pair_key) {
            better = left->pair_key < right->pair_key;
        } else if (left->feature != right->feature) {
            better = left->feature < right->feature;
        } else {
            better = index < best;
        }
        if (better) {
            best = index;
        }
    }
    return best;
}

RA_D static RA_INLINE float pl_imass(
        const RaRigidBody* body_a, RaVec3 point_a,
        const RaRigidBody* body_b, RaVec3 point_b, RaVec3 direction,
        const PlImpulseReaction* reaction, int direction_index) {
    float denominator = ra_impd(body_a, point_a, direction)
        + ra_impd(body_b, point_b, direction);
    if (reaction->active) {
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
    if (!reaction->active) {
        return;
    }
    ra_applyr(reaction_state,
        reaction->robot_response[direction_index], -impulse);
    reaction_state->gripper_velocity +=
        reaction->gripper_velocity_response[direction_index] * impulse;
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
        float velocity = reaction->gripper_jacobian[direction_index]
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
    if (!reaction->active) {
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
    point->point_a = ra_add(body_a->pose.position,
        ra_rotate(body_a->pose.rotation, point->local_a));
    point->point_b = ra_add(body_b->pose.position,
        ra_rotate(body_b->pose.rotation, point->local_b));
    point->separation = ra_dot(ra_sub(point->point_a, point->point_b),
        manifold->normal) + point->prescribed_separation_offset;
}

RA_D static RA_INLINE void pl_ildc(
        const PlImpulseCache* cache, PlImpulseManifold* manifold,
        PlImpulsePoint* point, int max_age) {
    point->normal_impulse = 0.0f;
    point->tangent_1_impulse = 0.0f;
    point->tangent_2_impulse = 0.0f;
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
        int max_age) {
    manifold->torsional_impulse = 0.0f;
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
    assert(manifold_count >= 0 && manifold_count <= PL_IMPULSE_MAX_MANIFOLDS);
    for (int manifold_index = 0; manifold_index < manifold_count;
            ++manifold_index) {
        PlImpulseManifold* manifold = &manifolds[manifold_index];
        assert(manifold->body_a >= 0 && manifold->body_b >= 0
            && manifold->body_a < body_count
            && manifold->body_b < body_count
            && manifold->body_a != manifold->body_b);
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
        assert(manifold->point_count >= 0
            && manifold->point_count <= PL_SAT_MAX_MANIFOLD_POINTS);
        for (int point_index = 0; point_index < manifold->point_count;
                ++point_index) {
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
            pl_ildc(cache, manifold, point, config->cache_max_age);
        }
        pl_ilda(cache, manifold, config->cache_max_age);
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
    for (int iteration = 0; iteration < config->position_iterations;
            ++iteration) {
        float maximum_penetration = 0.0f;
        for (int manifold_index = 0; manifold_index < manifold_count;
                ++manifold_index) {
            PlImpulseManifold* manifold = &manifolds[manifold_index];
            if (manifold->point_count <= 0) {
                continue;
            }
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
                float correction = ra_min(
                    config->position_beta * penetration,
                    config->max_position_correction);
                float denominator = pl_imass(body_a,
                    point->point_a, body_b, point->point_b, manifold->normal,
                    &point->reaction, 0) + point->normal_cfm;
                if (denominator <= PL_IMPULSE_EPSILON) {
                    continue;
                }
                float magnitude = ra_min(correction / denominator,
                    config->max_position_impulse);
                pl_iapos(body_a, point->point_a,
                    ra_scale(manifold->normal, magnitude));
                pl_iapos(body_b, point->point_b,
                    ra_scale(manifold->normal, -magnitude));
                float delta_q[RA_DOF] = {0.0f};
                float delta_width = 0.0f;
                if (point->reaction.active) {
                    for (int joint = 0; joint < RA_DOF; ++joint) {
                        delta_q[joint] = -magnitude
                            * point->reaction.robot_response[0][joint];
                    }
                    delta_width = magnitude
                        * point->reaction.gripper_velocity_response[0];
                    ra_applyr(reaction_state,
                        point->reaction.robot_response[0], -magnitude);
                    reaction_state->gripper_width +=
                        point->reaction.gripper_velocity_response[0] * magnitude;
                    reaction_state->gripper_width = ra_clamp(
                        reaction_state->gripper_width, 0.0f, 0.20f);
                }
                for (int other = 0; other < manifold_count; ++other) {
                    PlImpulseManifold* other_manifold = &manifolds[other];
                    for (int other_point = 0;
                            other_point < other_manifold->point_count;
                            ++other_point) {
                        PlImpulsePoint* offset =
                            &other_manifold->points[other_point];
                        if (!offset->reaction.active) {
                            continue;
                        }
                        float displacement = offset->reaction.gripper_jacobian[0]
                            * delta_width;
                        for (int joint = 0; joint < RA_DOF; ++joint) {
                            displacement +=
                                offset->reaction.robot_jacobian[0][joint]
                                * delta_q[joint];
                        }
                        offset->prescribed_separation_offset -= displacement;
                    }
                }
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
    for (int manifold_index = 0; manifold_index < manifold_count;
            ++manifold_index) {
        PlImpulseManifold* manifold = &manifolds[manifold_index];
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

RA_D static RA_INLINE void pl_iws(
        RaRigidBody* bodies, int body_count, PlImpulseManifold* manifolds,
        int manifold_count, RaState* reaction_state) {
    for (int manifold_index = 0; manifold_index < manifold_count;
            ++manifold_index) {
        PlImpulseManifold* manifold = &manifolds[manifold_index];
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
        if (manifold->angular_reaction.active == 0) {
            continue;
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
}

RA_D static RA_INLINE void pl_iafr(
        RaRigidBody* body_a, RaRigidBody* body_b,
        PlImpulseManifold* manifold, RaState* reaction_state) {
    if (manifold->angular_reaction.active == 0
            || manifold->torsional_radius <= 0.0f) {
        return;
    }
    float normal = 0.0f;
    for (int point = 0; point < manifold->point_count; ++point) {
        normal += ra_max(manifold->points[point].normal_impulse, 0.0f);
    }
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

RA_D static RA_INLINE void pl_isvel(
        RaRigidBody* bodies, int body_count, PlImpulseManifold* manifolds,
        int manifold_count, const PlImpulseConfig* config,
        RaState* reaction_state) {
    pl_iws(bodies, body_count, manifolds,
        manifold_count, reaction_state);
    for (int iteration = 0; iteration < config->velocity_iterations;
            ++iteration) {
        float maximum_impulse_delta = 0.0f;
        for (int manifold_index = 0; manifold_index < manifold_count;
                ++manifold_index) {
            PlImpulseManifold* manifold = &manifolds[manifold_index];
            if (manifold->point_count <= 0) {
                continue;
            }
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
                    config->max_normal_impulse);
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
    for (int manifold_index = 0; manifold_index < manifold_count;
            ++manifold_index) {
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
    cache->tick += 1u;
    if (cache->tick == 0u) {
        cache->tick = 1u;
    }
    int write = 0;
    for (int read = 0; read < cache->count; ++read) {
        PlImpulseCacheEntry* entry = &cache->entries[read];
        uint32_t age = cache->tick - entry->stamp;
        if (age > (uint32_t)config->cache_max_age) {
            continue;
        }
        if (write != read) {
            cache->entries[write] = *entry;
        }
        ++write;
    }
    cache->count = write;
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

typedef struct RaStaged {
    float actions[RA_ACTIONS];
    float target_width;
    float energy;
    int first_grasp;
    int grasp_broken;
    int released;
    RaPose links[RA_LINKS];
    RaVec3 origins[RA_DOF];
    RaVec3 axes[RA_DOF];
} RaStaged;

typedef struct RaWorld {
    RaState state;
    RaCudaRigidWorld rigid;
    RaStaged staged;
} RaWorld;

RA_D static RA_INLINE float ra_csup(
        RaQuat rotation, RaVec3 direction) {
    RaVec3 axes[3];
    ra_caxes(rotation, axes);
    return RA_CUBE_HALF * (fabsf(ra_dot(axes[0], direction))
        + fabsf(ra_dot(axes[1], direction))
        + fabsf(ra_dot(axes[2], direction)));
}

RA_D static RA_INLINE float ra_cup(RaQuat rotation) {
    RaVec3 axes[3];
    ra_caxes(rotation, axes);
    float best = ra_max(fabsf(axes[0].y),
        ra_max(fabsf(axes[1].y), fabsf(axes[2].y)));
    return 1.0f - best;
}

RA_HD static RA_INLINE float ra_rand(
        uint32_t* state, float low, float high) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value ? value : 0x9e3779b9u;
    float unit = (*state >> 8) * (1.0f / 16777216.0f);
    return low + (high - low) * unit;
}

RA_D static RA_INLINE float ra_jmin(int joint) {
    const float values[RA_DOF] = {-2.8973f, -1.7628f, -2.8973f, -3.0718f,
        -2.8973f, -0.0175f, -2.8973f};
    return values[joint];
}

RA_D static RA_INLINE float ra_jmax(int joint) {
    const float values[RA_DOF] = {2.8973f, 1.7628f, 2.8973f, -0.0698f,
        2.8973f, 3.7525f, 2.8973f};
    return values[joint];
}

RA_D static void ra_massg(const RaState* state,
        float matrix[RA_DOF][RA_DOF], float torque[RA_DOF]) {
    float mass[RA_DYN_BODIES] = {
        4.970684f, 0.646926f, 3.228604f, 3.587895f, 1.225946f,
        1.666555f, 0.735522f, 0.730000f, 0.015000f, 0.015000f,
    };
    RaVec3 com_local[RA_DYN_BODIES] = {
        { 0.003875f,  0.002081f, -0.047620f},
        {-0.003141f, -0.028720f,  0.003495f},
        { 0.027518f,  0.039252f, -0.066502f},
        {-0.053170f,  0.104419f,  0.027454f},
        {-0.011953f,  0.041065f, -0.038437f},
        { 0.060149f, -0.014117f, -0.010517f},
        { 0.010517f, -0.004252f,  0.061597f},
        {-0.010000f,  0.000000f,  0.030000f},
        { 0.000000f,  0.000000f,  0.000000f},
        { 0.000000f,  0.000000f,  0.000000f},
    };
    RaInertia3 inertia[RA_DYN_BODIES] = {
        {0.703370f, 0.706610f, 0.009117f,
            -0.000139f,  0.006772f,  0.019169f},
        {0.007962f, 0.028110f, 0.025995f,
            -0.003925f,  0.010254f,  0.000704f},
        {0.037242f, 0.036155f, 0.010830f,
            -0.004761f, -0.011396f, -0.012805f},
        {0.025853f, 0.019552f, 0.028323f,
             0.007796f, -0.001332f,  0.008641f},
        {0.035549f, 0.029474f, 0.008627f,
            -0.002117f, -0.004037f,  0.000229f},
        {0.001964f, 0.004354f, 0.005433f,
             0.000109f, -0.001158f,  0.000341f},
        {0.012516f, 0.010027f, 0.004815f,
            -0.000428f, -0.001196f, -0.000741f},
        {0.001000f, 0.002500f, 0.001700f,
             0.000000f,  0.000000f,  0.000000f},
        {0.000002375f, 0.000002375f, 0.000000750f,
             0.000000000f, 0.000000000f, 0.000000000f},
        {0.000002375f, 0.000002375f, 0.000000750f,
             0.000000000f, 0.000000000f, 0.000000000f},
    };
    for (int joint = 0; joint < RA_DOF; ++joint) {
        torque[joint] = 0.0f;
        for (int column = 0; column < RA_DOF; ++column) {
            matrix[joint][column] = 0.0f;
        }
    }
    RaPose links[RA_LINKS];
    RaPose bodies[RA_DYN_BODIES];
    RaVec3 origins[RA_DOF];
    RaVec3 axes[RA_DOF];
    ra_fk(state->q, state->gripper_width, links, origins, axes, NULL);
    for (int body = 0; body < RA_DOF; ++body) {
        bodies[body] = links[body + 1];
    }
    bodies[7].rotation = links[RA_DOF].rotation;
    bodies[7].position = ra_add(links[RA_DOF].position,
        ra_rotate(links[RA_DOF].rotation, ra_v3(0, 0, 0.107f)));
    bodies[7].rotation = ra_qnorm(ra_qmul(bodies[7].rotation,
        ra_qaxis(ra_v3(0, 0, 1), -0.78539816339f)));
    bodies[8] = links[RA_DOF + 1];
    bodies[9] = links[RA_DOF + 2];
    RaVec3 gravity = {0, -9.81f, 0};
    for (int body = 0; body < RA_DYN_BODIES; ++body) {
        RaVec3 com = ra_add(bodies[body].position,
            ra_rotate(bodies[body].rotation, com_local[body]));
        int last = body < RA_DOF ? body : RA_DOF - 1;
        for (int row = 0; row <= last; ++row) {
            RaVec3 linear_row = ra_cross(
                axes[row], ra_sub(com, origins[row]));
            torque[row] += mass[body] * ra_dot(linear_row, gravity);
            RaVec3 local_axis = ra_rotate(
                ra_qconj(bodies[body].rotation), axes[row]);
            RaInertia3 body_inertia = inertia[body];
            RaVec3 local_inertia = ra_v3(
                body_inertia.xx*local_axis.x + body_inertia.xy*local_axis.y
                    + body_inertia.xz*local_axis.z,
                body_inertia.xy*local_axis.x + body_inertia.yy*local_axis.y
                    + body_inertia.yz*local_axis.z,
                body_inertia.xz*local_axis.x + body_inertia.yz*local_axis.y
                    + body_inertia.zz*local_axis.z);
            RaVec3 inertia_row = ra_rotate(
                bodies[body].rotation, local_inertia);
            for (int column = 0; column <= row; ++column) {
                RaVec3 linear_column = ra_cross(
                    axes[column], ra_sub(com, origins[column]));
                float value = mass[body] * ra_dot(linear_row, linear_column)
                    + ra_dot(axes[column], inertia_row);
                matrix[row][column] += value;
                if (row != column) {
                    matrix[column][row] += value;
                }
            }
        }
    }
    for (int joint = 0; joint < RA_DOF; ++joint) {
        matrix[joint][joint] += 0.1f;
    }
}

RA_D static void ra_masss(
        const float lower[RA_DOF][RA_DOF],
        const float rhs[RA_DOF], float solution[RA_DOF]) {
    float y[RA_DOF] = {0};
    for (int row = 0; row < RA_DOF; ++row) {
        float sum = rhs[row];
        for (int k = 0; k < row; ++k) {
            sum -= lower[row][k] * y[k];
        }
        y[row] = sum / lower[row][row];
    }
    for (int row = RA_DOF - 1; row >= 0; --row) {
        float sum = y[row];
        for (int k = row + 1; k < RA_DOF; ++k) {
            sum -= lower[k][row] * solution[k];
        }
        solution[row] = sum / lower[row][row];
    }
}

RA_D static void ra_jacfk(
        const float lower[RA_DOF][RA_DOF], const RaVec3* origins,
        const RaVec3* axes, int last_joint, RaVec3 point,
        RaVec3 linear_direction, RaVec3 angular_direction,
        float jacobian[RA_DOF], float response[RA_DOF],
        float* inverse_mass) {
    for (int joint = 0; joint < RA_DOF; ++joint) {
        if (joint <= last_joint) {
            RaVec3 linear = ra_cross(
                axes[joint], ra_sub(point, origins[joint]));
            jacobian[joint] = ra_dot(linear, linear_direction)
                + ra_dot(axes[joint], angular_direction);
        } else {
            jacobian[joint] = 0.0f;
        }
        response[joint] = 0.0f;
    }
    ra_masss(lower, jacobian, response);
    *inverse_mass = 0.0f;
    for (int joint = 0; joint < RA_DOF; ++joint) {
        *inverse_mass += jacobian[joint] * response[joint];
    }
}

RA_D static RA_INLINE RaVec3 ra_ptvel(
        const float* qd, const RaVec3* origins, const RaVec3* axes,
        int last_joint, RaVec3 point) {
    RaVec3 velocity = ra_v3(0, 0, 0);
    for (int joint = 0; joint <= last_joint; ++joint) {
        velocity = ra_add(velocity, ra_scale(
            ra_cross(axes[joint], ra_sub(point, origins[joint])),
            qd[joint]));
    }
    return velocity;
}

RA_D static RA_INLINE RaVec3 ra_angvel(
        const float* qd, const RaVec3* axes, int last_joint) {
    RaVec3 velocity = ra_v3(0, 0, 0);
    for (int joint = 0; joint <= last_joint; ++joint) {
        velocity = ra_add(velocity, ra_scale(axes[joint], qd[joint]));
    }
    return velocity;
}

RA_D static RA_INLINE RaConvexShape ra_padsh(
        RaPose finger, int index) {
    RaVec3 local_position[RA_PAD_BOXES] = {
        {0.0f, 0.0055f, 0.0445f},
        {0.0055f, 0.0020f, 0.0500f},
        {-0.0055f, 0.0020f, 0.0500f},
        {0.0055f, 0.0020f, 0.0395f},
        {-0.0055f, 0.0020f, 0.0395f},
    };
    RaVec3 half_extents[RA_PAD_BOXES] = {
        {0.0085f, 0.0040f, 0.0085f},
        {0.0030f, 0.0020f, 0.0030f},
        {0.0030f, 0.0020f, 0.0030f},
        {0.0030f, 0.0020f, 0.0035f},
        {0.0030f, 0.0020f, 0.0035f},
    };
    return (RaConvexShape){
        RA_CONVEX_BOX,
        {ra_add(finger.position,
            ra_rotate(finger.rotation, local_position[index])),
            finger.rotation},
        half_extents[index],
    };
}

RA_D static RA_INLINE float ra_oboxr(
        const RaVec3 axes[3], RaVec3 half_extents, RaVec3 direction) {
    return half_extents.x * fabsf(ra_dot(axes[0], direction))
        + half_extents.y * fabsf(ra_dot(axes[1], direction))
        + half_extents.z * fabsf(ra_dot(axes[2], direction));
}

RA_D static RA_INLINE int ra_padhit(
        RaVec3 cube_position, RaQuat cube_rotation,
        RaPose finger, float margin, RaConvexContact* best) {
    RaVec3 inward = ra_scale(
        ra_rotate(finger.rotation, ra_v3(0, 1, 0)), -1.0f);
    RaVec3 cube_axes[3];
    RaVec3 pad_axes[3];
    ra_caxes(cube_rotation, cube_axes);
    ra_caxes(finger.rotation, pad_axes);
    int found = 0;
    RaVec3 point_a_sum = ra_v3(0, 0, 0);
    RaVec3 point_b_sum = ra_v3(0, 0, 0);
    int manifold_points = 0;
    memset(best, 0, sizeof(*best));
    best->separation = 1.0e30f;
    for (int index = 0; index < RA_PAD_BOXES; ++index) {
        RaConvexShape pad = ra_padsh(finger, index);
        RaConvexContact candidate;
        memset(&candidate, 0, sizeof(candidate));
        RaVec3 delta = ra_sub(cube_position, pad.pose.position);
        RaVec3 cube_half = ra_v3(RA_CUBE_HALF, RA_CUBE_HALF, RA_CUBE_HALF);
        int sat_ok = 1;
        for (int axis_index = 0; axis_index < 15; ++axis_index) {
            RaVec3 axis;
            if (axis_index < 3) {
                axis = cube_axes[axis_index];
            } else if (axis_index < 6) {
                axis = pad_axes[axis_index - 3];
            } else {
                int pair = axis_index - 6;
                axis = ra_cross(cube_axes[pair / 3], pad_axes[pair % 3]);
                float length = ra_length(axis);
                if (length < 1.0e-6f) {
                    continue;
                }
                axis = ra_scale(axis, 1.0f / length);
            }
            float reach = ra_oboxr(
                    cube_axes, cube_half, axis)
                + ra_oboxr(
                    pad_axes, pad.half_extents, axis)
                + margin;
            if (fabsf(ra_dot(delta, axis)) > reach) {
                sat_ok = 0;
                break;
            }
        }
        if (!sat_ok || ra_dot(delta, inward) < 0.0f) {
            continue;
        }
        RaVec3 inner_surface = ra_add(pad.pose.position,
            ra_scale(inward, pad.half_extents.y));
        RaVec3 cube_surface = cube_position;
        RaVec3 support_dir = ra_scale(inward, -1.0f);
        for (int axis = 0; axis < 3; ++axis) {
            float sign = ra_dot(cube_axes[axis], support_dir) < 0.0f
                ? -1.0f : 1.0f;
            cube_surface = ra_add(cube_surface,
                ra_scale(cube_axes[axis], sign * RA_CUBE_HALF));
        }
        float face_separation = ra_dot(
            ra_sub(cube_surface, inner_surface), inward);
        if (face_separation > margin) {
            continue;
        }
        float local_x = ra_dot(delta, pad_axes[0]);
        float local_z = ra_dot(delta, pad_axes[2]);
        float cube_radius_x = ra_oboxr(
            cube_axes, cube_half, pad_axes[0]);
        float cube_radius_z = ra_oboxr(
            cube_axes, cube_half, pad_axes[2]);
        float low_x = ra_max(-pad.half_extents.x,
            local_x - cube_radius_x);
        float high_x = ra_min(pad.half_extents.x,
            local_x + cube_radius_x);
        float low_z = ra_max(-pad.half_extents.z,
            local_z - cube_radius_z);
        float high_z = ra_min(pad.half_extents.z,
            local_z + cube_radius_z);
        float patch_x = low_x <= high_x
            ? 0.5f * (low_x + high_x)
            : ra_clamp(local_x, -pad.half_extents.x, pad.half_extents.x);
        float patch_z = low_z <= high_z
            ? 0.5f * (low_z + high_z)
            : ra_clamp(local_z, -pad.half_extents.z, pad.half_extents.z);
        candidate.hit = 1;
        candidate.iterations = 15;
        candidate.separation = face_separation;
        candidate.normal = inward;
        candidate.point_b = ra_add(inner_surface,
            ra_add(ra_scale(pad_axes[0], patch_x),
                ra_scale(pad_axes[2], patch_z)));
        candidate.point_a = ra_add(candidate.point_b,
            ra_scale(inward, face_separation));
        if (!found || candidate.separation
                < best->separation - 2.0e-5f) {
            *best = candidate;
            point_a_sum = candidate.point_a;
            point_b_sum = candidate.point_b;
            manifold_points = 1;
            found = 1;
        } else if (candidate.separation
                <= best->separation + 2.0e-5f) {
            point_a_sum = ra_add(point_a_sum, candidate.point_a);
            point_b_sum = ra_add(point_b_sum, candidate.point_b);
            manifold_points++;
        }
    }
    if (found) {
        float inverse_points = 1.0f / (float)manifold_points;
        best->point_a = ra_scale(point_a_sum, inverse_points);
        best->point_b = ra_scale(point_b_sum, inverse_points);
        best->normal = inward;
    }
    return found;
}

typedef struct RaGripperCollisionFrame {
    RaPose hand;
    RaPose left_finger;
    RaPose right_finger;
} RaGripperCollisionFrame;

typedef struct RaCollisionBox {
    RaPose pose;
    RaVec3 half_extents;
} RaCollisionBox;

RA_D static RA_INLINE RaGripperCollisionFrame ra_gripf(
        const RaPose* links, RaVec3 end_effector) {
    RaGripperCollisionFrame frame;
    frame.hand.rotation = links[RA_DOF + 1].rotation;
    frame.hand.position = ra_sub(end_effector,
        ra_rotate(frame.hand.rotation, ra_v3(0, 0, 0.115f)));
    frame.left_finger = links[RA_DOF + 1];
    frame.right_finger = links[RA_DOF + 2];
    return frame;
}

RA_D static RA_INLINE RaPose ra_offp(
        RaPose parent, RaVec3 local_position) {
    RaPose pose;
    pose.position = ra_add(parent.position,
        ra_rotate(parent.rotation, local_position));
    pose.rotation = parent.rotation;
    return pose;
}

RA_D static RA_INLINE RaCollisionBox ra_linkb(
        const RaPose* links, int index) {
    const RaVec3 center[RA_DOF] = {
        {-0.00001f, -0.03719f, -0.06850f},
        {-0.00001f, -0.06949f,  0.03720f},
        { 0.04124f,  0.02803f, -0.03300f},
        {-0.04126f,  0.03450f,  0.02803f},
        {-0.00001f,  0.03747f, -0.10340f},
        { 0.04206f,  0.01523f,  0.00613f},
        { 0.01864f,  0.01863f,  0.07940f},
    };
    const RaVec3 half_extents[RA_DOF] = {
        {0.05501f, 0.09220f, 0.12350f},
        {0.05502f, 0.12451f, 0.09220f},
        {0.09626f, 0.08303f, 0.08800f},
        {0.09625f, 0.08950f, 0.08303f},
        {0.05500f, 0.09246f, 0.15560f},
        {0.08996f, 0.06643f, 0.05012f},
        {0.06267f, 0.06265f, 0.02740f},
    };
    return (RaCollisionBox){
        ra_offp(links[index + 1], center[index]),
        half_extents[index],
    };
}

RA_D static RA_INLINE RaCollisionBox ra_gripb(
        const RaGripperCollisionFrame* frame, int index) {
    RaCollisionBox box;
    if (index == 0) {
        box.pose = ra_offp(frame->hand, ra_v3(0, 0, -0.0055f));
        box.half_extents = ra_v3(0.0320f, 0.1040f, 0.0205f);
    } else if (index == 1) {
        box.pose = ra_offp(frame->hand, ra_v3(0, 0, 0.0250f));
        box.half_extents = ra_v3(0.0240f, 0.1020f, 0.0100f);
    } else if (index == 2) {
        box.pose = ra_offp(frame->hand, ra_v3(0, 0, 0.0505f));
        box.half_extents = ra_v3(0.0220f, 0.1010f, 0.0155f);
    } else {
        int right = index >= 5;
        int distal = index == 4 || index == 6;
        RaPose finger = right
            ? frame->right_finger : frame->left_finger;
        if (distal) {
            box.pose = ra_offp(finger, ra_v3(0, 0.0080f, 0.0420f));
            box.half_extents = ra_v3(0.0095f, 0.0080f, 0.0120f);
        } else {
            box.pose = ra_offp(finger, ra_v3(0, 0.0144f, 0.0150f));
            box.half_extents = ra_v3(0.0105f, 0.0120f, 0.0150f);
        }
    }
    return box;
}

RA_HD static RA_INLINE RaVec3 ra_gctr(
        RaVec3 end_effector, RaQuat hand_rotation) {
    return ra_sub(end_effector, ra_rotate(hand_rotation,
        ra_v3(0, 0, RA_BASKETBALL_GRASP_CENTER_OFFSET)));
}

RA_HD static RA_INLINE float ra_blq(
        RaVec3 position, RaVec3 velocity) {
    const float gravity = 9.81f;
    const float drag = RA_BALL_LINEAR_DRAG;
    RaVec3 delta = ra_sub(ra_hoop(), position);
    float horizontal = sqrtf(delta.x*delta.x + delta.z*delta.z);
    float flight_time = ra_clamp(horizontal / 2.20f, 0.45f, 0.75f);
    float travel = (1.0f - expf(-drag * flight_time)) / drag;
    RaVec3 target = ra_v3(delta.x / travel,
        (delta.y + gravity * flight_time / drag) / travel - gravity / drag,
        delta.z / travel);
    RaVec3 error = ra_sub(velocity, target);
    float error_squared = ra_dot(error, error);
    const float sigma = 1.50f;
    return expf(-0.5f * error_squared / (sigma*sigma));
}

RA_D static RA_INLINE void ra_obs_xyz(
        float* observation, int* index, RaVec3 value) {
    observation[(*index)++] = value.x;
    observation[(*index)++] = value.y;
    observation[(*index)++] = value.z;
}

RA_D static RA_INLINE void ra_obs3(
        float* observation, int* index, RaVec3 value, float scale) {
    observation[(*index)++] = ra_clamp(scale * value.x, -1.0f, 1.0f);
    observation[(*index)++] = ra_clamp(scale * value.y, -1.0f, 1.0f);
    observation[(*index)++] = ra_clamp(scale * value.z, -1.0f, 1.0f);
}

RA_D static RA_INLINE float ra_obs_pad(float impulse) {
    return ra_clamp(impulse / (RA_PHYSICS_DT * RA_GRIPPER_MAX_FORCE),
        0.0f, 1.0f);
}

RA_D static void ra_observe(const RaState* state, float* observation) {
    int index = 0;
    for (int joint = 0; joint < RA_DOF; ++joint) {
        float low = ra_jmin(joint);
        float high = ra_jmax(joint);
        float midpoint = 0.5f * (low + high);
        float half_range = 0.5f * (high - low);
        observation[index++] = ra_clamp(
            (state->q[joint] - midpoint) / half_range, -1.0f, 1.0f);
    }
    for (int joint = 0; joint < RA_DOF; ++joint) {
        observation[index++] = ra_clamp(state->qd[joint] / 6.0f, -1.0f, 1.0f);
    }
    for (int action = 0; action < RA_ACTIONS; ++action) {
        observation[index++] = ra_clamp(
            state->previous_action[action], -1.0f, 1.0f);
    }
    RaPose gripper_links[RA_LINKS];
    RaVec3 origins[RA_DOF];
    RaVec3 axes[RA_DOF];
    ra_fk(state->q, state->gripper_width, gripper_links, origins, axes, NULL);
    RaQuat hand_rotation = gripper_links[RA_DOF + 1].rotation;
    RaVec3 reach_origin = state->basketball_mode
        ? ra_gctr(state->end_effector, hand_rotation)
        : state->end_effector;
    ra_obs3(observation, &index,
        ra_sub(state->cube_position, reach_origin), RA_OBS_POS_SCALE);
    ra_obs3(observation, &index,
        ra_sub(state->target_position, state->cube_position),
        RA_OBS_POS_SCALE);
    ra_obs3(observation, &index, state->cube_velocity, RA_OBS_LIN_VEL_SCALE);
    ra_obs3(observation, &index, state->end_effector, RA_OBS_POS_SCALE);
    RaQuat gripper_in_cube = state->basketball_mode
        ? hand_rotation
        : ra_qmul(ra_qconj(state->cube_rotation), hand_rotation);
    ra_obs_xyz(observation, &index,
        ra_rotate(gripper_in_cube, ra_v3(0, 1, 0)));
    ra_obs_xyz(observation, &index,
        ra_rotate(gripper_in_cube, ra_v3(0, 0, 1)));
    ra_obs3(observation, &index, state->cube_angular_velocity,
        RA_OBS_ANG_VEL_SCALE);
    float grip_vel = ra_clamp(
        RA_OBS_GRIP_VEL_SCALE * state->gripper_velocity, -1.0f, 1.0f);
    if (state->stack_mode) {
        ra_obs_xyz(observation, &index,
            ra_rotate(state->base_cube_rotation, ra_v3(1, 0, 0)));
        ra_obs_xyz(observation, &index,
            ra_rotate(state->base_cube_rotation, ra_v3(0, 1, 0)));
        ra_obs3(observation, &index, state->base_cube_velocity,
            RA_OBS_LIN_VEL_SCALE);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[0]);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[1]);
        observation[index++] = ra_clamp(
            RA_OBS_ANG_VEL_SCALE * ra_length(state->base_cube_angular_velocity),
            0.0f, 1.0f);
    } else if (state->basketball_mode) {
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[0]);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[1]);
        observation[index++] = grip_vel;
        ra_obs3(observation, &index, state->cube_position, RA_OBS_POS_SCALE);
        ra_obs3(observation, &index, ra_rotate(
            ra_qconj(hand_rotation), state->cube_velocity),
            RA_OBS_LIN_VEL_SCALE);
        ra_obs3(observation, &index, ra_ptvel(
            state->qd, origins, axes, RA_DOF - 1, state->end_effector),
            RA_OBS_LIN_VEL_SCALE);
    } else {
        ra_obs3(observation, &index, state->target_position, RA_OBS_POS_SCALE);
        ra_obs_xyz(observation, &index,
            ra_rotate(state->cube_rotation, ra_v3(0, 1, 0)));
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[0]);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[1]);
        ra_obs_xyz(observation, &index,
            ra_rotate(hand_rotation, ra_v3(0, 0, 1)));
        observation[index++] = grip_vel;
    }
    observation[index++] = ra_clamp(
        state->gripper_width / 0.08f, 0.0f, 1.0f);
    observation[index++] = ra_clamp(
        state->gripper_force / RA_GRIPPER_MAX_FORCE, 0.0f, 1.0f);
    ra_obs3(observation, &index, ra_rotate(
        ra_qconj(hand_rotation),
        ra_scale(state->wrist_linear_impulse, 1.0f / RA_CONTROL_DT)), 0.01f);
    ra_obs3(observation, &index, ra_rotate(
        ra_qconj(hand_rotation),
        ra_scale(state->wrist_angular_impulse, 1.0f / RA_CONTROL_DT)), 0.20f);
    observation[index++] = state->transported ? 1.0f : 0.0f;
    observation[index++] = state->basketball_mode
        ? (state->basketball_close_ready ? 1.0f : 0.0f)
        : (state->stack_aligned ? 1.0f : 0.0f);
    observation[index++] = state->basketball_mode
        ? (state->basketball_in_flight ? 1.0f : 0.0f)
        : (state->released_near_target ? 1.0f : 0.0f);
    observation[index++] = state->grasped ? 1.0f : 0.0f;
    observation[index++] = state->lifted ? 1.0f : 0.0f;
    int maximum_steps = state->basketball_mode
        ? RA_BASKETBALL_MAX_STEPS : RA_MAX_STEPS;
    observation[index++] = ra_clamp(
        (float)state->step / (float)maximum_steps, 0.0f, 1.0f);
    assert(index == OBS_SIZE);
}

RA_HD static void ra_resetb(RaState* state) {
    state->cube_position = ra_v3(
        ra_rand(&state->rng, 0.42f, 0.54f),
        RA_TABLE_TOP + RA_BALL_RADIUS,
        ra_rand(&state->rng, 0.20f, 0.32f));
    state->cube_velocity = ra_v3(0, 0, 0);
    state->cube_rotation = ra_quat(0, 0, 0, 1);
    state->cube_angular_velocity = ra_v3(0, 0, 0);
    state->previous_cube_position = state->cube_position;
    state->target_position = ra_hoop();
    state->basketball_in_flight = 0;
    state->basketball_grounded_steps = 0;
    state->grasped = 0;
    state->grasp_cooldown = 0;
    state->grasp_contact_misses = 0;
    state->ever_grasped = 0;
    state->lifted = 0;
    state->transported = 0;
    state->released_near_target = 0;
    state->placement_settle_steps = 0;
    state->basketball_close_ready = 0;
    state->basketball_release_ready = 0;
    state->basketball_release_commanded = 0;
    state->gripper_force = 0.0f;
    memset(state->pad_normal_impulse, 0,
        sizeof(state->pad_normal_impulse));
    RaPose links[RA_LINKS];
    ra_fk(state->q, state->gripper_width, links, NULL, NULL, NULL);
    RaVec3 grasp_center = ra_gctr(
        state->end_effector, links[RA_DOF + 1].rotation);
    state->previous_reach_distance = ra_length(
        ra_sub(state->cube_position, grasp_center));
    state->previous_place_distance = ra_length(
        ra_sub(state->target_position, state->cube_position));
    state->previous_lift_height = 0.0f;
    state->previous_grip_error = fabsf(
        state->gripper_width - RA_BASKETBALL_OPEN_WIDTH);
    float launch_quality = ra_blq(
        state->cube_position, state->cube_velocity);
    float trajectory_quality = ra_btq(
        state->cube_position, state->cube_velocity);
    state->previous_throw_quality = 0.35f*launch_quality
        + 0.65f*trajectory_quality;
}

RA_HD static void ra_reset(RaState* state) {
    uint32_t rng = state->rng ? state->rng : 1u;
    int no_timeout = state->no_timeout;
    int stack_mode = state->stack_mode;
    int basketball_mode = state->basketball_mode;
    memset(state, 0, sizeof(*state));
    state->rng = rng;
    state->no_timeout = no_timeout;
    state->stack_mode = stack_mode;
    state->basketball_mode = basketball_mode;
    state->cube_rotation = ra_quat(0, 0, 0, 1);
    state->base_cube_rotation = ra_quat(0, 0, 0, 1);
    for (int joint = 0; joint < RA_DOF; ++joint) {
        state->q[joint] = ra_jhome(joint)
            + ra_rand(&state->rng, -0.035f, 0.035f);
        state->target_q[joint] = state->q[joint];
    }
    state->gripper_width = 0.080f;
    RaPose links[RA_LINKS];
    ra_fk(state->q, state->gripper_width, links, NULL, NULL,
        &state->end_effector);

    if (state->basketball_mode) {
        ra_resetb(state);
        return;
    }

    float cube_angle = ra_rand(&state->rng, -0.72f, -0.28f);
    float cube_radius = ra_rand(&state->rng, 0.43f, 0.62f);
    state->cube_position = ra_v3(cube_radius*cosf(cube_angle),
        RA_TABLE_TOP + RA_CUBE_HALF, -cube_radius*sinf(cube_angle));
    float target_angle = ra_rand(&state->rng, 0.28f, 0.72f);
    float target_radius = ra_rand(&state->rng, 0.43f, 0.62f);
    if (state->stack_mode) {
        state->base_cube_position = ra_v3(target_radius*cosf(target_angle),
            RA_TABLE_TOP + RA_CUBE_HALF, -target_radius*sinf(target_angle));
        state->base_cube_start_position = state->base_cube_position;
        state->previous_base_cube_position = state->base_cube_position;
        state->target_position = ra_add(state->base_cube_position,
            ra_v3(0, 2.0f * RA_CUBE_HALF, 0));
    } else {
        state->target_position = ra_v3(target_radius*cosf(target_angle),
            RA_TABLE_TOP + 0.008f, -target_radius*sinf(target_angle));
    }
    state->previous_reach_distance = ra_length(
        ra_sub(state->cube_position, state->end_effector));
    state->previous_place_distance = ra_length(
        ra_sub(state->target_position, state->cube_position));
    state->previous_lift_height = 0.0f;
    RaVec3 stack_delta = ra_sub(
        state->cube_position, state->base_cube_position);
    state->previous_stack_horizontal = sqrtf(
        stack_delta.x*stack_delta.x + stack_delta.z*stack_delta.z);
    float stack_clearance = stack_delta.y - 2.0f*RA_CUBE_HALF;
    state->previous_stack_drop_error = fabsf(
        stack_clearance - RA_STACK_HOVER_CLEARANCE);
    state->previous_stack_orientation_error = 0.0f;
}

RA_D static float ra_stepb(RaState* state,
        const float* actions, float energy, int first_grasp, int released,
        const RaPose* links) {
    RaVec3 hoop = ra_hoop();
    RaVec3 grasp_center = ra_gctr(
        state->end_effector, links[RA_DOF + 1].rotation);
    float reach_distance = ra_length(
        ra_sub(state->cube_position, grasp_center));
    float hoop_distance = ra_length(ra_sub(hoop, state->cube_position));
    float lift_height = ra_max(0.0f,
        state->cube_position.y - RA_BALL_RADIUS - RA_TABLE_TOP);
    float reward = -0.0001f;

    if (!state->grasped && !state->basketball_in_flight) {
        reward += 0.08f * ra_clamp(
            state->previous_reach_distance - reach_distance,
            -0.05f, 0.05f);
    } else if (state->grasped && !state->lifted) {
        reward += 0.08f * ra_clamp(
            lift_height - state->previous_lift_height, -0.03f, 0.03f);
    } else if (state->grasped) {
        reward += 0.04f * ra_clamp(
            state->previous_place_distance - hoop_distance,
            -0.05f, 0.05f);
    } else if (state->basketball_in_flight) {
        reward += 0.02f * ra_clamp(
            state->previous_place_distance - hoop_distance,
            -0.05f, 0.05f);
    }
    int open_enough = state->gripper_width > 0.062f;
    int entered_close_phase = !state->basketball_close_ready
        && !state->grasped && !state->basketball_in_flight
        && open_enough && reach_distance < 0.045f;
    if (entered_close_phase) {
        state->basketball_close_ready = 1;
        state->previous_grip_error = fabsf(
            state->gripper_width - RA_BASKETBALL_GRIP_WIDTH);
    }
    float target_width = state->basketball_close_ready
        ? RA_BASKETBALL_GRIP_WIDTH : RA_BASKETBALL_OPEN_WIDTH;
    float grip_error = fabsf(state->gripper_width - target_width);
    if (!state->grasped && !state->basketball_in_flight
            && !entered_close_phase) {
        reward += 0.15f * ra_clamp(
            state->previous_grip_error - grip_error, -0.02f, 0.02f);
    }
    if (first_grasp) {
        state->basketball_grasps += 1;
        reward += 0.050f;
    }
    if (!state->lifted && state->grasped && lift_height >= RA_LIFT_HEIGHT) {
        state->lifted = 1;
        reward += 0.025f;
    }
    if (!state->transported && state->grasped && state->lifted
            && hoop_distance < RA_BASKETBALL_RELEASE_DISTANCE) {
        state->transported = 1;
        reward += 0.020f;
    }
    float launch_quality = ra_blq(
        state->cube_position, state->cube_velocity);
    float trajectory_quality = ra_btq(
        state->cube_position, state->cube_velocity);
    float release_quality = 0.35f*launch_quality
        + 0.65f*trajectory_quality;
    if (state->grasped && state->lifted) {
        reward += 0.040f * ra_clamp(
            release_quality - state->previous_throw_quality, -0.05f, 0.05f);
        if (!state->basketball_release_ready
                && release_quality >= RA_BASKETBALL_RELEASE_READY_QUALITY) {
            state->basketball_release_ready = 1;
        }
        if (state->basketball_release_ready) {
            reward -= 0.0005f;
            if (actions[RA_DOF] > 0.25f) {
                reward += 0.010f*release_quality;
            }
        }
    }
    int opening_for_release = state->lifted && state->ever_grasped
        && !state->basketball_in_flight && actions[RA_DOF] > 0.25f
        && (state->grasped || released);
    if (opening_for_release && !state->basketball_release_commanded) {
        state->basketball_release_commanded = 1;
        reward += 0.015f + 0.015f*release_quality;
    }
    int thrown = state->lifted && !state->grasped
        && state->ever_grasped && released;
    if (!state->basketball_in_flight && thrown) {
        state->basketball_in_flight = 1;
        state->basketball_releases += 1;
        RaVec3 predicted_crossing;
        float center_miss = RA_BASKETBALL_PREDICTED_MISS_CAP;
        if (ra_bxing(
                state->cube_position, state->cube_velocity,
                &predicted_crossing, NULL, NULL)) {
            float dx = predicted_crossing.x - hoop.x;
            float dz = predicted_crossing.z - hoop.z;
            center_miss = ra_min(sqrtf(dx*dx + dz*dz),
                RA_BASKETBALL_PREDICTED_MISS_CAP);
        }
        state->basketball_release_center_miss_cm_sum += 100.0f*center_miss;
        state->basketball_release_ready = 0;
        reward += 0.020f*launch_quality
            + 0.100f*trajectory_quality;
    }

    int crossed_down = state->basketball_in_flight
        && state->previous_cube_position.y > hoop.y
        && state->cube_position.y <= hoop.y;
    int scored = 0;
    if (crossed_down) {
        float height_delta = state->previous_cube_position.y
            - state->cube_position.y;
        float fraction = height_delta > 1.0e-7f
            ? (state->previous_cube_position.y - hoop.y) / height_delta
            : 0.0f;
        RaVec3 crossing = ra_add(state->previous_cube_position,
            ra_scale(ra_sub(state->cube_position,
                state->previous_cube_position), fraction));
        float offset_x = crossing.x - hoop.x;
        float offset_z = crossing.z - hoop.z;
        float clearance = RA_HOOP_INNER_RADIUS - RA_BALL_RADIUS;
        scored = offset_x*offset_x + offset_z*offset_z
            < clearance*clearance;
    }
    int grounded = !state->grasped
        && state->cube_position.y <= RA_TABLE_TOP + RA_BALL_RADIUS
            + RA_BASKETBALL_GROUNDED_HEIGHT_SLOP
        && fabsf(state->cube_velocity.y)
            <= RA_BASKETBALL_GROUNDED_MAX_VERTICAL_SPEED;
    float ball_base_distance = ra_length(state->cube_position);
    int out_of_reach = ball_base_distance - RA_BALL_RADIUS
        > RA_ARM_GEOMETRIC_REACH_BOUND;
    if (grounded && out_of_reach) {
        state->basketball_grounded_steps += 1;
    } else {
        state->basketball_grounded_steps = 0;
    }
    int grounded_reset = state->basketball_grounded_steps
        >= RA_BASKETBALL_GROUNDED_RESET_STEPS;
    if (scored) {
        state->baskets += 1;
        state->attempts += 1;
        state->success = 1;
        reward = 1.0f;
        state->basketball_in_flight = 0;
    } else if (grounded && state->basketball_in_flight) {
        state->attempts += 1;
        reward = -0.010f;
        state->basketball_in_flight = 0;
        state->basketball_release_ready = 0;
    }
    if (grounded_reset) {
        ra_resetb(state);
        state->basketball_reset = 1;
    }

    if (!state->no_timeout && state->step >= RA_BASKETBALL_MAX_STEPS) {
        state->done = 1;
    }
    if (!state->basketball_reset) {
        state->previous_reach_distance = reach_distance;
        state->previous_place_distance = hoop_distance;
        state->previous_lift_height = lift_height;
        state->previous_grip_error = grip_error;
        state->previous_throw_quality = release_quality;
    }
    state->episode_energy += energy;
    state->episode_return += reward;
    for (int action = 0; action < RA_ACTIONS; ++action) {
        state->previous_action[action] = ra_clamp(
            actions[action], -1.0f, 1.0f);
    }
    return reward;
}

RA_D static float ra_stept(RaState* state, const float* actions,
        float energy, int first_grasp, int released, const RaPose* links) {
    float grip_action = ra_clamp(actions[RA_DOF], -1.0f, 1.0f);
    float reach_distance = ra_length(
        ra_sub(state->cube_position, state->end_effector));
    RaQuat hand_rotation = links[RA_DOF + 1].rotation;
    RaVec3 hand_position = ra_sub(state->end_effector,
        ra_rotate(hand_rotation, ra_v3(0, 0, 0.115f)));
    float half_width = 0.5f * state->gripper_width;
    RaPose fingers[2];
    fingers[0].position = ra_add(hand_position,
        ra_rotate(hand_rotation, ra_v3(0, half_width, 0.0584f)));
    fingers[0].rotation = hand_rotation;
    fingers[1].position = ra_add(hand_position,
        ra_rotate(hand_rotation, ra_v3(0, -half_width, 0.0584f)));
    fingers[1].rotation = ra_qnorm(ra_qmul(hand_rotation,
        ra_qaxis(ra_v3(0, 0, 1), 3.14159265359f)));
    RaConvexContact clear_contact;
    int gripper_clear = 1;
    for (int finger = 0; finger < 2; ++finger) {
        gripper_clear &= !ra_padhit(
            state->cube_position, state->cube_rotation,
            fingers[finger], RA_GRIPPER_CLEARANCE_MARGIN, &clear_contact);
        if (state->stack_mode) {
            gripper_clear &= !ra_padhit(
                state->base_cube_position, state->base_cube_rotation,
                fingers[finger], RA_GRIPPER_CLEARANCE_MARGIN, &clear_contact);
        }
    }
    RaVec3 place_offset = ra_sub(
        state->target_position, state->cube_position);
    float place_distance = ra_length(place_offset);
    float place_horizontal_distance = sqrtf(
        place_offset.x*place_offset.x + place_offset.z*place_offset.z);
    float main_support_y = ra_csup(
        state->cube_rotation, ra_v3(0, 1, 0));
    float base_support_y = ra_csup(
        state->base_cube_rotation, ra_v3(0, 1, 0));
    float expected_stack_separation = main_support_y + base_support_y;
    float stack_height_error = fabsf(
        (state->cube_position.y - state->base_cube_position.y)
            - expected_stack_separation);
    float stack_clearance = (state->cube_position.y
        - state->base_cube_position.y) - expected_stack_separation;
    float stack_drop_error = fabsf(
        stack_clearance - RA_STACK_HOVER_CLEARANCE);
    float stack_orientation_error = ra_cup(
        state->cube_rotation);
    RaVec3 base_motion_delta = ra_sub(
        state->base_cube_position, state->previous_base_cube_position);
    float base_motion = sqrtf(base_motion_delta.x*base_motion_delta.x
        + base_motion_delta.z*base_motion_delta.z);
    int stack_release_pose = place_horizontal_distance
            < RA_STACK_RELEASE_RADIUS
        && stack_clearance >= -RA_STACK_HEIGHT_TOLERANCE
        && stack_clearance < RA_STACK_RELEASE_CLEARANCE;
    int stack_alignment_pose = place_horizontal_distance < 0.050f
        && stack_clearance >= -RA_STACK_HEIGHT_TOLERANCE
        && stack_clearance < 0.080f;
    float lift_height = ra_max(0.0f,
        state->cube_position.y - main_support_y - RA_TABLE_TOP);
    int stack_contact = state->stack_mode && !state->grasped
        && place_horizontal_distance < 2.0f*RA_CUBE_HALF
        && stack_height_error < RA_STACK_HEIGHT_TOLERANCE;
    if (stack_contact) {
        state->ever_stacked = 1;
    }
    int placement_disturbed = state->released_near_target
        && (state->grasped
            || place_horizontal_distance >= (state->stack_mode
                ? RA_STACK_RELEASE_RADIUS : RA_PLACE_RADIUS));
    if (placement_disturbed) {
        state->released_near_target = 0;
        state->placement_settle_steps = 0;
    }
    float reward = -0.002f;
    if (!state->grasped) {
        if (state->released_near_target) {
            reward += (state->stack_mode ? 8.0f : 1.8f) * ra_clamp(
                reach_distance - state->previous_reach_distance,
                -0.05f, 0.05f);
        } else {
            reward += 1.8f * ra_clamp(
                state->previous_reach_distance - reach_distance,
                -0.05f, 0.05f);
        }
    } else if (!state->lifted) {
        reward += (state->stack_mode ? 12.0f : 5.0f) * ra_clamp(
            lift_height - state->previous_lift_height, -0.03f, 0.03f);
        if (lift_height >= RA_LIFT_HEIGHT) {
            state->lifted = 1;
            reward += state->stack_mode ? 2.0f : 0.75f;
        }
    }
    if (state->grasped && state->lifted) {
        if (state->stack_mode) {
            reward += RA_STACK_HORIZONTAL_PROGRESS_REWARD * ra_clamp(
                state->previous_stack_horizontal
                    - place_horizontal_distance,
                -0.05f, 0.05f);
            if (place_horizontal_distance < 0.12f) {
                reward += RA_STACK_HEIGHT_PROGRESS_REWARD * ra_clamp(
                    state->previous_stack_drop_error - stack_drop_error,
                    -0.04f, 0.04f);
                reward += RA_STACK_UPRIGHT_PROGRESS_REWARD * ra_clamp(
                    state->previous_stack_orientation_error
                        - stack_orientation_error,
                    -0.05f, 0.05f);
            }
        } else {
            reward += 6.0f * ra_clamp(
                state->previous_place_distance - place_distance,
                -0.05f, 0.05f);
        }
    }
    if (first_grasp) {
        reward += state->stack_mode ? 0.40f : 0.5f;
    }
    if (released) {
        if (state->stack_mode) {
            if (grip_action <= 0.25f) {
                float slip_penalty = state->stack_aligned ? 2.00f
                    : (state->transported ? 1.00f
                    : (state->lifted ? RA_STACK_SLIP_PENALTY : 0.10f));
                reward -= slip_penalty;
            } else if (!state->transported) {
                reward -= state->lifted ? 0.75f : 0.15f;
            }
        } else if (!state->transported) {
            reward -= state->lifted ? 0.50f : 0.10f;
        }
    }
    if (placement_disturbed) {
        reward -= 0.25f;
    }
    int transport_pose = state->stack_mode
        ? (place_horizontal_distance < RA_STACK_TRANSPORT_RADIUS
            && stack_clearance >= -RA_STACK_HEIGHT_TOLERANCE
            && stack_clearance < 0.10f)
        : place_distance < 0.12f;
    if (!state->transported && state->grasped && state->lifted
            && lift_height >= RA_CARRY_HEIGHT && transport_pose) {
        state->transported = 1;
        reward += state->stack_mode ? 2.0f : 0.5f;
    }
    int stack_release_ready = stack_release_pose
        && stack_orientation_error < 0.050f
        && ra_length(state->cube_velocity) < 0.50f
        && ra_length(state->cube_angular_velocity) < 2.0f;
    if (state->stack_mode && state->transported && state->grasped
            && stack_release_ready && !state->stack_aligned) {
        state->stack_aligned = 1;
        reward += 2.0f;
    }
    if (state->stack_mode && state->transported && state->stack_aligned
            && state->grasped) {
        reward -= 0.030f;
        if (!state->stack_opening_credited && grip_action > 0.25f) {
            state->stack_opening_credited = 1;
            if (stack_release_ready) {
                reward += 1.000f;
            } else if (stack_alignment_pose) {
                reward += 0.200f;
            }
        }
    }
    int valid_release = released && (state->stack_mode
        ? (grip_action > 0.25f && state->stack_aligned
            && stack_alignment_pose)
        : place_distance < 0.12f);
    int first_valid_release = !state->valid_release_achieved
        && state->transported && valid_release;
    if (!state->released_near_target && state->transported && valid_release) {
        state->released_near_target = 1;
        state->valid_release_achieved = 1;
        if (state->stack_mode && first_valid_release) {
            float release_quality_penalty = ra_clamp(
                2.0f*ra_length(state->cube_velocity)
                    + 0.5f*ra_length(state->cube_angular_velocity)
                    + 20.0f*stack_orientation_error,
                0.0f, 4.0f);
            reward += 6.0f - release_quality_penalty;
        } else if (!state->stack_mode) {
            reward += 0.25f;
        }
    }
    if (state->stack_mode && state->transported && released
            && !valid_release && grip_action > 0.25f) {
        reward -= 2.0f;
    }
    if (state->stack_mode && stack_contact
            && state->valid_release_achieved
            && !state->valid_stack_contact) {
        state->valid_stack_contact = 1;
        reward += 4.0f;
    }
    if (state->stack_mode && state->released_near_target
            && gripper_clear
            && !state->cleared_after_release) {
        state->cleared_after_release = 1;
        reward += 4.0f;
    }
    if (state->stack_mode && state->released_near_target
            && place_horizontal_distance < 0.040f
            && stack_height_error < 0.010f) {
        if (stack_orientation_error < RA_STACK_UPRIGHT_ERROR) {
            reward += 0.020f;
        }
        if (ra_length(state->cube_velocity) < RA_STACK_SETTLE_SPEED
                && ra_length(state->base_cube_velocity)
                    < RA_STACK_SETTLE_SPEED) {
            reward += 0.020f;
        }
        if (ra_length(state->cube_angular_velocity)
                < RA_STACK_SETTLE_ANGULAR_SPEED
                && ra_length(state->base_cube_angular_velocity)
                    < RA_STACK_SETTLE_ANGULAR_SPEED) {
            reward += 0.020f;
        }
        if (gripper_clear) {
            reward += 0.030f;
        }
    }
    if (state->stack_mode) {
        reward -= 8.0f * ra_min(base_motion, 0.03f);
    }
    float action_cost = 0.0f;
    float action_delta = 0.0f;
    for (int action = 0; action < RA_DOF; ++action) {
        float value = ra_clamp(actions[action], -1.0f, 1.0f);
        action_cost += value * value;
        float delta = value - state->previous_action[action];
        action_delta += delta * delta;
    }

    int placement_stable;
    int settle_steps_required;
    if (state->stack_mode) {
        placement_stable = !state->grasped
            && state->released_near_target
            && state->lifted && state->transported
            && place_horizontal_distance < RA_STACK_ALIGN_RADIUS
            && stack_height_error < RA_STACK_HEIGHT_TOLERANCE
            && ra_cup(state->cube_rotation)
                < RA_STACK_UPRIGHT_ERROR
            && ra_cup(state->base_cube_rotation)
                < RA_STACK_UPRIGHT_ERROR
            && state->base_cube_position.y
                <= RA_TABLE_TOP + base_support_y + 0.004f
            && ra_length(state->cube_velocity) < RA_STACK_SETTLE_SPEED
            && ra_length(state->base_cube_velocity) < RA_STACK_SETTLE_SPEED
            && ra_length(state->cube_angular_velocity)
                < RA_STACK_SETTLE_ANGULAR_SPEED
            && ra_length(state->base_cube_angular_velocity)
                < RA_STACK_SETTLE_ANGULAR_SPEED
            && gripper_clear;
        settle_steps_required = RA_STACK_SETTLE_STEPS;
    } else {
        placement_stable = !state->grasped
            && state->released_near_target
            && state->lifted && state->transported
            && place_horizontal_distance < RA_PLACE_RADIUS
            && state->cube_position.y
                <= RA_TABLE_TOP + main_support_y + 0.008f
            && ra_length(state->cube_velocity) < RA_PLACE_SETTLE_SPEED
            && ra_length(state->cube_angular_velocity)
                < RA_PLACE_SETTLE_ANGULAR_SPEED
            && reach_distance > RA_PLACE_CLEARANCE;
        settle_steps_required = RA_PLACE_SETTLE_STEPS;
    }
    if (placement_stable) {
        state->placement_settle_steps += 1;
        if (state->placement_settle_steps
                > state->max_placement_settle_steps) {
            state->max_placement_settle_steps
                = state->placement_settle_steps;
        }
        reward += state->stack_mode ? 0.12f : 0.01f;
    } else {
        state->placement_settle_steps = 0;
    }
    if (state->placement_settle_steps >= settle_steps_required) {
        state->success = 1;
        state->done = 1;
        reward += state->stack_mode ? 20.0f : 10.0f;
    }
    if (!state->no_timeout && state->step >= RA_MAX_STEPS) {
        state->done = 1;
    }
    if (state->cube_position.y < -0.25f) {
        state->done = 1;
        reward -= 0.25f;
    }
    if (state->stack_mode && state->base_cube_position.y < -0.25f) {
        state->done = 1;
        reward -= 0.25f;
    }

    reward *= state->stack_mode ? RA_STACK_REWARD_SCALE : RA_PICK_REWARD_SCALE;
    reward -= 0.001f * action_cost + 0.0004f * action_delta;
    reward -= 0.00002f * energy;

    state->previous_reach_distance = reach_distance;
    state->previous_place_distance = place_distance;
    state->previous_lift_height = lift_height;
    state->previous_stack_horizontal = place_horizontal_distance;
    state->previous_stack_drop_error = stack_drop_error;
    state->previous_stack_orientation_error = stack_orientation_error;
    state->previous_base_cube_position = state->base_cube_position;
    state->episode_energy += energy;
    state->episode_return += reward;
    for (int action = 0; action < RA_ACTIONS; ++action) {
        state->previous_action[action] = ra_clamp(actions[action], -1.0f, 1.0f);
    }
    return reward;
}

RA_HD static RA_INLINE void ra_rbrst(
        RaCudaRigidWorld* world, unsigned int topology) {
    memset(world, 0, sizeof(*world));
    world->topology = topology;
    world->config = (PlImpulseConfig){
        .velocity_iterations = RA_CONTACT_VELOCITY_ITERS,
        .position_iterations = RA_CONTACT_POSITION_ITERS,
        .velocity_impulse_tolerance = 1.0e-7f,
        .position_beta = 0.80f,
        .slop = RA_CUDA_CONTACT_SLOP,
        .speculative_margin = RA_CONTACT_MARGIN,
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
    if (world->manifold_count >= PL_IMPULSE_MAX_MANIFOLDS) {
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
        RaWorld* world) {
    RaState* state = &world->state;
    RaCudaRigidWorld* rigid = &world->rigid;
    unsigned int topology = state->basketball_mode ? 3u
        : (state->stack_mode ? 2u : 1u);
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
            box.pose, box.half_extents, 0.0f, velocity, hand_angular);
    }
    for (int item = 0; item < RA_DOF; ++item) {
        RaCollisionBox box = ra_linkb(links, item);
        RaVec3 velocity = ra_ptvel(
            state->qd, origins, axes, item, box.pose.position);
        RaVec3 angular = ra_angvel(
            state->qd, axes, item);
        ra_setbox(rigid, RA_CUDA_BODY_LINK_START + item,
            box.pose, box.half_extents, 0.0f, velocity, angular);
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
        RaWorld* world,
        const float mass_factor[RA_DOF][RA_DOF]) {
    RaState* state = &world->state;
    RaCudaRigidWorld* rigid = &world->rigid;
    const RaPose* links = world->staged.links;
    const RaVec3* origins = world->staged.origins;
    const RaVec3* axes = world->staged.axes;
    const int manifold_count = rigid->manifold_count;
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
                    float gripper_jacobian = 0.5f * ra_dot(outward,
                        directions[direction_index]);
                    reaction->gripper_jacobian[direction_index] = gripper_jacobian;
                    reaction->inverse_mass[direction_index] +=
                        gripper_jacobian * gripper_jacobian
                        / RA_GRIPPER_EFFECTIVE_MASS;
                    reaction->gripper_velocity_response[direction_index]
                        = -gripper_jacobian / RA_GRIPPER_EFFECTIVE_MASS;
                }
            }
            reaction->active = 1;
            if (side >= 0 && point->patch_group != 0
                    && point->patch.area > 0.0f) {
                int group_points = 0;
                for (int member = 0; member < manifold->point_count;
                        ++member) {
                    if (manifold->points[member].patch_group
                            == point->patch_group) {
                        group_points++;
                    }
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
        RaWorld* world, int side) {
    RaCudaRigidWorld* rigid = &world->rigid;
    int pad_start = side == 0 ? RA_CUDA_BODY_PAD_LEFT_START
        : RA_CUDA_BODY_PAD_RIGHT_START;
    RaVec3 inward = ra_scale(ra_rotate(
        rigid->shapes[pad_start].pose.rotation, ra_v3(0, 1, 0)), -1.0f);
    PlImpulseCandidate best;
    memset(&best, 0, sizeof(best));
    float best_separation = 3.402823466e+38f;
    int best_pad = -1;
    for (int pad = 0; pad < RA_PAD_BOXES; ++pad) {
        int body = pad_start + pad;
        PlSatQuery query = pl_sq(
            &rigid->shapes[RA_CUDA_BODY_CUBE], &rigid->shapes[body],
            RA_CONTACT_MARGIN);
        if (ra_dot(query.contact.normal, inward) < 0.45f) {
            continue;
        }
        const RaRigidBody* ball = &rigid->bodies[RA_CUDA_BODY_CUBE];
        const RaRigidBody* pad_body = &rigid->bodies[body];
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
    if (rigid->manifold_count >= PL_IMPULSE_MAX_MANIFOLDS) {
        return 0;
    }
    int body = pad_start + best_pad;
    PlImpulseManifold* manifold = &rigid->manifolds[rigid->manifold_count];
    int count = pl_iman(RA_CUDA_BODY_CUBE, body,
        &best, 1, RA_CONTACT_MARGIN, RA_ROBOT_FRICTION,
        RA_ROBOT_FRICTION, 0.0f, manifold);
    if (count <= 0) {
        return 0;
    }
    rigid->compound_pad_component_mask[side] = 1u << best_pad;
    rigid->manifold_count += 1;
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
            intersection.point = ra_add(previous.point,
                ra_scale(ra_sub(current.point, previous.point), fraction));
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
        RaWorld* world, int side, int object_body) {
    RaCudaRigidWorld* rigid = &world->rigid;
    if (rigid->manifold_count >= PL_IMPULSE_MAX_MANIFOLDS) {
        return 0;
    }
    int pad_start = side == 0 ? RA_CUDA_BODY_PAD_LEFT_START
        : RA_CUDA_BODY_PAD_RIGHT_START;
    float margin = RA_CONTACT_MARGIN;
    float static_friction = RA_ROBOT_FRICTION;
    float dynamic_friction = RA_ROBOT_FRICTION;
    float restitution = 0.0f;
    PlImpulseCandidate* candidates = rigid->compound_candidate_scratch;
    int candidate_count = 0;
    RaVec3 inward = ra_scale(ra_rotate(
        rigid->shapes[pad_start].pose.rotation, ra_v3(0, 1, 0)), -1.0f);
    const RaConvexShape* object_shape = &rigid->shapes[object_body];
    const RaRigidBody* object_state = &rigid->bodies[object_body];
    PlSatObb object_obb = pl_sobb(object_shape);
    RaVec3 object_axes[3];
    ra_caxes(object_shape->pose.rotation, object_axes);
    const RaVec3 object_extents = object_shape->half_extents;
    const RaVec3 frame_origin = rigid->shapes[pad_start].pose.position;
    RaVec3 pad_axes[3];
    ra_caxes(rigid->shapes[pad_start].pose.rotation, pad_axes);
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
        const RaConvexShape* pad_shape = &rigid->shapes[pad_start + pad];
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
            const RaConvexShape* pad_shape = &rigid->shapes[body];
        PlSatQuery sat_query = pl_sq(
            object_shape, pad_shape, margin);

        RaVec3 delta = ra_sub(object_shape->pose.position,
            pad_shape->pose.position);
        if (ra_dot(delta, inward) < 0.0f) {
            continue;
        }
        RaVec3 object_surface = pl_ssup(&object_obb,
            ra_scale(inward, -1.0f));
        float face_separation = ra_dot(
            ra_sub(object_surface, inner_surface[pad]), inward);
        const RaRigidBody* pad_body_state = &rigid->bodies[body];
        float angular_bound = ra_length(object_state->angular_velocity)
                * ra_brad(object_shape)
            + ra_length(pad_body_state->angular_velocity)
                * ra_brad(pad_shape);
        RaVec3 sat_normal = pl_inrm(
            sat_query.contact.normal, ra_sub(
                sat_query.contact.point_a, sat_query.contact.point_b));
        RaVec3 sat_velocity_a = ra_add(object_state->linear_velocity,
            ra_cross(object_state->angular_velocity,
                ra_sub(sat_query.contact.point_a,
                    object_state->pose.position)));
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
                && sat_projected_separation > margin) {
            continue;
        }
        float object_radius_x = ra_oboxr(
            object_axes, object_extents, pad_axes[0]);
        float object_radius_z = ra_oboxr(
            object_axes, object_extents, pad_axes[2]);
        float local_x = ra_dot(delta, pad_axes[0]);
        float local_z = ra_dot(delta, pad_axes[2]);
        if (fabsf(local_x) > pad_shape->half_extents.x
                + object_radius_x + margin
            || fabsf(local_z) > pad_shape->half_extents.z
                + object_radius_z + margin) {
            continue;
        }

        RaVec3 object_velocity = ra_add(object_state->linear_velocity,
            ra_cross(object_state->angular_velocity,
                ra_sub(object_surface, object_state->pose.position)));
        RaVec3 pad_velocity = ra_add(pad_body_state->linear_velocity,
            ra_cross(pad_body_state->angular_velocity,
                ra_sub(inner_surface[pad], pad_body_state->pose.position)));
        float normal_velocity = ra_dot(
            ra_sub(object_velocity, pad_velocity), inward);
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

    int incident_axis = pl_saxis(&object_obb,
        ra_scale(inward, -1.0f));
    float incident_sign = ra_dot(object_obb.axis[incident_axis],
        ra_scale(inward, -1.0f)) < 0.0f ? -1.0f : 1.0f;
    RaVec3 incident_face[4];
    pl_sface(&object_obb, incident_axis, incident_sign,
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
                    <= RA_PAD_CSG_BOUNDARY_EPSILON) {
                continue;
            }
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
                        + RA_PAD_CSG_BOUNDARY_EPSILON) {
                    continue;
                }
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
                            + RA_PAD_CSG_BOUNDARY_EPSILON) {
                        continue;
                    }
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
                            + RA_PAD_CSG_BOUNDARY_EPSILON) {
                    continue;
                }
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
    if (rigid->manifold_count >= PL_IMPULSE_MAX_MANIFOLDS) {
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
    if (rigid->manifold_count >= PL_IMPULSE_MAX_MANIFOLDS) {
        return 0;
    }
    int selected[PL_SAT_MAX_MANIFOLD_POINTS];
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
    while (selected_count < PL_SAT_MAX_MANIFOLD_POINTS
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
    PlImpulseCandidate reduced[PL_SAT_MAX_MANIFOLD_POINTS];
    for (int index = 0; index < selected_count; ++index) {
        reduced[index] = candidates[selected[index]];
    }
    PlImpulseManifold* manifold = &rigid->manifolds[
        rigid->manifold_count];
    int made_count = pl_iman(object_body, pad_start, reduced,
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
    rigid->compound_pad_component_mask[side] |= component_mask;
    rigid->manifold_count++;
    return 1;
}

RA_D static RA_INLINE void ra_advbox(
        RaWorld* world, int body_index, int other_index) {
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

RA_D static RA_INLINE float ra_toi(
        RaConvexSweep sweep, RaVec3 velocity, float advance) {
    float normal_speed = ra_dot(velocity, sweep.contact.normal);
    if (sweep.hit && normal_speed < 0.0f && sweep.toi < advance) {
        return sweep.toi;
    }
    return advance;
}

RA_D static RA_INLINE int ra_tbllo(
        RaPose pose, RaVec3 half_extents, float margin) {
    RaVec3 axis_x = ra_rotate(pose.rotation, ra_v3(1, 0, 0));
    RaVec3 axis_y = ra_rotate(pose.rotation, ra_v3(0, 1, 0));
    RaVec3 axis_z = ra_rotate(pose.rotation, ra_v3(0, 0, 1));
    float radius_x = fabsf(axis_x.x)*half_extents.x
        + fabsf(axis_y.x)*half_extents.y
        + fabsf(axis_z.x)*half_extents.z;
    float radius_y = fabsf(axis_x.y)*half_extents.x
        + fabsf(axis_y.y)*half_extents.y
        + fabsf(axis_z.y)*half_extents.z;
    float radius_z = fabsf(axis_x.z)*half_extents.x
        + fabsf(axis_y.z)*half_extents.y
        + fabsf(axis_z.z)*half_extents.z;
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
        RaPose pose, RaVec3 half_extents, float margin) {
    if (!ra_tbllo(pose, half_extents, margin)) {
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
    shape.half_extents = half_extents;
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
                box.pose, box.half_extents, RA_CONTACT_MARGIN)) {
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
                box.pose, box.half_extents, RA_CONTACT_MARGIN)) {
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
        RaWorld* world) {
    RaState* state = &world->state;
    if (state->basketball_mode) {
        RaCudaRigidWorld* rigid = &world->rigid;
        RaRigidBody* ball = &rigid->bodies[RA_CUDA_BODY_CUBE];
        RaVec3 velocity = ball->linear_velocity;
        float advance = RA_PHYSICS_DT;
        advance = ra_toi(ra_boxccd(
            &rigid->shapes[RA_CUDA_BODY_CUBE],
            &rigid->shapes[RA_CUDA_BODY_TABLE],
            velocity, ball->angular_velocity,
            ra_v3(0, 0, 0), ra_v3(0, 0, 0),
            RA_PHYSICS_DT, 0.0f), velocity, advance);
        advance = ra_toi(ra_boxccd(
            &rigid->shapes[RA_CUDA_BODY_CUBE],
            &rigid->shapes[RA_CUDA_BODY_BACKBOARD],
            velocity, ball->angular_velocity,
            ra_v3(0, 0, 0), ra_v3(0, 0, 0),
            RA_PHYSICS_DT, 0.0f), velocity, advance);
        RaConvexSweep sweep;
        memset(&sweep, 0, sizeof(sweep));
        sweep.toi = RA_PHYSICS_DT;
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
                break;
            }
            if (speed <= 1.0e-8f) {
                break;
            }
            float rim_step = sweep.contact.separation / speed;
            if (rim_step <= 1.0e-7f) {
                sweep.hit = 1;
                sweep.toi = time;
                break;
            }
            time += rim_step;
            if (time > RA_PHYSICS_DT) {
                break;
            }
        }
        advance = ra_toi(sweep, velocity, advance);
        ball->pose.position = ra_add(ball->pose.position,
            ra_scale(velocity, advance));
        ball->pose.rotation = ra_qint(
            ball->pose.rotation, ball->angular_velocity, advance);
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
    RaRigidBody* cube = &world->rigid.bodies[RA_CUDA_BODY_CUBE];
    state->cube_position = cube->pose.position;
    state->cube_rotation = cube->pose.rotation;
    state->cube_velocity = cube->linear_velocity;
    state->cube_angular_velocity = cube->angular_velocity;
    if (state->stack_mode) {
        RaRigidBody* base = &world->rigid.bodies[RA_CUDA_BODY_BASE];
        state->base_cube_position = base->pose.position;
        state->base_cube_rotation = base->pose.rotation;
        state->base_cube_velocity = base->linear_velocity;
        state->base_cube_angular_velocity = base->angular_velocity;
    }
}

RA_D static RA_INLINE void ra_prep(
        RaWorld* world,
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
    ra_massg(state, matrix, gravity);
    for (int joint = 0; joint < RA_DOF; ++joint) {
        float kp = joint < 2 ? 4500.0f : (joint < 4 ? 3500.0f : 2000.0f);
        float kd = joint < 2 ? 450.0f : (joint < 4 ? 350.0f : 200.0f);
        float torque_limit = joint < 4 ? 87.0f : 12.0f;
        float arm_motor = ra_clamp(
            kp * (state->target_q[joint] - state->q[joint])
                - kd * state->qd[joint],
            -torque_limit, torque_limit);
        rhs[joint] = arm_motor - state->qd[joint] + gravity[joint];
        world->staged.energy += fabsf(arm_motor * state->qd[joint])
            * RA_PHYSICS_DT;
    }
    for (int row = 0; row < RA_DOF; ++row) {
        for (int column = 0; column <= row; ++column) {
            float sum = matrix[row][column];
            for (int k = 0; k < column; ++k) {
                sum -= mass_factor[row][k] * mass_factor[column][k];
            }
            if (row == column) {
                mass_factor[row][column] = sqrtf(ra_max(sum, 1.0e-8f));
            } else {
                mass_factor[row][column] = sum / mass_factor[column][column];
            }
        }
    }
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
        RaWorld* world) {
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
    state->pad_normal_impulse[0] = 0.0f;
    state->pad_normal_impulse[1] = 0.0f;
    state->wrist_linear_impulse = ra_v3(0, 0, 0);
    state->wrist_angular_impulse = ra_v3(0, 0, 0);
    for (int manifold_index = 0;
            manifold_index < world->rigid.manifold_count; ++manifold_index) {
        PlImpulseManifold* contact_manifold =
            &world->rigid.manifolds[manifold_index];
        int robot_contact = contact_manifold->body_b >= RA_CUDA_BODY_SHELL_START
            && contact_manifold->body_b < RA_CUDA_ROBOT_BODY_END;
        for (int point_index = 0; point_index < contact_manifold->point_count;
                ++point_index) {
            PlImpulsePoint* point = &contact_manifold->points[point_index];
            float impulse = ra_max(point->normal_impulse, 0.0f);
            if (contact_manifold->body_b >= RA_CUDA_BODY_PAD_LEFT_START
                    && contact_manifold->body_b < RA_CUDA_BODY_PAD_RIGHT_START) {
                state->pad_normal_impulse[0] += impulse;
            } else if (contact_manifold->body_b >= RA_CUDA_BODY_PAD_RIGHT_START
                    && contact_manifold->body_b < RA_CUDA_ROBOT_BODY_END) {
                state->pad_normal_impulse[1] += impulse;
            }
            if (robot_contact && impulse > 0.0f) {
                RaVec3 reaction = ra_scale(contact_manifold->normal, -impulse);
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
        RaWorld* world, int object_body, int pad_mask) {
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
            RA_ROBOT_FRICTION, RA_ROBOT_FRICTION, 0.0f);
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
            RA_ROBOT_FRICTION, RA_ROBOT_FRICTION, 0.0f);
    }
}

RA_D static RA_INLINE void ra_buildc(
        RaWorld* world) {
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
                && rigid->manifold_count < PL_IMPULSE_MAX_MANIFOLDS) {
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
    if (state->basketball_mode) {
        ra_bpad(world, 0);
        ra_bpad(world, 1);
    } else {
        ra_padc(world, 0, RA_CUDA_BODY_CUBE);
        ra_padc(world, 1, RA_CUDA_BODY_CUBE);
        if (state->stack_mode) {
            ra_padc(world, 0, RA_CUDA_BODY_BASE);
            ra_padc(world, 1, RA_CUDA_BODY_BASE);
        }
    }
    int pad_mask = (rigid->compound_pad_component_mask[0] != 0 ? 1 : 0)
        | (rigid->compound_pad_component_mask[1] != 0 ? 2 : 0);
    ra_objr(world, RA_CUDA_BODY_CUBE, pad_mask);
    if (state->stack_mode) {
        ra_objr(world, RA_CUDA_BODY_BASE, pad_mask);
    }
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
                RA_CONTACT_MARGIN, RA_ROBOT_FRICTION, 0.70f, 0.0f);
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
    RaWorld world;
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
    RaWorld* world = &envs[state_index].world;
    RaState* state = &world->state;
    for (int action = 0; action < RA_ACTIONS; ++action) {
        world->staged.actions[action] = actions[state_index * RA_ACTIONS
            + action];
    }
    float span[RA_DOF] = {2.30f, 1.45f, 2.30f, 1.20f,
        2.30f, 1.50f, 2.20f};
    for (int joint = 0; joint < RA_DOF; ++joint) {
        float action = ra_clamp(world->staged.actions[joint], -1.0f, 1.0f);
        state->target_q[joint] = ra_clamp(ra_jhome(joint)
            + action * span[joint],
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
        state->previous_cube_position = state->cube_position;
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
    RaWorld* world = &envs[start + local].world;
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
    RaWorld* world = &env->world;
    float reward = world->state.basketball_mode
        ? ra_stepb(&world->state, world->staged.actions,
            world->staged.energy, world->staged.first_grasp,
            world->staged.released, world->staged.links)
        : ra_stept(&world->state, world->staged.actions,
            world->staged.energy, world->staged.first_grasp,
            world->staged.released, world->staged.links);
    float terminal = world->state.done ? 1.0f : 0.0f;
    if (terminal != 0.0f) {
        RaState* state = &world->state;
        struct Log* log = &env->log;
        if (state->basketball_mode) {
            float grasp_denominator = ra_max(
                (float)state->attempts, (float)state->basketball_grasps);
            float release_denominator = ra_max(
                (float)state->basketball_grasps, 1.0f);
            log->basketball_mode += 1.0f;
            log->score += (float)state->baskets;
            log->baskets += (float)state->baskets;
            log->grasp_rate += grasp_denominator > 0.0f
                ? (float)state->basketball_grasps / grasp_denominator
                : 0.0f;
            log->lift_rate += state->lifted ? 1.0f : 0.0f;
            log->slip_rate += state->slip_events > 0 ? 1.0f : 0.0f;
            log->release_rate += (float)state->basketball_releases
                / release_denominator;
            log->release_center_miss_cm_sum +=
                state->basketball_release_center_miss_cm_sum;
            log->release_center_miss_count +=
                (float)state->basketball_releases;
            log->episode_length += (float)state->step;
            log->n += 1.0f;
        } else {
            log->score += state->success ? 1.0f : 0.0f;
            log->episode_length += (float)state->step;
            log->success_rate += state->success ? 1.0f : 0.0f;
            log->grasp_rate += state->ever_grasped ? 1.0f : 0.0f;
            log->lift_rate += state->lifted ? 1.0f : 0.0f;
            log->transport_rate += state->transported ? 1.0f : 0.0f;
            log->release_rate += state->stack_mode
                ? (state->valid_release_achieved ? 1.0f : 0.0f)
                : (state->released_near_target ? 1.0f : 0.0f);
            log->return_value += state->episode_return;
            log->reach_distance += ra_length(
                ra_sub(state->cube_position, state->end_effector));
            log->place_distance += ra_length(
                ra_sub(state->target_position, state->cube_position));
            log->energy += state->episode_energy
                / ra_max((float)state->step, 1.0f);
            log->pinch_force += state->episode_pinch_force
                / ra_max((float)state->pinch_substeps, 1.0f);
            log->slip_rate += state->slip_events > 0 ? 1.0f : 0.0f;
            log->cube_angular_speed += ra_length(
                state->cube_angular_velocity);
            log->base_angular_speed += state->stack_mode
                ? ra_length(state->base_cube_angular_velocity) : 0.0f;
            log->orientation_error += state->stack_mode
                ? ra_max(ra_cup(state->cube_rotation),
                    ra_cup(state->base_cube_rotation))
                : ra_cup(state->cube_rotation);
            if (state->stack_mode) {
                RaVec3 alignment = ra_sub(
                    state->cube_position, state->base_cube_position);
                RaVec3 base_slide = ra_sub(
                    state->base_cube_position,
                    state->base_cube_start_position);
                log->stack_rate += state->ever_stacked ? 1.0f : 0.0f;
                log->stable_stack_rate += state->success ? 1.0f : 0.0f;
                log->stack_alignment_rate += state->stack_aligned
                    ? 1.0f : 0.0f;
                log->valid_stack_contact_rate +=
                    state->valid_stack_contact ? 1.0f : 0.0f;
                log->clearance_rate += state->cleared_after_release
                    ? 1.0f : 0.0f;
                log->settle_rate += state->max_placement_settle_steps > 0
                    ? 1.0f : 0.0f;
                log->stack_alignment += sqrtf(
                    alignment.x*alignment.x + alignment.z*alignment.z);
                log->base_slide_distance += sqrtf(
                    base_slide.x*base_slide.x + base_slide.z*base_slide.z);
            }
            log->n += 1.0f;
        }
        unsigned int topology = state->basketball_mode ? 3u
            : (state->stack_mode ? 2u : 1u);
        ra_reset(state);
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
