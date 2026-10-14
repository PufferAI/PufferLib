#pragma once
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "contact_traits.cuh"

// Margin-aware SAT, exact-float feature selection and stable manifold feature IDs.
// T supplies vector/shape types and math; PfContactTraits is the native default.
// Supported shapes are boxes and spheres. Normals point from B to A.
template<class T = PfContactTraits>
struct PfSatCollisionT {
    using Vec3 = typename T::Vec3;
    using Shape = typename T::Shape;
    using Contact = typename T::Contact;
    static constexpr float parallel_epsilon = 1.0e-8f;
    static constexpr float duplicate_epsilon = 1.0e-8f;
    static constexpr int max_points = 4, max_clip_vertices = 8;
    typedef enum Feature {
        PF_SAT_FACE_A_X = 0,
        PF_SAT_FACE_A_Y = 1,
        PF_SAT_FACE_A_Z = 2,
        PF_SAT_FACE_B_X = 3,
        PF_SAT_FACE_B_Y = 4,
        PF_SAT_FACE_B_Z = 5,
        PF_SAT_EDGE_A0_B0 = 6,
    } Feature;

    typedef struct Obb {
        Vec3 center;
        Vec3 axis[3];
        Vec3 half_extents;
    } Obb;

    typedef struct Query {
        Contact contact;
        int feature;
    } Query;

    typedef struct Manifold {
        int count;
        int feature;
        Contact point[max_points];
        uint32_t point_feature[max_points]; // warm-start key
    } Manifold;

    __device__ static __forceinline__ Vec3 negate(Vec3 value) {
        return T::scale(value, -1.0f);
    }

    __device__ static __forceinline__ float half_extent(Vec3 half_extents, int index) {
        return index == 0 ? half_extents.x : index == 1 ? half_extents.y : half_extents.z;
    }

    __device__ static __forceinline__ Obb obb(const Shape* shape) {
        Obb box;
        box.center = shape->pose.position;
        T::axes(shape->pose.rotation, box.axis);
        box.half_extents = shape->half_extents;
        return box;
    }

    __device__ static __forceinline__ Vec3 support(const Obb* box, Vec3 direction) {
        float sx = T::dot(box->axis[0], direction) < 0.0f ? -box->half_extents.x : box->half_extents.x;
        float sy = T::dot(box->axis[1], direction) < 0.0f ? -box->half_extents.y : box->half_extents.y;
        float sz = T::dot(box->axis[2], direction) < 0.0f ? -box->half_extents.z : box->half_extents.z;
        return T::add(box->center,
            T::add(T::scale(box->axis[0], sx),
                T::add(T::scale(box->axis[1], sy), T::scale(box->axis[2], sz))));
    }

    __device__ static __forceinline__ Vec3 orient(Vec3 axis, float projection_a_minus_b) {
        return projection_a_minus_b < 0.0f ? negate(axis) : axis; // zero -> +axis
    }

    __device__ static __forceinline__ void consider(
        Query* query, float separation, Vec3 normal, int feature) {
        if (separation > query->contact.separation) { // exact-float tie keeps first axis
            query->contact.separation = separation;
            query->contact.normal = normal;
            query->feature = feature;
        }
    }

    __device__ static __forceinline__ void consider_cross(Query* query, float unnormalised_separation,
        float projection_b_minus_a, Vec3 axis_a, Vec3 axis_b, int feature) {
        Vec3 cross_axis = T::cross(axis_a, axis_b);
        float length_squared = T::dot(cross_axis, cross_axis);
        float parallel_squared = parallel_epsilon * parallel_epsilon;
        if (!(length_squared > parallel_squared)) {
            return;
        }
        float inverse_length = 1.0f / sqrtf(length_squared);
        float separation = unnormalised_separation * inverse_length;
        if (separation <= query->contact.separation) {
            return;
        }
        Vec3 normal = T::scale(cross_axis, inverse_length);
        if (projection_b_minus_a > 0.0f) {
            normal = negate(normal); // normal always B->A
        }
        query->contact.separation = separation;
        query->contact.normal = normal;
        query->feature = feature;
    }

    __device__ static __forceinline__ Query query_obb(const Obb* a, const Obb* b, float margin) {
        const Vec3 a0 = a->axis[0];
        const Vec3 a1 = a->axis[1];
        const Vec3 a2 = a->axis[2];
        const Vec3 b0 = b->axis[0];
        const Vec3 b1 = b->axis[1];
        const Vec3 b2 = b->axis[2];
        const float ax = a->half_extents.x;
        const float ay = a->half_extents.y;
        const float az = a->half_extents.z;
        const float bx = b->half_extents.x;
        const float by = b->half_extents.y;
        const float bz = b->half_extents.z;

        const Vec3 b_minus_a = T::sub(b->center, a->center);
        const float t0 = T::dot(b_minus_a, a0);
        const float t1 = T::dot(b_minus_a, a1);
        const float t2 = T::dot(b_minus_a, a2);

        const float r00 = T::dot(a0, b0);
        const float r01 = T::dot(a0, b1);
        const float r02 = T::dot(a0, b2);
        const float r10 = T::dot(a1, b0);
        const float r11 = T::dot(a1, b1);
        const float r12 = T::dot(a1, b2);
        const float r20 = T::dot(a2, b0);
        const float r21 = T::dot(a2, b1);
        const float r22 = T::dot(a2, b2);
        const float ar00 = fabsf(r00);
        const float ar01 = fabsf(r01);
        const float ar02 = fabsf(r02);
        const float ar10 = fabsf(r10);
        const float ar11 = fabsf(r11);
        const float ar12 = fabsf(r12);
        const float ar20 = fabsf(r20);
        const float ar21 = fabsf(r21);
        const float ar22 = fabsf(r22);

        Query query;
        query.contact.hit = 0;
        query.contact.iterations = 15;
        query.contact.separation = -3.402823466e+38f;
        query.contact.normal = a0;
        query.contact.point_a = a->center;
        query.contact.point_b = b->center;
        query.feature = PF_SAT_FACE_A_X;

        consider(&query, fabsf(t0) - ax - (bx * ar00 + by * ar01 + bz * ar02), orient(a0, -t0),
            PF_SAT_FACE_A_X);
        consider(&query, fabsf(t1) - ay - (bx * ar10 + by * ar11 + bz * ar12), orient(a1, -t1),
            PF_SAT_FACE_A_Y);
        consider(&query, fabsf(t2) - az - (bx * ar20 + by * ar21 + bz * ar22), orient(a2, -t2),
            PF_SAT_FACE_A_Z);

        const float u0 = t0 * r00 + t1 * r10 + t2 * r20;
        const float u1 = t0 * r01 + t1 * r11 + t2 * r21;
        const float u2 = t0 * r02 + t1 * r12 + t2 * r22;
        consider(&query, fabsf(u0) - bx - (ax * ar00 + ay * ar10 + az * ar20), orient(b0, -u0),
            PF_SAT_FACE_B_X);
        consider(&query, fabsf(u1) - by - (ax * ar01 + ay * ar11 + az * ar21), orient(b1, -u1),
            PF_SAT_FACE_B_Y);
        consider(&query, fabsf(u2) - bz - (ax * ar02 + ay * ar12 + az * ar22), orient(b2, -u2),
            PF_SAT_FACE_B_Z);

        consider_cross(&query, fabsf(t2 * r10 - t1 * r20) - (ay * ar20 + az * ar10) - (by * ar02 + bz * ar01),
            t2 * r10 - t1 * r20, a0, b0, PF_SAT_EDGE_A0_B0 + 0);
        consider_cross(&query, fabsf(t2 * r11 - t1 * r21) - (ay * ar21 + az * ar11) - (bz * ar00 + bx * ar02),
            t2 * r11 - t1 * r21, a0, b1, PF_SAT_EDGE_A0_B0 + 1);
        consider_cross(&query, fabsf(t2 * r12 - t1 * r22) - (ay * ar22 + az * ar12) - (bx * ar01 + by * ar00),
            t2 * r12 - t1 * r22, a0, b2, PF_SAT_EDGE_A0_B0 + 2);

        consider_cross(&query, fabsf(t0 * r20 - t2 * r00) - (az * ar00 + ax * ar20) - (by * ar12 + bz * ar11),
            t0 * r20 - t2 * r00, a1, b0, PF_SAT_EDGE_A0_B0 + 3);
        consider_cross(&query, fabsf(t0 * r21 - t2 * r01) - (az * ar01 + ax * ar21) - (bz * ar10 + bx * ar12),
            t0 * r21 - t2 * r01, a1, b1, PF_SAT_EDGE_A0_B0 + 4);
        consider_cross(&query, fabsf(t0 * r22 - t2 * r02) - (az * ar02 + ax * ar22) - (bx * ar11 + by * ar10),
            t0 * r22 - t2 * r02, a1, b2, PF_SAT_EDGE_A0_B0 + 5);

        consider_cross(&query, fabsf(t1 * r00 - t0 * r10) - (ax * ar10 + ay * ar00) - (by * ar22 + bz * ar21),
            t1 * r00 - t0 * r10, a2, b0, PF_SAT_EDGE_A0_B0 + 6);
        consider_cross(&query, fabsf(t1 * r01 - t0 * r11) - (ax * ar11 + ay * ar01) - (bz * ar20 + bx * ar22),
            t1 * r01 - t0 * r11, a2, b1, PF_SAT_EDGE_A0_B0 + 7);
        consider_cross(&query, fabsf(t1 * r02 - t0 * r12) - (ax * ar12 + ay * ar02) - (bx * ar21 + by * ar20),
            t1 * r02 - t0 * r12, a2, b2, PF_SAT_EDGE_A0_B0 + 8);

        query.contact.hit = query.contact.separation <= margin;
        Vec3 normal = query.contact.normal;
        query.contact.point_a = support(a, negate(normal));
        query.contact.point_b = support(b, normal);
        return query;
    }

    __device__ static __forceinline__ int clip(
        const Vec3* input, int count, Vec3* output, Vec3 normal, float offset) {
        if (count <= 0) {
            return 0;
        }
        int output_count = 0;
        Vec3 previous = input[count - 1];
        float previous_distance = T::dot(previous, normal) - offset;
        int previous_inside = previous_distance <= 0.0f;
        for (int index = 0; index < count; ++index) {
            Vec3 current = input[index];
            float current_distance = T::dot(current, normal) - offset;
            int current_inside = current_distance <= 0.0f;
            if (current_inside != previous_inside && output_count < max_clip_vertices) {
                float fraction = previous_distance / (previous_distance - current_distance);
                output[output_count++] = T::lerp(previous, current, fraction);
            }
            if (current_inside && output_count < max_clip_vertices) {
                output[output_count++] = current;
            }
            previous = current;
            previous_distance = current_distance;
            previous_inside = current_inside;
        }
        return output_count;
    }

    __device__ static __forceinline__ void face(
        const Obb* box, int normal_axis, float sign, Vec3 output[4]) {
        const int tangent_a = (normal_axis + 1) % 3;
        const int tangent_b = (normal_axis + 2) % 3;
        const float face_half = half_extent(box->half_extents, normal_axis);
        const float half_a = half_extent(box->half_extents, tangent_a);
        const float half_b = half_extent(box->half_extents, tangent_b);
        const Vec3 center = T::add(box->center, T::scale(box->axis[normal_axis], sign * face_half));
        const Vec3 along_a = T::scale(box->axis[tangent_a], half_a);
        const Vec3 along_b = T::scale(box->axis[tangent_b], half_b);
        output[0] = T::sub(T::sub(center, along_a), along_b);
        output[1] = T::add(T::sub(center, along_a), along_b);
        output[2] = T::add(T::add(center, along_a), along_b);
        output[3] = T::sub(T::add(center, along_a), along_b);
    }

    __device__ static __forceinline__ int aligned_axis(const Obb* box, Vec3 target_normal) {
        float best = fabsf(T::dot(box->axis[0], target_normal));
        int best_axis = 0;
        float candidate = fabsf(T::dot(box->axis[1], target_normal));
        if (candidate > best) {
            best = candidate;
            best_axis = 1;
        }
        candidate = fabsf(T::dot(box->axis[2], target_normal));
        if (candidate > best) {
            best_axis = 2;
        }
        return best_axis;
    }

    __device__ static __forceinline__ int is_selected(const int* selected, int count, int candidate) {
        for (int index = 0; index < count; ++index) {
            if (selected[index] == candidate) {
                return 1;
            }
        }
        return 0;
    }

    __device__ static __forceinline__ int duplicate(const Manifold* manifold, Vec3 point_a, Vec3 point_b) {
        const float epsilon = duplicate_epsilon;
        const float epsilon_squared = epsilon * epsilon;
        for (int index = 0; index < manifold->count; ++index) {
            Vec3 delta_a = T::sub(point_a, manifold->point[index].point_a);
            Vec3 delta_b = T::sub(point_b, manifold->point[index].point_b);
            if (T::dot(delta_a, delta_a) <= epsilon_squared
                && T::dot(delta_b, delta_b) <= epsilon_squared) {
                return 1;
            }
        }
        return 0;
    }

    __device__ static __forceinline__ uint32_t feature_id(int manifold_feature, int point_index) {
        return ((uint32_t)(manifold_feature + 1) << 8) // 0 is invalid
            | (uint32_t)(point_index + 1);
    }

    __device__ static __forceinline__ void clear(Manifold* manifold) {
        manifold->count = 0;
        manifold->feature = PF_SAT_FACE_A_X;
        for (int index = 0; index < max_points; ++index) {
            Contact empty = {};
            manifold->point[index] = empty;
            manifold->point_feature[index] = 0;
        }
    }

    __device__ static __forceinline__ int manifold_obb(const Obb* a, const Obb* b, float margin,
        const Query* query, Manifold* manifold) {
        clear(manifold);
        manifold->feature = query->feature;
        if (!query->contact.hit) {
            return 0;
        }

        if (query->feature >= PF_SAT_EDGE_A0_B0) {
            manifold->point[0] = query->contact;
            manifold->point_feature[0] = feature_id(query->feature, 0);
            manifold->count = 1;
            return 1;
        }

        const int reference_is_a = query->feature < PF_SAT_FACE_B_X;
        const int reference_axis = query->feature % 3;
        const Obb* reference = reference_is_a ? a : b;
        const Obb* incident = reference_is_a ? b : a;
        const Vec3 contact_normal = query->contact.normal;

        Vec3 reference_normal = reference_is_a ? negate(contact_normal) : contact_normal;
        const float reference_sign =
            T::dot(reference->axis[reference_axis], reference_normal) < 0.0f ? -1.0f : 1.0f;
        reference_normal = T::scale(reference->axis[reference_axis], reference_sign);
        const Vec3 reference_center = T::add(reference->center,
            T::scale(reference_normal, half_extent(reference->half_extents, reference_axis)));

        const Vec3 incident_target = negate(reference_normal);
        const int incident_axis = aligned_axis(incident, incident_target);
        const float incident_sign =
            T::dot(incident->axis[incident_axis], incident_target) < 0.0f ? -1.0f : 1.0f;

        Vec3 input[max_clip_vertices];
        Vec3 scratch[max_clip_vertices];
        face(incident, incident_axis, incident_sign, input);
        int count = 4;

        const int tangent_a = (reference_axis + 1) % 3;
        const int tangent_b = (reference_axis + 2) % 3;
        const Vec3 side_a = reference->axis[tangent_a];
        const Vec3 side_b = reference->axis[tangent_b];
        const float half_a = half_extent(reference->half_extents, tangent_a);
        const float half_b = half_extent(reference->half_extents, tangent_b);
        const float center_a = T::dot(reference_center, side_a);
        const float center_b = T::dot(reference_center, side_b);
        count = clip(input, count, scratch, side_a, center_a + half_a);
        count = clip(scratch, count, input, negate(side_a), -center_a + half_a);
        count = clip(input, count, scratch, side_b, center_b + half_b);
        count = clip(scratch, count, input, negate(side_b), -center_b + half_b);

        if (count <= 0) {
            manifold->point[0] = query->contact;
            manifold->point_feature[0] = feature_id(query->feature, 0);
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
                if (is_selected(selected, selected_count, index)) {
                    continue;
                }
                const float projection = direction_a[corner] * T::dot(input[index], side_a)
                    + direction_b[corner] * T::dot(input[index], side_b);
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
            if (!is_selected(selected, selected_count, index)) {
                selected[selected_count++] = index;
            }
        }

        for (int output = 0; output < selected_count; ++output) {
            const Vec3 incident_point = input[selected[output]];
            const float separation = T::dot(T::sub(incident_point, reference_center), reference_normal);
            if (separation > margin) {
                continue;
            }
            const Vec3 reference_point =
                T::sub(incident_point, T::scale(reference_normal, separation));
            Contact contact;
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
            if (duplicate(manifold, contact.point_a, contact.point_b)) {
                continue;
            }
            int output_index = manifold->count++;
            manifold->point[output_index] = contact;
            manifold->point_feature[output_index] = feature_id(query->feature, selected[output]);
            if (manifold->count == max_points) {
                break;
            }
        }

        if (manifold->count == 0) {
            manifold->point[0] = query->contact;
            manifold->point_feature[0] = feature_id(query->feature, 0);
            manifold->count = 1;
        }
        return manifold->count;
    }

    __device__ static __forceinline__ Query sphere_sphere(
        const Shape* a, const Shape* b, float margin) {
        Query query;
        memset(&query, 0, sizeof(query));
        Vec3 delta = T::sub(a->pose.position, b->pose.position);
        float distance_squared = T::dot(delta, delta);
        float distance = sqrtf(T::max(distance_squared, 0.0f));
        Vec3 normal = distance > 1.0e-10f ? T::scale(delta, 1.0f / distance) : T::v3(1, 0, 0);
        float radius_a = a->half_extents.x;
        float radius_b = b->half_extents.x;
        query.contact.iterations = 1;
        query.contact.separation = distance - radius_a - radius_b;
        query.contact.normal = normal;
        query.contact.point_a = T::sub(a->pose.position, T::scale(normal, radius_a));
        query.contact.point_b = T::add(b->pose.position, T::scale(normal, radius_b));
        query.contact.hit = query.contact.separation <= margin;
        query.feature = PF_SAT_FACE_A_X;
        return query;
    }

    __device__ static __forceinline__ Query sphere_box(
        const Shape* sphere, const Shape* box, float margin) {
        Query query;
        memset(&query, 0, sizeof(query));
        Obb obb = PfSatCollisionT::obb(box);
        Vec3 center_delta = T::sub(sphere->pose.position, obb.center);
        float local[3] = {
            T::dot(center_delta, obb.axis[0]),
            T::dot(center_delta, obb.axis[1]),
            T::dot(center_delta, obb.axis[2]),
        };
        float half[3] = {obb.half_extents.x, obb.half_extents.y, obb.half_extents.z};
        Vec3 closest = obb.center;
        for (int axis = 0; axis < 3; ++axis) {
            closest = T::add(
                closest, T::scale(obb.axis[axis], T::clamp(local[axis], -half[axis], half[axis])));
        }
        Vec3 delta = T::sub(sphere->pose.position, closest);
        float distance_squared = T::dot(delta, delta);
        float radius = sphere->half_extents.x;
        int feature_axis = 0;
        if (distance_squared > 1.0e-20f) {
            float distance = sqrtf(distance_squared);
            query.contact.normal = T::scale(delta, 1.0f / distance);
            query.contact.separation = distance - radius;
            float best_axis = fabsf(T::dot(query.contact.normal, obb.axis[0]));
            for (int axis = 1; axis < 3; ++axis) {
                float alignment = fabsf(T::dot(query.contact.normal, obb.axis[axis]));
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
            query.contact.normal = T::scale(obb.axis[feature_axis], sign);
            query.contact.separation = -radius - clearance;
            query.contact.point_b =
                T::add(sphere->pose.position, T::scale(query.contact.normal, clearance));
        }
        query.contact.point_a = T::sub(sphere->pose.position, T::scale(query.contact.normal, radius));
        query.contact.hit = query.contact.separation <= margin;
        query.contact.iterations = 1;
        query.feature = PF_SAT_FACE_B_X + feature_axis;
        return query;
    }

    __device__ static __forceinline__ Query query(
        const Shape* a, const Shape* b, float margin) {
        if (a->type == T::sphere_kind && b->type == T::sphere_kind) {
            return sphere_sphere(a, b, margin);
        }
        if (a->type == T::sphere_kind && b->type == T::box_kind) {
            return sphere_box(a, b, margin);
        }
        if (a->type == T::box_kind && b->type == T::sphere_kind) {
            Query query = sphere_box(b, a, margin);
            Vec3 point = query.contact.point_a;
            query.contact.point_a = query.contact.point_b;
            query.contact.point_b = point;
            query.contact.normal = negate(query.contact.normal);
            query.feature -= PF_SAT_FACE_B_X;
            return query;
        }
        Obb box_a = obb(a);
        Obb box_b = obb(b);
        return query_obb(&box_a, &box_b, margin);
    }

    __device__ static __forceinline__ int manifold(
        const Shape* a, const Shape* b, float margin, Manifold* manifold) {
        clear(manifold);
        if (a->type != T::box_kind || b->type != T::box_kind) {
            Query query = PfSatCollisionT::query(a, b, margin);
            if (!query.contact.hit) {
                return 0;
            }
            manifold->count = 1;
            manifold->feature = query.feature;
            manifold->point[0] = query.contact;
            manifold->point_feature[0] = feature_id(query.feature, 0);
            return 1;
        }
        Obb box_a = obb(a);
        Obb box_b = obb(b);
        Query query = query_obb(&box_a, &box_b, margin);
        return manifold_obb(&box_a, &box_b, margin, &query, manifold);
    }

};
using PfSatCollision = PfSatCollisionT<>;
