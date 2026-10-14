#pragma once
#include "contact_traits.cuh"

// Boolean OBB tests with finite centers, orthonormal axes and positive extents.
// Use nonnegative margins and a positive axis_epsilon. Margin is
// applied to each normalized SAT axis; cross axes below axis_epsilon are skipped.
// This is a clearance predicate, independent of manifold clipping/feature IDs.
template<class T = PfContactTraits>
struct PfBoxClearanceT {
    using Vec3 = typename T::Vec3;
    __host__ __device__ static __forceinline__ float radius(const Vec3* axes, Vec3 half, Vec3 direction) {
        return half.x * fabsf(T::dot(axes[0], direction))
            + half.y * fabsf(T::dot(axes[1], direction))
            + half.z * fabsf(T::dot(axes[2], direction));
    }
    // Signed distance of the lowest box support point to a plane. normal is unit
    // length and dot(normal, point) = offset; positive clearance is outside.
    __host__ __device__ static __forceinline__ float plane(Vec3 center, const Vec3* axes,
        Vec3 half, Vec3 normal, float offset) {
        return T::dot(center, normal) - offset - radius(axes, half, normal);
    }
    __host__ __device__ static __forceinline__ bool overlap(Vec3 delta, const Vec3* axes_a, Vec3 half_a,
        const Vec3* axes_b, Vec3 half_b, float margin, float axis_epsilon = 1.0e-6f) {
        for (int index = 0; index < 15; ++index) {
            Vec3 axis;
            if (index < 3) axis = axes_a[index];
            else if (index < 6) axis = axes_b[index - 3];
            else {
                int pair = index - 6;
                axis = T::cross(axes_a[pair / 3], axes_b[pair % 3]);
                float length = sqrtf(T::dot(axis, axis));
                if (length < axis_epsilon) continue;
                axis = T::scale(axis, 1.0f / length);
            }
            float reach = radius(axes_a, half_a, axis) + radius(axes_b, half_b, axis) + margin;
            if (fabsf(T::dot(delta, axis)) > reach) return false;
        }
        return true;
    }
    // Query overlap on either local-Y face of B. inward must be +axes_b[1]
    // or -axes_b[1]; the caller chooses the exposed face.
    __host__ __device__ static __forceinline__ bool face(Vec3 center_a, const Vec3* axes_a, Vec3 half_a,
        Vec3 center_b, const Vec3* axes_b, Vec3 half_b, Vec3 inward, float margin) {
        Vec3 delta = T::sub(center_a, center_b);
        if (!overlap(delta, axes_a, half_a, axes_b, half_b, margin) || T::dot(delta,inward) < 0) return false;
        Vec3 inner_surface = T::add(center_b, T::scale(inward, half_b.y));
        Vec3 surface_a = center_a, support_dir = T::scale(inward, -1.0f);
        const float half[3] = {half_a.x, half_a.y, half_a.z};
        for (int axis = 0; axis < 3; ++axis) {
            float sign = T::dot(axes_a[axis], support_dir) < 0 ? -1.0f : 1.0f;
            surface_a = T::add(surface_a, T::scale(axes_a[axis], sign * half[axis]));
        }
        float separation = T::dot(T::sub(surface_a, inner_surface), inward);
        return separation <= margin;
    }
};
