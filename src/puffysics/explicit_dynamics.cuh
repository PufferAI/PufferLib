#pragma once
#include "contact_traits.cuh"

// Explicit updates: coefficients, limits and limit response are caller policy.
// These helpers do not add implicit damping or velocity-bias forces. Inputs
// must be finite: dt > 0, ordered limits, nonnegative stiffness/damping, force
// and speed limits, and physical inertia. Compliance needs effective_mass > 0,
// damping_ratio >= 0 and epsilon > 0. No validation/status is performed here.
template<class T = PfContactTraits>
struct PfExplicitDynamicsT {
    using Vec3 = typename T::Vec3;
    __host__ __device__ static __forceinline__ float servo(float target, float position, float velocity,
        float stiffness, float damping, float force_limit) {
        return T::clamp(stiffness * (target - position) - damping * velocity,
            -force_limit, force_limit);
    }
    __host__ __device__ static __forceinline__ float velocity(float current, float acceleration,
        float dt, float speed_limit) {
        return T::clamp(current + acceleration * dt, -speed_limit, speed_limit);
    }
    __host__ __device__ static __forceinline__ float coordinate(float position, float& velocity,
        float dt, float low, float high, float limit_response) {
        position += velocity * dt;
        if (position < low) {
            position = low;
            velocity = T::max(velocity, 0.0f) * limit_response;
        } else if (position > high) {
            position = high;
            velocity = T::min(velocity, 0.0f) * limit_response;
        }
        return position;
    }
    // Clamped slides stop outward motion even when exactly on a boundary.
    __host__ __device__ static __forceinline__ float slide(float position, float& velocity,
        float dt, float low, float high) {
        position = T::clamp(position + velocity * dt, low, high);
        if ((position <= low && velocity < 0) || (position >= high && velocity > 0)) velocity = 0;
        return position;
    }
    struct Compliance { float cfm, erp; };
    __host__ __device__ static __forceinline__ Compliance compliance(float stiffness, float effective_mass,
        float damping_ratio, float dt, float epsilon) {
        float damping = 2.0f * damping_ratio * sqrtf(T::max(stiffness * effective_mass, 0.0f));
        float spring_damping = damping + dt * stiffness;
        return {1.0f / T::max(dt * spring_damping, epsilon),
            stiffness / T::max(spring_damping, epsilon)};
    }
    template<class Tensor>
    __host__ __device__ static __forceinline__ Tensor box_inertia(float mass, Vec3 half) {
        if (mass <= 0) return {0,0,0,0,0,0};
        float x2 = half.x * half.x, y2 = half.y * half.y, z2 = half.z * half.z;
        return {mass * (y2 + z2) / 3.0f, mass * (x2 + z2) / 3.0f,
            mass * (x2 + y2) / 3.0f, 0,0,0};
    }
    template<class Tensor>
    __host__ __device__ static __forceinline__ Tensor sphere_inertia(float mass, float radius) {
        float diagonal = mass > 0 ? 0.4f * mass * radius * radius : 0;
        return {diagonal,diagonal,diagonal,0,0,0};
    }
    // Invert a full symmetric local inertia tensor. determinant_floor must be
    // positive; this regularization is caller policy, not physical validation.
    template<class Tensor, class Quat>
    __host__ __device__ static __forceinline__ Vec3 inverse_inertia(Tensor inertia, Quat rotation,
        Vec3 vector, float determinant_floor) {
        Vec3 local = T::rotate(T::conjugate(rotation), vector);
        float cofactor_xx = inertia.yy * inertia.zz - inertia.yz * inertia.yz;
        float cofactor_xy = inertia.xz * inertia.yz - inertia.xy * inertia.zz;
        float cofactor_xz = inertia.xy * inertia.yz - inertia.xz * inertia.yy;
        float cofactor_yy = inertia.xx * inertia.zz - inertia.xz * inertia.xz;
        float cofactor_yz = inertia.xy * inertia.xz - inertia.xx * inertia.yz;
        float cofactor_zz = inertia.xx * inertia.yy - inertia.xy * inertia.xy;
        float determinant = inertia.xx * cofactor_xx + inertia.xy * cofactor_xy + inertia.xz * cofactor_xz;
        float inverse = 1.0f / T::max(fabsf(determinant), determinant_floor);
        if (determinant < 0.0f) inverse = -inverse;
        Vec3 product = T::v3(
            inverse * (cofactor_xx * local.x + cofactor_xy * local.y + cofactor_xz * local.z),
            inverse * (cofactor_xy * local.x + cofactor_yy * local.y + cofactor_yz * local.z),
            inverse * (cofactor_xz * local.x + cofactor_yz * local.y + cofactor_zz * local.z));
        return T::rotate(rotation, product);
    }
};
