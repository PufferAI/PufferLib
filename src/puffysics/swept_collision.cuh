#pragma once
#include "sat_manifold.cuh"

// Conservative advancement with constant world linear/angular velocities.
// Queries do not integrate bodies. A miss may mean the iteration budget was
// exhausted; iterations and the last contact are returned for caller policy.
template<class T=PfContactTraits>
struct PfSweptCollisionT {
    using Vec3=typename T::Vec3;
    using Shape=typename T::Shape;
    using Contact=typename T::Contact;
    using Sweep=typename T::Sweep;
    using Sat=PfSatCollisionT<T>;
    struct Ring {
        Vec3 center, axis_x, axis_z, normal; // orthonormal frame, X/Z in ring plane
        float major_radius, tube_radius;
    };
    __device__ static __forceinline__ Sweep shape(const Shape* initial_a,
        const Shape* initial_b, Vec3 linear_a, Vec3 angular_a, Vec3 linear_b,
        Vec3 angular_b, float maximum_time, float target_margin) {
        Sweep sweep;
        memset(&sweep, 0, sizeof(sweep));
        sweep.toi = maximum_time;
        float time = 0.0f;
        float angular_bound =
            initial_a->type == T::sphere_kind ? 0.0f : T::length(angular_a) * T::radius(initial_a);
        angular_bound +=
            initial_b->type == T::sphere_kind ? 0.0f : T::length(angular_b) * T::radius(initial_b);
        for (int iteration = 0; iteration < 12; ++iteration) {
            sweep.iterations = iteration + 1;
            Shape a = *initial_a;
            Shape b = *initial_b;
            a.pose.position = T::add(a.pose.position, T::scale(linear_a, time));
            b.pose.position = T::add(b.pose.position, T::scale(linear_b, time));
            a.pose.rotation = T::integrate_rotation(a.pose.rotation, angular_a, time);
            b.pose.rotation = T::integrate_rotation(b.pose.rotation, angular_b, time);
            Contact contact = Sat::query(&a, &b, target_margin).contact;
            sweep.contact = contact;
            if (contact.hit) {
                sweep.hit = 1;
                sweep.toi = time;
                return sweep;
            }
            float closing_speed = -T::dot(T::sub(linear_a, linear_b), contact.normal) + angular_bound;
            if (closing_speed <= 1.0e-8f) {
                return sweep;
            }
            float advance = (contact.separation - target_margin) / closing_speed;
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

    // Avoid changing arithmetic for the common world-X/Z ring frame.
    __device__ static __forceinline__ Contact sphere_ring_xz(Vec3 sphere_position,
            float sphere_radius, const Ring& ring, float margin) {
        Contact contact;
        memset(&contact,0,sizeof(contact));
        Vec3 center=ring.center;
        float dx=sphere_position.x-center.x;
        float dz=sphere_position.z-center.z;
        float radial=sqrtf(dx*dx+dz*dz);
        float inverse_radial=radial>1.0e-10f?1.0f/radial:0.0f;
        Vec3 centerline=T::v3(
            center.x+ring.major_radius*(radial>1.0e-10f?dx*inverse_radial:1.0f),center.y,
            center.z+ring.major_radius*(radial>1.0e-10f?dz*inverse_radial:0.0f));
        Vec3 delta=T::sub(sphere_position,centerline);
        float distance=T::length(delta);
        Vec3 normal=distance>1.0e-10f?T::scale(delta,1.0f/distance):ring.normal;
        contact.hit=distance-sphere_radius-ring.tube_radius<=margin;
        contact.iterations=1;
        contact.separation=distance-sphere_radius-ring.tube_radius;
        contact.normal=normal;
        contact.point_a=T::sub(sphere_position,T::scale(normal,sphere_radius));
        contact.point_b=T::add(centerline,T::scale(normal,ring.tube_radius));
        return contact;
    }

    __device__ static __forceinline__ Contact sphere_ring(Vec3 sphere_position, float sphere_radius, const Ring& ring, float margin) {
        if (ring.axis_x.x==1 && ring.axis_x.y==0 && ring.axis_x.z==0
                && ring.axis_z.x==0 && ring.axis_z.y==0 && ring.axis_z.z==1)
            return sphere_ring_xz(sphere_position,sphere_radius,ring,margin);
        Contact contact;
        memset(&contact, 0, sizeof(contact));
        Vec3 center=ring.center;
        Vec3 offset=T::sub(sphere_position,center);
        float dx=T::dot(offset,ring.axis_x);
        float dz=T::dot(offset,ring.axis_z);
        float radial = sqrtf(dx * dx + dz * dz);
        float inverse_radial = radial > 1.0e-10f ? 1.0f / radial : 0.0f;
        Vec3 centerline=T::add(center,T::add(
            T::scale(ring.axis_x,ring.major_radius*(radial>1.0e-10f?dx*inverse_radial:1.0f)),
            T::scale(ring.axis_z,ring.major_radius*(radial>1.0e-10f?dz*inverse_radial:0.0f))));
        Vec3 delta = T::sub(sphere_position, centerline);
        float distance = T::length(delta);
        Vec3 normal = distance > 1.0e-10f ? T::scale(delta, 1.0f / distance) : ring.normal;
        contact.hit = distance - sphere_radius - ring.tube_radius <= margin;
        contact.iterations = 1;
        contact.separation = distance - sphere_radius - ring.tube_radius;
        contact.normal = normal;
        contact.point_a = T::sub(sphere_position, T::scale(normal, sphere_radius));
        contact.point_b = T::add(centerline, T::scale(normal, ring.tube_radius));
        return contact;
    }

    __device__ static __forceinline__ Sweep sweep_sphere_ring(Vec3 initial_position, Vec3 velocity, float sphere_radius, const Ring& ring, float maximum_time) {
        Sweep sweep;
        memset(&sweep, 0, sizeof(sweep));
        sweep.toi = maximum_time;
        float speed = T::length(velocity);
        float time = 0.0f;
        for (int iteration = 0; iteration < 12; ++iteration) {
            sweep.iterations = iteration + 1;
            Vec3 position = T::add(initial_position, T::scale(velocity, time));
            sweep.contact = sphere_ring(position, sphere_radius, ring, 0.0f);
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

    __device__ static __forceinline__ float approaching_time(Sweep sweep, Vec3 velocity, float advance) {
        float normal_speed = T::dot(velocity, sweep.contact.normal);
        if (sweep.hit && normal_speed < 0.0f && sweep.toi < advance) {
            return sweep.toi;
        }
        return advance;
    }

};
using PfSweptCollision=PfSweptCollisionT<>;
