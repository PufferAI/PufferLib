#pragma once

#include "math.cuh"

#define PF_VELOCITY_ITERATIONS 16
#define PF_RESTITUTION_THRESHOLD 0.5f
#define PF_TOUCHED_MASK_BITS 32

// Orientation is fixed from preparation through the velocity iterations.
__device__ static inline void pf_contact_inertia(const PfBody* body, PfVec3 columns[3]) {
    for (int i = 0; i < 3; i++) columns[i] = pf_mode_is_dynamic(body->mode)
        ? pf_inverse_inertia_world(body, pf_v3(i == 0, i == 1, i == 2)) : pf_v3(0,0,0);
}
__device__ static inline void pf_apply_contact_impulse(PfBody* body, PfVec3 point,
        PfVec3 impulse, const PfVec3 columns[3]) {
    if (!pf_mode_is_dynamic(body->mode)) return;
    body->linear_velocity = pf_add(body->linear_velocity, pf_scale(impulse, body->inverse_mass));
    PfVec3 torque = pf_cross(pf_sub(point, body->position), impulse);
    body->angular_velocity = pf_add(body->angular_velocity,
        pf_add(pf_add(pf_scale(columns[0],torque.x),pf_scale(columns[1],torque.y)),pf_scale(columns[2],torque.z)));
}

// Cache masses and incoming restitution targets before any impulse is applied.
// Each geometric point owns a constraint; manifold impulses are diagnostic sums.
__device__ static inline void pf_prepare_velocity_contacts(
        PfBody* bodies, PfManifold* manifolds, int manifold_count) {
    for (int manifold_index = 0; manifold_index < manifold_count;
            ++manifold_index) {
        PfManifold* manifold = &manifolds[manifold_index];
        PfBody* body_a = &bodies[manifold->body_a];
        PfBody* body_b = &bodies[manifold->body_b];
        pf_contact_inertia(body_a, manifold->inertia_a);
        pf_contact_inertia(body_b, manifold->inertia_b);
        for (int point_index = 0; point_index < manifold->point_count; ++point_index) {
            const PfContactPoint* point = &manifold->points[point_index];
            PfContactVelocity* row = &manifold->velocity[point_index];
            *row = (PfContactVelocity){0};
            float normal_mass = pf_effective_mass(body_a, body_b,
                point->point_a, point->point_b, manifold->normal);
            row->normal_mass = normal_mass > 1.0e-8f ? 1.0f / normal_mass : 0.0f;
            if (manifold->static_friction > 0.0f || manifold->dynamic_friction > 0.0f) {
                float tangent_1_mass = pf_effective_mass(body_a, body_b,
                    point->point_a, point->point_b, manifold->tangent_1);
                float tangent_2_mass = pf_effective_mass(body_a, body_b,
                    point->point_a, point->point_b, manifold->tangent_2);
                row->tangent_1_mass = tangent_1_mass > 1.0e-8f ? 1.0f / tangent_1_mass : 0.0f;
                row->tangent_2_mass = tangent_2_mass > 1.0e-8f ? 1.0f / tangent_2_mass : 0.0f;
            }
            float incoming = pf_dot(pf_sub(pf_point_velocity(body_a, point->point_a),
                pf_point_velocity(body_b, point->point_b)), manifold->normal);
            row->restitution_bias = incoming < -PF_RESTITUTION_THRESHOLD
                ? -manifold->restitution * incoming : 0.0f;
        }
        manifold->normal_impulse = 0.0f;
        manifold->tangent_1_impulse = manifold->tangent_2_impulse = 0.0f;
        manifold->converged = 0;
    }
}

// Shared by standalone, joint-interleaved and soft-body contact passes.
__device__ static inline bool pf_solve_velocity_manifold(
        PfBody* bodies, PfManifold* manifold) {
    PfBody* body_a = &bodies[manifold->body_a];
    PfBody* body_b = &bodies[manifold->body_b];
    bool applied = false;
    manifold->normal_impulse = 0.0f;
    manifold->tangent_1_impulse = manifold->tangent_2_impulse = 0.0f;
    for (int index = 0; index < manifold->point_count; ++index) {
        const PfContactPoint* point = &manifold->points[index];
        PfContactVelocity* row = &manifold->velocity[index];
        PfVec3 relative = pf_sub(pf_point_velocity(body_a, point->point_a),
            pf_point_velocity(body_b, point->point_b));
        float old_normal = row->normal_impulse;
        row->normal_impulse = fmaxf(0.0f, old_normal
            + (row->restitution_bias - pf_dot(relative, manifold->normal)) * row->normal_mass);
        float normal_delta = row->normal_impulse - old_normal;
        if (normal_delta != 0.0f) {
            applied = true;
            PfVec3 impulse = pf_scale(manifold->normal, normal_delta);
            pf_apply_contact_impulse(body_a, point->point_a, impulse, manifold->inertia_a);
            pf_apply_contact_impulse(body_b, point->point_b, pf_scale(impulse, -1.0f), manifold->inertia_b);
        }
        manifold->normal_impulse += row->normal_impulse;
        if (row->tangent_1_mass == 0.0f && row->tangent_2_mass == 0.0f) continue;
        relative = pf_sub(pf_point_velocity(body_a, point->point_a),
            pf_point_velocity(body_b, point->point_b));
        float old_1 = row->tangent_1_impulse, old_2 = row->tangent_2_impulse;
        float candidate_1 = old_1 - pf_dot(relative, manifold->tangent_1) * row->tangent_1_mass;
        float candidate_2 = old_2 - pf_dot(relative, manifold->tangent_2) * row->tangent_2_mass;
        float length_squared = candidate_1 * candidate_1 + candidate_2 * candidate_2;
        float static_limit = manifold->static_friction * row->normal_impulse;
        if (length_squared > static_limit * static_limit) {
            float scale = fminf(1.0f, manifold->dynamic_friction * row->normal_impulse
                * rsqrtf(fmaxf(length_squared, 1.0e-20f)));
            candidate_1 *= scale;
            candidate_2 *= scale;
        }
        row->tangent_1_impulse = candidate_1;
        row->tangent_2_impulse = candidate_2;
        float delta_1 = candidate_1 - old_1, delta_2 = candidate_2 - old_2;
        if (delta_1 != 0.0f || delta_2 != 0.0f) {
            applied = true;
            PfVec3 impulse = pf_add(pf_scale(manifold->tangent_1, delta_1),
                pf_scale(manifold->tangent_2, delta_2));
            pf_apply_contact_impulse(body_a, point->point_a, impulse, manifold->inertia_a);
            pf_apply_contact_impulse(body_b, point->point_b, pf_scale(impulse, -1.0f), manifold->inertia_b);
        }
        manifold->tangent_1_impulse += candidate_1;
        manifold->tangent_2_impulse += candidate_2;
    }
    manifold->converged = !applied;
    return applied;
}

__device__ static inline void pf_solve_velocity_contacts(
        PfBody* bodies, PfManifold* manifolds, int manifold_count) {
    pf_prepare_velocity_contacts(bodies, manifolds, manifold_count);
    unsigned int carried = 0u;
    for (int iteration = 0; iteration < PF_VELOCITY_ITERATIONS; ++iteration) {
        bool applied = false;
        unsigned int touched = 0u;
        for (int manifold_index = 0; manifold_index < manifold_count;
                ++manifold_index) {
            PfManifold* manifold = &manifolds[manifold_index];
            // Only dynamic bodies can change velocity under an impulse, so
            // static and kinematic indices never need a bit.
            PfBody* body_a = &bodies[manifold->body_a];
            PfBody* body_b = &bodies[manifold->body_b];
            unsigned int body_mask = 0u;
            if (pf_mode_is_dynamic(body_a->mode)
                    && (unsigned int)manifold->body_a < PF_TOUCHED_MASK_BITS) {
                body_mask |= 1u << manifold->body_a;
            }
            if (pf_mode_is_dynamic(body_b->mode)
                    && (unsigned int)manifold->body_b < PF_TOUCHED_MASK_BITS) {
                body_mask |= 1u << manifold->body_b;
            }
            bool maskable = ((unsigned int)manifold->body_a
                | (unsigned int)manifold->body_b) < PF_TOUCHED_MASK_BITS;
            if (manifold->converged != 0 && maskable
                    && (body_mask & (touched | carried)) == 0u) {
                continue;
            }
            if (pf_solve_velocity_manifold(bodies, manifold)) {
                applied = true;
                touched |= body_mask;
            }
        }
        carried = touched;
        if (!applied) {
            break;
        }
    }
}
