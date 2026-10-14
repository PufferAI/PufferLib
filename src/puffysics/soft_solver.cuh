#pragma once

#include <cuda_runtime.h>

#include "collision_sphere.cuh"
#include "contact_solver.cuh"
#include "collision.cuh"
#include "world.cuh"

#define PF_SOFT_CONSTRAINT_DISTANCE 0
#define PF_SOFT_CONSTRAINT_VOLUME 1
#define PF_SOFT_SOLVER_BLOCK_SIZE 256
#define PF_SOFT_CONSTRAINT_CONTACT 2
#define PF_SOFT_CONTACT_MIN_DISTANCE 0.02f
#define PF_SOFT_CONTACT_CUTOFF 0.03f
#define PF_SOFT_CONTACT_COMPLIANCE 1.0e-7f
#define PF_SOFT_STRAIN_STIFFENING 20.0f
#define PF_SOFT_MIN_STRAIN_EXPONENT (-20.0f)
#define PF_SOFT_DEFAULT_DAMPING 0.0f
#define PF_SOFT_RIGID_CONTACT_MARGIN 0.02f

typedef struct PfSoftConstraintItem {
    int type;
    int index;
} PfSoftConstraintItem;

typedef struct PfSoftColoring {
    PfSoftConstraintItem* constraints;
    int* offsets;
    int* bucket_sizes;
    int constraint_count;
    int contact_count;
    int bucket_count;
} PfSoftColoring;

typedef struct PfSoftContactConstraint {
    int particle_a;
    int particle_b;
    float minimum_distance;
    float compliance;
    float lambda;
} PfSoftContactConstraint;

typedef struct PfSoftRigidContact {
    int particle_index;
    int body_index;
    PfVec3 normal;
    PfVec3 body_point;
    float separation;
    float normal_impulse;
} PfSoftRigidContact;
__device__ static inline void pf_soft_reset_lambdas(
        PfSoftWorld world, PfSoftContactConstraint* contacts,
        PfSoftColoring coloring, int lane, int lane_count);
__device__ static inline void pf_soft_project_color(
        PfSoftWorld world, PfSoftContactConstraint* contacts,
        PfSoftColoring coloring, int color, float alpha_scale,
        float strain_stiffening, int lane, int lane_count);

__device__ static inline int pf_detect_soft_rigid_contacts(
        PfSoftWorld soft, PfBody* bodies, int body_count,
        PfSoftRigidContact* contacts, int contact_capacity) {
    int count = 0;
    for (int particle = 0; particle < soft.particle_count; ++particle) {
        for (int body = 0; body < body_count; ++body) {
            PfPointQuery query;
            bool hit = bodies[body].shape == PF_SPHERE
                ? pf_point_sphere_query(soft.particles[particle].position,
                    bodies[body].position, bodies[body].half_extents.x, &query)
                : pf_point_box_query(soft.particles[particle].position,
                    &bodies[body], &query);
            if (query.separation <= PF_SOFT_RIGID_CONTACT_MARGIN
                    && count < contact_capacity) {
                contacts[count++] = (PfSoftRigidContact){particle, body,
                    query.normal, query.closest_point, query.separation, 0.0f};
            }
        }
    }
    return count;
}

__device__ static inline float pf_soft_rigid_effective_mass(
        PfBody* body, PfVec3 point, PfVec3 direction) {
    if (!pf_mode_is_dynamic(body->mode)) {
        return 0.0f;
    }
    PfVec3 lever = pf_cross(pf_sub(point, body->position), direction);
    return body->inverse_mass + pf_dot(lever,
        pf_inverse_inertia_world(body, lever));
}

__device__ static inline void pf_solve_rigid_manifold_row(
        PfBody* bodies, PfManifold* manifold) {
    pf_solve_velocity_manifold(bodies, manifold);
}

__device__ static inline void pf_prepare_soft_rigid_contacts(
        PfSoftRigidContact* contacts, int count) {
    for (int index = 0; index < count; ++index) {
        contacts[index].normal_impulse = 0.0f;
    }
}

__device__ static inline void pf_solve_soft_rigid_contact_row(
        PfSoftWorld soft, PfBody* bodies, PfSoftRigidContact* contact,
        float dt) {
    PfBody* body = &bodies[contact->body_index];
    PfVec3 direction = contact->normal;
    float particle_mass = soft.particles[contact->particle_index].inverse_mass;
    float mass = particle_mass + pf_soft_rigid_effective_mass(body,
        contact->body_point, direction);
    if (mass <= 1.0e-8f) {
        return;
    }
    PfVec3 relative_velocity = pf_sub(
        soft.particles[contact->particle_index].velocity,
        pf_point_velocity(body, contact->body_point));
    float normal_velocity = pf_dot(relative_velocity, direction);
    float old_impulse = contact->normal_impulse;
    float candidate = old_impulse - normal_velocity / mass;
    candidate = candidate < 0.0f ? 0.0f : candidate;
    contact->normal_impulse = candidate;
    float delta = candidate - old_impulse;
    if (delta == 0.0f) {
        return;
    }
    PfVec3 impulse = pf_scale(direction, delta);
    if (particle_mass > 0.0f) {
        PfVec3 kick = pf_scale(impulse, particle_mass);
        soft.particles[contact->particle_index].velocity = pf_add(
            soft.particles[contact->particle_index].velocity, kick);
        /* Velocity is reconstructed from the position delta at the end of
         * the substep, so the particle half of the impulse has to travel
         * into the position stream or the pair loses its momentum. */
        soft.particles[contact->particle_index].position = pf_add(
            soft.particles[contact->particle_index].position,
            pf_scale(kick, dt));
    }
    pf_apply_impulse(body, contact->body_point, pf_scale(impulse, -1.0f));
}

__device__ static inline void pf_project_soft_rigid_contact_positions(
        PfSoftWorld soft, PfBody* bodies, PfSoftRigidContact* contacts,
        int contact_count) {
    for (int index = 0; index < contact_count; ++index) {
        PfSoftRigidContact& contact = contacts[index];
        PfBody* body = &bodies[contact.body_index];
        PfVec3 direction = contact.normal;
        float particle_mass = soft.particles[contact.particle_index].inverse_mass;
        float body_mass = pf_mode_is_dynamic(body->mode)
            ? body->inverse_mass : 0.0f;
        float mass = particle_mass + body_mass;
        if (mass <= 1.0e-8f || contact.separation >= 0.0f) {
            continue;
        }
        float correction = fminf(-contact.separation, 0.01f);
        soft.particles[contact.particle_index].position = pf_add(
            soft.particles[contact.particle_index].position,
            pf_scale(direction, correction * particle_mass / mass));
        /* Closing speed is relative to the contact point, otherwise a
         * receding body still takes the particle's share of the impulse. */
        float normal_velocity = pf_dot(pf_sub(
            soft.particles[contact.particle_index].velocity,
            pf_point_velocity(body, contact.body_point)), direction);
        if (normal_velocity < 0.0f) {
            soft.particles[contact.particle_index].velocity = pf_sub(
                soft.particles[contact.particle_index].velocity,
                pf_scale(direction, normal_velocity));
        }
        if (body_mass > 0.0f) {
            body->position = pf_sub(body->position,
                pf_scale(direction, correction * body_mass / mass));
        }
        contact.separation += correction;
    }
}

/* Deterministic combined row order: rigid manifolds, colored soft rows,
 * then soft-rigid rows. The rigid-only caller never enters this helper. */
__device__ static inline void pf_solve_velocity_contacts_and_soft(
        PfWorld* rigid, PfSoftWorld soft, PfSoftColoring coloring,
        PfSoftContactConstraint* self_contacts,
        PfSoftRigidContact* soft_rigid_contacts, int soft_rigid_count,
        float dt, int iterations, float strain_stiffening) {
    pf_prepare_velocity_contacts(rigid->bodies, rigid->manifolds,
        rigid->manifold_count);
    pf_prepare_soft_rigid_contacts(soft_rigid_contacts, soft_rigid_count);
    float alpha_scale = dt * dt;
    pf_soft_reset_lambdas(soft, self_contacts, coloring, 0, 1);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int index = 0; index < rigid->manifold_count; ++index) {
            PfManifold* manifold = &rigid->manifolds[index];
            pf_solve_rigid_manifold_row(rigid->bodies, manifold);
        }
        for (int color = 0; color < coloring.bucket_count; ++color) {
            pf_soft_project_color(soft, self_contacts, coloring, color,
                alpha_scale, strain_stiffening, 0, 1);
        }
        for (int index = 0; index < soft_rigid_count; ++index) {
            pf_solve_soft_rigid_contact_row(soft, rigid->bodies,
                &soft_rigid_contacts[index], dt);
        }
    }
}

__global__ static void pf_soft_build_contacts_kernel(
        const PfSoftParticle* particles, int particle_count,
        PfSoftContactConstraint* contacts, int contact_capacity,
        float cutoff, float compliance, int* contact_count) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int count = 0;
    for (int a = 0; a < particle_count; ++a) {
        for (int b = a + 1; b < particle_count; ++b) {
            if (count >= contact_capacity) {
                *contact_count = -1;
                return;
            }
            if (pf_length_squared(pf_sub(particles[b].position,
                    particles[a].position)) <= cutoff * cutoff) {
                contacts[count++] = (PfSoftContactConstraint){a, b,
                    PF_SOFT_CONTACT_MIN_DISTANCE, compliance, 0.0f};
            }
        }
    }
    *contact_count = count;
}

static inline bool pf_soft_contacts_create(
        const PfSoftParticle* particles, int particle_count,
        PfSoftContactConstraint** out, int* out_count) {
    if (particles == NULL || particle_count <= 0 || out == NULL
            || out_count == NULL) {
        return false;
    }
    *out = NULL;
    *out_count = 0;
    size_t pair_count = (size_t)particle_count * (particle_count - 1) / 2;
    if (pair_count > 0x7fffffffu) {
        return false;
    }
    int capacity = (int)pair_count;
    int* device_count = NULL;
    bool ok = cudaMalloc((void**)out,
        (size_t)capacity * sizeof(PfSoftContactConstraint)) == cudaSuccess
        && cudaMalloc((void**)&device_count, sizeof(int)) == cudaSuccess;
    if (ok) {
        pf_soft_build_contacts_kernel<<<1, 1>>>(particles, particle_count,
            *out, capacity, PF_SOFT_CONTACT_CUTOFF,
            PF_SOFT_CONTACT_COMPLIANCE, device_count);
        ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize()
            == cudaSuccess && cudaMemcpy(out_count, device_count, sizeof(int),
                cudaMemcpyDeviceToHost) == cudaSuccess && *out_count >= 0;
    }
    cudaFree(device_count);
    if (!ok) {
        cudaFree(*out);
        *out = NULL;
        *out_count = 0;
    }
    return ok;
}

__device__ static inline bool pf_soft_items_conflict(
        const PfSoftDistanceConstraint* distances,
        const PfSoftVolumeConstraint* volumes,
        const PfSoftContactConstraint* contacts,
        PfSoftConstraintItem a, PfSoftConstraintItem b) {
    int a_vertices[4];
    int a_count;
    if (a.type == PF_SOFT_CONSTRAINT_DISTANCE) {
        a_vertices[0] = distances[a.index].particle_a;
        a_vertices[1] = distances[a.index].particle_b;
        a_count = 2;
    } else if (a.type == PF_SOFT_CONSTRAINT_VOLUME) {
        a_vertices[0] = volumes[a.index].p0;
        a_vertices[1] = volumes[a.index].p1;
        a_vertices[2] = volumes[a.index].p2;
        a_vertices[3] = volumes[a.index].p3;
        a_count = 4;
    } else {
        a_vertices[0] = contacts[a.index].particle_a;
        a_vertices[1] = contacts[a.index].particle_b;
        a_count = 2;
    }
    int b_vertices[4];
    int b_count;
    if (b.type == PF_SOFT_CONSTRAINT_DISTANCE) {
        b_vertices[0] = distances[b.index].particle_a;
        b_vertices[1] = distances[b.index].particle_b;
        b_count = 2;
    } else if (b.type == PF_SOFT_CONSTRAINT_VOLUME) {
        b_vertices[0] = volumes[b.index].p0;
        b_vertices[1] = volumes[b.index].p1;
        b_vertices[2] = volumes[b.index].p2;
        b_vertices[3] = volumes[b.index].p3;
        b_count = 4;
    } else {
        b_vertices[0] = contacts[b.index].particle_a;
        b_vertices[1] = contacts[b.index].particle_b;
        b_count = 2;
    }
    for (int i = 0; i < a_count; ++i) {
        for (int j = 0; j < b_count; ++j) {
            if (a_vertices[i] == b_vertices[j]) {
                return true;
            }
        }
    }
    return false;
}

__device__ static inline PfSoftConstraintItem pf_soft_constraint_item(
        int index, int distance_count, int volume_count) {
    if (index < distance_count) {
        return (PfSoftConstraintItem){PF_SOFT_CONSTRAINT_DISTANCE, index};
    }
    index -= distance_count;
    if (index < volume_count) {
        return (PfSoftConstraintItem){PF_SOFT_CONSTRAINT_VOLUME, index};
    }
    return (PfSoftConstraintItem){PF_SOFT_CONSTRAINT_CONTACT,
        index - volume_count};
}

/*
 * One device thread inserts constraints in fixed order. At insertion i, earlier
 * constraints occupy at most i colors, so color i is necessarily available.
 * The loop bound <= i is therefore a deterministic termination guarantee.
 */
__global__ static void pf_soft_build_coloring_kernel(
        const PfSoftDistanceConstraint* distances, int distance_count,
        const PfSoftVolumeConstraint* volumes, int volume_count,
        const PfSoftContactConstraint* contacts, int contact_count,
        int particle_count, PfSoftConstraintItem* ordered,
        int* offsets, int* bucket_sizes, int* item_colors, int* fill,
        int* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    *result = -1;
    int count = distance_count + volume_count + contact_count;
    if (count < distance_count || count < volume_count
            || count < contact_count || particle_count < 0) {
        return;
    }
    for (int index = 0; index < distance_count; ++index) {
        const PfSoftDistanceConstraint& q = distances[index];
        if (q.particle_a < 0 || q.particle_b < 0
                || q.particle_a >= particle_count
                || q.particle_b >= particle_count
                || q.particle_a == q.particle_b) {
            return;
        }
    }
    for (int index = 0; index < volume_count; ++index) {
        const PfSoftVolumeConstraint& q = volumes[index];
        if (q.p0 < 0 || q.p1 < 0 || q.p2 < 0 || q.p3 < 0
                || q.p0 >= particle_count || q.p1 >= particle_count
                || q.p2 >= particle_count || q.p3 >= particle_count
                || q.p0 == q.p1 || q.p0 == q.p2 || q.p0 == q.p3
                || q.p1 == q.p2 || q.p1 == q.p3 || q.p2 == q.p3) {
            return;
        }
    }
    for (int index = 0; index < contact_count; ++index) {
        const PfSoftContactConstraint& q = contacts[index];
        if (q.particle_a < 0 || q.particle_b < 0
                || q.particle_a >= particle_count
                || q.particle_b >= particle_count
                || q.particle_a == q.particle_b
                || !pf_number(q.minimum_distance)
                || q.minimum_distance <= 0.0f
                || !pf_number(q.compliance) || q.compliance < 0.0f) {
            return;
        }
    }
    for (int color = 0; color < count; ++color) {
        bucket_sizes[color] = 0;
        fill[color] = 0;
    }
    offsets[0] = 0;
    for (int i = 0; i < count; ++i) {
        PfSoftConstraintItem item = pf_soft_constraint_item(i,
            distance_count, volume_count);
        for (int color = 0; color <= i; ++color) {
            bool available = true;
            for (int previous = 0; previous < i && available; ++previous) {
                if (item_colors[previous] == color) {
                    PfSoftConstraintItem other = pf_soft_constraint_item(previous,
                        distance_count, volume_count);
                    available = !pf_soft_items_conflict(distances, volumes,
                        contacts, item, other);
                }
            }
            if (available) {
                item_colors[i] = color;
                ++bucket_sizes[color];
                break;
            }
        }
    }
    int bucket_count = 0;
    for (int color = 0; color < count; ++color) {
        offsets[color + 1] = offsets[color] + bucket_sizes[color];
        if (bucket_sizes[color] != 0) {
            bucket_count = color + 1;
        }
    }
    for (int i = 0; i < count; ++i) {
        int color = item_colors[i];
        PfSoftConstraintItem item = pf_soft_constraint_item(i,
            distance_count, volume_count);
        ordered[offsets[color] + fill[color]++] = item;
    }
    *result = bucket_count;
}

static inline void pf_soft_coloring_destroy(PfSoftColoring* coloring) {
    if (coloring == NULL) {
        return;
    }
    cudaFree(coloring->constraints);
    cudaFree(coloring->offsets);
    cudaFree(coloring->bucket_sizes);
    *coloring = {};
}

static inline bool pf_soft_coloring_create(
        const PfSoftDistanceConstraint* distances, int distance_count,
        const PfSoftVolumeConstraint* volumes, int volume_count,
        const PfSoftContactConstraint* contacts, int contact_count,
        int particle_count, PfSoftColoring* out) {
    if (out == NULL || distance_count < 0 || volume_count < 0
            || contact_count < 0 || particle_count < 0
            || distance_count > 0x7fffffff - volume_count
            || distance_count + volume_count > 0x7fffffff - contact_count
            || (distance_count > 0 && distances == NULL)
            || (volume_count > 0 && volumes == NULL)
            || (contact_count > 0 && contacts == NULL)) {
        return false;
    }
    *out = {};
    int count = distance_count + volume_count + contact_count;
    int scratch_count = count > 0 ? count : 1;
    int* item_colors = NULL;
    int* fill = NULL;
    int* device_status = NULL;
    int status = 0;
    bool ok = cudaMalloc((void**)&out->constraints,
        (size_t)scratch_count * sizeof(PfSoftConstraintItem)) == cudaSuccess
        && cudaMalloc((void**)&out->offsets,
            (size_t)(count + 1) * sizeof(int)) == cudaSuccess
        && cudaMalloc((void**)&out->bucket_sizes,
            (size_t)scratch_count * sizeof(int)) == cudaSuccess
        && cudaMalloc((void**)&item_colors,
            (size_t)scratch_count * sizeof(int)) == cudaSuccess
        && cudaMalloc((void**)&fill,
            (size_t)scratch_count * sizeof(int)) == cudaSuccess
        && cudaMalloc((void**)&device_status, sizeof(int)) == cudaSuccess;
    if (ok) {
        ok = cudaMemset(device_status, 0xff, sizeof(int)) == cudaSuccess;
    }
    if (ok) {
        pf_soft_build_coloring_kernel<<<1, 1>>>(distances, distance_count,
            volumes, volume_count, contacts, contact_count, particle_count,
            out->constraints, out->offsets, out->bucket_sizes, item_colors, fill,
            device_status);
        ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize()
            == cudaSuccess && cudaMemcpy(&status, device_status, sizeof(int),
                cudaMemcpyDeviceToHost) == cudaSuccess && status >= 0
            && status <= count;
    }
    cudaFree(item_colors);
    cudaFree(fill);
    cudaFree(device_status);
    if (!ok) {
        pf_soft_coloring_destroy(out);
        return false;
    }
    out->constraint_count = count;
    out->contact_count = contact_count;
    out->bucket_count = status;
    return true;
}

/* Caller-owned force scratch lives only for the current substep launch. */
__device__ static inline void pf_soft_accumulate_particle_force(
        PfVec3* accumulated, int particle_index, int particle_count,
        PfVec3 contribution) {
    if (accumulated != NULL && particle_index >= 0
            && particle_index < particle_count) {
        accumulated[particle_index] = pf_add(accumulated[particle_index],
            contribution);
    }
}
__device__ static inline void pf_soft_project_distance(
        PfSoftWorld world, PfSoftConstraintItem item, float alpha_scale,
        float strain_stiffening) {
    PfSoftDistanceConstraint* constraint =
        &world.distance_constraints[item.index];
    int a = constraint->particle_a;
    int b = constraint->particle_b;
    float wa = world.particles[a].inverse_mass;
    float wb = world.particles[b].inverse_mass;
    float weight = wa + wb;
    if (weight <= 0.0f) {
        return;
    }
    PfVec3 delta = pf_sub(world.particles[b].position,
        world.particles[a].position);
    float length = pf_length(delta);
    PfVec3 direction = pf_normalize_or(delta, pf_v3(0.0f, 1.0f, 0.0f));
    float strain = fabsf(length - constraint->rest_length)
        / constraint->rest_length;
    float exponent = strain_stiffening > 0.0f
        ? fmaxf(PF_SOFT_MIN_STRAIN_EXPONENT, -strain_stiffening * strain)
        : 0.0f;
    float effective_compliance = constraint->compliance * expf(exponent);
    float alpha_tilde = effective_compliance / alpha_scale;
    float delta_lambda = (-(length - constraint->rest_length)
        - alpha_tilde * constraint->lambda) / (weight + alpha_tilde);
    constraint->lambda += delta_lambda;
    if (wa > 0.0f) {
        world.particles[a].position = pf_sub(world.particles[a].position,
            pf_scale(direction, wa * delta_lambda));
    }
    if (wb > 0.0f) {
        world.particles[b].position = pf_add(world.particles[b].position,
            pf_scale(direction, wb * delta_lambda));
    }
}

__device__ static inline void pf_soft_project_volume(
        PfSoftWorld world, PfSoftConstraintItem item, float alpha_tilde) {
    PfSoftVolumeConstraint* constraint = &world.volume_constraints[item.index];
    int p0 = constraint->p0;
    int p1 = constraint->p1;
    int p2 = constraint->p2;
    int p3 = constraint->p3;
    float w0 = world.particles[p0].inverse_mass;
    float w1 = world.particles[p1].inverse_mass;
    float w2 = world.particles[p2].inverse_mass;
    float w3 = world.particles[p3].inverse_mass;
    float weight = w0 + w1 + w2 + w3;
    if (weight <= 0.0f) {
        return;
    }
    PfVec3 a = world.particles[p0].position;
    PfVec3 b = world.particles[p1].position;
    PfVec3 c = world.particles[p2].position;
    PfVec3 d = world.particles[p3].position;
    PfVec3 ab = pf_sub(b, a);
    PfVec3 ac = pf_sub(c, a);
    PfVec3 ad = pf_sub(d, a);
    PfVec3 g1 = pf_scale(pf_cross(ac, ad), 1.0f / 6.0f);
    PfVec3 g2 = pf_scale(pf_cross(ad, ab), 1.0f / 6.0f);
    PfVec3 g3 = pf_scale(pf_cross(ab, ac), 1.0f / 6.0f);
    PfVec3 g0 = pf_scale(pf_add(pf_add(g1, g2), g3), -1.0f);
    float denominator = w0 * pf_dot(g0, g0) + w1 * pf_dot(g1, g1)
        + w2 * pf_dot(g2, g2) + w3 * pf_dot(g3, g3) + alpha_tilde;
    if (denominator <= 0.0f) {
        return;
    }
    float volume = pf_dot(ab, pf_cross(ac, ad)) / 6.0f;
    float delta_lambda = (-(volume - constraint->rest_volume)
        - alpha_tilde * constraint->lambda) / denominator;
    constraint->lambda += delta_lambda;
    if (w0 > 0.0f) {
        world.particles[p0].position = pf_add(a, pf_scale(g0, w0 * delta_lambda));
    }
    if (w1 > 0.0f) {
        world.particles[p1].position = pf_add(b, pf_scale(g1, w1 * delta_lambda));
    }
    if (w2 > 0.0f) {
        world.particles[p2].position = pf_add(c, pf_scale(g2, w2 * delta_lambda));
    }
    if (w3 > 0.0f) {
        world.particles[p3].position = pf_add(d, pf_scale(g3, w3 * delta_lambda));
    }
}

/* Self-contact rows carry a mutable XPBD multiplier, so every environment
 * needs its own block. Topology is per environment and laid out like
 * particles, distance and volume constraints; the colouring stays shared. */
__device__ static inline PfSoftContactConstraint* pf_soft_env_contacts(
        PfSoftContactConstraint* contacts, PfSoftColoring coloring,
        int environment) {
    if (contacts == NULL || coloring.contact_count <= 0) {
        return contacts;
    }
    return contacts + (size_t)environment * (size_t)coloring.contact_count;
}

__device__ static inline void pf_soft_project_contact(
        PfSoftWorld world, PfSoftContactConstraint* contacts,
        PfSoftConstraintItem item, float alpha_scale) {
    PfSoftContactConstraint* constraint = &contacts[item.index];
    int a = constraint->particle_a;
    int b = constraint->particle_b;
    float wa = world.particles[a].inverse_mass;
    float wb = world.particles[b].inverse_mass;
    float weight = wa + wb;
    if (weight <= 0.0f) {
        return;
    }
    PfVec3 delta = pf_sub(world.particles[b].position,
        world.particles[a].position);
    float length = pf_length(delta);
    if (length >= constraint->minimum_distance) {
        constraint->lambda = 0.0f;
        return;
    }
    PfVec3 direction = pf_normalize_or(delta, pf_v3(0.0f, 1.0f, 0.0f));
    float alpha_tilde = constraint->compliance / alpha_scale;
    float delta_lambda = (-(length - constraint->minimum_distance)
        - alpha_tilde * constraint->lambda) / (weight + alpha_tilde);
    constraint->lambda += delta_lambda;
    if (wa > 0.0f) {
        world.particles[a].position = pf_sub(world.particles[a].position,
            pf_scale(direction, wa * delta_lambda));
    }
    if (wb > 0.0f) {
        world.particles[b].position = pf_add(world.particles[b].position,
            pf_scale(direction, wb * delta_lambda));
    }
}
__device__ static inline void pf_soft_project_item(
        PfSoftWorld world, PfSoftContactConstraint* contacts,
        PfSoftConstraintItem item, float alpha_scale,
        float strain_stiffening) {
    if (item.type == PF_SOFT_CONSTRAINT_DISTANCE) {
        pf_soft_project_distance(world, item, alpha_scale, strain_stiffening);
    } else if (item.type == PF_SOFT_CONSTRAINT_VOLUME) {
        float alpha_tilde = world.volume_constraints[item.index].compliance
            / alpha_scale;
        pf_soft_project_volume(world, item, alpha_tilde);
    } else {
        pf_soft_project_contact(world, contacts, item, alpha_scale);
    }
}

__device__ static inline void pf_soft_predict(
        PfSoftWorld world, PfVec3 gravity, const PfVec3* forces, float dt,
        int lane, int lane_count) {
    for (int i = lane; i < world.particle_count; i += lane_count) {
        PfSoftParticle* particle = &world.particles[i];
        particle->previous_position = particle->position;
        if (particle->inverse_mass > 0.0f) {
            PfVec3 acceleration = gravity;
            if (forces != NULL) {
                acceleration = pf_add(acceleration, pf_scale(forces[
                    (size_t)world.env_index * world.particle_count + i],
                    particle->inverse_mass));
            }
            particle->velocity = pf_add(particle->velocity,
                pf_scale(acceleration, dt));
            particle->position = pf_add(particle->position,
                pf_scale(particle->velocity, dt));
        } else {
            particle->velocity = pf_v3(0.0f, 0.0f, 0.0f);
        }
    }
}

__device__ static inline void pf_soft_finish(
        PfSoftWorld world, float inverse_dt, int lane, int lane_count) {
    for (int i = lane; i < world.particle_count; i += lane_count) {
        PfSoftParticle* particle = &world.particles[i];
        if (particle->inverse_mass > 0.0f) {
            particle->velocity = pf_scale(pf_sub(particle->position,
                particle->previous_position), inverse_dt);
        }
    }
}

__device__ static inline void pf_soft_finish_damped(
        PfSoftWorld world, float dt, float inverse_dt, float damping,
        int lane, int lane_count) {
    pf_soft_finish(world, inverse_dt, lane, lane_count);
    float factor = damping > 0.0f ? expf(-damping * dt) : 1.0f;
    for (int i = lane; i < world.particle_count; i += lane_count) {
        if (world.particles[i].inverse_mass > 0.0f) {
            world.particles[i].velocity = pf_scale(
                world.particles[i].velocity, factor);
        }
    }
}

__device__ static inline void pf_soft_reset_lambdas(
        PfSoftWorld world, PfSoftContactConstraint* contacts,
        PfSoftColoring coloring, int lane, int lane_count) {
    for (int i = lane; i < coloring.constraint_count; i += lane_count) {
        PfSoftConstraintItem item = coloring.constraints[i];
        if (item.type == PF_SOFT_CONSTRAINT_DISTANCE) {
            world.distance_constraints[item.index].lambda = 0.0f;
        } else if (item.type == PF_SOFT_CONSTRAINT_VOLUME) {
            world.volume_constraints[item.index].lambda = 0.0f;
        } else {
            contacts[item.index].lambda = 0.0f;
        }
    }
}

__device__ static inline void pf_soft_project_color(
        PfSoftWorld world, PfSoftContactConstraint* contacts,
        PfSoftColoring coloring, int color, float alpha_scale,
        float strain_stiffening, int lane, int lane_count) {
    for (int i = coloring.offsets[color] + lane;
            i < coloring.offsets[color + 1]; i += lane_count) {
        pf_soft_project_item(world, contacts, coloring.constraints[i],
            alpha_scale, strain_stiffening);
    }
}

__global__ static void pf_soft_substep_block_kernel(
        PfSoftWorld* worlds, int environment_count, PfSoftColoring coloring,
        PfSoftContactConstraint* contacts, PfVec3 gravity,
        const PfVec3* forces, float dt, int substeps, int iterations,
        float strain_stiffening, float damping) {
    int environment = blockIdx.x;
    if (environment >= environment_count) {
        return;
    }
    PfSoftWorld world = worlds[environment];
    PfSoftContactConstraint* self_contacts = pf_soft_env_contacts(contacts,
        coloring, environment);
    float h = dt / (float)substeps;
    float alpha_scale = h * h;
    for (int substep = 0; substep < substeps; ++substep) {
        pf_soft_predict(world, gravity, forces, h, threadIdx.x, blockDim.x);
        __syncthreads();
        pf_soft_reset_lambdas(world, self_contacts, coloring, threadIdx.x,
            blockDim.x);
        __syncthreads();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (int color = 0; color < coloring.bucket_count; ++color) {
                pf_soft_project_color(world, self_contacts, coloring, color,
                    alpha_scale, strain_stiffening, threadIdx.x, blockDim.x);
                __syncthreads();
            }
        }
        pf_soft_finish_damped(world, h, 1.0f / h, damping,
            threadIdx.x, blockDim.x);
        __syncthreads();
    }
}

__global__ static void pf_soft_substep_serial_kernel(
        PfSoftWorld* worlds, int environment_count, PfSoftColoring coloring,
        PfSoftContactConstraint* contacts, PfVec3 gravity,
        const PfVec3* forces, float dt, int substeps, int iterations,
        float strain_stiffening, float damping) {
    int environment = blockIdx.x;
    if (environment >= environment_count) {
        return;
    }
    PfSoftWorld world = worlds[environment];
    PfSoftContactConstraint* self_contacts = pf_soft_env_contacts(contacts,
        coloring, environment);
    float h = dt / (float)substeps;
    float alpha_scale = h * h;
    for (int substep = 0; substep < substeps; ++substep) {
        pf_soft_predict(world, gravity, forces, h, 0, 1);
        pf_soft_reset_lambdas(world, self_contacts, coloring, 0, 1);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (int color = 0; color < coloring.bucket_count; ++color) {
                pf_soft_project_color(world, self_contacts, coloring, color,
                    alpha_scale, strain_stiffening, 0, 1);
            }
        }
        pf_soft_finish_damped(world, h, 1.0f / h, damping, 0, 1);
    }
}

/* Pointer form keeps manifold/contact counts observable to soft callers. */
__device__ static inline bool pf_soft_step(
        PfWorld* rigid, PfSoftWorld* soft, PfSoftColoring coloring,
        PfSoftContactConstraint* self_contacts,
        PfSoftRigidContact* rigid_contacts, int rigid_contact_capacity,
        int* rigid_contact_count, PfVec3 gravity, const PfVec3* forces,
        float dt, int substeps, int iterations, float strain_stiffening,
        float damping) {
    if (rigid == NULL || rigid_contact_count == NULL || !pf_number(dt)
            || dt <= 0.0f || substeps <= 0 || iterations <= 0) {
        return false;
    }
    if (soft == NULL) {
        return pf_step(*rigid, gravity, dt, substeps);
    }
    if (soft->particles == NULL || soft->particle_count <= 0
            || soft->particle_count > soft->particle_capacity
            || soft->distance_count < 0
            || soft->distance_count > soft->distance_capacity
            || soft->volume_count < 0
            || soft->volume_count > soft->volume_capacity
            || rigid->body_count < 0
            || (rigid->body_count > 0 && rigid->bodies == NULL)
            || rigid_contact_capacity < 0
            || (rigid_contact_capacity > 0 && rigid_contacts == NULL)) {
        return false;
    }
    self_contacts = pf_soft_env_contacts(self_contacts, coloring,
        soft->env_index);
    rigid->manifold_count = 0;
    bool has_dynamic = false;
    for (int index = 0; index < rigid->body_count; ++index) {
        has_dynamic = has_dynamic || pf_mode_is_dynamic(rigid->bodies[index].mode);
    }
    float h = dt / (float)substeps;
    for (int substep = 0; substep < substeps; ++substep) {
        pf_integrate(rigid->bodies, rigid->body_count, gravity, h);
        pf_soft_predict(*soft, gravity, forces, h, 0, 1);
        if (rigid->body_count > 1 && has_dynamic) {
            pf_detect_contacts(rigid);
        }
        *rigid_contact_count = pf_detect_soft_rigid_contacts(*soft,
            rigid->bodies, rigid->body_count, rigid_contacts,
            rigid_contact_capacity);
        pf_solve_velocity_contacts_and_soft(rigid, *soft, coloring,
            self_contacts, rigid_contacts, *rigid_contact_count, h, iterations,
            strain_stiffening);
        if (rigid->manifold_count > 0) {
            pf_solve_positions(rigid);
        }
        for (int color = 0; color < coloring.bucket_count; ++color) {
            pf_soft_project_color(*soft, self_contacts, coloring, color,
                h * h, strain_stiffening, 0, 1);
        }
        pf_soft_finish_damped(*soft, h, 1.0f / h, damping, 0, 1);
        pf_project_soft_rigid_contact_positions(*soft, rigid->bodies,
            rigid_contacts, *rigid_contact_count);
    }
    if (rigid->bodies != NULL) {
        pf_clear_forces(rigid->bodies, rigid->body_count);
    }
    return true;
}
