#pragma once

#include "math.cuh"

#define PF_POSITION_ITERATIONS 4
#define PF_POSITION_SLOP 0.005f
#define PF_POSITION_PERCENT 0.05f
#define PF_MAX_POSITION_CORRECTION 0.01f

// Corrects each manifold once per iteration at its deepest point. The
// correction is a rigid translation of the two centres, so the denominator is
// the sum of the dynamic inverse masses and nothing else: the lever arms that
// `pf_effective_mass` adds are rotational terms that a translation-only
// correction never realises, and dividing by the full effective mass
// under-corrects a body by exactly that rotational share. With one dynamic
// body the whole penetration is removed; with two it is split in inverse-mass
// proportion, which is the whole point of a mass-weighted correction.
//
// The stored separation is advanced by the ACTUAL relative translation
// projected on the normal, not by the nominal `correction`. The two differ
// whenever a static or kinematic partner is present, or whenever the
// effective-mass guard skips, and assuming the nominal amount leaves the
// reported penetration disagreeing with the geometry the solver just
// produced. Every point of the manifold moves by that same projection,
// because a translation is rigid, so one scalar updates them all
// consistently. Detection rebuilds every manifold from scratch at the start
// of the next substep, so no other manifold is stale: this pass only moves
// centres, and a manifold that shares a body is corrected against the
// positions this same pass has already written.
__device__ static inline void pf_solve_positions(PfWorld* world) {
    for (int iteration = 0; iteration < PF_POSITION_ITERATIONS; ++iteration) {
        bool corrected = false;
        for (int manifold_index = 0; manifold_index < world->manifold_count;
                ++manifold_index) {
            PfManifold* manifold = &world->manifolds[manifold_index];
            if (manifold->point_count <= 0) {
                continue;
            }
            int deepest = -1;
            float deepest_depth = 0.0f;
            for (int point_index = 0; point_index < manifold->point_count;
                    ++point_index) {
                float depth = -manifold->points[point_index].separation
                    - PF_POSITION_SLOP;
                if (depth > deepest_depth) {
                    deepest_depth = depth;
                    deepest = point_index;
                }
            }
            if (deepest < 0) {
                continue;
            }
            PfBody* body_a = &world->bodies[manifold->body_a];
            PfBody* body_b = &world->bodies[manifold->body_b];
            PfVec3 normal = manifold->normal;
            float inverse_a = pf_mode_is_dynamic(body_a->mode)
                ? body_a->inverse_mass : 0.0f;
            float inverse_b = pf_mode_is_dynamic(body_b->mode)
                ? body_b->inverse_mass : 0.0f;
            float inverse_sum = inverse_a + inverse_b;
            if (inverse_sum <= 1.0e-8f) {
                continue;
            }
            float correction = fminf(deepest_depth * PF_POSITION_PERCENT,
                PF_MAX_POSITION_CORRECTION);
            // Each body takes its inverse-mass share OF THE TOTAL, so the two
            // translations sum to exactly `correction` whatever the masses.
            // Without the division the pair is displaced by
            // `correction * inverse_sum` instead, which is mass dependent: a
            // mass-10 body against a static floor has inverse_sum = 0.1 and
            // receives one tenth of the requested depth, while a mass-1 body
            // gets it in full, and a light body against a heavy one is
            // over-corrected. Penetration recovery then scales with 1/mass
            // and quietly stops working for exactly the heavy links that
            // need it.
            // Moving a body moves every contact attached to it, not just the
            // one being corrected here, so the translation is propagated to
            // every manifold that involves either body. Reconstructing each
            // affected contact from the transforms instead would mean running
            // the narrow phase again for them, which is the cost this pass
            // exists to avoid, and would still leave the stored points stale.
            // Propagating is exact under the same assumption the per-manifold
            // update already made: the contact normal does not change during
            // the pass, and the stored points ride with their bodies.
            //
            // The two shares sum to exactly `correction` by construction, so
            // the current manifold needs no separate bookkeeping - it matches
            // the propagation and is updated with the rest.
            float share_a = inverse_a / inverse_sum;
            float share_b = inverse_b / inverse_sum;
            pf_apply_position_correction(body_a,
                manifold->points[deepest].point_a, normal, correction * share_a);
            pf_apply_position_correction(body_b,
                manifold->points[deepest].point_b, pf_scale(normal, -1.0f),
                correction * share_b);
            PfVec3 delta_a = pf_scale(normal, correction * share_a);
            PfVec3 delta_b = pf_scale(normal, -correction * share_b);
            for (int other = 0; other < world->manifold_count; ++other) {
                PfManifold* target = &world->manifolds[other];
                if (target->point_count <= 0) {
                    continue;
                }
                bool moves_a = target->body_a == manifold->body_a
                    || target->body_b == manifold->body_a;
                bool moves_b = target->body_a == manifold->body_b
                    || target->body_b == manifold->body_b;
                if (!moves_a && !moves_b) {
                    continue;
                }
                float along_a = pf_dot(delta_a, target->normal);
                float along_b = pf_dot(delta_b, target->normal);
                for (int point_index = 0; point_index < target->point_count;
                        ++point_index) {
                    PfContactPoint* point = &target->points[point_index];
                    // A body may occupy either slot, so both are tested
                    // rather than assuming a slot convention.
                    if (moves_a && target->body_a == manifold->body_a) {
                        point->point_a = pf_add(point->point_a, delta_a);
                        point->separation += along_a;
                    }
                    if (moves_a && target->body_b == manifold->body_a) {
                        point->point_b = pf_add(point->point_b, delta_a);
                        point->separation -= along_a;
                    }
                    if (moves_b && target->body_a == manifold->body_b) {
                        point->point_a = pf_add(point->point_a, delta_b);
                        point->separation += along_b;
                    }
                    if (moves_b && target->body_b == manifold->body_b) {
                        point->point_b = pf_add(point->point_b, delta_b);
                        point->separation -= along_b;
                    }
                }
            }
            corrected = true;
        }
        if (!corrected) {
            break;
        }
    }
}
