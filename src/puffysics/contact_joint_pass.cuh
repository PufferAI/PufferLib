#pragma once

#include "contact_solver.cuh"

// One unchanged contact row pass, exposed for joint/contact interleaving.
//
// The converged-row skip of the pure-contact path is deliberately NOT
// reproduced here. That skip is sound only when the contact mask is a
// complete record of every velocity change since the row last converged,
// and in the interleaved path it is not: `pf_joint_solve_velocity_once`
// changes linear and angular velocity directly, without appearing in the
// contact touched mask. A row marked converged against a body a joint then
// accelerated is therefore skipped with a stale flag, and the contact that
// should arrest the joint's motion never sees it. The joint solver would
// have to return the set of bodies it moved - the previous iteration's set
// as well as the current pass's - before the mask could be trusted, and it
// does not. `carried` is kept because world.cuh and the test harness pass
// it, and is still written so that a future joint-side record has somewhere
// to land.
//
// Rows whose body index is at or above PF_TOUCHED_MASK_BITS were never
// skippable, so the interleaved path is now exactly as conservative as a
// solver that never skips.
__device__ static inline void pf_solve_velocity_contacts_one(
        PfBody* bodies, PfManifold* manifolds, int manifold_count,
        bool first, unsigned int* carried) {
    (void)first; // Restitution was captured for all points during preparation.
    unsigned int touched = 0u;
    for (int manifold_index = 0; manifold_index < manifold_count;
            ++manifold_index) {
        PfManifold* manifold = &manifolds[manifold_index];
        PfBody* body_a = &bodies[manifold->body_a];
        PfBody* body_b = &bodies[manifold->body_b];
        unsigned int body_mask = 0u;
        if (pf_mode_is_dynamic(body_a->mode)
                && (unsigned int)manifold->body_a < PF_TOUCHED_MASK_BITS)
            body_mask |= 1u << manifold->body_a;
        if (pf_mode_is_dynamic(body_b->mode)
                && (unsigned int)manifold->body_b < PF_TOUCHED_MASK_BITS)
            body_mask |= 1u << manifold->body_b;
        if (pf_solve_velocity_manifold(bodies, manifold)) touched |= body_mask;
    }
    *carried = touched;
}
