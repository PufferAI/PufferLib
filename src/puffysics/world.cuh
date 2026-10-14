#pragma once

#include "contact_solver.cuh"
#include "contact_joint_pass.cuh"
#include "integrate.cuh"
#include "joints.cuh"
#include "position_solver.cuh"


__device__ static inline bool pf_step(
        PfWorld world, PfVec3 gravity, float dt, int substeps) {
    if (world.bodies == NULL || world.body_count < 0 || !pf_number(dt)
            || dt <= 0.0f || substeps <= 0) {
        return false;
    }
    if (world.body_count == 0) {
        world.manifold_count = 0;
        return true;
    }
    if (world.body_count == 1) {
        world.manifold_count = 0;
        for (int substep = 0; substep < substeps; ++substep) {
            pf_integrate(world.bodies, world.body_count, gravity, dt);
            pf_clear_forces(world.bodies, world.body_count);
        }
        return true;
    }
    bool has_dynamic = false;
    for (int index = 0; index < world.body_count; ++index) {
        if (pf_mode_is_dynamic(world.bodies[index].mode)) {
            has_dynamic = true;
            break;
        }
    }
    if (!has_dynamic) {
        world.manifold_count = 0;
    }
    for (int substep = 0; substep < substeps; ++substep) {
        pf_integrate(world.bodies, world.body_count, gravity, dt);
        if (has_dynamic && pf_detect_contacts(&world) > 0) {
            pf_solve_velocity_contacts(world.bodies, world.manifolds,
                world.manifold_count);
            pf_solve_positions(&world);
        }
        /* Cleared inside the substep loop, not once after it. A force the
         * caller set before the call must be integrated exactly once
         * whatever the substep count; clearing only after the loop
         * integrates it `substeps` times. */
        pf_clear_forces(world.bodies, world.body_count);
    }
    return true;
}

__device__ static inline bool pf_step_joints(PfWorld world,
        PfJoint* joints, int joint_count, PfVec3 gravity, float dt,
        int substeps) {
    if (joint_count <= 0) return pf_step(world, gravity, dt, substeps);
    if (world.bodies == NULL || world.body_count < joint_count
            || !pf_joint_constraints_valid(joints, joint_count, world.bodies,
                world.body_count) || !pf_number(dt) || dt <= 0.0f
            || substeps <= 0 || !pf_vec_valid(gravity)) {
        return false;
    }
    bool has_dynamic = false;
    for (int index = 0; index < world.body_count; ++index) {
        if (pf_mode_is_dynamic(world.bodies[index].mode)) {
            has_dynamic = true;
            break;
        }
    }
    for (int substep = 0; substep < substeps; ++substep) {
        pf_joint_apply_motors(joints, joint_count, world.bodies);
        pf_integrate(world.bodies, world.body_count, gravity, dt);
        bool contacts = has_dynamic && pf_detect_contacts(&world) > 0;
        if (contacts) {
            pf_prepare_velocity_contacts(world.bodies, world.manifolds,
                world.manifold_count);
            unsigned int carried = 0u;
            for (int iteration = 0; iteration < PF_JOINT_VELOCITY_ITERATIONS;
                    ++iteration) {
                pf_solve_velocity_contacts_one(world.bodies, world.manifolds,
                    world.manifold_count, iteration == 0, &carried);
                pf_joint_solve_velocity_once(joints, joint_count, world.bodies);
            }
        } else {
            pf_joint_solve_velocity(joints, joint_count, world.bodies);
        }
        pf_joint_solve_positions(joints, joint_count, world.bodies);
        if (contacts) {
            for (int iteration = 0; iteration < PF_POSITION_ITERATIONS; ++iteration)
                pf_solve_positions(&world);
        }
        pf_joint_refresh_state(joints, joint_count, world.bodies);
        /* Cleared inside the substep loop, not once after it.
         * pf_joint_apply_motors ADDS into body->torque, so clearing only
         * after the loop integrates substep k with k copies of the
         * motor torque (tau, 2tau, 3tau, ... - a 2.5x mean error at
         * substeps == 4). Clearing per substep leaves each one holding
         * exactly the single copy apply_motors just added. */
        pf_clear_forces(world.bodies, world.body_count);
    }
    return true;
}
