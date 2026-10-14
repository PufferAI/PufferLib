#pragma once

#include "joints.cuh"

__device__ static inline bool pf_joint_forward_kinematics(
        PfJointWorld world) {
    if (world.joints == NULL || world.bodies == NULL
            || world.joint_count <= 0 || world.joint_count > world.joint_capacity
            || world.body_count < world.joint_count
            || world.body_count > world.body_capacity) {
        return false;
    }
    for (int index = 0; index < world.joint_count; ++index) {
        if (!pf_joint_valid(&world.joints[index], world.body_count)
                || !pf_mode_is_dynamic(world.bodies[
                    world.joints[index].child_body].mode)) {
            return false;
        }
    }
    /* Every joint reads its parent's pose, so a body that another joint
     * rewrites must be written by an earlier joint. Roots, which no joint
     * owns, keep the pose the caller gave them. ponytail: O(joint_count^2)
     * scan; the articulated robots here carry tens of joints, so a scratch
     * incoming-joint map would cost more memory than it saves. */
    for (int index = 0; index < world.joint_count; ++index) {
        int parent = world.joints[index].parent_body;
        if (parent < 0) continue;
        for (int other = 0; other < world.joint_count; ++other)
            if (world.joints[other].child_body == parent && other >= index)
                return false;
    }
    for (int index = 0; index < world.joint_count; ++index) {
        const PfJoint* joint = &world.joints[index];
        PfBody* child = &world.bodies[joint->child_body];
        PfQuat delta = pf_quat_from_axis_angle(joint->axis, joint->angle);
        if (joint->parent_body < 0) {
            child->rotation = delta;
            child->position = pf_sub(joint->parent_anchor,
                pf_quat_rotate(child->rotation, joint->child_anchor));
        } else {
            const PfBody* parent = &world.bodies[joint->parent_body];
            /* joint->axis lives in the parent frame, so the local joint delta
             * composes on the right: child = parent * delta. Anchors are body
             * local on both sides, which is the same convention. */
            child->rotation = pf_quat_normalize(pf_quat_multiply(
                parent->rotation, delta));
            PfVec3 anchor = pf_add(parent->position,
                pf_quat_rotate(parent->rotation, joint->parent_anchor));
            child->position = pf_sub(anchor,
                pf_quat_rotate(child->rotation, joint->child_anchor));
        }
    }
    return true;
}
