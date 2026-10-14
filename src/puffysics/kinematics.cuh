#pragma once
#include "jacobian_storage.cuh"

// Ball velocities are expressed in the zero-pose joint frame. Free-joint
// translation and angular velocities are world-space; quaternion order is wxyz.
__device__ static inline bool pf_kinematics(const PfModel& m,
        const PfState& s, PfWorkspace& w) {
    for (int i = 0; i < m.link_count; ++i) {
        const PfLink& l = m.links[i];
        PfLinkState parent = {};
        parent.pose = pf_pose_identity();
        if (l.parent >= 0) parent = w.links[l.parent];
        PfLinkState out = {};
        PfPose base = pf_pose_compose(parent.pose, l.rest);
        out.pose = base;
        PfVec3 joint_axis = pf_quat_rotate(base.rotation, l.axis);
        PfVec3 anchor = pf_pose_point(base, l.anchor);
        PfVec3 relative_spin = pf_v3(0,0,0);
        if (l.joint == PF_FREE) {
            const float* q = s.qpos + l.qpos;
            out.pose = {pf_v3(q[0],q[1],q[2]), {q[3],q[4],q[5],q[6]}};
            const float* v = s.qvel + l.dof;
            out.velocity = pf_v3(v[0],v[1],v[2]);
            out.angular_velocity = pf_v3(v[3],v[4],v[5]);
        } else {
            if (l.joint == PF_SLIDE)
                out.pose.position = pf_add(base.position,
                    pf_scale(joint_axis, pf_joint_position(l,s.qpos)));
            if (l.joint == PF_HINGE || l.joint == PF_BALL) {
                PfQuat delta;
                if (l.joint == PF_HINGE) {
                    delta = pf_quat_from_axis_angle(l.axis, pf_joint_position(l,s.qpos));
                    relative_spin = pf_scale(joint_axis, pf_joint_velocity(l,s.qvel));
                } else {
                    const float* q = s.qpos + l.qpos;
                    delta = {q[0],q[1],q[2],q[3]};
                    relative_spin = pf_quat_rotate(base.rotation,
                        pf_v3(s.qvel[l.dof],s.qvel[l.dof+1],s.qvel[l.dof+2]));
                }
                out.pose.rotation = pf_quat_multiply(base.rotation, delta);
                out.pose.position = pf_sub(anchor,
                    pf_quat_rotate(out.pose.rotation, l.anchor));
            }
            out.angular_velocity = pf_add(parent.angular_velocity, relative_spin);
            out.angular_bias = pf_add(parent.angular_bias,
                pf_cross(parent.angular_velocity, relative_spin));
            PfVec3 offset = pf_sub(out.pose.position, parent.pose.position);
            out.velocity = pf_add(parent.velocity,
                pf_cross(parent.angular_velocity, offset));
            out.acceleration_bias = pf_add(parent.acceleration_bias,
                pf_add(pf_cross(parent.angular_bias, offset),
                    pf_cross(parent.angular_velocity,
                        pf_cross(parent.angular_velocity, offset))));
            if (l.joint == PF_SLIDE) {
                PfVec3 slide = pf_scale(joint_axis, pf_joint_velocity(l,s.qvel));
                out.velocity = pf_add(out.velocity, slide);
                out.acceleration_bias = pf_add(out.acceleration_bias,
                    pf_scale(pf_cross(parent.angular_velocity, slide), 2));
            } else if (l.joint == PF_HINGE || l.joint == PF_BALL) {
                PfVec3 r = pf_sub(out.pose.position, anchor);
                out.velocity = pf_add(out.velocity, pf_cross(relative_spin, r));
                // Differentiate the anchor-to-origin lever using the child's
                // angular velocity, including the moving parent joint frame.
                out.acceleration_bias = pf_add(out.acceleration_bias,
                    pf_add(pf_cross(pf_cross(parent.angular_velocity, relative_spin), r),
                    pf_sub(pf_cross(out.angular_velocity, pf_cross(out.angular_velocity,r)),
                        pf_cross(parent.angular_velocity,pf_cross(parent.angular_velocity,r)))));
            }
        }
        if (!pf_vec_valid(out.pose.position) || !pf_quat_valid(out.pose.rotation)
                || !pf_vec_valid(out.velocity) || !pf_vec_valid(out.angular_velocity)
                || !pf_vec_valid(out.acceleration_bias) || !pf_vec_valid(out.angular_bias))
            return false;
        w.links[i] = out;
        bool sparse=w.sparse_jacobian && m.link_dof_offsets;
        int begin=sparse?m.link_dof_offsets[i]:0;
        int end=sparse?m.link_dof_offsets[i+1]:m.nv;
        for (int at=begin;at<end;++at) {
            int d=sparse?m.link_dofs[at]:at;
            PfVec3 jv = pf_v3(0,0,0), jw = jv;
            if (l.parent >= 0) {
                jw = pf_jacobian_entry(m,w,l.parent,d,true);
                jv = pf_add(pf_jacobian_entry(m,w,l.parent,d),
                    pf_cross(jw, pf_sub(out.pose.position, parent.pose.position)));
            }
            int k = d - l.dof;
            if (k >= 0 && k < pf_joint_nv(l.joint)) {
                if (l.joint == PF_SLIDE) jv = pf_add(jv, pf_scale(joint_axis,pf_joint_scale(l)));
                else if (l.joint == PF_FREE && k < 3) jv = pf_basis(k);
                else {
                    PfVec3 axis = l.joint == PF_HINGE ? pf_scale(joint_axis,pf_joint_scale(l))
                        : l.joint == PF_FREE ? pf_basis(k-3)
                        : pf_quat_rotate(base.rotation, pf_basis(k));
                    jw = pf_add(jw, axis);
                    if (l.joint != PF_FREE) jv = pf_add(jv,
                        pf_cross(axis, pf_sub(out.pose.position, anchor)));
                }
            }
            int index=w.compact_jacobian?at:i*m.nv+d;
            w.linear_jacobian[index] = jv;
            w.angular_jacobian[index] = jw;
        }
    }
    return true;
}

__device__ static inline PfVec3 pf_point_jacobian(const PfModel& m,
        const PfWorkspace& w, int link, PfVec3 point, int dof) {
    if (link < 0) return pf_v3(0,0,0);
    return pf_add(pf_jacobian_entry(m,w,link,dof),
        pf_cross(pf_jacobian_entry(m,w,link,dof,true),
            pf_sub(point, w.links[link].pose.position)));
}
