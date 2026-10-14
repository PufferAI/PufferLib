#pragma once
#include "state.cuh"

// Scalar joint transmissions: length = gear*q, velocity = gear*qvel.
// Force bounds are actuator-space bounds, before transmission by gear.
__device__ static inline bool pf_actuators(const PfModel& m, PfState& s,
        PfWorkspace& w, float dt) {
    for (int i = 0; i < m.actuator_count; ++i) {
        const PfActuator& a = m.actuators[i];
        const PfLink& l = m.links[a.link];
        float control = s.control[i];
        if (!pf_number(control) || !pf_number(s.activation[i])) return false;
        control = pf_clamp(control,a.control_min,a.control_max);
        float input = a.time_constant > 0
            ? s.activation[i] + (control-s.activation[i])*(-expm1f(-dt/a.time_constant))
            : control;
        s.activation[i] = input;
        float transmission = a.gear*pf_joint_scale(l);
        float position = a.gear*pf_joint_position(l,s.qpos);
        float velocity = a.gear*pf_joint_velocity(l,s.qvel);
        float kp = a.kind == PF_POSITION_SERVO ? a.kp : 0;
        float kd = a.kind == PF_MOTOR ? 0 : a.kd;
        float force = a.kind == PF_MOTOR ? input
            : a.kind == PF_POSITION_SERVO ? kp*(input-position-dt*velocity)-kd*velocity
            : kd*(input-velocity);
        float bounded = pf_clamp(force,a.force_min,a.force_max);
        w.rhs[l.dof] += transmission*bounded;
        // The unsaturated linear branch is implicit. Saturated force is held
        // constant this step; bounds are not an implicit active-set solve.
        if (force == bounded)
            w.diagonal[l.dof] += transmission*transmission*(dt*kd+dt*dt*kp);
    }
    return true;
}
