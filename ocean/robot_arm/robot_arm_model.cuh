#pragma once
#include "../../src/puffysics/model_builder.cuh"
#include "../../src/puffysics/inertia.cuh"

// Native articulation data. Collision geometry and contact policy are added
// separately before this model can replace the production environment solver.
#include "robot_arm_parameters.h"

enum RaNativeMode { RA_NATIVE_PICK, RA_NATIVE_STACK, RA_NATIVE_BASKETBALL };
enum RaNativeLink {
    RA_NATIVE_JOINT_1,
    RA_NATIVE_JOINT_2,
    RA_NATIVE_JOINT_3,
    RA_NATIVE_JOINT_4,
    RA_NATIVE_JOINT_5,
    RA_NATIVE_JOINT_6,
    RA_NATIVE_JOINT_7,
    RA_NATIVE_HAND,
    RA_NATIVE_LEFT,
    RA_NATIVE_RIGHT,
    RA_NATIVE_OBJECT,
    RA_NATIVE_BASE
};
enum RaNativeSite { RA_NATIVE_END_EFFECTOR };

// q[7] is HALF width. Both finger axes are local +Y; the right finger's
// 180-degree rest rotation reverses its world axis. Its mimic ratio is +1.
static const char* ra_native_articulation(RaNativeMode mode, PfModelStorage& h, PfModel& model) {
    model = {};
    h = PfModelStorage{};
    if (mode < RA_NATIVE_PICK || mode > RA_NATIVE_BASKETBALL) {
        return "invalid robot arm mode";
    }
    const float mass[10] = RA_MODEL_MASSES;
    const PfVec3 center[10] = RA_MODEL_CENTERS;
    const PfInertiaTensor tensor[10] = RA_MODEL_INERTIAS;
    const PfVec3 offsets[7] = RA_MODEL_OFFSETS;
    const float lower[7] = RA_MODEL_LOWER, upper[7] = RA_MODEL_UPPER, home[7] = RA_MODEL_HOME;
    const float half = 0.70710678118f;
    const PfQuat rotations[7] = RA_MODEL_ROTATIONS(RA_MODEL_WXYZ, half);
    h.links.resize(mode == RA_NATIVE_STACK ? 12 : 11);
    for (int i = 0; i < 10; ++i) {
        PfLink& l = h.links[i];
        l.mass = mass[i];
        l.center = center[i];
        l.rest = pf_pose_identity();
        if (!pf_principal_inertia(tensor[i], l.inertia, l.inertia_rotation)) {
            return "robot arm inertia conversion failed";
        }
        l.parent = i - 1;
        if (i < 7) {
            l.joint = PF_HINGE;
            l.axis = pf_v3(0, 0, 1);
            l.rest = {offsets[i], rotations[i]};
            l.limited = true;
            l.lower = lower[i];
            l.upper = upper[i];
            if (i == 0) {
                l.rest = pf_pose_compose({pf_v3(0, 0, 0), {half, -half, 0, 0}}, l.rest);
            }
        } else if (i == RA_NATIVE_HAND) {
            l.joint = PF_FIXED;
            l.rest.position = pf_v3(0, 0, 0.107f);
            l.rest.rotation = pf_quat_from_axis_angle(pf_v3(0, 0, 1), -0.78539816339f);
        } else {
            l.parent = RA_NATIVE_HAND;
            l.joint = PF_SLIDE;
            l.axis = pf_v3(0, 1, 0);
            l.rest.position = pf_v3(0, 0, 0.0584f);
            if (i == RA_NATIVE_LEFT) {
                l.limited = true;
                l.lower = 0.002f;
                l.upper = 0.040f;
            } else {
                l.rest.rotation = {0, 0, 0, 1};
                l.mimic = true;
                l.source = RA_NATIVE_LEFT;
                l.ratio = 1;
            }
        }
    }
    for (int i = RA_NATIVE_OBJECT; i < (int)h.links.size(); ++i) {
        PfLink& l = h.links[i];
        l.parent = -1;
        l.joint = PF_FREE;
        l.rest = pf_pose_identity();
        l.inertia_rotation = pf_quat_identity();
        bool ball = mode == RA_NATIVE_BASKETBALL;
        l.mass = ball ? 0.080f : mode == RA_NATIVE_STACK ? 1.0f : 0.10f;
        float moment =
            ball ? 0.4f * l.mass * 0.028f * 0.028f : (2.0f / 3.0f) * l.mass * 0.035f * 0.035f;
        l.inertia = pf_v3(moment, moment, moment);
        // Nominal poses only. Episode RNG will overwrite these during adapter reset.
        l.rest.position =
            pf_v3(0.48f, ball ? 0.028f : 0.035f, i == RA_NATIVE_BASE ? -0.26f : 0.26f);
    }
    h.dofs.resize(mode == RA_NATIVE_STACK ? 20 : 14);
    for (int i = 0; i < 7; ++i) {
        h.dofs[i].armature = 0.1f;
        h.dofs[i].damping = 1;
        PfActuator motor = {};
        motor.link = i;
        motor.kind = PF_POSITION_SERVO;
        motor.gear = 1;
        motor.kp = i < 2 ? 4500 : i < 4 ? 3500 : 2000;
        motor.kd = i < 2 ? 450 : i < 4 ? 350 : 200;
        motor.control_min = lower[i];
        motor.control_max = upper[i];
        motor.force_max = i < 4 ? 87 : 12;
        motor.force_min = -motor.force_max;
        h.actuators.push_back(motor);
    }
    // Fixed-arm M_half_width = 2*0.015 + 0.20 = 4*0.0575.
    // Physical arm/finger coupling is retained by the native mass assembly.
    h.dofs[7].armature = 0.20f;
    PfActuator jaw = {};
    jaw.link = RA_NATIVE_LEFT;
    jaw.kind = PF_POSITION_SERVO;
    jaw.gear = 2;
    jaw.kp = 1500;
    jaw.kd = 18.57f;
    jaw.control_min = 0.004f;
    jaw.control_max = 0.080f;
    jaw.force_min = -100;
    jaw.force_max = 100;
    h.actuators.push_back(jaw);
    h.sites.push_back({RA_NATIVE_HAND, {pf_v3(0, 0, 0.115f), pf_quat_identity()}});
    const char* error = pf_compile_model(h, model);
    if (error) {
        return error;
    }
    for (int i = 0; i < 7; ++i) {
        h.initial_qpos[h.links[i].qpos] = home[i];
    }
    h.initial_qpos[h.links[RA_NATIVE_LEFT].qpos] = 0.040f;
    return NULL;
}
