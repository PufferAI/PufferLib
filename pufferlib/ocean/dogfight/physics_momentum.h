// physics_momentum.h - Momentum-based RK4 flight physics for dogfight environment
//
// This is a STUB file that passes through to rate-based physics (flightlib.h).
// Future agent: Implement full momentum-based physics here.
//
// DELETE THESE SCAFFOLDING COMMENTS AFTER IMPLEMENTATION IS COMPLETE.
//
// ============================================================================
// OVERVIEW FOR FUTURE IMPLEMENTER
// ============================================================================
//
// This file should provide the SAME INTERFACE as flightlib.h:
//   - Plane struct (with omega as state variable, not computed)
//   - reset_plane(Plane *p, Vec3 pos, Vec3 vel)
//   - step_plane_with_physics(Plane *p, float *actions, float dt)
//   - step_plane(Plane *p, float dt)  (simple forward motion for opponent)
//
// KEY DIFFERENCE from rate-based physics:
//   - Rate-based: actions directly set angular rates (omega_body = f(actions))
//   - Momentum-based: actions set control surface deflections -> aerodynamic
//     moments -> angular acceleration -> omega integrated via RK4
//
// The agent policy outputs the SAME 5 actions: [throttle, pitch, roll, yaw, trigger]
// Momentum physics interprets pitch/roll/yaw as control surface deflections
// that create aerodynamic moments, rather than direct rate commands.
//
// ============================================================================
// REFERENCE: drone_race/dronelib.h
// ============================================================================
//
// dronelib.h implements full RK4 momentum physics for a quadrotor. Key patterns:
//
// 1. State struct contains omega as a state variable (not computed from actions):
//    typedef struct {
//        Vec3 pos, vel;
//        Quat quat;
//        Vec3 omega;    // angular velocity (p, q, r) - integrated, not commanded
//        float rpms[4];
//    } State;
//
// 2. StateDerivative struct for RK4:
//    typedef struct {
//        Vec3 vel;       // d(pos)/dt
//        Vec3 v_dot;     // d(vel)/dt = acceleration
//        Quat q_dot;     // d(quat)/dt
//        Vec3 w_dot;     // d(omega)/dt = angular acceleration
//    } StateDerivative;
//
// 3. compute_derivatives() calculates derivatives from current state:
//    - Converts actions to forces/torques
//    - Computes v_dot = F/m (linear acceleration)
//    - Computes w_dot = tau/I (angular acceleration, with inertia tensor)
//    - Computes q_dot = 0.5 * q * omega_quat
//
// 4. rk4_step() integrates using 4th-order Runge-Kutta:
//    k1 = compute_derivatives(state)
//    k2 = compute_derivatives(state + k1*dt/2)
//    k3 = compute_derivatives(state + k2*dt/2)
//    k4 = compute_derivatives(state + k3*dt)
//    state += (k1 + 2*k2 + 2*k3 + k4) * dt/6
//
// ============================================================================
// AIRCRAFT PARAMETERS NEEDED
// ============================================================================
//
// For momentum-based aircraft physics, you'll need:
//
// INERTIA TENSOR (moments of inertia about body axes):
//   float Ixx;  // roll inertia (kg*m^2) - P-51D: ~5,000-8,000
//   float Iyy;  // pitch inertia (kg*m^2) - P-51D: ~15,000-25,000
//   float Izz;  // yaw inertia (kg*m^2) - P-51D: ~18,000-30,000
//
// STABILITY DERIVATIVES (moment coefficients):
//   float Cm_alpha;    // pitching moment vs AOA (negative = stable)
//   float Cl_beta;     // rolling moment vs sideslip
//   float Cn_beta;     // yawing moment vs sideslip (weathervane)
//   float Cm_q;        // pitch damping (moment vs pitch rate)
//   float Cl_p;        // roll damping (moment vs roll rate)
//   float Cn_r;        // yaw damping (moment vs yaw rate)
//
// CONTROL DERIVATIVES (control effectiveness):
//   float Cm_delta_e;  // pitching moment vs elevator deflection
//   float Cl_delta_a;  // rolling moment vs aileron deflection
//   float Cn_delta_r;  // yawing moment vs rudder deflection
//
// CROSS-COUPLING (optional, for realism):
//   float Cl_delta_r;  // adverse yaw from rudder
//   float Cn_delta_a;  // adverse yaw from aileron
//
// See JSBSim or X-Plane data for P-51D values.
//
// ============================================================================
// FORCES AND MOMENTS TO COMPUTE
// ============================================================================
//
// In compute_derivatives(), calculate:
//
// FORCES (world frame, for v_dot):
//   - Thrust: T = f(throttle, airspeed) along body X-axis
//   - Lift: L = 0.5 * rho * V^2 * S * C_L(alpha), perpendicular to velocity
//   - Drag: D = 0.5 * rho * V^2 * S * C_D(alpha), opposite to velocity
//   - Weight: W = m * g, -Z world
//
// MOMENTS (body frame, for w_dot):
//   - Pitching moment: M = 0.5 * rho * V^2 * S * c * Cm
//     where Cm = Cm_0 + Cm_alpha*alpha + Cm_q*(q*c/2V) + Cm_delta_e*delta_e
//   - Rolling moment: L = 0.5 * rho * V^2 * S * b * Cl
//     where Cl = Cl_beta*beta + Cl_p*(p*b/2V) + Cl_delta_a*delta_a
//   - Yawing moment: N = 0.5 * rho * V^2 * S * b * Cn
//     where Cn = Cn_beta*beta + Cn_r*(r*b/2V) + Cn_delta_r*delta_r
//
// ANGULAR ACCELERATION:
//   w_dot.x = (L + (Iyy - Izz) * q * r) / Ixx
//   w_dot.y = (M + (Izz - Ixx) * p * r) / Iyy
//   w_dot.z = (N + (Ixx - Iyy) * p * q) / Izz
//
// ============================================================================
// IMPLEMENTATION STEPS
// ============================================================================
//
// 1. Define aircraft parameters (Ixx, Iyy, Izz, stability/control derivatives)
// 2. Create StateDerivative struct
// 3. Implement compute_derivatives():
//    - Map actions[1:3] to control surface deflections (delta_e, delta_a, delta_r)
//    - Compute aerodynamic forces (lift, drag, thrust)
//    - Compute aerodynamic moments (L, M, N)
//    - Compute v_dot = F_total / mass
//    - Compute w_dot from Euler's equations
//    - Compute q_dot from quaternion kinematics
// 4. Implement rk4_step() following dronelib.h pattern
// 5. Update step_plane_with_physics() to use rk4_step()
// 6. Run all tests, verify behavior similar to rate-based
// 7. DELETE THESE SCAFFOLDING COMMENTS
//
// ============================================================================

#ifndef PHYSICS_MOMENTUM_H
#define PHYSICS_MOMENTUM_H

// For now, include rate-based physics and pass through
// Future: Replace this with full momentum-based implementation
#include "flightlib.h"

// ============================================================================
// STUB IMPLEMENTATION - Passes through to rate-based physics
// ============================================================================
//
// All functions below just call the corresponding flightlib.h functions.
// This allows PHYSICS_MODE=1 to compile and work identically to PHYSICS_MODE=0.
//
// Future: Replace these with actual momentum-based implementations.
// ============================================================================

// Note: Plane struct, reset_plane(), step_plane_with_physics(), step_plane()
// are all provided by flightlib.h included above.
//
// When implementing momentum physics:
// 1. Remove the #include "flightlib.h" above
// 2. Copy the Plane struct definition here (it's the same)
// 3. Copy the math functions (vec3, quat, etc.) or factor into shared header
// 4. Implement step_plane_with_physics() using RK4
// 5. Keep step_plane() simple (opponent doesn't need full physics)

#endif // PHYSICS_MOMENTUM_H
