#pragma once
#include "../../src/puffysics/integrate.cuh"
#include "car.h"
#include "time_trial.cuh"

// Environment-owned raycast vehicle. SI units, body +Z forward, +X left, +Y up.
struct RacingCarConfig {
    float mass, power, torque, wheelbase, track_width, radius;
    float spring, damper, rest_length, travel, wheel_inertia;
    float steer_limit, steer_rate, steer_speed_scale, throttle_rate;
    float final_drive, efficiency, brake_torque;
    float longitudinal_stiffness, lateral_stiffness, drag_area;
    float lidar_mount_y, lidar_range;
    float tarmac_grip, curb_grip, grass_grip, gravel_grip;
    float tarmac_rolling, curb_rolling, grass_rolling, gravel_rolling;
};
struct RacingCar {
    PfBody body;
    RacingTrial trial;
    float omega[4], spin[4], wheel_y[4], steer[4], throttle, steering, rpm, shift_time;
    float grip[4];
    int gear, crashed, material[4], contacts[4];
#ifdef RACING_MULTI
    unsigned tire_saturation_mask;
#endif
};
static constexpr float RACING_DT = 1.0f / 240;
__host__ __device__ static float racing_clamp(float x, float low, float high) {
    return fminf(high, fmaxf(low, x));
}
__device__ static PfVec3 racing_anchor(RacingCarConfig c, int i) {
    return pf_v3((i % 2 ? -0.5f : 0.5f) * c.track_width, 0, (i < 2 ? 0.5f : -0.5f) * c.wheelbase);
}
__device__ static void racing_impulse(PfBody *b, PfVec3 r, PfVec3 impulse) {
    b->linear_velocity = pf_add(b->linear_velocity, pf_scale(impulse, b->inverse_mass));
    b->angular_velocity =
        pf_add(b->angular_velocity, pf_inverse_inertia_world(b, pf_cross(r, impulse)));
}
__device__ static float racing_inverse_mass(PfBody *b, PfVec3 r, PfVec3 direction) {
    PfVec3 cross = pf_cross(r, direction);
    return b->inverse_mass + pf_dot(cross, pf_inverse_inertia_world(b, cross));
}
#include "task.cuh"

__device__ static void racing_reset_kernel_device(
    RacingCar *car, RacingCarConfig c, float3 ground, float heading) {
    RacingTrial previous = car->trial;
    *car = {};
    car->trial.best = previous.best;
    car->trial.last = previous.last;
    car->trial.laps = previous.laps;
    car->body.mode = PF_DYNAMIC;
    car->body.position = pf_v3(
        ground.x, ground.y + c.radius + c.rest_length - c.mass * 9.81f / (4 * c.spring), ground.z);
    car->body.rotation = pf_quat_from_axis_angle(pf_v3(0, 1, 0), heading);
    car->body.inverse_mass = 1 / c.mass;
    // Box approximation for chassis inertia; dimensions are provisional.
    float length = c.wheelbase * 1.72f, width = c.track_width + 0.25f, height = 1.2f;
    car->body.inverse_inertia_local = pf_v3(12 / (c.mass * (length * length + height * height)),
        12 / (c.mass * (length * length + width * width)),
        12 / (c.mass * (width * width + height * height)));
    car->gear = 1;
    car->rpm = 1200;
    for (int i = 0; i < 4; i++) {
        car->wheel_y[i] = -c.rest_length + c.mass * 9.81f / (4 * c.spring);
        car->material[i] = -1;
    }
}
__device__ static void racing_car_rays_device(RacingCar *car, RacingCarConfig c, PfOptixRay *rays) {
    PfBody *body = &car->body;
    PfVec3 up = pf_quat_rotate(body->rotation, pf_v3(0, 1, 0));
    for (int i = 0; i < 4; i++) {
        PfVec3 p = pf_add(body->position, pf_quat_rotate(body->rotation, racing_anchor(c, i)));
        rays[i] = {make_float3(p.x, p.y, p.z), 0.001f, make_float3(-up.x, -up.y, -up.z),
            c.rest_length + c.radius + 0.05f};
    }
    // Swept corner probes stop the manual preview on obstacle impact. Not a convex CCD solver.
    for (int i = 0; i < 8; i++) {
        PfVec3 r = pf_quat_rotate(body->rotation,
            pf_v3((i & 1 ? 1 : -1) * (c.track_width * 0.5f + 0.13f), i & 2 ? 0.35f : -0.25f,
                (i & 4 ? 1 : -1) * c.wheelbase * 0.85f));
        PfVec3 p = pf_add(body->position, r);
        PfVec3 delta =
            pf_scale(pf_add(body->linear_velocity, pf_cross(body->angular_velocity, r)), RACING_DT);
        float length = pf_length(delta);
        PfVec3 direction = length > 1e-6f ? pf_scale(delta, 1 / length) : pf_v3(0, 1, 0);
        rays[i + 4] = {make_float3(p.x, p.y, p.z), 0.001f,
            make_float3(direction.x, direction.y, direction.z), fmaxf(0.002f, length + 0.02f)};
    }
}
// Sensor directions are local to the chassis, including pitch and roll.
__device__ static void racing_car_sensor_ray(
    const RacingCar *car, RacingCarConfig c, PfOptixRay *rays, int i) {
    if (i >= 260)
        return;
    PfVec3 origin, direction;
    float range;
    if (i < 256) {
        float azimuth = ((i % 128) / 127.0f - 0.5f) * 4.7123889804f;
#ifdef RACING_MULTI
        // Level upper ring sees other cars at range from the lower race-mode mount.
        float elevation = i < 128 ? -0.0523598776f : 0.0f;
#else
        float elevation = i < 128 ? -0.0523598776f : 0.0174532925f;
#endif
        origin = pf_v3(0, c.lidar_mount_y, 0);
        direction = pf_v3(
            sinf(azimuth) * cosf(elevation), sinf(elevation), cosf(azimuth) * cosf(elevation));
        range = c.lidar_range;
    } else {
        origin = racing_anchor(c, i - 256);
        direction = pf_v3(0, -1, 0);
        range = c.rest_length + c.radius + 0.05f;
    }
    origin = pf_add(car->body.position, pf_quat_rotate(car->body.rotation, origin));
    direction = pf_quat_rotate(car->body.rotation, direction);
    rays[i] = {make_float3(origin.x, origin.y, origin.z), i < 256 ? 0.01f : 0.001f,
        make_float3(direction.x, direction.y, direction.z), range};
}
__device__ static void racing_car_integrate_device(RacingCar *car, RacingCarConfig c,
    const PfOptixRay *rays, const PfOptixHit *hits, const RacingGate *gates, int gate_count,
    RacingTask *task, const RacingRoutePoint *route, int route_count, float route_length,
    bool update_task = true) {
    if (task->done)
        return;
#ifdef RACING_MULTI
    car->tire_saturation_mask = 0;
#endif
    float steering = task->action[0], throttle = task->action[1], brake = task->action[2];
    PfBody *b = &car->body;
    PfVec3 axes[3];
    pf_quat_axes(b->rotation, axes);
#ifndef RACING_MULTI
    for (int i = 4; i < 12; i++) {
        if (hits[i].triangle >= 0 && pf_length(b->linear_velocity) > 0.5f)
            car->crashed = 1;
    }
    if (axes[1].y < 0.25f) car->crashed = 1;
#endif
    if (b->position.y < -50 || !pf_vec_valid(b->linear_velocity) ||
        !pf_vec_valid(b->angular_velocity) || !pf_vec_valid(b->position) ||
        !pf_quat_valid(b->rotation)) {
        car->crashed = 1;
    }
    if (car->crashed) {
        if (car->trial.active)
            car->trial.invalid = 3;
        b->linear_velocity = b->angular_velocity = pf_v3(0, 0, 0);
        if (update_task) racing_task_step(task, car, c, route, route_count, route_length, b->position);
        return;
    }
    float dt = RACING_DT, speed = pf_length(b->linear_velocity);
    car->throttle +=
        racing_clamp(throttle - car->throttle, -c.throttle_rate * dt, c.throttle_rate * dt);
    float steer_target = -steering * c.steer_limit / (1 + speed / c.steer_speed_scale);
    car->steering +=
        racing_clamp(steer_target - car->steering, -c.steer_rate * dt, c.steer_rate * dt);
    float ratios[6] = {3.0f, 2.1f, 1.6f, 1.28f, 1.04f, 0.85f};
    float wheel_speed = 0;
    for (int i = 0; i < 4; i++)
        wheel_speed += fabsf(car->omega[i]) * 0.25f;
    car->rpm = fmaxf(1200, wheel_speed * ratios[car->gear - 1] * c.final_drive * 9.5492966f);
    car->shift_time = fmaxf(0, car->shift_time - dt);
    if (car->shift_time == 0 && car->rpm > 6900 && car->gear < 6) {
        car->gear++;
        car->shift_time = 0.12f;
    } else if (car->shift_time == 0 && car->rpm < 2900 && car->gear > 1) {
        car->gear--;
        car->shift_time = 0.12f;
    }
    float engine_omega = car->rpm / 9.5492966f;
    float curve = racing_clamp((car->rpm - 800) / 2700, 0.35f, 1);
    // Brake override acts immediately, even while the throttle actuator is closing.
    float engine_torque = fminf(c.torque * curve, c.power / engine_omega)
        * car->throttle * (1 - brake);
    if (car->rpm > 8000 || car->shift_time > 0)
        engine_torque = 0;
    float drive = engine_torque * ratios[car->gear - 1] * c.final_drive * c.efficiency * 0.25f;
    pf_clear_forces(b, 1);
    b->force = pf_scale(b->linear_velocity, -0.5f * 1.225f * c.drag_area * speed);
    for (int i = 0; i < 4; i++) {
        car->contacts[i] = 0;
        car->material[i] = hits[i].triangle >= 0 ? hits[i].material : -1;
        car->grip[i] = 0;
        car->wheel_y[i] = -c.rest_length;
        float angle = 0;
        if (i < 2 && fabsf(car->steering) > 1e-5f) {
            float turn_radius = c.wheelbase / tanf(car->steering);
            angle = atanf(c.wheelbase / (turn_radius - (i % 2 ? -0.5f : 0.5f) * c.track_width));
        }
        car->steer[i] = angle;
        car->omega[i] += drive * dt / c.wheel_inertia;
        float brake_step = brake * c.brake_torque * dt / c.wheel_inertia;
        car->omega[i] -= racing_clamp(car->omega[i], -brake_step, brake_step);
        if (hits[i].triangle >= 0) {
            float compression = c.rest_length + c.radius - hits[i].distance;
            PfVec3 normal = pf_v3(hits[i].normal.x, hits[i].normal.y, hits[i].normal.z);
            if (compression >= 0 && pf_dot(normal, axes[1]) > 0.35f) {
                PfVec3 p = pf_v3(rays[i].origin.x + rays[i].direction.x * hits[i].distance,
                    rays[i].origin.y + rays[i].direction.y * hits[i].distance,
                    rays[i].origin.z + rays[i].direction.z * hits[i].distance);
                PfVec3 r = pf_sub(p, b->position);
                PfVec3 velocity = pf_add(b->linear_velocity, pf_cross(b->angular_velocity, r));
                float load = fmaxf(0, c.spring * compression - c.damper * pf_dot(velocity, normal));
                load += fmaxf(0, compression - c.travel) * c.spring * 4;
                load = fminf(load, c.mass * 9.81f * 3);
                PfVec3 force = pf_scale(normal, load);
                b->force = pf_add(b->force, force);
                b->torque = pf_add(b->torque, pf_cross(r, force));
                float grip = c.tarmac_grip, rolling = c.tarmac_rolling;
                int mat = hits[i].material;
                if (mat == 44 || mat == 56) {
                    grip = c.curb_grip;
                    rolling = c.curb_rolling;
                }
                if (mat == 53 || mat == 58) {
                    grip = c.grass_grip;
                    rolling = c.grass_rolling;
                }
                if (mat == 32 || mat == 59) {
                    grip = c.gravel_grip;
                    rolling = c.gravel_rolling;
                }
                car->contacts[i] = 1;
                car->grip[i] = grip;
                car->wheel_y[i] = c.radius - hits[i].distance;
                PfVec3 forward =
                    pf_add(pf_scale(axes[2], cosf(angle)), pf_scale(axes[0], sinf(angle)));
                forward = pf_normalize_or(
                    pf_sub(forward, pf_scale(normal, pf_dot(forward, normal))), axes[2]);
                PfVec3 left = pf_cross(normal, forward);
                float vx = pf_dot(velocity, forward), vy = pf_dot(velocity, left);
                float mx = racing_inverse_mass(b, r, forward), my = racing_inverse_mass(b, r, left);
                // Implicit wheel/body coupling avoids explicit stiff-slip instability at low speed.
                float jx = (car->omega[i] * c.radius - vx) /
                    (mx + c.radius * c.radius / c.wheel_inertia +
                        1 / (c.longitudinal_stiffness * dt));
                float jy = -vy / (my + 1 / (c.lateral_stiffness * dt));
                float budget = grip * load * dt, magnitude = sqrtf(jx * jx + jy * jy);
                float scale = fminf(1, budget / fmaxf(magnitude, 1e-8f));
#ifdef RACING_MULTI
                if (scale < 0.99f) car->tire_saturation_mask |= 1u << i;
#endif
                jx *= scale;
                jy *= scale;
                racing_impulse(b, r, pf_add(pf_scale(forward, jx), pf_scale(left, jy)));
                car->omega[i] -= jx * c.radius / c.wheel_inertia;
                float resistance =
                    racing_clamp(vx / fmaxf(mx, 1e-8f), -rolling * load * dt, rolling * load * dt);
                racing_impulse(b, r, pf_scale(forward, -resistance));
            }
        }
        car->spin[i] = remainderf(car->spin[i] + car->omega[i] * dt, 6.283185307f);
    }
    // pf_integrate consumes applied torque; include the rigid body's gyroscopic term explicitly.
    PfVec3 local = pf_quat_rotate(pf_quat_conjugate(b->rotation), b->angular_velocity);
    PfVec3 momentum = pf_v3(local.x / b->inverse_inertia_local.x,
        local.y / b->inverse_inertia_local.y, local.z / b->inverse_inertia_local.z);
    b->torque = pf_sub(b->torque, pf_quat_rotate(b->rotation, pf_cross(local, momentum)));
    PfVec3 before = b->position;
    pf_integrate(b, 1, pf_v3(0, -9.81f, 0), dt);
    if (!update_task) return;
    racing_task_step(task, car, c, route, route_count, route_length, before);
    int laps = car->trial.laps;
    if (!task->done) {
        racing_trial_step(&car->trial, gates, gate_count, before, b->position,
            task->s, task->route_delta, route_length, task->route_jump);
        if (car->trial.laps > laps) {
            task->pending += 10;
            task->progress_credit = 0; // Completed-lap credit is retained on a later crash.
        }
    }
}
__device__ static void racing_car_snapshot_device(
    RacingCar *car, RacingCarConfig c, RacingCarFrame *out, const RacingTask *task) {
    PfBody b = car->body;
    *out = {};
    out->position[0] = b.position.x;
    out->position[1] = b.position.y;
    out->position[2] = b.position.z;
    out->rotation[0] = b.rotation.x;
    out->rotation[1] = b.rotation.y;
    out->rotation[2] = b.rotation.z;
    out->rotation[3] = b.rotation.w;
    out->reward = task->done && task->pending != 0 ? task->pending : task->reward;
    out->episode_return = task->total_reward + task->pending;
    out->task_done = task->done;
    out->progress = task->furthest;
    out->lap_time = car->trial.active ? car->trial.clock - car->trial.started : 0;
    out->last_lap = car->trial.last;
    out->best_lap = car->trial.best;
    out->lap_active = car->trial.active;
    out->next_gate = car->trial.next;
    out->lap_invalid = car->trial.invalid;
    out->completed_laps = car->trial.laps;
    out->speed = pf_length(b.linear_velocity);
    out->rpm = car->rpm;
    out->throttle = car->throttle;
    out->brake = task->action[2];
    out->steering = task->action[0];
    out->gear = car->gear;
    out->crashed = car->crashed;
    out->wheelbase = c.wheelbase;
    out->track_width = c.track_width;
    out->radius = c.radius;
    out->lidar_mount_y = c.lidar_mount_y;
    for (int i = 0; i < 4; i++) {
        out->wheel_y[i] = car->wheel_y[i];
        out->wheel_spin[i] = car->spin[i];
        out->steer[i] = car->steer[i];
        out->material[i] = car->material[i];
        out->grip[i] = car->grip[i];
        out->contacts[i] = car->contacts[i];
    }
}

__global__ static void racing_reset_kernel(
    RacingCar *car, RacingCarConfig c, float3 ground, float heading) {
    racing_reset_kernel_device(car, c, ground, heading);
}
__global__ static void racing_car_rays(RacingCar *car, RacingCarConfig c, PfOptixRay *rays) {
    racing_car_rays_device(car, c, rays);
}
__global__ static void racing_car_sensor_rays(
    const RacingCar *car, RacingCarConfig c, PfOptixRay *rays) {
    racing_car_sensor_ray(car, c, rays, threadIdx.x + blockIdx.x * blockDim.x);
}
__global__ static void racing_car_integrate(RacingCar *car, RacingCarConfig c,
    const PfOptixRay *rays, const PfOptixHit *hits, const RacingGate *gates, int gate_count,
    RacingTask *task, const RacingRoutePoint *route, int route_count, float route_length) {
    racing_car_integrate_device(
        car, c, rays, hits, gates, gate_count, task, route, route_count, route_length);
}

__global__ static void racing_car_snapshot(
    RacingCar *car, RacingCarConfig c, RacingCarFrame *out, const RacingTask *task) {
    racing_car_snapshot_device(car, c, out, task);
}
