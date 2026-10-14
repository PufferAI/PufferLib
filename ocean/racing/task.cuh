#pragma once

// Included after RacingCar and its geometry helpers. All state and outputs stay on CUDA.
static constexpr int RACING_BASE_OBS_SIZE = 568;
#ifdef RACING_MULTI
static constexpr int RACING_CAR_BINS = 16;
static constexpr int RACING_NEAR_CARS = 3;
static constexpr int RACING_GROUND_PROBES = 6;
static constexpr int RACING_OBS_SIZE = RACING_BASE_OBS_SIZE + RACING_CAR_BINS
    + RACING_NEAR_CARS * 7 + RACING_GROUND_PROBES * 3;
#else
static constexpr int RACING_OBS_SIZE = RACING_BASE_OBS_SIZE;
#endif
struct RacingRoutePoint {
    float s;
    PfVec3 center, left, right;
};
struct RacingTask {
    int cursor, ticks, done, offroad, offroad_wheels, route_jump;
    float route_delta, motion;
    float time_limit;
    float s, distance, furthest, lateral, heading, stalled, wrongway;
    float action[3], pending, reward, total_reward, progress_credit;
};

__device__ static float racing_project(PfVec3 p, PfVec3 a, PfVec3 b) {
    PfVec3 d = pf_sub(b, a);
    return racing_clamp(pf_dot(pf_sub(p, a), d) / fmaxf(pf_dot(d, d), 1e-8f), 0, 1);
}
__device__ static int racing_route_near(
    const RacingRoutePoint *route, int count, PfVec3 p, int cursor, int radius, float *fraction) {
    float best = 1e30f;
    int found = cursor;
    for (int j = -radius; j <= radius; j++) {
        int i = (cursor + j + count) % count, next = (i + 1) % count;
        float f = racing_project(p, route[i].center, route[next].center);
        PfVec3 point =
            pf_add(route[i].center, pf_scale(pf_sub(route[next].center, route[i].center), f));
        PfVec3 delta = pf_sub(p, point);
        float d = pf_dot(delta, delta);
        if (d < best) {
            best = d;
            found = i;
            *fraction = f;
        }
    }
    return found;
}
__device__ static float racing_route_s(
    const RacingRoutePoint *route, int count, float length, int i, float f) {
    float end = i + 1 == count ? length : route[i + 1].s;
    return route[i].s + f * (end - route[i].s);
}
// Reconcile standings with the current route projection, preserving the lap at wraparound.
// Reward distance still rejects route jumps; standings must not retain those lost metres.
__host__ __device__ static double racing_route_progress(double previous, float s, float length) {
    return s + floor((previous - s) / length + 0.5) * length;
}
__device__ static bool racing_on_road(
    const RacingRoutePoint *route, int count, int cursor, PfVec3 wheel) {
    float f = 0;
    int i = racing_route_near(route, count, wheel, cursor, 2, &f), next = (i + 1) % count;
    PfVec3 left = pf_add(route[i].left, pf_scale(pf_sub(route[next].left, route[i].left), f));
    PfVec3 right = pf_add(route[i].right, pf_scale(pf_sub(route[next].right, route[i].right), f));
    PfVec3 span = pf_sub(right, left), d = pf_sub(wheel, left);
    float across = (d.x * span.x + d.z * span.z) / fmaxf(span.x * span.x + span.z * span.z, 1e-8f);
    return across >= 0 && across <= 1 && fabsf(wheel.y - left.y - across * span.y) <= 2;
}

__device__ static void racing_task_reset_device(
    RacingTask *t, const RacingCar *car, const RacingRoutePoint *route, int count, float length) {
    *t = {};
    float f = 0, best = 1e30f;
    // Global search only on reset. Stepping uses a bounded local cursor.
    for (int i = 0; i < count; i++) {
        float u =
            racing_project(car->body.position, route[i].center, route[(i + 1) % count].center);
        PfVec3 point = pf_add(
            route[i].center, pf_scale(pf_sub(route[(i + 1) % count].center, route[i].center), u));
        PfVec3 d = pf_sub(car->body.position, point);
        if (pf_dot(d, d) < best) {
            best = pf_dot(d, d);
            t->cursor = i;
            f = u;
        }
    }
    t->s = racing_route_s(route, count, length, t->cursor, f);
    PfVec3 forward = pf_normalize_or(
        pf_sub(route[(t->cursor + 1) % count].center, route[t->cursor].center), pf_v3(0, 0, 1));
    t->heading = pf_dot(forward, pf_quat_rotate(car->body.rotation, pf_v3(0, 0, 1)));
}

// Called every physics step. Road/cursor checks affect shaping, never automatic resets.
__device__ static void racing_task_step(RacingTask *t, RacingCar *car, RacingCarConfig c,
    const RacingRoutePoint *route, int count, float length, PfVec3 before) {
    if (t->done)
        return;
    t->ticks++;
    int previous_cursor = t->cursor;
    float previous_s = t->s;
    float f = 0;
    int i = racing_route_near(route, count, car->body.position, t->cursor, 6, &f);
    float s = racing_route_s(route, count, length, i, f);
    float delta = s - t->s;
    if (delta < -length * 0.5f)
        delta += length;
    if (delta > length * 0.5f)
        delta -= length;
    t->cursor = i;
    t->s = s;
    PfVec3 forward =
        pf_normalize_or(pf_sub(route[(i + 1) % count].center, route[i].center), pf_v3(0, 0, 1));
    PfVec3 center = pf_add(
        route[i].center, pf_scale(pf_sub(route[(i + 1) % count].center, route[i].center), f));
    t->lateral = pf_dot(pf_sub(car->body.position, center),
        pf_normalize_or(pf_cross(pf_v3(0, 1, 0), forward), pf_v3(1, 0, 0)));
    t->heading = pf_dot(forward, pf_quat_rotate(car->body.rotation, pf_v3(0, 0, 1)));
    t->offroad = 0;
    t->offroad_wheels = 0;
    for (int w = 0; w < 4; w++) {
        PfVec3 local = racing_anchor(c, w);
        local.y = car->wheel_y[w] - c.radius;
        PfVec3 wheel = pf_add(car->body.position, pf_quat_rotate(car->body.rotation, local));
        if (!racing_on_road(route, count, i, wheel)) {
            t->offroad = 1;
            t->offroad_wheels |= 1 << w;
        }
    }
    // Reject cursor discontinuities independently of the road/reward checks.
    float motion = pf_length(pf_sub(car->body.position, before));
    bool continuous = isfinite(delta) && fabsf(delta) <= motion * 2 + 0.1f;
    // The nearest polyline projection can change abruptly at a corner while the
    // car moves smoothly. Verify the previous projection against
    // the physical before-pose, then measure motion in the new segment's frame.
    // Only adjacent segment switches qualify; a stale s or a non-local shortcut
    // still invalidates the traversal rather than manufacturing progress.
    int cursor_step = (i - previous_cursor + count) % count;
    // A wheel crossing the curb suppresses progress reward, but must not
    // disable this geometric continuity proof and turn a projection correction
    // into a fatal route jump.
    if (!continuous && (cursor_step == 1 || cursor_step == count - 1)) {
        int old_next = (previous_cursor + 1) % count, new_next = (i + 1) % count;
        float old_before_s = racing_route_s(route, count, length, previous_cursor,
            racing_project(before, route[previous_cursor].center, route[old_next].center));
        float new_before_s = racing_route_s(route, count, length, i,
            racing_project(before, route[i].center, route[new_next].center));
        float previous_error = old_before_s - previous_s;
        float physical_delta = s - new_before_s;
        if (previous_error < -length * 0.5f) previous_error += length;
        if (previous_error > length * 0.5f) previous_error -= length;
        if (physical_delta < -length * 0.5f) physical_delta += length;
        if (physical_delta > length * 0.5f) physical_delta -= length;
        continuous = isfinite(previous_error) && isfinite(physical_delta)
            && fabsf(previous_error) <= 0.1f && fabsf(physical_delta) <= motion * 2 + 0.1f;
    }
    t->route_jump = !continuous;
    t->route_delta = delta;
    t->motion = motion;
    bool valid = !t->offroad && !car->trial.invalid && continuous;
    // Track backward/off-road motion too, so returning to the road cannot replay progress credit.
    if (continuous) t->distance += delta;
    float fresh = fmaxf(0, t->distance - t->furthest);
    t->furthest = fmaxf(t->furthest, t->distance);
    if (valid) {
        float credit = 0.1f * fresh;
        t->pending += credit;
        t->progress_credit += credit;
    }
    t->stalled = valid && fresh > 0.001f ? 0 : t->stalled + RACING_DT;
    t->wrongway = t->heading < -0.25f && pf_length(car->body.linear_velocity) > 1
        ? t->wrongway + RACING_DT
        : 0;
    t->pending -= 0.01f * RACING_DT;
    if (car->crashed) {
        t->done = 1;
        // Refund only shaping actually earned on the unfinished lap.
        t->pending -= t->progress_credit + 5;
    }
}

__device__ static void racing_task_observe_device(const RacingTask *t, const RacingCar *car,
    RacingCarConfig c, const RacingRoutePoint *route, int count, float length,
    const PfOptixHit *hits, int gate_count, float *obs, int k) {
    if (k < 256) {
        bool hit = hits[k].triangle >= 0;
        obs[k] = hit ? racing_clamp(hits[k].distance / c.lidar_range, 0, 1) : 1;
        obs[256 + k] = hit ? 1 : 0;
    }
    if (k != 0)
        return;
    PfQuat inverse = pf_quat_conjugate(car->body.rotation);
    PfVec3 v = pf_quat_rotate(inverse, car->body.linear_velocity);
    PfVec3 w = pf_quat_rotate(inverse, car->body.angular_velocity);
    PfVec3 up = pf_quat_rotate(inverse, pf_v3(0, 1, 0));
    obs[512] = v.x / 100;
    obs[513] = v.y / 100;
    obs[514] = v.z / 100;
    obs[515] = w.x / 5;
    obs[516] = w.y / 5;
    obs[517] = w.z / 5;
    obs[518] = up.x;
    obs[519] = up.y;
    obs[520] = up.z;
    obs[521] = -car->steering / c.steer_limit;
    obs[522] = car->throttle;
    obs[523] = car->rpm / 8000;
    obs[524] = car->gear / 6.0f;
    for (int i = 0; i < 4; i++) {
        obs[525 + i] = car->omega[i] * c.radius / 100;
        obs[529 + i] = (car->wheel_y[i] + c.rest_length) / c.travel;
        obs[533 + i] = car->contacts[i];
        obs[537 + i] = car->grip[i] / c.tarmac_grip;
    }
    for (int i = 0; i < 3; i++)
        obs[541 + i] = t->action[i];
#ifdef RACING_MULTI
    const float lookahead[6] = {10, 25, 50, 100, 200, 350};
#else
    const float lookahead[6] = {5, 10, 20, 40, 80, 120};
#endif
    int index = t->cursor;
    for (int j = 0; j < 6; j++) {
        float target = fmodf(t->s + lookahead[j], length);
        for (int n = 0; n < count; n++) {
            float end = index + 1 == count ? length : route[index + 1].s;
            if (target >= route[index].s && target <= end)
                break;
            index = (index + 1) % count;
        }
        int next = (index + 1) % count;
        float end = next == 0 ? length : route[next].s;
        float f = (target - route[index].s) / fmaxf(end - route[index].s, 1e-8f);
        PfVec3 point = pf_add(
            route[index].center, pf_scale(pf_sub(route[next].center, route[index].center), f));
        point = pf_scale(pf_quat_rotate(inverse, pf_sub(point, car->body.position)), 1.0f / 120);
        obs[544 + 3 * j] = point.x;
        obs[545 + 3 * j] = point.y;
        obs[546 + 3 * j] = point.z;
    }
    obs[562] = t->lateral / 30;
    obs[563] = t->heading;
    obs[564] = car->trial.next / (float)gate_count;
    obs[565] = t->ticks / (240.0f * (t->time_limit > 0 ? t->time_limit : 300.0f));
    obs[566] = t->offroad;
    obs[567] = t->furthest / length;
    for (int i = 512; i < RACING_BASE_OBS_SIZE; i++)
        obs[i] = isfinite(obs[i]) ? racing_clamp(obs[i], -5, 5) : 0;
}

__global__ static void racing_task_action(
    RacingTask *t, const float *actions, float steering, float throttle, float brake) {
    float input[3] = {steering, throttle, brake};
    for (int i = 0; i < 3; i++) {
        float value = actions ? actions[i] : input[i];
        t->action[i] = isfinite(value) ? racing_clamp(value, i == 0 ? -1 : 0, 1) : 0;
    }
}
__global__ static void racing_task_publish(RacingTask *t, float *reward, float *terminal) {
    t->reward = t->pending;
    t->pending = 0;
    t->total_reward += t->reward;
    if (reward)
        *reward = t->reward;
    if (terminal)
        *terminal = t->done != 0;
}

__global__ static void racing_task_reset(
    RacingTask *t, const RacingCar *car, const RacingRoutePoint *route, int count, float length) {
    racing_task_reset_device(t, car, route, count, length);
}
__global__ static void racing_task_observe(const RacingTask *t, const RacingCar *car,
    RacingCarConfig c, const RacingRoutePoint *route, int count, float length,
    const PfOptixHit *hits, int gate_count, float *obs) {
    racing_task_observe_device(t, car, c, route, count, length, hits, gate_count, obs,
        threadIdx.x + blockIdx.x * blockDim.x);
}
