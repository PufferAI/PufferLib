#pragma once

static constexpr int RACING_NEAR_BASE = RACING_BASE_OBS_SIZE + RACING_CAR_BINS;
static constexpr int RACING_GROUND_BASE = RACING_NEAR_BASE + RACING_NEAR_CARS * 7;

__device__ static PfOptixRay racing_ground_ray(const RacingCar &car,
    RacingCarConfig config, int probe) {
    const float distance[3] = {10, 25, 50};
    PfVec3 forward = pf_quat_rotate(car.body.rotation, pf_v3(0, 0, 1));
    forward = pf_normalize_or(pf_v3(forward.x, 0, forward.z), pf_v3(0, 0, 1));
    PfVec3 left = pf_v3(forward.z, 0, -forward.x);
    float side = (probe & 1 ? -0.5f : 0.5f) * config.track_width;
    PfVec3 origin = pf_add(car.body.position,
        pf_add(pf_scale(forward, distance[probe / 2]), pf_scale(left, side)));
    origin.y += 5;
    return {make_float3(origin.x, origin.y, origin.z), 0.01f,
        make_float3(0, -1, 0), 20};
}

__device__ static void racing_extra_observe(const Env *envs, int cars, int races,
    int i, int k, RacingCarConfig config, const PfOptixHit *lidar,
    const PfOptixRay *ground_rays, const PfOptixHit *ground_hits, float *obs) {
    if (k < RACING_CAR_BINS) {
        float proximity = 0;
        for (int j = 0; j < 256 / RACING_CAR_BINS; j++) {
            int ring = j / (128 / RACING_CAR_BINS);
            int azimuth = k * (128 / RACING_CAR_BINS) + j % (128 / RACING_CAR_BINS);
            PfOptixHit hit = lidar[ring * 128 + azimuth];
            if (hit.triangle >= 0 && hit.material == -1)
                proximity = fmaxf(proximity, 1 - hit.distance / config.lidar_range);
        }
        obs[RACING_BASE_OBS_SIZE + k] = racing_clamp(proximity, 0, 1);
    }
    if (k < RACING_GROUND_PROBES) {
        PfOptixHit hit = ground_hits[k];
        int base = RACING_GROUND_BASE + k * 3;
        bool valid = hit.triangle >= 0 && hit.normal.y > 0.35f;
        float grip = config.tarmac_grip;
        if (hit.material == 44 || hit.material == 56) grip = config.curb_grip;
        if (hit.material == 53 || hit.material == 58) grip = config.grass_grip;
        if (hit.material == 32 || hit.material == 59) grip = config.gravel_grip;
        obs[base] = valid ? grip / config.tarmac_grip : 0;
        obs[base + 1] = valid ? racing_clamp(
            (ground_rays[k].origin.y - hit.distance - envs[i].car.body.position.y) / 10,
            -2, 2) : 0;
        obs[base + 2] = valid ? 1 : 0;
    }
    if (k != 0) return;

    int nearest[RACING_NEAR_CARS] = {-1, -1, -1};
    float distance_sq[RACING_NEAR_CARS] = {1e30f, 1e30f, 1e30f};
    const PfBody &self = envs[i].car.body;
    int world = i % races;
    for (int slot = 0; slot < cars; slot++) {
        int other = slot * races + world;
        if (other == i || envs[other].task.done || envs[other].car.crashed) continue;
        PfVec3 delta = pf_sub(envs[other].car.body.position, self.position);
        float d2 = delta.x * delta.x + delta.z * delta.z;
        for (int rank = 0; rank < RACING_NEAR_CARS; rank++) {
            if (d2 >= distance_sq[rank]) continue;
            for (int move = RACING_NEAR_CARS - 1; move > rank; move--) {
                nearest[move] = nearest[move - 1];
                distance_sq[move] = distance_sq[move - 1];
            }
            nearest[rank] = other;
            distance_sq[rank] = d2;
            break;
        }
    }
    PfQuat inverse = pf_quat_conjugate(self.rotation);
    for (int rank = 0; rank < RACING_NEAR_CARS; rank++) {
        int base = RACING_NEAR_BASE + rank * 7;
        if (nearest[rank] < 0) {
            for (int j = 0; j < 7; j++) obs[base + j] = 0;
            continue;
        }
        const PfBody &other = envs[nearest[rank]].car.body;
        PfVec3 position = pf_quat_rotate(inverse, pf_sub(other.position, self.position));
        PfVec3 velocity = pf_quat_rotate(inverse,
            pf_sub(other.linear_velocity, self.linear_velocity));
        PfVec3 heading = pf_quat_rotate(inverse,
            pf_quat_rotate(other.rotation, pf_v3(0, 0, 1)));
        obs[base] = racing_clamp(position.x / 100, -5, 5);
        obs[base + 1] = racing_clamp(position.z / 100, -5, 5);
        obs[base + 2] = racing_clamp(velocity.x / 100, -5, 5);
        obs[base + 3] = racing_clamp(velocity.z / 100, -5, 5);
        obs[base + 4] = heading.x;
        obs[base + 5] = heading.z;
        obs[base + 6] = 1;
    }
}
