#pragma once
#include "../../src/puffysics/collision.cuh"
#include "../../src/puffysics/contact_solver.cuh"
#include "../../src/puffysics/position_solver.cuh"
#include "barriers.cuh"
#include "rewards.cuh"

static constexpr int RACING_RACE_MAX = 8;
static constexpr float RACING_STALL_SPEED = 0.5f; // Retire persistent wall crawling below 1.8 km/h.
static constexpr int RACING_STALL_TICKS = 5 * 240;
struct RacingRaceState {
    int ticks; unsigned seed;
    float lap_record, lap_target;
    unsigned promotion_generation;
};
struct RacingRaceBatch {
    int cars, races, max_ticks;
    float length, initial_lap_record, reward_discount;
    unsigned promotion_generation, promotion_active;
    unsigned *promotion_races, *promotion_points;
    RacingRaceState *states;
    RacingCar *initial;
    RacingTask *initial_tasks;
    PfBody *bodies;
    PfManifold *contacts;
};
static RacingRaceBatch racing_race;

// Cars are policy-major: slot*races + race. Each race has exactly one car from each slot.
// Positive means a leads b. Finishers rank by absolute finish time, then active cars, then DNFs.
__device__ static int racing_race_compare(const Env &a, const Env &b) {
    bool af = a.car.trial.laps > 0, bf = b.car.trial.laps > 0;
    if (af != bf) return af ? 1 : -1;
    if (af) {
        double delta = b.car.trial.started - a.car.trial.started;
        return delta > 1e-7 ? 1 : delta < -1e-7 ? -1 : 0;
    }
    bool a_dnf = a.car.crashed || a.task.done == 5 || a.task.done == 3;
    bool b_dnf = b.car.crashed || b.task.done == 5 || b.task.done == 3;
    if (a_dnf != b_dnf) return a_dnf ? -1 : 1;
    double delta = a.race_progress - b.race_progress;
    return delta > 1e-4f ? 1 : delta < -1e-4f ? -1 : 0;
}
__device__ static float racing_race_score(Env *envs, RacingRaceBatch race, int world, int slot,
    int *rank) {
    float score = 0;
    *rank = 1;
    for (int other = 0; other < race.cars; other++) {
        if (other == slot) continue;
        int cmp = racing_race_compare(envs[slot * race.races + world], envs[other * race.races + world]);
        score += cmp;
        *rank += cmp < 0;
    }
    return score / (race.cars - 1);
}
__device__ static void racing_race_reset_one(Env *envs, RacingRaceBatch race, int world,
    bool clear) {
    auto &state = race.states[world];
    unsigned seed = clear ? (unsigned)(world + 1) * 747796405u : state.seed;
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    float record = clear ? race.initial_lap_record : state.lap_record;
    state = {0, seed, record, record, race.promotion_generation};
    for (int slot = 0; slot < race.cars; slot++) {
        Env &e = envs[slot * race.races + world];
        if (clear) e = {};
        int grid = (slot + seed % race.cars) % race.cars;
        if (seed & 0x100) grid = race.cars - 1 - grid;
        e.car = race.initial[grid];
        e.task = race.initial_tasks[grid];
        e.num_agents = 1;
        e.race_before = e.car.body.position;
        e.race_rotation = e.car.body.rotation;
        e.race_reward = 0;
        e.wall_impact_penalty = 0;
        e.wall_contact_penalty = e.wall_contact_seconds = 0;
        e.wall_contact = e.car_contact = false;
        e.car_contact_penalty = e.car_contact_seconds = 0;
        e.offroad_seconds = e.wrongway_seconds = e.active_seconds = 0;
        e.route_jumps = 0; e.checkpoint_bonus = e.saturated_seconds = 0;
        e.progress_reward = e.time_penalty = e.delay_penalty = 0;
        e.speed_integral = e.throttle_integral = e.brake_integral = 0;
        e.race_contacts = e.barrier_contacts = e.barrier_overflow = 0;
        e.overturned_ticks = e.stopped_ticks = e.race_settled = 0;
        e.car_impacts = e.wall_impacts = 0;
        e.last_car_impact_tick = e.last_wall_impact_tick = -RACING_IMPACT_COOLDOWN;
        e.race_progress = e.task.s - race.length;
        e.race_progress_limit = race.length - e.race_progress;
        e.task.time_limit = race.max_ticks * RACING_DT;
    }
    for (int slot = 0; slot < race.cars; slot++) {
        Env &e = envs[slot * race.races + world];
        e.race_score = racing_race_score(envs, race, world, slot, &e.race_rank);
    }
}
__global__ static void racing_race_templates(RacingRaceBatch race, RacingCar initial,
    RacingCarConfig config, const RacingRoutePoint *route, int count, float length) {
    int i = threadIdx.x;
    if (i >= race.cars) return;
    RacingCar car = initial;
    float column = fmaxf(3.6f, config.track_width + 0.25f + 1.0f);
    float row = fmaxf(6.0f, config.wheelbase * 1.72f + 1.0f);
    PfVec3 offset = pf_quat_rotate(car.body.rotation,
        pf_v3((i % 2 ? -0.5f : 0.5f) * column, 0, -row * (i / 2)));
    car.body.position = pf_add(car.body.position, offset);
    RacingTask task;
    racing_task_reset_device(&task, &car, route, count, length);
    int next = (task.cursor + 1) % count;
    float f = racing_project(car.body.position, route[task.cursor].center, route[next].center);
    car.body.position.y = route[task.cursor].center.y * (1 - f) + route[next].center.y * f
        + config.radius + config.rest_length - config.mass * 9.81f / (4 * config.spring);
    car.body.shape = PF_BOX;
    car.body.half_extents = pf_v3((config.track_width + 0.25f) * 0.5f, 0.6f, config.wheelbase * 0.86f);
    car.body.friction = 0.5f;
    car.body.restitution = 0.05f;
    racing_task_reset_device(&task, &car, route, count, length);
    race.initial[i] = car;
    race.initial_tasks[i] = task;
}
// One-time ground placement on the actual support mesh, cached for every reset.
__global__ static void racing_race_grid_rays(RacingRaceBatch race, PfOptixRay *rays) {
    int i = threadIdx.x;
    if (i >= race.cars) return;
    PfVec3 p = race.initial[i].body.position;
    rays[i] = {make_float3(p.x, p.y + 5, p.z), 0.001f, make_float3(0,-1,0), 15};
}
__global__ static void racing_race_grid_ground(RacingRaceBatch race, RacingCarConfig config,
    const PfOptixRay *rays, const PfOptixHit *hits, const RacingRoutePoint *route,
    int count, float length) {
    int i = threadIdx.x;
    if (i >= race.cars) return;
    assert(hits[i].triangle >= 0 && "Race grid must sit on the support mesh");
    auto &car = race.initial[i];
    PfVec3 normal = pf_v3(hits[i].normal.x, hits[i].normal.y, hits[i].normal.z);
    if (normal.y < 0) normal = pf_scale(normal, -1);
    normal = pf_normalize_or(normal, pf_v3(0,1,0));
    assert(normal.y > 0.9f && "Race grid needs a nearly level road surface");
    PfQuat tilt = pf_quat_normalize(PfQuat{1 + normal.y, normal.z, 0, -normal.x});
    car.body.rotation = pf_quat_multiply(tilt, car.body.rotation);
    float height = config.radius + config.rest_length - config.mass * 9.81f / (4 * config.spring);
    PfVec3 ground = pf_v3(rays[i].origin.x, rays[i].origin.y - hits[i].distance, rays[i].origin.z);
    car.body.position = pf_add(ground, pf_scale(normal, height));
    racing_task_reset_device(&race.initial_tasks[i], &car, route, count, length);
    for (int k = 0; k < 4; k++) {
        PfVec3 corner = pf_v3((k & 1 ? 1 : -1) * car.body.half_extents.x,
            -height, (k & 2 ? 1 : -1) * car.body.half_extents.z);
        PfVec3 p = pf_add(car.body.position, pf_quat_rotate(car.body.rotation, corner));
        assert(racing_on_road(route, count, race.initial_tasks[i].cursor, p)
            && "Race grid chassis must fit within the road boundaries");
    }
}
__global__ static void racing_race_grid_wheels(RacingRaceBatch race, RacingCarConfig config,
    PfOptixRay *rays) {
    int i = threadIdx.x;
    if (i >= race.cars) return;
    PfOptixRay local[12];
    racing_car_rays_device(&race.initial[i], config, local);
    for (int wheel = 0; wheel < 4; wheel++) rays[i*4+wheel] = local[wheel];
}
__global__ static void racing_race_grid_supported(RacingRaceBatch race, RacingCarConfig config,
    const PfOptixRay *rays, const PfOptixHit *hits) {
    int i = threadIdx.x;
    if (i >= race.cars) return;
    const PfBody &body = race.initial[i].body;
    PfVec3 up = pf_quat_rotate(body.rotation,pf_v3(0,1,0));
    for (int wheel = 0; wheel < 4; wheel++) {
        int j = i*4+wheel;
        assert(hits[j].triangle >= 0 && "Starting grid must support all four wheels");
        PfVec3 anchor = pf_add(body.position,pf_quat_rotate(body.rotation,racing_anchor(config,wheel)));
        PfVec3 origin = pf_v3(rays[j].origin.x,rays[j].origin.y,rays[j].origin.z);
        float distance = hits[j].distance-pf_dot(pf_sub(origin,anchor),up);
        PfVec3 normal = pf_v3(hits[j].normal.x,hits[j].normal.y,hits[j].normal.z);
        assert(pf_dot(normal,up)>0.35f && distance <= config.rest_length+config.radius
            && distance >= config.rest_length+config.radius-config.travel
            && "Starting grid wheel support must be within suspension travel");
    }
}
__global__ static void racing_race_grid_clearance(RacingRaceBatch race) {
    int i = threadIdx.x;
    if (i >= race.cars) return;
    for (int j = i + 1; j < race.cars; j++) {
        PfManifold contact;
        assert(!pf_collision(i, &race.initial[i].body, j, &race.initial[j].body, &contact)
            && "Race grid cars must not overlap");
    }
}
__global__ static void racing_race_reset(Env *envs, RacingRaceBatch race, float *rewards,
    float *terminals, bool clear) {
    int world = blockIdx.x * blockDim.x + threadIdx.x;
    if (world >= race.races) return;
    racing_race_reset_one(envs, race, world, clear);
    for (int slot = 0; slot < race.cars; slot++) {
        int i = slot * race.races + world;
        rewards[i] = terminals[i] = 0;
    }
}
__global__ static void racing_race_contacts(Env *envs, RacingRaceBatch race) {
    int world = blockIdx.x * blockDim.x + threadIdx.x;
    if (world >= race.races) return;
    PfBody *bodies = race.bodies + world * race.cars;
    int slots[RACING_RACE_MAX], count = 0;
    for (int slot = 0; slot < race.cars; slot++) {
        Env &e = envs[slot * race.races + world];
        e.car_contact = false;
        if (e.task.done || e.car.crashed) continue;
        slots[count] = slot;
        bodies[count++] = e.car.body;
    }
    if (count < 2) return;
    int capacity = race.cars * (race.cars - 1) / 2;
    PfWorld contacts = {};
    contacts.bodies = bodies;
    contacts.body_count = count;
    contacts.manifolds = race.contacts + world * capacity;
    contacts.manifold_capacity = capacity;
    // pf_collision rejects separated bounding spheres before box SAT/manifold work.
    // Rechecked every 1/240 s at current poses; proximity never applies a force.
    if (pf_detect_contacts(&contacts) > 0) {
        // Applied at contact points: linear momentum transfer plus off-centre angular impulse.
        pf_solve_velocity_contacts(bodies, contacts.manifolds, contacts.manifold_count);
        float impact[RACING_RACE_MAX] = {};
        for (int j = 0; j < contacts.manifold_count; j++) {
            const PfManifold &m = contacts.manifolds[j];
            impact[m.body_a] = fmaxf(impact[m.body_a], m.normal_impulse);
            impact[m.body_b] = fmaxf(impact[m.body_b], m.normal_impulse);
            envs[slots[m.body_a] * race.races + world].car_contact = true;
            envs[slots[m.body_b] * race.races + world].car_contact = true;
        }
        pf_solve_positions(&contacts);
        for (int j = 0; j < contacts.manifold_count; j++) {
            const PfManifold &m = contacts.manifolds[j];
            envs[slots[m.body_a] * race.races + world].race_contacts++;
            envs[slots[m.body_b] * race.races + world].race_contacts++;
        }
        for (int j = 0; j < count; j++) {
            Env &e = envs[slots[j] * race.races + world];
            if (e.car_contact) {
                // Once per car per physics step, even if several opponents touch it.
                float cost = RACING_CONTACT_COST_PER_SECOND * RACING_DT;
                e.race_reward -= cost;
                e.car_contact_penalty += cost;
                e.car_contact_seconds += RACING_DT;
            }
            float cost = racing_impact_cost(impact[j], bodies[j].inverse_mass);
            if (cost > 0 && e.last_car_impact_tick + RACING_IMPACT_COOLDOWN <= race.states[world].ticks) {
                e.race_reward -= cost;
                e.car_impacts++;
                e.last_car_impact_tick = race.states[world].ticks;
            }
        }
    }
    for (int j = 0; j < count; j++) envs[slots[j] * race.races + world].car.body = bodies[j];
}
__global__ static void racing_race_tasks(Env *envs, RacingRaceBatch race, RacingCarConfig config,
    const RacingRoutePoint *route, int route_count, float length,
    const RacingGate *gates, int gate_count) {
    int world = blockIdx.x, slot = threadIdx.x;
    // One block per race, one lane per car. Standings read all updated cars after the barrier.
    if (slot == 0) race.states[world].ticks++;
    __shared__ float deltas[RACING_RACE_MAX];
    __shared__ bool active[RACING_RACE_MAX];
    deltas[slot] = 0;
    active[slot] = false;
    if (!envs[slot * race.races + world].task.done) {
        Env &e = envs[slot * race.races + world];
        active[slot] = true;
        float previous_furthest = e.task.furthest;
        int previous_checkpoints = e.car.trial.checkpoints;
        PfVec3 up = pf_quat_rotate(e.car.body.rotation,pf_v3(0,1,0));
        e.overturned_ticks = up.y < 0.25f ? e.overturned_ticks + 1 : 0;
        if (e.overturned_ticks >= 3 * 240) e.car.crashed = 1;
        if (e.car.crashed) {
            if (e.car.trial.active) e.car.trial.invalid = 3;
            // Retired cars remain in the batch until race end: keep their observations finite.
            if (!pf_vec_valid(e.car.body.position) || !pf_quat_valid(e.car.body.rotation))
                e.car.body = race.initial[slot].body;
            e.car.body.linear_velocity = e.car.body.angular_velocity = pf_v3(0,0,0);
        }
        racing_task_step(&e.task, &e.car, config, route, route_count, length, e.race_before);
        // Race rewards replace time-trial shaping/refunds, keeping geometry bookkeeping.
        e.task.pending = e.task.progress_credit = 0;
        e.active_seconds += RACING_DT;
        e.speed_integral += pf_length(e.car.body.linear_velocity) * RACING_DT;
        e.throttle_integral += e.task.action[1] * RACING_DT;
        e.brake_integral += e.task.action[2] * RACING_DT;
        e.saturated_seconds += (fabsf(e.task.action[0]) >= 0.95f
            || e.task.action[1] >= 0.95f || e.task.action[2] >= 0.95f) ? RACING_DT : 0;
        e.offroad_seconds += e.task.offroad ? RACING_DT : 0;
        e.wrongway_seconds += e.task.wrongway > 0 ? RACING_DT : 0;
        e.route_jumps += e.task.route_jump != 0;
        if (!e.task.done) {
            PfVec3 v = e.car.body.linear_velocity;
            e.stopped_ticks = v.x*v.x + v.z*v.z < RACING_STALL_SPEED*RACING_STALL_SPEED
                ? e.stopped_ticks + 1 : 0;
            if (e.stopped_ticks >= RACING_STALL_TICKS) e.task.done = 5;
        }
        if (e.car.crashed || e.task.done == 5) {
            e.race_reward -= 0.1f;
        } else {
            racing_trial_step(&e.car.trial, gates, gate_count, e.race_before, e.car.body.position,
                e.task.s, e.task.route_delta, length, e.task.route_jump);
            if (e.car.trial.invalid == 2) e.task.done = 3;
            else if (e.car.trial.laps > 0) e.task.done = 7;
            bool clean = !e.task.offroad && !e.task.route_jump && !e.car.trial.invalid
                && !e.wall_contact && !e.car_contact;
            float bonus = racing_checkpoint_reward(e.car.trial.checkpoints - previous_checkpoints,
                race.states[world].ticks * RACING_DT, race.max_ticks * RACING_DT, clean);
            e.race_reward += bonus; e.checkpoint_bonus += bonus;
        }
        e.race_progress = racing_route_progress(e.race_progress, e.task.s, length);
        // Standing uses signed metres; shaping pays each new maximum only once.
        deltas[slot] = racing_progress_reward(e.task, previous_furthest, e.race_progress_limit,
            e.wall_contact || e.car_contact || e.car.crashed);
        e.progress_reward += deltas[slot];
        e.time_penalty += 0.005f * RACING_DT;
        e.race_reward -= 0.005f * RACING_DT;
    }
    __syncthreads();
    Env &e = envs[slot * race.races + world];
    float score = racing_race_score(envs, race, world, slot, &e.race_rank);
    bool clean = !e.task.offroad && !e.task.route_jump && !e.car.trial.invalid
        && !e.wall_contact && !e.car_contact && !e.car.crashed && e.task.done != 5;
    if (active[slot]) e.race_reward += racing_position_change_reward(score - e.race_score, clean)
        + deltas[slot];
    e.race_score = score;
}
__global__ static void racing_race_finish(Env *envs, RacingRaceBatch race,
    float *rewards, float *terminals) {
    int world = blockIdx.x * blockDim.x + threadIdx.x;
    if (world >= race.races) return;
    int remaining = 0;
    for (int slot = 0; slot < race.cars; slot++) remaining += !envs[slot * race.races + world].task.done;
    bool timeout = race.states[world].ticks >= race.max_ticks;
    bool done = remaining == 0 || timeout;
    if (done && race.promotion_active &&
        race.states[world].promotion_generation == race.promotion_generation) {
        unsigned seed = race.states[world].seed;
        int grid = (2 + seed % race.cars) % race.cars;
        if (seed & 0x100) grid = race.cars - 1 - grid;
        int comparison = racing_race_compare(envs[2 * race.races + world],
            envs[race.races + world]);
        atomicAdd(&race.promotion_races[grid], 1u);
        atomicAdd(&race.promotion_points[grid], comparison > 0 ? 2u : comparison == 0 ? 1u : 0u);
    }
    for (int slot = 0; slot < race.cars; slot++) {
        int i = slot * race.races + world;
        Env &e = envs[i];
        bool learning_end = !e.race_settled && (e.task.done || done);
        // Finish order is fixed at crossing; crashes, stalls and unfinished timeouts lose.
        // Waiting for other cars must neither create training samples nor delayed credit.
        float outcome = !e.car.trial.laps || e.car.crashed || e.task.done == 5
            ? racing_failure_outcome(race.reward_discount) : RACING_POSITION_REWARD * e.race_score;
        float reward = e.race_settled ? 0 : e.race_reward + (learning_end ? outcome : 0);
        if (!e.race_settled) {
            // One charge per active action, including a terminal action. No
            // further cost or terminal credit while a retired car waits.
            float cost = racing_failure_delay_cost(race.reward_discount);
            reward -= cost;
            e.delay_penalty += cost;
        }
        if (learning_end && e.car.trial.laps > 0) {
            auto &state = race.states[world];
            float lap = (float)e.car.trial.last;
            float finish = (float)e.car.trial.started;
            float limit = race.max_ticks * RACING_DT;
            // Absolute finish time includes the run-up: waiting before the line cannot
            // improve this bonus. Race records are telemetry, not reward targets.
            reward += racing_lap_bonus(finish, limit);
            state.lap_record = fminf(state.lap_record, lap);
        }
        if (learning_end) e.race_settled = 1;
        reward = racing_reward_for_ppo(reward);
        e.task.reward = rewards[i] = reward;
        e.task.total_reward += reward;
        e.race_reward = 0;
        terminals[i] = done || e.task.done;
        if (!done) continue;
        auto &end = e.last_end;
        end.serial++;
        end.reason = e.car.crashed ? 1 : e.task.done == 3 ? 3 : e.task.done == 5 ? 5 : e.car.trial.laps ? 7 : 6;
        end.invalid = e.car.trial.invalid; end.route_jump = e.task.route_jump;
        end.offroad_wheels = e.task.offroad_wheels;
        end.route_delta = e.task.route_delta; end.motion = e.task.motion;
        end.throttle = e.task.action[1]; end.brake = e.task.action[2];
        end.seconds = race.states[world].ticks * RACING_DT;
        end.speed = pf_length(e.car.body.linear_velocity);
        end.progress = e.task.furthest;
        end.checkpoints = e.car.trial.checkpoints;
        end.next_gate = e.car.trial.next;
        end.position[0] = e.car.body.position.x;
        end.position[1] = e.car.body.position.y;
        end.position[2] = e.car.body.position.z;
        e.log.episode_return += e.task.total_reward;
        e.log.progress += e.task.furthest;
        e.log.progress_reward += racing_reward_for_ppo(e.progress_reward);
        e.log.time_penalty += racing_reward_for_ppo(e.time_penalty);
        e.log.delay_penalty += racing_reward_for_ppo(e.delay_penalty);
        float inv_seconds = e.active_seconds > 0 ? 1 / e.active_seconds : 0;
        e.log.mean_speed += e.speed_integral * inv_seconds;
        e.log.mean_throttle += e.throttle_integral * inv_seconds;
        e.log.mean_brake += e.brake_integral * inv_seconds;
        e.log.checkpoints += e.car.trial.checkpoints;
        e.log.laps += e.car.trial.laps;
        if (e.car.trial.laps > 0) {
            float lap = (float)e.car.trial.last;
            float target = race.states[world].lap_target;
            e.log.lap_seconds += lap;
            e.log.record_beats += lap < target;
            e.log.lap_bonus += racing_reward_for_ppo(racing_lap_bonus(
                (float)e.car.trial.started, race.max_ticks * RACING_DT));
        }
        e.log.crashes += e.car.crashed != 0;
        e.log.offroad += e.offroad_seconds > 0;
        e.log.invalid += e.car.trial.invalid == 2;
        e.log.wrongway += e.wrongway_seconds > 0;
        e.log.offroad_seconds += e.offroad_seconds;
        e.log.wrongway_seconds += e.wrongway_seconds;
        e.log.route_jumps += e.route_jumps;
        e.log.control_saturation += e.active_seconds > 0 ? e.saturated_seconds / e.active_seconds : 0;
        e.log.checkpoint_bonus += racing_reward_for_ppo(e.checkpoint_bonus);
        e.log.stalled += e.task.done == 5;
        e.log.timeout += !e.task.done;
        e.log.contacts += e.race_contacts;
        e.log.wall_contacts += e.barrier_contacts;
        e.log.car_impacts += e.car_impacts;
        e.log.wall_impacts += e.wall_impacts;
        e.log.wall_impact_penalty += racing_reward_for_ppo(e.wall_impact_penalty);
        e.log.wall_contact_penalty += racing_reward_for_ppo(e.wall_contact_penalty);
        e.log.wall_contact_seconds += e.wall_contact_seconds;
        e.log.car_contact_penalty += racing_reward_for_ppo(e.car_contact_penalty);
        e.log.car_contact_seconds += e.car_contact_seconds;
        e.log.contact_overflow += e.barrier_overflow;
        e.log.position += e.race_rank;
        e.log.wins += e.car.trial.laps > 0 && e.race_rank == 1;
        e.log.n++;
    }
    if (done) racing_race_reset_one(envs, race, world, false);
}

// Small dynamic set: intersect opponent boxes after the shared static OptiX scene.
__device__ static bool racing_race_ray_box(const PfBody &body, PfVec3 origin, PfVec3 direction,
    float tmin, float *distance, PfVec3 *normal) {
    PfQuat inverse = pf_quat_conjugate(body.rotation);
    PfVec3 o = pf_quat_rotate(inverse, pf_sub(origin, body.position));
    PfVec3 d = pf_quat_rotate(inverse, direction);
    float ov[3] = {o.x,o.y,o.z}, dv[3] = {d.x,d.y,d.z};
    float extent[3] = {body.half_extents.x,body.half_extents.y,body.half_extents.z};
    float enter = -1e30f, exit = 1e30f;
    PfVec3 enter_normal = {}, exit_normal = {};
    for (int axis = 0; axis < 3; axis++) {
        if (fabsf(dv[axis]) < 1e-8f) {
            if (fabsf(ov[axis]) > extent[axis]) return false;
            continue;
        }
        float a = (-extent[axis] - ov[axis]) / dv[axis];
        float b = (extent[axis] - ov[axis]) / dv[axis];
        float sign = dv[axis] > 0 ? -1 : 1;
        if (a > b) { float tmp = a; a = b; b = tmp; }
        PfVec3 n = pf_v3(axis == 0 ? sign : 0, axis == 1 ? sign : 0, axis == 2 ? sign : 0);
        if (a > enter) { enter = a; enter_normal = n; }
        if (b < exit) { exit = b; exit_normal = pf_scale(n, -1); }
    }
    float t = enter >= tmin ? enter : exit;
    if (enter > exit || t < tmin || t >= *distance) return false;
    *distance = t;
    *normal = pf_quat_rotate(body.rotation, enter >= tmin ? enter_normal : exit_normal);
    return true;
}
__global__ static void racing_race_lidar(const Env *envs, int cars, int races,
    const PfOptixRay *rays, PfOptixHit *hits) {
    int i = blockIdx.x, k = threadIdx.x, world = i % races;
    int index = i * 256 + k;
    PfOptixRay ray = rays[index];
    PfOptixHit hit = hits[index];
    if (hit.triangle == -2) return;
    for (int slot = 0; slot < cars; slot++) {
        int other = slot * races + world;
        if (other == i || envs[other].task.done || envs[other].car.crashed) continue;
        PfVec3 normal;
        if (racing_race_ray_box(envs[other].car.body, pf_v3(ray.origin.x,ray.origin.y,ray.origin.z),
            pf_v3(ray.direction.x,ray.direction.y,ray.direction.z), ray.tmin, &hit.distance, &normal)) {
            hit.normal = make_float3(normal.x,normal.y,normal.z);
            hit.triangle = 0; // Observation uses hit/miss only, never this dynamic primitive index.
            hit.material = -1;
        }
    }
    hits[index] = hit;
}
static void racing_race_create(Dict *kwargs) {
    auto &race = racing_race;
    auto &b = racing_batch;
    // Bumper-height sensor intersects nearby chassis instead of passing above them.
    car_config.lidar_mount_y = 0.45f;
    car_config.lidar_range = 300.0f;
    race.cars = (int)dict_get(kwargs, "race_cars");
    assert(race.cars >= 2 && race.cars <= RACING_RACE_MAX && b.count % race.cars == 0);
    race.races = b.count / race.cars;
    float seconds = dict_get(kwargs, "race_seconds");
    assert(isfinite(seconds) && seconds >= 1 && seconds <= 3600);
    race.max_ticks = (int)(seconds * 240);
    race.reward_discount = dict_find(kwargs, "reward_discount")
        ? (float)dict_get(kwargs, "reward_discount") : 0.9999f;
    assert(isfinite(race.reward_discount) && race.reward_discount >= 0 && race.reward_discount <= 1);
    race.initial_lap_record = dict_get(kwargs, "lap_record_seconds");
    assert(isfinite(race.initial_lap_record) && race.initial_lap_record > 0);
    race.length = task_route_length;
    pf_optix_cuda(cudaMalloc(&race.states, race.races * sizeof(RacingRaceState)));
    pf_optix_cuda(cudaMalloc(&race.promotion_races, RACING_RACE_MAX * sizeof(unsigned)));
    pf_optix_cuda(cudaMalloc(&race.promotion_points, RACING_RACE_MAX * sizeof(unsigned)));
    pf_optix_cuda(cudaMemset(race.promotion_races, 0, RACING_RACE_MAX * sizeof(unsigned)));
    pf_optix_cuda(cudaMemset(race.promotion_points, 0, RACING_RACE_MAX * sizeof(unsigned)));
    pf_optix_cuda(cudaMalloc(&race.initial, race.cars * sizeof(RacingCar)));
    pf_optix_cuda(cudaMalloc(&race.initial_tasks, race.cars * sizeof(RacingTask)));
    pf_optix_cuda(cudaMalloc(&race.bodies, b.count * sizeof(PfBody)));
    pf_optix_cuda(cudaMalloc(&race.contacts, race.races * race.cars * (race.cars - 1) / 2 * sizeof(PfManifold)));
    racing_race_templates<<<1, 32>>>(race, b.initial, car_config, task_route, task_route_count, task_route_length);
    pf_optix_cuda(cudaGetLastError());
    racing_race_grid_rays<<<1, 32>>>(race, b.rays);
    pf_optix_launch(&rt, b.queries, race.cars, 0);
    racing_race_grid_ground<<<1, 32>>>(race, car_config, b.rays, b.hits,
        task_route, task_route_count, task_route_length);
    racing_race_grid_clearance<<<1, 32>>>(race);
    pf_optix_cuda(cudaGetLastError());
    pf_optix_cuda(cudaDeviceSynchronize());
    float wall_crash_speed = dict_get(kwargs, "wall_crash_speed");
    assert(isfinite(wall_crash_speed) && wall_crash_speed > 0);
    racing_barrier_create(meshes[1], b.count, wall_crash_speed);
    // Use the same support scene and suspension queries as time trials.
    racing_race_grid_wheels<<<1, 32>>>(race, car_config, b.rays);
    pf_optix_launch(&rt, b.queries, race.cars*4, 0);
    racing_race_grid_supported<<<1, 32>>>(race, car_config, b.rays, b.hits);
    pf_optix_cuda(cudaGetLastError());
    pf_optix_cuda(cudaDeviceSynchronize());
    if (race.races == 1) {
        if (!racing_ghosts.count) racing_shuffle_names(race.cars);
        racing_ghosts.count = race.cars;
        if (!b.ghost_snapshot) pf_optix_cuda(cudaMalloc(&b.ghost_snapshot, race.cars * sizeof(RacingGhostSnapshot)));
    }
}
static void racing_race_close() {
    racing_barrier_close();
    cudaFree(racing_race.states); cudaFree(racing_race.initial); cudaFree(racing_race.initial_tasks);
    cudaFree(racing_race.promotion_races); cudaFree(racing_race.promotion_points);
    cudaFree(racing_race.bodies); cudaFree(racing_race.contacts);
    racing_race = {};
}
