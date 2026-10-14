#ifndef PUFFER_RACING_CU
#define PUFFER_RACING_CU
#define PUF_BACKEND PUF_GPU
#include <cuda_bf16.h>
typedef __nv_bfloat16 obs_t;
#include "pufferenv.h"
#include "map_queries.cu"
#include "eval_render.h"

#define NUM_ATNS 3 // Continuous: steer [-1 left, +1 right], throttle [0,1], brake [0,1].
#define ACT_SIZES {1, 1, 1}
#define OBS_SIZE RACING_OBS_SIZE

struct Log {
    float episode_return, progress, checkpoints, laps, crashes, offroad, invalid, wrongway, stalled, timeout, n;
#ifdef RACING_MULTI
    float position, wins, contacts, wall_contacts, contact_overflow;
    float car_impacts, wall_impacts, wall_impact_penalty;
    float wall_contact_penalty, wall_contact_seconds;
    float car_contact_penalty, car_contact_seconds;
    float lap_seconds, record_beats, lap_bonus;
    float offroad_seconds, wrongway_seconds, route_jumps, checkpoint_bonus, control_saturation;
    float progress_reward, time_penalty, delay_penalty, mean_speed, mean_throttle, mean_brake;
#endif
};
struct Env {
    Log log;
    Agent agents[1];
    int num_agents, tag, boundary_reached;
    unsigned rng;
    RacingCar car;
    RacingTask task;
    RacingEpisodeEnd last_end;
#ifdef RACING_MULTI
    PfVec3 race_before;
    PfQuat race_rotation;
    float race_reward, race_score, wall_impact_penalty, race_progress_limit;
    float wall_contact_penalty, wall_contact_seconds;
    float car_contact_penalty, car_contact_seconds;
    bool wall_contact, car_contact;
    float offroad_seconds, wrongway_seconds, active_seconds, checkpoint_bonus, saturated_seconds;
    float progress_reward, time_penalty, delay_penalty, speed_integral, throttle_integral, brake_integral;
    int route_jumps;
    double race_progress;
    int race_rank, race_contacts, barrier_contacts, barrier_overflow, overturned_ticks, stopped_ticks, race_settled;
    int last_car_impact_tick, last_wall_impact_tick, car_impacts, wall_impacts;
#endif
};
static struct {
    Env *envs;
    int count;
    obs_t *observations;
    float *actions, *rewards, *terminals, *float_obs;
    PfOptixRay *rays, *sensor_rays;
    PfOptixHit *hits, *sensor_hits;
#ifdef RACING_MULTI
    PfOptixRay *ground_rays;
    PfOptixHit *ground_hits;
#endif
    PfOptixQuery *queries;
    RacingCar initial;
    RacingTask initial_task;
    RacingGhostSnapshot *ghost_snapshot;
    cudaStream_t stream;
} racing_batch;
#ifdef RACING_MULTI
#include "../racing_multi/race.cuh"
#include "../racing_multi/observations.cuh"
#include "../racing_multi/handling_trace.cuh"
#endif
#include "trainer_hooks.cuh"

__global__ static void racing_batch_reset(Env *envs, int count, RacingCar initial,
    RacingTask initial_task, float *rewards, float *terminals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    envs[i] = {};
    envs[i].num_agents = 1;
    envs[i].rng = i + 1;
    envs[i].car = initial;
    envs[i].task = initial_task;
    rewards[i] = terminals[i] = 0;
}
__global__ static void racing_batch_actions(Env *envs, int count, const float *actions, int manual,
    float steering, float throttle, float brake) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
#ifdef RACING_MULTI
    if (envs[i].task.done) return;
#endif
    float input[3] = {steering, throttle, brake};
    if (manual != i + 1) {
        for (int k = 0; k < 3; k++) {
            float value = actions[i * 3 + k];
            input[k] = isfinite(value) ? value : 0;
        }
#ifdef RACING_MULTI
        float controls[3];
        racing_policy_controls(input, controls);
        for (int k = 0; k < 3; k++) input[k] = controls[k];
#endif
    }
    for (int k = 0; k < 3; k++)
        envs[i].task.action[k] = isfinite(input[k])
            ? racing_clamp(input[k], k == 0 ? -1 : 0, 1) : 0;
}
__global__ static void racing_batch_rays(
    Env *envs, int count, RacingCarConfig config, PfOptixRay *rays) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
#ifdef RACING_MULTI
    if (envs[i].task.done) {
        for (int j = 0; j < 4; j++) rays[i * 4 + j] = {};
        return;
    }
#endif
    PfOptixRay local[12];
    racing_car_rays_device(&envs[i].car, config, local);
    for (int j = 0; j < 4; j++)
        rays[i * 4 + j] = local[j];
#ifndef RACING_MULTI
    for (int j = 0; j < 8; j++)
        rays[count * 4 + i * 8 + j] = local[j + 4];
#endif
}
__global__ static void racing_batch_physics(Env *envs, int count, RacingCarConfig config,
    const PfOptixRay *rays, const PfOptixHit *hits, const RacingGate *gates, int gate_count,
    const RacingRoutePoint *route, int route_count, float length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
#ifdef RACING_MULTI
    if (envs[i].task.done) return;
#endif
    PfOptixRay local_rays[12] = {};
    PfOptixHit local_hits[12] = {};
#ifdef RACING_MULTI
    const int queries = 4; // Static contacts use finite triangles, not the old crash probes.
#else
    const int queries = 12;
#endif
    for (int j = 0; j < queries; j++) {
        int index = j < 4 ? i * 4 + j : count * 4 + i * 8 + j - 4;
        local_rays[j] = rays[index];
        local_hits[j] = hits[index];
    }
#ifdef RACING_MULTI
    envs[i].race_before = envs[i].car.body.position;
    envs[i].race_rotation = envs[i].car.body.rotation;
    racing_car_integrate_device(&envs[i].car, config, local_rays, local_hits, gates, gate_count,
        &envs[i].task, route, route_count, length, false);
#else
    racing_car_integrate_device(&envs[i].car, config, local_rays, local_hits, gates, gate_count,
        &envs[i].task, route, route_count, length);
#endif
}
__global__ static void racing_batch_finish(Env *envs, int count, RacingCar initial,
    RacingTask initial_task, float *rewards, float *terminals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    Env *env = &envs[i];
    RacingTask *t = &env->task;
    rewards[i] = t->pending;
    t->reward = t->pending;
    t->total_reward += t->pending;
    t->pending = 0;
    terminals[i] = t->done != 0;
    if (!t->done)
        return;
    RacingEpisodeEnd *end = &env->last_end;
    end->serial++;
    end->reason = t->done;
    end->invalid = env->car.trial.invalid;
    end->route_jump = t->route_jump;
    end->next_gate = env->car.trial.next;
    end->checkpoints = env->car.trial.checkpoints;
    end->offroad_wheels = t->offroad_wheels;
    end->seconds = t->ticks / 240.0f;
    end->speed = pf_length(env->car.body.linear_velocity);
    end->progress = t->furthest;
    end->position[0] = env->car.body.position.x;
    end->position[1] = env->car.body.position.y;
    end->position[2] = env->car.body.position.z;
    end->throttle = t->action[1]; end->brake = t->action[2];
    end->route_delta = t->route_delta; end->motion = t->motion;
    env->log.episode_return += t->total_reward;
    env->log.progress += t->furthest;
    env->log.checkpoints += env->car.trial.checkpoints;
    env->log.laps += env->car.trial.laps;
    env->log.crashes += t->done == 1;
    env->log.offroad += t->done == 2;
    env->log.invalid += t->done == 3;
    env->log.wrongway += t->done == 4;
    env->log.stalled += t->done == 5;
    env->log.timeout += t->done == 6;
    env->log.n++;
    env->car = initial;
    *t = initial_task;
}
__global__ static void racing_batch_sensor(Env *envs, RacingCarConfig config, PfOptixRay *rays
#ifdef RACING_MULTI
    , PfOptixRay *ground_rays
#endif
) {
    int i = blockIdx.x;
#ifdef RACING_MULTI
    if (envs[i].task.done) {
        rays[i * 256 + threadIdx.x] = {};
        if (threadIdx.x < RACING_GROUND_PROBES)
            ground_rays[i * RACING_GROUND_PROBES + threadIdx.x] = {};
        return;
    }
#endif
    racing_car_sensor_ray(&envs[i].car, config, rays + i * 256, threadIdx.x);
#ifdef RACING_MULTI
    if (threadIdx.x < RACING_GROUND_PROBES)
        ground_rays[i * RACING_GROUND_PROBES + threadIdx.x] =
            racing_ground_ray(envs[i].car, config, threadIdx.x);
#endif
}
__global__ static void racing_batch_observe(Env *envs, RacingCarConfig config,
    const RacingRoutePoint *route, int route_count, float length, const PfOptixHit *hits,
    int gate_count, float *values, obs_t *observations
#ifdef RACING_MULTI
    , const PfOptixRay *ground_rays, const PfOptixHit *ground_hits, int cars, int races
#endif
) {
    int i = blockIdx.x, k = threadIdx.x;
    float *out = values + i * OBS_SIZE;
#ifdef RACING_MULTI
    if (envs[i].task.done) {
        for (int j = k; j < OBS_SIZE; j += blockDim.x) {
            out[j] = 0; observations[i * OBS_SIZE + j] = __float2bfloat16(0);
        }
        return;
    }
#endif
    racing_task_observe_device(&envs[i].task, &envs[i].car, config, route, route_count, length,
        hits + i * 256, gate_count, out, k);
#ifdef RACING_MULTI
    racing_extra_observe(envs, cars, races, i, k, config, hits + i * 256,
        ground_rays + i * RACING_GROUND_PROBES,
        ground_hits + i * RACING_GROUND_PROBES, out);
#endif
    __syncthreads();
    for (int j = k; j < OBS_SIZE; j += blockDim.x) {
#ifdef RACING_MULTI
        // Inactive slots still pass through dense inference. Supply finite padding,
        // so a retired car's invalid telemetry cannot leak through zero-gradient rows.
        float value = envs[i].task.done ? 0.0f : out[j];
#else
        float value = out[j];
#endif
        observations[i * OBS_SIZE + j] = __float2bfloat16(value);
    }
}
static void racing_batch_observations() {
    auto &b = racing_batch;
    racing_batch_sensor<<<b.count, 256, 0, b.stream>>>(b.envs, car_config, b.sensor_rays
#ifdef RACING_MULTI
        , b.ground_rays
#endif
    );
    pf_optix_cuda(cudaGetLastError());
    pf_cuda_stage(b.stream, "LiDAR ray generation");
    pf_optix_launch(&rt, b.queries + 2, b.count * 256, b.stream);
#ifdef RACING_MULTI
    pf_optix_launch(&rt, b.queries + 3, b.count * RACING_GROUND_PROBES, b.stream);
    pf_cuda_stage(b.stream, "forward ground preview");
    racing_race_lidar<<<b.count, 256, 0, b.stream>>>(b.envs, racing_race.cars, racing_race.races,
        b.sensor_rays, b.sensor_hits);
    pf_cuda_stage(b.stream, "opponent LiDAR");
#endif
    racing_batch_observe<<<b.count, 256, 0, b.stream>>>(b.envs, car_config, task_route,
        task_route_count, task_route_length, b.sensor_hits, trial_gate_count, b.float_obs,
        b.observations
#ifdef RACING_MULTI
        , b.ground_rays, b.ground_hits, racing_race.cars, racing_race.races
#endif
    );
    pf_cuda_stage(b.stream, "observations");
    pf_optix_cuda(cudaGetLastError());
}
void puf_init(Env *env, Dict *kwargs) {
#ifdef RACING_MULTI
    env->num_agents = (int)dict_get(kwargs, "race_cars");
#else
    env->num_agents = 1;
#endif
}
Env *puf_vec_create(
    int count, Dict *kwargs, obs_t *observations, float *actions, float *rewards, float *terminals) {
    assert(count > 0 && count <= 65536);
    assert(!racing_ghosts.count || (count == racing_ghosts.count && count <= RACING_MAX_GHOSTS));
    // Initialize the shared scene/config via the manual bridge; no per-world geometry copies.
    racing_queries_create();
    FILE *file = fopen("ocean/racing/route.bin", "rb");
    assert(file && fseek(file, 8, SEEK_SET) == 0);
    unsigned points;
    assert(fread(&points, 4, 1, file) == 1 && points > 3);
    assert(fseek(file, 16 + (long)(points - 3) * 44 + 4, SEEK_SET) == 0);
    float3 ground, next;
    assert(fread(&ground, 4, 3, file) == 3);
    assert(fseek(file, 16 + (long)(points - 2) * 44 + 4, SEEK_SET) == 0);
    assert(fread(&next, 4, 3, file) == 3);
    fclose(file);
    RacingCarFrame frame;
    racing_car_reset(
        ground.x, ground.y, ground.z, atan2f(next.x - ground.x, next.z - ground.z), &frame);
    auto &b = racing_batch;
    b.count = count;
    b.observations = observations;
    b.actions = actions;
    b.rewards = rewards;
    b.terminals = terminals;
    pf_optix_cuda(cudaMemcpy(&b.initial, car, sizeof(RacingCar), cudaMemcpyDeviceToHost));
    // The manual bridge already computed the task state for this fixed spawn.
    pf_optix_cuda(cudaMemcpy(&b.initial_task, car_task, sizeof(RacingTask), cudaMemcpyDeviceToHost));
    pf_optix_cuda(cudaMalloc(&b.envs, count * sizeof(Env)));
    if (racing_ghosts.count)
        pf_optix_cuda(cudaMalloc(&b.ghost_snapshot, count * sizeof(RacingGhostSnapshot)));
    pf_optix_cuda(cudaMalloc(&b.rays, count * 12 * sizeof(PfOptixRay)));
    pf_optix_cuda(cudaMalloc(&b.hits, count * 12 * sizeof(PfOptixHit)));
    pf_optix_cuda(cudaMalloc(&b.sensor_rays, count * 256 * sizeof(PfOptixRay)));
    pf_optix_cuda(cudaMalloc(&b.sensor_hits, count * 256 * sizeof(PfOptixHit)));
#ifdef RACING_MULTI
    pf_optix_cuda(cudaMalloc(&b.ground_rays, count * RACING_GROUND_PROBES * sizeof(PfOptixRay)));
    pf_optix_cuda(cudaMalloc(&b.ground_hits, count * RACING_GROUND_PROBES * sizeof(PfOptixHit)));
#endif
    pf_optix_cuda(cudaMalloc(&b.float_obs, count * OBS_SIZE * sizeof(float)));
    pf_optix_cuda(cudaMalloc(&b.queries,
#ifdef RACING_MULTI
        4
#else
        3
#endif
        * sizeof(PfOptixQuery)));
    PfOptixQuery queries[] = {
        {meshes[0].handle, meshes[0].vertices, meshes[0].materials, b.rays, b.hits},
        {meshes[1].handle, meshes[1].vertices, meshes[1].materials, b.rays + count * 4,
            b.hits + count * 4},
        {meshes[1].handle, meshes[1].vertices, meshes[1].materials, b.sensor_rays, b.sensor_hits}
#ifdef RACING_MULTI
        , {meshes[0].handle, meshes[0].vertices, meshes[0].materials,
            b.ground_rays, b.ground_hits}
#endif
    };
    pf_optix_cuda(cudaMemcpy(b.queries, queries, sizeof(queries), cudaMemcpyHostToDevice));
#ifdef RACING_MULTI
    racing_race_create(kwargs);
    racing_trace_open(racing_race.cars, racing_race.races);
#endif
    return b.envs;
}
void puf_bind_stream(cudaStream_t stream) {
    racing_batch.stream = stream;
}
void puf_reset(Env *) {
    auto &b = racing_batch;
#ifdef RACING_MULTI
    racing_race_reset<<<(racing_race.races + 31) / 32, 32, 0, b.stream>>>(b.envs, racing_race,
        b.rewards, b.terminals, true);
#else
    racing_batch_reset<<<(b.count + 127) / 128, 128, 0, b.stream>>>(b.envs, b.count, b.initial,
        b.initial_task, b.rewards, b.terminals);
#endif
    pf_optix_cuda(cudaGetLastError());
    racing_batch_observations();
    pf_optix_cuda(cudaStreamSynchronize(b.stream));
}
void puf_step(Env *) {
    auto &b = racing_batch;
    int blocks = (b.count + 127) / 128;
    racing_batch_actions<<<blocks, 128, 0, b.stream>>>(b.envs, b.count, b.actions,
        racing_view.opened && racing_view.manual ? racing_ghosts.selected + 1 : 0,
        racing_view.steering, racing_view.throttle,
        racing_view.brake);
    pf_optix_cuda(cudaGetLastError());
    pf_cuda_stage(b.stream, "policy inference and actions");
    for (int step = 0; step < 8; step++) {
        racing_batch_rays<<<blocks, 128, 0, b.stream>>>(b.envs, b.count, car_config, b.rays);
        pf_optix_cuda(cudaGetLastError());
        pf_cuda_stage(b.stream, "wheel ray generation");
        pf_optix_launch(&rt, b.queries, b.count * 4, b.stream);
#ifndef RACING_MULTI
        pf_optix_launch(&rt, b.queries + 1, b.count * 8, b.stream);
#endif
        // One warp per block exposes more independent worlds at small batch sizes.
        racing_batch_physics<<<(b.count + 31) / 32, 32, 0, b.stream>>>(b.envs, b.count, car_config, b.rays,
            b.hits, trial_gates, trial_gate_count, task_route, task_route_count, task_route_length);
        pf_cuda_stage(b.stream, "vehicle physics");
#ifdef RACING_MULTI
        // Spread independent race solves across SMs.
        racing_race_contacts<<<racing_race.races, 1, 0, b.stream>>>(b.envs, racing_race);
        pf_cuda_stage(b.stream, "car contacts");
        // Resolve static contacts after car pairs, including displacement caused by a shove.
        racing_barrier_solve<<<b.count, 1, 0, b.stream>>>(b.envs, b.count, racing_barriers);
        pf_cuda_stage(b.stream, "barrier contacts");
        racing_race_tasks<<<racing_race.races, racing_race.cars, 0, b.stream>>>(b.envs, racing_race,
            car_config, task_route, task_route_count, task_route_length, trial_gates, trial_gate_count);
        pf_cuda_stage(b.stream, "race progress and laps");
#endif
        pf_optix_cuda(cudaGetLastError());
    }
#ifdef RACING_MULTI
    racing_race_finish<<<(racing_race.races + 31) / 32, 32, 0, b.stream>>>(b.envs, racing_race,
        b.rewards, b.terminals);
#else
    racing_batch_finish<<<blocks, 128, 0, b.stream>>>(b.envs, b.count, b.initial,
        b.initial_task, b.rewards, b.terminals);
#endif
    pf_optix_cuda(cudaGetLastError());
    pf_cuda_stage(b.stream, "finish and reset");
    racing_batch_observations();
#ifdef RACING_MULTI
    racing_trace_step(b.stream, b.envs, racing_race.cars);
#endif
}
__global__ static void racing_ghost_snapshot(Env *envs, int count, RacingCarConfig config,
    RacingGhostSnapshot *out) {
    int i = threadIdx.x;
    if (i >= count) return;
    racing_car_snapshot_device(&envs[i].car, config, &out[i].frame, &envs[i].task);
    out[i].end = envs[i].last_end;
    out[i].checkpoints = envs[i].car.trial.checkpoints;
#ifdef RACING_MULTI
    out[i].rank = envs[i].race_rank;
    out[i].frame.progress = (float)envs[i].race_progress;
#endif
}
void puf_render(Env *) {
    auto &b = racing_batch;
    if (!racing_view.opened)
        racing_eval_open();
    racing_eval_input();
    RacingEpisodeEnd last_end = {};
    RacingCarFrame *frame = host_car_frame;
    static RacingGhostSnapshot snapshots[RACING_MAX_GHOSTS];
    if (racing_ghosts.count) {
        racing_ghost_snapshot<<<1, 32, 0, b.stream>>>(b.envs, b.count, car_config, b.ghost_snapshot);
        pf_optix_cuda(cudaGetLastError());
        pf_optix_cuda(cudaMemcpyAsync(snapshots, b.ghost_snapshot,
            b.count * sizeof(RacingGhostSnapshot), cudaMemcpyDeviceToHost, b.stream));
    } else {
        racing_car_snapshot<<<1, 1, 0, b.stream>>>(
            &b.envs[0].car, car_config, car_frame, &b.envs[0].task);
        pf_optix_cuda(cudaGetLastError());
        pf_optix_cuda(cudaMemcpyAsync(
            host_car_frame, car_frame, sizeof(*car_frame), cudaMemcpyDeviceToHost, b.stream));
        pf_optix_cuda(cudaMemcpyAsync(&last_end, &b.envs[0].last_end, sizeof(last_end),
            cudaMemcpyDeviceToHost, b.stream));
    }
    static PfOptixRay visible_rays[256];
    static PfOptixHit visible_hits[256];
    pf_optix_cuda(cudaStreamSynchronize(b.stream));
    if (racing_ghosts.count) {
        auto &g = racing_ghosts;
        for (int i = 0; i < g.count; i++) {
            g.frames[i] = snapshots[i].frame;
            g.ends[i] = snapshots[i].end;
            g.checkpoints[i] = snapshots[i].checkpoints;
#ifdef RACING_MULTI
            g.rank[i] = snapshots[i].rank;
#endif
            g.best_checkpoints[i] = (int)fmaxf(g.best_checkpoints[i], fmaxf(g.checkpoints[i], g.ends[i].checkpoints));
            g.best_progress[i] = fmaxf(g.best_progress[i], fmaxf(g.frames[i].progress, g.ends[i].progress));
            double lap = g.frames[i].best_lap;
            if (lap > 0 && (g.best_lap[i] == 0 || lap < g.best_lap[i])) g.best_lap[i] = lap;
        }
        racing_eval_select();
        frame = g.selected>=0 ? &g.frames[g.selected] : NULL;
        if (g.selected>=0) last_end=g.ends[g.selected];
    }
    if (!frame || frame->task_done || frame->crashed) racing_view.manual=0;
    if (racing_view.lidar && frame && !frame->task_done && !frame->crashed) {
        int selected=racing_ghosts.count ? racing_ghosts.selected : 0;
        pf_optix_cuda(cudaMemcpyAsync(visible_rays,b.sensor_rays+selected*256,
            sizeof(visible_rays),cudaMemcpyDeviceToHost,b.stream));
        pf_optix_cuda(cudaMemcpyAsync(visible_hits,b.sensor_hits+selected*256,
            sizeof(visible_hits),cudaMemcpyDeviceToHost,b.stream));
        pf_optix_cuda(cudaStreamSynchronize(b.stream));
    }
    if (!racing_ghosts.count && last_end.serial != racing_view.last_serial) {
        racing_view.last_serial = last_end.serial;
        printf("Racing reset: %s; t=%.2fs speed=%.2fkm/h progress=%.2fm checkpoints=%d next=%d "
            "position=(%.2f,%.2f,%.2f) throttle=%.3f brake=%.3f wheels=0x%x "
            "route_delta=%.6f motion=%.6f\n", racing_reset_reason(&last_end),
            last_end.seconds, last_end.speed * 3.6f, last_end.progress, last_end.checkpoints,
            last_end.next_gate, last_end.position[0], last_end.position[1], last_end.position[2],
            last_end.throttle, last_end.brake, last_end.offroad_wheels,
            last_end.route_delta, last_end.motion);
        fflush(stdout);
    }
    racing_eval_draw(frame, visible_rays, visible_hits, &last_end);
}
void puf_close(Env *) {
    auto &b = racing_batch;
    pf_optix_cuda(cudaStreamSynchronize(b.stream));
    racing_eval_close();
#ifdef RACING_MULTI
    racing_race_close();
    racing_trace_close();
    racing_ghosts.count = 0;
#endif
    cudaFree(b.ghost_snapshot);
    cudaFree(b.envs);
    cudaFree(b.rays);
    cudaFree(b.hits);
    cudaFree(b.sensor_rays);
    cudaFree(b.sensor_hits);
#ifdef RACING_MULTI
    cudaFree(b.ground_rays);
    cudaFree(b.ground_hits);
#endif
    cudaFree(b.float_obs);
    cudaFree(b.queries);
    racing_map_query_close();
    b = {};
}
void puf_log(Log *log, Dict *out) {
#ifdef RACING_MULTI
    dict_set(out, "score", log->wins);
#else
    dict_set(out, "score", log->progress);
#endif
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "progress", log->progress);
    dict_set(out, "checkpoints", log->checkpoints);
#ifdef RACING_MULTI
    dict_set(out, "mean_speed_mps", log->mean_speed);
    dict_set(out, "progress_reward", log->progress_reward);
    dict_set(out, "mean_throttle", log->mean_throttle);
    dict_set(out, "mean_brake", log->mean_brake);
    dict_set(out, "time_penalty", log->time_penalty);
    dict_set(out, "delay_penalty", log->delay_penalty);
#endif
    dict_set(out, "laps", log->laps);
    dict_set(out, "crashes", log->crashes);
    dict_set(out, "offroad", log->offroad);
    dict_set(out, "invalid", log->invalid);
    dict_set(out, "wrongway", log->wrongway);
    dict_set(out, "stalled", log->stalled);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "n", log->n);
#ifdef RACING_MULTI
    dict_set(out, "position", log->position);
    dict_set(out, "wins", log->wins);
    dict_set(out, "contacts", log->contacts);
    dict_set(out, "wall_contacts", log->wall_contacts);
    dict_set(out, "contact_overflow", log->contact_overflow);
    dict_set(out, "car_impacts", log->car_impacts);
    dict_set(out, "wall_impacts", log->wall_impacts);
    dict_set(out, "wall_impact_penalty", log->wall_impact_penalty);
    dict_set(out, "wall_contact_penalty", log->wall_contact_penalty);
    dict_set(out, "wall_contact_seconds", log->wall_contact_seconds);
    dict_set(out, "car_contact_penalty", log->car_contact_penalty);
    dict_set(out, "car_contact_seconds", log->car_contact_seconds);
    dict_set(out, "offroad_seconds", log->offroad_seconds);
    dict_set(out, "wrongway_seconds", log->wrongway_seconds);
    dict_set(out, "route_jumps", log->route_jumps);
    dict_set(out, "checkpoint_bonus", log->checkpoint_bonus);
    dict_set(out, "control_saturation", log->control_saturation);
    dict_set(out, "lap_seconds", log->laps > 0 ? log->lap_seconds / log->laps : 0);
    dict_set(out, "record_beats", log->record_beats);
    dict_set(out, "lap_bonus", log->lap_bonus);
#endif
}
#endif
