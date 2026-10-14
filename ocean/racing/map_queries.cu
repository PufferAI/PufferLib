// Optional viewer bridge. Production callers use puffysics device ray/hit buffers directly.
#include "../../src/puffysics/raycast_optix.cuh"
#include <string.h>
#include "car_physics.cuh"

static PfOptix rt;
static PfOptixMesh meshes[2];
static PfOptixRay *rays;
static PfOptixHit *hits;
static PfOptixQuery *params;
static float *segments;
static float *host_segments;
static cudaStream_t stream;
static bool ready;

__global__ static void map_generate_rays(PfOptixRay *rays, float3 point, float3 forward) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 260) {
        return;
    }
    float3 left = make_float3(forward.z, 0, -forward.x);
    if (i < 256) {
        float azimuth = ((i % 128) / 127.0f - 0.5f) * 4.7123889804f;
        float elevation = i < 128 ? -0.0523598776f : 0.0174532925f;
        float c = cosf(azimuth) * cosf(elevation), s = sinf(azimuth) * cosf(elevation);
        rays[i] = {make_float3(point.x, point.y + 1.5f, point.z), 0.01f,
            make_float3(forward.x * c + left.x * s, sinf(elevation), forward.z * c + left.z * s),
            150};
    } else {
        float x = (i & 1) ? 1 : -1, z = (i & 2) ? 1.5f : -1.5f;
        rays[i] = {make_float3(point.x + left.x * x + forward.x * z, point.y + 2,
                       point.z + left.z * x + forward.z * z),
            0.01f, make_float3(0, -1, 0), 5};
    }
}
__global__ static void map_query_segments(
    const PfOptixRay *rays, const PfOptixHit *hits, float *segments) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 260) {
        return;
    }
    PfOptixRay ray = rays[i];
    PfOptixHit hit = hits[i];
    float *segment = segments + i * 7;
    segment[0] = ray.origin.x;
    segment[1] = ray.origin.y;
    segment[2] = ray.origin.z;
    segment[3] = ray.origin.x + ray.direction.x * hit.distance;
    segment[4] = ray.origin.y + ray.direction.y * hit.distance;
    segment[5] = ray.origin.z + ray.direction.z * hit.distance;
    segment[6] = hit.triangle >= 0 ? (float)hit.material : (float)hit.triangle;
}

static void racing_queries_create() {
    if (!ready) {
        pf_optix_create(&rt, "ocean/racing/raycast_optix.ptx");
        pf_optix_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        FILE *file = fopen("ocean/racing/query_mesh.bin", "rb");
        assert(file);
        char magic[8];
        assert(fread(magic, 1, 8, file) == 8 && memcmp(magic, "PFQUERY1", 8) == 0);
        for (int scene = 0; scene < 2; scene++) {
            unsigned count;
            assert(fread(&count, 4, 1, file) == 1 && count > 0 && count < 10000000);
            float3 *vertices = (float3 *)malloc((size_t)count * 3 * sizeof(float3));
            unsigned *materials = (unsigned *)malloc((size_t)count * sizeof(unsigned));
            assert(vertices && materials);
            for (unsigned i = 0; i < count; i++) {
                assert(fread(vertices + i * 3, sizeof(float3), 3, file) == 3);
                assert(fread(materials + i, sizeof(unsigned), 1, file) == 1);
            }
            pf_optix_build(&rt, &meshes[scene], vertices, materials, count, stream);
            printf("%s: %u triangles, compact BVH %zu bytes\n", scene ? "LiDAR" : "Support", count,
                meshes[scene].accel_bytes);
            free(vertices);
            free(materials);
        }
        fclose(file);
        pf_optix_cuda(cudaMalloc(&rays, 260 * sizeof(PfOptixRay)));
        pf_optix_cuda(cudaMalloc(&hits, 260 * sizeof(PfOptixHit)));
        pf_optix_cuda(cudaMalloc(&params, 2 * sizeof(PfOptixQuery)));
        pf_optix_cuda(cudaMalloc(&segments, 260 * 7 * sizeof(float)));
        pf_optix_cuda(cudaMallocHost(&host_segments, 260 * 7 * sizeof(float)));
        PfOptixQuery query[2] = {
            {meshes[0].handle, meshes[0].vertices, meshes[0].materials, rays + 256, hits + 256},
            {meshes[1].handle, meshes[1].vertices, meshes[1].materials, rays, hits}};
        pf_optix_cuda(
            cudaMemcpyAsync(params, query, sizeof(query), cudaMemcpyHostToDevice, stream));
        pf_optix_cuda(cudaStreamSynchronize(stream));
        ready = true;
    }
}

extern "C" void racing_map_query(float x, float y, float z, float dx, float dz, float *result) {
    racing_queries_create();
    float length = sqrtf(dx * dx + dz * dz);
    assert(length > 0);
    map_generate_rays<<<2, 256, 0, stream>>>(
        rays, make_float3(x, y, z), make_float3(dx / length, 0, dz / length));
    pf_optix_cuda(cudaGetLastError());
    pf_optix_launch(&rt, params, 4, stream);
    pf_optix_launch(&rt, params + 1, 256, stream);
    map_query_segments<<<2, 256, 0, stream>>>(rays, hits, segments);
    pf_optix_cuda(cudaGetLastError());
    pf_optix_cuda(cudaMemcpyAsync(
        host_segments, segments, 260 * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    pf_optix_cuda(cudaStreamSynchronize(stream));
    memcpy(result, host_segments, 260 * 7 * sizeof(float));
}
extern "C" void racing_map_query_close() {
    if (!ready) {
        return;
    }
    pf_optix_cuda(cudaStreamSynchronize(stream));
    racing_car_close();
    pf_optix_cuda(cudaFree(rays));
    pf_optix_cuda(cudaFree(hits));
    pf_optix_cuda(cudaFree(params));
    pf_optix_cuda(cudaFree(segments));
    pf_optix_cuda(cudaFreeHost(host_segments));
    for (int i = 0; i < 2; i++) {
        pf_optix_mesh_destroy(&meshes[i]);
    }
    pf_optix_destroy(&rt);
    pf_optix_cuda(cudaStreamDestroy(stream));
    ready = false;
}

static RacingCar *car;
static RacingTask *car_task;
static RacingRoutePoint *task_route;
static int task_route_count;
static float task_route_length;
static float *task_observations;
static RacingGate *trial_gates;
static unsigned trial_gate_count;
static RacingCarConfig car_config;
static PfOptixRay *car_rays;
static PfOptixHit *car_hits;
static PfOptixQuery *car_params;
static RacingCarFrame *car_frame, *host_car_frame;
static int car_sensor_steps;

static void racing_car_sensor_submit() {
    racing_car_sensor_rays<<<2, 256, 0, stream>>>(car, car_config, rays);
    pf_optix_cuda(cudaGetLastError());
    pf_optix_launch(&rt, params, 4, stream);
    pf_optix_launch(&rt, params + 1, 256, stream);
    map_query_segments<<<2, 256, 0, stream>>>(rays, hits, segments);
    pf_optix_cuda(cudaGetLastError());
}

extern "C" void racing_car_lidar(float *lidar_lines) {
    assert(car && lidar_lines);
    racing_car_sensor_submit();
    pf_optix_cuda(cudaMemcpyAsync(
        host_segments, segments, 260 * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    pf_optix_cuda(cudaStreamSynchronize(stream));
    memcpy(lidar_lines, host_segments, 260 * 7 * sizeof(float));
}

extern "C" void racing_car_reset(float x, float y, float z, float heading, RacingCarFrame *frame) {
    racing_queries_create();
    if (!car) {
        pf_optix_cuda(cudaMalloc(&car_task, sizeof(*car_task)));
        pf_optix_cuda(cudaMalloc(&task_observations, RACING_OBS_SIZE * sizeof(float)));
        FILE *route_file = fopen("ocean/racing/route.bin", "rb");
        assert(route_file);
        char route_magic[8];
        unsigned landmarks;
        assert(
            fread(route_magic, 1, 8, route_file) == 8 && memcmp(route_magic, "PFROUTE1", 8) == 0);
        assert(fread(&task_route_count, 4, 1, route_file) == 1 && task_route_count > 3 &&
            task_route_count <= 4096);
        assert(fread(&landmarks, 4, 1, route_file) == 1);
        RacingRoutePoint host_route[4096];
        for (int i = 0; i < task_route_count; i++) {
            unsigned flags;
            assert(fread(&host_route[i].s, 4, 1, route_file) == 1);
            assert(fread(&host_route[i].center, 4, 3, route_file) == 3);
            assert(fread(&host_route[i].left, 4, 3, route_file) == 3);
            assert(fread(&host_route[i].right, 4, 3, route_file) == 3);
            assert(fread(&flags, 4, 1, route_file) == 1);
        }
        fclose(route_file);
        task_route_length = host_route[task_route_count - 1].s +
            pf_length(pf_sub(host_route[0].center, host_route[task_route_count - 1].center));
        pf_optix_cuda(cudaMalloc(&task_route, task_route_count * sizeof(RacingRoutePoint)));
        pf_optix_cuda(cudaMemcpyAsync(task_route, host_route,
            task_route_count * sizeof(RacingRoutePoint), cudaMemcpyHostToDevice, stream));
        pf_optix_cuda(cudaStreamSynchronize(stream));
        pf_optix_cuda(cudaMalloc(&car, sizeof(*car)));
        pf_optix_cuda(cudaMemsetAsync(car, 0, sizeof(*car), stream));
        FILE *track = fopen("ocean/racing/track.bin", "rb");
        assert(track);
        char magic[8];
        unsigned sections;
        assert(fread(magic, 1, 8, track) == 8 && memcmp(magic, "PFTRACK2", 8) == 0);
        assert(fread(&sections, 4, 1, track) == 1 && sections > 0 && sections <= 4096);
        assert(fread(&trial_gate_count, 4, 1, track) == 1 && trial_gate_count > 1 &&
            trial_gate_count <= 128);
        RacingGate host_gates[128];
        assert(fseek(track, (long)sections * 6 * sizeof(float), SEEK_CUR) == 0);
        for (unsigned i = 0; i < trial_gate_count; i++) {
            unsigned index;
            assert(fread(&index, 4, 1, track) == 1 && index < sections);
            assert(index < (unsigned)task_route_count);
            host_gates[i].s = host_route[index].s;
            assert(fread(&host_gates[i].forward, 4, 3, track) == 3);
            assert(fread(&host_gates[i].left, 4, 3, track) == 3);
            assert(fread(&host_gates[i].right, 4, 3, track) == 3);
            PfVec3 width = pf_sub(host_gates[i].right, host_gates[i].left);
            assert(width.x * width.x + width.z * width.z > 0);
        }
        fclose(track);
        pf_optix_cuda(cudaMalloc(&trial_gates, trial_gate_count * sizeof(RacingGate)));
        pf_optix_cuda(cudaMemcpyAsync(trial_gates, host_gates,
            trial_gate_count * sizeof(RacingGate), cudaMemcpyHostToDevice, stream));
        pf_optix_cuda(cudaStreamSynchronize(stream));
        pf_optix_cuda(cudaMalloc(&car_rays, 12 * sizeof(PfOptixRay)));
        pf_optix_cuda(cudaMalloc(&car_hits, 12 * sizeof(PfOptixHit)));
        pf_optix_cuda(cudaMalloc(&car_params, 2 * sizeof(PfOptixQuery)));
        pf_optix_cuda(cudaMalloc(&car_frame, sizeof(*car_frame)));
        pf_optix_cuda(cudaMallocHost(&host_car_frame, sizeof(*host_car_frame)));
        PfOptixQuery queries[2] = {
            {meshes[0].handle, meshes[0].vertices, meshes[0].materials, car_rays, car_hits},
            {meshes[1].handle, meshes[1].vertices, meshes[1].materials, car_rays + 4,
                car_hits + 4}};
        pf_optix_cuda(
            cudaMemcpyAsync(car_params, queries, sizeof(queries), cudaMemcpyHostToDevice, stream));
        pf_optix_cuda(cudaStreamSynchronize(stream));
    }
    struct Field {
        const char *name;
        size_t offset;
    };
    Field fields[] = {
        {"mass", offsetof(RacingCarConfig, mass)},
        {"power", offsetof(RacingCarConfig, power)},
        {"torque", offsetof(RacingCarConfig, torque)},
        {"wheelbase", offsetof(RacingCarConfig, wheelbase)},
        {"track_width", offsetof(RacingCarConfig, track_width)},
        {"radius", offsetof(RacingCarConfig, radius)},
        {"spring", offsetof(RacingCarConfig, spring)},
        {"damper", offsetof(RacingCarConfig, damper)},
        {"rest_length", offsetof(RacingCarConfig, rest_length)},
        {"travel", offsetof(RacingCarConfig, travel)},
        {"wheel_inertia", offsetof(RacingCarConfig, wheel_inertia)},
        {"steer_limit", offsetof(RacingCarConfig, steer_limit)},
        {"steer_rate", offsetof(RacingCarConfig, steer_rate)},
        {"steer_speed_scale", offsetof(RacingCarConfig, steer_speed_scale)},
        {"throttle_rate", offsetof(RacingCarConfig, throttle_rate)},
        {"final_drive", offsetof(RacingCarConfig, final_drive)},
        {"efficiency", offsetof(RacingCarConfig, efficiency)},
        {"brake_torque", offsetof(RacingCarConfig, brake_torque)},
        {"longitudinal_stiffness", offsetof(RacingCarConfig, longitudinal_stiffness)},
        {"lateral_stiffness", offsetof(RacingCarConfig, lateral_stiffness)},
        {"drag_area", offsetof(RacingCarConfig, drag_area)},
        {"lidar_mount_y", offsetof(RacingCarConfig, lidar_mount_y)},
        {"lidar_range", offsetof(RacingCarConfig, lidar_range)},
        {"tarmac_grip", offsetof(RacingCarConfig, tarmac_grip)},
        {"curb_grip", offsetof(RacingCarConfig, curb_grip)},
        {"grass_grip", offsetof(RacingCarConfig, grass_grip)},
        {"gravel_grip", offsetof(RacingCarConfig, gravel_grip)},
        {"tarmac_rolling", offsetof(RacingCarConfig, tarmac_rolling)},
        {"curb_rolling", offsetof(RacingCarConfig, curb_rolling)},
        {"grass_rolling", offsetof(RacingCarConfig, grass_rolling)},
        {"gravel_rolling", offsetof(RacingCarConfig, gravel_rolling)},
    };
    FILE *file = fopen("ocean/racing/car.cfg", "r");
    assert(file);
    bool seen[sizeof(fields) / sizeof(fields[0])] = {};
    char line[256], name[80];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        float value;
        if (sscanf(line, "%79s %f", name, &value) != 2 || !isfinite(value) || value <= 0) {
            fprintf(stderr, "Invalid car.cfg line: %s", line);
            exit(1);
        }
        bool found = false;
        for (unsigned i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
            if (strcmp(name, fields[i].name) == 0) {
                memcpy((char *)&car_config + fields[i].offset, &value, sizeof(value));
                assert(!seen[i]);
                seen[i] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "Unknown car.cfg setting: %s\n", name);
            exit(1);
        }
    }
    fclose(file);
    for (unsigned i = 0; i < sizeof(fields) / sizeof(fields[0]); i++)
        assert(seen[i]);
    assert(car_config.efficiency <= 1 && car_config.steer_limit < 1.2f);
    assert(car_config.mass * 9.81f / (4 * car_config.spring) < car_config.travel);
    assert(car_config.travel < car_config.rest_length);
    racing_reset_kernel<<<1, 1, 0, stream>>>(car, car_config, make_float3(x, y, z), heading);
    pf_optix_cuda(cudaGetLastError());
    racing_task_reset<<<1, 1, 0, stream>>>(
        car_task, car, task_route, task_route_count, task_route_length);
    pf_optix_cuda(cudaGetLastError());
    car_sensor_steps = 0;
    racing_car_step(0, 0, 0, 0, frame, NULL);
}
extern "C" void racing_car_step(int steps, float throttle, float brake, float steering,
    RacingCarFrame *frame, float *lidar_lines) {
    assert(car && steps >= 0 && steps <= 16);
    throttle = fminf(1, fmaxf(0, throttle));
    brake = fminf(1, fmaxf(0, brake));
    steering = fminf(1, fmaxf(-1, steering));
    bool sensor_updated = false;
    for (int i = 0; i < steps; i++) {
        if (car_sensor_steps == 0) {
            racing_task_action<<<1, 1, 0, stream>>>(car_task, NULL, steering, throttle, brake);
            pf_optix_cuda(cudaGetLastError());
        }
        racing_car_rays<<<1, 1, 0, stream>>>(car, car_config, car_rays);
        pf_optix_cuda(cudaGetLastError());
        pf_optix_launch(&rt, car_params, 4, stream);
        pf_optix_launch(&rt, car_params + 1, 8, stream);
        racing_car_integrate<<<1, 1, 0, stream>>>(car, car_config, car_rays, car_hits, trial_gates,
            trial_gate_count, car_task, task_route, task_route_count, task_route_length);
        pf_optix_cuda(cudaGetLastError());
        if (++car_sensor_steps == 8) {
            car_sensor_steps = 0;
            racing_car_sensor_submit();
            racing_task_observe<<<1, 256, 0, stream>>>(car_task, car, car_config, task_route,
                task_route_count, task_route_length, hits, trial_gate_count, task_observations);
            racing_task_publish<<<1, 1, 0, stream>>>(car_task, NULL, NULL);
            pf_optix_cuda(cudaGetLastError());
            sensor_updated = true;
        }
    }
    if (sensor_updated && lidar_lines) {
        pf_optix_cuda(cudaMemcpyAsync(
            host_segments, segments, 260 * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    racing_car_snapshot<<<1, 1, 0, stream>>>(car, car_config, car_frame, car_task);
    pf_optix_cuda(cudaGetLastError());
    pf_optix_cuda(cudaMemcpyAsync(
        host_car_frame, car_frame, sizeof(*car_frame), cudaMemcpyDeviceToHost, stream));
    pf_optix_cuda(cudaStreamSynchronize(stream));
    *frame = *host_car_frame;
    if (sensor_updated && lidar_lines)
        memcpy(lidar_lines, host_segments, 260 * 7 * sizeof(float));
}
extern "C" void racing_car_close() {
    if (!car)
        return;
    pf_optix_cuda(cudaStreamSynchronize(stream));
    pf_optix_cuda(cudaFree(car));
    pf_optix_cuda(cudaFree(car_task));
    pf_optix_cuda(cudaFree(task_route));
    pf_optix_cuda(cudaFree(task_observations));
    pf_optix_cuda(cudaFree(trial_gates));
    pf_optix_cuda(cudaFree(car_rays));
    pf_optix_cuda(cudaFree(car_hits));
    pf_optix_cuda(cudaFree(car_params));
    pf_optix_cuda(cudaFree(car_frame));
    pf_optix_cuda(cudaFreeHost(host_car_frame));
    car = NULL;
}
