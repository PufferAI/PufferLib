#pragma once

// Optional single-race CSV. Normal training never copies vehicle state to the CPU.
static FILE *racing_trace_file;
static Env racing_trace_envs[RACING_RACE_MAX];
static unsigned long racing_trace_step_count;

static void racing_trace_open(int cars, int races) {
    const char *path = getenv("RACING_TRACE_PATH");
    if (!path || !*path) return;
    if (races != 1 || cars > RACING_RACE_MAX) {
        fprintf(stderr, "RACING_TRACE_PATH requires one race (use racing_multi race --headless)\n");
        exit(1);
    }
    racing_trace_file = fopen(path, "wx");
    if (!racing_trace_file) { perror(path); exit(1); }
    fprintf(racing_trace_file, "step,car,seconds,route_m,speed_mps,steer_action,steer_rad,"
        "throttle_action,throttle,brake,yaw_radps,grounded_wheels,saturated_wheels,"
        "mean_grip,lateral_m,heading_alignment,offroad,crashed,done,wall_impacts\n");
}

static void racing_trace_step(cudaStream_t stream, Env *envs, int cars) {
    if (!racing_trace_file) return;
    pf_optix_cuda(cudaMemcpyAsync(racing_trace_envs, envs, cars * sizeof(Env),
        cudaMemcpyDeviceToHost, stream));
    pf_optix_cuda(cudaStreamSynchronize(stream));
    for (int i = 0; i < cars; i++) {
        const Env &e = racing_trace_envs[i];
        const RacingCar &c = e.car;
        int grounded = 0, saturated = 0;
        float grip = 0;
        for (int wheel = 0; wheel < 4; wheel++) {
            grounded += c.contacts[wheel] != 0;
            saturated += (c.tire_saturation_mask >> wheel) & 1u;
            grip += c.grip[wheel];
        }
        PfVec3 angular = pf_quat_rotate(pf_quat_conjugate(c.body.rotation),
            c.body.angular_velocity);
        fprintf(racing_trace_file,
            "%lu,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%.4f,%d,%d,%.3f,%.3f,%.4f,%d,%d,%d,%d\n",
            racing_trace_step_count, i, e.task.ticks * RACING_DT, e.task.s,
            pf_length(c.body.linear_velocity), e.task.action[0], c.steering,
            e.task.action[1], c.throttle, e.task.action[2], angular.y,
            grounded, saturated, grounded ? grip / grounded : 0,
            e.task.lateral, e.task.heading, e.task.offroad, c.crashed != 0,
            e.task.done, e.wall_impacts);
    }
    racing_trace_step_count++;
}

static void racing_trace_close() {
    if (racing_trace_file) fclose(racing_trace_file);
    racing_trace_file = nullptr;
    racing_trace_step_count = 0;
}
