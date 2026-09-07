#include "fc_projectile_visual.h"

#include <math.h>

float fc_projectile_profile_end_cycle(float launch_cycle,
                                      float length_adjustment,
                                      float step_multiplier,
                                      int tile_distance) {
    if (tile_distance < 0) tile_distance = 0;
    float end_cycle = launch_cycle + length_adjustment +
                      step_multiplier * (float)tile_distance;
    return end_cycle > launch_cycle ? end_cycle : launch_cycle + 1.0f;
}

int fc_projectile_timing_from_client_cycles(float launch_cycle,
                                            float end_cycle,
                                            float ticks_per_second,
                                            FcProjectileTiming* timing) {
    if (!timing || ticks_per_second <= 0.0f || launch_cycle < 0.0f ||
        end_cycle <= launch_cycle)
        return 0;

    float seconds_per_client_cycle = 1.0f / (30.0f * ticks_per_second);
    timing->launch_delay = launch_cycle * seconds_per_client_cycle;
    timing->flight_duration =
        (end_cycle + 1.0f - launch_cycle) * seconds_per_client_cycle;
    timing->total_duration = timing->launch_delay + timing->flight_duration;
    return 1;
}

float fc_projectile_effect_duration_seconds(float animation_client_cycles,
                                            float retain_client_cycles,
                                            float ticks_per_second) {
    if (ticks_per_second <= 0.0f || retain_client_cycles <= 0.0f)
        return 0.0f;
    if (animation_client_cycles <= 0.0f)
        animation_client_cycles = 30.0f;
    float visible_cycles = animation_client_cycles < retain_client_cycles
        ? animation_client_cycles : retain_client_cycles;
    return visible_cycles / (30.0f * ticks_per_second);
}

int fc_projectile_path_sample(const FcProjectilePath* path,
                              float elapsed,
                              FcProjectileSample* sample) {
    if (!path || !sample || path->duration <= 0.0f)
        return 0;

    float dx = path->target_x - path->source_x;
    float dz = path->target_z - path->source_z;
    float horizontal = sqrtf(dx * dx + dz * dz);
    float direction_x = 0.0f;
    float direction_z = 1.0f;
    if (horizontal > 0.00001f) {
        direction_x = dx / horizontal;
        direction_z = dz / horizontal;
    }

    float source_x = path->source_x + direction_x * path->progress;
    float source_z = path->source_z + direction_z * path->progress;
    float velocity_x = (path->target_x - source_x) / path->duration;
    float velocity_z = (path->target_z - source_z) / path->duration;
    float horizontal_speed = sqrtf(
        velocity_x * velocity_x + velocity_z * velocity_z);
    float velocity_y = horizontal_speed *
        tanf(path->angle * (3.14159265358979323846f / 128.0f));
    float acceleration_y = 2.0f *
        (path->target_y - path->source_y - velocity_y * path->duration) /
        (path->duration * path->duration);

    float t = elapsed;
    if (t < 0.0f) t = 0.0f;
    if (t > path->duration) t = path->duration;
    sample->x = source_x + velocity_x * t;
    sample->y = path->source_y + velocity_y * t +
                0.5f * acceleration_y * t * t;
    sample->z = source_z + velocity_z * t;
    sample->velocity_x = velocity_x;
    sample->velocity_y = velocity_y + acceleration_y * t;
    sample->velocity_z = velocity_z;
    return 1;
}
