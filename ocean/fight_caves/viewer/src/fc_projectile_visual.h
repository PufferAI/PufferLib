#ifndef FC_PROJECTILE_VISUAL_H
#define FC_PROJECTILE_VISUAL_H

typedef struct {
    float source_x;
    float source_y;
    float source_z;
    float target_x;
    float target_y;
    float target_z;
    float duration;
    float angle;
    float progress;
} FcProjectilePath;

typedef struct {
    float x;
    float y;
    float z;
    float velocity_x;
    float velocity_y;
    float velocity_z;
} FcProjectileSample;

typedef struct {
    float launch_delay;
    float flight_duration;
    float total_duration;
} FcProjectileTiming;

float fc_projectile_profile_end_cycle(float launch_cycle,
                                      float length_adjustment,
                                      float step_multiplier,
                                      int tile_distance);

/* Convert the client's 30-cycle-per-game-tick projectile profile to viewer
 * seconds. The extra client cycle in flight_duration matches the client
 * endpoint convention used by RuneC's projectile sampler. */
int fc_projectile_timing_from_client_cycles(float launch_cycle,
                                            float end_cycle,
                                            float ticks_per_second,
                                            FcProjectileTiming* timing);

/* Effects are retained only until either their animation ends or the client
 * retention window closes. This prevents short spot animations from looping. */
float fc_projectile_effect_duration_seconds(float animation_client_cycles,
                                            float retain_client_cycles,
                                            float ticks_per_second);

/* Sample the client projectile curve at an absolute point in its flight.
 * The target may be replaced on every render frame to reproduce the client's
 * actor-targeted homing behavior without frame-rate-dependent integration. */
int fc_projectile_path_sample(const FcProjectilePath* path,
                              float elapsed,
                              FcProjectileSample* sample);

#endif /* FC_PROJECTILE_VISUAL_H */
