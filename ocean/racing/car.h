#pragma once
// C/CUDA interface: only this small selected-world snapshot is read back for rendering.
typedef struct RacingCarFrame {
    float reward, episode_return, progress;
    int task_done;
    double lap_time, last_lap, best_lap;
    int lap_active, next_gate, lap_invalid, completed_laps;
    float position[3], rotation[4]; // xyzw quaternion
    float wheel_y[4], wheel_spin[4], steer[4];
    float speed, rpm, throttle, brake, steering, wheelbase, track_width, radius, lidar_mount_y, grip[4];
    int gear, crashed, material[4], contacts[4];
} RacingCarFrame;
#ifdef __cplusplus
extern "C" {
#endif
void racing_car_reset(float x, float y, float z, float heading, RacingCarFrame *frame);
// Steering: -1 left, 0 straight, +1 right; throttle/brake: 0 released, 1 fully pressed.
void racing_car_step(int steps, float throttle, float brake, float steering, RacingCarFrame *frame,
    float *lidar_lines);
void racing_car_lidar(float *lidar_lines);
void racing_car_close(void);
#ifdef __cplusplus
}
#endif

// Retained across automatic resets for evaluation diagnostics.
typedef struct RacingEpisodeEnd {
    int serial, reason, invalid, route_jump, next_gate, checkpoints, offroad_wheels;
    float seconds, speed, progress, position[3], throttle, brake, route_delta, motion;
} RacingEpisodeEnd;
