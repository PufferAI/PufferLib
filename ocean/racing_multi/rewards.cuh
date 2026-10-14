#pragma once

// After PPO scaling, each new clean metre earns 0.0005 reward.
static constexpr float RACING_PROGRESS_METRES_PER_REWARD = 200.0f;
// Per second of actual car or wall contact, independent of impulse, patch count and cooldown.
static constexpr float RACING_CONTACT_COST_PER_SECOND = 0.1f;
static constexpr float RACING_DNF_REWARD = -7.0f;
static constexpr float RACING_POSITION_REWARD = 3.0f;
static constexpr float RACING_LAP_REWARD = 3.5f;
static constexpr float RACING_LAP_SPEED_REWARD = 2.5f;
// Leave headroom for progress, position changes and contact costs on terminal steps.
static constexpr float RACING_REWARD_SCALE = 0.1f;

__host__ __device__ static float racing_progress_reward(const RacingTask &task,
    float previous_furthest, float limit, bool contact = false) {
    if (task.offroad || task.route_jump || contact || (task.done && task.done != 7)) return 0;
    // Furthest includes off-road travel: returning cannot claim it again.
    return racing_clamp((fminf(task.furthest, limit) - fminf(previous_furthest, limit)) /
        RACING_PROGRESS_METRES_PER_REWARD, 0, 0.02f);
}

__host__ __device__ static float racing_lap_bonus(float finish_seconds, float limit) {
    return RACING_LAP_REWARD + RACING_LAP_SPEED_REWARD
        * racing_clamp(1 - finish_seconds / limit, 0, 1);
}

__host__ __device__ static float racing_reward_for_ppo(float reward) {
    return reward * RACING_REWARD_SCALE;
}

// Charge the discount-erased part of a DNF on every active policy action.
// With terminal DNF = D*gamma, c + gamma*D = D, where
// c = D*(1-gamma). Thus merely postponing failure cannot improve its value.
// This is a time cost, never a reward for speed without actual route progress.
__host__ __device__ static float racing_failure_delay_cost(float discount) {
    return -RACING_DNF_REWARD * (1 - discount);
}
__host__ __device__ static float racing_failure_outcome(float discount) {
    return RACING_DNF_REWARD * discount;
}

// Positive place gains require a clean transition. Always preserve losses, and
// consume the standings baseline even when a gain is suppressed.
__host__ __device__ static float racing_position_change_reward(float change, bool clean) {
    return 0.25f * (clean ? change : fminf(change, 0.0f));
}
// Once-only gates provide earlier feedback. Absolute race time rewards pace,
// including the run-up, so waiting before the start cannot improve the bonus.
__host__ __device__ static float racing_checkpoint_reward(int crossed, float seconds,
    float limit, bool clean) {
    return clean && crossed > 0 ? 0.025f * racing_clamp(1 - seconds / limit, 0.25f, 1.0f) : 0;
}
// PPO stores the latent Gaussian action. Only dispatched controls are transformed;
// log probabilities remain those of the latent policy, with no clipped-action mismatch.
__host__ __device__ static void racing_policy_controls(const float *raw, float *out) {
    out[0] = tanhf(raw[0]);
    // Neutral Gaussian exploration should mostly coast/accelerate. An unbiased
    // brake head fires half the time and overpowers the rate-limited throttle.
    // Braking remains reachable through the same trainable latent action.
    float drive = fmaxf(0, tanhf(raw[1])) - fmaxf(0, tanhf(raw[2] - 1.5f));
    out[1] = fmaxf(0, drive);
    out[2] = fmaxf(0, -drive);
}
