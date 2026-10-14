#pragma once

// Included by robot_arm.h after the shared state and geometry helpers.

RA_D static RA_INLINE void ra_obs_xyz(float* observation, int* index, RaVec3 value) {
    observation[(*index)++] = value.x;
    observation[(*index)++] = value.y;
    observation[(*index)++] = value.z;
}

RA_D static RA_INLINE void ra_obs3(float* observation, int* index, RaVec3 value, float scale) {
    observation[(*index)++] = ra_clamp(scale * value.x, -1.0f, 1.0f);
    observation[(*index)++] = ra_clamp(scale * value.y, -1.0f, 1.0f);
    observation[(*index)++] = ra_clamp(scale * value.z, -1.0f, 1.0f);
}

RA_D static RA_INLINE float ra_obs_pad(float impulse) {
    return ra_clamp(impulse / (RA_PHYSICS_DT * RA_GRIPPER_MAX_FORCE), 0.0f, 1.0f);
}

RA_D static void ra_observe(const RaState* state, float* observation) {
    int index = 0;
    for (int joint = 0; joint < RA_DOF; ++joint) {
        float low = ra_jmin(joint);
        float high = ra_jmax(joint);
        float midpoint = 0.5f * (low + high);
        float half_range = 0.5f * (high - low);
        observation[index++] = ra_clamp((state->q[joint] - midpoint) / half_range, -1.0f, 1.0f);
    }
    for (int joint = 0; joint < RA_DOF; ++joint) {
        observation[index++] = ra_clamp(state->qd[joint] / 6.0f, -1.0f, 1.0f);
    }
    for (int action = 0; action < RA_ACTIONS; ++action) {
        observation[index++] = ra_clamp(state->previous_action[action], -1.0f, 1.0f);
    }
    RaPose gripper_links[RA_LINKS];
    RaVec3 origins[RA_DOF];
    RaVec3 axes[RA_DOF];
    ra_fk(state->q, state->gripper_width, gripper_links, origins, axes, NULL);
    RaQuat hand_rotation = gripper_links[RA_DOF + 1].rotation;
    RaVec3 reach_origin =
        state->basketball_mode ? ra_gctr(state->end_effector, hand_rotation) : state->end_effector;
    ra_obs3(observation, &index, ra_sub(state->cube_position, reach_origin), RA_OBS_POS_SCALE);
    ra_obs3(observation, &index, ra_sub(state->target_position, state->cube_position),
        RA_OBS_POS_SCALE);
    ra_obs3(observation, &index, state->cube_velocity, RA_OBS_LIN_VEL_SCALE);
    ra_obs3(observation, &index, state->end_effector, RA_OBS_POS_SCALE);
    RaQuat gripper_in_cube = state->basketball_mode
        ? hand_rotation
        : ra_qmul(ra_qconj(state->cube_rotation), hand_rotation);
    ra_obs_xyz(observation, &index, ra_rotate(gripper_in_cube, ra_v3(0, 1, 0)));
    ra_obs_xyz(observation, &index, ra_rotate(gripper_in_cube, ra_v3(0, 0, 1)));
    ra_obs3(observation, &index, state->cube_angular_velocity, RA_OBS_ANG_VEL_SCALE);
    float grip_vel = ra_clamp(RA_OBS_GRIP_VEL_SCALE * state->gripper_velocity, -1.0f, 1.0f);
    if (state->stack_mode) {
        ra_obs_xyz(observation, &index, ra_rotate(state->base_cube_rotation, ra_v3(1, 0, 0)));
        ra_obs_xyz(observation, &index, ra_rotate(state->base_cube_rotation, ra_v3(0, 1, 0)));
        ra_obs3(observation, &index, state->base_cube_velocity, RA_OBS_LIN_VEL_SCALE);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[0]);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[1]);
        observation[index++] = ra_clamp(
            RA_OBS_ANG_VEL_SCALE * ra_length(state->base_cube_angular_velocity), 0.0f, 1.0f);
    } else if (state->basketball_mode) {
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[0]);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[1]);
        observation[index++] = grip_vel;
        ra_obs3(observation, &index, state->cube_position, RA_OBS_POS_SCALE);
        ra_obs3(observation, &index, ra_rotate(ra_qconj(hand_rotation), state->cube_velocity),
            RA_OBS_LIN_VEL_SCALE);
        ra_obs3(observation, &index,
            RaDynamics::point_velocity(state->qd, origins, axes, RA_DOF - 1, state->end_effector),
            RA_OBS_LIN_VEL_SCALE);
    } else {
        ra_obs3(observation, &index, state->target_position, RA_OBS_POS_SCALE);
        ra_obs_xyz(observation, &index, ra_rotate(state->cube_rotation, ra_v3(0, 1, 0)));
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[0]);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[1]);
        ra_obs_xyz(observation, &index, ra_rotate(hand_rotation, ra_v3(0, 0, 1)));
        observation[index++] = grip_vel;
    }
    observation[index++] = ra_clamp(state->gripper_width / 0.08f, 0.0f, 1.0f);
    observation[index++] = ra_clamp(state->gripper_force / RA_GRIPPER_MAX_FORCE, 0.0f, 1.0f);
    ra_obs3(observation, &index,
        ra_rotate(
            ra_qconj(hand_rotation), ra_scale(state->wrist_linear_impulse, 1.0f / RA_CONTROL_DT)),
        0.01f);
    ra_obs3(observation, &index,
        ra_rotate(
            ra_qconj(hand_rotation), ra_scale(state->wrist_angular_impulse, 1.0f / RA_CONTROL_DT)),
        0.20f);
    observation[index++] = state->transported ? 1.0f : 0.0f;
    observation[index++] = state->basketball_mode ? (state->basketball_close_ready ? 1.0f : 0.0f)
        : (state->stack_aligned ? 1.0f : 0.0f);
    observation[index++] = state->basketball_mode ? (state->basketball_in_flight ? 1.0f : 0.0f)
        : (state->released_near_target ? 1.0f : 0.0f);
    observation[index++] = state->grasped ? 1.0f : 0.0f;
    observation[index++] = state->lifted ? 1.0f : 0.0f;
    int maximum_steps = state->basketball_mode ? RA_BASKETBALL_MAX_STEPS : RA_MAX_STEPS;
    observation[index++] = ra_clamp((float)state->step / (float)maximum_steps, 0.0f, 1.0f);
    float table_clearance[RA_DOF + 3];
    ra_table_clearances(gripper_links, state->end_effector, table_clearance);
    for (int body = 0; body < RA_DOF + 3; ++body) {
        observation[index++] = ra_clamp(table_clearance[body] / RA_OBS_TABLE_RANGE, -1.0f, 1.0f);
    }
    ra_obs_xyz(observation, &index, ra_rotate(ra_qconj(hand_rotation), ra_v3(0, 1, 0)));
    float hand_clearance = ra_min(table_clearance[RA_DOF],
        ra_min(table_clearance[RA_DOF + 1], table_clearance[RA_DOF + 2]));
    // Translate the current hand pose to the object's center without rotating it.
    float approach_clearance = hand_clearance + state->cube_position.y - reach_origin.y;
    observation[index++] = ra_clamp(approach_clearance / RA_OBS_TABLE_RANGE, -1.0f, 1.0f);
    observation[index++] = (float)state->table_blocked_substeps / (float)RA_SUBSTEPS;
    observation[index++] = state->basketball_mode
        ? ra_clamp(state->basketball_approach_reward_left / RA_BASKETBALL_APPROACH_BUDGET,
            0.0f, 1.0f) : 0.0f;
    observation[index++] = state->basketball_mode
        ? ra_clamp(state->previous_grip_error / 0.08f, 0.0f, 1.0f) : 0.0f;
    assert(index == OBS_SIZE);
}

RA_HD static void ra_resetb(RaState* state) {
    state->cube_position = ra_v3(ra_rand(&state->rng, 0.42f, 0.54f), RA_TABLE_TOP + RA_BALL_RADIUS,
        ra_rand(&state->rng, 0.20f, 0.32f));
    state->cube_velocity = ra_v3(0, 0, 0);
    state->cube_rotation = ra_quat(0, 0, 0, 1);
    state->cube_angular_velocity = ra_v3(0, 0, 0);
    state->previous_ball_position = state->cube_position;
    state->target_position = ra_hoop();
    state->basketball_in_flight = 0;
    state->basketball_grounded_steps = 0;
    state->grasped = 0;
    state->grasp_cooldown = 0;
    state->grasp_contact_misses = 0;
    state->ever_grasped = 0;
    state->lifted = 0;
    state->transported = 0;
    state->released_near_target = 0;
    state->placement_settle_steps = 0;
    state->basketball_close_ready = 0;
    state->basketball_release_ready = 0;
    state->basketball_release_commanded = 0;
    state->gripper_force = 0.0f;
    memset(state->pad_normal_impulse, 0, sizeof(state->pad_normal_impulse));
    RaPose links[RA_LINKS];
    ra_fk(state->q, state->gripper_width, links, NULL, NULL, NULL);
    RaVec3 grasp_center = ra_gctr(state->end_effector, links[RA_DOF + 1].rotation);
    state->previous_reach_distance = ra_length(ra_sub(state->cube_position, grasp_center));
    state->basketball_best_grasp_center = grasp_center;
    state->basketball_previous_grasp_center = grasp_center;
    state->basketball_approach_reward_left = RA_BASKETBALL_APPROACH_BUDGET;
    state->basketball_best_open_error =
        fabsf(state->gripper_width - RA_BASKETBALL_OPEN_WIDTH);
    state->basketball_best_close_error =
        fabsf(state->gripper_width - RA_BASKETBALL_GRIP_WIDTH);
    state->previous_place_distance =
        ra_length(ra_sub(state->target_position, state->cube_position));
    state->previous_lift_height = 0.0f;
    state->previous_grip_error = fabsf(state->gripper_width - RA_BASKETBALL_OPEN_WIDTH);
    float launch_quality = ra_blq(state->cube_position, state->cube_velocity);
    float trajectory_quality = ra_btq(state->cube_position, state->cube_velocity);
    state->previous_throw_quality = 0.35f * launch_quality + 0.65f * trajectory_quality;
}

RA_HD static void ra_reset(RaState* state) {
    uint32_t rng = state->rng ? state->rng : 1u;
    int no_timeout = state->no_timeout;
    int stack_mode = state->stack_mode;
    int basketball_mode = state->basketball_mode;
    memset(state, 0, sizeof(*state));
    state->rng = rng;
    state->no_timeout = no_timeout;
    state->stack_mode = stack_mode;
    state->basketball_mode = basketball_mode;
    state->cube_rotation = ra_quat(0, 0, 0, 1);
    state->base_cube_rotation = ra_quat(0, 0, 0, 1);
    for (int joint = 0; joint < RA_DOF; ++joint) {
        state->q[joint] = ra_jhome(joint) + ra_rand(&state->rng, -0.035f, 0.035f);
        state->target_q[joint] = state->q[joint];
    }
    state->gripper_width = 0.080f;
    RaPose links[RA_LINKS];
    ra_fk(state->q, state->gripper_width, links, NULL, NULL, &state->end_effector);

    if (state->basketball_mode) {
        ra_resetb(state);
        return;
    }

    float cube_angle = ra_rand(&state->rng, -0.72f, -0.28f);
    float cube_radius = ra_rand(&state->rng, 0.43f, 0.62f);
    state->cube_position = ra_v3(cube_radius * cosf(cube_angle), RA_TABLE_TOP + RA_CUBE_HALF,
        -cube_radius * sinf(cube_angle));
    float target_angle = ra_rand(&state->rng, 0.28f, 0.72f);
    float target_radius = ra_rand(&state->rng, 0.43f, 0.62f);
    if (state->stack_mode) {
        state->base_cube_position = ra_v3(target_radius * cosf(target_angle),
            RA_TABLE_TOP + RA_CUBE_HALF, -target_radius * sinf(target_angle));
        state->base_cube_start_position = state->base_cube_position;
        state->previous_base_cube_position = state->base_cube_position;
        state->target_position =
            ra_add(state->base_cube_position, ra_v3(0, 2.0f * RA_CUBE_HALF, 0));
    } else {
        state->target_position = ra_v3(target_radius * cosf(target_angle), RA_TABLE_TOP + 0.008f,
            -target_radius * sinf(target_angle));
    }
    state->previous_reach_distance = ra_length(ra_sub(state->cube_position, state->end_effector));
    state->previous_place_distance =
        ra_length(ra_sub(state->target_position, state->cube_position));
    state->previous_lift_height = 0.0f;
    RaVec3 stack_delta = ra_sub(state->cube_position, state->base_cube_position);
    state->previous_stack_horizontal =
        sqrtf(stack_delta.x * stack_delta.x + stack_delta.z * stack_delta.z);
    float stack_clearance = stack_delta.y - 2.0f * RA_CUBE_HALF;
    state->previous_stack_drop_error = fabsf(stack_clearance - RA_STACK_HOVER_CLEARANCE);
    state->previous_stack_orientation_error = 0.0f;
}

RA_D static void ra_logep(const RaState* state, struct Log* log) {
    log->episode_length += (float)state->step;
    log->lift_rate += state->lifted ? 1.0f : 0.0f;
    log->slip_rate += state->slip_events > 0 ? 1.0f : 0.0f;
    log->table_blocked_rate +=
        (float)state->table_blocked_steps / ra_max((float)state->step, 1.0f);
    log->n += 1.0f;
    if (state->basketball_mode) {
        float grasp_denominator = ra_max((float)state->attempts, (float)state->basketball_grasps);
        float release_denominator = ra_max((float)state->basketball_grasps, 1.0f);
        log->basketball_mode += 1.0f;
        log->score += (float)state->baskets;
        log->baskets += (float)state->baskets;
        log->grasp_rate +=
            grasp_denominator > 0.0f ? (float)state->basketball_grasps / grasp_denominator : 0.0f;
        log->release_rate += (float)state->basketball_releases / release_denominator;
        log->release_center_miss_cm_sum += state->basketball_release_center_miss_cm_sum;
        log->release_center_miss_count += (float)state->basketball_releases;
        return;
    }
    log->score += state->success ? 1.0f : 0.0f;
    log->success_rate += state->success ? 1.0f : 0.0f;
    log->grasp_rate += state->ever_grasped ? 1.0f : 0.0f;
    log->transport_rate += state->transported ? 1.0f : 0.0f;
    log->release_rate += state->stack_mode ? (state->valid_release_achieved ? 1.0f : 0.0f)
        : (state->released_near_target ? 1.0f : 0.0f);
    log->return_value += state->episode_return;
    log->reach_distance += ra_length(ra_sub(state->cube_position, state->end_effector));
    log->place_distance += ra_length(ra_sub(state->target_position, state->cube_position));
    log->energy += state->episode_energy / ra_max((float)state->step, 1.0f);
    log->pinch_force += state->episode_pinch_force / ra_max((float)state->pinch_substeps, 1.0f);
    log->cube_angular_speed += ra_length(state->cube_angular_velocity);
    log->base_angular_speed +=
        state->stack_mode ? ra_length(state->base_cube_angular_velocity) : 0.0f;
    log->orientation_error += state->stack_mode
        ? ra_max(ra_cup(state->cube_rotation), ra_cup(state->base_cube_rotation))
        : ra_cup(state->cube_rotation);
    if (state->stack_mode) {
        RaVec3 alignment = ra_sub(state->cube_position, state->base_cube_position);
        RaVec3 base_slide = ra_sub(state->base_cube_position, state->base_cube_start_position);
        log->stack_rate += state->ever_stacked ? 1.0f : 0.0f;
        log->stable_stack_rate += state->success ? 1.0f : 0.0f;
        log->stack_alignment_rate += state->stack_aligned ? 1.0f : 0.0f;
        log->valid_stack_contact_rate += state->valid_stack_contact ? 1.0f : 0.0f;
        log->clearance_rate += state->cleared_after_release ? 1.0f : 0.0f;
        log->settle_rate += state->max_placement_settle_steps > 0 ? 1.0f : 0.0f;
        log->stack_alignment += sqrtf(alignment.x * alignment.x + alignment.z * alignment.z);
        log->base_slide_distance +=
            sqrtf(base_slide.x * base_slide.x + base_slide.z * base_slide.z);
    }
}

RA_D static float ra_stepb(RaState* state, const float* actions, float energy, int first_grasp,
    int released, const RaPose* links) {
    RaVec3 hoop = ra_hoop();
    RaVec3 grasp_center = ra_gctr(state->end_effector, links[RA_DOF + 1].rotation);
    float reach_distance = ra_length(ra_sub(state->cube_position, grasp_center));
    float hoop_distance = ra_length(ra_sub(hoop, state->cube_position));
    float lift_height = ra_max(0.0f, state->cube_position.y - RA_BALL_RADIUS - RA_TABLE_TOP);
    float reward = -0.0001f;

    if (!state->grasped && !state->basketball_in_flight) {
        float best_distance = ra_length(ra_sub(state->cube_position,
            state->basketball_best_grasp_center));
        float previous_distance = ra_length(ra_sub(state->cube_position,
            state->basketball_previous_grasp_center));
        float baseline = ra_min(best_distance, previous_distance);
        float progress = 0.08f * (1.0f / (1.0f + reach_distance)
                - 1.0f / (1.0f + baseline))
            + 0.05f * (expf(-reach_distance / 0.03f) - expf(-baseline / 0.03f));
        float approach_reward = ra_min(ra_max(progress, 0.0f),
            state->basketball_approach_reward_left);
        reward += approach_reward;
        state->basketball_approach_reward_left -= approach_reward;
        if (reach_distance < best_distance) {
            state->basketball_best_grasp_center = grasp_center;
        }
    } else if (state->grasped && !state->lifted) {
        reward += 0.08f * ra_clamp(lift_height - state->previous_lift_height, -0.03f, 0.03f);
    } else if (state->grasped) {
        reward += 0.04f * ra_clamp(state->previous_place_distance - hoop_distance, -0.05f, 0.05f);
    } else if (state->basketball_in_flight) {
        reward += 0.02f * ra_clamp(state->previous_place_distance - hoop_distance, -0.05f, 0.05f);
    }
    if (!state->grasped && !state->basketball_in_flight
        && reach_distance > RA_BASKETBALL_REOPEN_DISTANCE) {
        state->basketball_close_ready = 0;
    }
    int open_enough = state->gripper_width > 0.062f;
    int entered_close_phase = !state->basketball_close_ready && !state->grasped
        && !state->basketball_in_flight && open_enough
        && reach_distance < RA_BASKETBALL_CLOSE_DISTANCE;
    if (entered_close_phase) {
        state->basketball_close_ready = 1;
        state->basketball_best_close_error = ra_min(state->basketball_best_close_error,
            fabsf(state->gripper_width - RA_BASKETBALL_GRIP_WIDTH));
    }
    float target_width =
        state->basketball_close_ready ? RA_BASKETBALL_GRIP_WIDTH : RA_BASKETBALL_OPEN_WIDTH;
    float grip_error = fabsf(state->gripper_width - target_width);
    float* best_grip_error = state->basketball_close_ready
        ? &state->basketball_best_close_error : &state->basketball_best_open_error;
    if (!state->grasped && !state->basketball_in_flight && !entered_close_phase) {
        reward += 0.15f * ra_max(*best_grip_error - grip_error, 0.0f);
        *best_grip_error = ra_min(*best_grip_error, grip_error);
    }
    state->previous_grip_error = *best_grip_error;
    if (first_grasp) {
        state->basketball_grasps += 1;
        reward += 0.150f;
    }
    if (!state->lifted && state->grasped && lift_height >= RA_LIFT_HEIGHT) {
        state->lifted = 1;
        reward += 0.025f;
    }
    if (!state->transported && state->grasped && state->lifted
        && hoop_distance < RA_BASKETBALL_RELEASE_DISTANCE) {
        state->transported = 1;
        reward += 0.020f;
    }
    float launch_quality = ra_blq(state->cube_position, state->cube_velocity);
    float trajectory_quality = ra_btq(state->cube_position, state->cube_velocity);
    float release_quality = 0.35f * launch_quality + 0.65f * trajectory_quality;
    if (state->grasped && state->lifted) {
        reward += 0.040f * ra_clamp(release_quality - state->previous_throw_quality, -0.05f, 0.05f);
        if (!state->basketball_release_ready
            && release_quality >= RA_BASKETBALL_RELEASE_READY_QUALITY) {
            state->basketball_release_ready = 1;
        }
        if (state->basketball_release_ready) {
            reward -= 0.0005f;
            if (actions[RA_DOF] > 0.25f) {
                reward += 0.010f * release_quality;
            }
        }
    }
    int opening_for_release = state->lifted && state->ever_grasped && !state->basketball_in_flight
        && actions[RA_DOF] > 0.25f && (state->grasped || released);
    if (opening_for_release && !state->basketball_release_commanded) {
        state->basketball_release_commanded = 1;
        reward += 0.015f + 0.015f * release_quality;
    }
    int thrown = state->lifted && !state->grasped && state->ever_grasped && released;
    if (!state->basketball_in_flight && thrown) {
        state->basketball_in_flight = 1;
        state->basketball_releases += 1;
        RaVec3 predicted_crossing;
        float center_miss = RA_BASKETBALL_PREDICTED_MISS_CAP;
        if (ra_bxing(state->cube_position, state->cube_velocity, &predicted_crossing, NULL, NULL)) {
            float dx = predicted_crossing.x - hoop.x;
            float dz = predicted_crossing.z - hoop.z;
            center_miss = ra_min(sqrtf(dx * dx + dz * dz), RA_BASKETBALL_PREDICTED_MISS_CAP);
        }
        state->basketball_release_center_miss_cm_sum += 100.0f * center_miss;
        state->basketball_release_ready = 0;
        reward += 0.020f * launch_quality + 0.100f * trajectory_quality;
    }

    int crossed_down = state->basketball_in_flight && state->previous_ball_position.y > hoop.y
        && state->cube_position.y <= hoop.y;
    int scored = 0;
    if (crossed_down) {
        float height_delta = state->previous_ball_position.y - state->cube_position.y;
        float fraction = height_delta > 1.0e-7f
            ? (state->previous_ball_position.y - hoop.y) / height_delta
            : 0.0f;
        RaVec3 crossing = ra_add(state->previous_ball_position,
            ra_scale(ra_sub(state->cube_position, state->previous_ball_position), fraction));
        float offset_x = crossing.x - hoop.x;
        float offset_z = crossing.z - hoop.z;
        float clearance = RA_HOOP_INNER_RADIUS - RA_BALL_RADIUS;
        scored = offset_x * offset_x + offset_z * offset_z < clearance * clearance;
    }
    int grounded = !state->grasped
        && state->cube_position.y
            <= RA_TABLE_TOP + RA_BALL_RADIUS + RA_BASKETBALL_GROUNDED_HEIGHT_SLOP
        && fabsf(state->cube_velocity.y) <= RA_BASKETBALL_GROUNDED_MAX_VERTICAL_SPEED;
    float ball_base_distance = ra_length(state->cube_position);
    int out_of_reach = ball_base_distance - RA_BALL_RADIUS > RA_ARM_GEOMETRIC_REACH_BOUND;
    if (grounded && out_of_reach) {
        state->basketball_grounded_steps += 1;
    } else {
        state->basketball_grounded_steps = 0;
    }
    int grounded_reset = state->basketball_grounded_steps >= RA_BASKETBALL_GROUNDED_RESET_STEPS;
    if (scored) {
        state->baskets += 1;
        state->attempts += 1;
        state->success = 1;
        reward = 1.0f;
        state->basketball_in_flight = 0;
    } else if (grounded && state->basketball_in_flight) {
        state->attempts += 1;
        reward = -0.010f;
        state->basketball_in_flight = 0;
        state->basketball_release_ready = 0;
    }
    if (grounded_reset) {
        ra_resetb(state);
        state->basketball_reset = 1;
    }

    if (!state->no_timeout && state->step >= RA_BASKETBALL_MAX_STEPS) {
        state->done = 1;
    }
    if (!state->basketball_reset) {
        state->previous_reach_distance = reach_distance;
        state->previous_place_distance = hoop_distance;
        state->previous_lift_height = lift_height;
        state->previous_throw_quality = release_quality;
    }
    return reward;
}

RA_D static float ra_stept(RaState* state, const float* actions, float energy, int first_grasp,
    int released, const RaPose* links) {
    float grip_action = ra_clamp(actions[RA_DOF], -1.0f, 1.0f);
    float reach_distance = ra_length(ra_sub(state->cube_position, state->end_effector));
    RaQuat hand_rotation = links[RA_DOF + 1].rotation;
    RaVec3 hand_position =
        ra_sub(state->end_effector, ra_rotate(hand_rotation, ra_v3(0, 0, 0.115f)));
    float half_width = 0.5f * state->gripper_width;
    RaPose fingers[2];
    fingers[0].position =
        ra_add(hand_position, ra_rotate(hand_rotation, ra_v3(0, half_width, 0.0584f)));
    fingers[0].rotation = hand_rotation;
    fingers[1].position =
        ra_add(hand_position, ra_rotate(hand_rotation, ra_v3(0, -half_width, 0.0584f)));
    fingers[1].rotation =
        ra_qnorm(ra_qmul(hand_rotation, ra_qaxis(ra_v3(0, 0, 1), 3.14159265359f)));
    int gripper_clear = 1;
    for (int finger = 0; finger < 2; ++finger) {
        gripper_clear &= !ra_pad_clearance(state->cube_position, state->cube_rotation,
            fingers[finger], RA_GRIPPER_CLEARANCE_MARGIN);
        if (state->stack_mode) {
            gripper_clear &= !ra_pad_clearance(state->base_cube_position, state->base_cube_rotation,
                fingers[finger], RA_GRIPPER_CLEARANCE_MARGIN);
        }
    }
    RaVec3 place_offset = ra_sub(state->target_position, state->cube_position);
    float place_distance = ra_length(place_offset);
    float place_horizontal_distance =
        sqrtf(place_offset.x * place_offset.x + place_offset.z * place_offset.z);
    float main_support_y = ra_csup(state->cube_rotation, ra_v3(0, 1, 0));
    float base_support_y = ra_csup(state->base_cube_rotation, ra_v3(0, 1, 0));
    float expected_stack_separation = main_support_y + base_support_y;
    float stack_height_error =
        fabsf((state->cube_position.y - state->base_cube_position.y) - expected_stack_separation);
    float stack_clearance =
        (state->cube_position.y - state->base_cube_position.y) - expected_stack_separation;
    float stack_drop_error = fabsf(stack_clearance - RA_STACK_HOVER_CLEARANCE);
    float stack_orientation_error = ra_cup(state->cube_rotation);
    RaVec3 base_motion_delta =
        ra_sub(state->base_cube_position, state->previous_base_cube_position);
    float base_motion = sqrtf(
        base_motion_delta.x * base_motion_delta.x + base_motion_delta.z * base_motion_delta.z);
    int stack_release_pose = place_horizontal_distance < RA_STACK_RELEASE_RADIUS
        && stack_clearance >= -RA_STACK_HEIGHT_TOLERANCE
        && stack_clearance < RA_STACK_RELEASE_CLEARANCE;
    int stack_alignment_pose = place_horizontal_distance < 0.050f
        && stack_clearance >= -RA_STACK_HEIGHT_TOLERANCE && stack_clearance < 0.080f;
    float lift_height = ra_max(0.0f, state->cube_position.y - main_support_y - RA_TABLE_TOP);
    int stack_contact = state->stack_mode && !state->grasped
        && place_horizontal_distance < 2.0f * RA_CUBE_HALF
        && stack_height_error < RA_STACK_HEIGHT_TOLERANCE;
    if (stack_contact) {
        state->ever_stacked = 1;
    }
    int placement_disturbed = state->released_near_target
        && (state->grasped
            || place_horizontal_distance
                >= (state->stack_mode ? RA_STACK_RELEASE_RADIUS : RA_PLACE_RADIUS));
    if (placement_disturbed) {
        state->released_near_target = 0;
        state->placement_settle_steps = 0;
    }
    float reward = -0.002f;
    if (!state->grasped) {
        if (state->released_near_target) {
            reward += (state->stack_mode ? 8.0f : 1.8f)
                * ra_clamp(reach_distance - state->previous_reach_distance, -0.05f, 0.05f);
        } else {
            reward +=
                1.8f * ra_clamp(state->previous_reach_distance - reach_distance, -0.05f, 0.05f);
        }
    } else if (!state->lifted) {
        reward += (state->stack_mode ? 12.0f : 5.0f)
            * ra_clamp(lift_height - state->previous_lift_height, -0.03f, 0.03f);
        if (lift_height >= RA_LIFT_HEIGHT) {
            state->lifted = 1;
            reward += state->stack_mode ? 2.0f : 0.75f;
        }
    }
    if (state->grasped && state->lifted) {
        if (state->stack_mode) {
            reward += RA_STACK_HORIZONTAL_PROGRESS_REWARD
                * ra_clamp(
                    state->previous_stack_horizontal - place_horizontal_distance, -0.05f, 0.05f);
            if (place_horizontal_distance < 0.12f) {
                reward += RA_STACK_HEIGHT_PROGRESS_REWARD
                    * ra_clamp(state->previous_stack_drop_error - stack_drop_error, -0.04f, 0.04f);
                reward += RA_STACK_UPRIGHT_PROGRESS_REWARD
                    * ra_clamp(state->previous_stack_orientation_error - stack_orientation_error,
                        -0.05f, 0.05f);
            }
        } else {
            reward +=
                6.0f * ra_clamp(state->previous_place_distance - place_distance, -0.05f, 0.05f);
        }
    }
    if (first_grasp) {
        reward += state->stack_mode ? 0.40f : 0.5f;
    }
    if (released && state->stack_mode) {
        if (grip_action <= 0.25f) {
            float slip_penalty = state->stack_aligned
                ? 2.00f
                : (state->transported ? 1.00f : (state->lifted ? RA_STACK_SLIP_PENALTY : 0.10f));
            reward -= slip_penalty;
        } else if (!state->transported) {
            reward -= state->lifted ? 0.75f : 0.15f;
        }
    } else if (released && !state->transported) {
        reward -= state->lifted ? 0.50f : 0.10f;
    }
    if (placement_disturbed) {
        reward -= 0.25f;
    }
    int transport_pose = state->stack_mode
        ? (place_horizontal_distance < RA_STACK_TRANSPORT_RADIUS
            && stack_clearance >= -RA_STACK_HEIGHT_TOLERANCE && stack_clearance < 0.10f)
        : place_distance < 0.12f;
    if (!state->transported && state->grasped && state->lifted && lift_height >= RA_CARRY_HEIGHT
        && transport_pose) {
        state->transported = 1;
        reward += state->stack_mode ? 2.0f : 0.5f;
    }
    int stack_release_ready = stack_release_pose && stack_orientation_error < 0.050f
        && ra_length(state->cube_velocity) < 0.50f
        && ra_length(state->cube_angular_velocity) < 2.0f;
    if (state->stack_mode && state->transported && state->grasped && stack_release_ready
        && !state->stack_aligned) {
        state->stack_aligned = 1;
        reward += 2.0f;
    }
    if (state->stack_mode && state->transported && state->stack_aligned && state->grasped) {
        reward -= 0.030f;
        if (!state->stack_opening_credited && grip_action > 0.25f) {
            state->stack_opening_credited = 1;
            if (stack_release_ready) {
                reward += 1.000f;
            } else if (stack_alignment_pose) {
                reward += 0.200f;
            }
        }
    }
    int valid_release = released
        && (state->stack_mode
                ? (grip_action > 0.25f && state->stack_aligned && stack_alignment_pose)
                : place_distance < 0.12f);
    int first_valid_release = !state->valid_release_achieved && state->transported && valid_release;
    if (!state->released_near_target && state->transported && valid_release) {
        state->released_near_target = 1;
        state->valid_release_achieved = 1;
        if (state->stack_mode && first_valid_release) {
            float release_quality_penalty = ra_clamp(2.0f * ra_length(state->cube_velocity)
                    + 0.5f * ra_length(state->cube_angular_velocity)
                    + 20.0f * stack_orientation_error,
                0.0f, 4.0f);
            reward += 6.0f - release_quality_penalty;
        } else if (!state->stack_mode) {
            reward += 0.25f;
        }
    }
    if (state->stack_mode && state->transported && released && !valid_release
        && grip_action > 0.25f) {
        reward -= 2.0f;
    }
    if (state->stack_mode && stack_contact && state->valid_release_achieved
        && !state->valid_stack_contact) {
        state->valid_stack_contact = 1;
        reward += 4.0f;
    }
    if (state->stack_mode && state->released_near_target && gripper_clear
        && !state->cleared_after_release) {
        state->cleared_after_release = 1;
        reward += 4.0f;
    }
    if (state->stack_mode && state->released_near_target && place_horizontal_distance < 0.040f
        && stack_height_error < 0.010f) {
        if (stack_orientation_error < RA_STACK_UPRIGHT_ERROR) {
            reward += 0.020f;
        }
        if (ra_length(state->cube_velocity) < RA_STACK_SETTLE_SPEED
            && ra_length(state->base_cube_velocity) < RA_STACK_SETTLE_SPEED) {
            reward += 0.020f;
        }
        if (ra_length(state->cube_angular_velocity) < RA_STACK_SETTLE_ANGULAR_SPEED
            && ra_length(state->base_cube_angular_velocity) < RA_STACK_SETTLE_ANGULAR_SPEED) {
            reward += 0.020f;
        }
        if (gripper_clear) {
            reward += 0.030f;
        }
    }
    if (state->stack_mode) {
        reward -= 8.0f * ra_min(base_motion, 0.03f);
    }
    float action_cost = 0.0f;
    float action_delta = 0.0f;
    for (int action = 0; action < RA_DOF; ++action) {
        float value = ra_clamp(actions[action], -1.0f, 1.0f);
        action_cost += value * value;
        float delta = value - state->previous_action[action];
        action_delta += delta * delta;
    }

    int placement_stable;
    int settle_steps_required;
    if (state->stack_mode) {
        placement_stable = !state->grasped && state->released_near_target && state->lifted
            && state->transported && place_horizontal_distance < RA_STACK_ALIGN_RADIUS
            && stack_height_error < RA_STACK_HEIGHT_TOLERANCE
            && ra_cup(state->cube_rotation) < RA_STACK_UPRIGHT_ERROR
            && ra_cup(state->base_cube_rotation) < RA_STACK_UPRIGHT_ERROR
            && state->base_cube_position.y <= RA_TABLE_TOP + base_support_y + 0.004f
            && ra_length(state->cube_velocity) < RA_STACK_SETTLE_SPEED
            && ra_length(state->base_cube_velocity) < RA_STACK_SETTLE_SPEED
            && ra_length(state->cube_angular_velocity) < RA_STACK_SETTLE_ANGULAR_SPEED
            && ra_length(state->base_cube_angular_velocity) < RA_STACK_SETTLE_ANGULAR_SPEED
            && gripper_clear;
        settle_steps_required = RA_STACK_SETTLE_STEPS;
    } else {
        placement_stable = !state->grasped && state->released_near_target && state->lifted
            && state->transported && place_horizontal_distance < RA_PLACE_RADIUS
            && state->cube_position.y <= RA_TABLE_TOP + main_support_y + 0.008f
            && ra_length(state->cube_velocity) < RA_PLACE_SETTLE_SPEED
            && ra_length(state->cube_angular_velocity) < RA_PLACE_SETTLE_ANGULAR_SPEED
            && reach_distance > RA_PLACE_CLEARANCE;
        settle_steps_required = RA_PLACE_SETTLE_STEPS;
    }
    if (placement_stable) {
        state->placement_settle_steps += 1;
        if (state->placement_settle_steps > state->max_placement_settle_steps) {
            state->max_placement_settle_steps = state->placement_settle_steps;
        }
        reward += state->stack_mode ? 0.12f : 0.01f;
    } else {
        state->placement_settle_steps = 0;
    }
    if (state->placement_settle_steps >= settle_steps_required) {
        state->success = 1;
        state->done = 1;
        reward += state->stack_mode ? 20.0f : 10.0f;
    }
    if (!state->no_timeout && state->step >= RA_MAX_STEPS) {
        state->done = 1;
    }
    if (state->cube_position.y < -0.25f) {
        state->done = 1;
        reward -= 0.25f;
    }
    if (state->stack_mode && state->base_cube_position.y < -0.25f) {
        state->done = 1;
        reward -= 0.25f;
    }

    reward *= state->stack_mode ? RA_STACK_REWARD_SCALE : RA_PICK_REWARD_SCALE;
    reward -= 0.001f * action_cost + 0.0004f * action_delta;
    reward -= 0.00002f * energy;

    state->previous_reach_distance = reach_distance;
    state->previous_place_distance = place_distance;
    state->previous_lift_height = lift_height;
    state->previous_stack_horizontal = place_horizontal_distance;
    state->previous_stack_drop_error = stack_drop_error;
    state->previous_stack_orientation_error = stack_orientation_error;
    state->previous_base_cube_position = state->base_cube_position;
    return reward;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "table_blocked_rate", log->table_blocked_rate);
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->score);
    if (log->basketball_mode > 0.5f) {
        dict_set(out, "baskets", log->baskets);
        dict_set(out, "grasp_rate", log->grasp_rate);
        dict_set(out, "lift_rate", log->lift_rate);
        dict_set(out, "release_rate", log->release_rate);
        dict_set(out, "slip_rate", log->slip_rate);
        float release_count = log->release_center_miss_count;
        dict_set(out, "avg_release_miss_cm",
            release_count > 0.0f ? log->release_center_miss_cm_sum / release_count : 0.0f);
        dict_set(out, "episode_length", log->episode_length);
        return;
    }
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "grasp_rate", log->grasp_rate);
    dict_set(out, "lift_rate", log->lift_rate);
    dict_set(out, "transport_rate", log->transport_rate);
    dict_set(out, "release_rate", log->release_rate);
    dict_set(out, "episode_return", log->return_value);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "reach_distance", log->reach_distance);
    dict_set(out, "place_distance", log->place_distance);
    dict_set(out, "energy", log->energy);
    dict_set(out, "pinch_force", log->pinch_force);
    dict_set(out, "slip_rate", log->slip_rate);
    dict_set(out, "stack_rate", log->stack_rate);
    dict_set(out, "stable_stack_rate", log->stable_stack_rate);
    dict_set(out, "stack_alignment_rate", log->stack_alignment_rate);
    dict_set(out, "valid_stack_contact_rate", log->valid_stack_contact_rate);
    dict_set(out, "clearance_rate", log->clearance_rate);
    dict_set(out, "settle_rate", log->settle_rate);
    dict_set(out, "stack_alignment", log->stack_alignment);
    dict_set(out, "base_slide_distance", log->base_slide_distance);
    dict_set(out, "cube_angular_speed", log->cube_angular_speed);
    dict_set(out, "base_angular_speed", log->base_angular_speed);
    dict_set(out, "orientation_error", log->orientation_error);
    dict_set(out, "n", log->n);
}
