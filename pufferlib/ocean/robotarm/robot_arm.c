#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "robot_arm.h"

 
static const Color ORANGE_MAIN   = {255, 140,  0, 255};
static const Color ORANGE_LIGHT  = {255, 180, 80, 255};
static const Color METAL_DARK    = { 45,  45, 45, 255};
static const Color METAL_MEDIUM  = {110, 110,110,255};
static const Color METAL_LIGHT   = {170, 170,170,255};
static const Color FLOOR_COLOR   = {235, 238, 240,255};
static const Color SHADOW_COLOR  = {  0,   0,  0, 40};
static const Color CLR_RED   = {231,  76,  60,255};
static const Color CLR_BLUE  = { 52, 152, 219,255};
static const Color CLR_GREEN = { 46, 204, 113,255};
static const Color CLR_YEL   = {241, 196,  15,255};

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline float randf(float a, float b) {
    return a + (float)rand() / (float)RAND_MAX * (b - a);
}
static inline float safe_sqrt(float x) {
    return sqrtf(fmaxf(0.0f, x));
}
static inline float distance3d(const float a[3], const float b[3]) {
    float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return safe_sqrt(dx*dx + dy*dy + dz*dz);
}

static inline void vec3_copy(float dst[3], const float src[3]) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

static inline void vec3_add(float dst[3], const float a[3], const float b[3]) {
    dst[0] = a[0] + b[0]; dst[1] = a[1] + b[1]; dst[2] = a[2] + b[2];
}

static inline void vec3_scale(float dst[3], const float src[3], float scale) {
    dst[0] = src[0] * scale; dst[1] = src[1] * scale; dst[2] = src[2] * scale;
}

static inline void vec3_zero(float v[3]) {
    v[0] = v[1] = v[2] = 0.0f;
}

static inline float vec3_length(const float v[3]) {
    return safe_sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static inline void vec3_normalize(float dst[3], const float src[3]) {
    float len = vec3_length(src);
    if (len > 1e-6f) {
        dst[0] = src[0] / len; dst[1] = src[1] / len; dst[2] = src[2] / len;
    } else {
        dst[0] = dst[1] = dst[2] = 0.0f;
    }
}

static void init_physics_body(PhysicsBody* body, const float pos[3]) {
    vec3_copy(body->pos, pos);
    vec3_zero(body->vel);
    vec3_zero(body->angular_vel);
    vec3_zero(body->force);
    vec3_zero(body->torque);
    body->orientation[0] = 1.0f;
    body->orientation[1] = 0.0f;
    body->orientation[2] = 0.0f;
    body->orientation[3] = 0.0f;
    body->on_surface = false;
    body->in_contact = false;
    body->contact_time = 0.0f;
}

static bool check_ground_collision(const ManipObject* obj) {
    float bottom = obj->physics.pos[2] - obj->size[2] * 0.5f;
    return bottom <= TABLE_HEIGHT;
}

static bool check_gripper_collision(const ManipObject* obj, const GripperFinger* finger) {
    float dx = obj->physics.pos[0] - finger->pos[0];
    float dy = obj->physics.pos[1] - finger->pos[1];
    float dz = obj->physics.pos[2] - finger->pos[2];
    float dist_sq = dx*dx + dy*dy + dz*dz;
    float obj_radius = obj->size[0] * 0.5f;
    float contact_dist = GRIPPER_CONTACT_RADIUS + obj_radius;
    return dist_sq < (contact_dist * contact_dist);
}

static void update_object_physics(ManipObject* obj, float dt) {
    if (obj->in_basket) return;
    
    PhysicsBody* body = &obj->physics;

    if (!obj->grasped) {
        body->force[2] += obj->mass * GRAVITY;
    }

    if (check_ground_collision(obj)) {
        float bottom = body->pos[2] - obj->size[2] * 0.5f;
        if (bottom < TABLE_HEIGHT) {

            if (body->vel[2] < 0.0f) {
                body->vel[2] *= -obj->restitution;

                float horizontal_speed = safe_sqrt(body->vel[0]*body->vel[0] + body->vel[1]*body->vel[1]);
                if (horizontal_speed > 1e-3f) {
                    float friction_force = obj->friction * obj->mass * fabsf(GRAVITY);
                    float friction_decel = friction_force / obj->mass;
                    float new_speed = fmaxf(0.0f, horizontal_speed - friction_decel * dt);
                    float scale = new_speed / horizontal_speed;
                    body->vel[0] *= scale;
                    body->vel[1] *= scale;
                }
            }
            body->on_surface = true;
        }
    } else {
        body->on_surface = false;
    }

    if (obj->mass > 0.0f) {
        body->vel[0] += (body->force[0] / obj->mass) * dt;
        body->vel[1] += (body->force[1] / obj->mass) * dt;
        body->vel[2] += (body->force[2] / obj->mass) * dt;
    }

    body->vel[0] *= powf(AIR_DAMPING, dt);
    body->vel[1] *= powf(AIR_DAMPING, dt);
    body->vel[2] *= powf(AIR_DAMPING, dt);

    body->pos[0] += body->vel[0] * dt;
    body->pos[1] += body->vel[1] * dt;
    body->pos[2] += body->vel[2] * dt;

    vec3_zero(body->force);
    vec3_zero(body->torque);
    
    if (body->in_contact) {
        body->contact_time += dt;
    } else {
        body->contact_time = 0.0f;
    }

    float bottom = body->pos[2] - obj->size[2] * 0.5f;
    if (bottom < TABLE_HEIGHT) {
        body->pos[2] = TABLE_HEIGHT + obj->size[2] * 0.5f;
        if (body->vel[2] < 0.0f) {
            body->vel[2] *= -obj->restitution;
            float horizontal_speed = safe_sqrt(body->vel[0]*body->vel[0] + body->vel[1]*body->vel[1]);
            if (horizontal_speed > 1e-3f) {
                float friction_force = obj->friction * obj->mass * fabsf(GRAVITY);
                float friction_decel = friction_force / obj->mass;
                float new_speed = fmaxf(0.0f, horizontal_speed - friction_decel * dt);
                float scale = (horizontal_speed > 0.0f) ? (new_speed / horizontal_speed) : 0.0f;
                body->vel[0] *= scale;
                body->vel[1] *= scale;
            }
        }
        body->on_surface = true;
    }
}

static inline void rotate_world_to_ee(const float ee_rpy[3], const float v[3], float out[3]) {
    float cr = cosf(ee_rpy[0]), sr = sinf(ee_rpy[0]);
    float cp = cosf(ee_rpy[1]), sp = sinf(ee_rpy[1]);
    float cy = cosf(ee_rpy[2]), sy = sinf(ee_rpy[2]);
    float vx = v[0], vy = v[1], vz = v[2];
    float x1 =  cy*vx + sy*vy;
    float y1 = -sy*vx + cy*vy;
    float z1 = vz;
    float x2 =  cp*x1 + sp*z1;
    float y2 =  y1;
    float z2 = -sp*x1 + cp*z1;
    out[0] = x2;
    out[1] =  cr*y2 + sr*z2;
    out[2] = -sr*y2 + cr*z2;
}

static inline void update_trig_cache(RobotArm *env) {
    for (int i=0;i<6;i++) {
        env->cached_sin[i] = sinf(env->joint_angles[i]);
        env->cached_cos[i] = cosf(env->joint_angles[i]);
    }
}

static void compute_forward_kinematics(RobotArm *env) {
    update_trig_cache(env);

    const float cos_base = env->cached_cos[0];
    const float sin_base = env->cached_sin[0];
    const float cos_sh = env->cached_cos[1];
    const float sin_sh = env->cached_sin[1];
    const float cos_wp = env->cached_cos[4];
    const float sin_wp = env->cached_sin[4];
    const float cos_se = cosf(env->joint_angles[1] + env->joint_angles[2]);
    const float sin_se = sinf(env->joint_angles[1] + env->joint_angles[2]);

    const float L1 = env->link1_length > 0 ? env->link1_length : ARM_LINK1_LENGTH;
    const float L2 = env->link2_length > 0 ? env->link2_length : ARM_LINK2_LENGTH;
    const float L3 = env->link3_length > 0 ? env->link3_length : ARM_LINK3_LENGTH;
    const float base_h = 0.2f;

    const float reach = L1*cos_sh + L2*cos_se;
    const float height = L1*sin_sh + L2*sin_se;

    const float wrist_xy = L3 * cos_wp;
    const float wrist_z  = L3 * sin_wp;

    const float yaw_c = cosf(env->joint_angles[0] + env->joint_angles[5]);
    const float yaw_s = sinf(env->joint_angles[0] + env->joint_angles[5]);

    env->end_effector[0] = reach * cos_base + wrist_xy * yaw_c;
    env->end_effector[1] = reach * sin_base + wrist_xy * yaw_s;
    env->end_effector[2] = base_h + height + wrist_z;

    env->end_effector_orient[0] = env->joint_angles[3];
    env->end_effector_orient[1] = env->joint_angles[4];
    env->end_effector_orient[2] = env->joint_angles[0] + env->joint_angles[5];
}

static void update_observations(RobotArm *env) {
    float *obs = env->observations;
    if (!obs) return;

    int obs_idx = 0;
    
    if (env->pick_and_place_mode) {
        
        for (int i = 0; i < 6; i++) {
            obs[obs_idx++] = clampf(env->joint_angles[i] / M_PI, -1.0f, 1.0f);
        }
        obs[obs_idx++] = env->gripper_state;
        obs[obs_idx++] = env->gripper_force / GRIPPER_MAX_FORCE;
        
        float min_dist = 1000.0f;
        int closest_obj = -1;
        for (int i = 0; i < MAX_OBJECTS; i++) {
            if (!env->objects[i].in_basket && env->objects[i].type == env->target_type) {
                float dist = distance3d(env->end_effector, env->objects[i].physics.pos);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_obj = i;
                }
            }
        }
        
        if (closest_obj >= 0) {
            ManipObject* obj = &env->objects[closest_obj];
            obs[obs_idx++] = (obj->physics.pos[0] - env->end_effector[0]) / 0.5f;
            obs[obs_idx++] = (obj->physics.pos[1] - env->end_effector[1]) / 0.5f; 
            obs[obs_idx++] = (obj->physics.pos[2] - env->end_effector[2]) / 0.5f;
            obs[obs_idx++] = obj->grasped ? 1.0f : 0.0f;
            
            int target_basket_idx = obj->target_basket;
            obs[obs_idx++] = (env->baskets[target_basket_idx].pos[0] - env->end_effector[0]) / 0.8f;
            obs[obs_idx++] = (env->baskets[target_basket_idx].pos[1] - env->end_effector[1]) / 0.8f;
            obs[obs_idx++] = (env->baskets[target_basket_idx].pos[2] - env->end_effector[2]) / 0.8f;
            BasketType tb_type = env->baskets[target_basket_idx].type;
            bool color_match =
                (obj->type == OBJ_RED   && tb_type == BASKET_RED)  ||
                (obj->type == OBJ_BLUE  && tb_type == BASKET_BLUE) ||
                (obj->type == OBJ_GREEN && tb_type == BASKET_GREEN);
            obs[obs_idx++] = color_match ? 1.0f : 0.0f;
        } else {
            for (int i = 0; i < 8; i++) obs[obs_idx++] = 0.0f;
        }
        
        obs[obs_idx++] = (env->target_type == OBJ_RED)   ? 1.0f : 0.0f;
        obs[obs_idx++] = (env->target_type == OBJ_BLUE)  ? 1.0f : 0.0f;
        obs[obs_idx++] = (env->target_type == OBJ_GREEN) ? 1.0f : 0.0f;

        // Extended observation: append joint velocities and contact flags
        if (env->extended_observation) {
            for (int i = 0; i < 6; i++) {
                obs[obs_idx++] = clampf(env->joint_vel[i] / 3.0f, -1.0f, 1.0f);
            }
            obs[obs_idx++] = env->left_finger.in_contact ? 1.0f : 0.0f;
            obs[obs_idx++] = env->right_finger.in_contact ? 1.0f : 0.0f;
        }
        
    } else {
        for (int i = 0; i < 6; i++) {
            obs[obs_idx++] = env->joint_angles[i] / M_PI;
        }

        float rel_world[3] = {
            env->target_pos[0] - env->end_effector[0],
            env->target_pos[1] - env->end_effector[1],
            env->target_pos[2] - env->end_effector[2],
        };
        float rel_ee[3];
        rotate_world_to_ee(env->end_effector_orient, rel_world, rel_ee);
        
        for (int k = 0; k < 3; k++) {
            float v = rel_ee[k];
            if (env->obs_noise_std > 0) v += randf(-env->obs_noise_std, env->obs_noise_std);
            obs[obs_idx++] = clampf(v, -1.0f, 1.0f);
        }
        
        obs[obs_idx++] = env->target_orient[0] - env->end_effector_orient[0];
        obs[obs_idx++] = env->target_orient[1] - env->end_effector_orient[1];
        obs[obs_idx++] = env->target_orient[2] - env->end_effector_orient[2];
        float d = env->current_distance;
        obs[obs_idx++] = d;
        
        obs[obs_idx++] = (env->prev_distance - d) / 0.1f;
    }
}

static float compute_reward(RobotArm *env) {
    if (env->pick_and_place_mode) {
        float reward = 0.0f;

        float min_obj_dist = 1e9f;
        int closest_obj = -1;
        for (int i = 0; i < MAX_OBJECTS; i++) {
            if (env->objects[i].in_basket || env->objects[i].type != env->target_type) continue;
            float dist = distance3d(env->end_effector, env->objects[i].physics.pos);
            if (dist < min_obj_dist) { min_obj_dist = dist; closest_obj = i; }
        }

        if (env->grasped_object_id < 0 && closest_obj >= 0) {

            float current_dist = min_obj_dist;
            float prev_dist = env->prev_distance;
            float movement_delta = prev_dist - current_dist;
            reward += (movement_delta > 0.0f) ? (movement_delta * 2.0f) : (movement_delta * 0.5f);
            reward += fmaxf(0.0f, 0.20f - current_dist) * 2.0f;
            {
                int ci = closest_obj;
                if (ci >= 0) {
                    ManipObject* obj = &env->objects[ci];
                    float dz = fabsf(env->end_effector[2] - obj->physics.pos[2]);
                    reward += fmaxf(0.0f, 0.05f - dz) * 1.0f;
                }
            }
            if (current_dist < 0.10f && env->gripper_state > 0.5f) reward += 2.0f;
            if (current_dist >= 0.20f && env->gripper_state > 0.6f) reward -= 1.0f;
            
            if (env->grasp_event) {
                reward += 5.0f;
                env->grasp_event = 0;
                env->episode_pick_count += 1.0f;

                env->log.pick_success_rate += 1.0f;
            }
            env->prev_distance = current_dist;

            {
                float base = reward;
                float scale = (env->reward_scale > 0.0f ? env->reward_scale : 1.0f);
                base *= scale;
                float penalty = 0.0f;
                if (env->action_penalty_coef > 0.0f && env->actions) {
                    for (int i = 0; i < 7; i++) { float a = env->actions[i]; penalty += a*a; }
                    penalty *= env->action_penalty_coef;
                }
                reward = base - penalty;
            }
            if (env->use_unified_clamp) {
                return clampf(reward, env->unified_clamp_min, env->unified_clamp_max);
            }
            return clampf(reward, -3.0f, 8.0f);
        }


        if (env->grasped_object_id >= 0) {
            ManipObject* o = &env->objects[env->grasped_object_id];
            if (o->type != env->target_type) {
                return -1.0f;
            }
            
            int bidx = o->target_basket;
            float current_dist = distance3d(o->physics.pos, env->baskets[bidx].pos);
            float prev_dist = env->prev_distance;
            float movement_delta = prev_dist - current_dist;
            reward += (movement_delta > 0.0f) ? (movement_delta * 2.0f) : (movement_delta * 0.5f);
            reward += fmaxf(0.0f, (BASKET_SIZE * 0.6f) - current_dist) * 2.0f;
            float lift = o->physics.pos[2] - (TABLE_HEIGHT + OBJECT_SIZE * 0.5f);
            if (lift > 0.03f) {
                reward += fminf(2.0f, lift * 20.0f);
                reward += 0.02f;
            }
            
            bool near_basket = current_dist < (BASKET_SIZE * 0.6f);
            bool opened = env->gripper_state < 0.3f;
            
            if (near_basket && opened) {
                o->in_basket = true;
                env->grasped_object_id = -1;
                reward += 8.0f;
                env->episode_score_accum += 1.0f;
                env->episode_place_count += 1.0f;

                env->log.place_success_rate += 1.0f;
                env->baskets[bidx].collected_count += 1.0f;
                env->placed_event = 1;
                // Select next target for continuous runs
                env->target_type = (ObjectType)(rand() % 3);
                env->current_target_object = -1;
                for (int i = 0; i < MAX_OBJECTS; i++) {
                    if (!env->objects[i].in_basket && env->objects[i].type == env->target_type) { env->current_target_object = i; break; }
                }
            }

            if (env->was_grasped_last_step && !o->physics.in_contact && env->gripper_state < 0.9f) {
                reward -= 3.0f;
            }
            env->prev_distance = current_dist;
            {
                float base = reward;
                float scale = (env->reward_scale > 0.0f ? env->reward_scale : 1.0f);
                base *= scale;
                float penalty = 0.0f;
                if (env->action_penalty_coef > 0.0f && env->actions) {
                    for (int i = 0; i < 7; i++) { float a = env->actions[i]; penalty += a*a; }
                    penalty *= env->action_penalty_coef;
                }
                reward = base - penalty;
            }
            if (env->use_unified_clamp) {
                return clampf(reward, env->unified_clamp_min, env->unified_clamp_max);
            }
            return clampf(reward, -4.0f, 10.0f);
        }
        {
            float base = reward;
            float scale = (env->reward_scale > 0.0f ? env->reward_scale : 1.0f);
            base *= scale;
            float penalty = 0.0f;
            if (env->action_penalty_coef > 0.0f && env->actions) {
                for (int i = 0; i < 7; i++) { float a = env->actions[i]; penalty += a*a; }
                penalty *= env->action_penalty_coef;
            }
            reward = base - penalty;
        }
        if (env->use_unified_clamp) {
            return clampf(reward, env->unified_clamp_min, env->unified_clamp_max);
        }
        return clampf(reward, -2.0f, 120.0f);
        
    } else {

        float current_dist = env->current_distance;
        float prev_dist = env->prev_distance; 
        float movement_delta = prev_dist - current_dist;
        
        float base_reward = 2.0f / (1.0f + current_dist * 5.0f);
        
        float movement_reward = 0.0f;
        if (movement_delta > 0.0f) {
            movement_reward = movement_delta * 5.0f;
        } else {
            movement_reward = movement_delta * 2.0f;
        }
        
        float proximity_bonus = 0.0f;
        if (current_dist < 0.4f) proximity_bonus += 0.5f;
        if (current_dist < 0.3f) proximity_bonus += 1.0f;
        if (current_dist < 0.2f) proximity_bonus += 2.0f;
        if (current_dist < 0.1f) proximity_bonus += 4.0f;
        if (current_dist < 0.05f) proximity_bonus += 8.0f;
        

        float action_bonus = 0.1f;
        
        float reward = base_reward + movement_reward + proximity_bonus + action_bonus;
        {
            float base = reward;
            float scale = (env->reward_scale > 0.0f ? env->reward_scale : 1.0f);
            base *= scale;
            float penalty = 0.0f;
            if (env->action_penalty_coef > 0.0f && env->actions) {
                for (int i = 0; i < 7; i++) { float a = env->actions[i]; penalty += a*a; }
                penalty *= env->action_penalty_coef;
            }
            reward = base - penalty;
        }
        return clampf(reward, -2.0f, 15.0f);
    }
}


static void update_gripper_fingers(RobotArm *env) {
    if (!env->pick_and_place_mode) return;
    float finger_separation = (1.0f - env->gripper_state) * (GRIPPER_FINGER_LENGTH * 0.6f);
    env->left_finger.pos[0] = env->end_effector[0] - finger_separation * 0.5f;
    env->left_finger.pos[1] = env->end_effector[1];
    env->left_finger.pos[2] = env->end_effector[2] - 0.015f; 
    env->right_finger.pos[0] = env->end_effector[0] + finger_separation * 0.5f;
    env->right_finger.pos[1] = env->end_effector[1];
    env->right_finger.pos[2] = env->end_effector[2] - 0.015f;
    env->left_finger.in_contact = false;
    env->left_finger.contact_object_id = -1;
    env->right_finger.in_contact = false;
    env->right_finger.contact_object_id = -1;
    vec3_zero(env->left_finger.force);
    vec3_zero(env->right_finger.force);
}

static void apply_gripper_forces(RobotArm *env) {
    if (!env->pick_and_place_mode) return;
    
    env->gripper_force = 0.0f;
    
    env->grasp_event = 0;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        ManipObject* obj = &env->objects[i];
        if (obj->in_basket) continue;
        

        bool left_contact = check_gripper_collision(obj, &env->left_finger);
        bool right_contact = check_gripper_collision(obj, &env->right_finger);
        
        obj->physics.in_contact = left_contact || right_contact;
        
        if (left_contact) {
            env->left_finger.in_contact = true;
            env->left_finger.contact_object_id = i;
        }
        if (right_contact) {
            env->right_finger.in_contact = true;
            env->right_finger.contact_object_id = i;
        }
        
        if (obj->physics.in_contact && env->gripper_state > 0.1f) {
            float center[3] = {
                env->end_effector[0],
                env->end_effector[1],
                env->end_effector[2] - 0.02f
            };
            float to_center[3] = {
                center[0] - obj->physics.pos[0],
                center[1] - obj->physics.pos[1],
                center[2] - obj->physics.pos[2]
            };
            float dist = vec3_length(to_center);
            if (dist > 1e-4f) {
                vec3_scale(to_center, to_center, 1.0f / dist);
            }
            float k = 8.0f;
            float c = 1.5f;
            float f_mag = env->gripper_state * fminf(0.5f, k * dist);
            obj->physics.force[0] += to_center[0] * f_mag - c * obj->physics.vel[0];
            obj->physics.force[1] += to_center[1] * f_mag - c * obj->physics.vel[1];
            obj->physics.force[2] += to_center[2] * f_mag - c * obj->physics.vel[2];

            env->gripper_force = f_mag;

            float contact_threshold = 0.010f;
            if (obj->physics.contact_time > contact_threshold && 
                env->gripper_state > 0.5f &&
                (left_contact || right_contact)) {
                if (!obj->grasped) {
                    obj->grasped = true;
                    env->grasped_object_id = i;
                    env->grasp_stability = 1.0f;
                    env->grasp_event = 1;
                }
            }
        }
    }
}

static void update_gripper(RobotArm *env) {
    if (!env->pick_and_place_mode) return;
    
    float gripper_cmd = env->actions ? env->actions[6] : 0.0f;
    if (env->continuous_gripper) {
        float mapped = 0.5f * (gripper_cmd + 1.0f);
        env->gripper_command = clampf(mapped, 0.0f, 1.0f);
    } else {
        if (gripper_cmd > 0.5f) {
            env->gripper_command = 1.0f;
        } else if (gripper_cmd < -0.5f) {
            env->gripper_command = 0.0f;
        }
    }
    
    float gripper_speed = 5.0f;
    if (env->gripper_command > env->gripper_state) {
        env->gripper_state = fminf(env->gripper_state + gripper_speed * ARM_DT, env->gripper_command);
    } else if (env->gripper_command < env->gripper_state) {
        env->gripper_state = fmaxf(env->gripper_state - gripper_speed * ARM_DT, env->gripper_command);
    }

    update_gripper_fingers(env);

    apply_gripper_forces(env);

    if (env->grasped_object_id >= 0) {
        ManipObject* grasped_obj = &env->objects[env->grasped_object_id];

        if (env->gripper_state < 0.3f || !grasped_obj->physics.in_contact) {
            grasped_obj->grasped = false;
            env->grasped_object_id = -1;
            env->grasp_stability = 0.0f;
        } else {
            float offset[3] = {0.0f, 0.0f, -0.03f};
            grasped_obj->physics.pos[0] = env->end_effector[0] + offset[0];
            grasped_obj->physics.pos[1] = env->end_effector[1] + offset[1];
            grasped_obj->physics.pos[2] = env->end_effector[2] + offset[2];
            
            vec3_zero(grasped_obj->physics.vel);
        }
    }
}

static void apply_actions(RobotArm *env) {

    static const float vmax[6] = {
        1.5f,
        1.2f,
        2.0f,
        2.5f,
        2.5f,
        2.5f
    };

    float alpha = 0.4f;
    float amax  = 30.0f;
    float damp  = 0.10f;

    float v_des[6];
    for (int i = 0; i < 6; i++) {
        float a = env->actions ? env->actions[i] : 0.0f;
        if (env->actuation_noise_std > 0.0f) a += randf(-env->actuation_noise_std, env->actuation_noise_std);
        a = clampf(a, -1.0f, 1.0f);
        float target_vel = a * vmax[i];
        env->cmd_filt[i] = (1.0f - alpha) * env->cmd_filt[i] + alpha * target_vel;
        v_des[i] = env->cmd_filt[i];
    }

    for (int i = 0; i < 6; i++) {
        float dv = v_des[i] - env->joint_vel[i];
        float dv_max = amax * ARM_DT * (env->episode_steps < 400 ? 0.5f : 1.0f);
        dv = clampf(dv, -dv_max, dv_max);
        env->joint_vel[i] += dv;
        
        env->joint_vel[i] *= (1.0f - damp * ARM_DT);

        // Soft-limit velocity scaling near joint limits
        const float lo[6] = {
            -M_PI * 0.75f,
            -M_PI_2 * 0.6f,
            -2.0f,
            -M_PI * 0.6f,
            -M_PI_2 * 0.6f,
            -M_PI_2 * 0.7f
        };
        const float hi[6] = {
            M_PI * 0.75f,
            M_PI_2 * 0.9f,
            -0.1f,
            M_PI * 0.6f,
            M_PI_2 * 0.6f,
            M_PI_2 * 0.7f
        };
        float margin_lo = env->joint_angles[i] - lo[i];
        float margin_hi = hi[i] - env->joint_angles[i];
        float margin = fminf(margin_lo, margin_hi);
        // Scale velocity when within 0.1–0.3 rad of a limit
        float scale = clampf((margin - 0.10f) / 0.20f, 0.15f, 1.0f);
        env->joint_vel[i] *= scale;
        
        float vel_limit = vmax[i] * scale;
        env->joint_vel[i] = clampf(env->joint_vel[i], -vel_limit, vel_limit);
        float jitter = 0.0f; // Disable early jitter to avoid pushing into limits
        env->joint_angles[i] += (env->joint_vel[i] + jitter) * ARM_DT;
    }

    env->joint_angles[0] = clampf(env->joint_angles[0], -M_PI * 0.75f,  M_PI * 0.75f);    // Base: ±135°
    env->joint_angles[1] = clampf(env->joint_angles[1], -M_PI_2 * 0.6f, M_PI_2 * 0.9f);  // Shoulder: -54° to +81°
    env->joint_angles[2] = clampf(env->joint_angles[2], -2.0f, -0.1f);                    // Elbow: -115° to -6° (no hyperextension)
    env->joint_angles[3] = clampf(env->joint_angles[3], -M_PI * 0.6f,   M_PI * 0.6f);    // Wrist roll: ±108°
    env->joint_angles[4] = clampf(env->joint_angles[4], -M_PI_2 * 0.6f, M_PI_2 * 0.6f);  // Wrist pitch: ±54°
    env->joint_angles[5] = clampf(env->joint_angles[5], -M_PI_2 * 0.7f, M_PI_2 * 0.7f);  // Wrist yaw: ±63°
    
    if (env->pick_and_place_mode) {
        update_gripper(env);
    }
}

static void init_pick_place_scene(RobotArm *env) {
    
    const ObjectType obj_types[MAX_OBJECTS] = {OBJ_RED, OBJ_BLUE, OBJ_GREEN, OBJ_YELLOW};
    for (int i = 0; i < MAX_OBJECTS; i++) {
        env->objects[i].type = obj_types[i];
        env->objects[i].grasped = false;
        env->objects[i].in_basket = false;
        env->objects[i].target_basket = i % MAX_BASKETS;
        
        // Physics properties
        env->objects[i].size[0] = OBJECT_SIZE;
        env->objects[i].size[1] = OBJECT_SIZE;
        env->objects[i].size[2] = OBJECT_SIZE;
    env->objects[i].mass = OBJECT_MASS * 2.0f;
    env->objects[i].restitution = OBJECT_RESTITUTION*0.5f;
    env->objects[i].friction = fminf(1.0f, OBJECT_FRICTION*1.2f);
        
        float spawn_difficulty = fminf(1.0f, (float)env->episodes_completed / (float)fmaxf(1, env->curriculum_episodes));
        
        const float L1 = env->link1_length > 0 ? env->link1_length : ARM_LINK1_LENGTH;
        const float L2 = env->link2_length > 0 ? env->link2_length : ARM_LINK2_LENGTH;
        const float L3 = env->link3_length > 0 ? env->link3_length : ARM_LINK3_LENGTH;
        const float max_reach = L1 + L2 + L3;
        
        float initial_pos[3];
        
        if (randf(0.0f, 1.0f) < (1.0f - spawn_difficulty) * 0.5f) {
            float offset_dist = randf(0.08f, 0.15f);
            float angle = randf(0.0f, 2.0f * M_PI);
            initial_pos[0] = env->end_effector[0] + offset_dist * cosf(angle);
            initial_pos[1] = env->end_effector[1] + offset_dist * sinf(angle);
            initial_pos[2] = env->end_effector[2] + randf(-0.05f, 0.05f);
        } else {

            float min_factor = 0.3f + spawn_difficulty * 0.2f;
            float max_factor = 0.7f + spawn_difficulty * 0.2f;
            float min_radius = max_reach * min_factor;
            float max_radius = max_reach * max_factor;
            
            float angle = randf(0.0f, 2.0f * M_PI);
            float radius = randf(min_radius, max_radius);
            initial_pos[0] = radius * cosf(angle);
            initial_pos[1] = radius * sinf(angle);
            initial_pos[2] = TABLE_HEIGHT + OBJECT_SIZE*0.5f;
        }

        initial_pos[0] = clampf(initial_pos[0], WORKSPACE_X_MIN + OBJECT_SIZE, WORKSPACE_X_MAX - OBJECT_SIZE);
        initial_pos[1] = clampf(initial_pos[1], WORKSPACE_Y_MIN + OBJECT_SIZE, WORKSPACE_Y_MAX - OBJECT_SIZE);
        initial_pos[2] = clampf(initial_pos[2], TABLE_HEIGHT + OBJECT_SIZE*0.5f, WORKSPACE_Z_MAX - OBJECT_SIZE);

        init_physics_body(&env->objects[i].physics, initial_pos);
        env->objects[i].physics.on_surface = true;
    }
    

    const BasketType basket_types[MAX_BASKETS] = {BASKET_RED, BASKET_BLUE, BASKET_GREEN};
    const float basket_angles[MAX_BASKETS] = {0.0f, 2.094f, 4.188f};
    for (int i = 0; i < MAX_BASKETS; i++) {
        env->baskets[i].type = basket_types[i];
        env->baskets[i].collected_count = 0.0f;

        float r = 0.4f; 
        env->baskets[i].pos[0] = r * cosf(basket_angles[i]);
        env->baskets[i].pos[1] = r * sinf(basket_angles[i]);
        env->baskets[i].pos[2] = 0.25f + BASKET_SIZE/2.0f;
    }
    
    env->target_type = (ObjectType)(rand() % 3);
    env->current_target_object = -1;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (!env->objects[i].in_basket && env->objects[i].type == env->target_type) {
            env->current_target_object = i;
            break;
        }
    }
    if (env->current_target_object < 0) env->current_target_object = 0;
    env->task_stage = 0;
    env->grasped_object_id = -1;
    env->gripper_state = 0.0f;
    
    float sgp = env->start_grasp_prob > 0.0f ? env->start_grasp_prob : 0.10f;
    if (randf(0.0f, 1.0f) < sgp) {
        int idx = rand() % MAX_OBJECTS;
        env->grasped_object_id = idx;
        env->objects[idx].grasped = true;
        env->gripper_state = 1.0f;
        env->objects[idx].physics.pos[0] = env->end_effector[0];
        env->objects[idx].physics.pos[1] = env->end_effector[1];
        env->objects[idx].physics.pos[2] = env->end_effector[2] - 0.03f;
        vec3_zero(env->objects[idx].physics.vel);
    }

    float min_d = 1e9f;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (env->objects[i].type != env->target_type) continue;
        float d = distance3d(env->end_effector, env->objects[i].physics.pos);
        if (d < min_d) min_d = d;
    }
    env->prev_distance = min_d;
}

static void pick_new_target(RobotArm *env) {
    float spawn_p = env->on_gripper_spawn_prob > 0.0f ? env->on_gripper_spawn_prob : 0.30f;
    if (!env->pick_and_place_mode && randf(0.0f, 1.0f) < spawn_p) {
        env->target_pos[0] = env->end_effector[0];
        env->target_pos[1] = env->end_effector[1];
        env->target_pos[2] = env->end_effector[2];
        env->target_pos[0] = clampf(env->target_pos[0], WORKSPACE_X_MIN, WORKSPACE_X_MAX);
        env->target_pos[1] = clampf(env->target_pos[1], WORKSPACE_Y_MIN, WORKSPACE_Y_MAX);
        env->target_pos[2] = clampf(env->target_pos[2], WORKSPACE_Z_MIN, WORKSPACE_Z_MAX);
        env->target_orient[0] = 0.0f;
        env->target_orient[1] = 0.0f;
        env->target_orient[2] = 0.0f;
        return;
    }

    const float base_h = 0.2f;
    const float L1 = env->link1_length > 0 ? env->link1_length : ARM_LINK1_LENGTH;
    const float L2 = env->link2_length > 0 ? env->link2_length : ARM_LINK2_LENGTH;
    const float L3 = env->link3_length > 0 ? env->link3_length : ARM_LINK3_LENGTH;
    const float Lmax = L1 + L2 + 0.9f * L3;

    const float r_min = 0.30f;
    const float r_ws_max = fminf(WORKSPACE_X_MAX, WORKSPACE_Y_MAX);
    const float r_abs_max = fminf(0.48f, fminf(r_ws_max, Lmax - 0.02f));

    float z_max_reach = base_h + safe_sqrt(fmaxf(0.0f, Lmax*Lmax - r_min*r_min));
    float z_min = fmaxf(WORKSPACE_Z_MIN + 0.20f, 0.25f);
    float z_max = fminf(WORKSPACE_Z_MAX - 0.05f, z_max_reach - 0.01f);
    if (z_max <= z_min) {
        z_max = fminf(WORKSPACE_Z_MAX - 0.05f, base_h + Lmax - 0.02f);
    }

    bool placed = false;
    for (int attempt = 0; attempt < 32 && !placed; attempt++) {
        float theta = randf(0.0f, 2.0f * M_PI);
        float z = randf(z_min, z_max);
        float r_reach = safe_sqrt(fmaxf(0.0f, Lmax*Lmax - (z - base_h)*(z - base_h)));
        float r_hi = fminf(r_abs_max, r_reach - 0.01f);
        float r_lo = fminf(fmaxf(r_min, 0.0f), r_hi);
        if (r_hi <= r_lo) continue;
        float r = randf(r_lo, r_hi);
        env->target_pos[0] = r * cosf(theta);
        env->target_pos[1] = r * sinf(theta);
        env->target_pos[2] = z;
        env->target_pos[0] = clampf(env->target_pos[0], WORKSPACE_X_MIN, WORKSPACE_X_MAX);
        env->target_pos[1] = clampf(env->target_pos[1], WORKSPACE_Y_MIN, WORKSPACE_Y_MAX);
        env->target_pos[2] = clampf(env->target_pos[2], WORKSPACE_Z_MIN, WORKSPACE_Z_MAX);
        placed = true;
    }
    if (!placed) {
        float z = 0.5f * (z_min + z_max);
        float r = 0.5f * (r_min + r_abs_max);
        float theta = randf(0.0f, 2.0f * M_PI);
        env->target_pos[0] = r * cosf(theta);
        env->target_pos[1] = r * sinf(theta);
        env->target_pos[2] = z;
    }

    env->target_orient[0] = 0.0f;
    env->target_orient[1] = 0.0f;
    env->target_orient[2] = 0.0f;
}

void c_reset(RobotArm *env) {
    if (!env) return;

    env->episode_steps = 0;

    env->episode_score_accum = 0.0f;
    env->episode_return_accum = 0.0f;
    env->episode_pick_count = 0.0f;
    env->episode_place_count = 0.0f;
    env->log.pick_success_rate = 0.0f;
    env->log.place_success_rate = 0.0f;

    if (env->domain_randomization) {
        env->link1_length = ARM_LINK1_LENGTH * randf(0.95f, 1.05f);
        env->link2_length = ARM_LINK2_LENGTH * randf(0.95f, 1.05f);
        env->link3_length = ARM_LINK3_LENGTH * randf(0.95f, 1.05f);
    } else {
        env->link1_length = ARM_LINK1_LENGTH;
        env->link2_length = ARM_LINK2_LENGTH;
        env->link3_length = ARM_LINK3_LENGTH;
    }

    env->joint_angles[0] = randf(-M_PI * 0.5f, M_PI * 0.5f);        // base: ±90°
    env->joint_angles[1] = randf(-0.3f, 1.0f);                      // shoulder: -17° to +57° (safe range)
    env->joint_angles[2] = randf(-1.5f, -0.3f);                     // elbow: -86° to -17° (avoid extremes)
    env->joint_angles[3] = randf(-M_PI * 0.4f, M_PI * 0.4f);        // wrist roll: ±72°
    env->joint_angles[4] = randf(-M_PI_2 * 0.4f, M_PI_2 * 0.4f);    // wrist pitch: ±36°
    env->joint_angles[5] = randf(-M_PI_2 * 0.5f, M_PI_2 * 0.5f);    // wrist yaw: ±45°

    for (int i = 0; i < 6; i++) {
        env->joint_vel[i] = 0.0f;
        env->cmd_filt[i] = 0.0f;
    }
    
    env->gripper_state = 0.0f;
    env->gripper_command = 0.0f;
    env->gripper_force = 0.0f;
    env->grasped_object_id = -1;
    env->grasp_stability = 0.0f;
    env->was_grasped_last_step = 0;
    
    if (env->pick_and_place_mode && env->assist_enabled && 
        env->episodes_completed < env->assist_episodes &&
        randf(0.0f, 1.0f) < env->start_grasp_prob) {
        env->grasped_object_id = 0;
        env->objects[0].grasped = true;
        env->gripper_state = 0.8f;
        env->grasp_stability = 1.0f;
    }

    vec3_zero(env->left_finger.pos);
    vec3_zero(env->left_finger.vel);
    vec3_zero(env->left_finger.force);
    env->left_finger.in_contact = false;
    env->left_finger.contact_object_id = -1;
    
    vec3_zero(env->right_finger.pos);
    vec3_zero(env->right_finger.vel);
    vec3_zero(env->right_finger.force);
    env->right_finger.in_contact = false;
    env->right_finger.contact_object_id = -1;

    compute_forward_kinematics(env);

    for (int it=0; it<4; it++) {
        if (env->end_effector[0] >= WORKSPACE_X_MIN && env->end_effector[0] <= WORKSPACE_X_MAX &&
            env->end_effector[1] >= WORKSPACE_Y_MIN && env->end_effector[1] <= WORKSPACE_Y_MAX &&
            env->end_effector[2] >= WORKSPACE_Z_MIN && env->end_effector[2] <= WORKSPACE_Z_MAX) break;
        env->joint_angles[4] = fminf(env->joint_angles[4] + 0.2f, M_PI_2);
        env->joint_angles[2] = fmaxf(env->joint_angles[2] - 0.2f, -2.0f);
        compute_forward_kinematics(env);
    }

    if (env->pick_and_place_mode) {
        init_pick_place_scene(env);
    } else {
        pick_new_target(env);
        env->current_distance = distance3d(env->end_effector, env->target_pos);
        env->prev_distance = env->current_distance;
        env->target_spawn_step = env->episode_steps;
        env->target_touch_awarded = 0;
    }

    if (env->frame_skip <= 0) env->frame_skip = 2;
    if (env->success_distance <= 0.0f) env->success_distance = 0.05f;
    if (env->obs_noise_std < 0.0f) env->obs_noise_std = 0.0f;
    if (env->actuation_noise_std < 0.0f) env->actuation_noise_std = 0.0f;

    if (env->rewards) env->rewards[0] = 0.0f;

    update_observations(env);

    env->camera_initialized = false;
    env->placed_event = 0;
    env->continuous_gripper = 1;
    env->use_unified_clamp = 1;
    env->unified_clamp_min = -5.0f;
    env->unified_clamp_max = 15.0f;
    env->terminate_on_place = 1;

    if (env->episodes_completed == 0) {
        if (env->success_distance_start > 0.0f) env->success_distance = env->success_distance_start;
        if (env->on_gripper_spawn_start > 0.0f) env->on_gripper_spawn_prob = env->on_gripper_spawn_start;
        if (env->success_distance <= 0.0f) env->success_distance = 0.08f;
        if (env->on_gripper_spawn_prob <= 0.0f) env->on_gripper_spawn_prob = 0.02f;
        if (env->success_distance_min <= 0.0f) env->success_distance_min = 0.03f;
        if (env->on_gripper_spawn_min <= 0.0f) env->on_gripper_spawn_min = 0.005f;
        if (env->curriculum_episodes <= 0) env->curriculum_episodes = 200;
        if (env->assist_episodes <= 0) env->assist_episodes = 200;
    }

    env->stagnation_steps = 0;
    env->stagnation_limit = 400;
    env->best_metric = 1e9f;
}

void c_step(RobotArm *env) {
    if (!env) return;

    env->episode_steps++;
    
    if (env->rewards) env->rewards[0] = 0.0f;

    int skip = env->frame_skip > 0 ? env->frame_skip : 1;
    for (int s=0; s<skip; s++) {
        apply_actions(env);
        compute_forward_kinematics(env);

        if (env->pick_and_place_mode) {
            for (int i = 0; i < MAX_OBJECTS; i++) {
                update_object_physics(&env->objects[i], ARM_DT);
            }
        }

        if (env->end_effector[0] < WORKSPACE_X_MIN || env->end_effector[0] > WORKSPACE_X_MAX ||
            env->end_effector[1] < WORKSPACE_Y_MIN || env->end_effector[1] > WORKSPACE_Y_MAX ||
            env->end_effector[2] < WORKSPACE_Z_MIN || env->end_effector[2] > WORKSPACE_Z_MAX) {
            if (env->rewards) env->rewards[0] = -2.0f;
            if (env->terminals) env->terminals[0] = 1;
            env->log.n += 1.0f;
            env->log.episode_length += env->episode_steps;
            env->log.episode_return += env->episode_return_accum;
            env->log.score += env->episode_score_accum;
            float avg_rps = env->episode_steps > 0 ? (env->episode_return_accum / (float)env->episode_steps) : 0.0f;
            float max_rps = env->pick_and_place_mode ? 60.0f : 200.0f;
            float perf_ep = max_rps > 0.0f ? (avg_rps / max_rps) * 100.0f : 0.0f;
            perf_ep = clampf(perf_ep, 0.0f, 100.0f);
            env->log.perf += perf_ep;
            return;
        }
    }

    if (!env->pick_and_place_mode) {
        env->current_distance = distance3d(env->end_effector, env->target_pos);
        float metric = env->current_distance;
        if (metric + 1e-4f < env->best_metric) { env->best_metric = metric; env->stagnation_steps = 0; }
        else env->stagnation_steps++;
    } else {
        float metric = 1e9f;
        if (env->grasped_object_id < 0) {
            for (int i = 0; i < MAX_OBJECTS; i++) if (!env->objects[i].in_basket && env->objects[i].type == env->target_type) {
                float d = distance3d(env->end_effector, env->objects[i].physics.pos);
                if (d < metric) metric = d;
            }
        } else {
            ManipObject* o = &env->objects[env->grasped_object_id];
            int bidx = o->target_basket;
            metric = distance3d(o->physics.pos, env->baskets[bidx].pos);
        }
        if (metric + 1e-4f < env->best_metric) { env->best_metric = metric; env->stagnation_steps = 0; }
        else env->stagnation_steps++;
    }

    env->was_grasped_last_step = (env->grasped_object_id >= 0);

    env->placed_event = 0;
    float r = compute_reward(env);
    if (env->rewards) env->rewards[0] = r;
    env->episode_return_accum += r;
    
    if (!env->pick_and_place_mode) {
        env->prev_distance = env->current_distance;
    }
    if (!env->pick_and_place_mode) {
        float current_dist = env->current_distance;
        if (current_dist < env->success_distance) {
            if (env->rewards) env->rewards[0] += 50.0f;
            env->episode_score_accum += 1.0f;
            pick_new_target(env);
            env->current_distance = distance3d(env->end_effector, env->target_pos);
            env->prev_distance = env->current_distance;
            env->target_spawn_step = env->episode_steps;
            env->target_touch_awarded = 0;
        } else if (env->episode_steps % 256 == 0) {
            pick_new_target(env);
            env->current_distance = distance3d(env->end_effector, env->target_pos);
            env->target_spawn_step = env->episode_steps;
            env->target_touch_awarded = 0;
        }
    }

    if (env->stagnation_steps > env->stagnation_limit) {
        if (!env->pick_and_place_mode) {
            pick_new_target(env);
            env->current_distance = distance3d(env->end_effector, env->target_pos);
            env->prev_distance = env->current_distance;
        } else {
            env->target_type = (ObjectType)(rand() % 3);
            env->current_target_object = -1;
            for (int i = 0; i < MAX_OBJECTS; i++) {
                if (!env->objects[i].in_basket && env->objects[i].type == env->target_type) { env->current_target_object = i; break; }
            }
        }
        env->stagnation_steps = 0;
        env->best_metric = 1e9f;
    }

    if ((env->terminate_on_place && env->placed_event) || (env->episode_steps >= env->max_steps)) {
        if (env->terminals) env->terminals[0] = 1;
        env->log.n += 1.0f;
        env->log.episode_length += env->episode_steps;
        env->log.episode_return += env->episode_return_accum;
        env->log.score += env->episode_score_accum;
        float avg_rps = env->episode_steps > 0 ? (env->episode_return_accum / (float)env->episode_steps) : 0.0f;
        float max_rps = env->pick_and_place_mode ? 15.0f : 20.0f;
        float perf_ep = max_rps > 0.0f ? (avg_rps / max_rps) * 100.0f : 0.0f;
        perf_ep = clampf(perf_ep, 0.0f, 100.0f);
        env->log.perf += perf_ep;
        env->episodes_completed += 1;
        if (env->curriculum_episodes > 0) {
            float t = (float)env->episodes_completed / (float)env->curriculum_episodes;
            t = clampf(t, 0.0f, 1.0f);
            env->success_distance = env->success_distance_start * (1.0f - t) + env->success_distance_min * t;
            env->on_gripper_spawn_prob = env->on_gripper_spawn_start * (1.0f - t) + env->on_gripper_spawn_min * t;
        }
        return;
    }

    update_observations(env);
}

static void handle_camera(RobotArm *env) {
    Vector2 mouse = GetMousePosition();
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        env->is_dragging = true;
        env->last_mouse_pos = mouse;
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) env->is_dragging = false;

    if (env->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 d = {mouse.x - env->last_mouse_pos.x, mouse.y - env->last_mouse_pos.y};
        env->camera_azimuth   -= 0.005f * d.x;
        env->camera_elevation += 0.005f * d.y;
        env->camera_elevation = clampf(env->camera_elevation, -M_PI_2, M_PI_2);
        env->last_mouse_pos = mouse;
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) env->camera_distance = clampf(env->camera_distance - 0.5f*wheel, 1.0f, 10.0f);

    float r = env->camera_distance;
    float az = env->camera_azimuth;
    float el = env->camera_elevation;
    env->camera.position = (Vector3){r*cosf(el)*cosf(az), r*cosf(el)*sinf(az), r*sinf(el)};
    env->camera.target = (Vector3){0,0,0.3f};
}

void c_render(RobotArm *env) {
    if (!env) return;

    compute_forward_kinematics(env);

    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
        InitWindow(1200, 800, "PufferLib Robot Arm");
        SetTargetFPS(60);
    }

    if (!env->camera_initialized) {
        env->camera_distance = 3.5f;
        env->camera_azimuth = -0.8f;
        env->camera_elevation = 0.35f;
        env->camera.up = (Vector3){0,0,1};
        env->camera.fovy = 45.0f;
        env->camera.projection = CAMERA_PERSPECTIVE;
        env->camera_initialized = true;
    }

    if (IsKeyDown(KEY_ESCAPE)) return;

    handle_camera(env);

    BeginDrawing();
    ClearBackground(RAYWHITE);

    BeginMode3D(env->camera);

    DrawPlane((Vector3){0,0,0}, (Vector2){4.0f, 4.0f}, FLOOR_COLOR);
    Color gridCol = (Color){200,205,210,120};
    for (float i=-2.0f; i<=2.0f; i+=0.2f) {
        DrawLine3D((Vector3){-2.0f, i, 0.001f}, (Vector3){2.0f, i, 0.001f}, gridCol);
        DrawLine3D((Vector3){i, -2.0f, 0.001f}, (Vector3){i, 2.0f, 0.001f}, gridCol);
    }

    Vector3 base = {0,0,0.2f};
    Vector3 ee   = {env->end_effector[0], env->end_effector[1], env->end_effector[2]};
    float cosb = cosf(env->joint_angles[0]);
    float sinb = sinf(env->joint_angles[0]);
    float L1 = env->link1_length > 0 ? env->link1_length : ARM_LINK1_LENGTH;
    float L2 = env->link2_length > 0 ? env->link2_length : ARM_LINK2_LENGTH;
    Vector3 shoulder = {L1*cosf(env->joint_angles[1])*cosb, L1*cosf(env->joint_angles[1])*sinb, 0.2f + L1*sinf(env->joint_angles[1])};
    Vector3 elbow = { (L1*cosf(env->joint_angles[1]) + L2*cosf(env->joint_angles[1]+env->joint_angles[2]))*cosb,
                      (L1*cosf(env->joint_angles[1]) + L2*cosf(env->joint_angles[1]+env->joint_angles[2]))*sinb,
                      0.2f + L1*sinf(env->joint_angles[1]) + L2*sinf(env->joint_angles[1]+env->joint_angles[2]) };
    DrawCylinder((Vector3){0,0,0.05f}, 0.10f, 0.10f, 0.10f, 20, METAL_DARK);
    DrawCylinder((Vector3){0,0,0.12f}, 0.11f, 0.11f, 0.02f, 20, METAL_LIGHT);
    DrawCylinderEx(base, shoulder, 0.035f, 0.030f, 20, ORANGE_MAIN);
    DrawCylinderEx(shoulder, elbow, 0.030f, 0.024f, 20, ORANGE_MAIN);
    DrawCylinderEx(elbow, ee, 0.024f, 0.018f, 20, ORANGE_MAIN);
    DrawSphere(base, 0.055f, METAL_MEDIUM);
    DrawSphere(shoulder, 0.045f, METAL_MEDIUM);
    DrawSphere(elbow, 0.040f, METAL_MEDIUM);
    DrawSphere(ee, 0.030f, ORANGE_LIGHT);
    float open = 0.02f;
    Vector3 g1 = {ee.x - open, ee.y, ee.z - 0.03f};
    Vector3 g2 = {ee.x + open, ee.y, ee.z - 0.03f};
    DrawCylinderEx(ee, g1, 0.006f, 0.006f, 8, METAL_LIGHT);
    DrawCylinderEx(ee, g2, 0.006f, 0.006f, 8, METAL_LIGHT);

    if (!env->pick_and_place_mode) {
        Vector3 t = {env->target_pos[0], env->target_pos[1], env->target_pos[2]};
        float d = env->current_distance;
        if (d <= 0.0f || isnan(d)) {
            d = distance3d(env->end_effector, env->target_pos);
        }
        Color tcol = (Color){ 64,160,255,255};
        DrawSphere(t, 0.065f, tcol);
        DrawSphereWires(t, 0.085f, 10, 10, (Color){tcol.r,tcol.g,tcol.b,160});
    } else {
        float best_d = 1e9f;
        int   best_i = -1;

        for (int i = 0; i < MAX_BASKETS; i++) {
            Basket *b = &env->baskets[i];
            Color bc = (b->type == BASKET_RED)   ? CLR_RED :
                       (b->type == BASKET_BLUE)  ? CLR_BLUE :
                       (b->type == BASKET_GREEN) ? CLR_GREEN : METAL_LIGHT;
            Vector3 bp = {b->pos[0], b->pos[1], b->pos[2]};
            DrawCylinder(bp, BASKET_SIZE*0.5f, BASKET_SIZE*0.5f, 0.02f, 20, (Color){bc.r,bc.g,bc.b,120});
            DrawCylinderWires(bp, BASKET_SIZE*0.5f, BASKET_SIZE*0.5f, 0.021f, 20, bc);
        }
        for (int i = 0; i < MAX_OBJECTS; i++) {
            ManipObject *o = &env->objects[i];
            Color oc = (o->type == OBJ_RED)   ? CLR_RED :
                       (o->type == OBJ_BLUE)  ? CLR_BLUE :
                       (o->type == OBJ_GREEN) ? CLR_GREEN : CLR_YEL;
            Vector3 op = {o->physics.pos[0], o->physics.pos[1], o->physics.pos[2]};
            DrawCube(op, OBJECT_SIZE, OBJECT_SIZE, OBJECT_SIZE, oc);
            DrawCubeWires(op, OBJECT_SIZE*1.01f, OBJECT_SIZE*1.01f, OBJECT_SIZE*1.01f, (Color){oc.r,oc.g,oc.b,180});
            float d = distance3d(env->end_effector, o->physics.pos);
            if (d < best_d && !o->in_basket) { best_d = d; best_i = i; }
        }
        if (best_i >= 0) {
            ManipObject *closest = &env->objects[best_i];
            Vector3 cp = {closest->physics.pos[0], closest->physics.pos[1], closest->physics.pos[2]};
            DrawSphereWires(cp, OBJECT_SIZE*0.8f, 12, 12, (Color){255,255,255,200});
        }
    }
    Vector3 center = {(base.x + ee.x)*0.5f, (base.y + ee.y)*0.5f, (base.z + ee.z)*0.5f};
    env->camera.target = center;

    EndMode3D();

    DrawText("Robot Arm", 20, 20, 20, BLACK);
    if (!env->pick_and_place_mode) {
        DrawText(TextFormat("Distance: %.2f  Target:(%.2f, %.2f, %.2f)", env->current_distance,
                           env->target_pos[0], env->target_pos[1], env->target_pos[2]), 20, 50, 16, BLACK);
    } else {
        float best_d = 1e9f;
        for (int i = 0; i < MAX_OBJECTS; i++) {
            if (!env->objects[i].in_basket) {
                float d = distance3d(env->end_effector, env->objects[i].physics.pos);
                if (d < best_d) best_d = d;
            }
        }
        if (best_d < 1e8f) {
            DrawText(TextFormat("Closest Obj Dist: %.2f", best_d), 20, 50, 16, BLACK);
            DrawText(TextFormat("Target Type: %s", env->target_type == OBJ_RED ? "RED" :
                               env->target_type == OBJ_BLUE ? "BLUE" : "GREEN"), 20, 70, 16, BLACK);
        }
    }
    DrawText(TextFormat("Score: %.2f", env->log.score), 20, 80, 16, BLACK);

    // Joint angle indicator (degrees)
    DrawText("Joint Angles (deg)", 20, 100, 16, BLACK);
    DrawText(TextFormat("J0: %5.1f  J1: %5.1f  J2: %5.1f",
                       env->joint_angles[0] * (180.0f / M_PI),
                       env->joint_angles[1] * (180.0f / M_PI),
                       env->joint_angles[2] * (180.0f / M_PI)),
             20, 120, 16, BLACK);
    DrawText(TextFormat("J3: %5.1f  J4: %5.1f  J5: %5.1f",
                       env->joint_angles[3] * (180.0f / M_PI),
                       env->joint_angles[4] * (180.0f / M_PI),
                       env->joint_angles[5] * (180.0f / M_PI)),
             20, 140, 16, BLACK);

    EndDrawing();
}

void c_close(RobotArm *env) {
    if (!env) return;
    if (IsWindowReady()) CloseWindow();
}