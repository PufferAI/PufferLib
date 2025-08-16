#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
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

static const float AIR_DAMPING_LOG = 0.0f;
static const float FINGER_HARD_RADIUS = 0.020f;
static const float GRIPPER_TANGENTIAL_FRICTION = 1.2f;
static const float WALL_RESTITUTION = 0.3f;
static const float WALL_FRICTION = 0.6f;
static const float HOLD_SPRING_K = 400.0f;
static const float HOLD_DAMPING_C = 20.0f;
    static const float MAGNET_FORCE_CAP = 5.0f;
    static const float MAGNET_BASE_GAIN = 0.0f;

static inline float get_air_damping_log() {
    static int initialized = 0;
    static float value = 0.0f;
    if (!initialized) {
        value = logf(AIR_DAMPING);
        initialized = 1;
    }
    return value;
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

typedef struct {
    float pos[3];
    float vel[3];
} PhysicsState;


static void compute_physics_derivative(const ManipObject* obj, const PhysicsState* state, PhysicsState* derivative, float dt_unused) {

    vec3_copy(derivative->pos, state->vel);
    

    if (obj->mass > 0.0f) {
        derivative->vel[0] = obj->physics.force[0] / obj->mass;
        derivative->vel[1] = obj->physics.force[1] / obj->mass;
        derivative->vel[2] = obj->physics.force[2] / obj->mass;
    } else {
        vec3_zero(derivative->vel);
    }
    
    float damping_factor = get_air_damping_log();
    derivative->vel[0] += state->vel[0] * damping_factor;
    derivative->vel[1] += state->vel[1] * damping_factor;
    derivative->vel[2] += state->vel[2] * damping_factor;
}

#if !ARM_USE_EULER
static void rk4_integrate_physics(ManipObject* obj, float dt) {
    if (obj->in_basket) return;
    
    PhysicsBody* body = &obj->physics;
    
    PhysicsState current_state, k1, k2, k3, k4, temp_state;
    
    vec3_copy(current_state.pos, body->pos);
    vec3_copy(current_state.vel, body->vel);
    
    compute_physics_derivative(obj, &current_state, &k1, 0.0f);

    for (int i = 0; i < 3; i++) {
        temp_state.pos[i] = current_state.pos[i] + k1.pos[i] * dt * 0.5f;
        temp_state.vel[i] = current_state.vel[i] + k1.vel[i] * dt * 0.5f;
    }
    compute_physics_derivative(obj, &temp_state, &k2, dt * 0.5f);

    for (int i = 0; i < 3; i++) {
        temp_state.pos[i] = current_state.pos[i] + k2.pos[i] * dt * 0.5f;
        temp_state.vel[i] = current_state.vel[i] + k2.vel[i] * dt * 0.5f;
    }
    compute_physics_derivative(obj, &temp_state, &k3, dt * 0.5f);

    for (int i = 0; i < 3; i++) {
        temp_state.pos[i] = current_state.pos[i] + k3.pos[i] * dt;
        temp_state.vel[i] = current_state.vel[i] + k3.vel[i] * dt;
    }
    compute_physics_derivative(obj, &temp_state, &k4, dt);

    for (int i = 0; i < 3; i++) {
        body->pos[i] += (k1.pos[i] + 2.0f*k2.pos[i] + 2.0f*k3.pos[i] + k4.pos[i]) * dt / 6.0f;
        body->vel[i] += (k1.vel[i] + 2.0f*k2.vel[i] + 2.0f*k3.vel[i] + k4.vel[i]) * dt / 6.0f;
    }
}
#endif

static void euler_integrate_physics(ManipObject* obj, float dt) {
    if (obj->in_basket) return;
    PhysicsBody* body = &obj->physics;
    if (obj->mass > 0.0f) {
        body->vel[0] += (body->force[0] / obj->mass + get_air_damping_log() * body->vel[0]) * dt;
        body->vel[1] += (body->force[1] / obj->mass + get_air_damping_log() * body->vel[1]) * dt;
        body->vel[2] += (body->force[2] / obj->mass + get_air_damping_log() * body->vel[2]) * dt;
    }
    body->pos[0] += body->vel[0] * dt;
    body->pos[1] += body->vel[1] * dt;
    body->pos[2] += body->vel[2] * dt;
}


static void update_object_physics(ManipObject* obj, float dt) {
    if (obj->in_basket) return;
    if (obj->grasped) return;
    PhysicsBody* body = &obj->physics;
    if (body->on_surface) {
        float v2 = body->vel[0]*body->vel[0] + body->vel[1]*body->vel[1] + body->vel[2]*body->vel[2];
        if (v2 < 1e-6f) {
            float min_z = TABLE_HEIGHT + obj->size[2] * 0.5f;
            if (body->pos[2] < min_z) body->pos[2] = min_z;
            vec3_zero(body->vel);
            vec3_zero(body->force);
            vec3_zero(body->torque);
            body->in_contact = true;
            body->contact_time += dt;
            return;
        }
    }
    float half_x = obj->size[0]*0.5f;
    float half_y = obj->size[1]*0.5f;
    float half_z = obj->size[2]*0.5f;
    float wall_mu = (1.0f - WALL_FRICTION * dt);

    if (!obj->grasped) {
        body->force[2] += obj->mass * GRAVITY;
    }
    body->on_surface = false;

#if ARM_USE_EULER
    euler_integrate_physics(obj, dt);
#else
    rk4_integrate_physics(obj, dt);
#endif

    vec3_zero(body->force);
    vec3_zero(body->torque);
    
    if (body->in_contact) {
        body->contact_time += dt;
    } else {
        body->contact_time = 0.0f;
    }

    

    if (body->pos[0] < WORKSPACE_X_MIN + half_x) {
        body->pos[0] = WORKSPACE_X_MIN + half_x;
        if (body->vel[0] < 0.0f) body->vel[0] *= -WALL_RESTITUTION;
        body->vel[1] *= wall_mu;
        body->vel[2] *= wall_mu;
    }
    if (body->pos[0] > WORKSPACE_X_MAX - half_x) {
        body->pos[0] = WORKSPACE_X_MAX - half_x;
        if (body->vel[0] > 0.0f) body->vel[0] *= -WALL_RESTITUTION;
        body->vel[1] *= wall_mu;
        body->vel[2] *= wall_mu;
    }
    if (body->pos[1] < WORKSPACE_Y_MIN + half_y) {
        body->pos[1] = WORKSPACE_Y_MIN + half_y;
        if (body->vel[1] < 0.0f) body->vel[1] *= -WALL_RESTITUTION;
        body->vel[0] *= wall_mu;
        body->vel[2] *= wall_mu;
    }
    if (body->pos[1] > WORKSPACE_Y_MAX - half_y) {
        body->pos[1] = WORKSPACE_Y_MAX - half_y;
        if (body->vel[1] > 0.0f) body->vel[1] *= -WALL_RESTITUTION;
        body->vel[0] *= wall_mu;
        body->vel[2] *= wall_mu;
    }
    if (body->pos[2] > WORKSPACE_Z_MAX - half_z) {
        body->pos[2] = WORKSPACE_Z_MAX - half_z;
        if (body->vel[2] > 0.0f) body->vel[2] *= -WALL_RESTITUTION;
        body->vel[0] *= wall_mu;
        body->vel[1] *= wall_mu;
    }
    if (body->pos[2] < TABLE_HEIGHT + half_z) {
        body->pos[2] = TABLE_HEIGHT + half_z;
        if (body->vel[2] < 0.0f) body->vel[2] *= -obj->restitution;
        float horizontal_speed = safe_sqrt(body->vel[0]*body->vel[0] + body->vel[1]*body->vel[1]);
        if (horizontal_speed > 1e-3f) {
            float friction_force = obj->friction * obj->mass * fabsf(GRAVITY);
            float friction_decel = friction_force / obj->mass;
            float new_speed = fmaxf(0.0f, horizontal_speed - friction_decel * dt);
            float scale = (horizontal_speed > 0.0f) ? (new_speed / horizontal_speed) : 0.0f;
            body->vel[0] *= scale;
            body->vel[1] *= scale;
        }
        body->on_surface = true;
    }
}

static void resolve_object_collisions(RobotArm *env, float dt) {
    for (int i = 0; i < MAX_OBJECTS; i++) {
        ManipObject *oi = &env->objects[i];
        if (oi->in_basket) continue;
        if (env->grasped_object_id == i) continue;
        float ri = oi->size[0] * 0.5f;
        for (int j = i + 1; j < MAX_OBJECTS; j++) {
            ManipObject *oj = &env->objects[j];
            if (oj->in_basket) continue;
            if (env->grasped_object_id == j) continue;
            if (oi->physics.on_surface && oj->physics.on_surface) {
                float v2i = oi->physics.vel[0]*oi->physics.vel[0] + oi->physics.vel[1]*oi->physics.vel[1] + oi->physics.vel[2]*oi->physics.vel[2];
                float v2j = oj->physics.vel[0]*oj->physics.vel[0] + oj->physics.vel[1]*oj->physics.vel[1] + oj->physics.vel[2]*oj->physics.vel[2];
                if (v2i < 1e-6f && v2j < 1e-6f) continue;
            }
            float rj = oj->size[0] * 0.5f;

            float dx = oj->physics.pos[0] - oi->physics.pos[0];
            float dy = oj->physics.pos[1] - oi->physics.pos[1];
            float dz = oj->physics.pos[2] - oi->physics.pos[2];
            float dist2 = dx*dx + dy*dy + dz*dz;
            float minDist = ri + rj;
            if (dist2 >= minDist * minDist) continue;

            float dist = safe_sqrt(dist2);
            float nx, ny, nz;
            if (dist > 1e-6f) {
                nx = dx / dist; ny = dy / dist; nz = dz / dist;
            } else {

                nx = 1.0f; ny = 0.0f; nz = 0.0f;
                dist = 0.0f;
            }

            float penetration = minDist - dist;
            if (penetration <= 0.0f) continue;
            bool i_grasped = (env->grasped_object_id == i);
            bool j_grasped = (env->grasped_object_id == j);
            float mi = fmaxf(oi->mass, 1e-6f);
            float mj = fmaxf(oj->mass, 1e-6f);

            if (i_grasped && !j_grasped) {
                oj->physics.pos[0] += nx * penetration;
                oj->physics.pos[1] += ny * penetration;
                oj->physics.pos[2] += nz * penetration;
            } else if (!i_grasped && j_grasped) {
                oi->physics.pos[0] -= nx * penetration;
                oi->physics.pos[1] -= ny * penetration;
                oi->physics.pos[2] -= nz * penetration;
            } else {

                float total_m = mi + mj;
                float move_i = penetration * (mj / total_m);
                float move_j = penetration * (mi / total_m);
                oi->physics.pos[0] -= nx * move_i;
                oi->physics.pos[1] -= ny * move_i;
                oi->physics.pos[2] -= nz * move_i;
                oj->physics.pos[0] += nx * move_j;
                oj->physics.pos[1] += ny * move_j;
                oj->physics.pos[2] += nz * move_j;
            }

            float vi_n = oi->physics.vel[0]*nx + oi->physics.vel[1]*ny + oi->physics.vel[2]*nz;
            float vj_n = oj->physics.vel[0]*nx + oj->physics.vel[1]*ny + oj->physics.vel[2]*nz;
            float rel_n = vj_n - vi_n;
            if (rel_n < 0.0f) {
                float e = fminf(oi->restitution, oj->restitution);
                float jimp = -(1.0f + e) * rel_n / (1.0f/mi + 1.0f/mj);
                if (!i_grasped) {
                    oi->physics.vel[0] -= (jimp / mi) * nx;
                    oi->physics.vel[1] -= (jimp / mi) * ny;
                    oi->physics.vel[2] -= (jimp / mi) * nz;
                }
                if (!j_grasped) {
                    oj->physics.vel[0] += (jimp / mj) * nx;
                    oj->physics.vel[1] += (jimp / mj) * ny;
                    oj->physics.vel[2] += (jimp / mj) * nz;
                }
            }
        }
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

static const float JOINT_LO[6] = {
    -M_PI * 0.75f,
    -M_PI_2 * 0.6f,
    -2.0f,
    -M_PI * 0.6f,
    -M_PI_2 * 0.6f,
    -M_PI_2 * 0.7f
};
static const float JOINT_HI[6] = {
    M_PI * 0.75f,
    M_PI_2 * 0.9f,
    -0.1f,
    M_PI * 0.6f,
    M_PI_2 * 0.6f,
    M_PI_2 * 0.7f
};

static float compute_joint_limit_cost(RobotArm *env) {
    float cost = 0.0f;
    for (int i = 0; i < 6; i++) {
        float margin_lo = env->joint_angles[i] - JOINT_LO[i];
        float margin_hi = JOINT_HI[i] - env->joint_angles[i];
        float margin = fminf(margin_lo, margin_hi);
        float proximity = fmaxf(0.0f, 0.10f - margin) / 0.10f;
        cost += proximity * proximity;
    }
    return cost;
}

static void compute_forward_kinematics(RobotArm *env) {
    update_trig_cache(env);

    const float cos_base = env->cached_cos[0];
    const float sin_base = env->cached_sin[0];
    const float cos_sh = env->cached_cos[1];
    const float sin_sh = env->cached_sin[1];
    const float cos_wp = env->cached_cos[4];
    const float sin_wp = env->cached_sin[4];
    const float cos_e = env->cached_cos[2];
    const float sin_e = env->cached_sin[2];
    const float cos_se = cos_sh * cos_e - sin_sh * sin_e;
    const float sin_se = sin_sh * cos_e + cos_sh * sin_e;
    const float L1 = env->link1_length > 0 ? env->link1_length : ARM_LINK1_LENGTH;
    const float L2 = env->link2_length > 0 ? env->link2_length : ARM_LINK2_LENGTH;
    const float L3 = env->link3_length > 0 ? env->link3_length : ARM_LINK3_LENGTH;
    const float base_h = 0.2f;
    const float reach = L1*cos_sh + L2*cos_se;
    const float height = L1*sin_sh + L2*sin_se;
    const float wrist_xy = L3 * cos_wp;
    const float wrist_z  = L3 * sin_wp;
    const float cos_yaw = env->cached_cos[5];
    const float sin_yaw = env->cached_sin[5];
    const float yaw_c = cos_base * cos_yaw - sin_base * sin_yaw;
    const float yaw_s = sin_base * cos_yaw + cos_base * sin_yaw;
    env->end_effector[0] = reach * cos_base + wrist_xy * yaw_c;
    env->end_effector[1] = reach * sin_base + wrist_xy * yaw_s;
    env->end_effector[2] = base_h + height + wrist_z;
    env->end_effector_orient[0] = env->joint_angles[3];
    env->end_effector_orient[1] = env->joint_angles[4];
    env->end_effector_orient[2] = env->joint_angles[0] + env->joint_angles[5];
}

static inline void find_nearest_target(RobotArm *env) {
    int idx = -1;
    float best_d2 = 1e12f;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (env->objects[i].in_basket || env->objects[i].type != env->target_type) continue;
        float dx = env->end_effector[0] - env->objects[i].physics.pos[0];
        float dy = env->end_effector[1] - env->objects[i].physics.pos[1];
        float dz = env->end_effector[2] - env->objects[i].physics.pos[2];
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_d2) { best_d2 = d2; idx = i; }
    }
    env->nearest_target_idx = idx;
    env->nearest_target_d2 = best_d2;
}

static void update_observations(RobotArm *env) {
    float *obs = env->observations;
    if (!obs) return;

    int obs_idx = 0;
    
    {
        
        for (int i = 0; i < 6; i++) {
            obs[obs_idx++] = clampf(env->joint_angles[i] / M_PI, -1.0f, 1.0f);
        }
        obs[obs_idx++] = env->gripper_state;
        obs[obs_idx++] = env->gripper_force / GRIPPER_MAX_FORCE;
        
    int closest_obj = env->nearest_target_idx;
        
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

        if (env->extended_observation) {
            for (int i = 0; i < 6; i++) {
                obs[obs_idx++] = clampf(env->joint_vel[i] / 3.0f, -1.0f, 1.0f);
            }
            obs[obs_idx++] = env->left_finger.in_contact ? 1.0f : 0.0f;
            obs[obs_idx++] = env->right_finger.in_contact ? 1.0f : 0.0f;
        }
        
    }
}

static inline float finalize_reward(RobotArm *env, float r, float clamp_min, float clamp_max) {
    float scale = (env->reward_scale > 0.0f ? env->reward_scale : 1.0f);
    float out = r * scale;
    if (env->use_unified_clamp) return clampf(out, env->unified_clamp_min, env->unified_clamp_max);
    return clampf(out, clamp_min, clamp_max);
}

static float compute_reward(RobotArm *env) {
    float reward = 0.0f;
    const float base_step = 0.02f;
    reward += base_step;
    if (env->release_event && env->recent_release_object_id >= 0) {
        int oi = env->recent_release_object_id;
        ManipObject *o = &env->objects[oi];
        env->release_event = 0;
        if (!o->in_basket) {
            int bidx = o->target_basket;
            Basket *b = &env->baskets[bidx];
            float dx = o->physics.pos[0] - b->pos[0];
            float dy = o->physics.pos[1] - b->pos[1];
            float dxy = safe_sqrt(dx*dx + dy*dy);
            float rad_ok = BASKET_SIZE * 0.7f;
            if (dxy > rad_ok) {
                float excess = dxy - rad_ok;
                reward -= fminf(10.0f, 40.0f * excess);
            }
        }
    }

    int closest_obj = env->nearest_target_idx;
    float min_obj_dist = (closest_obj >= 0) ? safe_sqrt(env->nearest_target_d2) : 1e9f;

    if (env->grasped_object_id < 0) {
        for (int oi = 0; oi < MAX_OBJECTS; oi++) {
            ManipObject *o = &env->objects[oi];
            if (o->in_basket || o->grasped) continue;
            int bidx = o->target_basket;
            Basket *b = &env->baskets[bidx];
            float dx = o->physics.pos[0] - b->pos[0];
            float dy = o->physics.pos[1] - b->pos[1];
            float r2 = dx*dx + dy*dy;
            float rad = BASKET_SIZE * 0.55f;
            float topZ = b->pos[2] + 0.5f * BASKET_SIZE;
            bool inside_xy = (r2 < rad*rad);
            bool below_rim = (o->physics.pos[2] <= topZ + 0.02f);
            bool settled = (o->physics.on_surface || fabsf(o->physics.vel[2]) < 0.05f);
            if (inside_xy && below_rim && settled) {
                o->in_basket = true;
                float frac_remaining = 0.0f;
                if (env->max_steps > 0) {
                    float t = (float)env->episode_steps / (float)env->max_steps;
                    frac_remaining = clampf(1.0f - t, 0.0f, 1.0f);
                }
                float time_bonus = 80.0f * frac_remaining;
                reward += 20.0f + time_bonus;
                env->episode_score_accum += 1.0f;
                env->episode_place_count += 1.0f;
                env->log.place_success_rate += 1.0f;
                b->collected_count += 1.0f;
                env->placed_event = 1;
                env->best_place_dist = 1e9f;
                int disabled_idx = bidx;
                for (int attempt = 0; attempt < 3; attempt++) {
                    ObjectType new_type = (ObjectType)(rand() % 3);
                    int basket_idx = -1;
                    for (int bi = 0; bi < MAX_BASKETS; bi++) if (env->baskets[bi].type == (BasketType)new_type) { basket_idx = bi; break; }
                    if (basket_idx >= 0 && basket_idx != disabled_idx) {
                        int found = -1;
                        for (int oj = 0; oj < MAX_OBJECTS; oj++) if (!env->objects[oj].in_basket && env->objects[oj].type == new_type) { found = oj; break; }
                        if (found >= 0) { env->target_type = new_type; env->current_target_object = found; break; }
                    }
                }
                return finalize_reward(env, reward, 0.0f, 200.0f);
            }
        }
    }

    if (env->grasped_object_id < 0 && closest_obj >= 0) {
        float current_dist = min_obj_dist;
        float prev_dist = env->prev_distance;
        float movement_delta = prev_dist - current_dist;

        if (movement_delta > 0.0f) {
            reward += 4.0f * movement_delta;
            reward += 1.5f * movement_delta;
        }
        reward += fmaxf(0.0f, 0.30f - current_dist) * 3.0f;
        if (current_dist < 0.06f) reward += 0.8f;

        if (env->grasp_event) { reward += 2.5f; env->grasp_event = 0; }
        env->prev_distance = current_dist;
        return finalize_reward(env, reward, 0.0f, 30.0f);
    }

    if (env->grasped_object_id >= 0) {
        ManipObject* o = &env->objects[env->grasped_object_id];
        int bidx = o->target_basket;
        float dxb = o->physics.pos[0] - env->baskets[bidx].pos[0];
        float dyb = o->physics.pos[1] - env->baskets[bidx].pos[1];
        float dzb = o->physics.pos[2] - env->baskets[bidx].pos[2];
        float d2b = dxb*dxb + dyb*dyb + dzb*dzb;
        float current_dist = safe_sqrt(d2b);
        float prev_dist = env->prev_distance;
        float movement_delta = prev_dist - current_dist;

        if (current_dist + 1e-5f < env->best_place_dist) env->best_place_dist = current_dist;
        if (movement_delta > 0.0f) {
            reward += 10.0f * movement_delta;
            reward += 3.0f * movement_delta;
        }
        reward += fmaxf(0.0f, (BASKET_SIZE * 1.0f) - current_dist) * 8.0f;
        float lift = o->physics.pos[2] - (TABLE_HEIGHT + OBJECT_SIZE * 0.5f);
        if (lift > 0.03f) reward += fminf(2.0f, lift * 15.0f);

        float near_thresh2 = (BASKET_SIZE * 0.7f) * (BASKET_SIZE * 0.7f);
        bool near_basket = d2b < near_thresh2;
        bool opened = env->gripper_state < 0.3f;
        if (near_basket && opened) {
            o->in_basket = true;
            env->grasped_object_id = -1;
                float frac_remaining = 0.0f;
                if (env->max_steps > 0) {
                    float t = (float)env->episode_steps / (float)env->max_steps;
                    frac_remaining = clampf(1.0f - t, 0.0f, 1.0f);
                }
                float time_bonus = 80.0f * frac_remaining;
                reward += 20.0f + time_bonus;
            env->episode_score_accum += 1.0f;
            env->episode_place_count += 1.0f;
            env->log.place_success_rate += 1.0f;
            env->baskets[bidx].collected_count += 1.0f;
            env->placed_event = 1;
            env->best_place_dist = 1e9f;
            int disabled_idx = bidx;
            for (int attempt = 0; attempt < 3; attempt++) {
                ObjectType new_type = (ObjectType)(rand() % 3);
                int basket_idx = -1;
                for (int bi = 0; bi < MAX_BASKETS; bi++) if (env->baskets[bi].type == (BasketType)new_type) { basket_idx = bi; break; }
                if (basket_idx >= 0 && basket_idx != disabled_idx) {
                    int found = -1;
                    for (int oi = 0; oi < MAX_OBJECTS; oi++) if (!env->objects[oi].in_basket && env->objects[oi].type == new_type) { found = oi; break; }
                    if (found >= 0) { env->target_type = new_type; env->current_target_object = found; break; }
                }
            }
        }
        env->prev_distance = current_dist;
        return finalize_reward(env, reward, 0.0f, 30.0f);
    }

    return finalize_reward(env, reward, 0.0f, 30.0f);
}


static void update_gripper_fingers(RobotArm *env) {
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
    env->gripper_force = 0.0f;
    env->grasp_event = 0;

    if (env->grasped_object_id >= 0) return;
    int ci = env->nearest_target_idx;
    if (ci < 0) return;

    ManipObject* obj = &env->objects[ci];
    float center[3] = { env->end_effector[0], env->end_effector[1], env->end_effector[2] - 0.02f };
    float to_center[3] = { center[0] - obj->physics.pos[0], center[1] - obj->physics.pos[1], center[2] - obj->physics.pos[2] };
    float dist = vec3_length(to_center);
    if (dist > 1e-6f) vec3_scale(to_center, to_center, 1.0f / fmaxf(dist, 1e-6f));
    float k = HOLD_SPRING_K * 1.00f;
    float c = HOLD_DAMPING_C * 0.5f;
    float gain = 1.0f - fmaxf(0.0f, fminf(1.0f, env->gripper_state));
    float f_mag = gain * fminf(MAGNET_FORCE_CAP, k * dist);
    obj->physics.force[0] += to_center[0] * f_mag - c * obj->physics.vel[0] * 0.25f;
    obj->physics.force[1] += to_center[1] * f_mag - c * obj->physics.vel[1] * 0.25f;
    obj->physics.force[2] += to_center[2] * f_mag - c * obj->physics.vel[2] * 0.25f;
    env->gripper_force = f_mag;

    if (dist < 0.12f && env->gripper_state > 0.55f && !obj->grasped) {
        obj->grasped = true;
        env->grasped_object_id = ci;
        env->grasp_stability = 1.0f;
        env->grasp_event = 1;
    }
}

static void update_gripper(RobotArm *env, float dt) {

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
        env->gripper_state = fminf(env->gripper_state + gripper_speed * dt, env->gripper_command);
    } else if (env->gripper_command < env->gripper_state) {
        env->gripper_state = fmaxf(env->gripper_state - gripper_speed * dt, env->gripper_command);
    }

    update_gripper_fingers(env);

    apply_gripper_forces(env);

    if (env->grasped_object_id >= 0) {
        ManipObject* grasped_obj = &env->objects[env->grasped_object_id];

        if (env->gripper_state < 0.15f) {
            grasped_obj->grasped = false;
            env->grasped_object_id = -1;
            env->grasp_stability = 0.0f;
            // mark release event for shaping
            env->release_event = 1;
            env->recent_release_object_id = (int)(grasped_obj - env->objects);
            float min_z = TABLE_HEIGHT + grasped_obj->size[2] * 0.5f;
            if (grasped_obj->physics.pos[2] < min_z) {
                grasped_obj->physics.pos[2] = min_z;
                if (grasped_obj->physics.vel[2] < 0.0f) grasped_obj->physics.vel[2] = 0.0f;
            }
        } else {
            if (env->gripper_state > 0.40f) {
                env->grasp_stability = fminf(1.0f, env->grasp_stability + 0.05f * (dt / ARM_DT));
            } else if (env->gripper_state < 0.20f) {
                env->grasp_stability = fmaxf(0.0f, env->grasp_stability - 0.05f * (dt / ARM_DT));
            }

            if (env->grasp_stability <= 0.0f) {
                grasped_obj->grasped = false;
                env->grasped_object_id = -1;
            } else {
                float hold_target[3] = {env->end_effector[0], env->end_effector[1], env->end_effector[2] - 0.02f};
                grasped_obj->physics.pos[0] = hold_target[0];
                grasped_obj->physics.pos[1] = hold_target[1];
                grasped_obj->physics.pos[2] = hold_target[2];
                vec3_zero(grasped_obj->physics.vel);
                vec3_zero(grasped_obj->physics.force);
                vec3_zero(grasped_obj->physics.torque);
                grasped_obj->physics.in_contact = true;
            }
        }
    }
}

static void apply_actions(RobotArm *env) {

    static const float vmax[6] = {
        1.5f,
        1.2f,
        2.0f,
        1.2f,
        1.2f,
        1.2f
    };

    float alpha = (env->action_smoothing_alpha > 0.0f) ? env->action_smoothing_alpha : 0.4f;

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
        float amax = (env->accel_limit > 0.0f) ? env->accel_limit : 30.0f;
        float damp = (env->damping > 0.0f) ? env->damping : 0.10f;
        
        float dv = v_des[i] - env->joint_vel[i];
        float dv_max = amax * ARM_DT;
        dv = clampf(dv, -dv_max, dv_max);
        env->joint_vel[i] += dv;
    env->joint_vel[i] *= (1.0f - damp * ARM_DT);
        env->joint_angles[i] += env->joint_vel[i] * ARM_DT;
    }

    env->joint_angles[0] = clampf(env->joint_angles[0], -M_PI * 0.75f,  M_PI * 0.75f);
    env->joint_angles[1] = clampf(env->joint_angles[1], -M_PI_2 * 0.6f, M_PI_2 * 0.9f);
    env->joint_angles[2] = clampf(env->joint_angles[2], -2.0f, -0.1f);
    env->joint_angles[3] = clampf(env->joint_angles[3], -M_PI * 0.6f,   M_PI * 0.6f);
    env->joint_angles[4] = clampf(env->joint_angles[4], -M_PI_2 * 0.6f, M_PI_2 * 0.6f);
    env->joint_angles[5] = clampf(env->joint_angles[5], -M_PI_2 * 0.7f, M_PI_2 * 0.7f);
    
    env->fk_dirty = 1;
}

static void init_pick_place_scene(RobotArm *env) {
    
    const ObjectType obj_types[MAX_OBJECTS] = {OBJ_RED, OBJ_BLUE, OBJ_GREEN, OBJ_YELLOW};
    for (int i = 0; i < MAX_OBJECTS; i++) {
        env->objects[i].type = obj_types[i];
        env->objects[i].grasped = false;
        env->objects[i].in_basket = false;
        env->objects[i].target_basket = i % MAX_BASKETS;
 
        env->objects[i].size[0] = OBJECT_SIZE;
        env->objects[i].size[1] = OBJECT_SIZE;
        env->objects[i].size[2] = OBJECT_SIZE;
    env->objects[i].mass = OBJECT_MASS;
    env->objects[i].restitution = OBJECT_RESTITUTION*0.5f;
    env->objects[i].friction = fminf(1.0f, OBJECT_FRICTION*1.2f);
        
        float initial_pos[3];
        const float margin_x = OBJECT_SIZE + 0.02f;
        const float margin_y = OBJECT_SIZE + 0.02f;
        const float min_sep = OBJECT_SIZE * 1.5f;
        int placed = 0;
        for (int attempt = 0; attempt < 50 && !placed; attempt++) {
            float left_x_min = -0.50f;
            float left_x_max = -0.10f;
            float y_min = WORKSPACE_Y_MIN + margin_y;
            float y_max = WORKSPACE_Y_MAX - margin_y;
            initial_pos[0] = randf(left_x_min, left_x_max);
            initial_pos[1] = randf(y_min, y_max);
            initial_pos[2] = TABLE_HEIGHT + OBJECT_SIZE * 0.5f + randf(0.0f, 0.01f);
            int ok = 1;
            for (int j = 0; j < i; j++) {
                float dx = initial_pos[0] - env->objects[j].physics.pos[0];
                float dy = initial_pos[1] - env->objects[j].physics.pos[1];
                float dz = initial_pos[2] - env->objects[j].physics.pos[2];
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < (min_sep*min_sep)) { ok = 0; break; }
            }
            if (ok) placed = 1;
        }
        if (!placed) {
            initial_pos[0] = clampf(initial_pos[0], WORKSPACE_X_MIN + margin_x, WORKSPACE_X_MAX - margin_x);
            initial_pos[1] = clampf(initial_pos[1], WORKSPACE_Y_MIN + margin_y, WORKSPACE_Y_MAX - margin_y);
            initial_pos[2] = TABLE_HEIGHT + OBJECT_SIZE * 0.5f;
        }

        init_physics_body(&env->objects[i].physics, initial_pos);
        env->objects[i].physics.on_surface = true;
    }
    

    const BasketType basket_types[MAX_BASKETS] = {BASKET_BLUE, BASKET_RED, BASKET_GREEN};
        for (int i = 0; i < MAX_BASKETS; i++) {
            env->baskets[i].type = basket_types[i];
            env->baskets[i].collected_count = 0.0f;
        float x = 0.35f;
        float spacing = 0.22f;
        float y = spacing - i * spacing;
            env->baskets[i].pos[0] = x;
            env->baskets[i].pos[1] = y;
            env->baskets[i].pos[2] = TABLE_HEIGHT + BASKET_SIZE * 0.5f;
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
        env->target_type = env->objects[idx].type;
        env->current_target_object = idx;
    }

    float min_d = 1e9f;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (env->objects[i].type != env->target_type) continue;
        float d = distance3d(env->end_effector, env->objects[i].physics.pos);
        if (d < min_d) min_d = d;
    }
    env->prev_distance = min_d;
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

    env->joint_angles[0] = randf(-M_PI * 0.4f, M_PI * 0.4f);        // base: ±72°
    env->joint_angles[1] = randf(-0.2f, 1.0f);                      // shoulder: -11° to +57°
    env->joint_angles[2] = randf(-1.5f, -0.3f);                     // elbow: -86° to -17°
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

    env->joint_angles[0] = clampf(env->joint_angles[0], -M_PI * 0.25f, M_PI * 0.25f);  // Base: ±45°
    env->joint_angles[1] = clampf(env->joint_angles[1], 0.2f, 0.7f);                   // Shoulder: safe range
    env->joint_angles[2] = clampf(env->joint_angles[2], -1.0f, -0.5f);                 // Elbow: safe range
    compute_forward_kinematics(env);
    env->fk_dirty = 0;

    init_pick_place_scene(env);

    if (env->frame_skip <= 0) env->frame_skip = 2;
    if (env->physics_substeps <= 0) env->physics_substeps = 1;
    env->success_distance2 = env->success_distance * env->success_distance;
    if (env->success_distance <= 0.0f) env->success_distance = 0.05f;
    if (env->obs_noise_std < 0.0f) env->obs_noise_std = 0.0f;
    if (env->actuation_noise_std < 0.0f) env->actuation_noise_std = 0.0f;

    if (env->rewards) env->rewards[0] = 0.0f;

    find_nearest_target(env);
    update_observations(env);

    env->camera_initialized = false;
    env->placed_event = 0;
    env->release_event = 0;
    env->recent_release_object_id = -1;
    env->continuous_gripper = 1;
    env->use_unified_clamp = 1;
    env->unified_clamp_min = 0.0f;
    env->unified_clamp_max = 200.0f;
    env->terminate_on_place = 1;
    env->extended_observation = 0;
    // Render controls defaults
    if (env->render_decimation <= 0) env->render_decimation = 1;
    env->render_counter = 0;
    if (env->render_target_fps < 0) env->render_target_fps = 60;

    if (env->episodes_completed == 0) {
        if (env->success_distance_start > 0.0f) env->success_distance = env->success_distance_start;
        if (env->on_gripper_spawn_start > 0.0f) env->on_gripper_spawn_prob = env->on_gripper_spawn_start;
        if (env->success_distance <= 0.0f) env->success_distance = 0.08f;
        if (env->on_gripper_spawn_prob <= 0.0f) env->on_gripper_spawn_prob = 0.02f;
        if (env->success_distance_min <= 0.0f) env->success_distance_min = 0.03f;
        if (env->on_gripper_spawn_min <= 0.0f) env->on_gripper_spawn_min = 0.005f;
        if (env->curriculum_episodes <= 0) env->curriculum_episodes = 200;
        if (env->assist_episodes <= 0) env->assist_episodes = 200;
        if (env->early_reach_episodes < 0) env->early_reach_episodes = 0;
        if (env->early_reach_bonus < 0.0f) env->early_reach_bonus = 0.0f;
    }
    env->best_place_dist = 1e9f;
    env->near_object_steps = 0;

    env->stagnation_steps = 0;
    env->stagnation_limit = 400;
    env->best_metric = 1e9f;
    env->target_unreachable_steps = 0;
}

void c_step(RobotArm *env) {
    if (!env) return;

    env->episode_steps++;
    
    if (env->rewards) env->rewards[0] = 0.0f;
    if (env->terminals) env->terminals[0] = 0;
    int skip = env->frame_skip > 0 ? env->frame_skip : 1;
    int psub = env->physics_substeps > 0 ? env->physics_substeps : 1;
    float sub_dt = ARM_DT / (float)psub;
    for (int s=0; s<skip; s++) {
        apply_actions(env);
        if (env->fk_dirty) { compute_forward_kinematics(env); env->fk_dirty = 0; }
        for (int sub=0; sub<psub; sub++) {
            find_nearest_target(env);
            update_gripper(env, sub_dt);
            for (int i = 0; i < MAX_OBJECTS; i++) {
                update_object_physics(&env->objects[i], sub_dt);
            }
            resolve_object_collisions(env, sub_dt);
            if (env->end_effector[0] < WORKSPACE_X_MIN || env->end_effector[0] > WORKSPACE_X_MAX ||
                env->end_effector[1] < WORKSPACE_Y_MIN || env->end_effector[1] > WORKSPACE_Y_MAX ||
                env->end_effector[2] < WORKSPACE_Z_MIN || env->end_effector[2] > WORKSPACE_Z_MAX) {
                float p = finalize_reward(env, -2.0f, -5.0f, 0.0f);
                if (env->rewards) env->rewards[0] = p;
                if (env->terminals) env->terminals[0] = 1;
                env->episode_return_accum += p;
                goto END_EPISODE;
            }
        }
    }

    {
        float metric2;
        if (env->grasped_object_id < 0) {
            metric2 = env->nearest_target_d2;
        } else {
            ManipObject* o = &env->objects[env->grasped_object_id];
            int bidx = o->target_basket;
            float dx = o->physics.pos[0] - env->baskets[bidx].pos[0];
            float dy = o->physics.pos[1] - env->baskets[bidx].pos[1];
            float dz = o->physics.pos[2] - env->baskets[bidx].pos[2];
            metric2 = dx*dx + dy*dy + dz*dz;
        }
        float metric = safe_sqrt(metric2);
        if (metric + 1e-4f < env->best_metric) { env->best_metric = metric; env->stagnation_steps = 0; }
        else env->stagnation_steps++;
    }

    int was_grasped_prev = env->was_grasped_last_step;
    env->was_grasped_last_step = (env->grasped_object_id >= 0);

    env->placed_event = 0;
    float r = compute_reward(env);
    if (env->rewards) env->rewards[0] = r;
    env->episode_return_accum += r;
    if (!was_grasped_prev && env->grasped_object_id >= 0) {
        env->episode_pick_count += 1.0f;
        env->log.pick_success_rate += 1.0f;
    }

    if (env->grasped_object_id < 0) {
        if (env->nearest_target_idx >= 0) {
            if (env->nearest_target_d2 < env->success_distance2) env->near_object_steps += 1; else env->near_object_steps = 0;
        } else {
            env->near_object_steps = 0;
        }
    } else {
        env->near_object_steps = 0;
    }

    if (env->near_object_steps > 200) {
        float penalty = finalize_reward(env, -0.5f, -2.0f, 0.0f);
        env->episode_return_accum += penalty;
        if (env->rewards) env->rewards[0] = penalty;
        if (env->terminals) env->terminals[0] = 1;
        goto END_EPISODE;
    }
    if (env->stagnation_steps > env->stagnation_limit) {
        if (env->terminals) env->terminals[0] = 1;
        goto END_EPISODE;
    }

    if (env->early_reach_episodes > 0 && env->episodes_completed < env->early_reach_episodes && env->grasped_object_id < 0) {
        if (env->nearest_target_idx >= 0) {
            if (env->nearest_target_d2 < env->success_distance2) {
            float bonus = env->early_reach_bonus;
            if (env->use_unified_clamp) bonus = clampf(bonus, env->unified_clamp_min, env->unified_clamp_max);
            env->episode_return_accum += bonus;
            if (env->rewards) env->rewards[0] = bonus;
            if (env->terminals) env->terminals[0] = 1;
            goto END_EPISODE;
            }
        }
    }

    if ((env->terminate_on_place && env->placed_event) || (env->episode_steps >= env->max_steps)) {
        if (env->terminals) env->terminals[0] = 1;
        goto END_EPISODE;
    }

    update_observations(env);
    return;

END_EPISODE:
    env->log.n += 1.0f;
    env->log.episode_length += env->episode_steps;
    env->log.episode_return += env->episode_return_accum;
    env->log.score += env->episode_score_accum;
    {
        float avg_rps = env->episode_steps > 0 ? (env->episode_return_accum / (float)env->episode_steps) : 0.0f;
        float max_rps = 15.0f;
        float perf_ep = max_rps > 0.0f ? (avg_rps / max_rps) * 100.0f : 0.0f;
        perf_ep = clampf(perf_ep, 0.0f, 100.0f);
        env->log.perf += perf_ep;
    }
    env->episodes_completed += 1;
    if (env->curriculum_episodes > 0) {
        float t = (float)env->episodes_completed / (float)env->curriculum_episodes;
        t = clampf(t, 0.0f, 1.0f);
        env->success_distance = env->success_distance_start * (1.0f - t) + env->success_distance_min * t;
        env->on_gripper_spawn_prob = env->on_gripper_spawn_start * (1.0f - t) + env->on_gripper_spawn_min * t;
    env->success_distance2 = env->success_distance * env->success_distance;
    }
    update_observations(env);
    return;
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
    if (env->headless) return;

    if (env->fk_dirty) {
        compute_forward_kinematics(env);
        env->fk_dirty = 0;
    }

    if (!IsWindowReady()) {
        unsigned int flags = FLAG_MSAA_4X_HINT;
        if (env->vsync) flags |= FLAG_VSYNC_HINT;
        SetConfigFlags(flags);
        InitWindow(1200, 800, "Puffer Robot Arm");
        if (env->render_target_fps > 0) SetTargetFPS(env->render_target_fps);
    }

    if (!env->camera_initialized) {
        env->camera_distance = 3.5f;
        env->camera_azimuth = -0.8f;
        env->camera_elevation = 0.35f;
        env->camera.up = (Vector3){0,0,1};
        env->camera.fovy = 45.0f;
        env->camera.projection = CAMERA_PERSPECTIVE;
        env->camera_initialized = true;
    if (!env->cube_model_loaded) {
            char pathbuf[512] = {0};
            const char *envRoot = getenv("PUFFERLIB_ASSETS");
            if (envRoot && envRoot[0]) {
                snprintf(pathbuf, sizeof(pathbuf), "%s/puffer.glb", envRoot);
                if (FileExists(pathbuf)) {
                    env->cube_model = LoadModel(pathbuf);
                    if (env->cube_model.meshCount > 0) env->cube_model_loaded = 1;
                }
            }
            if (!env->cube_model_loaded) {
                const char *candidates[] = {
                    "resources/shared/puffer.glb",
                    "./resources/shared/puffer.glb",
                    "../resources/shared/puffer.glb",
                    "../../resources/shared/puffer.glb",
                    "pufferlib/resources/shared/puffer.glb",
                    "./pufferlib/resources/shared/puffer.glb",
                    "ocean/resources/shared/puffer.glb",
                    "./ocean/resources/shared/puffer.glb"
                };
                for (int ci = 0; ci < (int)(sizeof(candidates)/sizeof(candidates[0])); ++ci) {
                    if (FileExists(candidates[ci])) {
                        env->cube_model = LoadModel(candidates[ci]);
                        if (env->cube_model.meshCount > 0) { env->cube_model_loaded = 1; break; }
                    }
                }
            }
            env->cube_model_scale = OBJECT_SIZE;
            env->cube_model_visual_mul = (env->cube_model_visual_mul > 0.0f) ? env->cube_model_visual_mul : 2.0f;
            env->cube_model_offset[0] = env->cube_model_offset[1] = env->cube_model_offset[2] = 0.0f;
            if (env->cube_model_loaded) {
                BoundingBox bb = GetModelBoundingBox(env->cube_model);
                Vector3 size = {bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z};
                float maxdim = fmaxf(size.x, fmaxf(size.y, size.z));
                if (maxdim > 1e-6f) {
                    float fit = OBJECT_SIZE / maxdim;
                    env->cube_model_scale = fit * env->cube_model_visual_mul;
                }
                Vector3 center = {(bb.max.x + bb.min.x)*0.5f, (bb.max.y + bb.min.y)*0.5f, (bb.max.z + bb.min.z)*0.5f};
                env->cube_model_offset[0] = -center.x;
                env->cube_model_offset[1] = -center.y;
                env->cube_model_offset[2] = -center.z;
            }
        }
    }

    if (IsKeyDown(KEY_ESCAPE)) return;

    handle_camera(env);
    env->render_counter++;
    if (env->render_decimation > 1 && (env->render_counter % env->render_decimation) != 0) return;

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
    float cosb = env->cached_cos[0];
    float sinb = env->cached_sin[0];
    float L1 = env->link1_length > 0 ? env->link1_length : ARM_LINK1_LENGTH;
    float L2 = env->link2_length > 0 ? env->link2_length : ARM_LINK2_LENGTH;
    float c1 = env->cached_cos[1], s1 = env->cached_sin[1];
    float c12 = c1 * env->cached_cos[2] - s1 * env->cached_sin[2];
    float s12 = s1 * env->cached_cos[2] + c1 * env->cached_sin[2];
    Vector3 shoulder = {L1*c1*cosb, L1*c1*sinb, 0.2f + L1*s1};
    Vector3 elbow = { (L1*c1 + L2*c12)*cosb,
                      (L1*c1 + L2*c12)*sinb,
                      0.2f + L1*s1 + L2*s12 };
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

    {
        int   best_i = env->nearest_target_idx;

        for (int i = 0; i < MAX_BASKETS; i++) {
            Basket *b = &env->baskets[i];
            Color bc = (b->type == BASKET_RED)   ? CLR_RED :
                       (b->type == BASKET_BLUE)  ? CLR_BLUE :
                       (b->type == BASKET_GREEN) ? CLR_GREEN : METAL_LIGHT;
            float outerW = BASKET_SIZE * 1.2f;
            float outerD = BASKET_SIZE * 1.2f;
            float wallT  = 0.010f;
            float baseT  = 0.008f;
            float binH   = BASKET_SIZE;
            float bottomZ = b->pos[2] - 0.5f * binH;
            Vector3 baseCenter = (Vector3){b->pos[0], b->pos[1], bottomZ + baseT * 0.5f};
            Color baseCol = (Color){235,238,242,230};
            DrawCube(baseCenter, outerW, outerD, baseT, baseCol);
            float wallH = binH - baseT;
            float wallZ = bottomZ + baseT + 0.5f * wallH;
            Color wallTint = (Color){bc.r, bc.g, bc.b, 180};
            DrawCube((Vector3){b->pos[0] + (outerW*0.5f - wallT*0.5f), b->pos[1], wallZ}, wallT, outerD, wallH, wallTint);
            DrawCube((Vector3){b->pos[0] - (outerW*0.5f - wallT*0.5f), b->pos[1], wallZ}, wallT, outerD, wallH, wallTint);
            DrawCube((Vector3){b->pos[0], b->pos[1] + (outerD*0.5f - wallT*0.5f), wallZ}, outerW, wallT, wallH, wallTint);
            DrawCube((Vector3){b->pos[0], b->pos[1] - (outerD*0.5f - wallT*0.5f), wallZ}, outerW, wallT, wallH, wallTint);
        }
        for (int i = 0; i < MAX_OBJECTS; i++) {
            ManipObject *o = &env->objects[i];
            Color oc = (o->type == OBJ_RED)   ? CLR_RED :
                       (o->type == OBJ_BLUE)  ? CLR_BLUE :
                       (o->type == OBJ_GREEN) ? CLR_GREEN : CLR_YEL;
            Vector3 op = {o->physics.pos[0], o->physics.pos[1], o->physics.pos[2]};
            if (env->cube_model_loaded) {
                Vector3 pos = {op.x + env->cube_model_offset[0]*env->cube_model_scale,
                               op.y + env->cube_model_offset[1]*env->cube_model_scale,
                               op.z + env->cube_model_offset[2]*env->cube_model_scale};
                DrawModelEx(env->cube_model, pos, (Vector3){0,0,1}, 0.0f,
                            (Vector3){env->cube_model_scale, env->cube_model_scale, env->cube_model_scale}, oc);
            } else {
                DrawCube(op, OBJECT_SIZE, OBJECT_SIZE, OBJECT_SIZE, oc);
                DrawCubeWires(op, OBJECT_SIZE*1.01f, OBJECT_SIZE*1.01f, OBJECT_SIZE*1.01f, (Color){oc.r,oc.g,oc.b,180});
            }
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

    DrawText("Robot Arm - Pick and Place", 20, 20, 20, BLACK);
    {
        if (env->nearest_target_idx >= 0) {
            float best_d = safe_sqrt(env->nearest_target_d2);
            DrawText(TextFormat("Target Obj Dist: %.2f", best_d), 20, 50, 16, BLACK);
            DrawText(TextFormat("Target Type: %s", env->target_type == OBJ_RED ? "RED" :
                               env->target_type == OBJ_BLUE ? "BLUE" : "GREEN"), 20, 70, 16, BLACK);
        }
    }
    DrawText(TextFormat("Score: %.2f", env->log.score), 20, 80, 16, BLACK);
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
    if (env->cube_model_loaded) { UnloadModel(env->cube_model); env->cube_model_loaded = 0; }
    if (IsWindowReady()) CloseWindow();
}