#pragma once

#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define ROBOT_MIN_RADIUS 1.0
#define ROBOT_MAX_RADIUS 1.05
#define ROBOT_MAX_JUMP_SPEED 15.0
#define ROBOT_ACCELERATION 100.0
#define ROBOT_NITRO_ACCELERATION 30.0
#define ROBOT_MAX_GROUND_SPEED 30.0
#define ROBOT_ARENA_E 0.0
#define ROBOT_MASS 2.0
#define TICKS_PER_SECOND 30
#define MICROTICKS_PER_TICK 1
#define RESET_TICKS (2 * TICKS_PER_SECOND)
#define BALL_ARENA_E 0.7
#define BALL_RADIUS 2.0
#define BALL_MASS 1.0
#define MIN_HIT_E 0.4
#define MAX_HIT_E 0.5
#define MAX_ENTITY_SPEED 100.0
#define MAX_NITRO_AMOUNT 100.0
#define START_NITRO_AMOUNT 50.0
#define NITRO_POINT_VELOCITY_CHANGE 0.6
#define NITRO_PACK_X 20.0
#define NITRO_PACK_Y 1.0
#define NITRO_PACK_Z 30.0
#define NITRO_PACK_RADIUS 0.5
#define NITRO_PACK_RESPAWN_TICKS (10 * TICKS_PER_SECOND)
#define GRAVITY 30.0

#define LOG_BUFFER_SIZE 1024

typedef struct Log Log;
struct Log {
    double episode_return_side;
    double episode_return_total;
    double episode_return_abs;
    double episode_return_one;
    int episode_length;
    double winner;  // was there a winner?
};

typedef struct LogBuffer LogBuffer;
struct LogBuffer {
    Log* logs;
    int length;
    int idx;
};

LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        logs->idx = 0;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
}

Log aggregate(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return_side += logs->logs[i].episode_return_side;
        log.episode_return_total += logs->logs[i].episode_return_total;
        log.episode_return_abs += logs->logs[i].episode_return_abs;
        log.episode_return_one += fabs(logs->logs[i].episode_return_one);
        log.episode_length += logs->logs[i].episode_length;
        log.winner += logs->logs[i].winner;
    }
    log.episode_return_side /= logs->idx;
    log.episode_return_total /= logs->idx;
    log.episode_return_abs /= logs->idx;
    log.episode_return_one /= logs->idx;
    log.episode_length /= logs->idx;
    log.winner /= logs->idx;
    return log;
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    log = aggregate(logs);
    logs->idx = 0;
    return log;
}

// Arena parameters (from the description)
typedef struct {
    float width;
    float height;
    float depth;
    float bottom_radius;
    float top_radius;
    float corner_radius;
    float goal_top_radius;
    float goal_width;
    float goal_depth;
    float goal_height;
    float goal_side_radius;
} CodeBallArena;

CodeBallArena arena = {60, 20, 80, 3, 7, 13, 3, 30, 10, 10, 1};

// 3D Vector
typedef struct {
    float x;
    float y;
    float z;
} Vec3D;

// Action structure
typedef struct {
    Vec3D target_velocity;
    float jump_speed;
    bool use_nitro;
} Action;

// Entity (Robot or Ball)
typedef struct {
    Vec3D position;
    Vec3D velocity;
    float radius;
    float radius_change_speed;
    float mass;
    float arena_e;
    bool touch;
    Vec3D touch_normal;
    float nitro;
    Action action;
    bool side;
} Entity;

// Nitro Pack
typedef struct {
    Vec3D position;
    bool alive;
    int respawn_ticks;
    float radius;  // Nitro pack has radius
} NitroPack;

// Helper functions
float vec3d_length(Vec3D v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
Vec3D vec3d_normalize(Vec3D v) {
    float l = vec3d_length(v);
    if (l == 0) return (Vec3D){0, 0, 0};  // Avoid division by zero
    return (Vec3D){v.x / l, v.y / l, v.z / l};
}

float vec3d_dot(Vec3D a, Vec3D b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3D vec3d_subtract(Vec3D a, Vec3D b) {
    return (Vec3D){a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3D vec3d_add(Vec3D a, Vec3D b) {
    return (Vec3D){a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3D vec3d_multiply(Vec3D v, float s) {
    return (Vec3D){v.x * s, v.y * s, v.z * s};
}

float clamp(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}
Vec3D vec3d_clamp(Vec3D v, float max_length) {
    float length = vec3d_length(v);
    if (length > max_length) {
        return vec3d_multiply(v, max_length / length);
    }
    return v;
}

typedef struct {
    float distance;
    Vec3D normal;
} DistanceAndNormal;

// Distance and Normal functions
DistanceAndNormal dan_to_plane(Vec3D point, Vec3D point_on_plane,
                               Vec3D plane_normal) {
    return (DistanceAndNormal){
        vec3d_dot(vec3d_subtract(point, point_on_plane), plane_normal),
        plane_normal};
}

DistanceAndNormal dan_to_sphere_inner(Vec3D point, Vec3D sphere_center,
                                      float sphere_radius) {
    return (DistanceAndNormal){
        sphere_radius - vec3d_length(vec3d_subtract(point, sphere_center)),
        vec3d_normalize(vec3d_subtract(sphere_center, point))};
}

DistanceAndNormal dan_to_sphere_outer(Vec3D point, Vec3D sphere_center,
                                      float sphere_radius) {
    return (DistanceAndNormal){
        vec3d_length(vec3d_subtract(point, sphere_center)) - sphere_radius,
        vec3d_normalize(vec3d_subtract(point, sphere_center))};
}

DistanceAndNormal dan_to_arena_quarter(Vec3D point) {
    DistanceAndNormal dan;
    dan.distance = 1e9;  // Initialize with a large value

    // Ground
    DistanceAndNormal temp_dan =
        dan_to_plane(point, (Vec3D){0, 0, 0}, (Vec3D){0, 1, 0});
    if (temp_dan.distance < dan.distance) dan = temp_dan;

    // Ceiling
    temp_dan =
        dan_to_plane(point, (Vec3D){0, arena.height, 0}, (Vec3D){0, -1, 0});
    if (temp_dan.distance < dan.distance) dan = temp_dan;

    // Side x
    temp_dan = dan_to_plane(point, (Vec3D){arena.width / 2.0, 0, 0},
                            (Vec3D){-1, 0, 0});
    if (temp_dan.distance < dan.distance) dan = temp_dan;

    // Side z (goal)
    temp_dan = dan_to_plane(
        point, (Vec3D){0, 0, (arena.depth / 2.0) + arena.goal_depth},
        (Vec3D){0, 0, -1});
    if (temp_dan.distance < dan.distance) dan = temp_dan;

    // Side z (regular)
    Vec3D v = {point.x - ((arena.goal_width / 2.0) - arena.goal_top_radius),
               point.y - (arena.goal_height - arena.goal_top_radius), 0};
    if (point.x >= (arena.goal_width / 2.0) + arena.goal_side_radius ||
        point.y >= arena.goal_height + arena.goal_side_radius ||
        (v.x > 0 && v.y > 0 &&
         vec3d_length(v) >= arena.goal_top_radius + arena.goal_side_radius)) {
        temp_dan = dan_to_plane(point, (Vec3D){0, 0, arena.depth / 2.0},
                                (Vec3D){0, 0, -1});
        if (temp_dan.distance < dan.distance) dan = temp_dan;
    }

    // Goal back corners
    if (point.z >
        (arena.depth / 2.0) + arena.goal_depth - arena.bottom_radius) {
        temp_dan = dan_to_sphere_inner(
            point,
            (Vec3D){
                clamp(point.x, arena.bottom_radius - (arena.goal_width / 2.0),
                      (arena.goal_width / 2.0) - arena.bottom_radius),
                clamp(point.y, arena.bottom_radius,
                      arena.goal_height - arena.goal_top_radius),
                (arena.depth / 2.0) + arena.goal_depth - arena.bottom_radius},
            arena.bottom_radius);
        if (temp_dan.distance < dan.distance) dan = temp_dan;
    }

    // Corner
    if (point.x > (arena.width / 2.0) - arena.corner_radius &&
        point.z > (arena.depth / 2.0) - arena.corner_radius) {
        temp_dan = dan_to_sphere_inner(
            point,
            (Vec3D){(arena.width / 2.0) - arena.corner_radius, point.y,
                    (arena.depth / 2.0) - arena.corner_radius},
            arena.corner_radius);
        if (temp_dan.distance < dan.distance) dan = temp_dan;
    }

    // Goal outer corner
    if (point.z < (arena.depth / 2.0) + arena.goal_side_radius) {
        // Side x
        if (point.x < (arena.goal_width / 2.0) + arena.goal_side_radius) {
            temp_dan = dan_to_sphere_outer(
                point,
                (Vec3D){(arena.goal_width / 2.0) + arena.goal_side_radius,
                        point.y, (arena.depth / 2.0) + arena.goal_side_radius},
                arena.goal_side_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
        // Ceiling
        if (point.y < arena.goal_height + arena.goal_side_radius) {
            temp_dan = dan_to_sphere_outer(
                point,
                (Vec3D){point.x, arena.goal_height + arena.goal_side_radius,
                        (arena.depth / 2.0) + arena.goal_side_radius},
                arena.goal_side_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
        // Top corner
        Vec3D o = {(arena.goal_width / 2.0) - arena.goal_top_radius,
                   arena.goal_height - arena.goal_top_radius, 0};
        v = (Vec3D){point.x - o.x, point.y - o.y, 0};
        if (v.x > 0 && v.y > 0) {
            o.x += vec3d_normalize(v).x *
                   (arena.goal_top_radius + arena.goal_side_radius);
            o.y += vec3d_normalize(v).y *
                   (arena.goal_top_radius + arena.goal_side_radius);
            temp_dan = dan_to_sphere_outer(
                point,
                (Vec3D){o.x, o.y, (arena.depth / 2.0) + arena.goal_side_radius},
                arena.goal_side_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
    }

    // Goal inside top corners
    if (point.z > (arena.depth / 2.0) + arena.goal_side_radius &&
        point.y > arena.goal_height - arena.goal_top_radius) {
        // Side x
        if (point.x > (arena.goal_width / 2.0) - arena.goal_top_radius) {
            temp_dan = dan_to_sphere_inner(
                point,
                (Vec3D){(arena.goal_width / 2.0) - arena.goal_top_radius,
                        arena.goal_height - arena.goal_top_radius, point.z},
                arena.goal_top_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
        // Side z
        if (point.z >
            (arena.depth / 2.0) + arena.goal_depth - arena.goal_top_radius) {
            temp_dan = dan_to_sphere_inner(
                point,
                (Vec3D){point.x, arena.goal_height - arena.goal_top_radius,
                        (arena.depth / 2.0) + arena.goal_depth -
                            arena.goal_top_radius},
                arena.goal_top_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
    }

    // Bottom corners
    if (point.y < arena.bottom_radius) {
        // Side x
        if (point.x > (arena.width / 2.0) - arena.bottom_radius) {
            DistanceAndNormal temp_dan = dan_to_sphere_inner(
                point,
                (Vec3D){(arena.width / 2.0) - arena.bottom_radius,
                        arena.bottom_radius, point.z},
                arena.bottom_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
        // Side z
        if (point.z > (arena.depth / 2.0) - arena.bottom_radius &&
            point.x >= (arena.goal_width / 2.0) + arena.goal_side_radius) {
            DistanceAndNormal temp_dan = dan_to_sphere_inner(
                point,
                (Vec3D){point.x, arena.bottom_radius,
                        (arena.depth / 2.0) - arena.bottom_radius},
                arena.bottom_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
        // Side z (goal)
        if (point.z >
            (arena.depth / 2.0) + arena.goal_depth - arena.bottom_radius) {
            DistanceAndNormal temp_dan = dan_to_sphere_inner(
                point,
                (Vec3D){point.x, arena.bottom_radius,
                        (arena.depth / 2.0) + arena.goal_depth -
                            arena.bottom_radius},
                arena.bottom_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
        // Goal outer corner
        Vec3D o = {(arena.goal_width / 2.0) + arena.goal_side_radius,
                   (arena.depth / 2.0) + arena.goal_side_radius, 0};
        Vec3D v = {point.x - o.x, point.z - o.y, 0};
        if (v.x < 0 && v.y < 0 &&
            vec3d_length(v) < arena.goal_side_radius + arena.bottom_radius) {
            o.x += vec3d_normalize(v).x *
                   (arena.goal_side_radius + arena.bottom_radius);
            o.y += vec3d_normalize(v).y *
                   (arena.goal_side_radius + arena.bottom_radius);
            DistanceAndNormal temp_dan = dan_to_sphere_inner(
                point, (Vec3D){o.x, arena.bottom_radius, o.y},
                arena.bottom_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
        // Side x (goal)
        if (point.z >= (arena.depth / 2.0) + arena.goal_side_radius &&
            point.x > (arena.goal_width / 2.0) - arena.bottom_radius) {
            DistanceAndNormal temp_dan = dan_to_sphere_inner(
                point,
                (Vec3D){(arena.goal_width / 2.0) - arena.bottom_radius,
                        arena.bottom_radius, point.z},
                arena.bottom_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
        // Corner
        if (point.x > (arena.width / 2.0) - arena.corner_radius &&
            point.z > (arena.depth / 2.0) - arena.corner_radius) {
            Vec3D corner_o = {(arena.width / 2.0) - arena.corner_radius,
                              (arena.depth / 2.0) - arena.corner_radius, 0};
            Vec3D n = {point.x - corner_o.x, point.z - corner_o.y, 0};
            float dist = vec3d_length(n);
            if (dist > arena.corner_radius - arena.bottom_radius) {
                n = vec3d_normalize(n);
                Vec3D o2 = {corner_o.x + n.x * (arena.corner_radius -
                                                arena.bottom_radius),
                            corner_o.y + n.y * (arena.corner_radius -
                                                arena.bottom_radius),
                            0};
                DistanceAndNormal temp_dan = dan_to_sphere_inner(
                    point, (Vec3D){o2.x, arena.bottom_radius, o2.y},
                    arena.bottom_radius);
                if (temp_dan.distance < dan.distance) dan = temp_dan;
            }
        }
    }

    // Ceiling corners
    if (point.y > arena.height - arena.top_radius) {
        // Side x
        if (point.x > (arena.width / 2.0) - arena.top_radius) {
            DistanceAndNormal temp_dan = dan_to_sphere_inner(
                point,
                (Vec3D){(arena.width / 2.0) - arena.top_radius,
                        arena.height - arena.top_radius, point.z},
                arena.top_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
        // Side z
        if (point.z > (arena.depth / 2.0) - arena.top_radius) {
            DistanceAndNormal temp_dan = dan_to_sphere_inner(
                point,
                (Vec3D){point.x, arena.height - arena.top_radius,
                        (arena.depth / 2.0) - arena.top_radius},
                arena.top_radius);
            if (temp_dan.distance < dan.distance) dan = temp_dan;
        }
        // Corner
        if (point.x > (arena.width / 2.0) - arena.corner_radius &&
            point.z > (arena.depth / 2.0) - arena.corner_radius) {
            Vec3D corner_o = {(arena.width / 2.0) - arena.corner_radius,
                              (arena.depth / 2.0) - arena.corner_radius, 0};
            Vec3D dv = {point.x - corner_o.x, point.z - corner_o.y, 0};
            if (vec3d_length(dv) > arena.corner_radius - arena.top_radius) {
                Vec3D n = vec3d_normalize(dv);
                Vec3D o2 = {
                    corner_o.x + n.x * (arena.corner_radius - arena.top_radius),
                    arena.height - arena.top_radius, o2.y};
                DistanceAndNormal temp_dan =
                    dan_to_sphere_inner(point, o2, arena.top_radius);
                if (temp_dan.distance < dan.distance) dan = temp_dan;
            }
        }
    }
    return dan;
}

DistanceAndNormal dan_to_arena(Vec3D point) {
    bool negate_x = point.x < 0;
    bool negate_z = point.z < 0;
    if (negate_x) point.x = -point.x;
    if (negate_z) point.z = -point.z;
    DistanceAndNormal result = dan_to_arena_quarter(point);
    if (negate_x) result.normal.x = -result.normal.x;
    if (negate_z) result.normal.z = -result.normal.z;
    return result;
}

void collide_entities(Entity* a, Entity* b) {
    Vec3D delta_position = {b->position.x - a->position.x,
                            b->position.y - a->position.y,
                            b->position.z - a->position.z};
    float distance = vec3d_length(delta_position);
    float penetration = a->radius + b->radius - distance;
    if (penetration > 0) {
        float k_a = (1.0 / a->mass) / ((1.0 / a->mass) + (1.0 / b->mass));
        float k_b = (1.0 / b->mass) / ((1.0 / a->mass) + (1.0 / b->mass));
        Vec3D normal = vec3d_normalize(delta_position);
        a->position.x -= normal.x * penetration * k_a;
        a->position.y -= normal.y * penetration * k_a;
        a->position.z -= normal.z * penetration * k_a;
        b->position.x += normal.x * penetration * k_b;
        b->position.y += normal.y * penetration * k_b;
        b->position.z += normal.z * penetration * k_b;

        float delta_velocity =
            vec3d_dot((Vec3D){b->velocity.x - a->velocity.x,
                              b->velocity.y - a->velocity.y,
                              b->velocity.z - a->velocity.z},
                      normal) -
            b->radius_change_speed - a->radius_change_speed;

        if (delta_velocity < 0) {
            float e = ((float)rand() / RAND_MAX) * (MAX_HIT_E - MIN_HIT_E) +
                       MIN_HIT_E;
            Vec3D impulse = {normal.x * (1 + e) * delta_velocity,
                             normal.y * (1 + e) * delta_velocity,
                             normal.z * (1 + e) * delta_velocity};
            a->velocity.x += impulse.x * k_a;
            a->velocity.y += impulse.y * k_a;
            a->velocity.z += impulse.z * k_a;
            b->velocity.x -= impulse.x * k_b;
            b->velocity.y -= impulse.y * k_b;
            b->velocity.z -= impulse.z * k_b;
        }
    }
}

Vec3D collide_with_arena(Entity* e) {
    DistanceAndNormal dan = dan_to_arena(e->position);
    float penetration = e->radius - dan.distance;
    if (penetration > 0) {
        e->position.x += dan.normal.x * penetration;
        e->position.y += dan.normal.y * penetration;
        e->position.z += dan.normal.z * penetration;

        float velocity =
            vec3d_dot(e->velocity, dan.normal) - e->radius_change_speed;
        if (velocity < 0) {
            e->velocity.x -= (1 + e->arena_e) * velocity * dan.normal.x;
            e->velocity.y -= (1 + e->arena_e) * velocity * dan.normal.y;
            e->velocity.z -= (1 + e->arena_e) * velocity * dan.normal.z;
        }
        return dan.normal;
    }
    return (Vec3D){0, 0, 0};  // Return zero vector to indicate no collision
}

void move(Entity* e, float delta_time) {
    e->velocity = vec3d_clamp(e->velocity, MAX_ENTITY_SPEED);
    e->position.x += e->velocity.x * delta_time;
    e->position.y += e->velocity.y * delta_time;
    e->position.z += e->velocity.z * delta_time;
    e->position.y -= GRAVITY * delta_time * delta_time / 2.0;
    e->velocity.y -= GRAVITY * delta_time;
}

typedef enum BaselineType {
    DO_NOTHING,
    RANDOM_ACTIONS,
    RUN_TO_BALL
} BaselineType;

typedef struct CodeBall {
    Entity ball;
    int n_robots;
    Entity* robots;
    int n_nitros;
    NitroPack* nitro_packs;
    int tick;
    float* actions;
    float* rewards;
    bool terminal;
    int frame_skip;
    Log log;
    LogBuffer* log_buffer;
    bool is_single;
    BaselineType baseline;
    float goal_scored_reward;
    float loiter_penalty;
    float ball_reward;
} CodeBall;

void allocate(CodeBall* env) {
    env->robots = (Entity*)calloc(env->n_robots, sizeof(Entity));
    env->nitro_packs = (NitroPack*)calloc(env->n_nitros, sizeof(NitroPack));
    env->rewards = (float*)calloc(env->n_robots, sizeof(float));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_allocated(CodeBall* env) {
    free(env->robots);
    free(env->nitro_packs);
    free(env->rewards);
    free_logbuffer(env->log_buffer);
}

void reset_positions(CodeBall* env) {
    Entity* robots = env->robots;
    for (int i = 0; i < env->n_robots; i++) {
        float half = env->n_robots / 2 - 1;
        float distance = arena.width * 0.4;
        robots[i].position.x = ((i / 2) / half - 0.5) * 2 * distance * 0.8;
        robots[i].position.z =
            sqrtf(distance * distance -
                  robots[i].position.x * robots[i].position.x) *
            (i % 2 == 0 ? -1 : 1);
        robots[i].position.y = 0;
        robots[i].velocity = (Vec3D){0, 0, 0};
        robots[i].radius = ROBOT_MIN_RADIUS;
        robots[i].radius_change_speed = 0;
        robots[i].mass = ROBOT_MASS;
        robots[i].arena_e = ROBOT_ARENA_E;
        robots[i].touch = false;
        robots[i].touch_normal = (Vec3D){0, 0, 0};
        robots[i].nitro = START_NITRO_AMOUNT;
        robots[i].action.target_velocity = (Vec3D){0, 0, 0};
        robots[i].action.jump_speed = 0;
        robots[i].action.use_nitro = false;
        robots[i].side = robots[i].position.z > 0;
    }

    Entity ball;
    ball.position.x = 0;
    ball.position.z = 0;
    ball.position.y = ((float)rand() / RAND_MAX) * (3 * BALL_RADIUS) +
                      BALL_RADIUS;  // Random height
    ball.velocity = (Vec3D){0, 0, 0};
    ball.radius = BALL_RADIUS;
    ball.radius_change_speed = 0;
    ball.mass = BALL_MASS;
    ball.arena_e = BALL_ARENA_E;
    ball.touch = false;
    ball.touch_normal = (Vec3D){0, 0, 0};
    ball.nitro = 0;
    ball.action.target_velocity = (Vec3D){0, 0, 0};
    ball.action.jump_speed = 0;
    ball.action.use_nitro = false;
    env->ball = ball;

    for (int i = 0; i < env->n_nitros; i++) {
        env->nitro_packs[i].position.x = NITRO_PACK_X * (i % 2 == 0 ? -1 : 1);
        env->nitro_packs[i].position.z = NITRO_PACK_Z * (i / 2 == 0 ? -1 : 1);
        env->nitro_packs[i].position.y = NITRO_PACK_Y;
        env->nitro_packs[i].alive = true;
    }

    memset(env->actions, 0, env->n_robots * 4 * sizeof(float));
    env->tick = 0;

    if (env->frame_skip == 0) {
        env->frame_skip = 1;
    }
}

void goal_scored(CodeBall *env, bool side) {
    for (int i = 0; i < env->n_robots; i++) {
        env->rewards[i] += (env->robots[i].side == side ? 1 : -1) * env->goal_scored_reward;
    }
    env->terminal = true;
    env->log.winner = 1;

    add_log(env->log_buffer, &env->log);
    env->log = (Log){0};

    reset_positions(env);
}

void reset(CodeBall* env) {
    reset_positions(env);
    memset(env->rewards, 0, env->n_robots * sizeof(float));

    env->terminal = false;
    
    add_log(env->log_buffer, &env->log);
    env->log = (Log){0};
}

void update(float delta_time, CodeBall* env) {
    Entity* robots = env->robots;
    Entity* ball = &env->ball;
    NitroPack* nitro_packs = env->nitro_packs;

    for (int i = 0; i < env->n_robots; i++) {
        Vec3D target_velocity = vec3d_clamp(robots[i].action.target_velocity,
                                            ROBOT_MAX_GROUND_SPEED);

        if (robots[i].touch) {
            target_velocity.x -=
                robots[i].touch_normal.x *
                vec3d_dot(robots[i].touch_normal, target_velocity);
            target_velocity.y -=
                robots[i].touch_normal.y *
                vec3d_dot(robots[i].touch_normal, target_velocity);
            target_velocity.z -=
                robots[i].touch_normal.z *
                vec3d_dot(robots[i].touch_normal, target_velocity);

            Vec3D target_velocity_change = {
                target_velocity.x - robots[i].velocity.x,
                target_velocity.y - robots[i].velocity.y,
                target_velocity.z - robots[i].velocity.z};
            if (vec3d_length(target_velocity_change) > 0) {
                float acceleration =
                    ROBOT_ACCELERATION * fmax(0, robots[i].touch_normal.y);
                Vec3D acceleration_vec = vec3d_clamp(
                    (Vec3D){
                        target_velocity_change.x * acceleration * delta_time /
                            vec3d_length(target_velocity_change),
                        target_velocity_change.y * acceleration * delta_time /
                            vec3d_length(target_velocity_change),
                        target_velocity_change.z * acceleration * delta_time /
                            vec3d_length(target_velocity_change)},
                    vec3d_length(target_velocity_change));
                robots[i].velocity.x += acceleration_vec.x;
                robots[i].velocity.y += acceleration_vec.y;
                robots[i].velocity.z += acceleration_vec.z;
            }
        }

        if (robots[i].action.use_nitro &&
            robots[i].nitro > 0) {  // Check if nitro is available
            Vec3D target_velocity_change = {
                robots[i].action.target_velocity.x - robots[i].velocity.x,
                robots[i].action.target_velocity.y - robots[i].velocity.y,
                robots[i].action.target_velocity.z - robots[i].velocity.z};
            target_velocity_change =
                vec3d_clamp(target_velocity_change,
                            robots[i].nitro * NITRO_POINT_VELOCITY_CHANGE);
            if (vec3d_length(target_velocity_change) > 0) {
                Vec3D acceleration = vec3d_normalize(target_velocity_change);
                acceleration.x *= ROBOT_NITRO_ACCELERATION;
                acceleration.y *= ROBOT_NITRO_ACCELERATION;
                acceleration.z *= ROBOT_NITRO_ACCELERATION;
                Vec3D velocity_change =
                    vec3d_clamp((Vec3D){acceleration.x * delta_time,
                                        acceleration.y * delta_time,
                                        acceleration.z * delta_time},
                                vec3d_length(target_velocity_change));
                robots[i].velocity.x += velocity_change.x;
                robots[i].velocity.y += velocity_change.y;
                robots[i].velocity.z += velocity_change.z;
                robots[i].nitro -=
                    vec3d_length(velocity_change) / NITRO_POINT_VELOCITY_CHANGE;
                if (robots[i].nitro < 0)
                    robots[i].nitro = 0;  // Ensure nitro doesn't go negative
            }
        }

        move(&robots[i], delta_time);
        robots[i].radius =
            ROBOT_MIN_RADIUS + (ROBOT_MAX_RADIUS - ROBOT_MIN_RADIUS) *
                                   robots[i].action.jump_speed /
                                   ROBOT_MAX_JUMP_SPEED;
        robots[i].radius_change_speed = robots[i].action.jump_speed;
    }

    move(ball, delta_time);

    for (int i = 0; i < env->n_robots; i++) {
        for (int j = 0; j < i; j++) {
            collide_entities(&robots[i], &robots[j]);
        }
    }

    for (int i = 0; i < env->n_robots; i++) {
        collide_entities(&robots[i], ball);
        Vec3D collision_normal = collide_with_arena(&robots[i]);
        robots[i].touch = (collision_normal.x != 0 || collision_normal.y != 0 ||
                           collision_normal.z != 0);
        if (robots[i].touch) {
            robots[i].touch_normal = collision_normal;
        }
    }
    collide_with_arena(ball);

    for (int i = 0; i < env->n_robots; i++) {
        if (robots[i].nitro == MAX_NITRO_AMOUNT) continue;
        for (int j = 0; j < env->n_nitros; j++) {
            if (!nitro_packs[j].alive) continue;
            Vec3D diff =
                vec3d_subtract(robots[i].position, nitro_packs[j].position);

            if (vec3d_length(diff) <=
                robots[i].radius + nitro_packs[j].radius) {
                robots[i].nitro = MAX_NITRO_AMOUNT;
                nitro_packs[j].alive = false;
                nitro_packs[j].respawn_ticks = NITRO_PACK_RESPAWN_TICKS;
            }
        }
    }
}

float goal_potential(Vec3D position, CodeBallArena* arena, bool side);

void step(CodeBall* env) {
    env->terminal = false;
    memset(env->rewards, 0, env->n_robots * sizeof(float));
    float actions[4];

    for (int i = 0; i < env->n_robots; i++) {
        if (env->is_single) {
            if (i % 2 == 0) {
                for (int j = 0; j < 4; j++) {
                    actions[j] = env->actions[(i / 2) * 4 + j];
                }
            } else {
                switch (env->baseline) {
                    case DO_NOTHING:
                        actions[0] = 0;
                        actions[1] = 0;
                        actions[2] = 0;
                        actions[3] = 0;
                        break;
                    case RANDOM_ACTIONS: {
                        float x_vel = ((float)rand() / RAND_MAX) * 2 - 1;
                        float z_vel = ((float)rand() / RAND_MAX) * 2 - 1;
                        actions[0] = x_vel;
                        actions[1] = z_vel;
                        actions[2] = ((float)rand() / RAND_MAX) > 0.5;
                        actions[3] = ((float)rand() / RAND_MAX) > 0.5;
                        break;
                    }
                    case RUN_TO_BALL: {
                        Vec3D tgt = vec3d_subtract(env->ball.position,
                                                env->robots[i].position);
                        for (int k = 0; k < env->n_robots; k++) {
                            if (k != i) {
                                Vec3D diff = vec3d_subtract(env->robots[k].position,
                                                            env->robots[i].position);
                                double diff_len = vec3d_length(diff);
                                if (diff_len < 2.5) {
                                    tgt = vec3d_multiply(diff, -1.0);
                                }
                            }
                        }
                        tgt = vec3d_multiply(tgt, ROBOT_MAX_GROUND_SPEED);
                        actions[0] = tgt.x;
                        actions[1] = tgt.z;
                        actions[2] = 0;
                        actions[3] = 0;
                        break;
                    }
                }
            }
        } else {
            for (int j = 0; j < 4; j++) {
                actions[j] = env->actions[i * 4 + j];
            }
        }
        env->robots[i].action = (Action){
            .target_velocity = {actions[0] * ROBOT_MAX_GROUND_SPEED, 0.0,
                                actions[1] * ROBOT_MAX_GROUND_SPEED},
            .jump_speed = actions[2] > 0.5 ? ROBOT_MAX_JUMP_SPEED : 0,
            .use_nitro = actions[3] > 0};
    }

    Vec3D ball_initial = env->ball.position;
    Vec3D initial_positions[env->n_robots];
    for (int i = 0; i < env->n_robots; i++) {
        initial_positions[i] = env->robots[i].position;
    }

    float delta_time = 1.0 / TICKS_PER_SECOND;
    for (int i = 0; i < MICROTICKS_PER_TICK * env->frame_skip; i++) {
        update(delta_time / MICROTICKS_PER_TICK, env);
    }
    NitroPack* nitro_packs = env->nitro_packs;
    for (int i = 0; i < env->n_nitros; i++) {
        if (!nitro_packs[i].alive) {
            nitro_packs[i].respawn_ticks--;
            if (nitro_packs[i].respawn_ticks == 0) {
                nitro_packs[i].alive = true;
            }
        }
    }

    Vec3D ball_final = env->ball.position;

    float delta = delta_time * env->frame_skip;
    for (int i = 0; i < env->n_robots; i++) {
        float initial_potential = goal_potential(ball_initial, &arena,
                                                     env->robots[i].side);
        float final_potential = goal_potential(ball_final, &arena,
                                                    env->robots[i].side);
        env->rewards[i] =
            initial_potential == final_potential ? -env->loiter_penalty : 0.0 +
            (initial_potential > final_potential ? env->ball_reward : -env->ball_reward);
    }

    if (fabs(ball_final.z) > arena.depth / 2.0 + env->ball.radius) {
        goal_scored(env, ball_final.z > 0);
    }

    env->log.episode_length++;  // Increment episode length each step
    for (int i = 0; i < env->n_robots; i++) {
        env->log.episode_return_side +=
            env->rewards[i] * ((i % 2) * 2 - 1);
        env->log.episode_return_abs += fabs(env->rewards[i]);
        env->log.episode_return_total += env->rewards[i];
        env->log.episode_return_one += env->rewards[i];
    }

    env->tick++;
}

float goal_potential(Vec3D position,
                                            CodeBallArena *arena,
                                            bool side) {
    if (!side) {
        position.z = -position.z;
    }
    return -position.z;
    float z_offset = arena->depth / 2.0 - position.z;
    if (z_offset < 0) {
        return 0;
    }
    if (fabs(position.x) < arena->goal_width / 2.0) {
        return z_offset;
    }
    float x_offset = fabs(position.x) - arena->goal_width / 2.0;
    return sqrtf(x_offset * x_offset + z_offset * z_offset);
}

void make_observation(CodeBall* env, float *buffer) {
    Entity *ent;
    int target;
    for (int source = 0; source < env->n_robots; source++) {
        if (env->is_single) {
            if (source % 1 == 1) {
                continue;
            }
            target = source / 2;
        } else {
            target = source;
        }
        int o = target * (env->n_robots + 3) * 9;
        Vec3D target_pos = env->robots[target].position;
        for (int i = 0; i < env->n_robots + 2; i++) {
            if (source > 0 && i <= env->n_robots) {
                for (int j = 0; j < 9; j++) {
                    buffer[o + i * 9 + j] = buffer[i * 9 + j];
                }
            }
            if (i < env->n_robots) {
                ent = &env->robots[i];
            } else if (i == env->n_robots) {
                ent = &env->ball;
            } else {
                ent = &env->robots[source];
            }
            buffer[o + i * 9 + 0] = ent->position.x / arena.width * 1.0;
            buffer[o + i * 9 + 1] = ent->position.y / arena.height * 1.0;
            buffer[o + i * 9 + 2] = ent->position.z / arena.depth * 1.0;
            buffer[o + i * 9 + 3] = (ent->position.x - target_pos.x) / arena.width * 1.0;
            buffer[o + i * 9 + 4] = (ent->position.y - target_pos.y) / arena.height * 1.0;
            buffer[o + i * 9 + 5] = (ent->position.z - target_pos.z) / arena.depth * 1.0;
            buffer[o + i * 9 + 6] =
                ent->velocity.x / ROBOT_MAX_GROUND_SPEED * 3.0;
            buffer[o + i * 9 + 7] =
                ent->velocity.y / ROBOT_MAX_GROUND_SPEED * 3.0;
            buffer[o + i * 9 + 8] =
                ent->velocity.z / ROBOT_MAX_GROUND_SPEED * 3.0;
        }
        int base = o + (env->n_robots + 2) * 9;
        // relying on ent being set to the player robot
        buffer[base] = ent->side ? 1.0 : -1.0;
        buffer[base + 1] = (source / (float)env->n_robots) * 2 - 1;
        buffer[base + 2] = env->ball.velocity.y != 0;
    }
}
