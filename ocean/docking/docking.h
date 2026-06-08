#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

#define DOCKING_OBS_SIZE 8

const unsigned char DOCK_NOOP = 0;
const unsigned char DOCK_TURN_LEFT = 1;
const unsigned char DOCK_TURN_RIGHT = 2;
const unsigned char DOCK_THRUST = 3;
const unsigned char DOCK_BRAKE = 4;

const unsigned char DOCK_RESULT_SUCCESS = 0;
const unsigned char DOCK_RESULT_CRASH = 1;
const unsigned char DOCK_RESULT_TIMEOUT = 2;

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float success_rate;
    float crash_rate;
    float timeout_rate;
    float final_distance;
    float alignment_error;
    float n;
} Log;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int width;
    int height;
    int max_ticks;
    int tick;
    float ship_x;
    float ship_y;
    float ship_heading;
    float ship_speed;
    float dock_x;
    float dock_y;
    float dock_heading;
    float prev_distance;
    float episode_return;
    float max_speed;
    float turn_rate;
    float accel;
    float drag;
    float dock_radius;
    float dock_speed_threshold;
    float dock_heading_threshold;
    float step_penalty;
    float progress_reward_scale;
    int feedback_timer;
    unsigned char last_result;
    unsigned char last_action;
    unsigned char reset_pending;
    unsigned int rng;
} Docking;

static inline float docking_clipf(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

static inline float docking_randf(Docking* env) {
    return rand_r(&env->rng) / (float)RAND_MAX;
}

static inline float docking_wrap_angle(float angle) {
    while (angle < 0.0f) {
        angle += 2.0f*PI;
    }
    while (angle >= 2.0f*PI) {
        angle -= 2.0f*PI;
    }
    return angle;
}

static inline float docking_angle_error(float a, float b) {
    float diff = fabsf(docking_wrap_angle(a) - docking_wrap_angle(b));
    return fminf(diff, 2.0f*PI - diff);
}

static inline float docking_distance(Docking* env) {
    float dx = env->dock_x - env->ship_x;
    float dy = env->dock_y - env->ship_y;
    return sqrtf(dx*dx + dy*dy);
}

static inline float docking_diag(Docking* env) {
    return sqrtf((float)(env->width*env->width + env->height*env->height));
}

void c_init(Docking* env) {
    env->num_agents = 1;
    env->width = env->width < 64 ? 64 : env->width;
    env->height = env->height < 64 ? 64 : env->height;
    env->max_ticks = env->max_ticks == 0 ? 1024 : env->max_ticks;
    if (env->max_ticks > 0 && env->max_ticks < 16) env->max_ticks = 16;
    env->max_speed = env->max_speed <= 0.0f ? 6.0f : env->max_speed;
    env->turn_rate = env->turn_rate <= 0.0f ? 0.10f : env->turn_rate;
    env->accel = env->accel <= 0.0f ? 0.55f : env->accel;
    env->drag = docking_clipf(env->drag, 0.0f, 1.0f);
    if (env->drag == 0.0f) env->drag = 0.92f;
    env->dock_radius = env->dock_radius <= 1.0f ? 24.0f : env->dock_radius;
    env->dock_speed_threshold = env->dock_speed_threshold <= 0.0f ? 0.75f : env->dock_speed_threshold;
    env->dock_heading_threshold = env->dock_heading_threshold <= 0.0f ? 0.30f : env->dock_heading_threshold;
    env->step_penalty = env->step_penalty == 0.0f ? -0.01f : env->step_penalty;
    env->progress_reward_scale = env->progress_reward_scale == 0.0f ? 0.25f : env->progress_reward_scale;
    env->feedback_timer = 0;
    env->last_result = DOCK_RESULT_TIMEOUT;
    env->last_action = DOCK_NOOP;
    env->reset_pending = 0;
}

void compute_observations(Docking* env) {
    float dx = env->dock_x - env->ship_x;
    float dy = env->dock_y - env->ship_y;
    float dist = sqrtf(dx*dx + dy*dy);
    float diag = docking_diag(env);

    memset(env->observations, 0, DOCKING_OBS_SIZE * sizeof(float));
    env->observations[0] = dx / env->width;
    env->observations[1] = dy / env->height;
    env->observations[2] = cosf(env->ship_heading);
    env->observations[3] = sinf(env->ship_heading);
    env->observations[4] = cosf(env->dock_heading);
    env->observations[5] = sinf(env->dock_heading);
    env->observations[6] = env->ship_speed / env->max_speed;
    env->observations[7] = dist / diag;
}

void add_log(Docking* env, unsigned char result) {
    float dist = docking_distance(env) / docking_diag(env);
    float angle_error = docking_angle_error(env->ship_heading, env->dock_heading) / PI;

    env->log.perf += result == DOCK_RESULT_SUCCESS ? 1.0f : 0.0f;
    env->log.score += env->rewards[0];
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.success_rate += result == DOCK_RESULT_SUCCESS ? 1.0f : 0.0f;
    env->log.crash_rate += result == DOCK_RESULT_CRASH ? 1.0f : 0.0f;
    env->log.timeout_rate += result == DOCK_RESULT_TIMEOUT ? 1.0f : 0.0f;
    env->log.final_distance += dist;
    env->log.alignment_error += angle_error;
    env->log.n += 1.0f;
}

void c_reset(Docking* env) {
    float width = (float)env->width;
    float height = (float)env->height;

    env->tick = 0;
    env->episode_return = 0.0f;

    env->ship_x = width * (0.10f + 0.20f * docking_randf(env));
    env->ship_y = height * (0.20f + 0.60f * docking_randf(env));
    env->ship_heading = docking_wrap_angle((-0.35f + 0.70f * docking_randf(env)) * PI);
    env->ship_speed = 0.0f;
    env->last_action = DOCK_NOOP;
    env->reset_pending = 0;

    env->dock_x = width * (0.65f + 0.20f * docking_randf(env));
    env->dock_y = height * (0.20f + 0.60f * docking_randf(env));
    env->dock_heading = 0.0f;

    env->prev_distance = docking_distance(env);
    compute_observations(env);
}

void finish_episode(Docking* env, float reward, unsigned char result) {
    env->rewards[0] = reward;
    env->terminals[0] = 1.0f;
    env->episode_return += reward;
    env->last_result = result;
    env->feedback_timer = 72;
    add_log(env, result);
    env->reset_pending = 1;
}

void c_step(Docking* env) {
    if (env->reset_pending) {
        if (IsWindowReady() && env->feedback_timer > 66) {
            env->feedback_timer -= 1;
            return;
        }
        c_reset(env);
    }

    env->tick += 1;
    if (env->feedback_timer > 0) {
        env->feedback_timer -= 1;
    }

    memset(env->observations, 0, DOCKING_OBS_SIZE * sizeof(float));
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    env->last_action = (unsigned char)action;
    if (action == DOCK_TURN_LEFT) {
        env->ship_heading -= env->turn_rate;
    } else if (action == DOCK_TURN_RIGHT) {
        env->ship_heading += env->turn_rate;
    }
    env->ship_heading = docking_wrap_angle(env->ship_heading);

    if (action == DOCK_THRUST) {
        env->ship_speed += env->accel;
    } else if (action == DOCK_BRAKE) {
        env->ship_speed -= 2.2f * env->accel;
    }
    env->ship_speed = docking_clipf(env->ship_speed, 0.0f, env->max_speed);
    env->ship_speed *= env->drag;

    env->ship_x += env->ship_speed * cosf(env->ship_heading);
    env->ship_y += env->ship_speed * sinf(env->ship_heading);

    int hit_x = 0;
    int hit_y = 0;
    if (env->ship_x < 0.0f) {
        env->ship_x = 0.0f;
        hit_x = 1;
    } else if (env->ship_x > env->width) {
        env->ship_x = (float)env->width;
        hit_x = 1;
    }
    if (env->ship_y < 0.0f) {
        env->ship_y = 0.0f;
        hit_y = 1;
    } else if (env->ship_y > env->height) {
        env->ship_y = (float)env->height;
        hit_y = 1;
    }
    if (hit_x || hit_y) {
        if (env->ship_speed > 0.55f * env->max_speed) {
            finish_episode(env, -1.0f, DOCK_RESULT_CRASH);
            return;
        }
        if (hit_x) {
            env->ship_heading = PI - env->ship_heading;
        }
        if (hit_y) {
            env->ship_heading = -env->ship_heading;
        }
        env->ship_heading = docking_wrap_angle(env->ship_heading);
        env->ship_speed *= 0.32f;
    }

    float dist = docking_distance(env);
    float dist_delta = (env->prev_distance - dist) / docking_diag(env);
    env->prev_distance = dist;

    float reward = env->step_penalty + env->progress_reward_scale * dist_delta;
    reward = docking_clipf(reward, -1.0f, 1.0f);
    env->rewards[0] = reward;
    env->episode_return += reward;

    float angle_error = docking_angle_error(env->ship_heading, env->dock_heading);
    float bay_depth = env->dock_radius * 1.75f;
    float dock_entrance_x = env->dock_x - cosf(env->dock_heading) * bay_depth * 0.50f;
    float dock_entrance_y = env->dock_y - sinf(env->dock_heading) * bay_depth * 0.50f;
    float dx_to_entrance = env->ship_x - dock_entrance_x;
    float dy_to_entrance = env->ship_y - dock_entrance_y;
    float dist_to_entrance = sqrtf(dx_to_entrance * dx_to_entrance + dy_to_entrance * dy_to_entrance);

    float success_radius = env->dock_radius * 1.40f;
    float soft_radius = env->dock_radius * 1.60f;
    float crash_radius = env->dock_radius * 1.01f;

    if (dist_to_entrance <= soft_radius) {
        int can_dock = (angle_error <= env->dock_heading_threshold + 0.12f
            && env->ship_speed <= 2.0f * env->dock_speed_threshold);
        int soft_approach = (env->ship_speed <= 2.6f * env->dock_speed_threshold
            && angle_error <= env->dock_heading_threshold + 0.35f);

        if (dist_to_entrance <= success_radius && can_dock) {
            env->ship_x = dock_entrance_x;
            env->ship_y = dock_entrance_y;
            env->ship_speed = 0.0f;
            env->ship_heading = env->dock_heading;
            finish_episode(env, 1.0f, DOCK_RESULT_SUCCESS);
        } else if (dist_to_entrance <= crash_radius) {
            finish_episode(env, -1.0f, DOCK_RESULT_CRASH);
        } else if (soft_approach) {
            env->ship_speed *= 0.35f;
            env->prev_distance = docking_distance(env);
            compute_observations(env);
        } else {
            float inv_dist = dist_to_entrance > 1e-5f ? 1.0f / dist_to_entrance : 0.0f;
            if (dist_to_entrance <= 1e-5f) {
                dx_to_entrance = -cosf(env->dock_heading);
                dy_to_entrance = -sinf(env->dock_heading);
                inv_dist = 1.0f;
            }
            env->ship_x = dock_entrance_x + dx_to_entrance * inv_dist * (soft_radius + 1.5f);
            env->ship_y = dock_entrance_y + dy_to_entrance * inv_dist * (soft_radius + 1.5f);
            env->ship_speed *= 0.25f;

            float old_reward = env->rewards[0];
            env->rewards[0] = docking_clipf(old_reward - 0.10f, -1.0f, 1.0f);
            env->episode_return += env->rewards[0] - old_reward;
            env->prev_distance = docking_distance(env);
            compute_observations(env);
        }
        return;
    }

    if (env->max_ticks > 0 && env->tick >= env->max_ticks) {
        finish_episode(env, -0.25f, DOCK_RESULT_TIMEOUT);
        return;
    }

    compute_observations(env);
}

void draw_docking_bay(Vector2 center, Vector2 dir, float radius, Color bay_color, Color target_color) {
    Vector2 side = {-dir.y, dir.x};
    float bay_half_width = radius * 1.25f;
    float bay_depth = radius * 1.75f;
    float target_half_width = radius * 0.72f;
    float target_depth = radius * 0.95f;

    Vector2 mouth_center = {
        center.x - dir.x * bay_depth * 0.50f,
        center.y - dir.y * bay_depth * 0.50f,
    };
    Vector2 back_center = {
        center.x + dir.x * bay_depth * 0.50f,
        center.y + dir.y * bay_depth * 0.50f,
    };
    Vector2 target_center = {
        center.x + dir.x * bay_depth * 0.10f,
        center.y + dir.y * bay_depth * 0.10f,
    };

    Vector2 mouth_left = {
        mouth_center.x + side.x * bay_half_width,
        mouth_center.y + side.y * bay_half_width,
    };
    Vector2 mouth_right = {
        mouth_center.x - side.x * bay_half_width,
        mouth_center.y - side.y * bay_half_width,
    };
    Vector2 back_left = {
        back_center.x + side.x * bay_half_width,
        back_center.y + side.y * bay_half_width,
    };
    Vector2 back_right = {
        back_center.x - side.x * bay_half_width,
        back_center.y - side.y * bay_half_width,
    };
    Vector2 target_mouth_left = {
        target_center.x - dir.x * target_depth * 0.50f + side.x * target_half_width,
        target_center.y - dir.y * target_depth * 0.50f + side.y * target_half_width,
    };
    Vector2 target_mouth_right = {
        target_center.x - dir.x * target_depth * 0.50f - side.x * target_half_width,
        target_center.y - dir.y * target_depth * 0.50f - side.y * target_half_width,
    };
    Vector2 target_back_left = {
        target_center.x + dir.x * target_depth * 0.50f + side.x * target_half_width,
        target_center.y + dir.y * target_depth * 0.50f + side.y * target_half_width,
    };
    Vector2 target_back_right = {
        target_center.x + dir.x * target_depth * 0.50f - side.x * target_half_width,
        target_center.y + dir.y * target_depth * 0.50f - side.y * target_half_width,
    };

    DrawLineEx(mouth_left, back_left, 6.0f, bay_color);
    DrawLineEx(mouth_right, back_right, 6.0f, bay_color);
    DrawLineEx(back_left, back_right, 6.0f, bay_color);
    DrawTriangle(target_mouth_left, target_mouth_right, target_back_left, Fade(target_color, 0.35f));
    DrawTriangle(target_mouth_right, target_back_left, target_back_right, Fade(target_color, 0.35f));
    DrawLineEx(target_mouth_left, target_back_left, 3.0f, target_color);
    DrawLineEx(target_mouth_right, target_back_right, 3.0f, target_color);
    DrawLineEx(target_back_left, target_back_right, 3.0f, target_color);
}

void draw_ship(Vector2 center, Vector2 dir, float ship_len, float ship_wid, Color hull_color) {
    Vector2 side = {-dir.y, dir.x};
    Vector2 nose = {
        center.x + dir.x * ship_len,
        center.y + dir.y * ship_len,
    };
    Vector2 left = {
        center.x - dir.x * ship_len * 0.55f + side.x * ship_wid,
        center.y - dir.y * ship_len * 0.55f + side.y * ship_wid,
    };
    Vector2 right = {
        center.x - dir.x * ship_len * 0.55f - side.x * ship_wid,
        center.y - dir.y * ship_len * 0.55f - side.y * ship_wid,
    };
    Vector2 cockpit = {
        center.x + dir.x * ship_len * 0.18f,
        center.y + dir.y * ship_len * 0.18f,
    };

    DrawTriangle(nose, left, right, hull_color);
    DrawTriangleLines(nose, left, right, RAYWHITE);
    DrawCircleV(cockpit, ship_wid * 0.22f, (Color){220, 245, 255, 255});
}

void draw_thruster(Vector2 center, Vector2 dir, float ship_len, float ship_wid) {
    Vector2 side = {-dir.y, dir.x};
    Vector2 base = {
        center.x - dir.x * ship_len * 0.72f,
        center.y - dir.y * ship_len * 0.72f,
    };
    float flame_len = ship_len * 0.90f;
    Vector2 flame_tip = {
        base.x - dir.x * flame_len,
        base.y - dir.y * flame_len,
    };
    Vector2 left = {
        base.x + side.x * ship_wid * 0.32f,
        base.y + side.y * ship_wid * 0.32f,
    };
    Vector2 right = {
        base.x - side.x * ship_wid * 0.32f,
        base.y - side.y * ship_wid * 0.32f,
    };
    Vector2 inner_tip = {
        base.x - dir.x * flame_len * 0.55f,
        base.y - dir.y * flame_len * 0.55f,
    };

    DrawTriangle(flame_tip, left, right, (Color){255, 140, 40, 220});
    DrawTriangle(inner_tip, left, right, (Color){255, 220, 100, 220});
}

void c_render(Docking* env) {
    if (!IsWindowReady()) {
        int screen_width = 960;
        int screen_height = (int)(screen_width * (env->height / (float)env->width));
        if (screen_height < 540) screen_height = 540;
        InitWindow(screen_width, screen_height, "PufferLib Docking");
        SetTargetFPS(60);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    float margin = 40.0f;
    float scale_x = (GetScreenWidth() - 2.0f * margin) / env->width;
    float scale_y = (GetScreenHeight() - 2.0f * margin) / env->height;
    float scale = fminf(scale_x, scale_y);
    float ox = 0.5f * (GetScreenWidth() - env->width * scale);
    float oy = 0.5f * (GetScreenHeight() - env->height * scale);

    Vector2 dock_center = {
        ox + env->dock_x * scale,
        oy + env->dock_y * scale,
    };
    Vector2 dock_dir = {
        cosf(env->dock_heading),
        sinf(env->dock_heading),
    };
    Vector2 ship_center = {
        ox + env->ship_x * scale,
        oy + env->ship_y * scale,
    };
    Vector2 ship_dir = {
        cosf(env->ship_heading),
        sinf(env->ship_heading),
    };

    float ship_len = fmaxf(24.0f, env->dock_radius * scale * 1.35f);
    float ship_wid = ship_len * 0.50f;
    float speed_ratio = env->ship_speed / env->max_speed;
    float angle_error = docking_angle_error(env->ship_heading, env->dock_heading);
    Color bay_color = (Color){120, 180, 255, 255};
    Color target_color = (Color){60, 255, 140, 255};
    Color ship_color = (Color){0, 205, 215, 255};

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    if (env->feedback_timer > 0) {
        float alpha = env->feedback_timer / 72.0f;
        if (env->last_result == DOCK_RESULT_SUCCESS) {
            DrawRectangle(0, 0, GetScreenWidth(), GetScreenHeight(), Fade((Color){40, 170, 90, 255}, 0.28f * alpha));
        } else if (env->last_result == DOCK_RESULT_CRASH) {
            DrawRectangle(0, 0, GetScreenWidth(), GetScreenHeight(), Fade((Color){180, 40, 40, 255}, 0.12f * alpha));
        }
    }

    DrawRectangleLines(
        (int)ox, (int)oy,
        (int)(env->width * scale), (int)(env->height * scale),
        (Color){60, 110, 110, 255}
    );

    draw_docking_bay(dock_center, dock_dir, env->dock_radius * scale, bay_color, target_color);
    DrawLineEx(
        (Vector2){
            dock_center.x - dock_dir.x * env->dock_radius * scale * 2.4f,
            dock_center.y - dock_dir.y * env->dock_radius * scale * 2.4f,
        },
        dock_center,
        2.0f,
        Fade(target_color, 0.55f)
    );

    if (env->last_action == DOCK_THRUST && env->ship_speed > 0.05f) {
        draw_thruster(ship_center, ship_dir, ship_len, ship_wid);
    }
    draw_ship(ship_center, ship_dir, ship_len, ship_wid, ship_color);

    DrawRectangle(20, 18, 220, 118, Fade(BLACK, 0.25f));
    DrawText("Dock slowly into the green bay", 32, 28, 22, RAYWHITE);
    DrawText(TextFormat("speed  %.2f / %.2f", env->ship_speed, env->max_speed), 32, 58, 20, RAYWHITE);
    DrawText(TextFormat("align  %.2f", angle_error), 32, 84, 20, RAYWHITE);
    if (env->max_ticks > 0) {
        DrawText(TextFormat("steps  %d / %d", env->tick, env->max_ticks), 32, 110, 20, RAYWHITE);
    } else {
        DrawText(TextFormat("steps  %d", env->tick), 32, 110, 20, RAYWHITE);
    }

    DrawRectangle(32, GetScreenHeight() - 36, 180, 12, Fade(RAYWHITE, 0.15f));
    DrawRectangle(32, GetScreenHeight() - 36, (int)(180.0f * speed_ratio), 12, ship_color);

    if (env->feedback_timer > 0) {
        const char* text = "TIMEOUT";
        Color text_color = (Color){255, 220, 120, 255};
        if (env->last_result == DOCK_RESULT_SUCCESS) {
            text = "DOCKED";
            text_color = (Color){60, 255, 140, 255};
        } else if (env->last_result == DOCK_RESULT_CRASH) {
            text = "CRASH";
            text_color = (Color){255, 90, 90, 255};
        }
        int font_size = env->last_result == DOCK_RESULT_SUCCESS ? 58 : 42;
        int text_width = MeasureText(text, font_size);
        DrawRectangle(
            GetScreenWidth()/2 - text_width/2 - 24,
            18,
            text_width + 48,
            env->last_result == DOCK_RESULT_SUCCESS ? 78 : 60,
            Fade(BLACK, 0.35f)
        );
        DrawText(text, GetScreenWidth()/2 - text_width/2, env->last_result == DOCK_RESULT_SUCCESS ? 28 : 30, font_size, text_color);
    }

    EndDrawing();
}

void c_close(Docking* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
