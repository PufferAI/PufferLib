#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "raylib.h"

#define NUM_ACTIONS 5
#define NUM_BULLETS 16

float cos_deg(float deg) {
    return cos(deg * 3.14159265358979323846 / 180.0);
}

float sin_deg(float deg) {
    return sin(deg * 3.14159265358979323846 / 180.0);
}

typedef struct Bullet Bullet;
struct Bullet {
    float x;
    float y;
    float heading;
    float firepower;
    bool live;
};

typedef struct Robot Robot;
struct Robot {
    float x;
    float y;
    float v;
    float heading;
    float gun_heading;
    float radar_heading_prev;
    float radar_heading;
    float gun_heat;
    int energy;
    int bullet_idx;
};

typedef struct Env Env;
struct Env {
    int num_agents;
    int width;
    int height;
    Robot* robots;
    Bullet* bullets;
    float* actions;
};

void allocate_env(Env* env) {
    env->robots = (Robot*)calloc(env->num_agents, sizeof(Robot));
    env->bullets = (Bullet*)calloc(NUM_BULLETS*env->num_agents, sizeof(Bullet));
    env->actions = (float*)calloc(NUM_ACTIONS*env->num_agents, sizeof(float));
}

void free_env(Env* env) {
    free(env->robots);
    free(env->actions);
}

void move(Env* env, Robot* robot, float distance) {
    float dx = cos_deg(robot->heading);
    float dy = sin_deg(robot->heading);
    //float accel = 1.0;//2.0*distance / (robot->v * robot->v);
    float accel = distance;

    if (accel > 1.0) {
        accel = 1.0;
    } else if (accel < -2.0) {
        accel = -2.0;
    }

    robot->v += accel;
    if (robot->v > 8.0) {
        robot->v = 8.0;
    } else if (robot->v < -8.0) {
        robot->v = -8.0;
    }

    float new_x = robot->x + dx * robot->v;
    float new_y = robot->y + dy * robot->v;

    // Collision check
    for (int j = 0; j < env->num_agents; j++) {
        Robot* target = &env->robots[j];
        if (target == robot) {
            continue;
        }
        float dx = target->x - new_x;
        float dy = target->y - new_y;
        float dist = sqrt(dx*dx + dy*dy);
        if (dist > 32.0f) {
            continue;
        }

        target->energy -= 0.6;
        robot->energy -= 0.6;
        return;
    }
    
    robot->x = new_x;
    robot->y = new_y;

}

float turn(Env* env, Robot* robot, float degrees) {
    float abs_v = fabs(robot->v);
    float d_angle = 10 - 0.75*abs_v;
    if (degrees > d_angle) {
        degrees = d_angle;
    } else if (degrees < -d_angle) {
        degrees = -d_angle;
    }

    robot->heading += degrees;
    if (robot->heading > 360) {
        robot->heading -= 360;
    } else if (robot->heading < 0) {
        robot->heading += 360;
    }
    return degrees;
}

void fire(Env* env, Robot* robot, float firepower) {
    if (robot->gun_heat > 0) {
        return;
    }
    if (robot->energy < firepower) {
        return;
    }
    robot->energy -= firepower;

    Bullet* bullet = &env->bullets[robot->bullet_idx];
    robot->bullet_idx = (robot->bullet_idx + 1) % NUM_BULLETS;
    robot->gun_heat += 1.0f + firepower/5.0f;

    bullet->x = robot->x + 64*cos_deg(robot->gun_heading);
    bullet->y = robot->y + 64*sin_deg(robot->gun_heading);
    bullet->heading = robot->gun_heading;
    bullet->firepower = firepower;
    bullet->live = true;
}

void reset(Env* env) {
    int idx = 0;
    float x, y;
    while (idx < env->num_agents) {
        Robot* robot = &env->robots[idx];
        x = 16 + rand() % (env->width-32);
        y = 16 + rand() % (env->height-32);
        bool collided = false;
        for (int j = 0; j < idx; j++) {
            Robot* other = &env->robots[j];
            float dx = x - other->x;
            float dy = y - other->y;
            float dist = sqrt(dx*dx + dy*dy);
            if (dist < 32.0f) {
                collided = true;
                break;
            }
        }
        if (!collided) {
            robot->x = x;
            robot->y = y;
            robot->v = 0;
            robot->heading = 0;
            robot->energy = 100;
            robot->gun_heat = 3;
            idx += 1;
        }
    }
}

void step(Env* env) {
    // Update bullets
    for (int agent = 0; agent < env->num_agents; agent++) {
        Robot* robot = &env->robots[agent];
        if (robot->energy <= 0) {
            reset(env);
            return;
        }

        for (int blt = 0; blt < NUM_BULLETS; blt++) {
            Bullet* bullet = &env->bullets[agent*NUM_BULLETS + blt];
            if (!bullet->live) {
                continue;
            }

            float v = 20.0f - 3.0f*bullet->firepower;
            bullet->x += v*cos_deg(bullet->heading);
            bullet->y += v*sin_deg(bullet->heading);

            // Bounds check
            if (bullet->x < 0 || bullet->x > env->width
                    || bullet->y < 0 || bullet->y > env->height) {
                bullet->live = false;
                continue;
            }

            // Collision check
            for (int j = 0; j < env->num_agents; j++) {
                Robot* target = &env->robots[j];
                float dx = target->x - bullet->x;
                float dy = target->y - bullet->y;
                float dist = sqrt(dx*dx + dy*dy);
                if (dist > 32.0f) {
                    continue;
                }

                float damage = 4*bullet->firepower;
                if (bullet->firepower > 1.0f) {
                    damage += 2*(bullet->firepower - 1.0f);
                }

                target->energy -= damage;
                robot->energy += 3*bullet->firepower;
                bullet->live = false;
            }
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        int atn_offset = i*NUM_ACTIONS;

        // Cool down gun
        if (robot->gun_heat > 0) {
            robot->gun_heat -= 0.1f;
        }

        // Move
        int move_atn = env->actions[atn_offset];
        move(env, robot, move_atn);

        // Turn
        int turn_atn = env->actions[atn_offset + 1];
        float turn_degrees = turn(env, robot, turn_atn);

        // Gun 
        float gun_degrees = env->actions[atn_offset + 2] + turn_degrees;
        robot->gun_heading += gun_degrees;
        if (robot->gun_heading > 360) {
            robot->gun_heading -= 360;
        } else if (robot->gun_heading < 0) {
            robot->gun_heading += 360;
        }

        // Radar
        float radar_degrees = env->actions[atn_offset + 3] + gun_degrees;
        robot->radar_heading_prev = robot->radar_heading;
        robot->radar_heading += radar_degrees;
        if (robot->radar_heading > 360) {
            robot->radar_heading -= 360;
        } else if (robot->radar_heading < 0) {
            robot->radar_heading += 360;
        }

        // Fire
        float firepower = env->actions[atn_offset + 4];
        if (firepower > 0) {
            fire(env, robot, firepower);
        }

        // Clip position
        if (robot->x < 16) {
            robot->x = 16;
        } else if (robot->x > env->width - 16) {
            robot->x = env->width - 16;
        }
        if (robot->y < 16) {
            robot->y = 16;
        } else if (robot->y > env->height - 16) {
            robot->y = env->height - 16;
        }
    }
}

typedef struct Client Client;
struct Client {
    Texture2D atlas;
};

Client* make_client(Env* env) {
    InitWindow(768, 576, "PufferLib Ray Robocode");
    SetTargetFPS(60);
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->atlas = LoadTexture("robocode.png");
    return client;
}

void close_client(Client* client) {
    UnloadTexture(client->atlas);
    CloseWindow();
}

void render(Client* client, Env* env) {
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int x = 0; x < env->width; x+=64) {
        for (int y = 0; y < env->height; y+=64) {
            int src_x = 64 * ((x*33409+ y*30971) % 5);
            Rectangle src_rect = (Rectangle){src_x, 0, 64, 64};
            Vector2 dest_pos = (Vector2){x, y};
            DrawTextureRec(client->atlas, src_rect, dest_pos, WHITE);
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        int atn_offset = i*NUM_ACTIONS;
        int turn_atn = env->actions[atn_offset + 1];
        int gun_atn = env->actions[atn_offset + 2] + turn_atn;
        int radar_atn = env->actions[atn_offset + 3] + gun_atn;

        Robot robot = env->robots[i];
        Vector2 robot_pos = (Vector2){robot.x, robot.y};

        // Radar
        float radar_left = (radar_atn > 0) ? robot.radar_heading: robot.radar_heading_prev;
        float radar_right = (radar_atn > 0) ? robot.radar_heading_prev : robot.radar_heading;
        Vector2 radar_left_pos = (Vector2){
            robot.x + 1200*cos_deg(radar_left),
            robot.y + 1200*sin_deg(radar_left)
        };
        Vector2 radar_right_pos = (Vector2){
            robot.x + 1200*cos_deg(radar_right),
            robot.y + 1200*sin_deg(radar_right)
        };
        DrawTriangle(robot_pos, radar_left_pos, radar_right_pos, (Color){0, 255, 0, 128});

        // Gun 
        Vector2 gun_pos = (Vector2){
            robot.x + 64*cos_deg(robot.gun_heading),
            robot.y + 64*sin_deg(robot.gun_heading)
        };
        //DrawLineEx(robot_pos, gun_pos, 4, WHITE);

        // Robot
        //DrawCircle(robot.x, robot.y, 32, RED);
        //DrawCircle(robot.x, robot.y, 16, WHITE);
        float theta = robot.heading;
        float dx = cos_deg(theta);
        float dy = sin_deg(theta);
        int src_y = 64 + 64*(i%2);
        Rectangle body_rect = (Rectangle){0, src_y, 64, 64};
        Rectangle radar_rect = (Rectangle){64, src_y, 64, 64};
        Rectangle gun_rect = (Rectangle){128, src_y, 64, 64};
        Rectangle dest_rect = (Rectangle){robot.x, robot.y, 64, 64};
        Vector2 origin = (Vector2){32, 32};
        DrawTexturePro(client->atlas, body_rect, dest_rect, origin, robot.heading+90, WHITE);
        DrawTexturePro(client->atlas, radar_rect, dest_rect, origin, robot.radar_heading+90, WHITE);
        DrawTexturePro(client->atlas, gun_rect, dest_rect, origin, robot.gun_heading+90, WHITE);

        DrawText(TextFormat("%i", robot.energy), robot.x-16, robot.y-48, 12, WHITE);
    }

    for (int i = 0; i < env->num_agents*NUM_BULLETS; i++) {
        Bullet bullet = env->bullets[i];
        if (!bullet.live) {
            continue;
        }
        Vector2 bullet_pos = (Vector2){bullet.x, bullet.y};
        DrawCircleV(bullet_pos, 4, WHITE);
    }

    EndDrawing();
}
