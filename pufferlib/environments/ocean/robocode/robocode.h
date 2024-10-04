#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "raylib.h"

float cos_deg(float deg) {
    return cos(deg * 3.14159265358979323846 / 180.0);
}

float sin_deg(float deg) {
    return sin(deg * 3.14159265358979323846 / 180.0);
}

typedef struct Robot Robot;
struct Robot {
    float x;
    float y;
    float v;
    float heading;
    float gun_heading;
    float radar_heading;
    float gun_heat;
    int energy;
    int health;
};

typedef struct Env Env;
struct Env {
    int num_agents;
    int width;
    int height;
    Robot* robots;
    unsigned char* actions;
};

void allocate_env(Env* env) {
    env->robots = (Robot*)calloc(env->num_agents, sizeof(Robot));
    env->actions = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
}

void free_env(Env* env) {
    free(env->robots);
    free(env->actions);
}

void reset(Env* env) {
    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        robot->x = rand() % env->width;
        robot->y = rand() % env->height;
        robot->v = 0;
        robot->heading = 0;
        robot->energy = 100;
        robot->health = 100;
    }
}

void step(Env* env) {
    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        unsigned char atn = env->actions[i];
        printf("Robot %i atn: %i\n", i, atn);
        if (atn == 0) {
            robot->v += 1;
        } else if (atn == 1) {
            robot->v -= 1;
        } else {
            float abs_v = fabs(robot->v);
            float d_angle = 10 - 0.75*abs_v;

            if (atn == 2) {
                robot->heading -= d_angle;
            } else if (atn == 3) {
                robot->heading += d_angle;
            }
        }

        float dx = cos(robot->heading * 3.14159265358979323846 / 180.0);
        float dy = sin(robot->heading * 3.14159265358979323846 / 180.0);
        robot->x += dx * robot->v;
        robot->y += dy * robot->v;
        
        if (robot->x < 0) {
            robot->x = 0;
        } else if (robot->x > env->width) {
            robot->x = env->width;
        }
        if (robot->y < 0) {
            robot->y = 0;
        } else if (robot->y > env->height) {
            robot->y = env->height;
        }

        if (robot->heading > 360) {
            robot->heading -= 360;
        }
        if (robot->heading < 0) {
            robot->heading += 360;
        }

    }
}

void init_client(Env* env) {
    InitWindow(1080, 720, "PufferLib Ray Robocode");
    SetTargetFPS(60);
}

void close_client(Env* env) {
    CloseWindow();
}

void render(Env* env) {
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int i = 0; i < env->num_agents; i++) {
        Robot robot = env->robots[i];
        DrawCircle(robot.x, robot.y, 32, RED);
        DrawCircle(robot.x, robot.y, 16, WHITE);

        float theta = robot.heading;
        float dx = cos_deg(theta);
        float dy = sin_deg(theta);
        DrawLine(robot.x, robot.y, robot.x + dx*64, robot.y + dy*64, WHITE);

        // Draw radar
        float t_left = robot.radar_heading - 20;
        float t_right = robot.radar_heading + 20;
        float x_left = robot.x + 256*cos_deg(t_left);
        float y_left = robot.y + 256*sin_deg(t_left);
        float x_right = robot.x + 256*cos_deg(t_right);
        float y_right = robot.y + 256*sin_deg(t_right);
        DrawTriangle(
            (Vector2){x_left, y_left},
            (Vector2){x_right, y_right},
            (Vector2){robot.x, robot.y},
            GREEN
        );

        //float health = robot.health / 100.0;
        //DrawRectangle(robot.x, robot.y - 10, 64, 16, RED);
        //DrawRectangle(robot.x, robot.y - 10, 64*health, 16, GREEN);
    }

    EndDrawing();
}
