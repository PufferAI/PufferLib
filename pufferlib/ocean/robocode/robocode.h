//puffer train puffer_robocode
//puffer eval puffer_robocode
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <unistd.h> 
#include "raylib.h"
// #include "logger.h"
#define NUM_ACTIONS 5
#define NUM_BULLETS 16
#define MAX_VELOCITY 8.0f
// Reward
#define HIT_ENEMY_REWARD 10.0f
#define GOT_HIT_PENALTY 10.0f
#define RELATIVE_ENERGY_SCALE 0.1f

//IDS
#define AGENT_IDX 0
#define ADVERSARIAL_IDX 1

//RADAR
#define MAX_RADAR_DISTANCE 1200

// Deseralise Actions Idxs
#define MOVE_IDX 0
#define TURN_IDX 1
#define GUN_TURN_IDX 2
#define RADAR_TURN_IDX 3
#define FIRE_IDX 4


typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

typedef struct {
    Texture2D atlas;
} Client;

float cos_deg(float deg) {
    return cos(deg * PI / 180.0);
}

float sin_deg(float deg) {
    return sin(deg * PI / 180.0);
}

typedef struct Pose{
    float x;
    float y;
    float heading;
} Pose;

typedef struct Bullet Bullet;
struct Bullet {
    Pose pose;
    float firepower;
    bool live;
};

typedef struct Robot Robot;
struct Robot {
    int id;
    Pose pose;
    float vel;
    float gun_heading;
    float radar_heading_prev;
    float radar_heading;
    float gun_heat;
    int energy;
    int bullet_idx;

    bool enemy_detected;
    float enemy_distance;
    float enemy_bearing;

    bool was_hit;
    bool hit_enemy;
};

typedef  struct {
    Log log;
    Client* client; 
    int tick;
    int num_agents;
    int width;
    int height;
    Robot* robots;
    Bullet* bullets;

    //RL MDP
    float* actions;
    float* rewards;
    unsigned char* terminals;
    float* info;
    float* observations;
    
}Robocode;

void init(Robocode* env) {
    env->robots = (Robot*)calloc(env->num_agents, sizeof(Robot));
    env->bullets = (Bullet*)calloc(NUM_BULLETS*env->num_agents, sizeof(Bullet));
}

void move(Robocode* env, Robot* robot, float distance) {
    float dx = cos_deg(robot->pose.heading);
    float dy = sin_deg(robot->pose.heading);
    //float accel = 1.0;//2.0*distance / (robot->v * robot->v);
    float accel = distance;
    accel = fmax(-2.0f, fmin(1.0f, accel));

    float new_vel = robot->vel + accel;
    new_vel = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, new_vel));


    float new_pose_x = robot->pose.x + dx * new_vel;
    float new_pose_y = robot->pose.y + dy * new_vel;

    //Notes: Add Boundary check?
    // if (new_pose_x < 16 || new_pose_x > env->width - 16 || 
    //     new_pose_y < 16 || new_pose_y > env->height - 16) {
    //     // If we would hit boundary, stop movement
    //     robot->vel = 0;
    //     return;
    // }

    // Collision check
    for (int j = 0; j < env->num_agents; j++) {
        Robot* target = &env->robots[j];
        if (target == robot) {
            continue;
        }
        float dx = target->pose.x - new_pose_x;
        float dy = target->pose.y - new_pose_y;
        float dist = sqrt(dx*dx + dy*dy);
        if (dist > 32.0f) {
            continue;
        }
        //Note: return once code is done
        target->energy -= 0.6;
        robot->energy -= 0.6;
        robot->vel = 0; // stop on collision
        // Notes: maybe add a penalty and set the was hit boolean to true?
        return;
    }
    robot->vel = new_vel;
    robot->pose.x = new_pose_x;
    robot->pose.y = new_pose_y;
}

float turn(Robocode* env, Robot* robot, float degrees) {
    float abs_v = fabs(robot->vel);
    float d_angle = 10 - 0.75 * abs_v;
    degrees = fmax(-d_angle, fmin(d_angle, degrees));

    robot->pose.heading += degrees;

    if (robot->pose.heading > 360) {
        robot->pose.heading -= 360;
    } else if (robot->pose.heading < 0) {
        robot->pose.heading += 360;
    }
    return degrees;
}

void fire(Robocode* env, Robot* robot, float firepower) {
    if (robot->gun_heat > 0) {
        return;
    }
    if (robot->energy < firepower) {
        return;
    }
    robot->energy -= firepower;

    Bullet* bullet = &env->bullets[robot->id * NUM_BULLETS + robot->bullet_idx];

    robot->bullet_idx = (robot->bullet_idx + 1) % NUM_BULLETS;
    robot->gun_heat += 1.0f + firepower/5.0f;

    //Update bullet
    bullet->pose.x = robot->pose.x + 64*cos_deg(robot->gun_heading);
    bullet->pose.y = robot->pose.y + 64*sin_deg(robot->gun_heading);
    bullet->pose.heading = robot->gun_heading;
    bullet->firepower = firepower;
    bullet->live = true;
}

void adversarial_agent_step(Robocode* env){
    /*
    Move Forward by 1.0
    Turn Gun Right by 1.0
    back(0.5)
    turnGunRIght by 3.0
    if enemy detected
        fire(1.0)
    if we were hit
        turn perpendicular to the bullet

    */
    Robot* adversarial = &env->robots[ADVERSARIAL_IDX]; // Hardcoded Adversarial
    Robot* agent = &env->robots[AGENT_IDX]; // Our Agent
    int atn_offset = ADVERSARIAL_IDX * NUM_ACTIONS;

    // reset actions
    for(int i = 0; i < NUM_ACTIONS; i++){
        env->actions[atn_offset + i] = 0.0f;
    }
    // NEW STUFF
    // angle to target (agent) cause we are cheating
    float dx = agent->pose.x - adversarial->pose.x;
    float dy = agent->pose.y - adversarial->pose.y;
    float angle_to_op = 180 * atan2(dy, dx) / PI;
    float gun_delta = angle_to_op - adversarial->gun_heading;


    // Normalize gun delta to -180 to 180 range
    if (gun_delta < -180) gun_delta += 360;
    if (gun_delta > 180) gun_delta -= 360; 

    //movement
    static int state = 0;
    static int steps = 0;
    // Simple control to move back and forth (Change Direction after a few steps)
    //Note: Add Mechanism to move tangentilal to the bullet direction when hit
    switch (state)
    {
        case 0:
            env->actions[atn_offset + MOVE_IDX] = 1.0f;  // forward
            steps++;
            if(steps >= 100){
                state = 1;
                steps = 0;
            }
            break;
        case  1:    
            env->actions[atn_offset + MOVE_IDX] = -1.0f;  // backward
            steps++;
            if(steps >= 100){
                state = 0;
                steps = 0;
            }
            break;

        default:
            break;
    }

    // If enemy was detected by radar
    if (adversarial->enemy_detected) {
        // Turn gun based on enemy bearing
        // Note: compute enemy bearing
        float gun_delta = adversarial->enemy_bearing - 
                         (adversarial->gun_heading - adversarial->pose.heading);
        
        // Normalize gun delta to -180 to 180 range
        while (gun_delta > 180) gun_delta -= 360;
        while (gun_delta < -180) gun_delta += 360;

        // Turn gun towards enemy
        // Note: Not sure about thiis section.
        // Still sweeping oriented towards the enemy
        env->actions[atn_offset + GUN_TURN_IDX] = (gun_delta > 0) ? 3.0f : -3.0f;

        // Fire when aligned
        if (fabs(gun_delta) < 5) {
            env->actions[atn_offset + FIRE_IDX] = 1.0f;
        }
    }

}

bool line_segment_intersects(float x1, float y1, float x2, float y2,
                           float x3, float y3, float x4, float y4) {
    // Calculate denominators for parameters
    float denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
    if (denom == 0) return false;  // Lines are parallel

    float ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
    float ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;

    // Check if intersection occurs within both line segments
    return (ua >= 0 && ua <= 1) && (ub >= 0 && ub <= 1);
}

// Function to check if a line segment intersects a rectangle
bool check_line_intersection(float x1, float y1, float x2, float y2,
                           float rectX, float rectY, float rectW, float rectH) {
    // Rectangle edges as line segments
    float rectTopX1 = rectX, rectTopY1 = rectY;
    float rectTopX2 = rectX + rectW, rectTopY2 = rectY;
    float rectBottomX1 = rectX, rectBottomY1 = rectY + rectH;
    float rectBottomX2 = rectX + rectW, rectBottomY2 = rectY + rectH;
    float rectLeftX1 = rectX, rectLeftY1 = rectY;
    float rectLeftX2 = rectX, rectLeftY2 = rectY + rectH;
    float rectRightX1 = rectX + rectW, rectRightY1 = rectY;
    float rectRightX2 = rectX + rectW, rectRightY2 = rectY + rectH;

    // Check against all rectangle edges
    return (line_segment_intersects(x1, y1, x2, y2, rectTopX1, rectTopY1, rectTopX2, rectTopY2) ||
            line_segment_intersects(x1, y1, x2, y2, rectBottomX1, rectBottomY1, rectBottomX2, rectBottomY2) ||
            line_segment_intersects(x1, y1, x2, y2, rectLeftX1, rectLeftY1, rectLeftX2, rectLeftY2) ||
            line_segment_intersects(x1, y1, x2, y2, rectRightX1, rectRightY1, rectRightX2, rectRightY2));
}


void check_bullet_collision(Robocode* env, Bullet* bullet, int agent_idx) {
    float v = 20.0f - 3.0f * bullet->firepower;
    float new_x = bullet->pose.x + v * cos_deg(bullet->pose.heading);
    float new_y = bullet->pose.y + v * sin_deg(bullet->pose.heading);
    
    for (int j = 0; j < env->num_agents; j++) {
        if (j == agent_idx) continue;
        
        Robot* target = &env->robots[j];
        
        // Check if bullet path intersects with robot's rectangular hitbox
        // Using a 64x64 hitbox centered on the robot
        if (check_line_intersection(bullet->pose.x, bullet->pose.y,
                                  new_x, new_y,
                                  target->pose.x - 32, target->pose.y - 32,
                                  64, 64)) {
            // printf("COLLISION DETECTED!\n");
            Robot* shooter = &env->robots[agent_idx];
            if (agent_idx == AGENT_IDX) {
                shooter->hit_enemy = true;
                target->was_hit = true;
            } else {
                target->was_hit = true;
                shooter->hit_enemy = true;
            }

            float damage = 4 * bullet->firepower;
            if (bullet->firepower > 1.0f) {
                damage += 2 * (bullet->firepower - 1.0f);
            }

            target->energy -= damage;
            shooter->energy += 3 * bullet->firepower;
            bullet->live = false;
            return;
        }
    }
    
    bullet->pose.x = new_x;
    bullet->pose.y = new_y;
}

void update_bullets(Robocode* env) {
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        Robot* robot = &env->robots[agent_idx];

        if (robot->energy <= 0) {
            // c_reset(env);
            env->terminals[agent_idx] = 1;
            return;
        }

        for (int blt = 0; blt < NUM_BULLETS; blt++) {
            Bullet* bullet = &env->bullets[agent_idx*NUM_BULLETS + blt];
            if (!bullet->live) {
                continue;
            }
            check_bullet_collision(env, bullet, agent_idx);
          
            // Bounds check
            if (bullet->pose.x < 0 || bullet->pose.x > env->width
                    || bullet->pose.y < 0 || bullet->pose.y > env->height) {
                bullet->live = false;
                continue;
            }
        }
    }
}

int is_angle_between(float start, float end, float angle){
    if (start < end){
        return (angle >= start && angle <= end);
    } else {
        return (angle >= start || angle <= end);
    }
}

bool is_in_radar_sweep( Robot* enemy, Robot* robot, 
                    float radar_start, float radar_end){

    float dx = enemy->pose.x - robot->pose.x;
    float dy = enemy->pose.y - robot->pose.y;

    float dist = sqrt(dx*dx + dy*dy);
    if (dist > MAX_RADAR_DISTANCE) return false;

    float angle_to_enemy = fmod( atan2(dy, dx) * RAD2DEG + 360.0f, 360.0f);

    return is_angle_between(radar_start, radar_end, angle_to_enemy);


}
void radar_detection_step(Robocode* env){

    for (int i = 0; i < env->num_agents; i++){

        Robot* robot = &env->robots[i];
        robot->enemy_detected = false;

        float radar_start = robot->radar_heading_prev;
        float radar_end = robot->radar_heading;

        // check for enemy in radar
        for(int j = 0; j < env->num_agents; j++){
            if(i==j) continue; // skip self
            Robot* enemy = &env->robots[j];
            if (is_in_radar_sweep(enemy ,robot, radar_start, radar_end)){
                robot->enemy_detected = true;

                float dx = enemy->pose.x - robot->pose.x;
                float dy = enemy->pose.y - robot->pose.y;
                robot->enemy_distance = sqrt(dx*dx + dy*dy);

                float angle_to_enemy = fmod( atan2(dy, dx) * RAD2DEG + 360.0f, 360.0f);
                robot->enemy_bearing = fmod(angle_to_enemy - robot->pose.heading + 360.0f, 360.0f);
                break;
            }
        }
    }
}
// RADAR STUFF ENDS HERE
void compute_observations(Robocode* env){
    int obs_size = 15;
    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        int obs_offset = i * obs_size;
        env->observations[obs_offset + 0] = robot->pose.x / env->width;;
        env->observations[obs_offset + 1] = robot->pose.y /  env->height;;
        env->observations[obs_offset + 2] = robot->pose.heading /360.0f; // normalize heading to 0-1
        env->observations[obs_offset + 3] = robot->vel / MAX_VELOCITY;

        env->observations[obs_offset + 4] = robot->gun_heading / 360.0f; // normalize gun heading to 0-1
        env->observations[obs_offset + 5] = robot->radar_heading_prev / 360.0f; // normalize radar heading to 0-1
        env->observations[obs_offset + 6] = robot->radar_heading / 360.0f; // normalize radar heading to 0-1
        env->observations[obs_offset + 7] = robot->gun_heat / 10.0f; // normalize gun heat to 0-1 (max gun heat is 10)
        env->observations[obs_offset + 8] = robot->energy / 100.0f; // normalize energy to 0-1
        env->observations[obs_offset + 9] = robot->bullet_idx;
        env->observations[obs_offset + 10] = env->bullets[robot->bullet_idx].pose.x / env->width;
        env->observations[obs_offset + 11] = env->bullets[robot->bullet_idx].pose.y/ env->height;
        env->observations[obs_offset + 12] = env->bullets[robot->bullet_idx].firepower / 3.0f; // normalize firepower to 0-1 (max firepower is 3)
        env->observations[obs_offset + 13] = env->bullets[robot->bullet_idx].live ? 1.0f : 0.0f;
        env->observations[obs_offset + 14] = robot->enemy_detected? 1.0f : 0.0f;
    }
}

float compute_reward(Robocode* env){
    /* 
    Reward function:
        - Staying Alive
        - Hitting the Enemy
        - Getting hit by the enemy
        - Relative Energy to the enemy
        - Dodging the enemy bullets
        - Hitting a wall
        - Aiming at the enemy
    */
    Robot* advserial = &env->robots[ADVERSARIAL_IDX];
    Robot* agent = &env->robots[AGENT_IDX];
    float reward = 0.0f;
    reward += 0.1f;

    if (agent->hit_enemy){
        reward += HIT_ENEMY_REWARD;
        env->log.score += HIT_ENEMY_REWARD;
        env->log.perf += 1.0f;
    }

    if(agent->was_hit){
        reward -= GOT_HIT_PENALTY;
    }

    reward += RELATIVE_ENERGY_SCALE*(agent->energy - advserial->energy);
    // Reset flags after computing reward
    agent->hit_enemy = false;
    agent->was_hit = false;
    return reward;
}

void c_reset(Robocode* env) {
    env->tick = 0;
    int idx = 0;
    float x, y;
    while (idx < env->num_agents) {
        Robot* robot = &env->robots[idx];
        robot->id = idx;
        x = 16 + rand() % (env->width-32);
        y = 16 + rand() % (env->height-32);

        bool collided = false;
        for (int j = 0; j < idx; j++) {
            Robot* other = &env->robots[j];
            float dx = x - other->pose.x;
            float dy = y - other->pose.y;
            float dist = sqrt(dx*dx + dy*dy);
            if (dist < 32.0f) {
                collided = true;
                break;
            }
        }
        // If Robots are not initialised in the same place then initlaise
        if (!collided) {
            robot->pose.x = x;
            robot->pose.y = y;
            robot->pose.heading = 0;
            robot->vel = 0;
            robot->energy = 100;
            robot->gun_heat = 3;
            robot->gun_heading = 0;
            robot->radar_heading = 0;
            robot->radar_heading_prev = 0;
            robot->bullet_idx = 0;
            robot->enemy_detected = false;
            robot->was_hit = false;
            robot->hit_enemy = false;
            idx += 1;
        }
    }

    // Reset Bullets
    for (int i = 0; i < NUM_BULLETS * env->num_agents; i++) {
        env->bullets[i].live = false;
    }
      // Reset rewards and terminals
    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0;
    }
    
    compute_observations(env);
}


void c_step(Robocode* env) {    
    env->tick++;
    env->log.episode_length += 1;
    //Reset rewards and terminals
    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0;
    }

    update_bullets(env);
    radar_detection_step(env);
    adversarial_agent_step(env);


    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        Robot* robot = &env->robots[agent_idx];
        int atn_offset = agent_idx * NUM_ACTIONS;

        // Cool down gun
        if (robot->gun_heat > 0) {
            robot->gun_heat -= 0.1f;
        }
      
        // Move
        float move_atn = env->actions[atn_offset + MOVE_IDX];
        move(env, robot, move_atn);

        //turn
        float turn_atn = env->actions[atn_offset + TURN_IDX];
        float turn_degrees = turn(env, robot, turn_atn);

        float gun_degrees = env->actions[atn_offset + GUN_TURN_IDX] + turn_degrees;
        robot->gun_heading += gun_degrees;
        
        if (robot->gun_heading > 360) {
            robot->gun_heading -= 360;
        } else if (robot->gun_heading < 0) {
            robot->gun_heading += 360;
        }

        // Radar
        float radar_degrees = env->actions[atn_offset + RADAR_TURN_IDX] + gun_degrees;
        robot->radar_heading_prev = robot->radar_heading;
        robot->radar_heading += radar_degrees;

        if (robot->radar_heading > 360) {
            robot->radar_heading -= 360;
        } else if (robot->radar_heading < 0) {
            robot->radar_heading += 360;
        }

        // Fire
        float firepower = env->actions[atn_offset + FIRE_IDX];
        if (firepower > 0) {
            fire(env, robot, firepower);
        }

       
        robot->pose.x = fmax(16, fmin(env->width - 16, robot->pose.x));
        robot->pose.y = fmax(16, fmin(env->height - 16, robot->pose.y));
    }
    

    env->rewards[AGENT_IDX] = compute_reward(env);
}


void c_render(Robocode* env) {
    if(env->client == NULL) {
        InitWindow(768, 576, "PufferLib Ray Robocode");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->atlas = LoadTexture("resources/robocode/robocode.png");
    }    
    
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int x = 0; x < env->width; x+=64) {
        for (int y = 0; y < env->height; y+=64) {
            int src_x = 64 * ((x*33409+ y*30971) % 5);
            Rectangle src_rect = (Rectangle){src_x, 0, 64, 64};
            Vector2 dest_pos = (Vector2){x, y};
            DrawTextureRec(env->client->atlas, src_rect, dest_pos, WHITE);
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        int atn_offset = i * NUM_ACTIONS;
        int turn_atn = env->actions[atn_offset + 1];
        int gun_atn = env->actions[atn_offset + 2] + turn_atn;
        int radar_atn = env->actions[atn_offset + 3] + gun_atn;

        Robot robot = env->robots[i];
        Vector2 robot_pos = (Vector2){robot.pose.x, robot.pose.y};

        // Radar
        float radar_left = (radar_atn > 0) ? robot.radar_heading: robot.radar_heading_prev;
        float radar_right = (radar_atn > 0) ? robot.radar_heading_prev : robot.radar_heading;
        Vector2 radar_left_pos = (Vector2){
            robot.pose.x + MAX_RADAR_DISTANCE * cos_deg(radar_left),
            robot.pose.y + MAX_RADAR_DISTANCE * sin_deg(radar_left)
        };
        Vector2 radar_right_pos = (Vector2){
            robot.pose.x + MAX_RADAR_DISTANCE * cos_deg(radar_right),
            robot.pose.y + MAX_RADAR_DISTANCE * sin_deg(radar_right)
        };
        DrawTriangle(robot_pos, radar_left_pos, radar_right_pos, (Color){0, 255, 0, 128});

        // Gun 
        Vector2 gun_pos = (Vector2){
            robot.pose.x + 64*cos_deg(robot.gun_heading),
            robot.pose.y + 64*sin_deg(robot.gun_heading)
        };
        //DrawLineEx(robot_pos, gun_pos, 4, WHITE);

        // Robot
        //DrawCircle(robot.x, robot.y, 32, RED);
        //DrawCircle(robot.x, robot.y, 16, WHITE);
        float theta = robot.pose.heading;
        float dx = cos_deg(theta);
        float dy = sin_deg(theta);
        int src_y = 64 + 64*(i%2);
        Rectangle body_rect = (Rectangle){0, src_y, 64, 64};
        Rectangle radar_rect = (Rectangle){64, src_y, 64, 64};
        Rectangle gun_rect = (Rectangle){128, src_y, 64, 64};
        Rectangle dest_rect = (Rectangle){robot.pose.x, robot.pose.y, 64, 64};
        Vector2 origin = (Vector2){32, 32};
        DrawTexturePro(env->client->atlas, body_rect, dest_rect, origin, robot.pose.heading+90, WHITE);
        DrawTexturePro(env->client->atlas, radar_rect, dest_rect, origin, robot.radar_heading+90, WHITE);
        DrawTexturePro(env->client->atlas, gun_rect, dest_rect, origin, robot.gun_heading+90, WHITE);

        DrawText(TextFormat("%i", robot.energy), robot.pose.x-16, robot.pose.y-48, 12, WHITE);
    }

    for (int i = 0; i < env->num_agents*NUM_BULLETS; i++) {
        Bullet bullet = env->bullets[i];
        if (!bullet.live) {
            continue;
        }
        Vector2 bullet_pos = (Vector2){bullet.pose.x, bullet.pose.y};
        DrawCircleV(bullet_pos, 4, WHITE);
    }

    EndDrawing();
}

void c_close(Robocode* env) {
    free(env->robots);
    free(env->bullets);

    if (env->client != NULL) {
        Client* client = env->client;
        UnloadTexture(client->atlas);
        CloseWindow();
        free(client);
    }
}