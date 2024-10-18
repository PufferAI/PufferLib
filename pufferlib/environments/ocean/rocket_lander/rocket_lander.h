#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "raylib.h"
#include "box2d/box2d.h"

// This shows how to use Box2D v3 with raylib.
// It also show how to use Box2D with pixel units.
//
#define LOG_BUFFER_SIZE 1024

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
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
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
    //printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.score += logs->logs[i].score;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    logs->idx = 0;
    return log;
}
 
typedef struct Entity
{
	b2BodyId bodyId;
	b2Vec2 extent;
} Entity;

#define GROUND_COUNT 14
#define BOX_COUNT 10

const float SCALE = 30;
const float VIEWPORT_W = 1000;
const float VIEWPORT_H = 800;
const float GRAVITY = 9.8f;
const float W = VIEWPORT_W / SCALE;
const float H = VIEWPORT_H / SCALE;

const float BARGE_FRICTION = 2;
const float BARGE_HEIGHT = 10;
const float BARGE_WIDTH = 100;

const float LANDER_WIDTH = 20;
const float LANDER_HEIGHT = 227;

const float LEG_AWAY = 20;
const float LEG_DOWN = 0.3;
const float LEG_W = 30;
const float LEG_H = 10*LANDER_HEIGHT / 8;
const float LEG_SPRING_TORQUE = LANDER_HEIGHT / 2;

const float INITIAL_Y = 500;


const float PX_PER_METER = 1.0f;
const float DEGTORAD = PI / 180.0f;

const float THRUST_SCALE = 1000000;
const float SIDE_THRUST_SCALE = 1000000;

void DrawEntity(const Entity* entity, Color color)
{
	// The boxes were created centered on the bodies, but raylib draws textures starting at the top left corner.
	// b2Body_GetWorldPoint gets the top left corner of the box accounting for rotation.
	b2Vec2 p = b2Body_GetWorldPoint(entity->bodyId, (b2Vec2){0, 0});
    float width = 2*entity->extent.x;
    float height = 2*entity->extent.y;

    b2Rot rotation = b2Body_GetRotation(entity->bodyId);
    float radians = b2Rot_GetAngle(rotation);
    float degrees = radians / DEGTORAD;

	//b2Rot rotation = b2Body_GetRotation(entity->bodyId);
	//float radians = b2Rot_GetAngle(rotation);
    //printf("\t: x: %f, y: %f, w: %f, h: %f, deg: %f\n", p.x, p.y, width, height, degrees);

    Rectangle rec = (Rectangle){
        PX_PER_METER*p.x,
        -PX_PER_METER*p.y,
        PX_PER_METER*width,
        PX_PER_METER*height,
    };
    DrawRectanglePro(rec, (Vector2){rec.width/2, rec.height/2}, -degrees, color);
	//DrawTextureEx(entity->texture, ps, RAD2DEG * radians, 1.0f, WHITE);

	// I used these circles to ensure the coordinates are correct
	//DrawCircleV(ps, 5.0f, BLACK);
	//p = b2Body_GetWorldPoint(entity->bodyId, (b2Vec2){0.0f, 0.0f});
	//ps = (Vector2){ p.x, p.y };
	//DrawCircleV(ps, 5.0f, BLUE);
	//p = b2Body_GetWorldPoint(entity->bodyId, (b2Vec2){ entity->extent.x, entity->extent.y });
	//ps = (Vector2){ p.x, p.y };
	//DrawCircleV(ps, 5.0f, RED);
}


typedef struct Lander Lander;
struct Lander {
    float* observations;
    float* actions;
    float* reward;
    unsigned char* terminal;
    unsigned char* truncation;
    LogBuffer* log_buffer;
    Log log;
    int tick;
    b2WorldId world_id;
    b2BodyId barge_id;
    b2BodyId lander_id;
    Entity barge;
    Entity lander;
};

void init_lander(Lander* env) {
	b2SetLengthUnitsPerMeter(PX_PER_METER);

	// 128 pixels per meter is a appropriate for this scene. The boxes are 128 pixels wide.
	b2WorldDef worldDef = b2DefaultWorldDef();

	// Realistic gravity is achieved by multiplying gravity by the length unit.
	worldDef.gravity.y = -9.8f * PX_PER_METER;
	b2WorldId world_id = b2CreateWorld(&worldDef);
    env->world_id = world_id;

    b2BodyDef barge_body = b2DefaultBodyDef();
    barge_body.position = (b2Vec2){0, 0};
    barge_body.type = b2_staticBody;
    b2BodyId barge_id = b2CreateBody(world_id, &barge_body);
    env->barge_id = barge_id;

    b2Vec2 barge_extent = (b2Vec2){BARGE_WIDTH/2, BARGE_HEIGHT/2};
    b2Polygon barge_box = b2MakeBox(barge_extent.x, barge_extent.y);
    b2ShapeDef barge_shape = b2DefaultShapeDef();
    b2CreatePolygonShape(barge_id, &barge_shape, &barge_box);
    Entity barge = {
        .extent = barge_extent,
        .bodyId = barge_id,
    };
    env->barge = barge;

    b2BodyDef lander_body = b2DefaultBodyDef();
    lander_body.position = (b2Vec2){0, INITIAL_Y};
    lander_body.type = b2_dynamicBody;
    b2BodyId lander_id = b2CreateBody(world_id, &lander_body);
    env->lander_id = lander_id;

    b2Vec2 lander_extent = (b2Vec2){LANDER_WIDTH / 2, LANDER_HEIGHT / 2};
    b2Polygon lander_box = b2MakeBox(lander_extent.x, lander_extent.y);
    b2ShapeDef lander_shape = b2DefaultShapeDef();
    b2CreatePolygonShape(lander_id, &lander_shape, &lander_box);
    Entity lander = {
        .extent = lander_extent,
        .bodyId = lander_id,
    };
    env->lander = lander;
}

void allocate_lander(Lander* env) {
    env->observations = (float*)calloc(6, sizeof(float));
    env->actions = (float*)calloc(3, sizeof(float));
    env->reward = (float*)calloc(1, sizeof(float));
    env->terminal = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->truncation = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
    init_lander(env);
}

void compute_observations_and_reward(Lander* env, float prev_x, float prev_y) {
    b2Transform transform = b2Body_GetTransform(env->lander_id);
    b2Vec2 pos = transform.p;
    float rot = b2Rot_GetAngle(transform.q);
    b2Vec2 vel = b2Body_GetLinearVelocity(env->lander_id);
    float ang_vel = b2Body_GetAngularVelocity(env->lander_id);

    float reward_x = (fabsf(prev_x) - fabsf(pos.x))/ 1000;
    float reward_y = (fabsf(prev_y) - fabsf(pos.y))/ 1000;
    //float reward_rot = -fabsf(rot)/ PI / 10;
    //printf("Reward: %f, %f, %f\n", reward_x, reward_y, reward_rot);
    float reward = reward_x + reward_y;// + reward_rot;
    
    reward = 0;
    if (env->actions[0] == 0) {
        reward = 1;
    }

    env->reward[0] = reward;
    env->log.episode_return += reward;

    env->observations[0] = pos.x / 500;
    env->observations[1] = pos.y / 1200;
    env->observations[2] = vel.x / 200;
    env->observations[3] = vel.y / 200;
    env->observations[4] = rot / PI / 10;
    env->observations[5] = ang_vel / PI;
}

void reset(Lander* env) {
    env->log = (Log){0};
    env->tick = 0;

    b2Body_SetTransform(
        env->lander_id,
        (b2Vec2){0, INITIAL_Y},
        b2MakeRot(0)
    );
    b2Body_SetLinearVelocity(env->lander_id, (b2Vec2){0, 0});
    b2Body_SetAngularVelocity(env->lander_id, 0.0f); 

    b2Transform transform = b2Body_GetTransform(env->lander_id);
    b2Vec2 pos = transform.p;
    env->reward[0] = 0;
    compute_observations_and_reward(env, pos.x, pos.y);
    env->reward[0] = 0;
}

void step(Lander* env) {
    env->reward[0] = 0;
    b2Transform transform = b2Body_GetTransform(env->lander_id);
    b2Vec2 pos = transform.p;
    printf("Pos x: %f, y: %f\n", pos.x, pos.y);

    b2Vec2 p_thrust = b2Body_GetWorldPoint(env->lander_id,
        (b2Vec2){0, -LANDER_HEIGHT/2});
    b2Vec2 p_left = b2Body_GetWorldPoint(env->lander_id,
        (b2Vec2){-LANDER_WIDTH/2, LANDER_HEIGHT/2});
    b2Vec2 p_right= b2Body_GetWorldPoint(env->lander_id,
        (b2Vec2){LANDER_WIDTH/2, LANDER_HEIGHT/2});

    b2Vec2 force = (b2Vec2){0, 0};
    b2Rot rotation = b2Body_GetRotation(env->lander_id);
    float radians = b2Rot_GetAngle(rotation);


    // Main thruster
    float atn_thrust = THRUST_SCALE * env->actions[0];
    float rad_thrust = radians + 0.02*(float)rand()/RAND_MAX;
    force = (b2Vec2){
        atn_thrust*sin(rad_thrust),
        atn_thrust*cos(rad_thrust)
    };
    b2Body_ApplyForce(env->lander_id, force, p_thrust, true);

    // Top left thruster
    float atn_left = SIDE_THRUST_SCALE * env->actions[1];
    float rad_left = -radians + PI/2 + 0.02*(float)rand()/RAND_MAX;
    force = (b2Vec2){
        atn_left*sin(rad_left),
        atn_left*cos(rad_left)
    };
    b2Body_ApplyForce(env->lander_id, force, p_left, true);

    // Top right thruster
    float atn_right = SIDE_THRUST_SCALE * env->actions[2];
    float rad_right = -radians - PI/2 + 0.02*(float)rand()/RAND_MAX;
    force = (b2Vec2){
        atn_right*sin(rad_right),
        atn_right*cos(rad_right)
    };
    b2Body_ApplyForce(env->lander_id, force, p_right, true);

    b2World_Step(env->world_id, 60.0f, 4);


    transform = b2Body_GetTransform(env->lander_id);
    bool do_reset = false;
    if (transform.p.x < -500 || transform.p.x > 500 || transform.p.y > 1200) {
        do_reset = true;
        //env->reward[0] -= 1.0;
        //env->log.episode_return -= 1.0;
    }
    if (transform.p.y < 120) {
        do_reset = true;
        printf("Transform y: %f\n", transform.p.y);
    }
    if (env->tick > 1000) {
        do_reset = true;
        printf("Tick: %i\n", env->tick);
    }
    if (do_reset) {
        printf("Reset\n");
        env->log.episode_length = env->tick;
        add_log(env->log_buffer, &env->log);
        reset(env);
    }
    env->tick += 1;

    compute_observations_and_reward(env, pos.x, pos.y);
    //printf("Reward: %f\n", env->reward[0]);
    //env->reward[0] = -(atn_thrust + atn_left + atn_right) / 10000000;
}

void free_lander(Lander* env) {
    free_logbuffer(env->log_buffer);
    b2DestroyWorld(env->world_id);
}

typedef struct Client Client;
struct Client {
    Camera2D camera;
};

Client* make_client(Lander* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

	int width = 1920, height = 1080;
	InitWindow(width, height, "box2d-raylib");
	SetTargetFPS(60);

    client->camera = (Camera2D){
        .target = (Vector2){0, 0},
        .offset = (Vector2){width/2, 9*height/10},
        .rotation = 0.0f,
        .zoom = 1.0f,
    };
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void render(Client* client, Lander* env) {
    if (IsKeyPressed(KEY_ESCAPE)) {
        exit(0);
    }
    BeginDrawing();
    ClearBackground(DARKGRAY);
    BeginMode2D(client->camera);

    env->actions[0] = 0;
    env->actions[1] = 0;
    env->actions[2] = 0;
    if (IsKeyDown(KEY_W)) {
        env->actions[0] = 1;
    }
    if (IsKeyDown(KEY_Q)) {
        env->actions[1] = 1;
    }
    if (IsKeyDown(KEY_E)) {
        env->actions[2] = 1;
    }


    /*
    b2Rot rotation = b2Body_GetRotation(lander_id);
    float radians = b2Rot_GetAngle(rotation);
    float mag = 1000000;

    if (IsKeyDown(KEY_W)) {
        float rad_thrust = radians + 0.02*(float)rand()/RAND_MAX;
        b2Vec2 force = (b2Vec2){mag*sin(rad_thrust), mag*cos(rad_thrust)};
        b2Body_ApplyForce(lander_id, force, p_thrust, true);
        DrawCircle(p_thrust.x, -p_thrust.y, 20, RED);
    }
    if (IsKeyDown(KEY_Q)) {
        float rad_left = -radians + PI/2 + 0.02*(float)rand()/RAND_MAX;
        if (rad_left > PI) {
            rad_left -= 2*PI;
        }
        b2Vec2 force = (b2Vec2){mag*sin(rad_left), mag*cos(rad_left)};
        b2Body_ApplyForce(lander_id, force, p_left, true);
        DrawCircle(p_left.x, -p_left.y, 20, RED);
    }
    if (IsKeyDown(KEY_E)) {
        float rad_right = -radians - PI/2 + 0.02*(float)rand()/RAND_MAX;
        b2Vec2 force = (b2Vec2){mag*sin(rad_right), mag*cos(rad_right)};
        b2Body_ApplyForce(lander_id, force, p_right, true);
        DrawCircle(p_right.x, -p_right.y, 20, RED);
    }


    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 mousePos = GetScreenToWorld2D(GetMousePosition(), camera);
        float x = mousePos.x;
        float y = -mousePos.y;
        b2Vec2 origin = (b2Vec2){x, y};
        b2Vec2 p = b2Body_GetWorldPoint(lander_id,
            (b2Vec2){0, -LANDER_HEIGHT/2});
        float mag = 1000;
        b2Vec2 force = (b2Vec2){
            mag * (p.x - origin.x),
            mag * (p.y - origin.y),
        };
        b2Body_ApplyForce(lander_id, force, p, true);
        DrawLine(mousePos.x, mousePos.y, p.x, -p.y, RED);
    }

    b2Transform transform = b2Body_GetTransform(lander_id);
    printf("y: %f\n", transform.p.y);
    if (transform.p.y < 120) {
        reset(&env);
    }
    */

    //DrawRectangle(0, 0, 100, 100, RED);
    DrawEntity(&env->barge, RED);
    DrawEntity(&env->lander, BLUE);
    //DrawEntity(&legs[0], GREEN);
    //DrawEntity(&legs[1], GREEN);
    EndMode2D();
    EndDrawing();
}

void demo() {
    Lander env = {0};
    allocate_lander(&env);
    Client* client = make_client(&env);

    while (!WindowShouldClose()) {
        step(&env);
        render(client, &env);
    }
}

void test_render() {
    InitWindow(1920, 1080, "box2d-raylib");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(DARKGRAY);

        Rectangle rec = (Rectangle){500, 500, 200, 200};
        Vector2 origin = (Vector2){rec.width/2, rec.height/2};
        DrawRectanglePro(rec, origin, 45, RED);

        DrawCircle(500, 500, 30, BLUE);

        EndDrawing();
    }

}


