#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "raylib.h"
#include "box2d/box2d.h"

// This shows how to use Box2D v3 with raylib.
// It also show how to use Box2D with pixel units.

typedef struct Entity
{
	b2BodyId bodyId;
	b2Vec2 extent;
	Texture texture;
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
const float BARGE_HEIGHT = 2;
const float BARGE_WIDTH = 10;

const float LANDER_WIDTH = 1;
const float LANDER_HEIGHT = 5;

const float PX_PER_METER = 5.0f;

void DrawEntity(const Entity* entity)
{
	// The boxes were created centered on the bodies, but raylib draws textures starting at the top left corner.
	// b2Body_GetWorldPoint gets the top left corner of the box accounting for rotation.
	b2Vec2 p = b2Body_GetWorldPoint(entity->bodyId, (b2Vec2){-entity->extent.x, -entity->extent.y});
    float width = entity->extent.x;
    float height = entity->extent.y;
	//b2Rot rotation = b2Body_GetRotation(entity->bodyId);
	//float radians = b2Rot_GetAngle(rotation);

    DrawRectangle(
        PX_PER_METER*p.x,
        -2*PX_PER_METER*p.y,
        2*PX_PER_METER*width,
        2*PX_PER_METER*height,
        RED
    );
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

};

int main(void)
{
	b2SetLengthUnitsPerMeter(PX_PER_METER);

	// 128 pixels per meter is a appropriate for this scene. The boxes are 128 pixels wide.
	b2WorldDef worldDef = b2DefaultWorldDef();

	// Realistic gravity is achieved by multiplying gravity by the length unit.
	worldDef.gravity.y = -9.8f * PX_PER_METER;
	b2WorldId world_id = b2CreateWorld(&worldDef);

    b2BodyDef barge_body = b2DefaultBodyDef();
    barge_body.position = (b2Vec2){0, 0};
    barge_body.type = b2_staticBody;
    b2BodyId barge_id = b2CreateBody(world_id, &barge_body);

    b2Vec2 barge_extent = (b2Vec2){BARGE_WIDTH/2, BARGE_HEIGHT/2};
    b2Polygon barge_box = b2MakeBox(barge_extent.x, barge_extent.y);
    b2ShapeDef barge_shape = b2DefaultShapeDef();
    b2CreatePolygonShape(barge_id, &barge_shape, &barge_box);
    Entity barge = {
        .extent = barge_extent,
        .bodyId = barge_id,
    };

    b2BodyDef lander_body = b2DefaultBodyDef();
    lander_body.position = (b2Vec2){0, 100};
    lander_body.type = b2_dynamicBody;
    b2BodyId lander_id = b2CreateBody(world_id, &lander_body);
    b2Polygon lander_box = b2MakeBox(LANDER_WIDTH / 2, LANDER_HEIGHT / 2);
    b2ShapeDef lander_shape = b2DefaultShapeDef();
    b2CreatePolygonShape(lander_id, &lander_shape, &lander_box);
    Entity lander = {
        .extent = (b2Vec2){ LANDER_WIDTH, LANDER_HEIGHT },
        .bodyId = lander_id,
    };

	int width = 1920, height = 1080;
	InitWindow(width, height, "box2d-raylib");
	SetTargetFPS(60);

    Camera2D camera = {
        .target = (Vector2){0, 0},
        .offset = (Vector2){width/2, 9*height/10},
        .rotation = 0.0f,
        .zoom = 1.0f,
    };

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ESCAPE)) {
            exit(0);
        }

        float deltaTime = GetFrameTime();
        b2World_Step(world_id, deltaTime, 4);

	    b2Transform transform = b2Body_GetTransform(lander_id);
        printf("y: %f\n", transform.p.y);
        if (transform.p.y < 6) {
            b2Body_SetTransform(lander_id, (b2Vec2){0, 100}, transform.q);
            b2Body_SetLinearVelocity(lander_id, (b2Vec2){0, 0});
            b2Body_SetAngularVelocity(lander_id, 0.0f); 
        }

        BeginDrawing();
        ClearBackground(DARKGRAY);
        //DrawRectangle(0, 0, 100, 100, RED);
        BeginMode2D(camera);
        DrawEntity(&barge);
        DrawEntity(&lander);
        EndMode2D();
        EndDrawing();
    }
    return 0;
}
