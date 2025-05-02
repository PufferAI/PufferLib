#ifndef IMPULSE_WARS_GAME_H
#define IMPULSE_WARS_GAME_H

#include "helpers.h"
#include "settings.h"
#include "types.h"

// these functions call each other so need to be forward declared
void destroyProjectile(env *e, projectileEntity *projectile, const bool processExplosions, const bool full);
void createExplosion(env *e, droneEntity *drone, const projectileEntity *projectile, const b2ExplosionDef *def);

void updateTrailPoints(trailPoints *tp, const uint8_t maxLen, const b2Vec2 pos);

entity *createEntity(env *e, enum entityType type, void *entityData) {
    int32_t id = b2AllocId(&e->idPool);
    entity *ent = NULL;
    if (id == (int64_t)cc_array_size(e->entities)) {
        ent = fastCalloc(1, sizeof(entity));
        ent->id = fastCalloc(1, sizeof(entityID));
        cc_array_add(e->entities, ent);
    } else {
        ent = safe_array_get_at(e->entities, id);
    }

    ent->generation += 1;
    ent->type = type;
    ent->entity = entityData;
    ent->id->id = id + 1,
    ent->id->generation = ent->generation;

    return ent;
}

void destroyEntity(env *e, entity *ent) {
    b2FreeId(&e->idPool, ent->id->id - 1);
    ent->id->id = 0;
}

// will return a pointer to an entity or NULL if the given entity ID is
// invalid or orphaned
entity *getEntityByID(const env *e, const entityID *id) {
    if (id->id < 1 || (int64_t)cc_array_size(e->entities) < id->id) {
        // invalid index
        return NULL;
    }
    entity *ent = safe_array_get_at(e->entities, id->id - 1);
    if (ent->id->id == 0 || ent->generation != id->generation) {
        // orphaned entity
        return NULL;
    }
    return ent;
}

static inline bool entityTypeIsWall(const enum entityType type) {
    // walls are the first 3 entity types
    return type <= DEATH_WALL_ENTITY;
}

static inline int16_t cellIndex(const env *e, const int8_t col, const int8_t row) {
    return col + (row * e->map->columns);
}

// discretizes an entity's position into a cell index; -1 is returned if
// the position is out of bounds of the map
static inline int16_t entityPosToCellIdx(const env *e, const b2Vec2 pos) {
    const float cellX = pos.x + (((float)e->map->columns * WALL_THICKNESS) / 2.0f);
    const float cellY = pos.y + (((float)e->map->rows * WALL_THICKNESS) / 2.0f);
    const int8_t cellCol = cellX / WALL_THICKNESS;
    const int8_t cellRow = cellY / WALL_THICKNESS;
    const int16_t cellIdx = cellIndex(e, cellCol, cellRow);
    // set the cell to -1 if it's out of bounds
    if (cellIdx < 0 || (uint16_t)cellIdx >= cc_array_size(e->cells)) {
        DEBUG_LOGF("invalid cell index: %d from position: (%f, %f)", cellIdx, pos.x, pos.y);
        return -1;
    }
    return cellIdx;
}

typedef struct overlapAABBCtx {
    bool overlaps;
} overlapAABBCtx;

bool overlapAABBCallback(b2ShapeId shapeID, void *context) {
    if (!b2Shape_IsValid(shapeID)) {
        return true;
    }

    overlapAABBCtx *ctx = context;
    ctx->overlaps = true;
    return true;
}

// returns true if the given position overlaps with shapes in a bounding
// box with a height and width of distance
bool isOverlappingAABB(const env *e, const b2Vec2 pos, const float distance, const b2QueryFilter filter) {
    b2AABB bounds = {
        .lowerBound = {.x = pos.x - distance, .y = pos.y - distance},
        .upperBound = {.x = pos.x + distance, .y = pos.y + distance},
    };
    overlapAABBCtx ctx = {.overlaps = false};
    b2World_OverlapAABB(e->worldID, bounds, filter, overlapAABBCallback, &ctx);
    return ctx.overlaps;
}

// TODO: store a shape proxy in entities?
b2ShapeProxy makeDistanceProxyFromType(const enum entityType type, bool *isCircle) {
    b2ShapeProxy proxy = {0};
    switch (type) {
    case DRONE_ENTITY:
        *isCircle = true;
        proxy.count = 1;
        proxy.radius = DRONE_RADIUS;
        break;
    case SHIELD_ENTITY:
        *isCircle = true;
        proxy.count = 1;
        proxy.radius = DRONE_SHIELD_RADIUS;
        break;
    case WEAPON_PICKUP_ENTITY:
        proxy.count = 4;
        proxy.points[0] = (b2Vec2){.x = -PICKUP_THICKNESS / 2.0f, .y = -PICKUP_THICKNESS / 2.0f};
        proxy.points[1] = (b2Vec2){.x = -PICKUP_THICKNESS / 2.0f, .y = +PICKUP_THICKNESS / 2.0f};
        proxy.points[2] = (b2Vec2){.x = +PICKUP_THICKNESS / 2.0f, .y = -PICKUP_THICKNESS / 2.0f};
        proxy.points[3] = (b2Vec2){.x = +PICKUP_THICKNESS / 2.0f, .y = +PICKUP_THICKNESS / 2.0f};
        break;
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY:
        proxy.count = 4;
        proxy.points[0] = (b2Vec2){.x = -FLOATING_WALL_THICKNESS / 2.0f, .y = -FLOATING_WALL_THICKNESS / 2.0f};
        proxy.points[1] = (b2Vec2){.x = -FLOATING_WALL_THICKNESS / 2.0f, .y = +FLOATING_WALL_THICKNESS / 2.0f};
        proxy.points[2] = (b2Vec2){.x = +FLOATING_WALL_THICKNESS / 2.0f, .y = -FLOATING_WALL_THICKNESS / 2.0f};
        proxy.points[3] = (b2Vec2){.x = +FLOATING_WALL_THICKNESS / 2.0f, .y = +FLOATING_WALL_THICKNESS / 2.0f};
        break;
    default:
        ERRORF("unknown entity type for shape distance: %d", type);
    }

    return proxy;
}

b2ShapeProxy makeDistanceProxy(const entity *ent, bool *isCircle) {
    if (ent->type == PROJECTILE_ENTITY) {
        *isCircle = true;
        b2ShapeProxy proxy = {0};
        const projectileEntity *proj = ent->entity;
        proxy.count = 1;
        proxy.radius = proj->weaponInfo->radius;
        return proxy;
    }

    return makeDistanceProxyFromType(ent->type, isCircle);
}

b2Transform entityTransform(const entity *ent) {
    b2Transform transform;
    wallEntity *wall;
    projectileEntity *proj;
    droneEntity *drone;

    switch (ent->type) {
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY:
        wall = ent->entity;
        transform.p = wall->pos;
        transform.q = wall->rot;
        return transform;
    case PROJECTILE_ENTITY:
        proj = ent->entity;
        transform.p = proj->pos;
        transform.q = b2Rot_identity;
        return transform;
    case DRONE_ENTITY:
        drone = ent->entity;
        transform.p = drone->pos;
        transform.q = b2Rot_identity;
        return transform;
    default:
        ERRORF("unknown entity type: %d", ent->type);
    }
}

// returns the closest points between two entities
b2DistanceOutput closestPoint(const entity *srcEnt, const entity *dstEnt) {
    bool isCircle = false;
    b2DistanceInput input;
    input.proxyA = makeDistanceProxy(srcEnt, &isCircle);
    input.proxyB = makeDistanceProxy(dstEnt, &isCircle);
    input.transformA = entityTransform(srcEnt);
    input.transformB = entityTransform(dstEnt);
    if (input.proxyA.radius != 0.0f && input.proxyB.radius != 0.0f) {
        b2DistanceOutput output = {0};
        output.distance = b2Distance(input.transformB.p, input.transformA.p) - input.proxyA.radius - input.proxyB.radius;
        output.normal = b2Normalize(b2Sub(input.transformB.p, input.transformA.p));
        output.pointA = b2MulAdd(input.transformA.p, input.proxyA.radius, output.normal);
        output.pointB = b2MulAdd(input.transformB.p, input.proxyB.radius, output.normal);
        return output;
    }

    input.useRadii = isCircle;

    b2SimplexCache cache = {0};
    return b2ShapeDistance(&input, &cache, NULL, 0);
}

typedef struct behindWallContext {
    const entity *dstEnt;
    const enum entityType *targetType;
    bool hit;
} behindWallContext;

float posBehindWallCallback(b2ShapeId shapeID, b2Vec2 point, b2Vec2 normal, float fraction, void *context) {
    // these are unused but required by the b2CastResultFcn callback prototype
    MAYBE_UNUSED(point);
    MAYBE_UNUSED(normal);
    MAYBE_UNUSED(fraction);

    behindWallContext *ctx = context;
    const entity *ent = b2Shape_GetUserData(shapeID);
    if (ent == ctx->dstEnt || (ctx->targetType != NULL && ent->type == *ctx->targetType)) {
        return -1;
    }
    ctx->hit = true;
    return 0;
}

// returns true if there are shapes that match filter between startPos and endPos
bool posBehindWall(const env *e, const b2Vec2 srcPos, const b2Vec2 dstPos, const entity *dstEnt, const b2QueryFilter filter, const enum entityType *targetType) {
    const float rayDistance = b2Distance(srcPos, dstPos);
    // if the two points are extremely close we can safely assume the
    // entity isn't behind a wall
    if (rayDistance <= 1.0f) {
        return false;
    }

    const b2Vec2 translation = b2Sub(dstPos, srcPos);
    behindWallContext ctx = {
        .dstEnt = dstEnt,
        .targetType = targetType,
        .hit = false,
    };
    b2World_CastRay(e->worldID, srcPos, translation, filter, posBehindWallCallback, &ctx);
    return ctx.hit;
}

typedef struct overlapCircleCtx {
    const env *e;
    const entity *ent;
    const enum entityType *targetType;
    const b2QueryFilter filter;
    bool overlaps;
} overlapCircleCtx;

bool isOverlappingCircleCallback(b2ShapeId shapeID, void *context) {
    if (!b2Shape_IsValid(shapeID)) {
        return true;
    }

    overlapCircleCtx *ctx = context;
    const entity *overlappingEnt = b2Shape_GetUserData(shapeID);
    if (ctx->targetType != NULL && overlappingEnt->type != *ctx->targetType) {
        return true;
    }

    const b2DistanceOutput output = closestPoint(ctx->ent, overlappingEnt);
    const bool behind = posBehindWall(ctx->e, output.pointA, output.pointB, ctx->ent, ctx->filter, ctx->targetType);
    if (!behind) {
        ctx->overlaps = true;
    } else if (ctx->ent->type == PROJECTILE_ENTITY) {
        projectileEntity *proj = ctx->ent->entity;
        droneEntity *drone = overlappingEnt->entity;
        proj->dronesBehindWalls[proj->numDronesBehindWalls++] = drone->idx;
    }
    return behind;
}

bool isOverlappingCircleInLineOfSight(const env *e, const entity *ent, const b2Vec2 startPos, const float radius, const b2QueryFilter filter, const enum entityType *targetType) {
    const b2ShapeProxy cirProxy = b2MakeProxy(&startPos, 1, radius);
    overlapCircleCtx ctx = {
        .e = e,
        .ent = ent,
        .targetType = targetType,
        .filter = (b2QueryFilter){
            .categoryBits = filter.categoryBits,
            .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE,
        },
        .overlaps = false,
    };
    b2World_OverlapShape(e->worldID, &cirProxy, filter, isOverlappingCircleCallback, &ctx);
    return ctx.overlaps;
}

uint8_t cellOffsets[8][2] = {
    {-1, 0},  // left
    {1, 0},   // right
    {0, -1},  // up
    {0, 1},   // down
    {-1, -1}, // top-left
    {1, -1},  // top-right
    {-1, 1},  // bottom-left
    {1, 1},   // bottom-right
};

// returns true and sets emptyPos to the position of an empty cell
// that is an appropriate distance away from other entities if one exists;
// if quad is set to -1 a random valid position from anywhere on the map
// will be returned, otherwise a position within the specified quadrant
// will be returned
bool findOpenPos(env *e, const enum shapeCategory shapeType, b2Vec2 *emptyPos, int8_t quad) {
    uint8_t checkedCells[BITNSLOTS(MAX_CELLS)] = {0};
    const size_t nCells = cc_array_size(e->cells) - 1;
    uint16_t attempts = 0;
    bool skipDistanceChecks = false;

    while (true) {
        if (attempts == nCells) {
            // if we're trying to find a position for a drone and sudden
            // death walls have been placed, try again this time ignoring
            // distance checks; the drone must be spawned next
            // to a death wall in this case
            if (shapeType == DRONE_SHAPE && e->suddenDeathWallsPlaced && !skipDistanceChecks) {
                attempts = 0;
                memset(checkedCells, 0x0, BITNSLOTS(MAX_CELLS));
                skipDistanceChecks = true;
                continue;
            }
            return false;
        }

        uint16_t cellIdx;
        if (quad == -1) {
            cellIdx = randInt(&e->randState, 0, nCells);
        } else {
            const float minX = e->map->spawnQuads[quad].min.x;
            const float minY = e->map->spawnQuads[quad].min.y;
            const float maxX = e->map->spawnQuads[quad].max.x;
            const float maxY = e->map->spawnQuads[quad].max.y;

            b2Vec2 randPos = {.x = randFloat(&e->randState, minX, maxX), .y = randFloat(&e->randState, minY, maxY)};
            cellIdx = entityPosToCellIdx(e, randPos);
        }
        if (bitTest(checkedCells, cellIdx)) {
            continue;
        }
        bitSet(checkedCells, cellIdx);
        attempts++;

        const mapCell *cell = safe_array_get_at(e->cells, cellIdx);
        if (cell->ent != NULL) {
            continue;
        }
        if (skipDistanceChecks) {
            return true;
        }

        if (shapeType == WEAPON_PICKUP_SHAPE) {
            // ensure pickups don't spawn too close to other pickups
            bool tooClose = false;
            for (uint8_t i = 0; i < cc_array_size(e->pickups); i++) {
                const weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
                if (b2DistanceSquared(cell->pos, pickup->pos) < PICKUP_SPAWN_DISTANCE_SQUARED) {
                    tooClose = true;
                    break;
                }
            }
            if (tooClose) {
                continue;
            }
        } else if (shapeType == DRONE_SHAPE) {
            if (e->suddenDeathWallsPlaced) {
                // if sudden death walls have been placed, ignore the
                // spawn points as they may be covered by death walls;
                // instead just try and find a cell that doesn't neighbor
                // a death wall
                const uint8_t cellCol = cellIdx / e->map->columns;
                const uint8_t cellRow = cellIdx % e->map->columns;
                bool deathWallNeighboring = false;
                for (uint8_t i = 0; i < 8; i++) {
                    const int8_t col = cellCol + cellOffsets[i][0];
                    const int8_t row = cellRow + cellOffsets[i][1];
                    if (row < 0 || row >= e->map->rows || col < 0 || col >= e->map->columns) {
                        continue;
                    }
                    const int16_t testCellIdx = cellIndex(e, col, row);
                    const mapCell *testCell = safe_array_get_at(e->cells, testCellIdx);
                    if (testCell->ent != NULL && testCell->ent->type == DEATH_WALL_ENTITY) {
                        deathWallNeighboring = true;
                        break;
                    }
                }
                if (deathWallNeighboring) {
                    continue;
                }
            } else {
                if (!e->map->droneSpawns[cellIdx]) {
                    continue;
                }

                // ensure drones don't spawn too close to other drones
                bool tooClose = false;
                for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
                    const droneEntity *drone = safe_array_get_at(e->drones, i);
                    if (b2DistanceSquared(cell->pos, drone->pos) < DRONE_DRONE_SPAWN_DISTANCE_SQUARED) {
                        tooClose = true;
                        break;
                    }
                }
                if (tooClose) {
                    continue;
                }
            }
        }

        uint64_t maskBits = FLOATING_WALL_SHAPE | WEAPON_PICKUP_SHAPE | DRONE_SHAPE;
        if (shapeType != FLOATING_WALL_SHAPE) {
            maskBits &= ~shapeType;
        }
        const b2QueryFilter filter = {
            .categoryBits = shapeType,
            .maskBits = maskBits,
        };
        if (!isOverlappingAABB(e, cell->pos, MIN_SPAWN_DISTANCE, filter)) {
            *emptyPos = cell->pos;
            return true;
        }
    }
}

entity *createWall(env *e, const b2Vec2 pos, const float width, const float height, int16_t cellIdx, const enum entityType type, const bool floating) {
    ASSERT(cellIdx != -1);
    ASSERT(entityTypeIsWall(type));

    b2BodyDef wallBodyDef = b2DefaultBodyDef();
    wallBodyDef.position = pos;
    if (floating) {
        wallBodyDef.type = b2_dynamicBody;
        wallBodyDef.linearDamping = FLOATING_WALL_DAMPING;
        wallBodyDef.angularDamping = FLOATING_WALL_DAMPING;
        wallBodyDef.isAwake = false;
    }
    b2BodyId wallBodyID = b2CreateBody(e->worldID, &wallBodyDef);

    b2Vec2 extent = {.x = width / 2.0f, .y = height / 2.0f};
    b2ShapeDef wallShapeDef = b2DefaultShapeDef();
    wallShapeDef.invokeContactCreation = false;
    wallShapeDef.density = WALL_DENSITY;
    wallShapeDef.material.restitution = STANDARD_WALL_RESTITUTION;
    wallShapeDef.material.friction = STANDARD_WALL_FRICTION;
    wallShapeDef.filter.categoryBits = WALL_SHAPE;
    wallShapeDef.filter.maskBits = FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE | SHIELD_SHAPE | DRONE_PIECE_SHAPE;
    if (floating) {
        wallShapeDef.filter.categoryBits = FLOATING_WALL_SHAPE;
        wallShapeDef.filter.maskBits |= WALL_SHAPE | WEAPON_PICKUP_SHAPE;
        wallShapeDef.enableSensorEvents = true;
    }

    if (type == BOUNCY_WALL_ENTITY) {
        wallShapeDef.material.restitution = BOUNCY_WALL_RESTITUTION;
        wallShapeDef.material.friction = 0.0f;
    } else if (type == DEATH_WALL_ENTITY) {
        wallShapeDef.enableContactEvents = true;
    }

    wallEntity *wall = fastCalloc(1, sizeof(wallEntity));
    wall->bodyID = wallBodyID;
    wall->pos = pos;
    wall->rot = b2Rot_identity;
    wall->velocity = b2Vec2_zero;
    wall->extent = extent;
    wall->mapCellIdx = cellIdx;
    wall->isFloating = floating;
    wall->type = type;
    wall->isSuddenDeath = e->suddenDeathWallsPlaced;

    entity *ent = createEntity(e, type, wall);
    wall->ent = ent;

    wallShapeDef.userData = ent;
    const b2Polygon wallPolygon = b2MakeBox(extent.x, extent.y);
    wall->shapeID = b2CreatePolygonShape(wallBodyID, &wallShapeDef, &wallPolygon);
    b2Body_SetUserData(wall->bodyID, ent);

    if (floating) {
        cc_array_add(e->floatingWalls, wall);
    } else {
        cc_array_add(e->walls, wall);
    }

    return ent;
}

void destroyWall(env *e, wallEntity *wall, const bool full) {
    destroyEntity(e, wall->ent);

    if (full) {
        mapCell *cell = safe_array_get_at(e->cells, wall->mapCellIdx);
        cell->ent = NULL;
    }

    b2DestroyBody(wall->bodyID);
    fastFree(wall);
}

enum weaponType randWeaponPickupType(env *e) {
    // spawn weapon pickups according to their spawn weights and how many
    // pickups are currently spawned with different weapons
    float totalWeight = 0.0f;
    float spawnWeights[_NUM_WEAPONS - 1] = {0};
    for (uint8_t i = 1; i < NUM_WEAPONS; i++) {
        if (i == e->defaultWeapon->type) {
            continue;
        }
        spawnWeights[i - 1] = weaponInfos[i]->spawnWeight / ((e->spawnedWeaponPickups[i] + 1) * 2.0f);
        totalWeight += spawnWeights[i - 1];
    }

    const float randPick = randFloat(&e->randState, 0.0f, totalWeight);
    float cumulativeWeight = 0.0f;
    enum weaponType type = STANDARD_WEAPON;
    for (uint8_t i = 1; i < NUM_WEAPONS; i++) {
        if (i == e->defaultWeapon->type) {
            continue;
        }
        cumulativeWeight += spawnWeights[i - 1];
        if (randPick < cumulativeWeight) {
            type = i;
            break;
        }
    }
    ASSERT(type != STANDARD_WEAPON && type != e->defaultWeapon->type);
    e->spawnedWeaponPickups[type]++;

    return type;
}

void createWeaponPickupBodyShape(const env *e, weaponPickupEntity *pickup) {
    pickup->bodyDestroyed = false;

    b2BodyDef pickupBodyDef = b2DefaultBodyDef();
    pickupBodyDef.position = pickup->pos;
    pickupBodyDef.userData = pickup->ent;
    pickup->bodyID = b2CreateBody(e->worldID, &pickupBodyDef);

    b2ShapeDef pickupShapeDef = b2DefaultShapeDef();
    pickupShapeDef.filter.categoryBits = WEAPON_PICKUP_SHAPE;
    pickupShapeDef.filter.maskBits = FLOATING_WALL_SHAPE | DRONE_SHAPE;
    pickupShapeDef.isSensor = true;
    pickupShapeDef.enableSensorEvents = true;
    pickupShapeDef.userData = pickup->ent;
    const b2Polygon pickupPolygon = b2MakeBox(PICKUP_THICKNESS / 2.0f, PICKUP_THICKNESS / 2.0f);
    pickup->shapeID = b2CreatePolygonShape(pickup->bodyID, &pickupShapeDef, &pickupPolygon);
}

void createWeaponPickup(env *e) {
    // ensure weapon pickups are initially spawned somewhat uniformly
    b2Vec2 pos;
    e->lastSpawnQuad = (e->lastSpawnQuad + 1) % 4;
    if (!findOpenPos(e, WEAPON_PICKUP_SHAPE, &pos, e->lastSpawnQuad)) {
        ERROR("no open position for weapon pickup");
    }

    weaponPickupEntity *pickup = fastCalloc(1, sizeof(weaponPickupEntity));
    pickup->weapon = randWeaponPickupType(e);
    pickup->respawnWait = 0.0f;
    pickup->floatingWallsTouching = 0;
    pickup->pos = pos;

    entity *ent = createEntity(e, WEAPON_PICKUP_ENTITY, pickup);
    pickup->ent = ent;

    const int16_t cellIdx = entityPosToCellIdx(e, pos);
    if (cellIdx == -1) {
        ERRORF("invalid position for weapon pickup spawn: (%f, %f)", pos.x, pos.y);
    }
    pickup->mapCellIdx = cellIdx;
    mapCell *cell = safe_array_get_at(e->cells, cellIdx);
    cell->ent = ent;

    createWeaponPickupBodyShape(e, pickup);

    cc_array_add(e->pickups, pickup);
}

void destroyWeaponPickup(env *e, weaponPickupEntity *pickup) {
    destroyEntity(e, pickup->ent);

    mapCell *cell = safe_array_get_at(e->cells, pickup->mapCellIdx);
    cell->ent = NULL;

    if (!pickup->bodyDestroyed) {
        b2DestroyBody(pickup->bodyID);
    }

    fastFree(pickup);
}

// destroys the pickup body and shape while the pickup is waiting to
// respawn to avoid spurious sensor overlap checks; enabling/disabling
// a body is almost as expensive and creating a new body in box2d, and
// manually moving (teleporting) it is expensive as well so destroying
// the body now and re-creating it later is the fastest
void disableWeaponPickup(env *e, weaponPickupEntity *pickup) {
    DEBUG_LOGF("disabling weapon pickup at cell %d (%f, %f)", pickup->mapCellIdx, pickup->pos.x, pickup->pos.y);

    pickup->respawnWait = PICKUP_RESPAWN_WAIT;
    if (e->suddenDeathWallsPlaced) {
        pickup->respawnWait = SUDDEN_DEATH_PICKUP_RESPAWN_WAIT;
    }
    b2DestroyBody(pickup->bodyID);
    pickup->bodyDestroyed = true;

    mapCell *cell = safe_array_get_at(e->cells, pickup->mapCellIdx);
    ASSERT(cell->ent != NULL);
    cell->ent = NULL;

    e->spawnedWeaponPickups[pickup->weapon]--;
}

void createDroneShield(env *e, droneEntity *drone, const int8_t groupIdx) {
    // the shield is comprised of 2 shapes over 2 bodies:
    // 1. a kinematic body that allows the parent drone to be unaffected
    // by collisions since kinematic bodies have essentially infinite mass
    // 2. a shape on the drone body that collides with walls and other
    // shields, since kinematic bodies don't collide with other kinematic
    // bodies or static bodies

    b2BodyDef shieldBodyDef = b2DefaultBodyDef();
    shieldBodyDef.type = b2_kinematicBody;
    shieldBodyDef.fixedRotation = true;
    shieldBodyDef.position = drone->pos;
    b2BodyId shieldBodyID = b2CreateBody(e->worldID, &shieldBodyDef);

    b2ShapeDef shieldShapeDef = b2DefaultShapeDef();
    shieldShapeDef.filter.categoryBits = SHIELD_SHAPE;
    shieldShapeDef.filter.maskBits = PROJECTILE_SHAPE | DRONE_SHAPE;
    shieldShapeDef.filter.groupIndex = groupIdx;
    shieldShapeDef.enableContactEvents = true;

    b2ShapeDef shieldBufferShapeDef = b2DefaultShapeDef();
    shieldBufferShapeDef.density = 0.0f;
    shieldBufferShapeDef.material.friction = DRONE_FRICTION;
    shieldBufferShapeDef.material.restitution = DRONE_RESTITUTION;
    shieldBufferShapeDef.filter.categoryBits = SHIELD_SHAPE;
    shieldBufferShapeDef.filter.maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | SHIELD_SHAPE;
    shieldBufferShapeDef.filter.groupIndex = groupIdx;

    shieldEntity *shield = fastCalloc(1, sizeof(shieldEntity));
    shield->drone = drone;
    shield->bodyID = shieldBodyID;
    shield->pos = drone->pos;
    shield->health = DRONE_SHIELD_HEALTH;
    float duration = DRONE_SHIELD_START_DURATION;
    if (drone->livesLeft != DRONE_LIVES) {
        duration = DRONE_SHIELD_RESPAWN_DURATION;
    }
    shield->duration = duration;

    entity *ent = createEntity(e, SHIELD_ENTITY, shield);
    shield->ent = ent;

    shieldShapeDef.userData = ent;
    const b2Circle shieldCircle = {.center = b2Vec2_zero, .radius = DRONE_SHIELD_RADIUS};
    shield->shapeID = b2CreateCircleShape(shieldBodyID, &shieldShapeDef, &shieldCircle);
    b2Body_SetUserData(shield->bodyID, ent);

    shieldBufferShapeDef.userData = ent;
    shield->bufferShapeID = b2CreateCircleShape(drone->bodyID, &shieldBufferShapeDef, &shieldCircle);

    drone->shield = shield;
}

void createDrone(env *e, const uint8_t idx) {
    const int8_t groupIdx = -(idx + 1);
    b2BodyDef droneBodyDef = b2DefaultBodyDef();
    droneBodyDef.type = b2_dynamicBody;

    int8_t spawnQuad = -1;
    if (!e->isTraining) {
        // spawn drones in diagonal quadrants from each other so that
        // they're more likely to be further apart if we're not training;
        // doing this while training will result in much slower learning
        // due to drones starting much farther apart
        if (e->lastSpawnQuad == -1) {
            spawnQuad = randInt(&e->randState, 0, 3);
        } else if (e->numDrones == 2) {
            spawnQuad = 3 - e->lastSpawnQuad;
        } else {
            spawnQuad = (e->lastSpawnQuad + 1) % 4;
        }
        e->lastSpawnQuad = spawnQuad;
    }
    if (!findOpenPos(e, DRONE_SHAPE, &droneBodyDef.position, spawnQuad)) {
        ERROR("no open position for drone");
    }

    droneBodyDef.fixedRotation = true;
    droneBodyDef.linearDamping = DRONE_LINEAR_DAMPING;
    b2BodyId droneBodyID = b2CreateBody(e->worldID, &droneBodyDef);
    b2ShapeDef droneShapeDef = b2DefaultShapeDef();
    droneShapeDef.density = DRONE_DENSITY;
    droneShapeDef.material.friction = DRONE_FRICTION;
    droneShapeDef.material.restitution = DRONE_RESTITUTION;
    droneShapeDef.filter.categoryBits = DRONE_SHAPE;
    droneShapeDef.filter.maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | WEAPON_PICKUP_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE | SHIELD_SHAPE;
    droneShapeDef.filter.groupIndex = groupIdx;
    droneShapeDef.enableContactEvents = true;
    droneShapeDef.enableSensorEvents = true;
    const b2Circle droneCircle = {.center = b2Vec2_zero, .radius = DRONE_RADIUS};

    droneEntity *drone = fastCalloc(1, sizeof(droneEntity));
    drone->bodyID = droneBodyID;
    drone->weaponInfo = e->defaultWeapon;
    drone->ammo = weaponAmmo(e->defaultWeapon->type, drone->weaponInfo->type);
    drone->energyLeft = DRONE_ENERGY_MAX;
    drone->idx = idx;
    drone->team = idx;
    if (e->teamsEnabled) {
        drone->team = idx / (e->numDrones / 2);
    }
    drone->initalPos = droneBodyDef.position;
    drone->pos = droneBodyDef.position;
    drone->mapCellIdx = entityPosToCellIdx(e, droneBodyDef.position);
    drone->lastAim = (b2Vec2){.x = 0.0f, .y = -1.0f};
    drone->livesLeft = DRONE_LIVES;
    drone->respawnGuideLifetime = UINT16_MAX;
    memset(&drone->stepInfo, 0x0, sizeof(droneStepInfo));

    entity *ent = createEntity(e, DRONE_ENTITY, drone);
    drone->ent = ent;

    droneShapeDef.userData = ent;
    drone->shapeID = b2CreateCircleShape(droneBodyID, &droneShapeDef, &droneCircle);
    b2Body_SetUserData(drone->bodyID, ent);

    cc_array_add(e->drones, drone);

    createDroneShield(e, drone, groupIdx);
}

void droneAddEnergy(droneEntity *drone, float energy) {
    // if a burst is charging, add the energy to the burst charge
    if (drone->chargingBurst) {
        drone->burstCharge = clamp(drone->burstCharge + energy);
    } else {
        drone->energyLeft = clamp(drone->energyLeft + energy);
    }
}

void createDronePiece(env *e, droneEntity *drone, const bool fromShield) {
    const float distance = randFloat(&e->randState, DRONE_PIECE_MIN_DISTANCE, DRONE_PIECE_MAX_DISTANCE);
    const b2Vec2 direction = {.x = randFloat(&e->randState, -1.0f, 1.0f), .y = randFloat(&e->randState, -1.0f, 1.0f)};
    const b2Vec2 pos = b2MulAdd(drone->pos, distance, direction);
    const b2Rot rot = b2MakeRot(randFloat(&e->randState, -PI, PI));

    dronePieceEntity *piece = fastCalloc(1, sizeof(dronePieceEntity));
    piece->droneIdx = drone->idx;
    piece->pos = pos;
    piece->rot = rot;
    piece->isShieldPiece = fromShield;
    piece->lifetime = UINT16_MAX;

    entity *ent = createEntity(e, DRONE_PIECE_ENTITY, piece);
    piece->ent = ent;

    b2BodyDef pieceBodyDef = b2DefaultBodyDef();
    pieceBodyDef.type = b2_dynamicBody;

    pieceBodyDef.position = pos;
    pieceBodyDef.rotation = rot;
    pieceBodyDef.linearDamping = DRONE_PIECE_LINEAR_DAMPING;
    pieceBodyDef.angularDamping = DRONE_PIECE_ANGULAR_DAMPING;
    const float bonus = 1.0f + min(b2Length(drone->velocity) / 15.0f, 5.0f);
    const float speed = randFloat(&e->randState, DRONE_PIECE_MIN_SPEED, DRONE_PIECE_MAX_SPEED) * bonus;
    pieceBodyDef.linearVelocity = b2MulSV(speed, direction);
    pieceBodyDef.angularVelocity = randFloat(&e->randState, -PI, PI);
    pieceBodyDef.userData = ent;
    piece->bodyID = b2CreateBody(e->worldID, &pieceBodyDef);

    b2ShapeDef pieceShapeDef = b2DefaultShapeDef();
    pieceShapeDef.filter.categoryBits = DRONE_PIECE_SHAPE;
    pieceShapeDef.filter.maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | DRONE_PIECE_SHAPE;
    pieceShapeDef.density = 1.0f;
    pieceShapeDef.material.friction = 0.5f;
    pieceShapeDef.userData = ent;

    // make pieces from the shield a bit smaller
    if (fromShield) {
        piece->vertices[0] = (b2Vec2){.x = 0.0f, .y = -1.0f};
        piece->vertices[1] = (b2Vec2){.x = -0.5f, .y = 0.0f};
        piece->vertices[2] = (b2Vec2){.x = 0.5f, .y = 0.0f};
    } else {
        piece->vertices[0] = (b2Vec2){.x = 0.0f, .y = -1.5f};
        piece->vertices[1] = (b2Vec2){.x = -0.75f, .y = 0.0f};
        piece->vertices[2] = (b2Vec2){.x = 0.75f, .y = 0.0f};
    }

    const b2Hull pieceHull = b2ComputeHull(piece->vertices, 3);
    const b2Polygon piecePolygon = b2MakePolygon(&pieceHull, 0.0f);
    piece->shapeID = b2CreatePolygonShape(piece->bodyID, &pieceShapeDef, &piecePolygon);

    cc_array_add(e->dronePieces, piece);
}

void destroyDronePiece(env *e, dronePieceEntity *piece) {
    b2DestroyBody(piece->bodyID);
    destroyEntity(e, piece->ent);
    fastFree(piece);
}

void destroyDroneShield(env *e, shieldEntity *shield, const bool createPieces) {
    droneEntity *drone = shield->drone;
    const float health = shield->health;
    if (health <= 0.0f) {
        droneAddEnergy(drone, DRONE_SHIELD_BREAK_ENERGY_COST);
    }
    drone->shield = NULL;

    b2DestroyBody(shield->bodyID);
    b2DestroyShape(shield->bufferShapeID, false);
    destroyEntity(e, shield->ent);
    fastFree(shield);

    if (!createPieces || health > 0.0f) {
        return;
    }

    // only create pieces if the shield was broken early
    for (uint8_t i = 0; i < DRONE_PIECE_COUNT; i++) {
        createDronePiece(e, drone, true);
    }
}

void destroyDrone(env *e, droneEntity *drone) {
    destroyEntity(e, drone->ent);

    shieldEntity *shield = drone->shield;
    if (shield != NULL) {
        destroyDroneShield(e, shield, false);
    }

    b2DestroyBody(drone->bodyID);
    fastFree(drone);
}

void droneChangeWeapon(const env *e, droneEntity *drone, const enum weaponType newWeapon) {
    // top up ammo but change nothing else if the weapon is the same
    if (drone->weaponInfo->type != newWeapon || drone->dead) {
        drone->weaponInfo = weaponInfos[newWeapon];
        drone->weaponCooldown = 0.0f;
        drone->weaponCharge = 0.0f;
        drone->heat = 0;
    }
    drone->ammo = weaponAmmo(e->defaultWeapon->type, drone->weaponInfo->type);
}

void killDrone(env *e, droneEntity *drone) {
    if (drone->dead || drone->livesLeft == 0) {
        return;
    }
    DEBUG_LOGF("drone %d died", drone->idx);

    drone->livesLeft--;
    drone->dead = true;
    drone->diedThisStep = true;
    drone->respawnWait = DRONE_RESPAWN_WAIT;

    for (uint8_t i = 0; i < DRONE_PIECE_COUNT; i++) {
        createDronePiece(e, drone, false);
    }

    b2Body_Disable(drone->bodyID);
    droneChangeWeapon(e, drone, e->defaultWeapon->type);
    drone->braking = false;
    drone->chargingBurst = false;
    drone->energyFullyDepleted = false;
    drone->shotThisStep = false;
    drone->velocity = b2Vec2_zero;
    drone->lastVelocity = b2Vec2_zero;
}

bool respawnDrone(env *e, droneEntity *drone) {
    b2Vec2 pos;
    if (!findOpenPos(e, DRONE_SHAPE, &pos, -1)) {
        return false;
    }
    b2Body_SetTransform(drone->bodyID, pos, b2Rot_identity);
    b2Body_Enable(drone->bodyID);
    b2Body_SetLinearDamping(drone->bodyID, DRONE_LINEAR_DAMPING);

    drone->dead = false;
    drone->pos = pos;
    drone->respawnGuideLifetime = UINT16_MAX;

    droneAddEnergy(drone, DRONE_ENERGY_RESPAWN_REFILL);

    createDroneShield(e, drone, -(drone->idx + 1));

    if (e->client != NULL) {
        drone->trailPoints.length = 0;
    }

    return true;
}

void createProjectile(env *e, droneEntity *drone, const b2Vec2 normAim) {
    ASSERT_VEC_NORMALIZED(normAim);

    const float radius = drone->weaponInfo->radius;
    float droneRadius = DRONE_RADIUS;
    if (drone->shield != NULL) {
        droneRadius = DRONE_SHIELD_RADIUS;
    }
    // spawn the projectile just outside the drone so they don't
    // immediately collide
    b2Vec2 pos = b2MulAdd(drone->pos, droneRadius + (radius * 1.5f), normAim);
    // if the projectile is inside a wall or out of the map, move the
    // projectile to be just outside the wall
    bool projectileInWall = false;
    int16_t cellIdx = entityPosToCellIdx(e, pos);
    if (cellIdx == -1) {
        projectileInWall = true;
    } else {
        const mapCell *cell = safe_array_get_at(e->cells, cellIdx);
        if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
            projectileInWall = true;
        }
    }
    if (projectileInWall) {
        const b2Vec2 rayEnd = b2MulAdd(drone->pos, droneRadius + (radius * 2.5f), normAim);
        const b2Vec2 translation = b2Sub(rayEnd, drone->pos);
        const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE};
        const b2RayResult rayRes = b2World_CastRayClosest(e->worldID, drone->pos, translation, filter);
        if (rayRes.hit) {
            const b2Vec2 invNormAim = b2MulSV(-1.0f, normAim);
            pos = b2MulAdd(rayRes.point, radius * 1.5f, invNormAim);
        }
    }

    b2BodyDef projectileBodyDef = b2DefaultBodyDef();
    projectileBodyDef.type = b2_dynamicBody;
    projectileBodyDef.isBullet = drone->weaponInfo->isPhysicsBullet;
    projectileBodyDef.linearDamping = drone->weaponInfo->damping;
    projectileBodyDef.enableSleep = drone->weaponInfo->canSleep;
    projectileBodyDef.position = pos;
    b2BodyId projectileBodyID = b2CreateBody(e->worldID, &projectileBodyDef);
    b2ShapeDef projectileShapeDef = b2DefaultShapeDef();
    projectileShapeDef.enableContactEvents = true;
    projectileShapeDef.density = drone->weaponInfo->density;
    projectileShapeDef.material.restitution = 1.0f;
    projectileShapeDef.material.friction = 0.0f;
    projectileShapeDef.filter.categoryBits = PROJECTILE_SHAPE;
    projectileShapeDef.filter.maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE | SHIELD_SHAPE;
    const b2Circle projectileCircle = {.center = b2Vec2_zero, .radius = radius};

    b2ShapeId projectileShapeID = b2CreateCircleShape(projectileBodyID, &projectileShapeDef, &projectileCircle);

    // add a bit of lateral drone velocity to projectile
    b2Vec2 forwardVel = b2MulSV(b2Dot(drone->velocity, normAim), normAim);
    b2Vec2 lateralVel = b2Sub(drone->velocity, forwardVel);
    lateralVel = b2MulSV(projectileShapeDef.density * DRONE_MOVE_AIM_COEF, lateralVel);
    b2Vec2 aim = weaponAdjustAim(&e->randState, drone->weaponInfo->type, drone->heat, normAim);
    b2Vec2 fire = b2MulAdd(lateralVel, weaponFire(&e->randState, drone->weaponInfo->type), aim);
    b2Body_ApplyLinearImpulseToCenter(projectileBodyID, fire, true);

    projectileEntity *projectile = fastCalloc(1, sizeof(projectileEntity));
    projectile->droneIdx = drone->idx;
    projectile->bodyID = projectileBodyID;
    projectile->shapeID = projectileShapeID;
    projectile->weaponInfo = drone->weaponInfo;
    projectile->pos = projectileBodyDef.position;
    projectile->lastPos = projectileBodyDef.position;
    projectile->velocity = b2Body_GetLinearVelocity(projectileBodyID);
    projectile->lastVelocity = projectile->velocity;
    projectile->speed = b2Length(projectile->velocity);
    projectile->lastSpeed = projectile->speed;
    if (projectile->weaponInfo->type == BLACK_HOLE_WEAPON) {
        create_array(&projectile->entsInBlackHole, 4);
    }
    cc_array_add(e->projectiles, projectile);

    entity *ent = createEntity(e, PROJECTILE_ENTITY, projectile);
    projectile->ent = ent;
    b2Body_SetUserData(projectile->bodyID, ent);
    b2Shape_SetUserData(projectile->shapeID, ent);

    // create a sensor shape if needed
    if (projectile->weaponInfo->hasSensor) {
        projectile->sensorID = weaponSensor(projectile->bodyID, projectile->weaponInfo->type);
        b2Shape_SetUserData(projectile->sensorID, ent);
    }
}

// compute value generally from 0-1 based off of how much a projectile(s)
// or explosion(s) caused the hit drone to change velocity
float computeHitStrength(const droneEntity *hitDrone) {
    const float prevSpeed = b2Length(hitDrone->lastVelocity);
    const float curSpeed = b2Length(hitDrone->velocity);
    return fabsf(curSpeed - prevSpeed) / MAX_SPEED;
}

// simplified and copied from box2d/src/shape.c
float getShapeProjectedPerimeter(const b2ShapeId shapeID, const b2Vec2 line) {
    if (b2Shape_GetType(shapeID) == b2_circleShape) {
        const b2Circle circle = b2Shape_GetCircle(shapeID);
        return circle.radius * 2.0f;
    }

    const b2Polygon polygon = b2Shape_GetPolygon(shapeID);
    const b2Vec2 *points = polygon.vertices;
    int count = polygon.count;
    B2_ASSERT(count > 0);
    float value = b2Dot(points[0], line);
    float lower = value;
    float upper = value;
    for (int i = 1; i < count; ++i) {
        value = b2Dot(points[i], line);
        lower = b2MinFloat(lower, value);
        upper = b2MaxFloat(upper, value);
    }

    return upper - lower;
}

// explodes projectile and ensures any other projectiles that are caught
// in the explosion are also destroyed if necessary
void createProjectileExplosion(env *e, projectileEntity *projectile, const bool initalProjectile) {
    if (projectile->needsToBeDestroyed) {
        return;
    }
    projectile->needsToBeDestroyed = true;
    cc_array_add(e->explodingProjectiles, projectile);

    b2ExplosionDef explosion;
    weaponExplosion(projectile->weaponInfo->type, &explosion);
    explosion.position = projectile->pos;
    explosion.maskBits = FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE;
    droneEntity *parentDrone = safe_array_get_at(e->drones, projectile->droneIdx);
    createExplosion(e, parentDrone, projectile, &explosion);

    if (e->client != NULL) {
        explosionInfo *explInfo = fastCalloc(1, sizeof(explosionInfo));
        explInfo->def = explosion;
        explInfo->renderSteps = UINT16_MAX;
        cc_array_add(e->explosions, explInfo);
    }
    if (!initalProjectile) {
        return;
    }

    // if we're not destroying the projectiles now, we need to remove the initial projectile
    // from the list of exploding projectiles so it's not destroyed twice
    const enum cc_stat res = cc_array_remove_fast(e->explodingProjectiles, projectile, NULL);
    MAYBE_UNUSED(res);
    ASSERT(res == CC_OK);
}

// ensures projectiles don't slow down when an impluse is applied to them
void fixProjectileSpeed(projectileEntity *projectile) {
    b2Vec2 newVel = b2Body_GetLinearVelocity(projectile->bodyID);
    float newSpeed = b2Length(newVel);
    if (newSpeed < projectile->lastSpeed) {
        newSpeed = projectile->lastSpeed;
        newVel = b2MulSV(newSpeed, b2Normalize(newVel));
        b2Body_SetLinearVelocity(projectile->bodyID, newVel);
    }

    projectile->velocity = newVel;
    projectile->lastSpeed = newSpeed;
    projectile->speed = newSpeed;
}

#define MAX_WALL_HITS 8

typedef struct wallBurstImpulse {
    float distance;
    b2Vec2 direction;
    float magnitude;
    uint16_t wallCellIdx;
} wallBurstImpulse;

typedef struct explosionCtx {
    env *e;
    const bool isBurst;
    droneEntity *parentDrone;
    const projectileEntity *projectile;
    const b2ExplosionDef *def;

    wallBurstImpulse wallImpulses[MAX_WALL_HITS];
    int8_t closestWallIdx;
    uint8_t wallsHit;
} explosionCtx;

const float COS_85_DEGREES = 0.087155743f;

// b2World_Explode doesn't support filtering on shapes of the same category,
// so we have to do it manually
// mostly copied from box2d/src/world.c
bool explodeCallback(b2ShapeId shapeID, void *context) {
    if (!b2Shape_IsValid(shapeID)) {
        return true;
    }

    explosionCtx *ctx = context;
    const entity *entity = b2Shape_GetUserData(shapeID);
    projectileEntity *projectile = NULL;
    droneEntity *drone = NULL;
    wallEntity *wall = NULL;
    bool isStaticWall = false;
    b2Transform transform;

    switch (entity->type) {
    case PROJECTILE_ENTITY:
        // don't explode the parent projectile
        projectile = entity->entity;
        if (ctx->projectile != NULL && (ctx->projectile == projectile || projectile->needsToBeDestroyed)) {
            return true;
        }
        transform.p = projectile->pos;
        transform.q = b2Rot_identity;
        break;
    case DRONE_ENTITY:
        drone = entity->entity;
        // the explosion shouldn't affect the parent drone if this is a burst
        if (drone->idx == ctx->parentDrone->idx) {
            if (ctx->isBurst) {
                return true;
            }

            drone->stepInfo.ownShotTaken = true;
            ctx->e->stats[drone->idx].ownShotsTaken[ctx->projectile->weaponInfo->type]++;
            DEBUG_LOGF("drone %d hit itself with explosion from weapon %d", drone->idx, ctx->projectile->weaponInfo->type);
        }
        ctx->parentDrone->stepInfo.explosionHit[drone->idx] = true;
        if (ctx->isBurst) {
            DEBUG_LOGF("drone %d hit drone %d with burst", ctx->parentDrone->idx, drone->idx);
            ctx->e->stats[ctx->parentDrone->idx].burstsHit++;
            DEBUG_LOGF("drone %d hit by burst from drone %d", drone->idx, ctx->parentDrone->idx);
        } else {
            DEBUG_LOGF("drone %d hit drone %d with explosion from weapon %d", ctx->parentDrone->idx, drone->idx, ctx->projectile->weaponInfo->type);
            ctx->e->stats[ctx->parentDrone->idx].shotsHit[ctx->projectile->weaponInfo->type]++;
            DEBUG_LOGF("drone %d hit by explosion from weapon %d from drone %d", drone->idx, ctx->projectile->weaponInfo->type, ctx->parentDrone->idx);
        }
        drone->stepInfo.explosionTaken[ctx->parentDrone->idx] = true;
        transform.p = drone->pos;
        transform.q = b2Rot_identity;
        break;
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY:
        wall = entity->entity;
        isStaticWall = !wall->isFloating;
        // normal explosions don't affect static walls
        if (!ctx->isBurst && isStaticWall) {
            return true;
        }
        transform.p = wall->pos;
        transform.q = wall->rot;
        break;
    default:
        ERRORF("invalid entity type %d to explode", entity->type);
    }

    // find the closest point from the entity to the explosion center
    const b2BodyId bodyID = b2Shape_GetBody(shapeID);
    ASSERT(b2Body_IsValid(bodyID));

    bool isCircle = false;
    b2DistanceInput input;
    input.proxyA = makeDistanceProxy(entity, &isCircle);
    input.proxyB = b2MakeProxy(&ctx->def->position, 1, 0.0f);
    input.transformA = transform;
    input.transformB = b2Transform_identity;
    input.useRadii = isCircle;

    b2SimplexCache cache = {0};
    const b2DistanceOutput output = b2ShapeDistance(&input, &cache, NULL, 0);
    if (output.distance > ctx->def->radius) {
        return true;
    }

    // don't explode the entity if it's behind a static or floating wall,
    // but always consider floating walls for implosions
    const bool isImplosion = ctx->def->impulsePerLength < 0.0f;
    b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE};
    if (!isImplosion) {
        filter.maskBits |= FLOATING_WALL_SHAPE;
    }
    if (posBehindWall(ctx->e, ctx->def->position, output.pointA, entity, filter, NULL)) {
        return true;
    }

    const b2Vec2 closestPoint = output.pointA;
    b2Vec2 direction;
    if (isStaticWall) {
        direction = b2Normalize(b2Sub(ctx->def->position, closestPoint));
    } else {
        direction = b2Normalize(b2Sub(closestPoint, ctx->def->position));
    }
    // if the direction is zero, the magnitude cannot be calculated
    // correctly so set the direction randomly
    if (b2VecEqual(direction, b2Vec2_zero)) {
        direction.x = randFloat(&ctx->e->randState, -1.0f, 1.0f);
        direction.y = randFloat(&ctx->e->randState, -1.0f, 1.0f);
        direction = b2Normalize(direction);
    }

    b2Vec2 localLine = b2Vec2_zero;
    if (entityTypeIsWall(entity->type)) {
        // the localLine isn't used in perimeter calculations for circles
        localLine = b2InvRotateVector(transform.q, b2LeftPerp(direction));
    }
    float perimeter = getShapeProjectedPerimeter(shapeID, localLine);
    const float relDistance = output.distance / ctx->def->radius;
    const float scale = 1.0f - (SQUARED(relDistance) * relDistance);

    // the parent drone or projecile's velocity affects the direction
    // and magnitude of the explosion
    b2Vec2 parentVelocity;
    float parentSpeed;
    if (ctx->projectile != NULL) {
        // use the projectile's last velocity and speed if it is in
        // contact with another body, as the current velocity will be
        // the velocity after the projectile rebounds which is not
        // what we want
        if (ctx->projectile->contacts != 0) {
            parentVelocity = ctx->projectile->lastVelocity;
            parentSpeed = ctx->projectile->lastSpeed;
        } else {
            parentVelocity = ctx->projectile->velocity;
            parentSpeed = ctx->projectile->speed;
        }
        if (isImplosion) {
            parentSpeed *= -1.0f;
        }
    } else {
        parentVelocity = ctx->parentDrone->lastVelocity;
        parentSpeed = b2Length(parentVelocity);
    }
    const b2Vec2 parentDirection = b2Normalize(parentVelocity);

    // scale the parent speed by how close the movement direction of
    // the parent is to where the entity is to the parent, except if
    // we're bursting off of a wall to make it more predictable and
    // to prevent taking a log of a negative number
    if (!isStaticWall) {
        parentSpeed *= b2Dot(direction, parentDirection);
    }
    // the parent entity's velocity affects the direction of the impulse
    // depending on the speed
    const b2Vec2 baseImpulse = b2MulSV(fabsf(ctx->def->impulsePerLength), direction);
    direction = b2Normalize(b2Add(baseImpulse, parentVelocity));

    float shieldReduction = 1.0f;
    if (drone != NULL && drone->shield != NULL) {
        shieldReduction = DRONE_SHIELD_EXPLOSION_REDUCTION;
    }

    float magnitude = (ctx->def->impulsePerLength + parentSpeed) * perimeter * scale * shieldReduction;
    if (isStaticWall) {
        // ensure this wall faces at least 85 degrees away from the
        // closest hit wall to prevent multiple walls facing roughly
        // the same direction greatly increasing the impulse magnitude
        if (ctx->closestWallIdx == -1) {
            ctx->closestWallIdx = 0;
        } else {
            const float closestWallDistance = ctx->wallImpulses[ctx->closestWallIdx].distance;
            if (output.distance < closestWallDistance) {
                ctx->closestWallIdx = ctx->wallsHit;

                for (int8_t i = 0; i < ctx->wallsHit; i++) {
                    const wallBurstImpulse impulse = ctx->wallImpulses[i];
                    if (impulse.distance != output.distance && b2Dot(impulse.direction, direction) > COS_85_DEGREES) {
                        ctx->wallImpulses[i] = ctx->wallImpulses[--ctx->wallsHit];
                        i--;
                    }
                }
            }
        }

        // reduce the magnitude when pushing a drone away from a wall
        magnitude = log2f(magnitude) * (5.0f + (25.0f * ctx->parentDrone->burstCharge));

        ctx->wallImpulses[ctx->wallsHit++] = (wallBurstImpulse){
            .distance = output.distance,
            .direction = direction,
            .magnitude = magnitude,
            .wallCellIdx = wall->mapCellIdx,
        };
        return true;
    }
    const b2Vec2 impulse = b2MulSV(magnitude, direction);
    ASSERT(b2IsValidVec2(impulse));

    switch (entity->type) {
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY:
        b2Body_ApplyLinearImpulse(bodyID, impulse, output.pointA, true);
        wall->velocity = b2Body_GetLinearVelocity(wall->bodyID);
        break;
    case PROJECTILE_ENTITY:
        b2Body_ApplyLinearImpulseToCenter(bodyID, impulse, true);
        // mine launcher projectiles explode when caught in another
        // explosion, explode this mine only once
        if (projectile->weaponInfo->type == MINE_LAUNCHER_WEAPON && ctx->def->impulsePerLength > 0.0f) {
            createProjectileExplosion(ctx->e, projectile, false);
            break;
        }

        fixProjectileSpeed(projectile);

        break;
    case DRONE_ENTITY:
        b2Body_ApplyLinearImpulseToCenter(bodyID, impulse, true);
        drone->lastVelocity = drone->velocity;
        drone->velocity = b2Body_GetLinearVelocity(drone->bodyID);

        shieldEntity *shield = drone->shield;

        // add energy to the drone that fired the projectile that is
        // currently exploding if it hit another drone
        if (!ctx->isBurst && drone->team != ctx->parentDrone->team && shield == NULL) {
            const float energyRefill = computeHitStrength(drone) * EXPLOSION_ENERGY_REFILL_COEF;
            droneAddEnergy(ctx->parentDrone, energyRefill);
        } else if (shield != NULL && shield->health > 0.0f) {
            const float damage = fabsf(magnitude) * DRONE_SHIELD_HEALTH_EXPLOSION_COEF;
            DEBUG_LOGF("shield explosion damage: %f", damage);
            shield->health -= damage;
            if (shield->health <= 0.0f) {
                droneAddEnergy(ctx->parentDrone, DRONE_SHIELD_BREAK_ENERGY_REFILL);
            }
        }

        break;
    default:
        ERRORF("unknown entity type for burst impulse %d", entity->type);
    }

    return true;
}

void applyDroneBurstImpulse(env *e, explosionCtx *ctx, const droneEntity *drone) {
    wallBurstImpulse *hitWalls = ctx->wallImpulses;
    uint8_t wallsHit = ctx->wallsHit;

    // sort by distance
    for (int8_t i = 1; i < wallsHit; i++) {
        wallBurstImpulse key = hitWalls[i];
        int j = i - 1;

        while (j >= 0 && hitWalls[j].distance > key.distance) {
            hitWalls[j + 1] = hitWalls[j];
            j = j - 1;
        }

        hitWalls[j + 1] = key;
    }

    for (uint8_t i = 0; i < wallsHit; i++) {
        const wallBurstImpulse impulseA = hitWalls[i];
        if (impulseA.distance == FLT_MAX) {
            continue;
        }
        for (uint8_t j = 0; j < wallsHit; j++) {
            if (j == i) {
                continue;
            }
            wallBurstImpulse impulseB = hitWalls[j];
            if (impulseB.distance == FLT_MAX) {
                continue;
            }
            const uint8_t cellIdxDiff = abs(impulseB.wallCellIdx - impulseA.wallCellIdx);
            if (cellIdxDiff == e->map->columns) {
                hitWalls[j].distance = FLT_MAX;
                continue;
            }
            if (cellIdxDiff == 1) {
                int8_t colA = impulseA.wallCellIdx / e->map->columns;
                int8_t colB = impulseB.wallCellIdx / e->map->columns;
                if (colA == colB) {
                    hitWalls[j].distance = FLT_MAX;
                }
            }
        }
    }

    uint8_t wallsUsed = 0;
    float magnitude = 0.0f;
    b2Vec2 direction = b2Vec2_zero;
    for (uint8_t i = 0; i < wallsHit; i++) {
        const wallBurstImpulse impulse = hitWalls[i];
        if (impulse.distance == FLT_MAX) {
            continue;
        }

        wallsUsed++;
        magnitude += impulse.magnitude;
        direction = b2Add(direction, impulse.direction);
    }

    if (wallsUsed > 2) {
        DEBUG_LOG("more than 2 walls used");
    }

    DEBUG_LOGF("walls used: %d magnitude: %f final: %f", wallsUsed, magnitude, magnitude / (float)wallsUsed);
    const b2Vec2 finalImpulse = b2MulSV(magnitude / (float)wallsUsed, b2Normalize(direction));
    ASSERT(b2IsValidVec2(finalImpulse));
    b2Body_ApplyLinearImpulseToCenter(drone->bodyID, finalImpulse, true);
}

void createExplosion(env *e, droneEntity *drone, const projectileEntity *projectile, const b2ExplosionDef *def) {
    b2AABB aabb = {
        .lowerBound.x = def->position.x - def->radius,
        .lowerBound.y = def->position.y - def->radius,
        .upperBound.x = def->position.x + def->radius,
        .upperBound.y = def->position.y + def->radius,
    };

    b2QueryFilter filter = b2DefaultQueryFilter();
    filter.categoryBits = PROJECTILE_SHAPE;
    filter.maskBits = def->maskBits;

    const bool isBurst = projectile == NULL;
    explosionCtx ctx = {
        .e = e,
        .isBurst = isBurst,
        .parentDrone = drone,
        .projectile = projectile,
        .def = def,
        .wallImpulses = {{0}},
        .closestWallIdx = -1,
        .wallsHit = 0,
    };
    b2World_OverlapAABB(e->worldID, aabb, filter, explodeCallback, &ctx);

    uint8_t wallsHit = ctx.wallsHit;
    if (isBurst && wallsHit != 0) {
        applyDroneBurstImpulse(e, &ctx, drone);
    }
}

void destroyProjectile(env *e, projectileEntity *projectile, const bool processExplosions, const bool full) {
    // explode projectile if necessary
    if (processExplosions && projectile->weaponInfo->explosive) {
        createProjectileExplosion(e, projectile, true);
    }

    destroyEntity(e, projectile->ent);

    b2DestroyBody(projectile->bodyID);

    if (full) {
        enum cc_stat res = cc_array_remove_fast(e->projectiles, projectile, NULL);
        MAYBE_UNUSED(res);
        ASSERT(res == CC_OK);
    }

    e->stats[projectile->droneIdx].shotDistances[projectile->droneIdx] += projectile->distance;

    if (projectile->entsInBlackHole != NULL) {
        for (uint8_t i = 0; i < cc_array_size(projectile->entsInBlackHole); i++) {
            entityID *id = safe_array_get_at(projectile->entsInBlackHole, i);
            fastFree(id);
        }
        cc_array_destroy(projectile->entsInBlackHole);
    }

    fastFree(projectile);
}

// destroy projectiles that were caught in an explosion; projectiles
// can't be destroyed in explodeCallback because box2d assumes all shapes
// and bodies are valid for the lifetime of an AABB query
static inline void destroyExplodedProjectiles(env *e) {
    if (cc_array_size(e->explodingProjectiles) == 0) {
        return;
    }

    CC_ArrayIter iter;
    cc_array_iter_init(&iter, e->explodingProjectiles);
    projectileEntity *projectile;
    while (cc_array_iter_next(&iter, (void **)&projectile) != CC_ITER_END) {
        destroyProjectile(e, projectile, false, false);
        const enum cc_stat res = cc_array_remove_fast(e->projectiles, projectile, NULL);
        MAYBE_UNUSED(res);
        ASSERT(res == CC_OK);
    }
    cc_array_remove_all(e->explodingProjectiles);
}

void createSuddenDeathWalls(env *e, const b2Vec2 startPos, const b2Vec2 size) {
    int16_t endIdx;
    uint8_t indexIncrement;
    if (size.y == WALL_THICKNESS) {
        // horizontal walls
        const b2Vec2 endPos = (b2Vec2){.x = startPos.x + size.x, .y = startPos.y};
        endIdx = entityPosToCellIdx(e, endPos);
        if (endIdx == -1) {
            ERRORF("invalid position for sudden death wall: (%f, %f)", endPos.x, endPos.y);
        }
        indexIncrement = 1;
    } else {
        // vertical walls
        const b2Vec2 endPos = (b2Vec2){.x = startPos.x, .y = startPos.y + size.y};
        endIdx = entityPosToCellIdx(e, endPos);
        if (endIdx == -1) {
            ERRORF("invalid position for sudden death wall: (%f, %f)", endPos.x, endPos.y);
        }
        indexIncrement = e->map->columns;
    }
    const int16_t startIdx = entityPosToCellIdx(e, startPos);
    if (startIdx == -1) {
        ERRORF("invalid position for sudden death wall: (%f, %f)", startPos.x, startPos.y);
    }
    for (uint16_t i = startIdx; i <= endIdx; i += indexIncrement) {
        mapCell *cell = safe_array_get_at(e->cells, i);
        if (cell->ent != NULL) {
            if (cell->ent->type == WEAPON_PICKUP_ENTITY) {
                weaponPickupEntity *pickup = cell->ent->entity;
                disableWeaponPickup(e, pickup);
            } else {
                continue;
            }
        }
        entity *ent = createWall(e, cell->pos, WALL_THICKNESS, WALL_THICKNESS, i, DEATH_WALL_ENTITY, false);
        cell->ent = ent;
    }
}

// TODO: handle when all walls are placed
void handleSuddenDeath(env *e) {
    ASSERT(e->suddenDeathSteps == 0);

    // create new walls that will close in on the arena
    e->suddenDeathWallCounter++;
    e->suddenDeathWallsPlaced = true;

    const float leftX = (e->suddenDeathWallCounter - 1) * WALL_THICKNESS;
    const float yOffset = (WALL_THICKNESS * (e->suddenDeathWallCounter - 1)) + (WALL_THICKNESS / 2);
    const float xWidth = WALL_THICKNESS * (e->map->columns - (e->suddenDeathWallCounter * 2) - 1);

    // top walls
    createSuddenDeathWalls(
        e,
        (b2Vec2){
            .x = e->map->bounds.min.x + leftX,
            .y = e->map->bounds.min.y + yOffset,
        },
        (b2Vec2){
            .x = xWidth,
            .y = WALL_THICKNESS,
        }
    );
    // bottom walls
    createSuddenDeathWalls(
        e,
        (b2Vec2){
            .x = e->map->bounds.min.x + leftX,
            .y = e->map->bounds.max.y - yOffset,
        },
        (b2Vec2){
            .x = xWidth,
            .y = WALL_THICKNESS,
        }
    );
    // left walls
    createSuddenDeathWalls(
        e,
        (b2Vec2){
            .x = e->map->bounds.min.x + leftX,
            .y = e->map->bounds.min.y + (e->suddenDeathWallCounter * WALL_THICKNESS),
        },
        (b2Vec2){
            .x = WALL_THICKNESS,
            .y = WALL_THICKNESS * (e->map->rows - (e->suddenDeathWallCounter * 2) - 2),
        }
    );
    // right walls
    createSuddenDeathWalls(
        e,
        (b2Vec2){
            .x = e->map->bounds.min.x + ((e->map->columns - e->suddenDeathWallCounter - 2) * WALL_THICKNESS),
            .y = e->map->bounds.min.y + (e->suddenDeathWallCounter * WALL_THICKNESS),
        },
        (b2Vec2){
            .x = WALL_THICKNESS,
            .y = WALL_THICKNESS * (e->map->rows - (e->suddenDeathWallCounter * 2) - 2),
        }
    );

    // mark drones as dead if they touch a newly placed wall
    for (uint8_t i = 0; i < e->numDrones; i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        const b2QueryFilter filter = {
            .categoryBits = DRONE_SHAPE,
            .maskBits = WALL_SHAPE,
        };
        if (isOverlappingAABB(e, drone->pos, DRONE_RADIUS, filter)) {
            killDrone(e, drone);
        }
    }

    // make floating walls static bodies if they are now overlapping with
    // a newly placed wall, but destroy them if they are fully inside a wall
    CC_ArrayIter floatingWallIter;
    cc_array_iter_init(&floatingWallIter, e->floatingWalls);
    wallEntity *wall;
    while (cc_array_iter_next(&floatingWallIter, (void **)&wall) != CC_ITER_END) {
        const mapCell *cell = safe_array_get_at(e->cells, wall->mapCellIdx);
        if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
            // floating wall is overlapping with a wall, destroy it
            const enum cc_stat res = cc_array_iter_remove_fast(&floatingWallIter, NULL);
            MAYBE_UNUSED(res);
            ASSERT(res == CC_OK);

            const b2Vec2 wallPos = wall->pos;
            MAYBE_UNUSED(wallPos);
            destroyWall(e, wall, false);
            DEBUG_LOGF("destroyed floating wall at %f, %f", wallPos.x, wallPos.y);
            continue;
        }
    }

    // detroy all projectiles that are now overlapping with a newly placed wall
    CC_ArrayIter projectileIter;
    cc_array_iter_init(&projectileIter, e->projectiles);
    projectileEntity *projectile;
    while (cc_array_iter_next(&projectileIter, (void **)&projectile) != CC_ITER_END) {
        const mapCell *cell = safe_array_get_at(e->cells, projectile->mapCellIdx);
        if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
            cc_array_iter_remove_fast(&projectileIter, NULL);
            destroyProjectile(e, projectile, false, false);
        }
    }
}

void droneMove(droneEntity *drone, b2Vec2 direction) {
    ASSERT_VEC_BOUNDED(direction);

    // if energy is fully depleted halve movement until energy starts
    // to refill again
    if (drone->energyFullyDepleted && drone->energyRefillWait != 0.0f) {
        direction = b2MulSV(0.5f, direction);
        drone->lastMove = direction;
    }
    const b2Vec2 force = b2MulSV(DRONE_MOVE_MAGNITUDE, direction);
    b2Body_ApplyForceToCenter(drone->bodyID, force, true);
}

void droneShoot(env *e, droneEntity *drone, const b2Vec2 aim, const bool chargingWeapon) {
    ASSERT(drone->ammo != 0);

    drone->shotThisStep = true;
    // TODO: rework heat to only increase when projectiles are fired,
    // and only cool down after the next shot was skipped
    drone->heat++;
    if (drone->weaponCooldown != 0.0f) {
        return;
    }
    const bool weaponNeedsCharge = drone->weaponInfo->charge != 0.0f;
    if (weaponNeedsCharge) {
        if (chargingWeapon) {
            drone->chargingWeapon = true;
            drone->weaponCharge = min(drone->weaponCharge + e->deltaTime, drone->weaponInfo->charge);
        } else if (drone->weaponCharge < drone->weaponInfo->charge) {
            drone->chargingWeapon = false;
            drone->weaponCharge = max(drone->weaponCharge - e->deltaTime, 0.0f);
        }
    }
    // if the weapon needs to be charged, only fire the weapon if it's
    // fully charged and the agent released the trigger
    if (weaponNeedsCharge && (chargingWeapon || drone->weaponCharge < drone->weaponInfo->charge)) {
        return;
    }

    if (drone->ammo != INFINITE) {
        drone->ammo--;
    }
    drone->weaponCooldown = drone->weaponInfo->coolDown;
    drone->chargingWeapon = false;
    drone->weaponCharge = 0.0f;

    b2Vec2 normAim = drone->lastAim;
    if (!b2VecEqual(aim, b2Vec2_zero)) {
        normAim = b2Normalize(aim);
    }
    ASSERT_VEC_NORMALIZED(normAim);
    b2Vec2 recoil = b2MulSV(-drone->weaponInfo->recoilMagnitude, normAim);
    b2Body_ApplyLinearImpulseToCenter(drone->bodyID, recoil, true);

    for (int i = 0; i < drone->weaponInfo->numProjectiles; i++) {
        createProjectile(e, drone, normAim);

        e->stats[drone->idx].shotsFired[drone->weaponInfo->type]++;
    }
    drone->stepInfo.firedShot = true;

    if (drone->ammo == 0) {
        droneChangeWeapon(e, drone, e->defaultWeapon->type);
        drone->weaponCooldown = drone->weaponInfo->coolDown;
    }
}

void droneBrake(env *e, droneEntity *drone, const bool brake) {
    if (drone->shield != NULL) {
        return;
    }
    // if the drone isn't braking or energy is fully depleted, return
    // unless the drone was braking during the last step
    if (!brake || drone->energyFullyDepleted) {
        if (drone->braking) {
            drone->braking = false;
            b2Body_SetLinearDamping(drone->bodyID, DRONE_LINEAR_DAMPING);
            if (drone->energyRefillWait == 0.0f && !drone->chargingBurst) {
                drone->energyRefillWait = DRONE_ENERGY_REFILL_WAIT;
            }
        }
        return;
    }
    ASSERT(!drone->energyFullyDepleted);

    // apply additional brake damping and decrease energy
    if (brake) {
        if (!drone->braking) {
            drone->braking = true;
            b2Body_SetLinearDamping(drone->bodyID, DRONE_LINEAR_DAMPING * DRONE_BRAKE_DAMPING_COEF);
        }
        drone->energyLeft = max(drone->energyLeft - (DRONE_BRAKE_DRAIN_RATE * e->deltaTime), 0.0f);
        e->stats[drone->idx].brakeTime += e->deltaTime;
    }

    // if energy is empty but burst is being charged, let burst functions
    // handle energy refill
    if (drone->energyLeft == 0.0f && !drone->chargingBurst) {
        drone->energyFullyDepleted = true;
        drone->energyFullyDepletedThisStep = true;
        drone->energyRefillWait = DRONE_ENERGY_REFILL_EMPTY_WAIT;
        e->stats[drone->idx].energyEmptied++;
    }

    if (e->client != NULL) {
        brakeTrailPoint *trailPoint = fastCalloc(1, sizeof(brakeTrailPoint));
        trailPoint->pos = drone->pos;
        trailPoint->lifetime = UINT16_MAX;
        cc_array_add(e->brakeTrailPoints, trailPoint);
    }
}

void droneChargeBurst(env *e, droneEntity *drone) {
    if (drone->energyFullyDepleted || drone->burstCooldown != 0.0f || drone->shield != NULL || (!drone->chargingBurst && drone->energyLeft < DRONE_BURST_BASE_COST)) {
        return;
    }

    // take energy and put it into burst charge
    if (drone->chargingBurst) {
        drone->burstCharge = min(drone->burstCharge + (DRONE_BURST_CHARGE_RATE * e->deltaTime), DRONE_ENERGY_MAX);
        drone->energyLeft = max(drone->energyLeft - (DRONE_BURST_CHARGE_RATE * e->deltaTime), 0.0f);
    } else {
        drone->burstCharge = min(drone->burstCharge + DRONE_BURST_BASE_COST, DRONE_ENERGY_MAX);
        drone->energyLeft = max(drone->energyLeft - DRONE_BURST_BASE_COST, 0.0f);
        drone->chargingBurst = true;
    }

    if (drone->energyLeft == 0.0f) {
        drone->energyFullyDepleted = true;
        e->stats[drone->idx].energyEmptied++;
    }
}

void droneBurst(env *e, droneEntity *drone) {
    if (!drone->chargingBurst) {
        return;
    }

    const float radius = (DRONE_BURST_RADIUS_BASE * drone->burstCharge) + DRONE_BURST_RADIUS_MIN;
    b2ExplosionDef explosion = {
        .position = drone->pos,
        .radius = radius,
        .impulsePerLength = (DRONE_BURST_IMPACT_BASE * drone->burstCharge) + DRONE_BURST_IMPACT_MIN,
        .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE,
    };
    createExplosion(e, drone, NULL, &explosion);
    destroyExplodedProjectiles(e);

    drone->chargingBurst = false;
    drone->burstCharge = 0.0f;
    drone->burstCooldown = DRONE_BURST_COOLDOWN;
    if (drone->energyLeft == 0.0f) {
        drone->energyFullyDepletedThisStep = true;
        drone->energyRefillWait = DRONE_ENERGY_REFILL_EMPTY_WAIT;
    } else {
        drone->energyRefillWait = DRONE_ENERGY_REFILL_WAIT;
    }
    e->stats[drone->idx].totalBursts++;

    if (e->client != NULL) {
        explosionInfo *explInfo = fastCalloc(1, sizeof(explosionInfo));
        explInfo->def = explosion;
        explInfo->isBurst = true;
        explInfo->droneIdx = drone->idx;
        explInfo->renderSteps = UINT16_MAX;
        cc_array_add(e->explosions, explInfo);
    }
}

void droneDiscardWeapon(env *e, droneEntity *drone) {
    if (drone->weaponInfo->type == e->defaultWeapon->type || (drone->energyFullyDepleted && !drone->chargingBurst)) {
        return;
    }

    droneChangeWeapon(e, drone, e->defaultWeapon->type);
    droneAddEnergy(drone, -WEAPON_DISCARD_COST);
    if (drone->chargingBurst) {
        return;
    }

    if (drone->energyLeft == 0.0f) {
        drone->energyFullyDepleted = true;
        drone->energyFullyDepletedThisStep = true;
        drone->energyRefillWait = DRONE_ENERGY_REFILL_EMPTY_WAIT;
        e->stats[drone->idx].energyEmptied++;
    } else {
        drone->energyRefillWait = DRONE_ENERGY_REFILL_WAIT;
    }
}

// update drone state, respawn the drone if necessary; false is returned
// if no position could be found to respawn the drone at, and true otherwise
bool droneStep(env *e, droneEntity *drone) {
    if (drone->dead) {
        drone->respawnWait -= e->deltaTime;
        if (drone->respawnWait <= 0.0f) {
            const bool foundPos = respawnDrone(e, drone);
            if (!foundPos) {
                return false;
            }
        }
        return true;
    }

    // manage weapon charge and heat
    if (drone->weaponCooldown != 0.0f) {
        drone->weaponCooldown = max(drone->weaponCooldown - e->deltaTime, 0.0f);
    }
    if (!drone->shotThisStep) {
        drone->weaponCharge = max(drone->weaponCharge - e->deltaTime, 0);
        drone->heat = max(drone->heat - 1, 0);
    } else {
        drone->shotThisStep = false;
    }
    ASSERT(!drone->shotThisStep);

    // manage drone energy
    if (drone->burstCooldown != 0.0f) {
        drone->burstCooldown = max(drone->burstCooldown - e->deltaTime, 0.0f);
    }
    if (drone->energyFullyDepletedThisStep) {
        drone->energyFullyDepletedThisStep = false;
    } else if (drone->energyRefillWait != 0.0f) {
        drone->energyRefillWait = max(drone->energyRefillWait - e->deltaTime, 0.0f);
    } else if (drone->energyLeft != DRONE_ENERGY_MAX && !drone->chargingBurst) {
        // don't start recharging energy until the burst charge is used
        drone->energyLeft = min(drone->energyLeft + (DRONE_ENERGY_REFILL_RATE * e->deltaTime), DRONE_ENERGY_MAX);
    }
    if (drone->energyLeft == DRONE_ENERGY_MAX) {
        drone->energyFullyDepleted = false;
    }

    const float distance = b2Distance(drone->lastPos, drone->pos);
    e->stats[drone->idx].distanceTraveled += distance;

    shieldEntity *shield = drone->shield;
    if (shield != NULL) {
        shield->duration -= e->deltaTime;
        if (shield->duration <= 0.0f || shield->health <= 0.0f) {
            destroyDroneShield(e, shield, true);
        }
    }

    return true;
}

void handleBlackHolePull(env *e, projectileEntity *projectile) {
    ASSERT(projectile->weaponInfo->type == BLACK_HOLE_WEAPON);

    CC_ArrayIter entIter;
    cc_array_iter_init(&entIter, projectile->entsInBlackHole);
    entityID *id;
    while (cc_array_iter_next(&entIter, (void **)&id) != CC_ITER_END) {
        // check if the entity is still valid
        const entity *ent = getEntityByID(e, id);
        if (ent == NULL) {
            fastFree(id);
            enum cc_stat res = cc_array_iter_remove_fast(&entIter, NULL);
            MAYBE_UNUSED(res);
            ASSERT(res == CC_OK);
            continue;
        }
        const b2DistanceOutput output = closestPoint(projectile->ent, ent);
        const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
        if (posBehindWall(e, projectile->pos, output.pointB, ent, filter, NULL)) {
            continue;
        }

        b2BodyId bodyID;
        b2ShapeId shapeID;
        b2Rot rot;

        wallEntity *wall;
        droneEntity *drone;
        projectileEntity *proj;
        bool hasShield = false;

        switch (ent->type) {
        case STANDARD_WALL_ENTITY:
        case BOUNCY_WALL_ENTITY:
        case DEATH_WALL_ENTITY:
            wall = ent->entity;
            bodyID = wall->bodyID;
            shapeID = wall->shapeID;
            rot = wall->rot;
            break;
        case DRONE_ENTITY:
            drone = ent->entity;
            bodyID = drone->bodyID;
            shapeID = drone->shapeID;
            hasShield = drone->shield != NULL;
            break;
        case PROJECTILE_ENTITY:
            proj = ent->entity;
            bodyID = proj->bodyID;
            shapeID = proj->shapeID;
            break;
        default:
            ERRORF("unknown entity type %d for black hole suck", ent->type);
        }

        b2Vec2 direction = b2Normalize(b2Sub(output.pointB, projectile->pos));

        b2Vec2 localLine = b2Vec2_zero;
        if (entityTypeIsWall(ent->type)) {
            // the localLine isn't used in perimeter calculations for circles
            localLine = b2InvRotateVector(rot, b2LeftPerp(direction));
        }
        const float perimeter = getShapeProjectedPerimeter(shapeID, localLine);

        const float scale = 1.0f - (output.distance / BLACK_HOLE_PROXIMITY_RADIUS);

        // the projecile's velocity affects the direction
        // and magnitude of the force
        b2Vec2 parentVelocity = projectile->velocity;
        float parentSpeed = projectile->speed;
        if (projectile->contacts != 0) {
            parentVelocity = projectile->lastVelocity;
            parentSpeed = projectile->lastSpeed;
        }
        const b2Vec2 parentDirection = b2Normalize(parentVelocity);
        parentSpeed *= b2Dot(direction, parentDirection);
        // the parent entity's velocity affects the direction of the force
        // depending on the speed
        const b2Vec2 baseForce = b2MulSV(-1.0f * BLACK_HOLE_PULL_MAGNITUDE, direction);
        direction = b2Normalize(b2Add(baseForce, parentVelocity));

        float magnitude = (BLACK_HOLE_PULL_MAGNITUDE + (-1.0f * parentSpeed)) * perimeter * scale;
        if (hasShield) {
            magnitude *= DRONE_SHIELD_EXPLOSION_REDUCTION;
        }
        const b2Vec2 force = b2MulSV(magnitude, direction);

        if (entityTypeIsWall(ent->type)) {
            b2Body_ApplyForce(bodyID, force, output.pointB, true);
        } else {
            b2Body_ApplyForceToCenter(bodyID, force, true);
        }
    }
}

void projectilesStep(env *e) {
    CC_ArrayIter projIter;
    cc_array_iter_init(&projIter, e->projectiles);
    projectileEntity *projectile;
    while (cc_array_iter_next(&projIter, (void **)&projectile) != CC_ITER_END) {
        if (projectile->needsToBeDestroyed) {
            continue;
        }
        const float maxDistance = projectile->weaponInfo->maxDistance;
        const float distance = b2Distance(projectile->pos, projectile->lastPos);
        projectile->distance += distance;

        // if a drone is in a set mine's sensor range but behind a wall,
        // we need to check until the drone leaves the sensor range if
        // it's not behind the wall anymore as we normally only check if
        // we need to explode the mine when a drone touches the sensor
        if (projectile->numDronesBehindWalls != 0) {
            bool destroyed = false;
            for (uint8_t i = 0; i < projectile->numDronesBehindWalls; i++) {
                const uint8_t droneIdx = projectile->dronesBehindWalls[i];
                const droneEntity *drone = safe_array_get_at(e->drones, droneIdx);
                // TODO: do we need the closest point here?
                const b2DistanceOutput output = closestPoint(projectile->ent, drone->ent);
                const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
                if (posBehindWall(e, projectile->pos, output.pointB, NULL, filter, NULL)) {
                    continue;
                }

                // we have to destroy the projectile using the iterator so
                // we can continue to iterate correctly
                destroyProjectile(e, projectile, true, false);
                enum cc_stat res = cc_array_iter_remove_fast(&projIter, NULL);
                MAYBE_UNUSED(res);
                ASSERT(res == CC_OK);
                destroyed = true;
                break;
            }
            if (destroyed) {
                continue;
            }
        }

        if (projectile->entsInBlackHole != NULL) {
            handleBlackHolePull(e, projectile);
        }

        if (maxDistance == INFINITE) {
            continue;
        }
        if (projectile->distance >= maxDistance) {
            // we have to destroy the projectile using the iterator so
            // we can continue to iterate correctly
            destroyProjectile(e, projectile, true, false);
            enum cc_stat res = cc_array_iter_remove_fast(&projIter, NULL);
            MAYBE_UNUSED(res);
            ASSERT(res == CC_OK);
            continue;
        }
    }

    destroyExplodedProjectiles(e);
}

void weaponPickupsStep(env *e) {
    CC_ArrayIter iter;
    cc_array_iter_init(&iter, e->pickups);
    weaponPickupEntity *pickup;

    // respawn weapon pickups at a random location as a random weapon type
    // once the respawn wait has elapsed
    while (cc_array_iter_next(&iter, (void **)&pickup) != CC_ITER_END) {
        if (pickup->respawnWait == 0.0f) {
            continue;
        }
        pickup->respawnWait = max(pickup->respawnWait - e->deltaTime, 0.0f);
        if (pickup->respawnWait != 0.0f) {
            continue;
        }

        b2Vec2 pos;
        if (!findOpenPos(e, WEAPON_PICKUP_SHAPE, &pos, -1)) {
            const enum cc_stat res = cc_array_iter_remove_fast(&iter, NULL);
            MAYBE_UNUSED(res);
            ASSERT(res == CC_OK);
            DEBUG_LOG("destroying weapon pickup");
            destroyWeaponPickup(e, pickup);
            continue;
        }
        pickup->pos = pos;
        pickup->weapon = randWeaponPickupType(e);

        const int16_t cellIdx = entityPosToCellIdx(e, pos);
        if (cellIdx == -1) {
            ERRORF("invalid position for weapon pickup spawn: (%f, %f)", pos.x, pos.y);
        }
        DEBUG_LOGF("respawned weapon pickup at cell %d (%f, %f)", cellIdx, pos.x, pos.y);
        pickup->mapCellIdx = cellIdx;
        createWeaponPickupBodyShape(e, pickup);

        mapCell *cell = safe_array_get_at(e->cells, cellIdx);
        cell->ent = pickup->ent;
    }
}

// only update positions and velocities of dynamic bodies if they moved
// this step
void handleBodyMoveEvents(env *e) {
    b2BodyEvents events = b2World_GetBodyEvents(e->worldID);
    for (int i = 0; i < events.moveCount; i++) {
        const b2BodyMoveEvent *event = events.moveEvents + i;
        if (!b2Body_IsValid(event->bodyId)) {
            continue;
        }
        ASSERT(b2IsValidVec2(event->transform.p));
        const b2Vec2 newPos = event->transform.p;
        entity *ent = event->userData;
        if (ent == NULL) {
            continue;
        }

        wallEntity *wall;
        projectileEntity *proj;
        droneEntity *drone;
        shieldEntity *shield;
        dronePieceEntity *piece;
        int16_t mapIdx;

        // if the new position is out of bounds, destroy the entity unless
        // a drone is out of bounds, then just kill it
        switch (ent->type) {
        case STANDARD_WALL_ENTITY:
        case BOUNCY_WALL_ENTITY:
        case DEATH_WALL_ENTITY:
            wall = ent->entity;
            mapIdx = entityPosToCellIdx(e, newPos);
            if (mapIdx == -1) {
                DEBUG_LOGF("invalid position for floating wall: (%f, %f) destroying", newPos.x, newPos.y);
                cc_array_remove_fast(e->floatingWalls, wall, NULL);
                destroyWall(e, wall, false);
                continue;
            }
            wall->mapCellIdx = mapIdx;
            wall->pos = newPos;
            wall->rot = event->transform.q;
            wall->velocity = b2Body_GetLinearVelocity(wall->bodyID);
            break;
        case PROJECTILE_ENTITY:
            proj = ent->entity;
            mapIdx = entityPosToCellIdx(e, newPos);
            if (mapIdx == -1) {
                DEBUG_LOGF("invalid position for projectile: (%f, %f) destroying", newPos.x, newPos.y);
                destroyProjectile(e, proj, false, true);
                continue;
            }
            proj->mapCellIdx = mapIdx;
            proj->lastPos = proj->pos;
            proj->pos = newPos;
            proj->lastVelocity = proj->velocity;
            proj->velocity = b2Body_GetLinearVelocity(proj->bodyID);
            // if the projectile doesn't have damping its speed will
            // only change when colliding with a dynamic body or getting
            // hit by an explosion, and if it's currently colliding with
            // something we don't care about the current speed
            if (proj->weaponInfo->damping != 0.0f && proj->contacts == 0) {
                proj->lastSpeed = proj->speed;
                proj->speed = b2Length(proj->velocity);
            }

            if (e->client != NULL) {
                updateTrailPoints(&proj->trailPoints, MAX_PROJECTLE_TRAIL_POINTS, newPos);
            }
            break;
        case DRONE_ENTITY:
            drone = ent->entity;
            mapIdx = entityPosToCellIdx(e, newPos);
            if (mapIdx == -1) {
                DEBUG_LOGF("invalid position for drone: (%f, %f) killing it", newPos.x, newPos.y);
                killDrone(e, drone);
                continue;
            }
            drone->mapCellIdx = mapIdx;
            drone->lastPos = drone->pos;
            drone->pos = newPos;
            drone->lastVelocity = drone->velocity;
            drone->velocity = b2Body_GetLinearVelocity(drone->bodyID);

            if (e->client != NULL) {
                updateTrailPoints(&drone->trailPoints, MAX_DRONE_TRAIL_POINTS, newPos);
            }
            break;
        case SHIELD_ENTITY:
            shield = ent->entity;
            shield->pos = newPos;
            break;
        case DRONE_PIECE_ENTITY:
            piece = ent->entity;
            piece->pos = newPos;
            piece->rot = event->transform.q;
            break;
        default:
            ERRORF("unknown entity type for move event %d", ent->type);
        }
    }
}

// destroy the projectile if it has traveled enough or has bounced enough
// times, and update drone stats if a drone was hit
uint8_t handleProjectileBeginContact(env *e, const entity *proj, const entity *ent, const b2Manifold *manifold, const bool projIsShapeA) {
    projectileEntity *projectile = proj->entity;
    projectile->contacts++;

    // e (shape B in the collision) will be NULL if it's another
    // projectile that was just destroyed
    if (ent == NULL || ent->type == PROJECTILE_ENTITY) {
        // explode mines when hit by a projectile
        if (projectile->weaponInfo->type == MINE_LAUNCHER_WEAPON) {
            uint8_t numDestroyed = 1;
            if (ent != NULL) {
                const projectileEntity *projectile2 = ent->entity;
                // if both entities are mines both will be destroyed
                if (projectile2->weaponInfo->type == MINE_LAUNCHER_WEAPON) {
                    numDestroyed = 2;
                }
            }
            destroyProjectile(e, projectile, true, true);
            destroyExplodedProjectiles(e);
            return numDestroyed;
        }

        // always allow all other projectiles to bounce off each other
        return false;
    } else if (ent->type == BOUNCY_WALL_ENTITY) {
        // always allow projectiles to bounce off bouncy walls
        return false;
    } else if (ent->type == SHIELD_ENTITY) {
        // always allow projectiles to bounce off shields, and update shield health
        shieldEntity *shield = ent->entity;
        if (shield->health <= 0.0f) {
            return false;
        }

        const float damage = projectile->lastSpeed * projectile->weaponInfo->mass * DRONE_SHIELD_HEALTH_IMPULSE_COEF;
        DEBUG_LOGF("shield projectile damage: %f", damage);
        shield->health -= damage;
        if (shield->health <= 0.0f) {
            droneEntity *parentDrone = safe_array_get_at(e->drones, projectile->droneIdx);
            droneAddEnergy(parentDrone, DRONE_SHIELD_BREAK_ENERGY_REFILL);
        }

        return false;
    }

    if (projectile->weaponInfo->type != BLACK_HOLE_WEAPON || ent->type != DRONE_ENTITY) {
        projectile->bounces++;
    }
    if (ent->type == DRONE_ENTITY) {
        droneEntity *hitDrone = ent->entity;
        if (projectile->droneIdx != hitDrone->idx) {
            droneEntity *shooterDrone = safe_array_get_at(e->drones, projectile->droneIdx);

            if (shooterDrone->team != hitDrone->team) {
                const float impulseEnergy = projectile->lastSpeed * projectile->weaponInfo->mass * projectile->weaponInfo->energyRefillCoef;
                droneAddEnergy(shooterDrone, impulseEnergy);
            }
            // add 1 so we can differentiate between no weapon and weapon 0
            shooterDrone->stepInfo.shotHit[hitDrone->idx] = projectile->weaponInfo->type + 1;
            e->stats[shooterDrone->idx].shotsHit[projectile->weaponInfo->type]++;
            DEBUG_LOGF("drone %d hit drone %d with weapon %d", shooterDrone->idx, hitDrone->idx, projectile->weaponInfo->type);
            hitDrone->stepInfo.shotTaken[shooterDrone->idx] = projectile->weaponInfo->type + 1;
            e->stats[hitDrone->idx].shotsTaken[projectile->weaponInfo->type]++;
            DEBUG_LOGF("drone %d hit by drone %d with weapon %d", hitDrone->idx, shooterDrone->idx, projectile->weaponInfo->type);
        } else {
            hitDrone->stepInfo.ownShotTaken = true;
            e->stats[hitDrone->idx].ownShotsTaken[projectile->weaponInfo->type]++;
            DEBUG_LOGF("drone %d hit by own weapon %d", hitDrone->idx, projectile->weaponInfo->type);
        }

        if (projectile->weaponInfo->destroyedOnDroneHit) {
            destroyProjectile(e, projectile, projectile->weaponInfo->explodesOnDroneHit, true);
            destroyExplodedProjectiles(e);
            return 1;
        }
    } else if (projectile->weaponInfo->type == MINE_LAUNCHER_WEAPON && !projectile->setMine) {
        // if the mine is in explosion proximity of a drone now,
        // destroy it
        const b2QueryFilter filter = {
            .categoryBits = PROJECTILE_SHAPE,
            .maskBits = DRONE_SHAPE,
        };
        if (isOverlappingCircleInLineOfSight(e, projectile->ent, projectile->pos, MINE_LAUNCHER_PROXIMITY_RADIUS, filter, NULL)) {
            destroyProjectile(e, projectile, true, true);
            destroyExplodedProjectiles(e);
            return 1;
        }

        // create a weld joint to stick the mine to the wall
        ASSERT(entityTypeIsWall(ent->type));
        wallEntity *wall = ent->entity;
        ASSERT(manifold->pointCount == 1);

        b2WeldJointDef jointDef = b2DefaultWeldJointDef();
        const b2Rot projRot = b2Body_GetRotation(projectile->bodyID);
        if (projIsShapeA) {
            jointDef.bodyIdA = projectile->bodyID;
            jointDef.bodyIdB = wall->bodyID;
            jointDef.localAnchorA = b2InvRotateVector(projRot, manifold->points[0].anchorB);
            jointDef.localAnchorB = b2InvRotateVector(wall->rot, manifold->points[0].anchorA);
            jointDef.referenceAngle = b2RelativeAngle(wall->rot, projRot);
        } else {
            jointDef.bodyIdA = wall->bodyID;
            jointDef.bodyIdB = projectile->bodyID;
            jointDef.localAnchorA = b2InvRotateVector(wall->rot, manifold->points[0].anchorA);
            jointDef.localAnchorB = b2InvRotateVector(projRot, manifold->points[0].anchorB);
            jointDef.referenceAngle = b2RelativeAngle(projRot, wall->rot);
        }
        b2CreateWeldJoint(e->worldID, &jointDef);
        projectile->velocity = b2Vec2_zero;
        projectile->lastVelocity = b2Vec2_zero;
        projectile->speed = 0.0f;
        projectile->lastSpeed = 0.0f;
        projectile->setMine = true;
    }

    const uint8_t maxBounces = projectile->weaponInfo->maxBounces;
    if (projectile->bounces == maxBounces) {
        destroyProjectile(e, projectile, true, true);
        destroyExplodedProjectiles(e);
        return 1;
    }

    return 0;
}

// ensure speed is maintained when a projectile hits a dynamic body
void handleProjectileEndContact(const entity *proj, const entity *ent) {
    projectileEntity *projectile = proj->entity;
    projectile->contacts--;

    // mines stick to walls, explode when hitting another projectile
    // and are destroyed when hitting a drone so no matter what we don't
    // need to do anything here
    if (projectile->weaponInfo->type == MINE_LAUNCHER_WEAPON) {
        return;
    }

    if (ent != NULL) {
        if (ent->type == PROJECTILE_ENTITY) {
            const projectileEntity *projectile2 = ent->entity;
            // allow projectile speeds to increase when two different
            // projectile types collide
            if (projectile->weaponInfo->type != projectile2->weaponInfo->type) {
                b2Vec2 newVel = b2Body_GetLinearVelocity(projectile->bodyID);
                float newSpeed = b2Length(newVel);
                if (newSpeed < projectile->lastSpeed) {
                    newSpeed = projectile->lastSpeed;
                    newVel = b2MulSV(newSpeed, b2Normalize(newVel));
                    b2Body_SetLinearVelocity(projectile->bodyID, newVel);
                }
                projectile->velocity = newVel;
                projectile->speed = newSpeed;
                projectile->lastSpeed = newSpeed;
                return;
            }
        }
    }

    // the last speed is used here instead of the current speed because
    // the current speed will be the speed box2d set the projectile to
    // after a collision and we want to keep the speed consistent
    float newSpeed = projectile->lastSpeed;
    if (projectile->weaponInfo->type == ACCELERATOR_WEAPON) {
        newSpeed = min(projectile->lastSpeed * ACCELERATOR_BOUNCE_SPEED_COEF, ACCELERATOR_MAX_SPEED);
        projectile->speed = newSpeed;
    }

    // ensure the projectile's speed doesn't change after bouncing off
    // something
    b2Vec2 newVel = b2Body_GetLinearVelocity(projectile->bodyID);
    newVel = b2MulSV(newSpeed, b2Normalize(newVel));
    b2Body_SetLinearVelocity(projectile->bodyID, newVel);
    projectile->velocity = newVel;
    projectile->speed = newSpeed;
    projectile->lastSpeed = newSpeed;
}

// TODO: drone on drone collisions should reduce shield health
void handleContactEvents(env *e) {
    b2ContactEvents events = b2World_GetContactEvents(e->worldID);
    for (int i = 0; i < events.beginCount; ++i) {
        const b2ContactBeginTouchEvent *event = events.beginEvents + i;
        entity *e1 = NULL;
        entity *e2 = NULL;

        if (b2Shape_IsValid(event->shapeIdA)) {
            e1 = b2Shape_GetUserData(event->shapeIdA);
            ASSERT(e1 != NULL);
        }
        if (b2Shape_IsValid(event->shapeIdB)) {
            e2 = b2Shape_GetUserData(event->shapeIdB);
            ASSERT(e2 != NULL);
        }

        if (e1 != NULL) {
            if (e1->type == PROJECTILE_ENTITY) {
                uint8_t numDestroyed = handleProjectileBeginContact(e, e1, e2, &event->manifold, true);
                if (numDestroyed == 2) {
                    continue;
                } else if (numDestroyed == 1) {
                    e1 = NULL;
                }

            } else if (e1->type == DEATH_WALL_ENTITY && e2 != NULL) {
                if (e2->type == DRONE_ENTITY) {
                    droneEntity *drone = e2->entity;
                    killDrone(e, drone);
                } else if (e2->type == SHIELD_ENTITY) {
                    shieldEntity *shield = e2->entity;
                    shield->health = 0.0f;
                    destroyDroneShield(e, shield, true);
                    e2 = NULL;
                }
            }
        }
        if (e2 != NULL) {
            if (e2->type == PROJECTILE_ENTITY) {
                handleProjectileBeginContact(e, e2, e1, &event->manifold, false);
            } else if (e2->type == DEATH_WALL_ENTITY && e1 != NULL) {
                if (e1->type == DRONE_ENTITY) {
                    droneEntity *drone = e1->entity;
                    killDrone(e, drone);
                } else if (e1->type == SHIELD_ENTITY) {
                    shieldEntity *shield = e1->entity;
                    shield->health = 0.0f;
                    destroyDroneShield(e, shield, true);
                }
            }
        }
    }

    for (int i = 0; i < events.endCount; ++i) {
        const b2ContactEndTouchEvent *event = events.endEvents + i;
        entity *e1 = NULL;
        entity *e2 = NULL;
        if (b2Shape_IsValid(event->shapeIdA)) {
            e1 = b2Shape_GetUserData(event->shapeIdA);
            ASSERT(e1 != NULL);
        }
        if (b2Shape_IsValid(event->shapeIdB)) {
            e2 = b2Shape_GetUserData(event->shapeIdB);
            ASSERT(e2 != NULL);
        }
        if (e1 != NULL && e1->type == PROJECTILE_ENTITY) {
            handleProjectileEndContact(e1, e2);
        }
        if (e2 != NULL && e2->type == PROJECTILE_ENTITY) {
            handleProjectileEndContact(e2, e1);
        }
    }
}

// set pickup to respawn somewhere else randomly if a drone touched it,
// mark the pickup as disabled if a floating wall is touching it
void handleWeaponPickupBeginTouch(env *e, const entity *sensor, entity *visitor) {
    weaponPickupEntity *pickup = sensor->entity;
    if (pickup->floatingWallsTouching != 0) {
        return;
    }

    wallEntity *wall;

    switch (visitor->type) {
    case DRONE_ENTITY:
        disableWeaponPickup(e, pickup);

        droneEntity *drone = visitor->entity;
        drone->stepInfo.pickedUpWeapon = true;
        drone->stepInfo.prevWeapon = drone->weaponInfo->type;
        droneChangeWeapon(e, drone, pickup->weapon);

        e->stats[drone->idx].weaponsPickedUp[pickup->weapon]++;
        DEBUG_LOGF("drone %d picked up weapon %d", drone->idx, pickup->weapon);
        break;
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY:
        wall = visitor->entity;
        if (!wall->isFloating) {
            if (!wall->isSuddenDeath) {
                ERRORF("non sudden death wall type %d at cell %d touched weapon pickup", visitor->type, wall->mapCellIdx);
            }
            return;
        }

        pickup->floatingWallsTouching++;
        break;
    default:
        ERRORF("invalid weapon pickup begin touch visitor %d", visitor->type);
    }
}

// explode proximity detonating projectiles
void handleProjectileBeginTouch(env *e, const entity *sensor, entity *visitor) {
    projectileEntity *projectile = sensor->entity;

    switch (projectile->weaponInfo->type) {
    case FLAK_CANNON_WEAPON:
        if (projectile->distance < FLAK_CANNON_SAFE_DISTANCE) {
            return;
        }
        destroyProjectile(e, projectile, true, true);
        destroyExplodedProjectiles(e);
        break;
    case MINE_LAUNCHER_WEAPON:
        if (!projectile->setMine) {
            return;
        }

        ASSERT(visitor->type == DRONE_ENTITY);
        const b2DistanceOutput output = closestPoint(sensor, visitor);
        const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
        if (posBehindWall(e, projectile->pos, output.pointB, NULL, filter, NULL)) {
            const droneEntity *drone = visitor->entity;
            projectile->dronesBehindWalls[projectile->numDronesBehindWalls++] = drone->idx;
            return;
        }

        destroyProjectile(e, projectile, true, true);
        destroyExplodedProjectiles(e);
        break;
    case BLACK_HOLE_WEAPON:
        if (visitor->type == DRONE_ENTITY) {
            const droneEntity *drone = visitor->entity;
            if (projectile->droneIdx == drone->idx && projectile->distance < BLACK_HOLE_PARENT_IGNORE_DISTANCE) {
                return;
            }
        }

        // copy the entity ID so it won't be changed if the entity is
        // destroyed and reused later
        const entityID *visitorID = visitor->id;
        entityID *id = fastCalloc(1, sizeof(entityID));
        id->id = visitorID->id;
        id->generation = visitorID->generation;
        cc_array_add(projectile->entsInBlackHole, id);
        break;
    default:
        ERRORF("invalid projectile type %d for begin touch event", sensor->type);
    }
}

// mark the pickup as enabled if no floating walls are touching it
void handleWeaponPickupEndTouch(const entity *sensor, entity *visitor) {
    weaponPickupEntity *pickup = sensor->entity;
    if (pickup->respawnWait != 0.0f) {
        return;
    }

    wallEntity *wall;

    switch (visitor->type) {
    case DRONE_ENTITY:
        break;
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY:
        wall = visitor->entity;
        if (!wall->isFloating) {
            return;
        }

        pickup->floatingWallsTouching--;
        break;
    default:
        ERRORF("invalid weapon pickup end touch visitor %d", visitor->type);
    }
}

void handleProjectileEndTouch(const entity *sensor, entity *visitor) {
    projectileEntity *projectile = sensor->entity;

    switch (projectile->weaponInfo->type) {
    case FLAK_CANNON_WEAPON:
        break;
    case MINE_LAUNCHER_WEAPON:
        if (projectile->numDronesBehindWalls == 0) {
            return;
        }
        projectile->numDronesBehindWalls--;
        break;
    case BLACK_HOLE_WEAPON:
        if (visitor == NULL) {
            return;
        }

        const entityID *visitorID = visitor->id;
        for (uint8_t i = 0; i < cc_array_size(projectile->entsInBlackHole); ++i) {
            entityID *id = safe_array_get_at(projectile->entsInBlackHole, i);
            if (id->id == visitorID->id) {
                fastFree(id);
                cc_array_remove_fast_at(projectile->entsInBlackHole, i, NULL);
                return;
            }
        }
        break;
    default:
        ERRORF("invalid projectile type %d for end touch event", projectile->weaponInfo->type);
    }
}

void handleSensorEvents(env *e) {
    b2SensorEvents events = b2World_GetSensorEvents(e->worldID);
    for (int i = 0; i < events.beginCount; ++i) {
        const b2SensorBeginTouchEvent *event = events.beginEvents + i;
        if (!b2Shape_IsValid(event->sensorShapeId)) {
            DEBUG_LOG("could not find sensor shape for begin touch event");
            continue;
        }
        entity *s = b2Shape_GetUserData(event->sensorShapeId);
        ASSERT(s != NULL);

        if (!b2Shape_IsValid(event->visitorShapeId)) {
            DEBUG_LOG("could not find visitor shape for begin touch event");
            continue;
        }
        entity *v = b2Shape_GetUserData(event->visitorShapeId);
        ASSERT(v != NULL);

        switch (s->type) {
        case WEAPON_PICKUP_ENTITY:
            handleWeaponPickupBeginTouch(e, s, v);
            break;
        case PROJECTILE_ENTITY:
            handleProjectileBeginTouch(e, s, v);
            break;
        default:
            ERRORF("unknown entity type %d for sensor begin touch event", s->type);
        }
    }

    for (int i = 0; i < events.endCount; ++i) {
        const b2SensorEndTouchEvent *event = events.endEvents + i;
        if (!b2Shape_IsValid(event->sensorShapeId)) {
            DEBUG_LOG("could not find sensor shape for end touch event");
            continue;
        }
        entity *s = b2Shape_GetUserData(event->sensorShapeId);
        ASSERT(s != NULL);
        entity *v = NULL;
        if (b2Shape_IsValid(event->visitorShapeId)) {
            v = b2Shape_GetUserData(event->visitorShapeId);
            ASSERT(v != NULL);
        }

        if (s->type == PROJECTILE_ENTITY) {
            handleProjectileEndTouch(s, v);
            continue;
        }

        if (v != NULL) {
            handleWeaponPickupEndTouch(s, v);
        }
    }
}

void findNearWalls(const env *e, const droneEntity *drone, nearEntity nearestWalls[], const uint8_t nWalls) {
    nearEntity nearWalls[MAX_NEAREST_WALLS];

    for (uint8_t i = 0; i < MAX_NEAREST_WALLS; ++i) {
        const uint32_t idx = (MAX_NEAREST_WALLS * drone->mapCellIdx) + i;
        const uint16_t wallIdx = e->map->nearestWalls[idx].idx;
        wallEntity *wall = safe_array_get_at(e->walls, wallIdx);
        nearWalls[i].entity = wall;
        nearWalls[i].distanceSquared = b2DistanceSquared(drone->pos, wall->pos);
    }
    insertionSort(nearWalls, MAX_NEAREST_WALLS);
    memcpy(nearestWalls, nearWalls, nWalls * sizeof(nearEntity));
}

#endif
