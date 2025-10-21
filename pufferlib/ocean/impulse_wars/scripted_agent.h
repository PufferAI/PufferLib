#ifndef IMPULSE_WARS_SCRIPTED_BOT_H
#define IMPULSE_WARS_SCRIPTED_BOT_H

#include "game.h"
#include "types.h"

const uint8_t NUM_NEAR_WALLS = 3;
const uint8_t NUM_NEAR_PICKUPS = 1;

const float WALL_CHECK_DISTANCE_SQUARED = SQUARED(6.0f);
const float WALL_AVOID_DISTANCE = 4.0f;
const float WALL_DANGER_DISTANCE = 3.0f;
const float WALL_BURST_CHECK_DISTANCE = 5.0f;
const float WALL_BURST_CHECK_SPEED = 20.0f;
const float BURST_MIN_RADIUS_SQUARED = SQUARED(DRONE_BURST_RADIUS_MIN);
const float STABILIZE_MOVE_SPEED = 5.0f;
const float SHOTGUN_DANGER_DISTANCE = 4.0f;

void addDebugPoint(iwEnv *e, b2Vec2 pos, float size, Color color) {
#ifndef NDEBUG
    debugPoint *point = fastCalloc(1, sizeof(debugPoint));
    point->pos = pos;
    point->size = size;
    point->color = color;
    cc_array_add(e->debugPoints, point);
#else
    MAYBE_UNUSED(e);
    MAYBE_UNUSED(pos);
    MAYBE_UNUSED(size);
    MAYBE_UNUSED(color);
#endif
}

typedef struct castCircleCtx {
    bool hit;
    b2ShapeId shapeID;
    b2Vec2 point;
} castCircleCtx;

float castCircleCallback(b2ShapeId shapeId, b2Vec2 point, b2Vec2 normal, float fraction, void *context) {
    // these parameters are required by the callback signature
    MAYBE_UNUSED(point);
    MAYBE_UNUSED(normal);
    if (!b2Shape_IsValid(shapeId) || (fraction == 0.0f && b2VecEqual(normal, b2Vec2_zero))) {
        // skip this shape if it isn't valid or this is an initial overlap
        return -1.0f;
    }

    castCircleCtx *ctx = context;
    ctx->hit = true;
    ctx->shapeID = shapeId;
    ctx->point = point;

    return 0.0f;
}

static inline uint32_t pathOffset(const iwEnv *e, uint16_t srcCellIdx, uint16_t destCellIdx) {
    const uint8_t srcCol = srcCellIdx % e->map->columns;
    const uint8_t srcRow = srcCellIdx / e->map->columns;
    const uint8_t destCol = destCellIdx % e->map->columns;
    const uint8_t destRow = destCellIdx / e->map->columns;
    return (destRow * e->map->rows * e->map->columns * e->map->rows) + (destCol * e->map->rows * e->map->columns) + (srcRow * e->map->columns) + srcCol;
}

void pathfindBFS(const iwEnv *e, uint8_t *flatPaths, uint16_t destCellIdx) {
    uint8_t (*paths)[e->map->columns] = (uint8_t (*)[e->map->columns])flatPaths;
    int8_t (*buffer)[3] = (int8_t (*)[3])e->mapPathing[e->mapIdx].pathBuffer;

    uint16_t start = 0;
    uint16_t end = 1;

    const mapCell *cell = safe_array_get_at(e->cells, destCellIdx);
    if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
        return;
    }
    const int8_t destCol = destCellIdx % e->map->columns;
    const int8_t destRow = destCellIdx / e->map->columns;

    buffer[start][0] = 8;
    buffer[start][1] = destCol;
    buffer[start][2] = destRow;
    while (start < end) {
        const int8_t direction = buffer[start][0];
        const int8_t startCol = buffer[start][1];
        const int8_t startRow = buffer[start][2];
        start++;

        if (startCol < 0 || startCol >= e->map->columns || startRow < 0 || startRow >= e->map->rows || paths[startRow][startCol] != UINT8_MAX) {
            continue;
        }
        int16_t cellIdx = cellIndex(e, startCol, startRow);
        const mapCell *cell = safe_array_get_at(e->cells, cellIdx);
        if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
            paths[startRow][startCol] = 8;
            continue;
        }

        paths[startRow][startCol] = direction;

        buffer[end][0] = 6; // up
        buffer[end][1] = startCol;
        buffer[end][2] = startRow + 1;
        end++;

        buffer[end][0] = 2; // down
        buffer[end][1] = startCol;
        buffer[end][2] = startRow - 1;
        end++;

        buffer[end][0] = 0; // right
        buffer[end][1] = startCol - 1;
        buffer[end][2] = startRow;
        end++;

        buffer[end][0] = 4; // left
        buffer[end][1] = startCol + 1;
        buffer[end][2] = startRow;
        end++;

        buffer[end][0] = 5; // up left
        buffer[end][1] = startCol + 1;
        buffer[end][2] = startRow + 1;
        end++;

        buffer[end][0] = 3; // down left
        buffer[end][1] = startCol + 1;
        buffer[end][2] = startRow - 1;
        end++;

        buffer[end][0] = 1; // down right
        buffer[end][1] = startCol - 1;
        buffer[end][2] = startRow - 1;
        end++;

        buffer[end][0] = 7; // up right
        buffer[end][1] = startCol - 1;
        buffer[end][2] = startRow + 1;
        end++;
    }
}

float distanceWithDamping(const iwEnv *e, const float speed, const float linearDamping, const float steps) {
    const float damping = 1.0f + (linearDamping * e->deltaTime);
    return speed * (damping / linearDamping) * (1.0f - powf(1.0f / damping, steps));
}

b2Vec2 positionWithDamping(const iwEnv *e, const droneEntity *drone, const b2Vec2 impulse, const float linearDamping, const float steps) {
    const b2Vec2 newVel = b2Add(drone->velocity, impulse);
    const float speed = b2Length(newVel);
    const float distance = distanceWithDamping(e, speed, linearDamping, steps);
    const b2Vec2 direction = b2Normalize(newVel);
    return b2MulAdd(drone->pos, distance, direction);
}

// returns true if drone can fire in the given direction without hitting
// or getting too close to a death wall before it can fire again
bool safeToFire(iwEnv *e, const droneEntity *drone, const b2Vec2 direction) {
    // don't shoot shotgun point blank at walls, the shots will immediately
    // bounce back and send the drone flying uncontrollably
    if (drone->weaponInfo->type == SHOTGUN_WEAPON) {
        const b2Vec2 rayEnd = b2MulAdd(drone->pos, SHOTGUN_DANGER_DISTANCE, direction);
        const b2Vec2 translation = b2Sub(rayEnd, drone->pos);
        const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
        const b2RayResult rayRes = b2World_CastRayClosest(e->worldID, drone->pos, translation, filter);
        if (rayRes.hit) {
            return false;
        }
    }

    float shotWait;
    if (drone->ammo > 1) {
        shotWait = ((drone->weaponInfo->coolDown + drone->weaponInfo->charge) / e->deltaTime) * 1.5f;
    } else {
        shotWait = ((e->defaultWeapon->coolDown + e->defaultWeapon->charge) / e->deltaTime) * 1.5f;
    }
    const float recoilSpeed = -drone->weaponInfo->recoilMagnitude * DRONE_INV_MASS;
    const b2Vec2 recoil = b2MulSV(recoilSpeed, direction);
    const b2Vec2 recoilPos = positionWithDamping(e, drone, recoil, DRONE_LINEAR_DAMPING, shotWait);

    addDebugPoint(e, b2MulAdd(drone->pos, 2.0f * DRONE_RADIUS, direction), 0.5f, WHITE);

    const b2Vec2 pos = drone->pos;
    const b2Vec2 rayEnd = recoilPos;
    const b2Vec2 translation = b2Sub(rayEnd, pos);
    float radius = DRONE_RADIUS;
    if (drone->shield != NULL) {
        radius = DRONE_SHIELD_RADIUS;
    }
    const b2ShapeProxy cirProxy = b2MakeProxy(&pos, 1, radius);
    const b2QueryFilter filter = {.categoryBits = DRONE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | DRONE_SHAPE};

    castCircleCtx ctx = {0};
    b2World_CastShape(e->worldID, &cirProxy, translation, filter, castCircleCallback, &ctx);
    if (!ctx.hit) {
        addDebugPoint(e, recoilPos, 0.5f, LIME);
        return true;
    } else {
        const entity *ent = b2Shape_GetUserData(ctx.shapeID);
        if (entityTypeIsWall(ent->type) && ent->type != DEATH_WALL_ENTITY) {
            addDebugPoint(e, recoilPos, 0.5f, LIME);
            return true;
        }
    }

    addDebugPoint(e, recoilPos, 0.5f, MAROON);
    return false;
}

bool weaponSafeForMovement(const droneEntity *drone) {
    switch (drone->weaponInfo->type) {
    case IMPLODER_WEAPON:
    case MINE_LAUNCHER_WEAPON:
    case BLACK_HOLE_WEAPON:
    case NUKE_WEAPON:
        return false;
    default:
        return true;
    }
}

void scriptedAgentShoot(const droneEntity *drone, agentActions *actions) {
    actions->shoot = true;
    if (drone->chargingWeapon && drone->weaponCharge == drone->weaponInfo->charge) {
        actions->chargingWeapon = false;
    }
}

void moveTo(iwEnv *e, const droneEntity *drone, agentActions *actions, const b2Vec2 dstPos) {
    ASSERT(drone->mapCellIdx != -1);
    int16_t dstIdx = entityPosToCellIdx(e, dstPos);
    if (dstIdx == -1) {
        return;
    }

    uint32_t pathIdx = pathOffset(e, drone->mapCellIdx, dstIdx);
    uint8_t *paths = e->mapPathing[e->mapIdx].paths;
    uint8_t direction = paths[pathIdx];
    if (direction == UINT8_MAX) {
        uint32_t bfsIdx = pathOffset(e, 0, dstIdx);
        pathfindBFS(e, &paths[bfsIdx], dstIdx);
        direction = paths[pathIdx];
    }
    if (direction >= 8) {
        return;
    }
    actions->move.x += discMoveToContMoveMap[0][direction];
    actions->move.y += discMoveToContMoveMap[1][direction];
    actions->move = b2Normalize(actions->move);

    const b2Vec2 invDirection = b2MulSV(-1.0f, actions->move);
    if (!weaponSafeForMovement(drone) || !safeToFire(e, drone, invDirection)) {
        return;
    }
    actions->aim = invDirection;
    scriptedAgentShoot(drone, actions);
}

float weaponIdealRangeSquared(const droneEntity *drone) {
    switch (drone->weaponInfo->type) {
    case STANDARD_WEAPON:
        return SQUARED(20.0f);
    case MACHINEGUN_WEAPON:
        return SQUARED(30.0f);
    case SHOTGUN_WEAPON:
        return SQUARED(20.0f);
    case IMPLODER_WEAPON:
        return SQUARED(30.0f);
    case FLAK_CANNON_WEAPON:
        return SQUARED(FLAK_CANNON_SAFE_DISTANCE + 5.0f);
    case SNIPER_WEAPON:
    case ACCELERATOR_WEAPON:
    case MINE_LAUNCHER_WEAPON:
    case BLACK_HOLE_WEAPON:
    case NUKE_WEAPON:
        return FLT_MAX;

    default:
        ERRORF("unknown weapon type %d", drone->weaponInfo->type);
    }
}

bool shouldShootAtEnemy(iwEnv *e, const droneEntity *drone, const droneEntity *enemyDrone, const b2Vec2 enemyDroneDirection) {
    // don't shoot at shielded enemies with railguns since it will likely
    // bounce back and hit us
    if (drone->weaponInfo->type == SNIPER_WEAPON && enemyDrone->shield != NULL) {
        return false;
    }
    if (!safeToFire(e, drone, enemyDroneDirection)) {
        return false;
    }

    // cast a circle that's the size of a projectile of the current weapon
    const float enemyDroneDistance = b2Distance(enemyDrone->pos, drone->pos);
    const b2Vec2 castEnd = b2MulAdd(drone->pos, enemyDroneDistance, enemyDroneDirection);
    const b2Vec2 translation = b2Sub(castEnd, drone->pos);
    const b2ShapeProxy cirProxy = b2MakeProxy(&drone->pos, 1, drone->weaponInfo->radius);
    const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | DRONE_SHAPE};

    castCircleCtx ctx = {0};
    b2World_CastShape(e->worldID, &cirProxy, translation, filter, castCircleCallback, &ctx);
    if (!ctx.hit) {
        return false;
    }
    ASSERT(b2Shape_IsValid(ctx.shapeID));
    const entity *ent = b2Shape_GetUserData(ctx.shapeID);
    if (ent == NULL || ent->type != DRONE_ENTITY) {
        return false;
    }
    const droneEntity *hitDrone = ent->entity;
    if (hitDrone->idx != enemyDrone->idx) {
        return false;
    }

    return true;
}

b2Vec2 predictiveAim(const droneEntity *drone, const droneEntity *enemyDrone, const float distanceSquared) {
    const float timeToImpact = sqrtf(distanceSquared) / drone->weaponInfo->initialSpeed;
    const b2Vec2 predictedPos = b2MulAdd(enemyDrone->pos, timeToImpact, enemyDrone->velocity);
    return b2Normalize(b2Sub(predictedPos, drone->pos));
}

void scriptedAgentBurst(const droneEntity *drone, agentActions *actions) {
    if (drone->chargingBurst) {
        return;
    } else {
        actions->chargingBurst = true;
    }
}

void handleWallProximity(iwEnv *e, const droneEntity *drone, const wallEntity *wall, const float distance, agentActions *actions) {
    // shoot to move away faster from a death wall if we're too close and it's safe
    const b2Vec2 wallDirection = b2Normalize(b2Sub(wall->pos, drone->pos));
    if (distance <= WALL_DANGER_DISTANCE && weaponSafeForMovement(drone) && safeToFire(e, drone, wallDirection)) {
        actions->aim = b2MulAdd(actions->aim, distance, wallDirection);
        scriptedAgentShoot(drone, actions);
    }
    // move away from the wall if we're too close
    if (distance <= WALL_AVOID_DISTANCE) {
        actions->move = b2MulAdd(actions->move, -distance, wallDirection);
    }
}

// charge burst until we're close enough to a death wall to burst off
// of it
void wallBurst(iwEnv *e, const droneEntity *drone, const float distance, agentActions *actions) {
    if (distance < DRONE_BURST_RADIUS_MIN) {
        scriptedAgentBurst(drone, actions);
        return;
    }

    float damping = DRONE_LINEAR_DAMPING;
    if (drone->braking || actions->brake) {
        damping *= DRONE_BRAKE_DAMPING_COEF;
    }
    const float travelDistance = distanceWithDamping(e, b2Length(drone->velocity), damping, e->frameSkip);
    if (travelDistance > distance) {
        scriptedAgentBurst(drone, actions);
    } else {
        actions->chargingBurst = true;
    }
}

agentActions scriptedAgentActions(iwEnv *e, droneEntity *drone) {
    agentActions actions = {0};
    if (e->sittingDuck) {
        return actions;
    }

    // keep the weapon charged and ready
    if (drone->weaponInfo->charge != 0.0f) {
        actions.chargingWeapon = true;
        actions.shoot = true;
    }

    // find the N nearest death walls or floating walls
    nearEntity nearWalls[MAX_NEAREST_WALLS] = {0};
    findNearWalls(e, drone, nearWalls, NUM_NEAR_WALLS);

    // move away from and shoot at death walls if we're too close
    float closestWallDistance = FLT_MAX;
    for (uint8_t i = 0; i < NUM_NEAR_WALLS; i++) {
        const wallEntity *wall = nearWalls[i].entity;
        if (wall->type != DEATH_WALL_ENTITY) {
            continue;
        }

        const b2DistanceOutput output = closestPoint(drone->ent, wall->ent);
        closestWallDistance = min(closestWallDistance, output.distance);
        handleWallProximity(e, drone, wall, output.distance, &actions);
    }

    for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
        wallEntity *floatingWall = safe_array_get_at(e->floatingWalls, i);
        if (floatingWall->type != DEATH_WALL_ENTITY) {
            continue;
        }
        if (b2DistanceSquared(floatingWall->pos, drone->pos) > WALL_CHECK_DISTANCE_SQUARED) {
            continue;
        }

        const b2DistanceOutput output = closestPoint(drone->ent, floatingWall->ent);
        closestWallDistance = min(closestWallDistance, output.distance);
        handleWallProximity(e, drone, floatingWall, output.distance, &actions);
    }

    // TODO: shoot mines around enemy, only if not too close

    // if we are moving towards a death wall at a high speed, do everything
    // possible to avoid hitting it
    const float droneSpeed = b2Length(drone->velocity);
    if (drone->braking || drone->chargingBurst || closestWallDistance <= WALL_BURST_CHECK_DISTANCE || droneSpeed >= WALL_BURST_CHECK_SPEED) {
        float damping = DRONE_LINEAR_DAMPING;
        if (drone->braking || actions.brake) {
            damping *= DRONE_BRAKE_DAMPING_COEF;
        }
        const b2Vec2 recoilPos = positionWithDamping(e, drone, b2Vec2_zero, damping, 0.5f / e->deltaTime);
        const b2Vec2 pos = drone->pos;
        const b2Vec2 rayEnd = recoilPos;
        const b2Vec2 translation = b2Sub(rayEnd, pos);
        float radius = DRONE_RADIUS;
        if (drone->shield != NULL) {
            radius = DRONE_SHIELD_RADIUS;
        }
        const b2ShapeProxy cirProxy = b2MakeProxy(&pos, 1, radius);
        const b2QueryFilter filter = {.categoryBits = DRONE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};

        castCircleCtx ctx = {0};
        b2World_CastShape(e->worldID, &cirProxy, translation, filter, castCircleCallback, &ctx);
        if (ctx.hit) {
            const entity *ent = b2Shape_GetUserData(ctx.shapeID);
            if (entityTypeIsWall(ent->type) && ent->type == DEATH_WALL_ENTITY) {
                actions.brake = true;
                if (drone->shield == NULL) {
                    wallBurst(e, drone, b2Distance(drone->pos, ctx.point), &actions);
                }

                const b2Vec2 droneDirection = b2Normalize(drone->velocity);
                if (b2VecEqual(actions.move, b2Vec2_zero)) {
                    actions.move = b2MulSV(-1.0f, droneDirection);
                }
                if (b2VecEqual(actions.aim, b2Vec2_zero) && weaponSafeForMovement(drone)) {
                    actions.aim = droneDirection;
                    scriptedAgentShoot(drone, &actions);
                }
                return actions;
            }
        }
    }

    // if we're close enough to a wall to need to shoot at it, don't
    // worry about enemies
    if (!b2VecEqual(actions.aim, b2Vec2_zero)) {
        return actions;
    }

    // get a weapon if the standard weapon is active
    if (drone->weaponInfo->type == STANDARD_WEAPON && cc_array_size(e->pickups) != 0) {
        nearEntity nearPickups[MAX_WEAPON_PICKUPS] = {0};
        uint8_t numActivePickups = 0;
        for (uint8_t i = 0; i < cc_array_size(e->pickups); i++) {
            weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
            if (pickup->floatingWallsTouching > 0) {
                continue;
            }
            const nearEntity nearEnt = {
                .entity = pickup,
                .distanceSquared = b2DistanceSquared(pickup->pos, drone->pos),
            };
            nearPickups[numActivePickups++] = nearEnt;
        }
        if (numActivePickups > 0) {
            insertionSort(nearPickups, numActivePickups);
            const weaponPickupEntity *pickup = nearPickups[0].entity;
            moveTo(e, drone, &actions, pickup->pos);
            return actions;
        }
    }

    // find closest enemy drone
    droneEntity *enemyDrone = NULL;
    float closestDistanceSquared = FLT_MAX;
    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        if (i == drone->idx) {
            continue;
        }
        droneEntity *otherDrone = safe_array_get_at(e->drones, i);
        if (otherDrone->dead || otherDrone->team == drone->team) {
            continue;
        }
        const float distanceSquared = b2DistanceSquared(otherDrone->pos, drone->pos);
        if (distanceSquared < closestDistanceSquared) {
            closestDistanceSquared = distanceSquared;
            enemyDrone = otherDrone;
        }
    }
    if (enemyDrone == NULL) {
        // fight recoil if we're not otherwise moving
        if (b2VecEqual(actions.move, b2Vec2_zero) && droneSpeed >= STABILIZE_MOVE_SPEED) {
            actions.move = b2MulSV(-1.0f, b2Normalize(drone->velocity));
        }
        return actions;
    }

    // burst if the enemy drone is very close
    if (closestDistanceSquared <= BURST_MIN_RADIUS_SQUARED) {
        scriptedAgentBurst(drone, &actions);
    }
    // move into ideal range for the current weapon
    if (closestDistanceSquared > weaponIdealRangeSquared(drone)) {
        moveTo(e, drone, &actions, enemyDrone->pos);
        return actions;
    }

    // shoot at enemy drone if it's in line of sight and safe, otherwise move towards it
    const b2Vec2 enemyDroneDirection = b2Normalize(b2Sub(enemyDrone->pos, drone->pos));
    if (shouldShootAtEnemy(e, drone, enemyDrone, enemyDroneDirection)) {
        if (drone->weaponCooldown == 0.0f && drone->weaponCharge >= drone->weaponInfo->charge - e->deltaTime) {
            actions.move.x += enemyDroneDirection.x;
            actions.move.y += enemyDroneDirection.y;
            actions.move = b2Normalize(actions.move);
        }
        actions.aim = predictiveAim(drone, enemyDrone, closestDistanceSquared);
        scriptedAgentShoot(drone, &actions);
    } else {
        moveTo(e, drone, &actions, enemyDrone->pos);
    }

    // fight recoil if we're not otherwise moving
    if (b2VecEqual(actions.move, b2Vec2_zero) && droneSpeed >= STABILIZE_MOVE_SPEED) {
        actions.move = b2MulSV(-1.0f, b2Normalize(drone->velocity));
    }

    return actions;
}

#endif
