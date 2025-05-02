#ifndef IMPULSE_WARS_TYPES_H
#define IMPULSE_WARS_TYPES_H

#include "box2d/box2d.h"

// autopxd2 can't parse internal box2d headers or raylib headers
#ifndef AUTOPXD
#include "id_pool.h"
#include "raylib.h"
#include "rlights.h"
#else
typedef struct Vector2 {
    float x;
    float y;
} Vector2;

typedef struct dummyStruct {
    void *dummy;
} dummyStruct;

typedef struct dummyStruct b2IdPool, Camera3D, Camera2D, Shader, Texture2D, RenderTexture2D;
#endif

#include "include/cc_array.h"

#include "settings.h"

#define _MAX_DRONES 4

const uint8_t NUM_WALL_TYPES = 3;

#define MAX_TRAIL_POINTS 20
#define MAX_DRONE_TRAIL_POINTS 20
#define MAX_PROJECTLE_TRAIL_POINTS 10

enum entityType {
    STANDARD_WALL_ENTITY,
    BOUNCY_WALL_ENTITY,
    DEATH_WALL_ENTITY,
    WEAPON_PICKUP_ENTITY,
    PROJECTILE_ENTITY,
    DRONE_ENTITY,
    SHIELD_ENTITY,
    DRONE_PIECE_ENTITY,
};

// the category bit that will be set on each entity's shape; this is
// used to control what entities can collide with each other
enum shapeCategory {
    WALL_SHAPE = 1,
    FLOATING_WALL_SHAPE = 2,
    PROJECTILE_SHAPE = 4,
    WEAPON_PICKUP_SHAPE = 8,
    DRONE_SHAPE = 16,
    SHIELD_SHAPE = 32,
    DRONE_PIECE_SHAPE = 64,
};

typedef struct entityID {
    int32_t id;
    uint16_t generation;
} entityID;

// general purpose entity object
typedef struct entity {
    entityID *id;
    uint32_t generation;
    enum entityType type;
    void *entity;
} entity;

#define _NUM_WEAPONS 10
const uint8_t NUM_WEAPONS = _NUM_WEAPONS;

enum weaponType {
    STANDARD_WEAPON,
    MACHINEGUN_WEAPON,
    SNIPER_WEAPON,
    SHOTGUN_WEAPON,
    IMPLODER_WEAPON,
    ACCELERATOR_WEAPON,
    FLAK_CANNON_WEAPON,
    MINE_LAUNCHER_WEAPON,
    BLACK_HOLE_WEAPON,
    NUKE_WEAPON,
};

typedef struct mapBounds {
    b2Vec2 min;
    b2Vec2 max;
} mapBounds;

// used for N near entities observations
typedef struct nearEntity {
    uint16_t idx;
    void *entity;
    float distanceSquared;
} nearEntity;

typedef struct mapEntry {
    const char *layout;
    const uint8_t columns;
    const uint8_t rows;
    const uint8_t randFloatingStandardWalls;
    const uint8_t randFloatingBouncyWalls;
    const uint8_t randFloatingDeathWalls;
    // are there any floating walls that have consistent starting positions
    const bool hasSetFloatingWalls;
    const uint16_t weaponPickups;
    const enum weaponType defaultWeapon;

    mapBounds bounds;
    mapBounds spawnQuads[4];
    bool *droneSpawns;
    uint8_t *packedLayout;
    nearEntity *nearestWalls;
} mapEntry;

// a cell in the map; ent will be NULL if the cell is empty
typedef struct mapCell {
    entity *ent;
    b2Vec2 pos;
} mapCell;

typedef struct wallEntity {
    b2BodyId bodyID;
    b2ShapeId shapeID;
    b2Vec2 pos;
    b2Rot rot;
    b2Vec2 velocity;
    b2Vec2 extent;
    int16_t mapCellIdx;
    bool isFloating;
    enum entityType type;
    bool isSuddenDeath;

    entity *ent;
} wallEntity;

typedef struct weaponInformation {
    const enum weaponType type;
    // should the body be treated as a bullet by box2d; if so CCD
    // (continuous collision detection) will be enabled to prevent
    // tunneling through static bodies which is expensive so it's
    // only enabled for fast moving projectiles
    const bool isPhysicsBullet;
    // can the projectile ever be stationary? if so, it should be
    // allowed to sleep to save on physics updates
    const bool canSleep;
    const uint8_t numProjectiles;
    const float fireMagnitude;
    const float recoilMagnitude;
    const float damping;
    const float charge;
    const float coolDown;
    const float maxDistance;
    const float radius;
    const float density;
    const float mass;
    const float invMass;
    const float initialSpeed;
    const uint8_t maxBounces;
    const bool explosive;
    const bool destroyedOnDroneHit;
    const bool explodesOnDroneHit;
    const bool hasSensor;
    const float energyRefillCoef;
    const float spawnWeight;
} weaponInformation;

typedef struct weaponPickupEntity {
    b2BodyId bodyID;
    b2ShapeId shapeID;
    enum weaponType weapon;
    float respawnWait;
    // how many floating walls are touching this pickup
    uint8_t floatingWallsTouching;
    b2Vec2 pos;
    int16_t mapCellIdx;

    entity *ent;
    bool bodyDestroyed;
} weaponPickupEntity;

typedef struct droneEntity droneEntity;

typedef struct trailPoints {
    Vector2 points[MAX_TRAIL_POINTS];
    uint8_t length;
} trailPoints;

typedef struct projectileEntity {
    uint8_t droneIdx;

    b2BodyId bodyID;
    b2ShapeId shapeID;
    // used for proximity explosive projectiles
    b2ShapeId sensorID;
    weaponInformation *weaponInfo;
    b2Vec2 pos;
    int16_t mapCellIdx;
    b2Vec2 lastPos;
    b2Vec2 velocity;
    b2Vec2 lastVelocity;
    float speed;
    float lastSpeed;
    float distance;
    uint8_t bounces;
    uint8_t contacts;
    bool setMine;
    uint8_t numDronesBehindWalls;
    uint8_t dronesBehindWalls[_MAX_DRONES];
    CC_Array *entsInBlackHole;
    bool needsToBeDestroyed;

    entity *ent;

    // for rendering
    trailPoints trailPoints;
} projectileEntity;

// used to keep track of what happened each step for reward purposes
typedef struct droneStepInfo {
    bool firedShot;
    bool pickedUpWeapon;
    enum weaponType prevWeapon;
    uint8_t shotHit[_MAX_DRONES];
    bool explosionHit[_MAX_DRONES];
    uint8_t shotTaken[_MAX_DRONES];
    bool explosionTaken[_MAX_DRONES];
    bool ownShotTaken;
} droneStepInfo;

typedef struct shieldEntity {
    droneEntity *drone;

    b2BodyId bodyID;
    b2ShapeId shapeID;
    b2ShapeId bufferShapeID;
    b2Vec2 pos;
    float health;
    float duration;

    entity *ent;
} shieldEntity;

typedef struct dronePieceEntity {
    uint8_t droneIdx;

    b2BodyId bodyID;
    b2ShapeId shapeID;
    b2Vec2 pos;
    b2Rot rot;
    b2Vec2 vertices[3];
    bool isShieldPiece;

    entity *ent;

    uint16_t lifetime;
} dronePieceEntity;

typedef struct droneEntity {
    b2BodyId bodyID;
    b2ShapeId shapeID;
    weaponInformation *weaponInfo;
    int8_t ammo;
    float weaponCooldown;
    uint16_t heat;
    bool chargingWeapon;
    float weaponCharge;
    float energyLeft;
    bool braking;
    bool chargingBurst;
    float burstCharge;
    float burstCooldown;
    bool energyFullyDepleted;
    bool energyFullyDepletedThisStep;
    float energyRefillWait;
    bool shotThisStep;
    bool diedThisStep;

    uint8_t idx;
    uint8_t team;
    b2Vec2 initalPos;
    b2Vec2 pos;
    int16_t mapCellIdx;
    b2Vec2 lastPos;
    b2Vec2 lastMove;
    b2Vec2 lastAim;
    b2Vec2 velocity;
    b2Vec2 lastVelocity;
    droneStepInfo stepInfo;
    float respawnWait;
    uint8_t livesLeft;
    bool dead;

    shieldEntity *shield;
    entity *ent;

    // for rendering
    trailPoints trailPoints;
    uint16_t respawnGuideLifetime;
} droneEntity;

// stats for the whole episode
typedef struct droneStats {
    float reward;
    float distanceTraveled;
    float absDistanceTraveled;
    float shotsFired[_NUM_WEAPONS];
    float shotsHit[_NUM_WEAPONS];
    float shotsTaken[_NUM_WEAPONS];
    float ownShotsTaken[_NUM_WEAPONS];
    float weaponsPickedUp[_NUM_WEAPONS];
    float shotDistances[_NUM_WEAPONS];
    float brakeTime;
    float totalBursts;
    float burstsHit;
    float energyEmptied;
    float wins;
} droneStats;

typedef struct logEntry {
    float length;
    float ties;
    droneStats stats[_MAX_DRONES];
} logEntry;

typedef struct logBuffer {
    logEntry *logs;
    uint16_t size;
    uint16_t capacity;
} logBuffer;

typedef struct gameCamera {
    Camera3D camera3D;
    Camera2D camera2D;
    Vector2 targetPos;
    float maxZoom;
    bool orthographic;
} gameCamera;

typedef struct rayClient {
    float scale;
    uint16_t width;
    uint16_t height;
    uint16_t halfWidth;
    uint16_t halfHeight;

    gameCamera *camera;

    Shader blurShader;
    int32_t blurShaderDirLoc;
    Shader bloomShader;
    int32_t bloomIntensityLoc;
    int32_t bloomTexColorLoc;
    int32_t bloomTexBloomBlurLoc;
    Shader gridShader;
    int32_t gridShaderPosLoc[4];
    int32_t gridShaderColorLoc[4];
    Texture2D wallTexture;
    RenderTexture2D blurSrcTexture;
    RenderTexture2D blurDstTexture;
    RenderTexture2D projRawTex;
    RenderTexture2D projBloomTex;
    RenderTexture2D droneRawTex;
    RenderTexture2D droneBloomTex;
} rayClient;

typedef struct brakeTrailPoint {
    b2Vec2 pos;
    uint16_t lifetime;
} brakeTrailPoint;

typedef struct explosionInfo {
    b2ExplosionDef def;
    bool isBurst;
    uint8_t droneIdx;
    uint16_t renderSteps;
} explosionInfo;

typedef struct agentActions {
    b2Vec2 move;
    b2Vec2 aim;
    bool chargingWeapon;
    bool shoot;
    bool brake;
    bool chargingBurst;
    bool discardWeapon;
} agentActions;

typedef struct pathingInfo {
    uint8_t *paths;
    int8_t *pathBuffer;
} pathingInfo;

typedef struct env {
    uint8_t numDrones;
    uint8_t numAgents;
    uint8_t numTeams;
    bool teamsEnabled;
    bool sittingDuck;
    bool isTraining;

    uint16_t obsBytes;
    uint16_t discreteObsBytes;

    uint8_t *obs;
    float *rewards;
    bool discretizeActions;
    float *contActions;
    int32_t *discActions;
    uint8_t *masks;
    uint8_t *terminals;
    uint8_t *truncations;

    uint8_t frameRate;
    float deltaTime;
    uint8_t frameSkip;
    uint8_t box2dSubSteps;
    uint64_t randState;
    bool needsReset;

    uint16_t episodeLength;
    logBuffer *logs;
    droneStats stats[_MAX_DRONES];

    b2WorldId worldID;
    int8_t pinnedMapIdx;
    int8_t mapIdx;
    mapEntry *map;
    int8_t lastSpawnQuad;
    uint8_t spawnedWeaponPickups[_NUM_WEAPONS];
    weaponInformation *defaultWeapon;
    b2IdPool idPool;
    CC_Array *entities;
    CC_Array *cells;
    CC_Array *walls;
    CC_Array *floatingWalls;
    CC_Array *drones;
    CC_Array *pickups;
    CC_Array *projectiles;
    CC_Array *explodingProjectiles;
    CC_Array *dronePieces;

    pathingInfo *mapPathing;

    uint16_t totalSteps;
    uint16_t totalSuddenDeathSteps;
    // steps left until sudden death
    uint16_t stepsLeft;
    // steps left until the next set of sudden death walls are spawned
    uint16_t suddenDeathSteps;
    // the amount of sudden death walls that have been spawned
    uint8_t suddenDeathWallCounter;
    bool suddenDeathWallsPlaced;

    bool humanInput;
    uint8_t humanDroneInput;
    uint8_t connectedControllers;

    rayClient *client;
    float renderScale;
    CC_Array *brakeTrailPoints;
    // used for rendering explosions
    CC_Array *explosions;
    b2Vec2 debugPoint;
} env;

#endif
