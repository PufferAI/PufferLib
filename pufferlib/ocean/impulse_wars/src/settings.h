#ifndef IMPULSE_WARS_SETTINGS_H
#define IMPULSE_WARS_SETTINGS_H

#include "helpers.h"
#include "types.h"

#define INFINITE -1
const uint8_t TWO_BIT_MASK = 0x3;

// general settings
const uint8_t TRAINING_ACTIONS_PER_SECOND = 10;
const uint8_t TRAINING_FRAME_RATE = 30;
const uint8_t TRAINING_BOX2D_SUBSTEPS = 2;

const uint8_t EVAL_FRAME_RATE = 120;
const uint8_t EVAL_BOX2D_SUBSTEPS = 4;

const uint8_t NUM_MAPS = 9;
#define _MAX_MAP_COLUMNS 25
#define _MAX_MAP_ROWS 25
#define MAX_CELLS _MAX_MAP_COLUMNS *_MAX_MAP_ROWS + 1
#define MAX_FLOATING_WALLS 18
#define MAX_WEAPON_PICKUPS 12

#define MAX_NEAREST_WALLS 8

const uint8_t DRONE_LIVES = 1;
const float DRONE_RESPAWN_WAIT = 2.0f;
const uint8_t ROUND_STEPS = 90;
const uint8_t SUDDEN_DEATH_STEPS = 5;

const uint8_t MAX_DRONES = _MAX_DRONES;

const uint16_t LOG_BUFFER_SIZE = 1024;

// reward settings
const float WIN_REWARD = 1.5f;
const float ENEMY_DEATH_REWARD = 1.5f;
const float TEAMMATE_DEATH_PUNISHMENT = -1.0f;
const float DEATH_PUNISHMENT = 0.0f;
const float ENERGY_EMPTY_PUNISHMENT = -0.75f;
const float WEAPON_PICKUP_REWARD = 0.5f;
const float SHOT_HIT_REWARD_COEF = 0.000013333f;
const float EXPLOSION_HIT_REWARD_COEF = 5.0f;
const float APPROACH_REWARD = 0.0f;

const uint8_t DISTANCE_CUTOFF = 15.0f;

// observation constants

// map layout observations
const uint8_t MAP_OBS_ROWS = 11;
const uint8_t MAP_OBS_COLUMNS = 11;
const uint16_t MAP_OBS_SIZE = MAP_OBS_ROWS * MAP_OBS_COLUMNS;

// discrete observations
#define _NUM_NEAR_WALL_OBS 4
const uint8_t NUM_NEAR_WALL_OBS = _NUM_NEAR_WALL_OBS;
const uint16_t NEAR_WALL_TYPES_OBS_OFFSET = MAP_OBS_SIZE;

#define _NUM_FLOATING_WALL_OBS 4
const uint8_t NUM_FLOATING_WALL_OBS = _NUM_FLOATING_WALL_OBS;
const uint16_t FLOATING_WALL_TYPES_OBS_OFFSET = NEAR_WALL_TYPES_OBS_OFFSET + NUM_NEAR_WALL_OBS;

const uint8_t NUM_PROJECTILE_OBS = 30;
const uint16_t PROJECTILE_DRONE_OBS_OFFSET = FLOATING_WALL_TYPES_OBS_OFFSET + NUM_FLOATING_WALL_OBS;
const uint16_t PROJECTILE_WEAPONS_OBS_OFFSET = PROJECTILE_DRONE_OBS_OFFSET + NUM_PROJECTILE_OBS;

#define _NUM_WEAPON_PICKUP_OBS 3
const uint8_t NUM_WEAPON_PICKUP_OBS = _NUM_WEAPON_PICKUP_OBS;
const uint16_t WEAPON_PICKUP_WEAPONS_OBS_OFFSET = PROJECTILE_WEAPONS_OBS_OFFSET + NUM_PROJECTILE_OBS;

const uint16_t ENEMY_DRONE_WEAPONS_OBS_OFFSET = WEAPON_PICKUP_WEAPONS_OBS_OFFSET + NUM_WEAPON_PICKUP_OBS;

// continuous observations
const uint8_t NEAR_WALL_POS_OBS_SIZE = 2;
const uint8_t NEAR_WALL_OBS_SIZE = NUM_NEAR_WALL_OBS * NEAR_WALL_POS_OBS_SIZE;
const uint16_t NEAR_WALL_POS_OBS_OFFSET = 0;

const uint8_t FLOATING_WALL_INFO_OBS_SIZE = 5;
const uint8_t FLOATING_WALL_OBS_SIZE = NUM_FLOATING_WALL_OBS * FLOATING_WALL_INFO_OBS_SIZE;
const uint16_t FLOATING_WALL_INFO_OBS_OFFSET = NEAR_WALL_POS_OBS_OFFSET + NEAR_WALL_OBS_SIZE;

const uint8_t WEAPON_PICKUP_POS_OBS_SIZE = 2;
const uint8_t WEAPON_PICKUP_OBS_SIZE = NUM_WEAPON_PICKUP_OBS * WEAPON_PICKUP_POS_OBS_SIZE;
const uint16_t WEAPON_PICKUP_POS_OBS_OFFSET = FLOATING_WALL_INFO_OBS_OFFSET + FLOATING_WALL_OBS_SIZE;

const uint8_t PROJECTILE_INFO_OBS_SIZE = 4;
const uint8_t PROJECTILE_OBS_SIZE = NUM_PROJECTILE_OBS * PROJECTILE_INFO_OBS_SIZE;
const uint16_t PROJECTILE_INFO_OBS_OFFSET = WEAPON_PICKUP_POS_OBS_OFFSET + WEAPON_PICKUP_OBS_SIZE;

const uint16_t ENEMY_DRONE_OBS_OFFSET = PROJECTILE_INFO_OBS_OFFSET + PROJECTILE_OBS_SIZE;
const uint8_t ENEMY_DRONE_OBS_SIZE = 24;

const uint8_t DRONE_OBS_SIZE = 22;

const uint8_t MISC_OBS_SIZE = 1;

const uint16_t _DISCRETE_OBS_SIZE = MAP_OBS_SIZE + NUM_NEAR_WALL_OBS + NUM_FLOATING_WALL_OBS + (NUM_PROJECTILE_OBS * 2) + NUM_WEAPON_PICKUP_OBS + 1;
const uint16_t _CONTINUOUS_OBS_SIZE = NEAR_WALL_OBS_SIZE + FLOATING_WALL_OBS_SIZE + WEAPON_PICKUP_OBS_SIZE + PROJECTILE_OBS_SIZE + DRONE_OBS_SIZE + MISC_OBS_SIZE;

uint16_t discreteObsSize(uint8_t numDrones) {
    return _DISCRETE_OBS_SIZE + ((numDrones - 1));
}

uint16_t continuousObsSize(uint8_t numDrones) {
    return _CONTINUOUS_OBS_SIZE + ((numDrones - 1) * ENEMY_DRONE_OBS_SIZE);
}

uint16_t obsBytes(uint8_t numDrones) {
    return alignedSize((discreteObsSize(numDrones) * sizeof(uint8_t)) + (continuousObsSize(numDrones) * sizeof(float)), sizeof(float));
}

const float MAX_X_POS = 150.0f;
const float MAX_Y_POS = 150.0f;
const float MAX_DISTANCE = 200.0f;
const float MAX_SPEED = 500.0f;
const float MAX_ACCEL = 1000.0f;
const float MAX_ANGLE = PI;

// action constants
const uint8_t CONTINUOUS_ACTION_SIZE = 7;
const uint8_t DISCRETE_ACTION_SIZE = 5;
const float ACTION_NOOP_MAGNITUDE = 0.1f;

const float discMoveToContMoveMap[2][8] = {
    {1.0f, 0.707107f, 0.0f, -0.707107f, -1.0f, -0.707107f, 0.0f, 0.707107f},
    {0.0f, 0.707107f, 1.0f, 0.707107f, 0.0f, -0.707107f, -1.0f, -0.707107f},
};
const float discAimToContAimMap[2][16] = {
    {1.0f, 0.92388f, 0.707107f, 0.382683f, 0.0f, -0.382683f, -0.707107f, -0.92388f, -1.0f, -0.92388f, -0.707107f, -0.382683f, 0.0f, 0.382683f, 0.707107f, 0.92388f},
    {0.0f, 0.382683f, 0.707107f, 0.92388f, 1.0f, 0.92388f, 0.707107f, 0.382683f, 0.0f, -0.382683f, -0.707107f, -0.92388f, -1.0f, -0.92388f, -0.707107f, -0.382683f},
};

const float MIN_SPAWN_DISTANCE = 6.0f;

// wall settings
const float WALL_THICKNESS = 4.0f;
const float FLOATING_WALL_THICKNESS = 3.0f;
const float FLOATING_WALL_DAMPING = 0.75f;
const float STANDARD_WALL_RESTITUTION = 0.01f;
const float STANDARD_WALL_FRICTION = 0.3f;
const float BOUNCY_WALL_RESTITUTION = 1.0f;
const float WALL_DENSITY = 4.0f;

// weapon pickup settings
const float PICKUP_THICKNESS = 3.0f;
const float PICKUP_SPAWN_DISTANCE_SQUARED = SQUARED(10.0f);
const float PICKUP_RESPAWN_WAIT = 3.0f;
const float SUDDEN_DEATH_PICKUP_RESPAWN_WAIT = 2.0f;

// drone settings
const float DRONE_WALL_SPAWN_DISTANCE = 2.0f;
const float DRONE_DEATH_WALL_SPAWN_DISTANCE = 7.5f;
const float DRONE_DRONE_SPAWN_DISTANCE_SQUARED = SQUARED(10.0f);

#define DRONE_RADIUS 1.0f
#define DRONE_DENSITY 1.25f
const float DRONE_RESTITUTION = 0.3f;
const float DRONE_FRICTION = 0.1f;
#define DRONE_INV_MASS INV_MASS(MASS(DRONE_DENSITY, DRONE_RADIUS))
const float DRONE_MOVE_MAGNITUDE = 35.0f;
const float DRONE_LINEAR_DAMPING = 1.0f;
const float DRONE_MOVE_AIM_COEF = 0.1f;

const float DRONE_ENERGY_MAX = 1.0f;
const float DRONE_BRAKE_DAMPING_COEF = 2.5f;
const float DRONE_BRAKE_DRAIN_RATE = 0.5f;
const float DRONE_ENERGY_REFILL_WAIT = 1.0f;
const float DRONE_ENERGY_REFILL_EMPTY_WAIT = 3.0f;
const float DRONE_ENERGY_REFILL_RATE = 0.03f;
const float DRONE_ENERGY_RESPAWN_REFILL = 0.5f;

const float DRONE_BURST_BASE_COST = 0.1f;
const float DRONE_BURST_CHARGE_RATE = 0.6f;
const float DRONE_BURST_RADIUS_BASE = 5.0f;
const float DRONE_BURST_RADIUS_MIN = 3.5f;
const float DRONE_BURST_IMPACT_BASE = 140.0f;
const float DRONE_BURST_IMPACT_MIN = 35.0f;
const float DRONE_BURST_COOLDOWN = 0.5f;

const float DRONE_SHIELD_RADIUS = DRONE_RADIUS * 1.5f;
const float DRONE_SHIELD_HEALTH = 100.0f;
const float DRONE_SHIELD_START_DURATION = 1.5f;
const float DRONE_SHIELD_RESPAWN_DURATION = 3.0f;
const float DRONE_SHIELD_EXPLOSION_REDUCTION = 0.5f;
const float DRONE_SHIELD_HEALTH_IMPULSE_COEF = 0.5f;
const float DRONE_SHIELD_HEALTH_EXPLOSION_COEF = 0.8;
const float DRONE_SHIELD_BREAK_ENERGY_COST = -0.33f;
const float DRONE_SHIELD_BREAK_ENERGY_REFILL = 0.25f;

#define PROJECTILE_ENERGY_REFILL_COEF 0.001f
const float EXPLOSION_ENERGY_REFILL_COEF = 1.75f;
const float WEAPON_DISCARD_COST = 0.2f;

const uint8_t DRONE_PIECE_COUNT = 5;
const float DRONE_PIECE_MIN_DISTANCE = 1.5f;
const float DRONE_PIECE_MAX_DISTANCE = 3.0f;
const float DRONE_PIECE_LINEAR_DAMPING = 1.0f;
const float DRONE_PIECE_ANGULAR_DAMPING = 1.0f;
const float DRONE_PIECE_MIN_SPEED = 5.0f;
const float DRONE_PIECE_MAX_SPEED = 10.0f;

// TODO: increase impulse of imploder of entity it directly hits?
// TODO: long charge single shot slow big explosion weapon?
// weapon projectile settings
#define STANDARD_AMMO INFINITE
#define STANDARD_PROJECTILES 1
#define STANDARD_RECOIL_MAGNITUDE 20.0f
#define STANDARD_FIRE_MAGNITUDE 17.0f
#define STANDARD_DAMPING 0.0f
#define STANDARD_CHARGE 0.0f
#define STANDARD_COOL_DOWN 0.37f
#define STANDARD_MAX_DISTANCE 80.0f
#define STANDARD_RADIUS 0.2
#define STANDARD_DENSITY 3.25f
#define STANDARD_MASS MASS(STANDARD_DENSITY, STANDARD_RADIUS)
#define STANDARD_INV_MASS INV_MASS(STANDARD_MASS)
#define STANDARD_BOUNCE 2
#define STANDARD_SPAWN_WEIGHT 0

#define MACHINEGUN_AMMO 50
#define MACHINEGUN_PROJECTILES 1
#define MACHINEGUN_RECOIL_MAGNITUDE 12.8f
#define MACHINEGUN_FIRE_MAGNITUDE 30.0f
#define MACHINEGUN_DAMPING 0.1f
#define MACHINEGUN_CHARGE 0.0f
#define MACHINEGUN_COOL_DOWN 0.07f
#define MACHINEGUN_MAX_DISTANCE 225.0f
#define MACHINEGUN_RADIUS 0.15f
#define MACHINEGUN_DENSITY 3.2f
#define MACHINEGUN_MASS MASS(MACHINEGUN_DENSITY, MACHINEGUN_RADIUS)
#define MACHINEGUN_INV_MASS INV_MASS(MACHINEGUN_MASS)
#define MACHINEGUN_BOUNCE 1
#define MACHINEGUN_ENERGY_REFILL_COEF 0.2f
#define MACHINEGUN_SPAWN_WEIGHT 3.0f

#define SNIPER_AMMO 3
#define SNIPER_PROJECTILES 1
#define SNIPER_RECOIL_MAGNITUDE 96.0f
#define SNIPER_FIRE_MAGNITUDE 300.0f
#define SNIPER_DAMPING 0.05f
#define SNIPER_CHARGE 1.0f
#define SNIPER_COOL_DOWN 1.5f
#define SNIPER_MAX_DISTANCE INFINITE
#define SNIPER_RADIUS 0.5f
#define SNIPER_DENSITY 2.0f
#define SNIPER_MASS MASS(SNIPER_DENSITY, SNIPER_RADIUS)
#define SNIPER_INV_MASS INV_MASS(SNIPER_MASS)
#define SNIPER_BOUNCE 0
#define SNIPER_ENERGY_REFILL_COEF 1.2f
#define SNIPER_SPAWN_WEIGHT 3.0f

#define SHOTGUN_AMMO 8
#define SHOTGUN_PROJECTILES 8
#define SHOTGUN_RECOIL_MAGNITUDE 100.0f
#define SHOTGUN_FIRE_MAGNITUDE 25.0f
#define SHOTGUN_DAMPING 0.3f
#define SHOTGUN_CHARGE 0.0f
#define SHOTGUN_COOL_DOWN 0.8f
#define SHOTGUN_MAX_DISTANCE 100.0f
#define SHOTGUN_RADIUS 0.15f
#define SHOTGUN_DENSITY 2.5f
#define SHOTGUN_MASS MASS(SHOTGUN_DENSITY, SHOTGUN_RADIUS)
#define SHOTGUN_INV_MASS INV_MASS(SHOTGUN_MASS)
#define SHOTGUN_BOUNCE 1
#define SHOTGUN_ENERGY_REFILL_COEF 0.5f
#define SHOTGUN_SPAWN_WEIGHT 3.0f

#define IMPLODER_AMMO 1
#define IMPLODER_PROJECTILES 1
#define IMPLODER_RECOIL_MAGNITUDE 65.0f
#define IMPLODER_FIRE_MAGNITUDE 60.0f
#define IMPLODER_DAMPING 0.0f
#define IMPLODER_CHARGE 2.0f
#define IMPLODER_COOL_DOWN 1.5f
#define IMPLODER_MAX_DISTANCE INFINITE
#define IMPLODER_RADIUS 0.8f
#define IMPLODER_DENSITY 1.0f
#define IMPLODER_MASS MASS(IMPLODER_DENSITY, IMPLODER_RADIUS)
#define IMPLODER_INV_MASS INV_MASS(IMPLODER_MASS)
#define IMPLODER_BOUNCE 0
#define IMPLODER_SPAWN_WEIGHT 1.0f

#define ACCELERATOR_AMMO 1
#define ACCELERATOR_PROJECTILES 1
#define ACCELERATOR_RECOIL_MAGNITUDE 100.0f
#define ACCELERATOR_FIRE_MAGNITUDE 35.0f
#define ACCELERATOR_DAMPING 0.0f
#define ACCELERATOR_CHARGE 0.0f
#define ACCELERATOR_COOL_DOWN 2.0f
#define ACCELERATOR_MAX_DISTANCE INFINITE
#define ACCELERATOR_RADIUS 0.5f
#define ACCELERATOR_DENSITY 2.0f
#define ACCELERATOR_MASS MASS(ACCELERATOR_DENSITY, ACCELERATOR_RADIUS)
#define ACCELERATOR_INV_MASS INV_MASS(ACCELERATOR_MASS)
#define ACCELERATOR_BOUNCE 100
#define ACCELERATOR_BOUNCE_SPEED_COEF 1.07f
#define ACCELERATOR_MAX_SPEED 500.f
#define ACCELERATOR_SPAWN_WEIGHT 1.0f

#define FLAK_CANNON_AMMO 12
#define FLAK_CANNON_PROJECTILES 1
#define FLAK_CANNON_RECOIL_MAGNITUDE 30.0f
#define FLAK_CANNON_FIRE_MAGNITUDE 14.0f
#define FLAK_CANNON_DAMPING 0.15f
#define FLAK_CANNON_CHARGE 0.0f
#define FLAK_CANNON_COOL_DOWN 0.4f
#define FLAK_CANNON_MAX_DISTANCE 100.0f
#define FLAK_CANNON_RADIUS 0.3f
#define FLAK_CANNON_DENSITY 1.0f
#define FLAK_CANNON_MASS MASS(FLAK_CANNON_DENSITY, FLAK_CANNON_RADIUS)
#define FLAK_CANNON_INV_MASS INV_MASS(FLAK_CANNON_MASS)
#define FLAK_CANNON_BOUNCE INFINITE
#define FLAK_CANNON_SAFE_DISTANCE 25.0f
#define FLAK_CANNON_PROXIMITY_RADIUS 2.0f
#define FLAK_CANNON_SPAWN_WEIGHT 2.0f

#define MINE_LAUNCHER_AMMO 3
#define MINE_LAUNCHER_PROJECTILES 1
#define MINE_LAUNCHER_RECOIL_MAGNITUDE 20.0f
#define MINE_LAUNCHER_FIRE_MAGNITUDE 25.0f
#define MINE_LAUNCHER_DAMPING 0.2f
#define MINE_LAUNCHER_CHARGE 0.0f
#define MINE_LAUNCHER_COOL_DOWN 0.6f
#define MINE_LAUNCHER_MAX_DISTANCE INFINITE
#define MINE_LAUNCHER_RADIUS 0.5f
#define MINE_LAUNCHER_DENSITY 0.5f
#define MINE_LAUNCHER_MASS MASS(MINE_LAUNCHER_DENSITY, MINE_LAUNCHER_RADIUS)
#define MINE_LAUNCHER_INV_MASS INV_MASS(MINE_LAUNCHER_MASS)
#define MINE_LAUNCHER_BOUNCE INFINITE // this is to avoid mines sometimes exploding when hitting walls
#define MINE_LAUNCHER_SPAWN_WEIGHT 2.0f
#define MINE_LAUNCHER_PROXIMITY_RADIUS 7.5f

#define BLACK_HOLE_AMMO 3
#define BLACK_HOLE_PROJECTILES 1
#define BLACK_HOLE_RECOIL_MAGNITUDE 75.0f
#define BLACK_HOLE_FIRE_MAGNITUDE 125.0f
#define BLACK_HOLE_DAMPING 0.0f
#define BLACK_HOLE_CHARGE 0.75f
#define BLACK_HOLE_COOL_DOWN 1.0f
#define BLACK_HOLE_MAX_DISTANCE INFINITE
#define BLACK_HOLE_RADIUS 0.5f
#define BLACK_HOLE_DENSITY 10.0f
#define BLACK_HOLE_MASS MASS(BLACK_HOLE_DENSITY, BLACK_HOLE_RADIUS)
#define BLACK_HOLE_INV_MASS INV_MASS(BLACK_HOLE_MASS)
#define BLACK_HOLE_BOUNCE 0
#define BLACK_HOLE_SPAWN_WEIGHT 2.0f
#define BLACK_HOLE_PROXIMITY_RADIUS 10.0f
#define BLACK_HOLE_PARENT_IGNORE_DISTANCE BLACK_HOLE_PROXIMITY_RADIUS * 1.5f
#define BLACK_HOLE_PULL_MAGNITUDE -300.0f

#define NUKE_AMMO 1
#define NUKE_PROJECTILES 1
#define NUKE_RECOIL_MAGNITUDE 150.0f
#define NUKE_FIRE_MAGNITUDE 75.0f
#define NUKE_DAMPING 0.0f
#define NUKE_CHARGE 5.0f
#define NUKE_COOL_DOWN 3.0f
#define NUKE_MAX_DISTANCE INFINITE
#define NUKE_RADIUS 0.8f
#define NUKE_DENSITY 3.0f
#define NUKE_MASS MASS(NUKE_DENSITY, NUKE_RADIUS)
#define NUKE_INV_MASS INV_MASS(NUKE_MASS)
#define NUKE_BOUNCE 0
#define NUKE_SPAWN_WEIGHT 0.5f

const weaponInformation standard = {
    .type = STANDARD_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = STANDARD_PROJECTILES,
    .fireMagnitude = STANDARD_FIRE_MAGNITUDE,
    .recoilMagnitude = STANDARD_RECOIL_MAGNITUDE,
    .damping = STANDARD_DAMPING,
    .charge = STANDARD_CHARGE,
    .coolDown = STANDARD_COOL_DOWN,
    .maxDistance = STANDARD_MAX_DISTANCE,
    .radius = STANDARD_RADIUS,
    .density = STANDARD_DENSITY,
    .mass = STANDARD_MASS,
    .invMass = STANDARD_INV_MASS,
    .initialSpeed = STANDARD_FIRE_MAGNITUDE * STANDARD_INV_MASS,
    .maxBounces = STANDARD_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .spawnWeight = STANDARD_SPAWN_WEIGHT,
};

const weaponInformation machineGun = {
    .type = MACHINEGUN_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = MACHINEGUN_PROJECTILES,
    .fireMagnitude = MACHINEGUN_FIRE_MAGNITUDE,
    .recoilMagnitude = MACHINEGUN_RECOIL_MAGNITUDE,
    .damping = MACHINEGUN_DAMPING,
    .charge = MACHINEGUN_CHARGE,
    .coolDown = MACHINEGUN_COOL_DOWN,
    .maxDistance = MACHINEGUN_MAX_DISTANCE,
    .radius = MACHINEGUN_RADIUS,
    .density = MACHINEGUN_DENSITY,
    .mass = MACHINEGUN_MASS,
    .invMass = MACHINEGUN_INV_MASS,
    .initialSpeed = MACHINEGUN_FIRE_MAGNITUDE * MACHINEGUN_INV_MASS,
    .maxBounces = MACHINEGUN_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF * MACHINEGUN_ENERGY_REFILL_COEF,
    .spawnWeight = MACHINEGUN_SPAWN_WEIGHT,
};

const weaponInformation sniper = {
    .type = SNIPER_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = SNIPER_PROJECTILES,
    .fireMagnitude = SNIPER_FIRE_MAGNITUDE,
    .recoilMagnitude = SNIPER_RECOIL_MAGNITUDE,
    .damping = SNIPER_DAMPING,
    .charge = SNIPER_CHARGE,
    .coolDown = SNIPER_COOL_DOWN,
    .maxDistance = SNIPER_MAX_DISTANCE,
    .radius = SNIPER_RADIUS,
    .density = SNIPER_DENSITY,
    .mass = SNIPER_MASS,
    .invMass = SNIPER_INV_MASS,
    .initialSpeed = SNIPER_FIRE_MAGNITUDE * SNIPER_INV_MASS,
    .maxBounces = SNIPER_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = true,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF * SNIPER_ENERGY_REFILL_COEF,
    .spawnWeight = SNIPER_SPAWN_WEIGHT,
};

const weaponInformation shotgun = {
    .type = SHOTGUN_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = SHOTGUN_PROJECTILES,
    .fireMagnitude = SHOTGUN_FIRE_MAGNITUDE,
    .recoilMagnitude = SHOTGUN_RECOIL_MAGNITUDE,
    .damping = SHOTGUN_DAMPING,
    .charge = SHOTGUN_CHARGE,
    .coolDown = SHOTGUN_COOL_DOWN,
    .maxDistance = SHOTGUN_MAX_DISTANCE,
    .radius = SHOTGUN_RADIUS,
    .density = SHOTGUN_DENSITY,
    .mass = SHOTGUN_MASS,
    .invMass = SHOTGUN_INV_MASS,
    .initialSpeed = SHOTGUN_FIRE_MAGNITUDE * SHOTGUN_INV_MASS,
    .maxBounces = SHOTGUN_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF * SHOTGUN_ENERGY_REFILL_COEF,
    .spawnWeight = SHOTGUN_SPAWN_WEIGHT,
};

const weaponInformation imploder = {
    .type = IMPLODER_WEAPON,
    .isPhysicsBullet = false,
    .canSleep = false,
    .numProjectiles = IMPLODER_PROJECTILES,
    .fireMagnitude = IMPLODER_FIRE_MAGNITUDE,
    .recoilMagnitude = IMPLODER_RECOIL_MAGNITUDE,
    .damping = IMPLODER_DAMPING,
    .charge = IMPLODER_CHARGE,
    .coolDown = IMPLODER_COOL_DOWN,
    .maxDistance = IMPLODER_MAX_DISTANCE,
    .radius = IMPLODER_RADIUS,
    .density = IMPLODER_DENSITY,
    .mass = IMPLODER_MASS,
    .invMass = IMPLODER_INV_MASS,
    .initialSpeed = IMPLODER_FIRE_MAGNITUDE * IMPLODER_INV_MASS,
    .maxBounces = IMPLODER_BOUNCE + 1,
    .explosive = true,
    .destroyedOnDroneHit = true,
    .explodesOnDroneHit = true,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .spawnWeight = IMPLODER_SPAWN_WEIGHT,
};

const weaponInformation accelerator = {
    .type = ACCELERATOR_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = ACCELERATOR_PROJECTILES,
    .fireMagnitude = ACCELERATOR_FIRE_MAGNITUDE,
    .recoilMagnitude = ACCELERATOR_RECOIL_MAGNITUDE,
    .damping = ACCELERATOR_DAMPING,
    .charge = ACCELERATOR_CHARGE,
    .coolDown = ACCELERATOR_COOL_DOWN,
    .maxDistance = ACCELERATOR_MAX_DISTANCE,
    .radius = ACCELERATOR_RADIUS,
    .density = ACCELERATOR_DENSITY,
    .mass = ACCELERATOR_MASS,
    .invMass = ACCELERATOR_INV_MASS,
    .initialSpeed = ACCELERATOR_FIRE_MAGNITUDE * ACCELERATOR_INV_MASS,
    .maxBounces = ACCELERATOR_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = true,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .spawnWeight = ACCELERATOR_SPAWN_WEIGHT,
};

const weaponInformation flakCannon = {
    .type = FLAK_CANNON_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = FLAK_CANNON_PROJECTILES,
    .fireMagnitude = FLAK_CANNON_FIRE_MAGNITUDE,
    .recoilMagnitude = FLAK_CANNON_RECOIL_MAGNITUDE,
    .damping = FLAK_CANNON_DAMPING,
    .charge = FLAK_CANNON_CHARGE,
    .coolDown = FLAK_CANNON_COOL_DOWN,
    .maxDistance = FLAK_CANNON_MAX_DISTANCE,
    .radius = FLAK_CANNON_RADIUS,
    .density = FLAK_CANNON_DENSITY,
    .mass = FLAK_CANNON_MASS,
    .invMass = FLAK_CANNON_INV_MASS,
    .initialSpeed = FLAK_CANNON_FIRE_MAGNITUDE * FLAK_CANNON_INV_MASS,
    .maxBounces = FLAK_CANNON_BOUNCE + 1,
    .explosive = true,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = true,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .spawnWeight = FLAK_CANNON_SPAWN_WEIGHT,
};

const weaponInformation mineLauncher = {
    .type = MINE_LAUNCHER_WEAPON,
    .isPhysicsBullet = false,
    .canSleep = true,
    .numProjectiles = MINE_LAUNCHER_PROJECTILES,
    .fireMagnitude = MINE_LAUNCHER_FIRE_MAGNITUDE,
    .recoilMagnitude = MINE_LAUNCHER_RECOIL_MAGNITUDE,
    .damping = MINE_LAUNCHER_DAMPING,
    .charge = MINE_LAUNCHER_CHARGE,
    .coolDown = MINE_LAUNCHER_COOL_DOWN,
    .maxDistance = MINE_LAUNCHER_MAX_DISTANCE,
    .radius = MINE_LAUNCHER_RADIUS,
    .density = MINE_LAUNCHER_DENSITY,
    .mass = MINE_LAUNCHER_MASS,
    .invMass = MINE_LAUNCHER_INV_MASS,
    .initialSpeed = MINE_LAUNCHER_FIRE_MAGNITUDE * MINE_LAUNCHER_INV_MASS,
    .maxBounces = MINE_LAUNCHER_BOUNCE + 1,
    .explosive = true,
    .destroyedOnDroneHit = true,
    .explodesOnDroneHit = false,
    .hasSensor = true,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .spawnWeight = MINE_LAUNCHER_SPAWN_WEIGHT,
};

const weaponInformation blackHole = {
    .type = BLACK_HOLE_WEAPON,
    .isPhysicsBullet = false,
    .canSleep = false,
    .numProjectiles = BLACK_HOLE_PROJECTILES,
    .fireMagnitude = BLACK_HOLE_FIRE_MAGNITUDE,
    .recoilMagnitude = BLACK_HOLE_RECOIL_MAGNITUDE,
    .damping = BLACK_HOLE_DAMPING,
    .charge = BLACK_HOLE_CHARGE,
    .coolDown = BLACK_HOLE_COOL_DOWN,
    .maxDistance = BLACK_HOLE_MAX_DISTANCE,
    .radius = BLACK_HOLE_RADIUS,
    .density = BLACK_HOLE_DENSITY,
    .mass = BLACK_HOLE_MASS,
    .invMass = BLACK_HOLE_INV_MASS,
    .initialSpeed = BLACK_HOLE_FIRE_MAGNITUDE * BLACK_HOLE_INV_MASS,
    .maxBounces = BLACK_HOLE_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = true,
    .energyRefillCoef = 0.0f,
    .spawnWeight = BLACK_HOLE_SPAWN_WEIGHT,
};

const weaponInformation nuke = {
    .type = NUKE_WEAPON,
    .isPhysicsBullet = false,
    .canSleep = false,
    .numProjectiles = NUKE_PROJECTILES,
    .fireMagnitude = NUKE_FIRE_MAGNITUDE,
    .recoilMagnitude = NUKE_RECOIL_MAGNITUDE,
    .damping = NUKE_DAMPING,
    .charge = NUKE_CHARGE,
    .coolDown = NUKE_COOL_DOWN,
    .maxDistance = NUKE_MAX_DISTANCE,
    .radius = NUKE_RADIUS,
    .density = NUKE_DENSITY,
    .mass = NUKE_MASS,
    .invMass = NUKE_INV_MASS,
    .initialSpeed = NUKE_FIRE_MAGNITUDE * NUKE_INV_MASS,
    .maxBounces = NUKE_BOUNCE + 1,
    .explosive = true,
    .destroyedOnDroneHit = true,
    .explodesOnDroneHit = true,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .spawnWeight = NUKE_SPAWN_WEIGHT,
};

#ifndef AUTOPXD
weaponInformation *weaponInfos[] = {
    (weaponInformation *)&standard,
    (weaponInformation *)&machineGun,
    (weaponInformation *)&sniper,
    (weaponInformation *)&shotgun,
    (weaponInformation *)&imploder,
    (weaponInformation *)&accelerator,
    (weaponInformation *)&flakCannon,
    (weaponInformation *)&mineLauncher,
    (weaponInformation *)&blackHole,
    (weaponInformation *)&nuke,
};
#endif

// max ammo of weapon
int8_t weaponAmmo(const enum weaponType defaultWep, const enum weaponType type) {
    if (type == defaultWep) {
        return INFINITE;
    }
    switch (type) {
    case STANDARD_WEAPON:
        return STANDARD_AMMO;
    case MACHINEGUN_WEAPON:
        return MACHINEGUN_AMMO;
    case SNIPER_WEAPON:
        return SNIPER_AMMO;
    case SHOTGUN_WEAPON:
        return SHOTGUN_AMMO;
    case IMPLODER_WEAPON:
        return IMPLODER_AMMO;
    case ACCELERATOR_WEAPON:
        return ACCELERATOR_AMMO;
    case FLAK_CANNON_WEAPON:
        return FLAK_CANNON_AMMO;
    case MINE_LAUNCHER_WEAPON:
        return MINE_LAUNCHER_AMMO;
    case BLACK_HOLE_WEAPON:
        return BLACK_HOLE_AMMO;
    case NUKE_WEAPON:
        return NUKE_AMMO;
    default:
        ERRORF("unknown weapon type %d", type);
    }
}

b2ShapeId weaponSensor(const b2BodyId bodyID, const enum weaponType type) {
    b2ShapeDef sensorShapeDef = b2DefaultShapeDef();
    sensorShapeDef.density = 0.0f;
    sensorShapeDef.isSensor = true;
    sensorShapeDef.enableSensorEvents = true;
    b2Circle sensorCircle = {.center = b2Vec2_zero};

    switch (type) {
    case FLAK_CANNON_WEAPON:
        sensorShapeDef.filter.categoryBits = PROJECTILE_SHAPE;
        sensorShapeDef.filter.maskBits = DRONE_SHAPE;
        sensorCircle.radius = FLAK_CANNON_PROXIMITY_RADIUS;
        break;
    case MINE_LAUNCHER_WEAPON:
        sensorShapeDef.filter.categoryBits = PROJECTILE_SHAPE;
        sensorShapeDef.filter.maskBits = DRONE_SHAPE;
        sensorCircle.radius = MINE_LAUNCHER_PROXIMITY_RADIUS;
        break;
    case BLACK_HOLE_WEAPON:
        sensorShapeDef.filter.categoryBits = PROJECTILE_SHAPE;
        sensorShapeDef.filter.maskBits = FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE;
        sensorCircle.radius = BLACK_HOLE_PROXIMITY_RADIUS;
        break;
    default:
        ERRORF("unknown weapon type with sensor %d", type);
    }

    return b2CreateCircleShape(bodyID, &sensorShapeDef, &sensorCircle);
}

// amount of force to apply to projectile
float weaponFire(uint64_t *seed, const enum weaponType type) {
    switch (type) {
    case STANDARD_WEAPON:
        return STANDARD_FIRE_MAGNITUDE;
    case MACHINEGUN_WEAPON:
        return MACHINEGUN_FIRE_MAGNITUDE;
    case SNIPER_WEAPON:
        return SNIPER_FIRE_MAGNITUDE;
    case SHOTGUN_WEAPON: {
        const int maxOffset = 3;
        const int fireOffset = randInt(seed, -maxOffset, maxOffset);
        return SHOTGUN_FIRE_MAGNITUDE + fireOffset;
    }
    case IMPLODER_WEAPON:
        return IMPLODER_FIRE_MAGNITUDE;
    case ACCELERATOR_WEAPON:
        return ACCELERATOR_FIRE_MAGNITUDE;
    case FLAK_CANNON_WEAPON:
        return FLAK_CANNON_FIRE_MAGNITUDE;
    case MINE_LAUNCHER_WEAPON:
        return MINE_LAUNCHER_FIRE_MAGNITUDE;
    case BLACK_HOLE_WEAPON:
        return BLACK_HOLE_FIRE_MAGNITUDE;
    case NUKE_WEAPON:
        return NUKE_FIRE_MAGNITUDE;
    default:
        ERRORF("unknown weapon type %d", type);
        return 0;
    }
}

b2Vec2 weaponAdjustAim(uint64_t *seed, const enum weaponType type, const uint16_t heat, const b2Vec2 normAim) {
    switch (type) {
    case MACHINEGUN_WEAPON: {
        const float swayCoef = logBasef((heat / 5.0f) + 1, 180);
        const float maxSway = 0.1f;
        const float swayX = randFloat(seed, maxSway * -swayCoef, maxSway * swayCoef);
        const float swayY = randFloat(seed, maxSway * -swayCoef, maxSway * swayCoef);
        b2Vec2 machinegunAim = {.x = normAim.x + swayX, .y = normAim.y + swayY};
        return b2Normalize(machinegunAim);
    }
    case SHOTGUN_WEAPON: {
        const float maxOffset = 0.11f;
        const float offsetX = randFloat(seed, -maxOffset, maxOffset);
        const float offsetY = randFloat(seed, -maxOffset, maxOffset);
        b2Vec2 shotgunAim = {.x = normAim.x + offsetX, .y = normAim.y + offsetY};
        return b2Normalize(shotgunAim);
    }
    default:
        return normAim;
    }
}

// sets explosion parameters and returns true if an explosion should be created
// when a projectile is destroyed
void weaponExplosion(const enum weaponType type, b2ExplosionDef *explosionDef) {
    switch (type) {
    case IMPLODER_WEAPON:
        explosionDef->radius = 15.0f;
        explosionDef->impulsePerLength = -200.0f;
        return;
    case FLAK_CANNON_WEAPON:
        explosionDef->radius = 7.5f;
        // will typically be closer to 45 with projectile velocity
        // factored in
        explosionDef->impulsePerLength = 10.0f;
        return;
    case MINE_LAUNCHER_WEAPON:
        explosionDef->radius = 15.0f;
        explosionDef->impulsePerLength = 200.0f;
        return;
    case NUKE_WEAPON:
        explosionDef->radius = 30.0f;
        explosionDef->impulsePerLength = 300.0f;
        return;
    default:
        ERRORF("unknown weapon type %d for projectile explosion", type);
    }
}

// insertion sort should be faster than quicksort for small arrays
void insertionSort(nearEntity arr[], uint8_t size) {
    for (int16_t i = 1; i < size; i++) {
        nearEntity key = arr[i];
        int16_t j = i - 1;

        while (j >= 0 && arr[j].distanceSquared > key.distanceSquared) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }

        arr[j + 1] = key;
    }
}

#endif
