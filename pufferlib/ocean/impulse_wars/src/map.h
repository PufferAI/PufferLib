#ifndef IMPULSE_WARS_MAP_H
#define IMPULSE_WARS_MAP_H

#include <errno.h>
#include <string.h>

#include "env.h"
#include "settings.h"

// clang-format off

const char boringLayout[] = {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

mapEntry boringMap = {
    .layout = boringLayout,
    .columns = 21,
    .rows = 21,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = false,
    .weaponPickups = 8,
    .defaultWeapon = STANDARD_WEAPON,
};

const char prototypeArenaLayout[] = {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','O','O','O','O','O','O','O','d','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','w','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','W','W','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','D','D','W','W','D','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','d','O','D','D','D','D','D','O','O','O','O','O','O','O','D',
    'D','O','w','O','O','O','O','D','D','D','D','D','O','O','O','O','w','O','O','D',
    'D','O','O','O','O','O','O','D','D','D','D','D','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','W','W','O','O','O','d','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','w','O','O','D',
    'D','O','O','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

mapEntry prototypeArenaMap = {
    .layout = prototypeArenaLayout,
    .columns = 20,
    .rows = 20,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 6,
    .defaultWeapon = STANDARD_WEAPON,
};

const char snipersLayout[] = {
    'B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B',
    'B','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','B',
    'B','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','D','D','B','O','B','D','D','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','D','D','D','B','O','B','D','D','D','O','O','O','O','O','B',
    'B','O','O','O','O','O','D','D','D','B','O','B','D','D','D','O','O','O','O','O','B',
    'B','O','O','O','O','O','B','B','B','B','O','B','B','B','B','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','B','B','B','B','O','B','B','B','B','O','O','O','O','O','B',
    'B','O','O','O','O','O','D','D','D','B','O','B','D','D','D','O','O','O','O','O','B',
    'B','O','O','O','O','O','D','D','D','B','O','B','D','D','D','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','D','D','B','O','B','D','D','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','B',
    'B','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','B',
    'B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B',
};

mapEntry snipersMap = {
    .layout = snipersLayout,
    .columns = 21,
    .rows = 21,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = false,
    .weaponPickups = 6,
    .defaultWeapon = SNIPER_WEAPON,
};

const char roomsLayout[] = {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','D','D','W','O','O','O','W','D','D','D','D','D','W','O','O','O','W','D','D','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

mapEntry roomsMap = {
    .layout = roomsLayout,
    .columns = 21,
    .rows = 21,
    .randFloatingStandardWalls = 3,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 3,
    .hasSetFloatingWalls = false,
    .weaponPickups = 10,
    .defaultWeapon = STANDARD_WEAPON,
};

const char xArenaLayout[] = {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','d','O','O','O','O','O','O','O','O','O','d','O','D',
    'D','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','W','W','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','D','W','W','D','D','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','D','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','D',
    'D','O','O','O','O','O','D','D','O','O','w','O','O','O','O','D','W','W','O','O','O','O','D',
    'D','O','O','O','O','W','W','D','D','O','O','O','O','O','D','D','W','W','O','O','w','O','D',
    'D','O','w','O','O','W','W','D','O','O','O','O','d','O','O','D','D','O','O','O','O','O','D',
    'D','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','D','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','D','D','W','W','D','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','W','W','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','w','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','D',
    'D','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

mapEntry xArena = {
    .layout = xArenaLayout,
    .columns = 23,
    .rows = 23,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 8,
    .defaultWeapon = STANDARD_WEAPON,
};

const char crossBounceLayout[] = {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','B','B','B','B','O','O','O','O','B','D','D','D','D','B','O','O','O','O','B','B','B','B','D',
    'D','B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B','D',
    'D','B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B','D',
    'D','B','O','O','B','B','O','O','O','O','O','w','d','O','O','O','O','O','B','B','O','O','B','D',
    'D','O','O','O','B','D','D','O','O','O','O','O','O','O','O','O','O','D','D','B','O','O','O','D',
    'D','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','D','B','O','O','B','D','O','O','O','O','O','O','O','O','D',
    'D','B','O','O','O','O','O','O','D','D','B','O','O','B','D','D','O','O','O','O','O','O','B','D',
    'D','D','O','O','O','O','O','O','B','B','B','O','O','B','B','B','O','O','O','O','O','O','D','D',
    'D','D','O','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','w','O','O','D','D',
    'D','D','O','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','O','O','D','D',
    'D','D','O','O','O','O','O','O','B','B','B','O','O','B','B','B','O','O','O','O','O','O','D','D',
    'D','B','O','O','O','O','O','O','D','D','B','O','O','B','D','D','O','O','O','O','O','O','B','D',
    'D','O','O','O','O','O','O','O','O','D','B','O','O','B','D','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','D',
    'D','O','O','O','B','D','D','O','O','O','O','O','O','O','O','O','O','D','D','B','O','O','O','D',
    'D','B','O','O','B','B','O','O','O','O','O','d','w','O','O','O','O','O','B','B','O','O','B','D',
    'D','B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B','D',
    'D','B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B','D',
    'D','B','B','B','B','O','O','O','O','B','D','D','D','D','B','O','O','O','O','B','B','B','B','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

mapEntry crossBounce = {
    .layout = crossBounceLayout,
    .columns = 24,
    .rows = 24,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 8,
    .defaultWeapon = STANDARD_WEAPON,// TODO: make this exploding weapon
};

const char asteriskArenaLayout[]= {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','D','W','O','O','O','W','D','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','D','O','O','O','D','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','D','D','O','O','O','O','O','O','O','O','O','D','D','O','O','O','O','D',
    'D','O','O','O','O','W','W','D','O','O','O','O','O','O','O','D','W','W','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','W','W','D','O','O','O','O','O','O','O','D','W','W','O','O','O','O','D',
    'D','O','O','O','O','D','D','O','O','O','O','O','O','O','O','O','D','D','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','D','O','O','O','D','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','D','W','O','O','O','W','D','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

mapEntry asteriskArena = {
    .layout = asteriskArenaLayout,
    .columns = 23,
    .rows = 23,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = false,
    .weaponPickups = 8,
    .defaultWeapon = STANDARD_WEAPON,
};

const char foamPitLayout[] = {
    'B','B','B','W','W','W','D','D','D','B','B','D','D','D','W','W','W','B','B','B',
    'B','O','O','O','O','O','O','O','D','B','B','D','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','B','B','O','O','O','O','O','O','O','O','B',
    'W','O','O','d','O','O','O','O','O','O','O','O','O','O','O','O','d','O','O','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','d','O','O','O','O','d','O','O','O','O','O','O','D',
    'D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D',
    'B','B','B','O','O','O','O','O','O','d','d','O','O','O','O','O','O','B','B','B',
    'B','B','B','O','O','O','O','O','O','d','d','O','O','O','O','O','O','B','B','B',
    'D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D',
    'D','O','O','O','O','O','O','d','O','O','O','O','d','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','O','O','d','O','O','O','O','O','O','O','O','O','O','O','O','d','O','O','W',
    'B','O','O','O','O','O','O','O','O','B','B','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','D','B','B','D','O','O','O','O','O','O','O','B',
    'B','B','B','W','W','W','D','D','D','B','B','D','D','D','W','W','W','B','B','B',
};

mapEntry foamPitMap = {
    .layout = foamPitLayout,
    .columns = 20,
    .rows = 20,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 6,
    .defaultWeapon = STANDARD_WEAPON,
};

const char siegeLayout[] = {
    'B','B','B','W','W','W','W','W','D','D','D','D','D','D','D','D','D','W','W','W','W','W','B','B','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','O','B',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','W','O','O','O','W','D','D','W','W','O','O','O','O','O','W','W','D','D','W','O','O','O','W','W',
    'W','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','W',
    'W','O','O','O','b','O','W','O','b','O','O','O','O','O','O','O','b','O','W','O','b','O','O','O','W',
    'W','O','b','O','O','O','W','O','O','O','O','O','O','O','O','O','O','O','W','O','O','O','b','O','W',
    'W','O','O','O','O','O','W','O','O','O','O','B','B','B','O','O','O','O','W','O','O','O','O','O','W',
    'W','W','O','O','O','W','W','O','O','O','O','O','O','O','O','O','O','O','W','W','O','O','O','W','W',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','b','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'W','W','O','O','O','W','W','O','O','O','O','O','O','O','O','O','O','O','W','W','O','O','O','W','W',
    'W','O','O','O','O','O','W','O','O','O','O','B','B','B','O','O','O','O','W','O','O','O','O','O','W',
    'W','O','b','O','O','O','W','O','O','O','O','O','O','O','O','O','O','O','W','O','O','O','b','O','W',
    'W','O','O','O','b','O','W','O','b','O','O','O','O','O','O','O','b','O','W','O','b','O','O','O','W',
    'W','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','W',
    'W','W','O','O','O','W','D','D','W','W','O','O','O','O','O','W','W','D','D','W','O','O','O','W','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'B','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','B','B','W','W','W','W','W','D','D','D','D','D','D','D','D','D','W','W','W','W','W','B','B','B',
};

mapEntry siegeMap = {
    .layout = siegeLayout,
    .columns = 25,
    .rows = 24,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 5,
    .defaultWeapon = STANDARD_WEAPON,
};

// clang-format on

#ifndef AUTOPXD
mapEntry *maps[] = {
    &boringMap,
    &prototypeArenaMap,
    &snipersMap,
    &roomsMap,
    &xArena,
    &crossBounce,
    &asteriskArena,
    &foamPitMap,
    &siegeMap,
};
#endif

void resetMap(env *e) {
    // if sudden death walls were placed, remove them
    if (e->suddenDeathWallsPlaced) {
        e->suddenDeathWallsPlaced = false;
        DEBUG_LOG("removing sudden death walls");
        // remove walls from the end of the array, sudden death walls
        // are added last
        for (int16_t i = cc_array_size(e->walls) - 1; i >= 0; i--) {
            wallEntity *wall = safe_array_get_at(e->walls, i);
            if (!wall->isSuddenDeath) {
                // if we reached the first non sudden death wall, we're done
                break;
            }
            cc_array_remove_last(e->walls, NULL);
            destroyWall(e, wall, true);
        }
    }

    // place floating walls with a set position if there are any
    const mapEntry *map = maps[e->mapIdx];
    if (!map->hasSetFloatingWalls) {
        return;
    }
    DEBUG_LOG("placing set floating walls");

    const uint8_t columns = map->columns;
    const uint8_t rows = map->rows;
    const char *layout = map->layout;
    uint16_t cellIdx = 0;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
            char cellType = layout[col + (row * columns)];

            enum entityType wallType;
            switch (cellType) {
            case 'w':
                wallType = STANDARD_WALL_ENTITY;
                break;
            case 'b':
                wallType = BOUNCY_WALL_ENTITY;
                break;
            case 'd':
                wallType = DEATH_WALL_ENTITY;
                break;
            default:
                cellIdx++;
                continue;
            }

            const mapCell *cell = safe_array_get_at(e->cells, cellIdx);
            createWall(e, cell->pos, FLOATING_WALL_THICKNESS, FLOATING_WALL_THICKNESS, cellIdx, wallType, true);
            cellIdx++;
        }
    }
}

void setupMap(env *e, const uint8_t mapIdx) {
    // reset the map if we're switching to the same map
    if (e->mapIdx == mapIdx) {
        resetMap(e);
        return;
    }

    // clear the old map
    for (size_t i = 0; i < cc_array_size(e->walls); i++) {
        wallEntity *wall = safe_array_get_at(e->walls, i);
        destroyWall(e, wall, false);
    }

    for (size_t i = 0; i < cc_array_size(e->cells); i++) {
        mapCell *cell = safe_array_get_at(e->cells, i);
        fastFree(cell);
    }

    cc_array_remove_all(e->walls);
    cc_array_remove_all(e->cells);
    e->suddenDeathWallsPlaced = false;

    const uint8_t columns = maps[mapIdx]->columns;
    const uint8_t rows = maps[mapIdx]->rows;
    const char *layout = maps[mapIdx]->layout;

    e->mapIdx = mapIdx;
    e->map = maps[mapIdx];
    e->defaultWeapon = weaponInfos[maps[mapIdx]->defaultWeapon];
    if (e->isTraining && randFloat(&e->randState, 0.0f, 1.0f) < 0.25f) {
        e->defaultWeapon = weaponInfos[randInt(&e->randState, 0, NUM_WEAPONS - 1)];
    }

    uint16_t cellIdx = 0;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
            char cellType = layout[col + (row * columns)];
            enum entityType wallType;
            const float x = (col - ((columns - 1) * 0.5f)) * WALL_THICKNESS;
            const float y = (row - (rows - 1) * 0.5f) * WALL_THICKNESS;

            b2Vec2 pos = {.x = x, .y = y};
            mapCell *cell = fastCalloc(1, sizeof(mapCell));
            cell->ent = NULL;
            cell->pos = pos;
            cc_array_add(e->cells, cell);

            bool floating = false;
            float thickness = WALL_THICKNESS;
            switch (cellType) {
            case 'O':
                cellIdx++;
                continue;
            case 'w':
                thickness = FLOATING_WALL_THICKNESS;
                floating = true;
            case 'W':
                wallType = STANDARD_WALL_ENTITY;
                break;
            case 'b':
                thickness = FLOATING_WALL_THICKNESS;
                floating = true;
            case 'B':
                wallType = BOUNCY_WALL_ENTITY;
                break;
            case 'd':
                thickness = FLOATING_WALL_THICKNESS;
                floating = true;
            case 'D':
                wallType = DEATH_WALL_ENTITY;
                break;
            default:
                ERRORF("unknown map layout cell %c", cellType);
            }

            entity *ent = createWall(e, pos, thickness, thickness, cellIdx, wallType, floating);
            if (!floating) {
                cell->ent = ent;
            }
            cellIdx++;
        }
    }
}

void computeMapBoundsAndQuadrants(env *e, mapEntry *map) {
    mapBounds bounds = {.min = {.x = FLT_MAX, .y = FLT_MAX}, .max = {.x = FLT_MIN, .y = FLT_MIN}};
    for (size_t i = 0; i < cc_array_size(e->walls); i++) {
        const wallEntity *wall = safe_array_get_at(e->walls, i);
        bounds.min.x = min(wall->pos.x - wall->extent.x + WALL_THICKNESS, bounds.min.x);
        bounds.min.y = min(wall->pos.y - wall->extent.y + WALL_THICKNESS, bounds.min.y);
        bounds.max.x = max(wall->pos.x + wall->extent.x - WALL_THICKNESS, bounds.max.x);
        bounds.max.y = max(wall->pos.y + wall->extent.y - WALL_THICKNESS, bounds.max.y);
    }
    map->bounds = bounds;
    map->spawnQuads[0] = (mapBounds){
        .min = (b2Vec2){
            .x = map->bounds.min.x + WALL_THICKNESS,
            .y = map->bounds.min.y + WALL_THICKNESS,
        },
        .max = (b2Vec2){
            .x = 0.0f,
            .y = 0.0f,
        }
    };
    map->spawnQuads[1] = (mapBounds){
        .min = (b2Vec2){
            .x = 0.0f,
            .y = map->bounds.min.y + WALL_THICKNESS,
        },
        .max = (b2Vec2){
            .x = map->bounds.max.x - WALL_THICKNESS,
            .y = 0.0f,
        }
    };
    map->spawnQuads[2] = (mapBounds){
        .min = (b2Vec2){
            .x = map->bounds.min.x + WALL_THICKNESS,
            .y = 0.0f,
        },
        .max = (b2Vec2){
            .x = 0.0f,
            .y = map->bounds.max.y - WALL_THICKNESS,
        }
    };
    map->spawnQuads[3] = (mapBounds){
        .min = (b2Vec2){
            .x = 0.0f,
            .y = 0.0f,
        },
        .max = (b2Vec2){
            .x = map->bounds.max.x - WALL_THICKNESS,
            .y = map->bounds.max.y - WALL_THICKNESS,
        }
    };
}

#ifndef AUTOPXD
bool posValidDroneSpawnPoint(const env *e, const b2Vec2 pos) {
    const b2QueryFilter filter = {
        .categoryBits = DRONE_SHAPE,
        .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE,
    };
    droneEntity dummyDrone = {.pos = pos};
    const entity ent = {.type = DRONE_ENTITY, .entity = &dummyDrone};
    const enum entityType deathWallType = DEATH_WALL_ENTITY;

    if (isOverlappingCircleInLineOfSight(e, &ent, pos, DRONE_DEATH_WALL_SPAWN_DISTANCE, filter, &deathWallType)) {
        return false;
    }
    if (isOverlappingAABB(e, pos, DRONE_WALL_SPAWN_DISTANCE, filter)) {
        return false;
    }

    return true;
}
#endif

void initMaps(env *e) {
    for (uint8_t i = 0; i < NUM_MAPS; i++) {
        setupMap(e, i);
        mapEntry *map = maps[i];

        computeMapBoundsAndQuadrants(e, map);

        bool *droneSpawns = fastCalloc(map->columns * map->rows, sizeof(bool));
        uint8_t *packedLayout = fastCalloc(map->columns * map->rows, sizeof(uint8_t));
        nearEntity *nearestWalls = fastCalloc(MAX_NEAREST_WALLS * map->columns * map->rows, sizeof(nearEntity));

        for (uint16_t i = 0; i < cc_array_size(e->cells); i++) {
            const mapCell *cell = safe_array_get_at(e->cells, i);

            // precompute packed map layout
            if (cell->ent != NULL) {
                packedLayout[i] = ((cell->ent->type + 1) & TWO_BIT_MASK) << 5;
                continue;
            } else {
                // precompute valid cells for drones to spawn
                droneSpawns[i] = posValidDroneSpawnPoint(e, cell->pos);
            }

            // find nearest walls for each empty cell
            uint16_t wallIdx = 0;
            nearEntity walls[map->columns * map->rows];
            memset(walls, 0x0, map->columns * map->rows * sizeof(nearEntity));
            for (uint16_t j = 0; j < cc_array_size(e->cells); j++) {
                const mapCell *c = safe_array_get_at(e->cells, j);
                if (c->ent == NULL) {
                    continue;
                }

                walls[wallIdx].idx = wallIdx;
                walls[wallIdx].distanceSquared = b2DistanceSquared(cell->pos, c->pos);
                wallIdx++;
            }
            insertionSort(walls, wallIdx);

            const uint32_t startIdx = i * MAX_NEAREST_WALLS;
            memcpy(nearestWalls + startIdx, walls, MAX_NEAREST_WALLS * sizeof(nearEntity));
        }
        map->droneSpawns = droneSpawns;
        map->packedLayout = packedLayout;
        map->nearestWalls = nearestWalls;

        // clear floating walls from the map
        for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
            wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
            destroyWall(e, wall, false);
        }
        cc_array_remove_all(e->floatingWalls);
    }

    e->mapIdx = -1;
}

void destroyMaps() {
    for (uint8_t i = 0; i < NUM_MAPS; i++) {
        mapEntry *map = maps[i];
        fastFree(map->droneSpawns);
        fastFree(map->packedLayout);
        fastFree(map->nearestWalls);
    }
}

void placeRandFloatingWall(env *e, const enum entityType wallType) {
    b2Vec2 pos;
    if (!findOpenPos(e, FLOATING_WALL_SHAPE, &pos, -1)) {
        ERROR("failed to find open position for floating wall");
    }
    int16_t cellIdx = entityPosToCellIdx(e, pos);
    createWall(e, pos, FLOATING_WALL_THICKNESS, FLOATING_WALL_THICKNESS, cellIdx, wallType, true);
}

void placeRandFloatingWalls(env *e, const int mapIdx) {
    for (int i = 0; i < maps[mapIdx]->randFloatingStandardWalls; i++) {
        placeRandFloatingWall(e, STANDARD_WALL_ENTITY);
    }
    for (int i = 0; i < maps[mapIdx]->randFloatingBouncyWalls; i++) {
        placeRandFloatingWall(e, BOUNCY_WALL_ENTITY);
    }
    for (int i = 0; i < maps[mapIdx]->randFloatingDeathWalls; i++) {
        placeRandFloatingWall(e, DEATH_WALL_ENTITY);
    }
}

#endif
