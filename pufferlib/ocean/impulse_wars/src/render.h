#ifndef IMPULSE_WARS_RENDER_H
#define IMPULSE_WARS_RENDER_H

#include "raymath.h"
#include "rlgl.h"

#define RLIGHTS_IMPLEMENTATION
#include "include/rlights.h"

#include "helpers.h"

#if defined(PLATFORM_DESKTOP)
#define GLSL_VERSION 330
#else // PLATFORM_ANDROID, PLATFORM_WEB
#define GLSL_VERSION 100
#endif

#define LETTER_BOUNDRY_SIZE 0.25f
#define TEXT_MAX_LAYERS 32
#define LETTER_BOUNDRY_COLOR VIOLET

bool SHOW_LETTER_BOUNDRY = false;

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = RED;
const Color PUFF_GREEN = GREEN;
const Color PUFF_YELLOW = YELLOW;
const Color PUFF_CYAN = BLUE;
const Color PUFF_WHITE = RAYWHITE;
const Color PUFF_BACKGROUND = BLACK;
const Color PUFF_BACKGROUND2 = BLACK;

bool droneControlledByHuman(const env *e, uint8_t i);

const float DEFAULT_SCALE = 11.0f;
const uint16_t DEFAULT_WIDTH = 1500;
const uint16_t DEFAULT_HEIGHT = 1000;
const uint16_t HEIGHT_LEEWAY = 75;

const float START_READY_TIME = 1.5f;
const float END_WAIT_TIME = 2.0f;

const float EXPLOSION_TIME = 0.5f;

const float DRONE_RESPAWN_GUIDE_SHRINK_TIME = 0.75f;
const float DRONE_RESPAWN_GUIDE_HOLD_TIME = 0.75f;
const float DRONE_RESPAWN_GUIDE_MAX_RADIUS = DRONE_RADIUS * 5.5f;
const float DRONE_RESPAWN_GUIDE_MIN_RADIUS = DRONE_RADIUS * 2.5f;

const float DRONE_PIECE_LIFETIME = 2.0f;

const Color barolo = {.r = 165, .g = 37, .b = 8, .a = 255};
const Color bambooBrown = {.r = 204, .g = 129, .b = 0, .a = 255};

const float droneLightRadius = 0.1f;
const float halfDroneRadius = DRONE_RADIUS / 2.0f;
const float droneThrusterLength = 1.5f * DRONE_RADIUS;
const float aimGuideLength = 0.3f * DRONE_RADIUS;
const float chargedAimGuideLength = DRONE_RADIUS;

static inline b2Vec2 rayVecToB2Vec(const env *e, const Vector2 v) {
    return (b2Vec2){.x = (v.x - e->client->halfWidth) / e->renderScale, .y = ((v.y - e->client->halfHeight - (2 * e->renderScale)) / e->renderScale)};
}

void updateTrailPoints(trailPoints *tp, const uint8_t maxLen, const b2Vec2 pos) {
    const Vector2 v = (Vector2){.x = pos.x, .y = pos.y};
    if (tp->length < maxLen) {
        tp->points[tp->length++] = v;
        return;
    }

    for (uint8_t i = 0; i < maxLen - 1; i++) {
        tp->points[i] = tp->points[i + 1];
    }
    tp->points[maxLen - 1] = v;
}

rayClient *createRayClient() {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(DEFAULT_WIDTH, DEFAULT_HEIGHT, "Impulse Wars");

    rayClient *client = fastCalloc(1, sizeof(rayClient));

    if (client->height == 0) {
#ifndef __EMSCRIPTEN__
        const int monitor = GetCurrentMonitor();
        client->height = GetMonitorHeight(monitor) - HEIGHT_LEEWAY;
#else
        client->height = DEFAULT_HEIGHT;
#endif
    }
    if (client->width == 0) {
        client->width = ((float)client->height * ((float)DEFAULT_WIDTH / (float)DEFAULT_HEIGHT));
    }
    client->scale = (float)client->height * (float)(DEFAULT_SCALE / DEFAULT_HEIGHT);

    client->halfWidth = client->width / 2.0f;
    client->halfHeight = client->height / 2.0f;

    SetWindowSize(client->width, client->height);

#ifndef __EMSCRIPTEN__
    SetTargetFPS(EVAL_FRAME_RATE);
#endif

    client->camera = fastCalloc(1, sizeof(gameCamera));
    client->camera->camera3D = (Camera3D){
        .position = (Vector3){.x = 0.0f, .y = 100.0f, .z = 0.0f},
        .target = (Vector3){.x = 0.0f, .y = 0.0f, .z = 0.0f},
        .up = (Vector3){.x = 0.0f, .y = 0.0f, .z = -1.0f},
        .fovy = 45.0f,
        .projection = CAMERA_PERSPECTIVE,
    };
    client->camera->camera2D = (Camera2D){
        .offset = (Vector2){.x = client->width / 2.0f, .y = client->height / 2.0f},
        .target = (Vector2){.x = 0.0f, .y = 0.0f},
        .zoom = 1.0f,
    };
    client->camera->orthographic = false;

    client->wallTexture = LoadTexture("resources/impulse_wars/wall_texture_map.png");
    client->blurSrcTexture = LoadRenderTexture(client->width, client->height);
    client->blurDstTexture = LoadRenderTexture(client->width, client->height);
    client->droneRawTex = LoadRenderTexture(client->width, client->height);
    client->droneBloomTex = LoadRenderTexture(client->width, client->height);
    client->projRawTex = LoadRenderTexture(client->width, client->height);
    client->projBloomTex = LoadRenderTexture(client->width, client->height);

    const char *gridVSPath = TextFormat("resources/impulse_wars/shaders/gls%i/grid.vs", GLSL_VERSION);
    const char *gridFSPath = TextFormat("resources/impulse_wars/shaders/gls%i/grid.fs", GLSL_VERSION);
    client->gridShader = LoadShader(gridVSPath, gridFSPath);
    for (int i = 0; i < 4; i++) {
        client->gridShaderPosLoc[i] = GetShaderLocation(client->gridShader, TextFormat("pos[%i]", i));
        client->gridShaderColorLoc[i] = GetShaderLocation(client->gridShader, TextFormat("color[%i]", i));
    }

    const char *blurVSPath = TextFormat("resources/impulse_wars/shaders/gls%i/blur.vs", GLSL_VERSION);
    const char *blurFSPath = TextFormat("resources/impulse_wars/shaders/gls%i/blur.fs", GLSL_VERSION);
    client->blurShader = LoadShader(blurVSPath, blurFSPath);
    client->blurShaderDirLoc = GetShaderLocation(client->blurShader, "uTexelDir");

    const char *bloomVSPath = TextFormat("resources/impulse_wars/shaders/gls%i/bloom.vs", GLSL_VERSION);
    const char *bloomFSPath = TextFormat("resources/impulse_wars/shaders/gls%i/bloom.fs", GLSL_VERSION);
    client->bloomShader = LoadShader(bloomVSPath, bloomFSPath);
    int32_t bloomModeLoc = GetShaderLocation(client->bloomShader, "uBloomMode");
    const int32_t bloomMode = 1;
    SetShaderValue(client->bloomShader, bloomModeLoc, &bloomMode, SHADER_UNIFORM_INT);
    client->bloomIntensityLoc = GetShaderLocation(client->bloomShader, "uBloomIntensity");
    client->bloomTexColorLoc = GetShaderLocation(client->bloomShader, "uTexColor");
    client->bloomTexBloomBlurLoc = GetShaderLocation(client->bloomShader, "uTexBloomBlur");

    return client;
}

void destroyRayClient(rayClient *client) {
    UnloadTexture(client->wallTexture);
    UnloadRenderTexture(client->blurSrcTexture);
    UnloadRenderTexture(client->blurDstTexture);
    UnloadRenderTexture(client->droneRawTex);
    UnloadRenderTexture(client->droneBloomTex);
    UnloadRenderTexture(client->projRawTex);
    UnloadRenderTexture(client->projBloomTex);

    UnloadShader(client->gridShader);
    UnloadShader(client->blurShader);
    UnloadShader(client->bloomShader);

    CloseWindow();
    fastFree(client->camera);
    fastFree(client);
}

const float ZOOM_SPEED = 0.04f;
const float PAN_SPEED = 0.04f;
const float MAX_CAMERA_HEIGHT = 130.0f;
const float MIN_CAMERA_HEIGHT = 60.0f;
const float BOUNDS_PADDING = 4.0f;
const float MAP_MAX_X_OFFSET = 25.0f;
const float MAP_MAX_Y_OFFSET = 10.0f;

void setCamera2DZoom(env *e) {
    gameCamera *camera = e->client->camera;

    float screenHeightInWorldUnits = 2.0f * tanf((camera->camera3D.fovy * DEG2RAD) / 2.0f) * camera->camera3D.position.y;
    camera->camera2D.zoom = e->client->height / screenHeightInWorldUnits;
}

void setupEnvCamera(env *e) {
    const float BASE_ROWS = 21.0f;
    const float scale = e->client->scale * (BASE_ROWS / e->map->rows);
    // TODO: remove this field?
    e->renderScale = scale;

    gameCamera *camera = e->client->camera;
    if (camera->orthographic) {
        camera->camera3D.projection = CAMERA_ORTHOGRAPHIC;
        camera->camera3D.position.x = 0.0f;
        camera->camera3D.position.y = 25.0f;
        camera->camera3D.position.z = -2.0f;
        camera->camera3D.target.x = 0.0f;
        camera->camera3D.target.y = 0.0f;
        camera->camera3D.target.z = -2.0f;
        camera->camera3D.fovy = (5.0f * e->map->rows) - 15.0f;

        camera->camera2D.target.x = 0.0f;
        camera->camera2D.target.y = -2.0f;
        camera->camera2D.zoom = (float)e->client->height / camera->camera3D.fovy;
    } else {
        camera->targetPos = Vector2Zero();

        camera->camera3D.projection = CAMERA_PERSPECTIVE;
        camera->maxZoom = (5 * e->map->rows) + 10;
        const float startingZoom = max((camera->maxZoom + MIN_CAMERA_HEIGHT) / 2.0f, MIN_CAMERA_HEIGHT);
        camera->camera3D.position = (Vector3){.x = 0.0f, .y = startingZoom, .z = 0.0f};
        camera->camera3D.target = (Vector3){.x = 0.0f, .y = 0.0f, .z = 0.0f};
        camera->camera3D.fovy = 45.0f;

        camera->camera2D.target.x = 0.0f;
        camera->camera2D.target.y = 0.0f;
        setCamera2DZoom(e);
    }
}

Rectangle calculatePlayersBoundingBox(const env *e) {
    float minX = FLT_MAX;
    float minY = FLT_MAX;
    float maxX = -FLT_MAX;
    float maxY = -FLT_MAX;

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        const droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead && drone->livesLeft == 0 && !drone->diedThisStep) {
            continue;
        }

        minX = min(drone->pos.x, minX);
        minY = min(drone->pos.y, minY);
        maxX = max(drone->pos.x, maxX);
        maxY = max(drone->pos.y, maxY);
    }

    Rectangle bounds = {
        .x = (minX + maxX) / 2.0f,
        .y = (minY + maxY) / 2.0f,
        .width = (maxX - minX) + BOUNDS_PADDING,
        .height = (maxY - minY) + BOUNDS_PADDING,
    };
    return bounds;
}

float calculateZoom(const env *e, Rectangle droneBounds) {
    float boundsZoom = (droneBounds.width + droneBounds.height) * 0.5f;
    boundsZoom += max(droneBounds.width, droneBounds.height) * 0.5f;

    float fov = tanf((e->client->camera->camera3D.fovy * DEG2RAD) / 2.0f);
    float zoom = boundsZoom / fov;
    zoom = (MIN_CAMERA_HEIGHT + zoom) * 0.45f;
    zoom = Clamp(zoom, MIN_CAMERA_HEIGHT, e->client->camera->maxZoom);

    return zoom;
}

void updateCamera(env *e) {
    gameCamera *camera = e->client->camera;

    if (IsKeyPressed(KEY_TAB)) {
        camera->orthographic = !camera->orthographic;
        setupEnvCamera(e);
    }
    if (camera->orthographic) {
        return;
    }

    const Rectangle droneBounds = calculatePlayersBoundingBox(e);
    if (droneBounds.width == 0 && droneBounds.height == 0) {
        return;
    }

    // smoothly move towards the center of players
    const Vector2 centerPoint = {.x = droneBounds.x, .y = droneBounds.y};
    camera->targetPos = Vector2Lerp(camera->targetPos, centerPoint, PAN_SPEED);
    camera->targetPos.x = Clamp(camera->targetPos.x, e->map->bounds.min.x + MAP_MAX_X_OFFSET, e->map->bounds.max.x - MAP_MAX_X_OFFSET);
    camera->targetPos.y = Clamp(camera->targetPos.y, e->map->bounds.min.y + MAP_MAX_Y_OFFSET, e->map->bounds.max.y - MAP_MAX_Y_OFFSET);

    camera->camera3D.target.x = camera->targetPos.x;
    camera->camera3D.target.z = camera->targetPos.y;

    camera->camera3D.position.x = camera->targetPos.x;
    camera->camera3D.position.z = camera->targetPos.y;

    // smoothly zoom in or out
    const float zoom = calculateZoom(e, droneBounds);
    camera->camera3D.position.y = Lerp(camera->camera3D.position.y, zoom, ZOOM_SPEED);
    setCamera2DZoom(e);

    // update 2D camera
    camera->camera2D.target = camera->targetPos;
}

Color getDroneColor(const uint8_t droneIdx) {
    switch (droneIdx) {
    case 0:
        return barolo;
    case 1:
        return PUFF_GREEN;
    case 2:
        return PUFF_CYAN;
    case 3:
        return PUFF_YELLOW;
    default:
        ERRORF("unsupported number of drones %d", droneIdx + 1);
    }
}

char *getWeaponAbreviation(const enum weaponType type) {
    char *name = "";
    switch (type) {
    case MACHINEGUN_WEAPON:
        name = "MCGN";
        break;
    case SNIPER_WEAPON:
        // TODO: rename to railgun everywhere
        name = "RAIL";
        break;
    case SHOTGUN_WEAPON:
        name = "SHGN";
        break;
    case IMPLODER_WEAPON:
        name = "IMPL";
        break;
    case ACCELERATOR_WEAPON:
        name = "ACCL";
        break;
    case FLAK_CANNON_WEAPON:
        name = "FLAK";
        break;
    case MINE_LAUNCHER_WEAPON:
        name = "MINE";
        break;
    case BLACK_HOLE_WEAPON:
        name = "BLKH";
        break;
    case NUKE_WEAPON:
        name = "NUKE";
        break;
    default:
        ERRORF("unknown weapon pickup type %d", type);
    }
    return name;
}

char *getWeaponName(const enum weaponType type) {
    char *name = "";
    switch (type) {
    case STANDARD_WEAPON:
        name = "Standard";
        break;
    case MACHINEGUN_WEAPON:
        name = "Machine Gun";
        break;
    case SNIPER_WEAPON:
        name = "Railgun";
        break;
    case SHOTGUN_WEAPON:
        name = "Shotgun";
        break;
    case IMPLODER_WEAPON:
        name = "Imploder";
        break;
    case ACCELERATOR_WEAPON:
        name = "Accelerator";
        break;
    case FLAK_CANNON_WEAPON:
        name = "Flak Cannon";
        break;
    case MINE_LAUNCHER_WEAPON:
        name = "Mine Launcher";
        break;
    case BLACK_HOLE_WEAPON:
        name = "Black Hole";
        break;
    case NUKE_WEAPON:
        name = "Tactical Nuke";
        break;
    default:
        ERRORF("unknown weapon pickup type %d", type);
    }
    return name;
}

float getWeaponAimGuideWidth(const enum weaponType type) {
    switch (type) {
    case STANDARD_WEAPON:
    case IMPLODER_WEAPON:
    case ACCELERATOR_WEAPON:
    case NUKE_WEAPON:
        return 5.0f;
    case FLAK_CANNON_WEAPON:
    case BLACK_HOLE_WEAPON:
        return 7.5f;
    case MACHINEGUN_WEAPON:
    case MINE_LAUNCHER_WEAPON:
        return 10.0f;
    case SNIPER_WEAPON:
        return 150.0f;
    case SHOTGUN_WEAPON:
        return 3.0f;
    default:
        ERRORF("unknown weapon when getting aim guide width %d", type);
    }
}

Color getProjectileColor(const enum weaponType type) {
    Color color;
    switch (type) {
    case STANDARD_WEAPON:
        color = PURPLE;
        break;
    case IMPLODER_WEAPON:
    case ACCELERATOR_WEAPON:
        color = DARKBLUE;
        break;
    case FLAK_CANNON_WEAPON:
        color = MAROON;
        break;
    case MACHINEGUN_WEAPON:
    case SNIPER_WEAPON:
    case SHOTGUN_WEAPON:
        color = ORANGE;
        break;
    case MINE_LAUNCHER_WEAPON:
        color = BROWN;
        break;
    case BLACK_HOLE_WEAPON:
        color = DARKGRAY;
        break;
    case NUKE_WEAPON:
        color = MAROON;
        break;
    default:
        ERRORF("unknown weapon when getting projectile color %d", type);
    }

    color.r *= 0.5f;
    color.g *= 0.5f;
    color.b *= 0.5f;
    return color;
}

void DrawCubeTexture(Texture2D texture, Vector3 position, float width, float height, float length, Color color) {
    float x = position.x;
    float y = position.y;
    float z = position.z;

    // Set desired texture to be enabled while drawing following vertex data
    rlSetTexture(texture.id);

    rlBegin(RL_QUADS);
    rlColor4ub(color.r, color.g, color.b, color.a);
    // Front Face
    rlNormal3f(0.0f, 0.0f, 1.0f); // Normal Pointing Towards Viewer
    rlTexCoord2f(0.0f, 0.0f);
    rlVertex3f(x - width / 2, y - height / 2, z + length / 2); // Bottom Left Of The Texture and Quad
    rlTexCoord2f(1.0f, 0.0f);
    rlVertex3f(x + width / 2, y - height / 2, z + length / 2); // Bottom Right Of The Texture and Quad
    rlTexCoord2f(1.0f, 1.0f);
    rlVertex3f(x + width / 2, y + height / 2, z + length / 2); // Top Right Of The Texture and Quad
    rlTexCoord2f(0.0f, 1.0f);
    rlVertex3f(x - width / 2, y + height / 2, z + length / 2); // Top Left Of The Texture and Quad
    // Back Face
    rlNormal3f(0.0f, 0.0f, -1.0f); // Normal Pointing Away From Viewer
    rlTexCoord2f(1.0f, 0.0f);
    rlVertex3f(x - width / 2, y - height / 2, z - length / 2); // Bottom Right Of The Texture and Quad
    rlTexCoord2f(1.0f, 1.0f);
    rlVertex3f(x - width / 2, y + height / 2, z - length / 2); // Top Right Of The Texture and Quad
    rlTexCoord2f(0.0f, 1.0f);
    rlVertex3f(x + width / 2, y + height / 2, z - length / 2); // Top Left Of The Texture and Quad
    rlTexCoord2f(0.0f, 0.0f);
    rlVertex3f(x + width / 2, y - height / 2, z - length / 2); // Bottom Left Of The Texture and Quad
    // Top Face
    rlNormal3f(0.0f, 1.0f, 0.0f); // Normal Pointing Up
    rlTexCoord2f(0.0f, 1.0f);
    rlVertex3f(x - width / 2, y + height / 2, z - length / 2); // Top Left Of The Texture and Quad
    rlTexCoord2f(0.0f, 0.0f);
    rlVertex3f(x - width / 2, y + height / 2, z + length / 2); // Bottom Left Of The Texture and Quad
    rlTexCoord2f(1.0f, 0.0f);
    rlVertex3f(x + width / 2, y + height / 2, z + length / 2); // Bottom Right Of The Texture and Quad
    rlTexCoord2f(1.0f, 1.0f);
    rlVertex3f(x + width / 2, y + height / 2, z - length / 2); // Top Right Of The Texture and Quad
    // Bottom Face
    rlNormal3f(0.0f, -1.0f, 0.0f); // Normal Pointing Down
    rlTexCoord2f(1.0f, 1.0f);
    rlVertex3f(x - width / 2, y - height / 2, z - length / 2); // Top Right Of The Texture and Quad
    rlTexCoord2f(0.0f, 1.0f);
    rlVertex3f(x + width / 2, y - height / 2, z - length / 2); // Top Left Of The Texture and Quad
    rlTexCoord2f(0.0f, 0.0f);
    rlVertex3f(x + width / 2, y - height / 2, z + length / 2); // Bottom Left Of The Texture and Quad
    rlTexCoord2f(1.0f, 0.0f);
    rlVertex3f(x - width / 2, y - height / 2, z + length / 2); // Bottom Right Of The Texture and Quad
    // Right face
    rlNormal3f(1.0f, 0.0f, 0.0f); // Normal Pointing Right
    rlTexCoord2f(1.0f, 0.0f);
    rlVertex3f(x + width / 2, y - height / 2, z - length / 2); // Bottom Right Of The Texture and Quad
    rlTexCoord2f(1.0f, 1.0f);
    rlVertex3f(x + width / 2, y + height / 2, z - length / 2); // Top Right Of The Texture and Quad
    rlTexCoord2f(0.0f, 1.0f);
    rlVertex3f(x + width / 2, y + height / 2, z + length / 2); // Top Left Of The Texture and Quad
    rlTexCoord2f(0.0f, 0.0f);
    rlVertex3f(x + width / 2, y - height / 2, z + length / 2); // Bottom Left Of The Texture and Quad
    // Left Face
    rlNormal3f(-1.0f, 0.0f, 0.0f); // Normal Pointing Left
    rlTexCoord2f(0.0f, 0.0f);
    rlVertex3f(x - width / 2, y - height / 2, z - length / 2); // Bottom Left Of The Texture and Quad
    rlTexCoord2f(1.0f, 0.0f);
    rlVertex3f(x - width / 2, y - height / 2, z + length / 2); // Bottom Right Of The Texture and Quad
    rlTexCoord2f(1.0f, 1.0f);
    rlVertex3f(x - width / 2, y + height / 2, z + length / 2); // Top Right Of The Texture and Quad
    rlTexCoord2f(0.0f, 1.0f);
    rlVertex3f(x - width / 2, y + height / 2, z - length / 2); // Top Left Of The Texture and Quad
    rlEnd();
    // rlPopMatrix();

    rlSetTexture(0);
}

// Draw cube with texture piece applied to all faces
void DrawCubeTextureRec(Texture2D texture, Rectangle source, Vector3 position, float width, float height, float length, Color color) {
    float x = position.x;
    float y = position.y;
    float z = position.z;
    float texWidth = (float)texture.width;
    float texHeight = (float)texture.height;

    // Set desired texture to be enabled while drawing following vertex data
    rlSetTexture(texture.id);

    // We calculate the normalized texture coordinates for the desired texture-source-rectangle
    // It means converting from (tex.width, tex.height) coordinates to [0.0f, 1.0f] equivalent
    rlBegin(RL_QUADS);
    rlColor4ub(color.r, color.g, color.b, color.a);

    // Front face
    rlNormal3f(0.0f, 0.0f, 1.0f);
    rlTexCoord2f(source.x / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x - width / 2, y - height / 2, z + length / 2);
    rlTexCoord2f((source.x + source.width) / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x + width / 2, y - height / 2, z + length / 2);
    rlTexCoord2f((source.x + source.width) / texWidth, source.y / texHeight);
    rlVertex3f(x + width / 2, y + height / 2, z + length / 2);
    rlTexCoord2f(source.x / texWidth, source.y / texHeight);
    rlVertex3f(x - width / 2, y + height / 2, z + length / 2);

    // Back face
    rlNormal3f(0.0f, 0.0f, -1.0f);
    rlTexCoord2f((source.x + source.width) / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x - width / 2, y - height / 2, z - length / 2);
    rlTexCoord2f((source.x + source.width) / texWidth, source.y / texHeight);
    rlVertex3f(x - width / 2, y + height / 2, z - length / 2);
    rlTexCoord2f(source.x / texWidth, source.y / texHeight);
    rlVertex3f(x + width / 2, y + height / 2, z - length / 2);
    rlTexCoord2f(source.x / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x + width / 2, y - height / 2, z - length / 2);

    // Top face
    rlNormal3f(0.0f, 1.0f, 0.0f);
    rlTexCoord2f(source.x / texWidth, source.y / texHeight);
    rlVertex3f(x - width / 2, y + height / 2, z - length / 2);
    rlTexCoord2f(source.x / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x - width / 2, y + height / 2, z + length / 2);
    rlTexCoord2f((source.x + source.width) / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x + width / 2, y + height / 2, z + length / 2);
    rlTexCoord2f((source.x + source.width) / texWidth, source.y / texHeight);
    rlVertex3f(x + width / 2, y + height / 2, z - length / 2);

    // Bottom face
    rlNormal3f(0.0f, -1.0f, 0.0f);
    rlTexCoord2f((source.x + source.width) / texWidth, source.y / texHeight);
    rlVertex3f(x - width / 2, y - height / 2, z - length / 2);
    rlTexCoord2f(source.x / texWidth, source.y / texHeight);
    rlVertex3f(x + width / 2, y - height / 2, z - length / 2);
    rlTexCoord2f(source.x / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x + width / 2, y - height / 2, z + length / 2);
    rlTexCoord2f((source.x + source.width) / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x - width / 2, y - height / 2, z + length / 2);

    // Right face
    rlNormal3f(1.0f, 0.0f, 0.0f);
    rlTexCoord2f((source.x + source.width) / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x + width / 2, y - height / 2, z - length / 2);
    rlTexCoord2f((source.x + source.width) / texWidth, source.y / texHeight);
    rlVertex3f(x + width / 2, y + height / 2, z - length / 2);
    rlTexCoord2f(source.x / texWidth, source.y / texHeight);
    rlVertex3f(x + width / 2, y + height / 2, z + length / 2);
    rlTexCoord2f(source.x / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x + width / 2, y - height / 2, z + length / 2);

    // Left face
    rlNormal3f(-1.0f, 0.0f, 0.0f);
    rlTexCoord2f(source.x / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x - width / 2, y - height / 2, z - length / 2);
    rlTexCoord2f((source.x + source.width) / texWidth, (source.y + source.height) / texHeight);
    rlVertex3f(x - width / 2, y - height / 2, z + length / 2);
    rlTexCoord2f((source.x + source.width) / texWidth, source.y / texHeight);
    rlVertex3f(x - width / 2, y + height / 2, z + length / 2);
    rlTexCoord2f(source.x / texWidth, source.y / texHeight);
    rlVertex3f(x - width / 2, y + height / 2, z - length / 2);

    rlEnd();

    rlSetTexture(0);
}

static void DrawTextCodepoint3D(Font font, int codepoint, Vector3 position, float fontSize, bool backface, Color tint) {
    // Character index position in sprite font
    // NOTE: In case a codepoint is not available in the font, index returned points to '?'
    int index = GetGlyphIndex(font, codepoint);
    float scale = fontSize / (float)font.baseSize;

    // Character destination rectangle on screen
    // NOTE: We consider charsPadding on drawing
    position.x += (float)(font.glyphs[index].offsetX - font.glyphPadding) / (float)font.baseSize * scale;
    position.z += (float)(font.glyphs[index].offsetY - font.glyphPadding) / (float)font.baseSize * scale;

    // Character source rectangle from font texture atlas
    // NOTE: We consider chars padding when drawing, it could be required for outline/glow shader effects
    Rectangle srcRec = {
        .x = font.recs[index].x - (float)font.glyphPadding,
        .y = font.recs[index].y - (float)font.glyphPadding,
        .width = font.recs[index].width + 2.0f * font.glyphPadding,
        .height = font.recs[index].height + 2.0f * font.glyphPadding,
    };

    float width = (float)(font.recs[index].width + 2.0f * font.glyphPadding) / (float)font.baseSize * scale;
    float height = (float)(font.recs[index].height + 2.0f * font.glyphPadding) / (float)font.baseSize * scale;

    if (font.texture.id > 0) {
        const float x = 0.0f;
        const float y = 0.0f;
        const float z = 0.0f;

        // normalized texture coordinates of the glyph inside the font texture (0.0f -> 1.0f)
        const float tx = srcRec.x / font.texture.width;
        const float ty = srcRec.y / font.texture.height;
        const float tw = (srcRec.x + srcRec.width) / font.texture.width;
        const float th = (srcRec.y + srcRec.height) / font.texture.height;

        if (SHOW_LETTER_BOUNDRY) {
            DrawCubeWiresV((Vector3){position.x + width / 2, position.y, position.z + height / 2}, (Vector3){width, LETTER_BOUNDRY_SIZE, height}, LETTER_BOUNDRY_COLOR);
        }

        rlCheckRenderBatchLimit(4 + 4 * backface);
        rlSetTexture(font.texture.id);

        rlPushMatrix();
        rlTranslatef(position.x, position.y, position.z);

        rlBegin(RL_QUADS);
        rlColor4ub(tint.r, tint.g, tint.b, tint.a);

        // Front Face
        rlNormal3f(0.0f, 1.0f, 0.0f); // Normal Pointing Up
        rlTexCoord2f(tx, ty);
        rlVertex3f(x, y, z); // Top Left Of The Texture and Quad
        rlTexCoord2f(tx, th);
        rlVertex3f(x, y, z + height); // Bottom Left Of The Texture and Quad
        rlTexCoord2f(tw, th);
        rlVertex3f(x + width, y, z + height); // Bottom Right Of The Texture and Quad
        rlTexCoord2f(tw, ty);
        rlVertex3f(x + width, y, z); // Top Right Of The Texture and Quad

        if (backface) {
            // Back Face
            rlNormal3f(0.0f, -1.0f, 0.0f); // Normal Pointing Down
            rlTexCoord2f(tx, ty);
            rlVertex3f(x, y, z); // Top Right Of The Texture and Quad
            rlTexCoord2f(tw, ty);
            rlVertex3f(x + width, y, z); // Top Left Of The Texture and Quad
            rlTexCoord2f(tw, th);
            rlVertex3f(x + width, y, z + height); // Bottom Left Of The Texture and Quad
            rlTexCoord2f(tx, th);
            rlVertex3f(x, y, z + height); // Bottom Right Of The Texture and Quad
        }
        rlEnd();
        rlPopMatrix();

        rlSetTexture(0);
    }
}

static void DrawText3D(Font font, const char *text, Vector3 position, float fontSize, float fontSpacing, float lineSpacing, bool backface, Color tint) {
    int length = TextLength(text); // Total length in bytes of the text, scanned by codepoints in loop

    float textOffsetY = 0.0f; // Offset between lines (on line break '\n')
    float textOffsetX = 0.0f; // Offset X to next character to draw

    const float scale = fontSize / (float)font.baseSize;

    for (int i = 0; i < length;) {
        // Get next codepoint from byte string and glyph index in font
        int codepointByteCount = 0;
        int codepoint = GetCodepoint(&text[i], &codepointByteCount);
        int index = GetGlyphIndex(font, codepoint);

        // NOTE: Normally we exit the decoding sequence as soon as a bad byte is found (and return 0x3f)
        // but we need to draw all of the bad bytes using the '?' symbol moving one byte
        if (codepoint == 0x3f) {
            codepointByteCount = 1;
        }

        if (codepoint == '\n') {
            // NOTE: Fixed line spacing of 1.5 line-height
            // TODO: Support custom line spacing defined by user
            textOffsetY += scale + lineSpacing / (float)font.baseSize * scale;
            textOffsetX = 0.0f;
        } else {
            if ((codepoint != ' ') && (codepoint != '\t')) {
                DrawTextCodepoint3D(font, codepoint, (Vector3){position.x + textOffsetX, position.y, position.z + textOffsetY}, fontSize, backface, tint);
            }

            if (font.glyphs[index].advanceX == 0) {
                textOffsetX += (float)(font.recs[index].width + fontSpacing) / (float)font.baseSize * scale;
            } else {
                textOffsetX += (float)(font.glyphs[index].advanceX + fontSpacing) / (float)font.baseSize * scale;
            }
        }

        i += codepointByteCount; // Move text bytes counter to next codepoint
    }
}

void renderTimer(const env *e, const char *timerStr, const Color color) {
    int fontSize = 2.5 * e->client->scale;
    int textWidth = MeasureText(timerStr, fontSize);
    int posX = (e->client->width - textWidth) / 2;
    DrawText(timerStr, posX, e->client->scale, fontSize, color);
}

void renderUI(const env *e, const bool starting) {
    // render drone info
    const uint8_t fontSize = 2 * e->client->scale;
    const uint8_t xMargin = 5 * e->client->scale;
    const uint8_t yMargin = 12 * e->client->scale;

    for (int i = 0; i < e->numDrones; i++) {
        const droneEntity *drone = safe_array_get_at(e->drones, i);

        const char *droneNum = TextFormat("Drone %d", drone->idx + 1);
        const Vector2 textSize = MeasureTextEx(GetFontDefault(), droneNum, fontSize, fontSize / 10);
        const uint16_t lineWidth = textSize.x + (3 * (e->client->scale * 2.5f));

        uint16_t x = 0;
        uint16_t y = 0;
        switch (drone->idx) {
        case 0:
            x = xMargin;
            y = yMargin;
            break;
        case 1:
            x = e->client->width - lineWidth - xMargin;
            y = yMargin;
            break;
        case 2:
            x = xMargin;
            y = e->client->height - yMargin - (6 * e->client->scale);
            break;
        case 3:
            x = e->client->width - lineWidth - xMargin;
            y = e->client->height - yMargin - (6 * e->client->scale);
            break;
        }

        Color textColor = PUFF_WHITE;
        if (drone->livesLeft == 0) {
            textColor = Fade(PUFF_WHITE, 0.5f);
        }
        DrawText(droneNum, x, y, fontSize, textColor);

        uint16_t lifeX = x + textSize.x;
        uint16_t lifeY = y + (textSize.y / 2);
        const Color droneColor = getDroneColor(drone->idx);
        for (uint8_t i = 0; i < drone->livesLeft; i++) {
            lifeX += e->client->scale * 2.5f;
            DrawCircleLines(lifeX, lifeY, e->client->scale, droneColor);
        }

        y += textSize.y + e->client->scale;
        DrawLine(x, y, x + textSize.x, y, droneColor);

        y += e->client->scale;
        const char *weaponName = getWeaponName(drone->weaponInfo->type);
        DrawText(weaponName, x, y, fontSize, textColor);

        y += textSize.y + e->client->scale;
        DrawLine(x, y, x + MeasureText(weaponName, fontSize), y, droneColor);

        y += e->client->scale;
        char *playerType = "";
        if (droneControlledByHuman(e, drone->idx)) {
            playerType = "Human";
        } else if (drone->idx < e->numAgents) {
            playerType = "NN";
        } else {
            playerType = "Scripted";
        }
        char *droneInfo;
        if (e->teamsEnabled) {
            droneInfo = (char *)TextFormat("%s | Team %d", playerType, drone->team + 1);
        } else {
            droneInfo = playerType;
        }

        DrawText(droneInfo, x, y, fontSize, textColor);
    }

    // render timer
    if (starting) {
        renderTimer(e, "READY", PUFF_WHITE);
        return;
    } else if (e->stepsLeft > (ROUND_STEPS - 1) * e->frameRate) {
        renderTimer(e, "GO!", PUFF_WHITE);
        return;
    } else if (e->stepsLeft == 0) {
        renderTimer(e, "SUDDEN DEATH", PUFF_WHITE);
        return;
    }

    char *timerStr;
    if (e->stepsLeft >= 10 * e->frameRate) {
        timerStr = (char *)TextFormat("%d", (uint16_t)(e->stepsLeft / e->frameRate));
    } else {
        timerStr = (char *)TextFormat("0%d", (uint16_t)(e->stepsLeft / e->frameRate));
    }
    renderTimer(e, timerStr, PUFF_WHITE);
}

// TODO: fix and improve
void renderBrakeTrails(const env *e) {
    MAYBE_UNUSED(e);
    // const float maxLifetime = 3.0f * e->frameRate;
    // const float radius = 0.3f * e->renderScale;

    // CC_ArrayIter brakeTrailIter;
    // cc_array_iter_init(&brakeTrailIter, e->brakeTrailPoints);
    // brakeTrailPoint *trailPoint;
    // while (cc_array_iter_next(&brakeTrailIter, (void **)&trailPoint) != CC_ITER_END) {
    //     if (trailPoint->lifetime == UINT16_MAX) {
    //         trailPoint->lifetime = maxLifetime;
    //     } else if (trailPoint->lifetime == 0) {
    //         fastFree(trailPoint);
    //         cc_array_iter_remove(&brakeTrailIter, NULL);
    //         continue;
    //     }

    //     Color trailColor = Fade(GRAY, 0.133f * (trailPoint->lifetime / maxLifetime));
    //     DrawCircleV(b2VecToRayVec(e, trailPoint->pos), radius, trailColor);
    //     trailPoint->lifetime--;
    // }
}

// TODO: improve
void renderExplosions(const env *e) {
    const uint16_t maxRenderSteps = EXPLOSION_TIME * e->frameRate;

    CC_ArrayIter iter;
    cc_array_iter_init(&iter, e->explosions);
    explosionInfo *explosion;

    while (cc_array_iter_next(&iter, (void **)&explosion) != CC_ITER_END) {
        if (explosion->renderSteps == UINT16_MAX) {
            explosion->renderSteps = maxRenderSteps;
        } else if (explosion->renderSteps == 0) {
            fastFree(explosion);
            cc_array_iter_remove(&iter, NULL);
            continue;
        }

        // color bursts with a bit of the parent drone's color'
        const float alpha = (float)explosion->renderSteps / maxRenderSteps;
        BeginBlendMode(BLEND_ALPHA);
        if (false && explosion->isBurst) {
            const Color droneColor = Fade(getDroneColor(explosion->droneIdx), alpha);
            DrawSphereEx(
                (Vector3){.x = explosion->def.position.x, .y = 0.5f, .z = explosion->def.position.y},
                explosion->def.radius + explosion->def.falloff,
                20,
                50,
                DARKGRAY
            );
            DrawSphereEx(
                (Vector3){.x = explosion->def.position.x, .y = 0.5f, .z = explosion->def.position.y},
                explosion->def.radius,
                20,
                50,
                droneColor
            );
        } else {
            const Color falloffColor = Fade(GRAY, alpha);
            const Color explosionColor = Fade(RAYWHITE, alpha);

            DrawSphereEx(
                (Vector3){.x = explosion->def.position.x, .y = 0.5f, .z = explosion->def.position.y},
                explosion->def.radius + explosion->def.falloff,
                20,
                50,
                falloffColor
            );
            DrawSphereEx(
                (Vector3){.x = explosion->def.position.x, .y = 0.5f, .z = explosion->def.position.y},
                explosion->def.radius,
                20,
                50,
                explosionColor
            );
        }
        EndBlendMode();

        explosion->renderSteps = max(explosion->renderSteps - 1, 0);
    }
}

// TODO: add bloom lines at drone level
void renderWall(const env *e, const wallEntity *wall) {
    Color color = {0};
    Rectangle textureRec;
    switch (wall->type) {
    case STANDARD_WALL_ENTITY:
        color = PUFF_CYAN;
        textureRec = (Rectangle){
            0.0f,
            0.0f,
            e->client->wallTexture.width / 2.0f,
            e->client->wallTexture.height / 2.0f,
        };
        break;
    case BOUNCY_WALL_ENTITY:
        color = PUFF_YELLOW;
        textureRec = (Rectangle){
            0.0f,
            e->client->wallTexture.height / 2.0f,
            e->client->wallTexture.width / 2.0f,
            e->client->wallTexture.height / 2.0f,
        };
        break;
    case DEATH_WALL_ENTITY:
        color = PUFF_RED;
        textureRec = (Rectangle){
            e->client->wallTexture.width / 2.0f,
            0.0f,
            e->client->wallTexture.width / 2.0f,
            e->client->wallTexture.height / 2.0f,
        };
        break;
    default:
        ERRORF("unknown wall type %d", wall->type);
    }

    float angle = 0.0f;
    if (wall->isFloating) {
        angle = b2Rot_GetAngle(wall->rot);
        angle *= RAD2DEG;
    }

    float y;
    float y_size;
    if (wall->isFloating) {
        y = (FLOATING_WALL_THICKNESS / 2.0f) - 1.0f;
        y_size = FLOATING_WALL_THICKNESS;
    } else {
        y = (WALL_THICKNESS / 2.0f) - 1.0f;
        y_size = WALL_THICKNESS;
    }

    float x = 2.0f * wall->extent.x;
    float z = 2.0f * wall->extent.y;

    rlPushMatrix();
    rlTranslatef(wall->pos.x, 0.0f, wall->pos.y);
    rlRotatef(-angle, 0.0f, 1.0f, 0.0f);

    DrawCubeTextureRec(
        e->client->wallTexture,
        textureRec,
        (Vector3){.x = 0.0f, .y = y, .z = 0.0f},
        x,
        y_size,
        z,
        color
    );

    if (!wall->isFloating) {
        for (uint8_t i = 0; i < 3; i++) {
            y -= WALL_THICKNESS;
            DrawCubeTextureRec(
                e->client->wallTexture,
                textureRec,
                (Vector3){.x = 0.0f, .y = y, .z = 0.0f},
                x,
                y_size,
                z,
                color
            );
        }
    }

    rlPopMatrix();
}

void renderWeaponPickup(const env *e, const weaponPickupEntity *pickup) {
    if (pickup->respawnWait != 0.0f || pickup->floatingWallsTouching != 0) {
        return;
    }
    Rectangle textureRec = (Rectangle){
        e->client->wallTexture.width / 2.0f,
        e->client->wallTexture.height / 2.0f,
        e->client->wallTexture.width / 2.0f,
        e->client->wallTexture.height / 2.0f,
    };

    DrawCubeTextureRec(
        e->client->wallTexture,
        textureRec,
        (Vector3){.x = pickup->pos.x, .y = 0.5f, .z = pickup->pos.y},
        PICKUP_THICKNESS,
        0.0f,
        PICKUP_THICKNESS,
        WHITE
    );

    const char *weaponName = getWeaponAbreviation(pickup->weapon);
    Vector3 textPos = (Vector3){
        .x = pickup->pos.x - PICKUP_THICKNESS / 2.0f,
        .y = 1.0f,
        .z = pickup->pos.y - PICKUP_THICKNESS / 2.0f
    };

    DrawText3D(GetFontDefault(), weaponName, textPos, 12, 0.5f, -1.0f, false, PUFF_WHITE);
}

void renderDronePieces(env *e) {
    const float maxLifetime = e->frameRate * DRONE_PIECE_LIFETIME;

    CC_ArrayIter iter;
    cc_array_iter_init(&iter, e->dronePieces);
    dronePieceEntity *piece;

    while (cc_array_iter_next(&iter, (void **)&piece) != CC_ITER_END) {
        if (piece->lifetime == UINT16_MAX) {
            piece->lifetime = maxLifetime;
        }

        float baseAlpha = 1.0f;
        if (piece->isShieldPiece) {
            baseAlpha = 0.5f;
        }
        const float alpha = 1.0f - (baseAlpha * ((float)piece->lifetime / maxLifetime));
        const float finalAlpha = 1.0f - (SQUARED(alpha) * alpha);
        const Color color = Fade(getDroneColor(piece->droneIdx), finalAlpha);
        const float angle = RAD2DEG * b2Rot_GetAngle(piece->rot);

        // Draw edges of the triangle
        rlPushMatrix();
        rlTranslatef(piece->pos.x, 0.0f, piece->pos.y);
        rlRotatef(-angle, 0.0f, 1.0f, 0.0f);

        rlBegin(RL_LINES);
        rlColor4ub(color.r, color.g, color.b, color.a);

        rlVertex3f(piece->vertices[0].x, 0.5f, piece->vertices[0].y);
        rlVertex3f(piece->vertices[1].x, 0.5f, piece->vertices[1].y);

        rlVertex3f(piece->vertices[1].x, 0.5f, piece->vertices[1].y);
        rlVertex3f(piece->vertices[2].x, 0.5f, piece->vertices[2].y);

        rlVertex3f(piece->vertices[2].x, 0.5f, piece->vertices[2].y);
        rlVertex3f(piece->vertices[0].x, 0.5f, piece->vertices[0].y);

        rlEnd();

        rlPopMatrix();

        piece->lifetime--;
        if (piece->lifetime == 0) {
            destroyDronePiece(e, piece);
            cc_array_iter_remove_fast(&iter, NULL);
        }
    }
}

void renderDroneRespawnGuides(const env *e, droneEntity *drone) {
    if (drone->respawnGuideLifetime == 0) {
        return;
    }

    const float maxLifetime = e->frameRate * (DRONE_RESPAWN_GUIDE_SHRINK_TIME + DRONE_RESPAWN_GUIDE_HOLD_TIME);
    const uint16_t shrinkTime = e->frameRate * DRONE_RESPAWN_GUIDE_SHRINK_TIME;
    if (drone->respawnGuideLifetime == UINT16_MAX) {
        drone->respawnGuideLifetime = maxLifetime;
    }

    float radius = DRONE_RESPAWN_GUIDE_MIN_RADIUS;
    if (drone->respawnGuideLifetime >= maxLifetime - shrinkTime) {
        radius += DRONE_RESPAWN_GUIDE_MAX_RADIUS * ((drone->respawnGuideLifetime - (e->frameRate * DRONE_RESPAWN_GUIDE_HOLD_TIME)) / shrinkTime);
    }

    Vector2 dronePos = (Vector2){.x = drone->pos.x, .y = drone->pos.y};
    DrawCircleLinesV(dronePos, radius, getDroneColor(drone->idx));

    drone->respawnGuideLifetime--;
}

b2RayResult droneAimingAt(const env *e, const droneEntity *drone) {
    const b2Vec2 rayEnd = b2MulAdd(drone->pos, 150.0f, drone->lastAim);
    const b2Vec2 translation = b2Sub(rayEnd, drone->pos);
    const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | DRONE_SHAPE};
    return b2World_CastRayClosest(e->worldID, drone->pos, translation, filter);
}

void renderDroneAimGuide(const env *e, const droneEntity *drone) {
    // find length of laser aiming guide by where it touches the nearest shape
    const b2RayResult rayRes = droneAimingAt(e, drone);
    ASSERT(b2Shape_IsValid(rayRes.shapeId));
    const entity *ent = b2Shape_GetUserData(rayRes.shapeId);

    const b2DistanceOutput output = closestPoint(drone->ent, ent);
    float aimGuideWidth = getWeaponAimGuideWidth(drone->weaponInfo->type);
    aimGuideWidth = min(aimGuideWidth, output.distance + 0.1f) + (DRONE_RADIUS * 2.0f);

    // render laser aim guide
    const b2Vec2 pos = b2MulAdd(drone->pos, aimGuideWidth / 2.0f, drone->lastAim);
    const float aimAngle = RAD2DEG * b2Atan2(drone->lastAim.y, drone->lastAim.x);

    rlPushMatrix();
    rlTranslatef(pos.x, 0.0f, pos.y);
    rlRotatef(-aimAngle, 0.0f, 1.0f, 0.0f);

    const Color droneColor = getDroneColor(drone->idx);
    DrawCube((Vector3){.x = 0.0f, .y = 0.0f, .z = 0.0f}, aimGuideWidth, 0.0f, aimGuideLength, droneColor);

    rlPopMatrix();
}

void renderDroneGuides(env *e, const droneEntity *drone, const bool ending) {
    // render thruster move guide
    if (!b2VecEqual(drone->lastMove, b2Vec2_zero) && !ending) {
        const float moveMagnitude = b2Length(drone->lastMove);
        const float thrusterAngle = RAD2DEG * b2Atan2(-drone->lastMove.y, -drone->lastMove.x);
        const float flickerWidth = randFloat(&e->randState, -0.05f, 0.05f);
        const float thrusterWidth = 2.5f * ((halfDroneRadius * moveMagnitude) + halfDroneRadius + flickerWidth);
        const b2Vec2 thrusterPos = b2MulAdd(drone->pos, -thrusterWidth / 2.0f, drone->lastMove);
        const Color thrusterColor = Fade(getDroneColor(drone->idx), 0.9);

        rlPushMatrix();
        rlTranslatef(thrusterPos.x, 0.0f, thrusterPos.y);
        rlRotatef(-thrusterAngle, 0.0f, 1.0f, 0.0f);

        DrawCube((Vector3){.x = 0.0f, .y = 0.0f, .z = 0.0f}, thrusterWidth, 0.0f, droneThrusterLength, thrusterColor);

        rlPopMatrix();
    }

    renderDroneAimGuide(e, drone);
}

void renderDroneTrail(const droneEntity *drone) {
    if (drone->trailPoints.length < 2) {
        return;
    }

    const float trailWidth = DRONE_RADIUS;
    const float numPoints = drone->trailPoints.length;
    const Color droneColor = getDroneColor(drone->idx);

    for (uint8_t i = 0; i < drone->trailPoints.length - 1; i++) {
        const Vector2 p0 = drone->trailPoints.points[i];
        const Vector2 p1 = drone->trailPoints.points[i + 1];

        // compute direction and a perpendicular vector
        Vector2 segment = Vector2Subtract(p1, p0);
        if (Vector2Length(segment) == 0) {
            continue;
        }
        segment = Vector2Normalize(segment);
        const Vector2 perp = {-segment.y, segment.x};

        // compute four vertices for the quad segment
        const Vector2 v0 = Vector2Add(p0, Vector2Scale(perp, trailWidth));
        const Vector2 v1 = Vector2Subtract(p0, Vector2Scale(perp, trailWidth));
        const Vector2 v2 = Vector2Add(p1, Vector2Scale(perp, trailWidth));
        const Vector2 v3 = Vector2Subtract(p1, Vector2Scale(perp, trailWidth));

        // draw the quad as two triangles
        const float alpha0 = 0.7f * ((float)(i + 1) / numPoints);
        const float alpha1 = 0.7f * ((float)(i + 2) / numPoints);
        DrawTriangle3D(
            (Vector3){.x = v0.x, .y = 0.0f, .z = v0.y},
            (Vector3){.x = v2.x, .y = 0.0f, .z = v2.y},
            (Vector3){.x = v1.x, .y = 0.0f, .z = v1.y},
            Fade(droneColor, alpha0)
        );
        DrawTriangle3D(
            (Vector3){.x = v1.x, .y = 0.0f, .z = v1.y},
            (Vector3){.x = v2.x, .y = 0.0f, .z = v2.y},
            (Vector3){.x = v3.x, .y = 0.0f, .z = v3.y},
            Fade(droneColor, alpha1)
        );
    }
}

void renderDroneLight(const droneEntity *drone) {
    const Color droneColor = getDroneColor(drone->idx);

    DrawCylinder(
        (Vector3){.x = drone->pos.x, .y = 0.0, .z = drone->pos.y},
        DRONE_RADIUS,
        DRONE_RADIUS,
        0.0f,
        32,
        droneColor
    );
}

void renderDrone(const droneEntity *drone) {
    renderDroneLight(drone);

    DrawSphere(
        (Vector3){.x = drone->pos.x, .y = 0.0f, .z = drone->pos.y},
        DRONE_RADIUS - droneLightRadius,
        BLACK
    );

    if (drone->shield != NULL) {
        DrawSphereWires(
            (Vector3){.x = drone->shield->pos.x, .y = 0.0f, .z = drone->shield->pos.y},
            DRONE_SHIELD_RADIUS,
            6,
            12,
            Fade(getDroneColor(drone->idx), 0.5f)
        );
    }
}

void renderDroneAmmo(const env *e, const droneEntity *drone) {
    Vector2 worldPos = {.x = drone->pos.x, .y = drone->pos.y};
    Vector2 screenPos = GetWorldToScreen2D(worldPos, e->client->camera->camera2D);

    // draw ammo count
    const float fontSize = 1.5f * e->client->camera->camera2D.zoom;
    const char *ammoStr = TextFormat("%d", drone->ammo);
    const Vector2 textSize = MeasureTextEx(GetFontDefault(), ammoStr, fontSize, fontSize / 10.0f);
    const Vector2 textOrigin = {.x = screenPos.x - (textSize.x / 2), .y = screenPos.y + (1.3f * textSize.y)};
    DrawTextEx(GetFontDefault(), ammoStr, textOrigin, fontSize, fontSize / 10.f, RAYWHITE);
}

void renderDroneUI(const droneEntity *drone) {
    // draw energy meter
    const float energyMeterInnerRadius = 0.6f;
    const float energyMeterOuterRadius = 0.3f;
    const Vector2 energyMeterOrigin = {.x = drone->pos.x, .y = drone->pos.y};
    float energyMeterEndAngle = 360.f * drone->energyLeft;
    Color energyMeterColor = RAYWHITE;
    if (drone->shield != NULL) {
        energyMeterColor = bambooBrown;
    } else if (drone->energyFullyDepleted && drone->energyRefillWait != 0.0f) {
        energyMeterColor = bambooBrown;
        energyMeterEndAngle = 360.0f * (1.0f - (drone->energyRefillWait / (DRONE_ENERGY_REFILL_EMPTY_WAIT)));
    } else if (drone->energyFullyDepleted) {
        energyMeterColor = GRAY;
    }
    DrawRing(energyMeterOrigin, energyMeterInnerRadius, energyMeterOuterRadius, 0.0f, energyMeterEndAngle, 32, energyMeterColor);

    // draw burst charge indicator
    if (drone->chargingBurst) {
        const float alpha = min(drone->burstCharge + (50.0f / 255.0f), 1.0f);
        const Color burstChargeColor = Fade(RAYWHITE, alpha);
        const float burstChargeOuterRadius = (DRONE_BURST_RADIUS_BASE * drone->burstCharge) + DRONE_BURST_RADIUS_MIN;
        const float burstChargeInnerRadius = burstChargeOuterRadius - 0.15f;
        DrawRing(energyMeterOrigin, burstChargeInnerRadius, burstChargeOuterRadius, 0.0f, 360.0f, 50, burstChargeColor);
    }

    const float maxCharge = drone->weaponInfo->charge;
    if (maxCharge == 0) {
        return;
    }

    // draw charge meter
    const Vector2 chargeMeterOrigin = {.x = drone->pos.x, .y = drone->pos.y + 3.3f};
    const float chargeMeterInnerRadius = 1.0f;
    const float chargeMeterOuterRadius = 0.5f;
    const float chargeMeterStartAngle = 157.5f;
    const float chargeMeterEndAngle = chargeMeterStartAngle - (135.0f * (drone->weaponCharge / drone->weaponInfo->charge));

    rlPushMatrix();
    rlTranslatef(chargeMeterOrigin.x, chargeMeterOrigin.y, 0.0f);
    rlScalef(1.7f, 1.0f, 1.0f);

    DrawRing(Vector2Zero(), chargeMeterInnerRadius, chargeMeterOuterRadius, chargeMeterStartAngle, chargeMeterEndAngle, 32, RAYWHITE);
    DrawRingLines(Vector2Zero(), chargeMeterInnerRadius, chargeMeterOuterRadius, chargeMeterStartAngle, chargeMeterStartAngle - 135.0f, 10, RAYWHITE);

    rlPopMatrix();
}

void renderProjectileTrail(const projectileEntity *proj) {
    if (proj->trailPoints.length < 2) {
        return; // need at least two points
    }

    const float maxWidth = proj->weaponInfo->radius;
    const float numPoints = proj->trailPoints.length;

    for (uint8_t i = 0; i < proj->trailPoints.length - 1; i++) {
        const Vector2 p0 = proj->trailPoints.points[i];
        const Vector2 p1 = proj->trailPoints.points[i + 1];

        // Compute a perpendicular vector for the segment
        Vector2 dir = Vector2Subtract(p1, p0);
        if (Vector2Length(dir) == 0) {
            continue; // Avoid division by zero
        }
        dir = Vector2Normalize(dir);
        const Vector2 perp = {-dir.y, dir.x};

        // Compute widths for the start and end of the segment.
        // Taper so that older segments are narrower.
        const float taper0 = (float)(i + 1) / numPoints;
        const float taper1 = (float)(i + 2) / numPoints;
        const float width0 = maxWidth * taper0;
        const float width1 = maxWidth * taper1;

        // Calculate two vertices on each side of the segment.
        const Vector2 v0 = Vector2Add(p0, Vector2Scale(perp, width0));
        const Vector2 v1 = Vector2Subtract(p0, Vector2Scale(perp, width0));
        const Vector2 v2 = Vector2Add(p1, Vector2Scale(perp, width1));
        const Vector2 v3 = Vector2Subtract(p1, Vector2Scale(perp, width1));

        // Draw two triangles for the quad and fade the color with distance so older parts are more transparent.
        const Color color = getProjectileColor(proj->weaponInfo->type);
        DrawTriangle3D(
            (Vector3){.x = v0.x, .y = 0.5f, .z = v0.y},
            (Vector3){.x = v2.x, .y = 0.5f, .z = v2.y},
            (Vector3){.x = v1.x, .y = 0.5f, .z = v1.y},
            Fade(color, taper0)
        );
        DrawTriangle3D(
            (Vector3){.x = v1.x, .y = 0.5f, .z = v1.y},
            (Vector3){.x = v2.x, .y = 0.5f, .z = v2.y},
            (Vector3){.x = v3.x, .y = 0.5f, .z = v3.y},
            Fade(color, taper1)
        );
    }
}

void renderProjectile(const projectileEntity *projectile) {
    DrawSphere(
        (Vector3){.x = projectile->pos.x, .y = 0.5f, .z = projectile->pos.y},
        projectile->weaponInfo->radius,
        getProjectileColor(projectile->weaponInfo->type)
    );
}

void renderBannerText(env *e, const bool starting, const int8_t winner, const int8_t winningTeam) {
    char *winStr;
    Color color = PUFF_WHITE;

    if (starting) {
        winStr = "Ready?";
    } else if (winner == -1 && winningTeam == -1) {
        winStr = "Tie";
    } else if (e->teamsEnabled) {
        winStr = (char *)TextFormat("Team %d wins!", winningTeam + 1);
    } else {
        winStr = (char *)TextFormat("Player %d wins!", winner + 1);
        color = getDroneColor(winner);
    }

    uint16_t fontSize = 5 * e->client->scale;
    uint16_t textWidth = MeasureText(winStr, fontSize);
    uint16_t posX = (e->client->halfWidth - (textWidth / 2));
    DrawText(winStr, posX, e->client->halfHeight, fontSize, color);
}

void applyBloom(const env *e, RenderTexture2D srcTex, RenderTexture2D dstTex, const float bloomIntensity) {
    BeginTextureMode(e->client->blurSrcTexture);
    ClearBackground(BLANK);
    DrawTextureRec(srcTex.texture, (Rectangle){0.0f, 0.0f, e->client->width, -e->client->height}, Vector2Zero(), WHITE);
    EndTextureMode();

    // apply horizontal and vertical blurring
    RenderTexture2D blurSrcTex = e->client->blurDstTexture;
    RenderTexture2D blurDstTex = e->client->blurSrcTexture;
    for (uint8_t i = 0, horizontal = true; i < 10; i++, horizontal = !horizontal) {
        RenderTexture2D temp = blurSrcTex;
        blurSrcTex = blurDstTex;
        blurDstTex = temp;

        Vector2 blurDir;
        if (horizontal) {
            blurDir = (Vector2){1.0f / e->client->width, 0.0f};
        } else {
            blurDir = (Vector2){0.0f, 1.0f / e->client->height};
        }

        BeginTextureMode(blurDstTex);
        BeginShaderMode(e->client->blurShader);
        SetShaderValue(e->client->blurShader, e->client->blurShaderDirLoc, &blurDir, SHADER_UNIFORM_VEC2);
        DrawTextureRec(blurSrcTex.texture, (Rectangle){0.0f, 0.0f, e->client->width, -e->client->height}, Vector2Zero(), WHITE);
        EndShaderMode();
        EndTextureMode();
    }

    // bloom
    BeginTextureMode(dstTex);
    BeginShaderMode(e->client->bloomShader);

    SetShaderValue(e->client->bloomShader, e->client->bloomIntensityLoc, &bloomIntensity, SHADER_UNIFORM_FLOAT);
    SetShaderValueTexture(e->client->bloomShader, e->client->bloomTexColorLoc, srcTex.texture);
    SetShaderValueTexture(e->client->bloomShader, e->client->bloomTexBloomBlurLoc, e->client->blurDstTexture.texture);
    DrawTextureRec(e->client->blurDstTexture.texture, (Rectangle){0.0f, 0.0f, e->client->width, -e->client->height}, Vector2Zero(), WHITE);

    EndShaderMode();
    EndTextureMode();
}

void minimalStepEnv(env *e) {
    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead || drone->shield == NULL) {
            continue;
        }

        // update shield velocity if its active
        b2Body_SetLinearVelocity(drone->shield->bodyID, b2Body_GetLinearVelocity(drone->bodyID));
    };

    b2World_Step(e->worldID, e->deltaTime, e->box2dSubSteps);

    handleBodyMoveEvents(e);
    handleContactEvents(e);
    handleSensorEvents(e);

    projectilesStep(e);

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }
        droneStep(e, drone);
    }
}

void _renderEnv(env *e, const bool starting, const bool ending, const int8_t winner, const int8_t winningTeam) {
    if (ending) {
        minimalStepEnv(e);
    }

    // UpdateCamera(&e->client->camera3D, CAMERA_ORBITAL);

    updateCamera(e);

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        const droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            // TODO: is there a better way to do this?
            float gridPos[2] = {-1000, -1000};
            SetShaderValue(e->client->gridShader, e->client->gridShaderPosLoc[drone->idx], gridPos, SHADER_UNIFORM_VEC2);
            continue;
        }

        float gridPos[2] = {drone->pos.x, drone->pos.y};
        SetShaderValue(e->client->gridShader, e->client->gridShaderPosLoc[drone->idx], gridPos, SHADER_UNIFORM_VEC2);
        const Color droneColor = getDroneColor(drone->idx);
        float gridColor[4] = {droneColor.r, droneColor.g, droneColor.b, droneColor.a};
        SetShaderValue(e->client->gridShader, e->client->gridShaderColorLoc[drone->idx], gridColor, SHADER_UNIFORM_VEC4);
    }

    // apply bloom to parts of drones
    BeginTextureMode(e->client->droneRawTex);
    ClearBackground(BLACK);
    BeginMode3D(e->client->camera->camera3D);

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        const droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }

        // light up the laser aim guide if the drone's weapon is fully charged
        if (drone->weaponInfo->charge != 0.0f && drone->weaponCharge == drone->weaponInfo->charge) {
            renderDroneAimGuide(e, drone);
        }

        renderDroneLight(drone);
    }

    EndMode3D();
    EndTextureMode();

    applyBloom(e, e->client->droneRawTex, e->client->droneBloomTex, 2.0f);

    // apply bloom to projectiles
    BeginTextureMode(e->client->projRawTex);
    ClearBackground(BLACK);
    BeginMode3D(e->client->camera->camera3D);

    BeginBlendMode(BLEND_ALPHA);
    for (size_t i = 0; i < cc_array_size(e->projectiles); i++) {
        const projectileEntity *projectile = safe_array_get_at(e->projectiles, i);
        renderProjectileTrail(projectile);
    }
    EndBlendMode();

    for (size_t i = 0; i < cc_array_size(e->projectiles); i++) {
        const projectileEntity *projectile = safe_array_get_at(e->projectiles, i);
        renderProjectile(projectile);
    }

    EndMode3D();
    EndTextureMode();

    applyBloom(e, e->client->projRawTex, e->client->projBloomTex, 3.0f);

    BeginDrawing();
    ClearBackground(BLACK);

#ifndef __EMSCRIPTEN__
    DrawFPS(e->client->scale, e->client->scale);
#endif

    BeginMode3D(e->client->camera->camera3D);

    // TODO: fix for maps with different rows and columns
    // draw a thicker grid below
    float y = (-3.0f * WALL_THICKNESS) - 1.0f;
    Color color = (Color){.r = 94, .g = 59, .b = 136, .a = 128};
    for (int i = 0; i < e->map->columns; i++) {
        const float d = WALL_THICKNESS * e->map->columns;
        const Vector2 start = {.x = -d / 2.0f, .y = WALL_THICKNESS * i - d / 2.0f};
        const Vector2 end = {.x = d / 2.0f, .y = WALL_THICKNESS * i - d / 2.0f};
        const Vector3 pos = {.x = start.x, .y = y, .z = start.y};

        rlPushMatrix();
        rlTranslatef(pos.x, pos.y, pos.z);
        rlRotatef(-90.0f, 0.0f, 0.0f, 1.0f);

        DrawCylinder(Vector3Zero(), 0.1f, 0.1f, Vector2Distance(start, end), 1, color);

        rlPopMatrix();
    }
    for (int i = 0; i < e->map->rows; i++) {
        const float d = WALL_THICKNESS * e->map->rows;
        const Vector2 start = {.x = WALL_THICKNESS * i - d / 2.0f, .y = -d / 2.0f};
        const Vector2 end = {.x = WALL_THICKNESS * i - d / 2.0f, .y = d / 2.0f};
        const Vector3 pos = {.x = start.x, .y = y, .z = start.y};

        rlPushMatrix();
        rlTranslatef(pos.x, pos.y, pos.z);
        rlRotatef(90.0f, 1.0f, 0.0f, 0.0f);

        DrawCylinder(Vector3Zero(), 0.1f, 0.1f, Vector2Distance(start, end), 1, color);

        rlPopMatrix();
    }

    // render smaller higher grid
    BeginBlendMode(BLEND_ALPHA);
    BeginShaderMode(e->client->gridShader);
    y = -1.0f;
    color = PUFF_BACKGROUND;
    for (int i = 0; i < 2 * e->map->columns; i++) {
        const float d = WALL_THICKNESS * e->map->columns;
        const Vector3 start = {.x = -d / 2.0f, .y = y, .z = WALL_THICKNESS / 2.0f * i - d / 2.0f};
        const Vector3 end = {.x = (d - WALL_THICKNESS / 2.0f) / 2.0f, .y = y, .z = WALL_THICKNESS / 2.0f * i - d / 2.0f};

        DrawLine3D(start, end, color);
    }
    for (int i = 0; i < 2 * e->map->rows; i++) {
        float d = WALL_THICKNESS * e->map->rows;
        const Vector3 start = {.x = WALL_THICKNESS * i / 2.0f - d / 2.0f, .y = y, .z = -d / 2.0f};
        const Vector3 end = {.x = WALL_THICKNESS * i / 2.0f - d / 2.0f, .y = y, .z = (d - WALL_THICKNESS / 2.0f) / 2.0f};

        DrawLine3D(start, end, color);
    }
    EndShaderMode();
    EndBlendMode();

    for (size_t i = 0; i < cc_array_size(e->pickups); i++) {
        const weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
        renderWeaponPickup(e, pickup);
    }

    renderBrakeTrails(e);
    renderDronePieces(e);

    EndMode3D();

    BeginBlendMode(BLEND_ADDITIVE);
    DrawTextureRec(e->client->droneBloomTex.texture, (Rectangle){0.0f, 0.0f, e->client->width, -e->client->height}, Vector2Zero(), WHITE);
    EndBlendMode();

    BeginMode3D(e->client->camera->camera3D);

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }
        renderDroneTrail(drone);
    }

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }
        renderDroneGuides(e, drone, ending);
    }
    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        const droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }
        renderDrone(drone);
    }

    for (size_t i = 0; i < cc_array_size(e->walls); i++) {
        const wallEntity *wall = safe_array_get_at(e->walls, i);
        renderWall(e, wall);
    }

    for (size_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
        const wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
        renderWall(e, wall);
    }

    renderExplosions(e);
    EndMode3D();

    BeginBlendMode(BLEND_ADDITIVE);
    DrawTextureRec(e->client->projRawTex.texture, (Rectangle){0.0f, 0.0f, e->client->width, -e->client->height}, Vector2Zero(), WHITE);
    DrawTextureRec(e->client->projBloomTex.texture, (Rectangle){0.0f, 0.0f, e->client->width, -e->client->height}, Vector2Zero(), WHITE);
    EndBlendMode();

    BeginMode2D(e->client->camera->camera2D);

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }
        renderDroneRespawnGuides(e, drone);
        renderDroneUI(drone);
    }

    if (!b2VecEqual(e->debugPoint, b2Vec2_zero)) {
        const Vector2 pos = {.x = e->debugPoint.x, .y = e->debugPoint.y};
        DrawCircleV(pos, DRONE_RADIUS * 0.5f, WHITE);
    }

    EndMode2D();

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }
        renderDroneAmmo(e, drone);
    }

    renderUI(e, starting);

    if (starting || ending) {
        renderBannerText(e, starting, winner, winningTeam);
    }

    EndDrawing();
}

void renderWait(env *e, const bool starting, const bool ending, const int8_t winner, const int8_t winningTeam, const float time) {
#ifdef __EMSCRIPTEN__
    const double startTime = emscripten_get_now();
    while (time > (emscripten_get_now() - startTime) / 1000.0) {
        _renderEnv(e, starting, ending, winner, winningTeam);
        emscripten_sleep(e->deltaTime * 1000.0);
    }
#else
    for (uint16_t i = 0; i < (uint16_t)(time * e->frameRate); i++) {
        _renderEnv(e, starting, ending, winner, winningTeam);
    }
#endif
}

void renderEnv(env *e, const bool starting, const bool ending, const int8_t winner, const int8_t winningTeam) {
    if (starting) {
        renderWait(e, starting, ending, winner, winningTeam, START_READY_TIME);
    } else if (ending) {
        renderWait(e, starting, ending, winner, winningTeam, END_WAIT_TIME);
    } else {
        _renderEnv(e, starting, ending, winner, winningTeam);
    }
}

#endif
