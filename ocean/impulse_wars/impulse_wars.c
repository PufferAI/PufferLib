#include "env.h"
#include "render.h"

#ifdef __EMSCRIPTEN__
void emscriptenStep(void *e) {
    stepEnv((iwEnv *)e);
    return;
}
#endif

int main(void) {
    const int NUM_DRONES = 2;

    iwEnv *e = fastCalloc(1, sizeof(iwEnv));

    posix_memalign((void **)&e->observations, sizeof(void *), alignedSize(NUM_DRONES * obsBytes(NUM_DRONES), sizeof(float)));
    e->rewards = fastCalloc(NUM_DRONES, sizeof(float));
    e->actions = fastCalloc(NUM_DRONES * CONTINUOUS_ACTION_SIZE, sizeof(float));
    e->masks = fastCalloc(NUM_DRONES, sizeof(uint8_t));
    e->terminals = fastCalloc(NUM_DRONES, sizeof(uint8_t));
    e->truncations = fastCalloc(NUM_DRONES, sizeof(uint8_t));

    rayClient *client = createRayClient();
    e->client = client;

    initEnv(e, NUM_DRONES, 0, -1, time(NULL), false, false, false, false);
    initMaps(e);
    setupEnv(e);
    // e->humanInput = true;

#ifdef __EMSCRIPTEN__
    lastFrameTime = emscripten_get_now();
    emscripten_set_main_loop_arg(emscriptenStep, e, 0, true);
#else
    while (!WindowShouldClose()) {
        stepEnv(e);
    }

    destroyEnv(e);
    destroyMaps();
    free(e->observations);
    fastFree(e->actions);
    fastFree(e->rewards);
    fastFree(e->masks);
    fastFree(e->terminals);
    fastFree(e->truncations);
    fastFree(e);
    destroyRayClient(client);
#endif
    return 0;
}
