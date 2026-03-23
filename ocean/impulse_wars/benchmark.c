#include "env.h"

void randActions(iwEnv *e) {
    // e->lastRandState = e->randState;
    uint8_t actionOffset = 0;
    for (uint8_t i = 0; i < e->numDrones; i++) {
        e->actions[actionOffset + 0] = randFloat(&e->randState, -1.0f, 1.0f);
        e->actions[actionOffset + 1] = randFloat(&e->randState, -1.0f, 1.0f);
        e->actions[actionOffset + 2] = randFloat(&e->randState, -1.0f, 1.0f);
        e->actions[actionOffset + 3] = randFloat(&e->randState, -1.0f, 1.0f);
        e->actions[actionOffset + 4] = randFloat(&e->randState, -1.0f, 1.0f);
        e->actions[actionOffset + 5] = randFloat(&e->randState, -1.0f, 1.0f);
        e->actions[actionOffset + 6] = randFloat(&e->randState, -1.0f, 1.0f);

        actionOffset += CONTINUOUS_ACTION_SIZE;
    }
}

void perfTest(const uint32_t numSteps) {
    const uint8_t NUM_DRONES = 2;

    iwEnv *e = fastCalloc(1, sizeof(iwEnv));

    posix_memalign((void **)&e->observations, sizeof(void *), alignedSize(NUM_DRONES * obsBytes(NUM_DRONES), sizeof(float)));
    e->rewards = fastCalloc(NUM_DRONES, sizeof(float));
    e->actions = fastCalloc(NUM_DRONES * CONTINUOUS_ACTION_SIZE, sizeof(float));
    e->masks = fastCalloc(NUM_DRONES, sizeof(uint8_t));
    e->terminals = fastCalloc(NUM_DRONES, sizeof(uint8_t));
    e->truncations = fastCalloc(NUM_DRONES, sizeof(uint8_t));

    // rayClient *client = createRayClient();
    // e->client = client;

    uint64_t seed = time(NULL);
    printf("seed: %lu\n", seed);
    initEnv(e, NUM_DRONES, 0, -1, seed, false, false, true, false);
    initMaps(e);

    // randActions(e);
    setupEnv(e);
    stepEnv(e);

    uint32_t steps = 0;
    while (steps != numSteps) {
        // randActions(e);
        stepEnv(e);
        steps++;
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
}

int main(void) {
    perfTest(2500000);
    return 0;
}
