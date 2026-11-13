// Checks puf_resource prefers PUFFER_ENV_DIR/resources, falls back to
// PUFFER_ROOT/resources, and keeps eight results alive at once. Driven by
// tests/test_build.sh, which writes the fixture files and passes the defines.
#include <assert.h>
#include <stdio.h>
#include <string.h>
typedef unsigned char obs_t;
#include "pufferenv.h"

int main(void) {
    // own.png exists only in the env's resources/.
    assert(strcmp(puf_resource("own.png"), PUFFER_ENV_DIR "/resources/own.png") == 0);
    // shared/x.png exists only under the PufferLib checkout.
    assert(strcmp(puf_resource("shared/x.png"), PUFFER_ROOT "/resources/shared/x.png") == 0);
    // Missing files resolve to the checkout too, so loaders report that path.
    assert(strcmp(puf_resource("missing.png"), PUFFER_ROOT "/resources/missing.png") == 0);
    // printf-style, so callers do not need TextFormat.
    assert(strcmp(puf_resource("shared/%s_%d.png", "x", 3),
        PUFFER_ROOT "/resources/shared/x_3.png") == 0);
    // Eight results stay valid at once (a shader needs two per call).
    const char* held[8];
    char want[256];
    for (int i = 0; i < 8; i++) {
        held[i] = puf_resource("%d.png", i);
    }
    for (int i = 0; i < 8; i++) {
        snprintf(want, sizeof(want), PUFFER_ROOT "/resources/%d.png", i);
        assert(strcmp(held[i], want) == 0);
    }
    printf("test_resource: ok\n");
    return 0;
}
