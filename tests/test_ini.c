// Checks puf_ini_load_env resolves config from PUFFER_ROOT / PUFFER_ENV_DIR,
// not the working directory. Driven by tests/test_build.sh, which writes the
// fixture inis and compiles this with the two defines pointing at them.
#include <assert.h>
#include <stdio.h>
#include "ini.h"

int main(void) {
    Ini ini = {0};
    // myenv.ini lives in PUFFER_ENV_DIR (an out-of-tree env).
    puf_ini_load_env(&ini, "myenv", 0, NULL);
    assert(puf_ini_get(&ini, "env", "size") == 3);
    assert(puf_ini_get(&ini, "env", "only_in_default") == 7);
    puf_ini_free(&ini);

    // oceanenv.ini lives only in PUFFER_ROOT/config (an Ocean env).
    puf_ini_load_env(&ini, "oceanenv", 0, NULL);
    assert(puf_ini_get(&ini, "env", "size") == 2);
    puf_ini_free(&ini);
    printf("test_ini: ok\n");
    return 0;
}
