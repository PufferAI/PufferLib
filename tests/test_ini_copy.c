// Build: cc -std=c11 -I src tests/test_ini_copy.c -o test_ini_copy

#include <assert.h>
#include <string.h>

#include "ini.h"

int main(void) {
    Ini original = {0};
    Ini copy = {0};

    puf_ini_set(puf_ini_section(&original, "base", 1),
        "load_model_path", "train.pt");
    puf_ini_set(puf_ini_section(&original, "train", 1), "horizon", "64");
    puf_ini_set(puf_ini_section(&original, "env", 1), "weights", "1,2,3");

    puf_ini_copy(&copy, &original);

    assert(copy.sections != original.sections);
    assert(puf_ini_get(&copy, "train", "horizon") == 64);
    assert(strcmp(puf_ini_get_str(&copy, "base", "load_model_path"), "train.pt") == 0);
    assert(puf_ini_section(&copy, "base", 0)->items[0].str
        != puf_ini_section(&original, "base", 0)->items[0].str);
    assert(puf_ini_section(&copy, "env", 0)->items[0].values
        != puf_ini_section(&original, "env", 0)->items[0].values);

    puf_ini_put(&copy, "train.horizon", "1");
    puf_ini_put(&copy, "base.load_model_path", "eval.pt");
    assert(puf_ini_get(&original, "train", "horizon") == 64);
    assert(strcmp(puf_ini_get_str(&original, "base", "load_model_path"), "train.pt") == 0);

    puf_ini_free(&copy);
    assert(puf_ini_get(&original, "train", "horizon") == 64);
    puf_ini_free(&original);
    return 0;
}
