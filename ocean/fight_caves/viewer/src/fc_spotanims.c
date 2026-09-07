#include "fc_spotanims.h"
#include "fc_assets.h"
#include "fc_io.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SPOTANIM_MAGIC 0x544F5053u
#define SPOTANIM_VERSION 1u

SpotAnimSet *spotanims_load(const char *path) {
    FILE *f = fc_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "spotanims: can't open %s\n", path);
        return NULL;
    }

    uint32_t magic, version, count;
    if (!fc_read_exact(f, &magic, sizeof(magic), 1, path, "spotanim magic")
            || !fc_read_exact(f, &version, sizeof(version), 1, path,
                              "spotanim version")
            || !fc_read_exact(f, &count, sizeof(count), 1, path,
                              "spotanim count")
            || magic != SPOTANIM_MAGIC || version != SPOTANIM_VERSION) {
        fc_asset_close(f);
        return NULL;
    }

    SpotAnimSet *set = (SpotAnimSet *)calloc(1, sizeof(*set));
    if (!set) {
        fc_asset_close(f);
        return NULL;
    }
    set->defs = (SpotAnimDef *)calloc(count, sizeof(*set->defs));
    if (count > 0 && !set->defs) {
        fc_asset_close(f);
        spotanims_free(set);
        return NULL;
    }
    set->count = (int)count;
    for (uint32_t i = 0; i < count; i++) {
        if (!fc_read_exact(f, &set->defs[i], sizeof(set->defs[i]), 1, path,
                           "spotanim row")) {
            fc_asset_close(f);
            spotanims_free(set);
            return NULL;
        }
    }
    fc_asset_close(f);
    set->loaded = 1;
    fprintf(stderr, "spotanims: loaded %d from %s\n", set->count, path);
    return set;
}

const SpotAnimDef *spotanim_find(const SpotAnimSet *set, int id) {
    if (!set || !set->loaded || id < 0) return NULL;
    for (int i = 0; i < set->count; i++) {
        if ((int)set->defs[i].id == id) return &set->defs[i];
    }
    return NULL;
}

void spotanims_free(SpotAnimSet *set) {
    if (!set) return;
    free(set->defs);
    free(set);
}
