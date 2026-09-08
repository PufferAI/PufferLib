#include "ocean/osrs/encounters/encounter_inferno.h"

int main(void) {
    CollisionMap* map = collision_map_load("inferno.cmap");
    if (!map) return 1;

    InfernoContext ctx;
    inf_init_context_typed(&ctx);
    ctx.collision_map = map;
    ctx.world_offset_x = 2246;
    ctx.world_offset_y = 5315;
    inf_finalize_route_topology(&ctx);
    if (!osrs_asset_exists(INF_ROUTE_BAKE_PATH)) {
        fprintf(stderr, "bake_inferno_route: %s was not written\n",
            osrs_asset_path(INF_ROUTE_BAKE_PATH));
        return 1;
    }
    return 0;
}
