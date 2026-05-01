/**
 * @file test_metal_const_ring.mm
 * @brief Tests constant-ring reservation bounds.
 */

#include "src/metal_platform.h"

int main(void) {
    NSUInteger next_offset = 0;

    if (!mtl_const_ring_reserve_range(0, 1, &next_offset)) return 1;
    if (next_offset != 16) return 2;

    if (!mtl_const_ring_reserve_range(MTL_CONST_RING_SIZE - 16, 16, &next_offset)) return 3;
    if (next_offset != MTL_CONST_RING_SIZE) return 4;

    if (mtl_const_ring_reserve_range(MTL_CONST_RING_SIZE - 16, 17, &next_offset)) return 5;

    if (mtl_parse_int_config_value("total_agents", 0.0) != 0) return 6;
    if (mtl_parse_int_config_value("total_agents", 128.0) != 128) return 7;

    if (mtl_parse_int_config_value("horizon", 4.5) != 5) return 8;
    if (mtl_parse_int_config_value("horizon", 4.4) != 4) return 81;

    bool rejected_nondivisible = false;
    try {
        mtl_validate_divisible_config_values("total_agents", 10, "num_buffers", 3);
    } catch (const std::invalid_argument&) {
        rejected_nondivisible = true;
    }
    if (!rejected_nondivisible) return 9;

    return 0;
}
