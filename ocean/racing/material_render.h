#pragma once
#include <string.h>
typedef struct RacingRenderBinding {
    unsigned base, detail, normal, flags;
    float normal_strength;
} RacingRenderBinding;

// Extra surface channels stored after each base material in PFVIS004.
typedef struct RacingMaterialDetail {
    unsigned normal, detail;
    float normal_strength, detail_scale, detail_strength, detail_world, roughness, metallic;
    float emission[3];
    int ground;
} RacingMaterialDetail;
static void racing_shader_maps(Shader shader) {
    shader.locs[SHADER_LOC_MAP_NORMAL] = GetShaderLocation(shader, "texture2");
    shader.locs[SHADER_LOC_MAP_METALNESS] = GetShaderLocation(shader, "texture1");
    shader.locs[SHADER_LOC_MAP_HEIGHT] = GetShaderLocation(shader, "shadowTexture");
}
static void racing_material_values(Shader shader, const RacingMaterialDetail *d) {
    static unsigned program = 0;
    static int locations[8];
    static float previous[6];
    static float previous_emission[3];
    static int previous_ground = -1, valid = 0;
    if (program != shader.id) {
        const char *names[] = {"normalStrength", "detailScale", "detailStrength",
                               "detailWorld",    "roughness",   "metallic"};
        for (int i = 0; i < 6; i++)
            locations[i] = GetShaderLocation(shader, names[i]);
        program = shader.id;
        locations[6] = GetShaderLocation(shader, "emission");
        locations[7] = GetShaderLocation(shader, "groundSurface");
        valid = 0;
    }
    float values[] = {d->normal_strength, d->detail_scale, d->detail_strength,
                      d->detail_world,    d->roughness,    d->metallic};
    for (int i = 0; i < 6; i++) {
        if (!valid || values[i] != previous[i])
            SetShaderValue(shader, locations[i], &values[i], SHADER_UNIFORM_FLOAT);
        previous[i] = values[i];
    }
    if (!valid || memcmp(previous_emission, d->emission, sizeof(previous_emission)))
        SetShaderValue(shader, locations[6], d->emission, SHADER_UNIFORM_VEC3);
    if (!valid || previous_ground != d->ground)
        SetShaderValue(shader, locations[7], &d->ground, SHADER_UNIFORM_INT);
    memcpy(previous_emission, d->emission, sizeof(previous_emission));
    previous_ground = d->ground;
    valid = 1;
}
