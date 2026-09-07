#include "ui_assets.h"
#include "fc_asset_raylib.h"
#include "fc_assets.h"

#include <stdio.h>
#include <string.h>

typedef struct RuneCUiAssetSpec {
    const char *name;
    const char *file;
    int required;
} RuneCUiAssetSpec;

#define UI_ASSET(name) { name, name ".png", 1 }
#define UI_OPTIONAL_ASSET(name, file) { name, file, 0 }

static const RuneCUiAssetSpec g_ui_asset_specs[] = {
    UI_ASSET("tradebacking_dark"),
    UI_ASSET("chatbox_bg"),
    UI_ASSET("main_stones_bottom"),
    UI_ASSET("side_background"),
    UI_ASSET("side_background_bottom"),
    UI_ASSET("side_background_left1"),
    UI_ASSET("side_background_left2"),
    UI_ASSET("side_background_right"),
    UI_ASSET("side_background_top"),
    UI_ASSET("side_stone_highlights_0"),
    UI_ASSET("side_stone_highlights_1"),
    UI_ASSET("side_stone_highlights_2"),
    UI_ASSET("side_stone_highlights_3"),
    UI_ASSET("side_stone_highlights_4"),
    UI_ASSET("osrs_stretch_side_topbottom_0"),
    UI_ASSET("osrs_stretch_side_topbottom_1"),
    UI_ASSET("osrs_stretch_side_columns_0"),
    UI_ASSET("osrs_stretch_side_columns_1"),
    UI_ASSET("compass"),
    UI_ASSET("mini_left"),
    UI_ASSET("mini_right"),
    UI_ASSET("mini_topright"),
    UI_ASSET("osrs_stretch_mapsurround"),
    UI_ASSET("resize_map_mask"),
    UI_ASSET("resize_compass_mask"),
    UI_ASSET("fixed_minimap_cover"),
    UI_ASSET("fixed_map_mask"),
    UI_ASSET("fixed_compass_mask"),
    UI_ASSET("fixed_map_clickmask"),
    UI_ASSET("mini_bottom"),
    UI_ASSET("compass_outline"),
    UI_ASSET("border_map_compass"),
    UI_ASSET("side_icon_combat"),
    UI_ASSET("side_icon_stats"),
    UI_ASSET("side_icon_quests"),
    UI_ASSET("side_icon_inventory"),
    UI_ASSET("side_icon_equipment"),
    UI_ASSET("side_icon_prayer"),
    UI_ASSET("side_icon_magic"),
    UI_ASSET("side_icon_clan"),
    UI_ASSET("side_icon_friends"),
    UI_ASSET("side_icon_options"),
    UI_ASSET("side_icon_logout"),
    UI_ASSET("side_icon_emotes"),
    UI_ASSET("side_icon_music"),
    UI_ASSET("side_icon_grouping"),
    UI_ASSET("side_icon_logout_modern"),
    UI_ASSET("options_icons_16"),
    UI_ASSET("options_icons_18"),
    UI_ASSET("options_icons_28"),
    UI_ASSET("whistle"),
    UI_ASSET("orb_frame_0"),
    UI_ASSET("orb_frame_1"),
    UI_ASSET("orb_frame_2"),
    UI_ASSET("orb_filler_0"),
    UI_ASSET("orb_filler_1"),
    UI_ASSET("orb_filler_4"),
    UI_ASSET("orb_filler_5"),
    UI_ASSET("orb_filler_6"),
    UI_ASSET("orb_filler_9"),
    UI_ASSET("orb_icon_0"),
    UI_ASSET("orb_icon_1"),
    UI_ASSET("orb_icon_2"),
    UI_ASSET("orb_icon_3"),
    UI_ASSET("orb_icon_6"),
    UI_ASSET("tli_button01_orbinfo_65x34_0"),
    UI_ASSET("tli_button01_orbinfo_65x34_1"),
    UI_ASSET("tli_button01_orbinfo_65x34_2"),
    UI_ASSET("ring_30"),
    UI_ASSET("worldmap_icon_0"),
    UI_ASSET("chat_tab_button_0"),
    UI_ASSET("wornicons_0"),
    UI_ASSET("wornicons_1"),
    UI_ASSET("wornicons_2"),
    UI_ASSET("wornicons_3"),
    UI_ASSET("wornicons_4"),
    UI_ASSET("wornicons_5"),
    UI_ASSET("wornicons_6"),
    UI_ASSET("wornicons_7"),
    UI_ASSET("wornicons_8"),
    UI_ASSET("wornicons_9"),
    UI_ASSET("wornicons_10"),
    UI_ASSET("wornicons_11"),
    UI_ASSET("miscgraphics_2"),
    UI_ASSET("miscgraphics_3"),
    UI_ASSET("combatboxes_0"),
    UI_ASSET("combatboxes_1"),
    UI_ASSET("combatboxes_2"),
    UI_ASSET("combatboxes_3"),
    UI_ASSET("combatboxes_large_0"),
    UI_ASSET("combatboxes_large_1"),
    UI_ASSET("combatboxes_very_large_0"),
    UI_ASSET("combatboxes_very_large_1"),
    UI_ASSET("combatboxes_special_attack"),
    UI_ASSET("combat_shield"),
    UI_ASSET("combaticons_0"),
    UI_ASSET("combaticons_1"),
    UI_ASSET("combaticons_2"),
    UI_ASSET("combaticons_3"),
    UI_ASSET("combaticons_4"),
    UI_ASSET("combaticons_5"),
    UI_ASSET("combaticons_6"),
    UI_ASSET("combaticons_7"),
    UI_ASSET("combaticons_8"),
    UI_ASSET("combaticons_9"),
    UI_ASSET("combaticons_10"),
    UI_ASSET("combaticons_11"),
    UI_ASSET("combaticons_12"),
    UI_ASSET("combaticons_13"),
    UI_ASSET("combaticons_14"),
    UI_ASSET("combaticons_15"),
    UI_ASSET("combaticons_16"),
    UI_ASSET("combaticons_17"),
    UI_ASSET("combaticons_18"),
    UI_ASSET("combaticons_19"),
    UI_ASSET("combaticons2_0"),
    UI_ASSET("combaticons2_1"),
    UI_ASSET("combaticons2_2"),
    UI_ASSET("combaticons2_3"),
    UI_ASSET("combaticons2_4"),
    UI_ASSET("combaticons2_5"),
    UI_ASSET("combaticons2_6"),
    UI_ASSET("combaticons2_7"),
    UI_ASSET("combaticons2_8"),
    UI_ASSET("combaticons2_9"),
    UI_ASSET("combaticons2_10"),
    UI_ASSET("combaticons2_11"),
    UI_ASSET("combaticons2_12"),
    UI_ASSET("combaticons2_13"),
    UI_ASSET("combaticons2_14"),
    UI_ASSET("combaticons2_15"),
    UI_ASSET("combaticons2_16"),
    UI_ASSET("combaticons2_17"),
    UI_ASSET("combaticons2_18"),
    UI_ASSET("combaticons2_19"),
    UI_ASSET("combaticons3_0"),
    UI_ASSET("combaticons3_1"),
    UI_ASSET("combaticons3_2"),
    UI_ASSET("combaticons3_3"),
    UI_ASSET("combaticons3_4"),
    UI_ASSET("combaticons3_5"),
    UI_ASSET("combaticons3_6"),
    UI_ASSET("combaticons3_7"),
    UI_ASSET("combaticons3_8"),
    UI_ASSET("combaticons3_9"),
    UI_ASSET("combaticons3_10"),
    UI_ASSET("combaticons3_11"),
    UI_ASSET("combaticons3_12"),
    UI_ASSET("combaticons3_13"),
    UI_ASSET("combaticons3_14"),
    UI_ASSET("combaticons3_15"),
    UI_ASSET("combaticons3_16"),
    UI_ASSET("combaticons3_17"),
    UI_ASSET("combaticons3_18"),
    UI_ASSET("combaticons3_19"),
    UI_ASSET("sideicons_interface_0"),
    UI_ASSET("sideicons_interface_1"),
    UI_ASSET("sideicons_interface_2"),
    UI_ASSET("sideicons_interface_3"),
    UI_ASSET("sideicons_interface_4"),
    UI_ASSET("sideicons_interface_5"),
    UI_ASSET("sideicons_interface_6"),
    UI_ASSET("sideicons_interface_7"),
    UI_ASSET("sideicons_interface_8"),
    UI_ASSET("sideicons_interface_9"),
    UI_ASSET("sideicons_interface_10"),
    UI_ASSET("sideicons_interface_11"),
    UI_ASSET("sideicons_interface_12"),
    UI_ASSET("sideicons_interface_13"),
    UI_ASSET("sideicons_interface_14"),
    UI_ASSET("sideicons_interface_15"),
    UI_ASSET("sideicons_interface_16"),
    UI_OPTIONAL_ASSET("item_995", "../items/item_995.png"),
    UI_OPTIONAL_ASSET("item_995_1", "../items/item_995_1.png"),
    UI_OPTIONAL_ASSET("item_995_2", "../items/item_995_2.png"),
    UI_OPTIONAL_ASSET("item_995_3", "../items/item_995_3.png"),
    UI_OPTIONAL_ASSET("item_995_4", "../items/item_995_4.png"),
    UI_OPTIONAL_ASSET("item_995_5", "../items/item_995_5.png"),
    UI_OPTIONAL_ASSET("item_995_25", "../items/item_995_25.png"),
    UI_OPTIONAL_ASSET("item_995_100", "../items/item_995_100.png"),
    UI_OPTIONAL_ASSET("item_995_250", "../items/item_995_250.png"),
    UI_OPTIONAL_ASSET("item_995_1000", "../items/item_995_1000.png"),
    UI_OPTIONAL_ASSET("item_995_10000", "../items/item_995_10000.png"),
    UI_OPTIONAL_ASSET("item_1038", "../items/item_1038.png"),
    UI_OPTIONAL_ASSET("item_1040", "../items/item_1040.png"),
    UI_OPTIONAL_ASSET("item_1042", "../items/item_1042.png"),
    UI_OPTIONAL_ASSET("item_1044", "../items/item_1044.png"),
    UI_OPTIONAL_ASSET("item_1046", "../items/item_1046.png"),
    UI_OPTIONAL_ASSET("item_1048", "../items/item_1048.png"),
    UI_OPTIONAL_ASSET("item_4151", "../items/item_4151.png"),
    UI_OPTIONAL_ASSET("item_554", "../items/item_554.png"),
    UI_OPTIONAL_ASSET("item_555", "../items/item_555.png"),
    UI_OPTIONAL_ASSET("item_556", "../items/item_556.png"),
    UI_OPTIONAL_ASSET("item_557", "../items/item_557.png"),
    UI_OPTIONAL_ASSET("item_558", "../items/item_558.png"),
    UI_OPTIONAL_ASSET("item_560", "../items/item_560.png"),
    UI_OPTIONAL_ASSET("item_562", "../items/item_562.png"),
    UI_OPTIONAL_ASSET("item_861", "../items/item_861.png"),
    UI_OPTIONAL_ASSET("item_892", "../items/item_892.png"),
    UI_OPTIONAL_ASSET("item_1381", "../items/item_1381.png"),
    UI_OPTIONAL_ASSET("item_10346", "../items/item_10346.png"),
    UI_OPTIONAL_ASSET("item_10348", "../items/item_10348.png"),
    UI_OPTIONAL_ASSET("item_10350", "../items/item_10350.png"),
    UI_OPTIONAL_ASSET("item_10352", "../items/item_10352.png"),
    UI_OPTIONAL_ASSET("item_11802", "../items/item_11802.png"),
    UI_OPTIONAL_ASSET("item_11832", "../items/item_11832.png"),
    UI_OPTIONAL_ASSET("item_11834", "../items/item_11834.png"),
    UI_OPTIONAL_ASSET("item_26382", "../items/item_26382.png"),
    UI_OPTIONAL_ASSET("item_26384", "../items/item_26384.png"),
    UI_OPTIONAL_ASSET("item_26386", "../items/item_26386.png"),
    UI_ASSET("skill_icon_0"),
    UI_ASSET("skill_icon_1"),
    UI_ASSET("skill_icon_2"),
    UI_ASSET("skill_icon_3"),
    UI_ASSET("skill_icon_4"),
    UI_ASSET("skill_icon_5"),
    UI_ASSET("skill_icon_6"),
    UI_ASSET("skill_icon_7"),
    UI_ASSET("skill_icon_8"),
    UI_ASSET("skill_icon_9"),
    UI_ASSET("skill_icon_10"),
    UI_ASSET("skill_icon_11"),
    UI_ASSET("skill_icon_12"),
    UI_ASSET("skill_icon_13"),
    UI_ASSET("skill_icon_14"),
    UI_ASSET("skill_icon_15"),
    UI_ASSET("skill_icon_16"),
    UI_ASSET("skill_icon_17"),
    UI_ASSET("skill_icon_18"),
    UI_ASSET("skill_icon_19"),
    UI_ASSET("skill_icon_20"),
    UI_ASSET("skill_icon_21"),
    UI_ASSET("skill_icon_22"),
    UI_ASSET("skill_icon_23"),
    UI_ASSET("prayeroff_0"),
    UI_ASSET("prayeroff_1"),
    UI_ASSET("prayeroff_2"),
    UI_ASSET("prayeroff_3"),
    UI_ASSET("prayeroff_4"),
    UI_ASSET("prayeroff_5"),
    UI_ASSET("prayeroff_6"),
    UI_ASSET("prayeroff_7"),
    UI_ASSET("prayeroff_8"),
    UI_ASSET("prayeroff_9"),
    UI_ASSET("prayeroff_10"),
    UI_ASSET("prayeroff_11"),
    UI_ASSET("prayeroff_12"),
    UI_ASSET("prayeroff_13"),
    UI_ASSET("prayeroff_14"),
    UI_ASSET("prayeroff_15"),
    UI_ASSET("prayeroff_16"),
    UI_ASSET("prayeroff_17"),
    UI_ASSET("prayeroff_18"),
    UI_ASSET("prayeroff_19"),
    UI_ASSET("prayeroff_20"),
    UI_ASSET("prayeroff_21"),
    UI_ASSET("prayeroff_22"),
    UI_ASSET("prayeroff_23"),
    UI_ASSET("prayeroff_24"),
    UI_ASSET("magicon_0"),
    UI_ASSET("magicon_1"),
    UI_ASSET("magicon_2"),
    UI_ASSET("magicon_3"),
    UI_ASSET("magicon_4"),
    UI_ASSET("magicon_5"),
    UI_ASSET("magicon_6"),
    UI_ASSET("magicon_7"),
    UI_ASSET("magicon_8"),
    UI_ASSET("magicon_9"),
    UI_ASSET("magicon_10"),
    UI_ASSET("magicon_11"),
    UI_ASSET("magicon_12"),
    UI_ASSET("magicon_13"),
    UI_ASSET("magicon_14"),
    UI_ASSET("magicon_15"),
    UI_ASSET("magicon_16"),
    UI_ASSET("magicon_17"),
    UI_ASSET("magicon_18"),
    UI_ASSET("magicon_19"),
    UI_ASSET("magicon_20"),
    UI_ASSET("magicon_21"),
    UI_ASSET("magicon_22"),
    UI_ASSET("magicon_23"),
    UI_ASSET("magicon_24"),
    UI_ASSET("standard_spell_on_0"),
    UI_ASSET("standard_spell_on_1"),
    UI_ASSET("standard_spell_on_2"),
    UI_ASSET("standard_spell_on_3"),
    UI_ASSET("standard_spell_on_4"),
    UI_ASSET("standard_spell_on_5"),
    UI_ASSET("standard_spell_on_6"),
    UI_ASSET("standard_spell_on_7"),
    UI_ASSET("standard_spell_on_8"),
    UI_ASSET("standard_spell_on_9"),
    UI_ASSET("standard_spell_on_10"),
    UI_ASSET("standard_spell_on_11"),
    UI_ASSET("standard_spell_on_12"),
    UI_ASSET("standard_spell_on_13"),
    UI_ASSET("standard_spell_on_14"),
    UI_ASSET("standard_spell_on_15"),
    UI_ASSET("standard_spell_on_16"),
    UI_ASSET("standard_spell_on_17"),
    UI_ASSET("standard_spell_on_18"),
    UI_ASSET("standard_spell_on_19"),
    UI_ASSET("standard_spell_on_20"),
    UI_ASSET("standard_spell_on_21"),
    UI_ASSET("standard_spell_on_22"),
    UI_ASSET("standard_spell_on_23"),
    UI_ASSET("standard_spell_on_24"),
    UI_ASSET("standard_spell_on_25"),
    UI_ASSET("standard_spell_on_26"),
    UI_ASSET("standard_spell_on_27"),
    UI_ASSET("standard_spell_on_28"),
    UI_ASSET("standard_spell_on_29"),
    UI_ASSET("standard_spell_on_30"),
    UI_ASSET("standard_spell_on_31"),
    UI_ASSET("standard_spell_on_32"),
    UI_ASSET("standard_spell_on_33"),
    UI_ASSET("standard_spell_on_34"),
    UI_ASSET("standard_spell_on_35"),
    UI_ASSET("standard_spell_on_36"),
    UI_ASSET("standard_spell_on_37"),
    UI_ASSET("standard_spell_on_38"),
    UI_ASSET("standard_spell_on_39"),
    UI_ASSET("standard_spell_on_40"),
    UI_ASSET("standard_spell_on_41"),
    UI_ASSET("standard_spell_on_42"),
    UI_ASSET("standard_spell_on_43"),
    UI_ASSET("standard_spell_on_44"),
    UI_ASSET("standard_spell_on_45"),
    UI_ASSET("standard_spell_on_46"),
    UI_ASSET("standard_spell_on_47"),
    UI_ASSET("standard_spell_on_48"),
    UI_ASSET("standard_spell_on_49"),
    UI_ASSET("standard_spell_on_50"),
    UI_ASSET("standard_spell_on_51"),
    UI_ASSET("standard_spell_on_52"),
    UI_ASSET("standard_spell_on_53"),
    UI_ASSET("standard_spell_on_54"),
    UI_ASSET("standard_spell_on_55"),
    UI_ASSET("standard_spell_on_56"),
    UI_ASSET("standard_spell_on_57"),
    UI_ASSET("standard_spell_on_58"),
    UI_ASSET("standard_spell_on_59"),
    UI_ASSET("standard_spell_on_60"),
    UI_ASSET("standard_spell_on_61"),
    UI_ASSET("standard_spell_on_62"),
    UI_ASSET("standard_spell_on_63"),
    UI_ASSET("standard_spell_on_64"),
    UI_ASSET("standard_spell_on_65"),
    UI_ASSET("standard_spell_on_66"),
    UI_ASSET("standard_spell_on_67"),
    UI_ASSET("standard_spell_on_68"),
    UI_ASSET("standard_spell_on_69"),
    UI_ASSET("standard_spell_on_70"),
    UI_ASSET("standard_spell_on_71"),
    UI_ASSET("standard_spell_on_72"),
    UI_ASSET("standard_spell_on_73"),
    UI_ASSET("standard_spell_on_74"),
    UI_ASSET("standard_spell_on_75"),
    UI_ASSET("standard_spell_on_76"),
    UI_ASSET("standard_spell_on_77"),
    UI_ASSET("standard_spell_on_78"),
    UI_ASSET("standard_spell_on_79"),
};

static int ui_asset_count(void) {
    return (int)(sizeof(g_ui_asset_specs) / sizeof(g_ui_asset_specs[0]));
}

static int find_asset_index(const char *name) {
    int count = ui_asset_count();
    for (int i = 0; i < count; i++) {
        if (strcmp(g_ui_asset_specs[i].name, name) == 0)
            return i;
    }
    return -1;
}

static Texture2D load_compass_texture(const char *path) {
    /* The OSRS client clips the rotating compass through sprite 1179 instead
     * of drawing sprite 169 as an opaque square. Pre-applying that circular
     * mask is equivalent because its shape is invariant under rotation. */
    Image compass = fc_load_image_asset(path);
    Image mask = fc_load_image_asset("data/sprites/ui/resize_compass_mask.png");
    Texture2D texture = {0};

    if (!compass.data || !mask.data) {
        if (compass.data) UnloadImage(compass);
        if (mask.data) UnloadImage(mask);
        return texture;
    }

    ImageResizeNN(&mask, compass.width, compass.height);
    ImageFormat(&compass, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    ImageFormat(&mask, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Color *compass_pixels = compass.data;
    const Color *mask_pixels = mask.data;
    int pixel_count = compass.width * compass.height;
    for (int i = 0; i < pixel_count; i++) {
        if (mask_pixels[i].a != 0)
            compass_pixels[i].a = 0;
    }

    texture = LoadTextureFromImage(compass);
    UnloadImage(mask);
    UnloadImage(compass);
    return texture;
}

void runec_ui_assets_load(RuneCUiAssets *assets) {
    memset(assets, 0, sizeof(*assets));

    int count = ui_asset_count();
    if (count > RUNEC_UI_ASSET_MAX)
        count = RUNEC_UI_ASSET_MAX;

    for (int i = 0; i < count; i++) {
        char path[256];
        snprintf(path, sizeof(path), "data/sprites/ui/%s", g_ui_asset_specs[i].file);
        if (g_ui_asset_specs[i].required)
            assets->required_count++;
        if (!fc_asset_exists(path)) {
            assets->missing_count++;
            if (g_ui_asset_specs[i].required) {
                assets->missing_required_count++;
                fprintf(stderr, "ui_assets: missing required sprite %s (%s)\n",
                        g_ui_asset_specs[i].name, path);
            } else {
                assets->missing_optional_count++;
            }
            continue;
        }
        Texture2D tex = strcmp(g_ui_asset_specs[i].name, "compass") == 0
            ? load_compass_texture(path)
            : fc_load_texture_asset(path);
        if (tex.id == 0) {
            assets->missing_count++;
            if (g_ui_asset_specs[i].required) {
                assets->missing_required_count++;
                fprintf(stderr, "ui_assets: failed to load required sprite %s (%s)\n",
                        g_ui_asset_specs[i].name, path);
            } else {
                assets->missing_optional_count++;
            }
            continue;
        }
        SetTextureFilter(tex, TEXTURE_FILTER_POINT);
        assets->textures[i] = tex;
        assets->loaded[i] = 1;
        assets->loaded_count++;
    }
    fprintf(stderr,
            "ui_assets: loaded %d/%d required sprites, %d optional sprites missing\n",
            assets->required_count - assets->missing_required_count,
            assets->required_count, assets->missing_optional_count);

    const char *font_paths[] = {
        "data/fonts/runescape.ttf",
    };
    for (int i = 0; i < (int)(sizeof(font_paths) / sizeof(font_paths[0])); i++) {
        const char *font_path = font_paths[i];
        if (!fc_asset_exists(font_path))
            continue;
        assets->font = fc_load_font_asset(font_path, 14);
        if (assets->font.texture.id != 0) {
            SetTextureFilter(assets->font.texture, TEXTURE_FILTER_POINT);
            assets->font_loaded = 1;
            fprintf(stderr, "ui_assets: loaded font %s\n", font_path);
            break;
        }
    }
    const char *small_font_paths[] = {
        "data/fonts/runescape_small.ttf",
    };
    for (int i = 0; i < (int)(sizeof(small_font_paths) / sizeof(small_font_paths[0])); i++) {
        const char *font_path = small_font_paths[i];
        if (!fc_asset_exists(font_path))
            continue;
        assets->small_font = fc_load_font_asset(font_path, 12);
        if (assets->small_font.texture.id != 0) {
            SetTextureFilter(assets->small_font.texture, TEXTURE_FILTER_POINT);
            assets->small_font_loaded = 1;
            fprintf(stderr, "ui_assets: loaded small font %s\n", font_path);
            break;
        }
    }
}

void runec_ui_assets_unload(RuneCUiAssets *assets) {
    int count = ui_asset_count();
    if (count > RUNEC_UI_ASSET_MAX)
        count = RUNEC_UI_ASSET_MAX;

    for (int i = 0; i < count; i++) {
        if (assets->loaded[i]) {
            UnloadTexture(assets->textures[i]);
            assets->loaded[i] = 0;
        }
    }
    if (assets->font_loaded) {
        UnloadFont(assets->font);
        assets->font_loaded = 0;
    }
    if (assets->small_font_loaded) {
        UnloadFont(assets->small_font);
        assets->small_font_loaded = 0;
    }
    assets->loaded_count = 0;
    assets->missing_count = 0;
    assets->required_count = 0;
    assets->missing_required_count = 0;
    assets->missing_optional_count = 0;
}

const Texture2D *runec_ui_asset(const RuneCUiAssets *assets, const char *name) {
    int index = find_asset_index(name);
    if (index < 0 || index >= RUNEC_UI_ASSET_MAX || !assets->loaded[index])
        return NULL;
    return &assets->textures[index];
}

int runec_ui_asset_ready(const RuneCUiAssets *assets, const char *name) {
    return runec_ui_asset(assets, name) != NULL;
}

void runec_ui_draw_asset(const RuneCUiAssets *assets, const char *name,
                         Rectangle dst, Color tint) {
    const Texture2D *tex = runec_ui_asset(assets, name);
    if (!tex)
        return;
    Rectangle src = {0, 0, (float)tex->width, (float)tex->height};
    DrawTexturePro(*tex, src, dst, (Vector2){0, 0}, 0.0f, tint);
}

Font runec_ui_font(const RuneCUiAssets *assets) {
    if (assets->font_loaded)
        return assets->font;
    if (assets->small_font_loaded)
        return assets->small_font;
    return (Font){0};
}

Font runec_ui_font_for_size(const RuneCUiAssets *assets, float size) {
    if (size <= 12.0f && assets->small_font_loaded)
        return assets->small_font;
    return runec_ui_font(assets);
}

void runec_ui_draw_text_shadow(const RuneCUiAssets *assets, const char *text,
                               float x, float y, float size, Color color) {
    if (!text || !text[0])
        return;
    if (size < 12.0f)
        size = 12.0f;
    Font font = runec_ui_font_for_size(assets, size);
    if (font.texture.id == 0)
        return;
    x = (float)((int)(x + 0.5f));
    y = (float)((int)(y + 0.5f));
    DrawTextEx(font, text, (Vector2){x + 1.0f, y + 1.0f}, size, 0.0f, BLACK);
    DrawTextEx(font, text, (Vector2){x, y}, size, 0.0f, color);
}
