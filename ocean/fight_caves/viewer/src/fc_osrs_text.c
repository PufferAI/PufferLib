#include "fc_osrs_text.h"

#include "fc_asset_raylib.h"

#include <stddef.h>
#include <string.h>

#define FC_OSRS_FONT_ASSET "data/fonts/p11_full.png"
#define FC_OSRS_FONT_CELL_SIZE 20
#define FC_OSRS_FONT_COLUMNS 16
#define FC_OSRS_FONT_GLYPHS 256

typedef struct {
    Rectangle source;
    int offset_x;
    int offset_y;
    int advance;
    int drawable;
} FcOsrsGlyph;

static Texture2D g_font_texture;
static FcOsrsGlyph g_glyphs[FC_OSRS_FONT_GLYPHS];
static int g_font_height;
static int g_font_ready;

static int is_magenta(Color color) {
    return color.r == 255 && color.g == 0 && color.b == 255;
}

static int glyph_pixel_set(const Color* pixels, int image_width,
                           int cell_x, int cell_y, int x, int y) {
    return !is_magenta(
        pixels[(cell_y + y) * image_width + cell_x + x]);
}

static void build_glyph_metrics(const Color* pixels, int image_width,
                                int codepoint) {
    FcOsrsGlyph* glyph = &g_glyphs[codepoint];
    int cell_x = (codepoint % FC_OSRS_FONT_COLUMNS) *
        FC_OSRS_FONT_CELL_SIZE;
    int cell_y = (codepoint / FC_OSRS_FONT_COLUMNS) *
        FC_OSRS_FONT_CELL_SIZE;
    int left = FC_OSRS_FONT_CELL_SIZE;
    int top = FC_OSRS_FONT_CELL_SIZE;
    int right = -1;
    int bottom = -1;

    for (int y = 0; y < FC_OSRS_FONT_CELL_SIZE; y++) {
        for (int x = 0; x < FC_OSRS_FONT_CELL_SIZE; x++) {
            if (!glyph_pixel_set(pixels, image_width,
                                 cell_x, cell_y, x, y)) {
                continue;
            }
            if (x < left) left = x;
            if (x > right) right = x;
            if (y < top) top = y;
            if (y > bottom) bottom = y;
        }
    }

    if (right < left || bottom < top) {
        glyph->advance = 0;
        return;
    }

    int width = right - left + 1;
    int height = bottom - top + 1;
    glyph->source = (Rectangle){
        (float)(cell_x + left), (float)(cell_y + top),
        (float)width, (float)height,
    };
    glyph->offset_x = 1;
    glyph->offset_y = top;
    glyph->advance = width + 2;
    glyph->drawable = codepoint >= 33 && codepoint != 127;

    /* Exact cache-client PixFont advance calculation. */
    int edge_start = height / 7;
    int edge_threshold = height / 7;
    int edge_pixels = 0;
    for (int y = edge_start; y < height; y++) {
        edge_pixels += glyph_pixel_set(
            pixels, image_width, cell_x, cell_y,
            left, top + y);
    }
    if (edge_pixels <= edge_threshold) {
        glyph->advance--;
        glyph->offset_x = 0;
    }

    edge_pixels = 0;
    for (int y = edge_start; y < height; y++) {
        edge_pixels += glyph_pixel_set(
            pixels, image_width, cell_x, cell_y,
            right, top + y);
    }
    if (edge_pixels <= edge_threshold)
        glyph->advance--;

    if (codepoint < 128 && height > g_font_height)
        g_font_height = height;
}

void fc_osrs_text_shutdown(void) {
    if (g_font_texture.id != 0)
        UnloadTexture(g_font_texture);
    g_font_texture = (Texture2D){0};
    memset(g_glyphs, 0, sizeof(g_glyphs));
    g_font_height = 0;
    g_font_ready = 0;
}

int fc_osrs_text_init(void) {
    fc_osrs_text_shutdown();

    Image image = fc_load_image_asset(FC_OSRS_FONT_ASSET);
    if (!image.data || image.width != 320 || image.height != 320) {
        if (image.data)
            UnloadImage(image);
        return 0;
    }
    ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Color* pixels = (Color*)image.data;

    for (int codepoint = 0;
         codepoint < FC_OSRS_FONT_GLYPHS; codepoint++) {
        build_glyph_metrics(pixels, image.width, codepoint);
    }
    if (g_font_height <= 0) {
        UnloadImage(image);
        fc_osrs_text_shutdown();
        return 0;
    }
    /* Native chat uses lowercase-i width for spaces and never blits the
     * placeholder sprite stored in the source sheet. */
    g_glyphs[' '].advance = g_glyphs['i'].advance;
    g_glyphs[' '].drawable = 0;

    for (int i = 0; i < image.width * image.height; i++) {
        if (is_magenta(pixels[i]))
            pixels[i] = BLANK;
        else
            pixels[i] = WHITE;
    }

    g_font_texture = LoadTextureFromImage(image);
    UnloadImage(image);
    if (g_font_texture.id == 0) {
        fc_osrs_text_shutdown();
        return 0;
    }
    SetTextureFilter(g_font_texture, TEXTURE_FILTER_POINT);
    g_font_ready = 1;
    return 1;
}

static unsigned char next_osrs_character(const unsigned char* text,
                                         size_t remaining,
                                         size_t* consumed) {
    *consumed = 1;
    if (text[0] < 0x80)
        return text[0];

    /* Cache fonts are 8-bit. Normalize the Unicode punctuation used by the
     * viewer diagnostics to the client's supported character repertoire. */
    if (remaining >= 3 && text[0] == 0xe2 && text[1] == 0x80 &&
        (text[2] == 0x93 || text[2] == 0x94)) {
        *consumed = 3;
        return '-';
    }
    if (remaining >= 3 && text[0] == 0xe2 && text[1] == 0x86 &&
        text[2] == 0x92) {
        *consumed = 3;
        return '>';
    }

    if ((text[0] & 0xe0) == 0xc0 && remaining >= 2)
        *consumed = 2;
    else if ((text[0] & 0xf0) == 0xe0 && remaining >= 3)
        *consumed = 3;
    else if ((text[0] & 0xf8) == 0xf0 && remaining >= 4)
        *consumed = 4;
    return '?';
}

void fc_osrs_draw_text(const char* text, int x, int y, int font_size,
                       Color color) {
    if (!text || !text[0] || font_size <= 0 || !g_font_ready)
        return;

    const unsigned char* bytes = (const unsigned char*)text;
    size_t remaining = strlen(text);
    while (remaining > 0) {
        size_t consumed = 1;
        unsigned char codepoint = next_osrs_character(
            bytes, remaining, &consumed);
        const FcOsrsGlyph* glyph = &g_glyphs[codepoint];
        if (glyph->drawable) {
            DrawTextureRec(g_font_texture, glyph->source,
                           (Vector2){(float)(x + glyph->offset_x),
                                     (float)(y + glyph->offset_y)},
                           color);
        }
        x += glyph->advance;
        bytes += consumed;
        remaining -= consumed;
    }
}

int fc_osrs_measure_text(const char* text, int font_size) {
    if (!text || !text[0] || font_size <= 0 || !g_font_ready)
        return 0;

    int width = 0;
    const unsigned char* bytes = (const unsigned char*)text;
    size_t remaining = strlen(text);
    while (remaining > 0) {
        size_t consumed = 1;
        unsigned char codepoint = next_osrs_character(
            bytes, remaining, &consumed);
        width += g_glyphs[codepoint].advance;
        bytes += consumed;
        remaining -= consumed;
    }
    return width;
}
