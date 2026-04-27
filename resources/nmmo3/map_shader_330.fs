#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

// Input uniform values
uniform sampler2D terrain;
uniform sampler2D texture_tiles;    // Tile sprite sheet texture
uniform vec4 colDiffuse;
uniform vec3 resolution;
uniform vec4 mouse;
uniform float time;
uniform float camera_x;
uniform float camera_y;
uniform float map_width;
uniform float map_height;

// Output fragment color
out vec4 outputColor;

float TILE_SIZE = 64.0;

// Number of tiles per row in the sprite sheet
const int TILES_PER_ROW = 64; // Adjust this based on your sprite sheet layout

void main()
{
    float ts = TILE_SIZE * resolution.z;

    // Get the screen pixel coordinates
    vec2 pixelPos = gl_FragCoord.xy;

    float x_offset = camera_x/64.0 + pixelPos.x/ts - resolution.x/ts/2.0;
    float y_offset = camera_y/64.0 - pixelPos.y/ts + resolution.y/ts/2.0;

    float x_floor = floor(x_offset);
    float y_floor = floor(y_offset);

    float x_frac = x_offset - x_floor;
    float y_frac = y_offset - y_floor;

    // TODO: This is the env size
    vec2 uv = vec2(
        x_floor/map_width,
        y_floor/map_height
    );
    vec2 tile_rg = texture(terrain, uv).rg;

    int tile_high_byte = int(tile_rg.r*255.0);
    int tile_low_byte = int(tile_rg.g*255.0);

    int tile = tile_high_byte*64 + tile_low_byte;
    if (tile >= 240 && tile < 240+4*4*4*4) {
        tile += int(3.9*time);
    }

    tile_high_byte = int(tile/64.0);
    tile_low_byte = int(tile%64);
 
    vec2 tile_uv = vec2(
        tile_low_byte/64.0 + x_frac/64.0,
        tile_high_byte/64.0 + y_frac/64.0
    );

    outputColor = texture(texture_tiles, tile_uv);
}

