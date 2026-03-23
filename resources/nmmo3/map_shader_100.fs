precision mediump float;

// Input uniforms (unchanged from original)
uniform sampler2D terrain;
uniform sampler2D texture_tiles;
uniform vec4 colDiffuse;
uniform vec3 resolution;
uniform vec4 mouse;
uniform float time;
uniform float camera_x;
uniform float camera_y;
uniform float map_width;
uniform float map_height;

// Constants
const float TILE_SIZE = 64.0;
const float TILES_PER_ROW = 64.0;

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
    
    // Environment size calculation
    vec2 uv = vec2(
        x_floor/map_width,
        y_floor/map_height
    );
    
    vec2 tile_rg = texture2D(terrain, uv).rg;
    float tile_high_byte = floor(tile_rg.r * 255.0);
    float tile_low_byte = floor(tile_rg.g * 255.0);
    float tile = tile_high_byte * 64.0 + tile_low_byte;
    
    // Handle animated tiles
    if (tile >= 240.0 && tile < (240.0 + 4.0*4.0*4.0*4.0)) {
        tile += floor(3.9 * time);
    }
    
    tile_high_byte = floor(tile/64.0);
    tile_low_byte = floor(mod(tile, 64.0));
    
    vec2 tile_uv = vec2(
        tile_low_byte/64.0 + x_frac/64.0,
        tile_high_byte/64.0 + y_frac/64.0
    );
    
    gl_FragColor = texture2D(texture_tiles, tile_uv);
}
