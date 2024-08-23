#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;       // Fragment input: vertex attribute: texture coordinates
in vec4 fragColor;          // Fragment input: vertex attribute: color 

// Input uniform values
uniform sampler2D texture0; // Fragment input: texture
uniform sampler2D texture1; // Fragment input: texture
uniform vec4 colDiffuse;    // Fragment input: tint color normalized [0.0f..1.0f]

uniform vec3 resolution;    // Fragment input: .xy texture resolution in pixels, .z scale factor
uniform vec4 mouse;         // Fragment input: .xy mouse position on texture in pixels, .z mouse LMB down, .w mouse RMB down
uniform float time;         // Fragment input: elapsed time in seconds since program started
uniform float camera_x;
uniform float camera_y;

// Output fragment color
out vec4 outputColor;       // Fragment output: pixel color

// Fixed scale factor
const float SCALE_FACTOR = 16.0;

// Glow parameter (number of border pixels to alter)
const float GLOW = 4.0;

const float DIST = GLOW / SCALE_FACTOR;

float rand(vec2 co){
  return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{
    //float CONVERT_X = 255.0 / textureSize(texture0, 0).x / SCALE_FACTOR;
    //float CONVERT_Y = 255.0 / textureSize(texture0, 0).y / SCALE_FACTOR;

    //float CONVERT_X = 41.0 / 255.0;
    //float CONVERT_Y = 23.0 / 255.0;

    //float convert_x = 41.0 / 128.0;
    //float convert_y = 23.0 / 128.0;

    float CONVERT_X = textureSize(texture0, 0).x / 32 / 128.0;
    float CONVERT_Y = textureSize(texture0, 0).y / 32 / 128.0;

    // Calculate the scaled texture coordinates
    //vec2 scaledTexCoord = vec2(CONVERT_X*fragTexCoord.x, CONVERT_Y*fragTexCoord.y);
    vec2 scaledTexCoord = vec2(CONVERT_X*fragTexCoord.x + camera_x, CONVERT_Y*fragTexCoord.y + camera_y);
    float dist = DIST / textureSize(texture1, 0).x;

    // Calculate the scaled texture coordinates
    //vec2 scaledTexCoord = vec2(fragTexCoord.x/SCALE_FACTOR + camera_x, fragTexCoord.y/SCALE_FACTOR + camera_y);
    //float dist = DIST / textureSize(texture1, 0).x;

	float borderColor = 0.8 + 0.2*sin(time);

    // Texel color fetching from texture sampler with scaled coordinates
    vec4 texelColor = texture(texture1, scaledTexCoord) * colDiffuse * fragColor;
    vec4 texelRight = texture(texture1, vec2(scaledTexCoord.x + dist, scaledTexCoord.y)) * colDiffuse * fragColor;
    vec4 texelLeft = texture(texture1, vec2(scaledTexCoord.x - dist, scaledTexCoord.y)) * colDiffuse * fragColor;
    vec4 texelDown = texture(texture1, vec2(scaledTexCoord.x, scaledTexCoord.y + dist)) * colDiffuse * fragColor;
    vec4 texelUp = texture(texture1, vec2(scaledTexCoord.x, scaledTexCoord.y - dist)) * colDiffuse * fragColor;

    vec4 texelRightDown = texture(texture1, vec2(scaledTexCoord.x + dist, scaledTexCoord.y + dist)) * colDiffuse * fragColor;
    vec4 texelLeftDown = texture(texture1, vec2(scaledTexCoord.x - dist, scaledTexCoord.y + dist)) * colDiffuse * fragColor;
    vec4 texelRightUp = texture(texture1, vec2(scaledTexCoord.x + dist, scaledTexCoord.y - dist)) * colDiffuse * fragColor;
    vec4 texelLeftUp = texture(texture1, vec2(scaledTexCoord.x  - dist, scaledTexCoord.y - dist)) * colDiffuse * fragColor;
    
    // Calculate the position within the scaled tile
    vec2 tilePos = fract(scaledTexCoord * resolution.xy);
    
    // Check if the pixel is on the border of the tile
	bool isBorder = 
    	(texelColor.rgb == vec3(0.0, 0.0, 0.0)) && (
    	(texelColor != texelRight) ||
    	(texelColor != texelDown) ||
    	(texelColor != texelLeft) ||
    	(texelColor != texelUp) ||
    	(texelColor != texelRightDown) ||
    	(texelColor != texelLeftDown) ||
    	(texelColor != texelRightUp) ||
    	(texelColor != texelLeftUp));

    float lerp = 10*(scaledTexCoord.x - scaledTexCoord.y);
    float lerp_red = clamp(lerp, 0, 1);
    float lerp_cyan = clamp(1.0-lerp, 0, 1);

    // Add some noise
    //float inp_x = round(512 * scaledTexCoord.x);
    //float inp_y = round(512 * scaledTexCoord.y);

    float inp_x = round(4096 * scaledTexCoord.x) / 8.0;
    float inp_y = round(4096 * scaledTexCoord.y) / 8.0;

    vec2 inp = vec2(inp_x, inp_y);
    //float noise = sin(inp.x * 0.01 + time * 0.1) * sin(inp.y * 0.01 + time * 0.1);
    //float noise = sin(0.1*(inp.x - inp.y)) + sin(0.1*(inp.x + inp.y));
    //float noise = rand(inp);

    if (isBorder) {
        // Change border pixels to (0, 128, 128, 255)
        outputColor = vec4(lerp_red*borderColor, lerp_cyan*borderColor, (lerp_cyan+0.5)*borderColor, 1.0);
    } else if (texelColor.rgb == vec3(1.0, 1.0, 1.0)) {
        // Change white pixels to (6, 24, 24, 255)
        outputColor = vec4(18.0/255.0*lerp_red + 6.0/255.0, 18.0/255.0*lerp_cyan + 6.0/255.0, 18.0/255.0*lerp_cyan + 6.0/255.0, 1.0);
        float noise = sin(100*inp.x - 100*inp.y + cos(100*inp.y));
        outputColor.rgb += 0.005 + 0.005*vec3(lerp_red*noise, lerp_cyan*noise, lerp_cyan*noise);
    } else if (texelColor.rgb == vec3(0.0, 0.0, 0.0)) {
        // Change black pixels to cyan (0, 255, 255, 255)
        outputColor = vec4(0.5*lerp_red, 0.5*lerp_cyan, 0.5*lerp_cyan, 1.0);
    } else {
        // Keep other colors unchanged
        outputColor = texelColor;
    }


    /*
    if (scaledTexCoord.x < scaledTexCoord.y) {
        if (isBorder) {
            // Change border pixels to (0, 128, 128, 255)
            outputColor = vec4(0.0, borderColor, borderColor, 1.0);
        } else if (texelColor.rgb == vec3(1.0, 1.0, 1.0)) {
            // Change white pixels to (6, 24, 24, 255)
            outputColor = vec4(6.0/255.0, 24.0/255.0, 24.0/255.0, 1.0);
        } else if (texelColor.rgb == vec3(0.0, 0.0, 0.0)) {
            // Change black pixels to cyan (0, 255, 255, 255)
            outputColor = vec4(0.0, 0.5, 0.5, 1.0);
        } else {
            // Keep other colors unchanged
            outputColor = texelColor;
        }
    } else {
        if (isBorder) {
            // Change border pixels to (128, 0, 0, 255)
            outputColor = vec4(borderColor, 0.0, 0.0, 1.0);
        } else if (texelColor.rgb == vec3(1.0, 1.0, 1.0)) {
            // Change white pixels to (24, 6, 6, 255)
            outputColor = vec4(24.0/255.0, 6.0/255.0, 6.0/255.0, 1.0);
        } else if (texelColor.rgb == vec3(0.0, 0.0, 0.0)) {
            // Change black pixels to red (255, 0, 0, 255)
            outputColor = vec4(0.5, 0.0, 0.0, 1.0);
        } else {
            // Keep other colors unchanged
            outputColor = texelColor;
        }
    }
    */
}

