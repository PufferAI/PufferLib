#version 100

// Vertex shader will need to pass these as varyings
precision mediump float;

varying vec2 fragTexCoord; 
varying vec4 fragColor;          

// Input uniform values
uniform sampler2D texture0; 
uniform sampler2D texture1; 
uniform vec4 colDiffuse;    

//uniform vec3 resolution;    
uniform vec4 mouse;         
uniform float time;         
uniform float camera_x;
uniform float camera_y;

const float SCALE_FACTOR = 16.0;
const float GLOW = 32.0;
const float DIST = GLOW / SCALE_FACTOR;

float rand(vec2 co){
  return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float round(float x) {
    return floor(x + 0.5);
}

void main()
{
    vec3 resolution = vec3(2560.0, 1440.0, 1.0);
    float CONVERT_X = resolution.x / 32.0 / 128.0;
    float CONVERT_Y = resolution.y / 32.0 / 128.0;

    vec2 scaledTexCoord = vec2(CONVERT_X * fragTexCoord.x + camera_x, CONVERT_Y * fragTexCoord.y + camera_y);
    float dist = DIST / resolution.x; // Pass texture size as uniform from JS

    float borderColor = 0.8 + 0.2 * sin(time);

    vec4 texelColor = texture2D(texture1, scaledTexCoord) * colDiffuse * fragColor;
    vec4 texelRight = texture2D(texture1, vec2(scaledTexCoord.x + dist, scaledTexCoord.y)) * colDiffuse * fragColor;
    vec4 texelLeft = texture2D(texture1, vec2(scaledTexCoord.x - dist, scaledTexCoord.y)) * colDiffuse * fragColor;
    vec4 texelDown = texture2D(texture1, vec2(scaledTexCoord.x, scaledTexCoord.y + dist)) * colDiffuse * fragColor;
    vec4 texelUp = texture2D(texture1, vec2(scaledTexCoord.x, scaledTexCoord.y - dist)) * colDiffuse * fragColor;

    vec4 texelRightDown = texture2D(texture1, vec2(scaledTexCoord.x + dist, scaledTexCoord.y + dist)) * colDiffuse * fragColor;
    vec4 texelLeftDown = texture2D(texture1, vec2(scaledTexCoord.x - dist, scaledTexCoord.y + dist)) * colDiffuse * fragColor;
    vec4 texelRightUp = texture2D(texture1, vec2(scaledTexCoord.x + dist, scaledTexCoord.y - dist)) * colDiffuse * fragColor;
    vec4 texelLeftUp = texture2D(texture1, vec2(scaledTexCoord.x  - dist, scaledTexCoord.y - dist)) * colDiffuse * fragColor;

    vec2 tilePos = fract(scaledTexCoord * resolution.xy);

    bool isBorder = (texelColor.rgb == vec3(0.0, 0.0, 0.0)) && (
        (texelColor != texelRight) ||
        (texelColor != texelDown) ||
        (texelColor != texelLeft) ||
        (texelColor != texelUp) ||
        (texelColor != texelRightDown) ||
        (texelColor != texelLeftDown) ||
        (texelColor != texelRightUp) ||
        (texelColor != texelLeftUp));

    float lerp = 10.0 * (scaledTexCoord.x - scaledTexCoord.y);
    float lerp_red = clamp(lerp, 0.0, 1.0);
    float lerp_cyan = clamp(1.0 - lerp, 0.0, 1.0);

    float inp_x = round(4096.0 * scaledTexCoord.x) / 8.0;
    float inp_y = round(4096.0 * scaledTexCoord.y) / 8.0;

    vec2 inp = vec2(inp_x, inp_y);

    if (isBorder) {
        gl_FragColor = vec4(lerp_red * borderColor, lerp_cyan * borderColor, (lerp_cyan + 0.5) * borderColor, 1.0);
    } else if (texelColor.rgb == vec3(1.0, 1.0, 1.0)) {
        gl_FragColor = vec4(18.0 / 255.0 * lerp_red + 6.0 / 255.0, 18.0 / 255.0 * lerp_cyan + 6.0 / 255.0, 18.0 / 255.0 * lerp_cyan + 6.0 / 255.0, 1.0);
        float noise = sin(100.0 * inp.x - 100.0 * inp.y + cos(100.0 * inp.y));
        gl_FragColor.rgb += 0.005 + 0.005 * vec3(lerp_red * noise, lerp_cyan * noise, lerp_cyan * noise);
    } else if (texelColor.rgb == vec3(0.0, 0.0, 0.0)) {
        gl_FragColor = vec4(0.5 * lerp_red, 0.5 * lerp_cyan, 0.5 * lerp_cyan, 1.0);
    } else {
        gl_FragColor = texelColor;
    }
}
