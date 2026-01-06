#version 100

// Precision qualifiers (required in GLSL 100)
precision highp float;

// Input from vertex shader (use 'varying' instead of 'in')
varying vec2 fragTexCoord;   // Texture coordinates
varying vec4 fragColor;      // Vertex color
varying vec3 fragPosition;   // Vertex position

// Uniforms (texture0 is not used in this shader but kept for completeness)
uniform sampler2D texture0;

// Ashima's simplex noise (unchanged, but ensure precision)
vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x) {
    return mod289(((x * 34.0) + 10.0) * x);
}

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                       -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    // First corner
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);

    // Other corners
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                     + i.x + vec3(0.0, i1.x, 1.0));

    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;

    // Gradients
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    // Normalise gradients
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    // Compute final noise value
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    float height = fragPosition.y / 32.0;
    float delta = 0.0;
    for (float i = -4.0; i < 4.0; i += 1.0) { // Use float for loop counter
        float scale = pow(2.0, i);
        delta += 0.02 * snoise((1.0 + fragPosition.y) * fragPosition.xz / scale);
    }
    float val = 0.5 + height;

    float black = 0.25 - 0.25 * fract(20.0 * height);
    vec4 start_color = vec4(128.0 / 255.0, 48.0 / 255.0, 0.0, 1.0);
    vec4 mid_color = vec4(255.0 / 255.0, 184.0 / 255.0, 0.0 / 255.0, 1.0);
    vec4 end_color = vec4(220.0 / 255.0, 48.0 / 255.0, 0.0, 1.0);

    vec4 color = mix(mid_color, end_color, height);
    color.rgb -= black;

    if (height < 0.001) {
        color.rgb = start_color.rgb + delta;
    }

    gl_FragColor = color; // Output to gl_FragColor instead of finalColor
}
