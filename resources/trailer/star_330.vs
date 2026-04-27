#version 330

in vec3 vertexPosition;
in vec4 vertexColor;

uniform mat4 mvp;
uniform float currentTime;

out vec4 fragColor;

float twinkle(float seed, float t) {
    // Stable per-star seed passed via vertexColor.a from CPU
    float phase     = mod(seed * 137.5, 360.0);
    float frequency = 1.2 + mod(seed * 0.317, 3.0) * 0.9;
    float amplitude = 7.0;
    float base_size = 20.0;
    float size_variation = amplitude * sin(frequency * t + radians(phase));
    return max(1.0, base_size + size_variation);
}

void main()
{
    vec2 pos = vertexPosition.xy;
    float size_scale = vertexPosition.z;
    float seed = vertexColor.a;  // stable per-star identity, not alpha

    gl_Position = mvp * vec4(pos, 0.0, 1.0);
    gl_PointSize = twinkle(seed, currentTime) * size_scale;

    fragColor = vertexColor;
}
