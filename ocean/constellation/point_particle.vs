#version 330

// Input vertex attributes
in vec3 vertexPosition;
in vec4 vertexColor;

// Input uniform values
uniform mat4 mvp;
uniform float currentTime;

// Output to fragment shader
out vec4 fragColor;

// NOTE: Add your custom variables here

float twinkle(float idx, float t) {
    float base_size = 10.0;
    float phase = mod(idx * 137.5, 360.0);
    float frequency = 2.5 + mod(idx, 3.0) * 1.0;
    float amplitude = (mod(idx, 10.0) == 0.0) ? 10.0 : 2.0;
    float size_variation = amplitude * sin(frequency * t + radians(phase));
    return max(0.5, base_size + size_variation);
}

void main()
{
    // Unpack data from vertexPosition
    vec2  pos = vertexPosition.xy;
    float idx = vertexPosition.z;

    // Calculate final vertex position (jiggle it around a bit horizontally)
    //pos += vec2(100, 0)*sin(period*currentTime);
    gl_Position = mvp*vec4(pos, 0.0, 1.0);

    // Calculate the screen space size of this particle (also vary it over time)
    //gl_PointSize = 10 - 5*abs(sin(0.1*idx*currentTime));
    //gl_PointSize = 10.0;

    gl_PointSize = twinkle(idx, currentTime);

    fragColor = vertexColor;
}
