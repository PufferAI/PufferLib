#version 330

in vec2 position; // Screen-space position
in vec4 color;    // RGBA color
out vec4 fragColor;

uniform float screenWidth;
uniform float screenHeight;
uniform float pointSize;

void main() {
    // Convert screen-space to NDC
    vec2 ndc = vec2(
        position.x / screenWidth * 2.0 - 1.0,
        1.0 - position.y / screenHeight * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    fragColor = color;
    gl_PointSize = pointSize;
}
