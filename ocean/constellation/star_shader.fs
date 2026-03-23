#version 330

in vec4 fragColor;
out vec4 finalColor;

void main() {
    // Optional: Circular points
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5) discard;
    finalColor = fragColor;
}
