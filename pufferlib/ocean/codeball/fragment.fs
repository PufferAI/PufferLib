#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragPosition;
in vec3 fragNormal;

uniform vec4 colDiffuse;

// Output fragment color
out vec4 finalColor;
void main() {
    float height = 5.0;
    float sun_intensity = dot(fragNormal, vec3(0.0, 1.0, 0.0));
    float spotlight_radius = 60.0;
    float spotlight_distance = spotlight_radius - length(fragPosition.xz);
    float spotlight_intensity = max(sqrt(max(spotlight_distance, 0.0) / spotlight_radius), 0.3);
    float mul = spotlight_intensity + sun_intensity * 0.4;
    finalColor = fragColor * colDiffuse * vec4(mul, mul, mul, 1.0);
}