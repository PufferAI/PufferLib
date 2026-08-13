#version 330

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;

uniform sampler2D texture0;
uniform vec4 colDiffuse;
uniform vec3 lightDirection;
uniform vec3 viewPosition;

out vec4 finalColor;

void main() {
    vec4 surface = texture(texture0, fragTexCoord) * colDiffuse * fragColor;
    vec3 normal = normalize(fragNormal);
    vec3 light = normalize(lightDirection);
    vec3 view = normalize(viewPosition - fragPosition);
    vec3 halfway = normalize(light + view);

    float diffuse = max(dot(normal, light), 0.0);
    float sky = 0.5 + 0.5 * normal.y;
    float specular = pow(max(dot(normal, halfway), 0.0), 48.0);
    vec3 illumination = vec3(0.48 + 0.10 * sky + 0.50 * diffuse);
    vec3 color = surface.rgb * illumination + vec3(0.12 * specular);
    finalColor = vec4(color, surface.a);
}
