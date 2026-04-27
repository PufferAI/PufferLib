#version 330

in vec2 fragTexCoord;
out vec4 finalColor;

uniform sampler2D texture0;
uniform vec2 texelSize;     // 1.0 / vec2(textureWidth, textureHeight)
uniform float glowStrength; // driven by cos in time
uniform float fadeAlpha;    // 0..1 fade-in multiplier

void main()
{
    vec2 uv = fragTexCoord;
    float c = texture(texture0, uv).a;

    // Solid pixels stay fully opaque — no shader math needed
    if (c > 0.5) {
        vec3 color = vec3(0.6, 0.95, 0.95);
        finalColor = vec4(color * fadeAlpha, fadeAlpha);
        return;
    }

    // Find the distance to the nearest solid pixel, then apply a smooth
    // exponential falloff on that distance. This gives a proper gradient
    // rather than a flat accumulated plateau.
    float min_dist = 32.0;
    int R = 32;
    for (int dx = -R; dx <= R; dx++) {
        for (int dy = -R; dy <= R; dy++) {
            float s = texture(texture0, uv + texelSize * vec2(dx, dy)).a;
            if (s > 0.5) {
                min_dist = min(min_dist, length(vec2(dx, dy)));
            }
        }
    }

    float glow  = exp(-0.12 * min_dist);  // bright at d=1, fades over ~20px
    float halo  = glow * (0.4 + glowStrength * 0.6) * fadeAlpha;
    vec3 color  = vec3(0.6, 0.95, 0.95);
    finalColor  = vec4(color * halo, halo);
}
