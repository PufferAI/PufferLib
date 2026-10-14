#version 330

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;
in vec4 shadowPosition;
uniform sampler2D texture0;
uniform sampler2D shadowMap;
uniform vec4 colDiffuse;
uniform vec3 lightDirection;
uniform vec3 viewPosition;
uniform int surfaceKind;
out vec4 finalColor;

float visibility(float facing) {
    vec3 projected = shadowPosition.xyz / shadowPosition.w * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0
        || any(lessThan(projected.xy, vec2(0.0)))
        || any(greaterThan(projected.xy, vec2(1.0)))) {
        return 1.0;
    }
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    // Compare each texel against the receiver plane to avoid sloped-surface acne.
    vec3 dx = dFdx(projected);
    vec3 dy = dFdy(projected);
    float determinant = dx.x * dy.y - dx.y * dy.x;
    vec2 gradient = vec2(0.0);
    if (abs(determinant) > 1e-10) {
        gradient = vec2(dx.z * dy.y - dy.z * dx.y, dx.x * dy.z - dy.x * dx.z) / determinant;
    }
    float bias = max(0.00004, 0.00008 * (1.0 - facing));
    float visible = 0.0;
    float weights = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            float weight = float((3 - abs(x)) * (3 - abs(y)));
            vec2 sampleUV = (floor(projected.xy / texel) + vec2(x, y) + 0.5) * texel;
            float depth = texture(shadowMap, sampleUV).r;
            float receiverDepth = projected.z + dot(gradient, sampleUV - projected.xy);
            visible += weight * step(receiverDepth - bias, depth);
            weights += weight;
        }
    }
    float edge = min(min(projected.x, projected.y), min(1.0 - projected.x, 1.0 - projected.y));
    return mix(1.0, visible / weights, smoothstep(0.0, 0.035, edge));
}

vec3 toneMap(vec3 value) {
    return clamp((value * (2.51 * value + 0.03))
        / (value * (2.43 * value + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec4 surface = texture(texture0, fragTexCoord) * colDiffuse * fragColor;
    vec3 albedo = pow(max(surface.rgb, vec3(0.0)), vec3(2.2));
    vec3 n = normalize(fragNormal);
    vec3 l = normalize(lightDirection);
    vec3 v = normalize(viewPosition - fragPosition);
    vec3 h = normalize(l + v);
    float nl = max(dot(n, l), 0.0);
    float nv = max(dot(n, v), 0.001);
    float nh = max(dot(n, h), 0.0);
    float vh = max(dot(v, h), 0.0);
    float roughness = surfaceKind == 1 ? 0.78 : 0.38;
    if (surfaceKind == 1) {
        vec2 cell = fragPosition.xz * 10.0;
        vec2 distance = abs(fract(cell - 0.5) - 0.5) / max(fwidth(cell), vec2(0.001));
        float grid = 1.0 - min(min(distance.x, distance.y), 1.0);
        grid *= 1.0 - smoothstep(0.35, 1.0, max(fwidth(cell.x), fwidth(cell.y)));
        albedo *= 1.0 + 0.08 * grid;
    }
    float a = roughness * roughness;
    float a2 = a * a;
    float denominator = nh * nh * (a2 - 1.0) + 1.0;
    float distribution = a2 / max(3.14159265 * denominator * denominator, 0.0001);
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float geometry = nv / (nv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k);
    vec3 fresnel = vec3(0.04) + vec3(0.96) * pow(1.0 - vh, 5.0);
    vec3 specular = distribution * geometry * fresnel / max(4.0 * nv * nl, 0.001);
    vec3 ambient = mix(vec3(0.24, 0.26, 0.30), vec3(0.48, 0.54, 0.63), n.y * 0.5 + 0.5);
    vec3 direct = ((1.0 - fresnel) * albedo + specular) * vec3(1.75, 1.66, 1.52);
    vec3 color = albedo * ambient + direct * nl * visibility(nl);
    finalColor = vec4(pow(toneMap(color), vec3(1.0 / 2.2)), surface.a);
}
