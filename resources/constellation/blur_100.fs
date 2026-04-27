#version 100
precision mediump float;
varying vec2 fragTexCoord;
varying vec4 fragColor;
uniform sampler2D texture0;

const float renderWidth = 960.0;
const float offset0 = 0.0;
const float offset1 = 1.3846153846;
const float offset2 = 3.2307692308;
const float weight0 = 0.2270270270;
const float weight1 = 0.3162162162;
const float weight2 = 0.0702702703;

void main() {
    vec3 texelColor = texture2D(texture0, fragTexCoord).rgb * weight0;
    texelColor += texture2D(texture0, fragTexCoord + vec2(offset1) / renderWidth).rgb * weight1;
    texelColor += texture2D(texture0, fragTexCoord - vec2(offset1) / renderWidth).rgb * weight1;
    texelColor += texture2D(texture0, fragTexCoord + vec2(offset2) / renderWidth).rgb * weight2;
    texelColor += texture2D(texture0, fragTexCoord - vec2(offset2) / renderWidth).rgb * weight2;
    gl_FragColor = vec4(texelColor, 1.0);
}
