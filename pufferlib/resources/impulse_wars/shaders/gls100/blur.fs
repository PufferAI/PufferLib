/*
 * Copyright (c) 2025 Le Juez Victor
 *
 * This software is provided "as-is", without any express or implied warranty. In no event
 * will the authors be held liable for any damages arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose, including commercial
 * applications, and to alter it and redistribute it freely, subject to the following restrictions:
 *
 *   1. The origin of this software must not be misrepresented; you must not claim that you
 *   wrote the original software. If you use this software in a product, an acknowledgment
 *   in the product documentation would be appreciated but is not required.
 *
 *   2. Altered source versions must be plainly marked as such, and must not be misrepresented
 *   as being the original software.
 *
 *   3. This notice may not be removed or altered from any source distribution.
 */

// NOTE: The coefficients for the two-pass Gaussian blur were generated using:
//       https://lisyarus.github.io/blog/posts/blur-coefficients-generator.html

#version 100

precision mediump float;

varying vec2 fragTexCoord;

uniform sampler2D uTexture;
uniform vec2 uTexelDir;

// hard‐coded offsets & weights
const float O0 = -4.455269417428358;
const float O1 = -2.4751038298192056;
const float O2 = -0.4950160492928827;
const float O3 =  1.485055021558738;
const float O4 =  3.465172537482815;
const float O5 =  5.0;

const float W0 = 0.14587920530480702;
const float W1 = 0.19230308352110734;
const float W2 = 0.21647621943673803;
const float W3 = 0.20809835496561988;
const float W4 = 0.17082879595769634;
const float W5 = 0.06641434081403137;

void main() {
    vec3 result = vec3(0.0);
    result += texture2D(uTexture, fragTexCoord + uTexelDir * O0).rgb * W0;
    result += texture2D(uTexture, fragTexCoord + uTexelDir * O1).rgb * W1;
    result += texture2D(uTexture, fragTexCoord + uTexelDir * O2).rgb * W2;
    result += texture2D(uTexture, fragTexCoord + uTexelDir * O3).rgb * W3;
    result += texture2D(uTexture, fragTexCoord + uTexelDir * O4).rgb * W4;
    result += texture2D(uTexture, fragTexCoord + uTexelDir * O5).rgb * W5;

    gl_FragColor = vec4(result, 1.0);
}
