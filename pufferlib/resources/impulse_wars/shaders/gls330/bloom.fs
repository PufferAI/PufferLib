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

#version 330 core

#define BLOOM_DISABLED 0
#define BLOOM_ADDITIVE 1
#define BLOOM_SOFT_LIGHT 2

noperspective in vec2 fragTexCoord;

uniform sampler2D uTexColor;
uniform sampler2D uTexBloomBlur;

uniform lowp int uBloomMode;
uniform float uBloomIntensity;

out vec4 fragColor;

void main()
{
    // Sampling scene color texture
    vec3 result = texture(uTexColor, fragTexCoord).rgb;

    // Apply bloom
    vec3 bloom = texture(uTexBloomBlur, fragTexCoord).rgb;
    bloom *= uBloomIntensity;

    if (uBloomMode == BLOOM_SOFT_LIGHT) {
        bloom = clamp(bloom.rgb, vec3(0.0), vec3(1.0));
        result = max((result + bloom) - (result * bloom), vec3(0.0));
    } else if (uBloomMode == BLOOM_ADDITIVE) {
        result += bloom;
    }

    // Final color output
    fragColor = vec4(result, 1.0);
}