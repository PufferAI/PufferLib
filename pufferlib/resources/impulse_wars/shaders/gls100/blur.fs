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

noperspective in vec2 fragTexCoord;

uniform sampler2D uTexture;
uniform vec2 uTexelDir;

out vec4 fragColor;

const int SAMPLE_COUNT = 6;

const float OFFSETS[6] = float[6](
    -4.455269417428358,
    -2.4751038298192056,
    -0.4950160492928827,
    1.485055021558738,
    3.465172537482815,
    5
);

const float WEIGHTS[6] = float[6](
    0.14587920530480702,
    0.19230308352110734,
    0.21647621943673803,
    0.20809835496561988,
    0.17082879595769634,
    0.06641434081403137
);

void main()
{
    vec3 result = vec3(0.0);

    for (int i = 0; i < SAMPLE_COUNT; ++i)
    {
        result += texture(uTexture, fragTexCoord + uTexelDir * OFFSETS[i]).rgb * WEIGHTS[i];
    }

    fragColor = vec4(result, 1.0);
}
