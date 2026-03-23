#version 330

in vec4 fragColor;
out vec4 finalColor;


void main()
{

vec2 uv = gl_PointCoord - vec2(0.5);
float dist = length(uv); // distance from center of point

// Soft radial falloff (tightens with higher number)
float falloff = exp(-20.0 * dist * dist);

// Kill pixels too dark to be visible — avoids black ring
if (falloff < 0.01)
    discard;

// Final color, scaled by falloff
vec3 color = fragColor.rgb * falloff * 5.0;
finalColor = vec4(color, falloff);


}
