#version 330

in vec4 fragColor;
out vec4 finalColor;

void main()
{
    vec2 uv = gl_PointCoord - vec2(0.5);
    float r = length(uv);
    float angle = atan(uv.y, uv.x);

    // Multi-lobe Gaussian core (tight + medium + wide)
    float core   = exp(-120.0 * r * r);
    float mid    = exp(- 30.0 * r * r) * 0.5;
    float corona = exp(-  3.5 * r * r) * 0.35;

    // Diffraction spikes: 4-point cross, sharp along axes
    // pow(abs(cos(angle*2)), p) peaks at 0, 90, 180, 270 degrees
    float spike_mask = pow(abs(cos(angle * 2.0)), 24.0);
    float spike = spike_mask * exp(-18.0 * r * r) * exp(-r * 4.0) * 1.2;

    float brightness = core + mid + corona + spike;

    if (brightness < 0.01)
        discard;

    // Slight chromatic shift on spikes/corona toward cool blue
    vec3 color = fragColor.rgb * brightness;
    color += vec3(0.0, 0.05, 0.15) * spike;
    color += vec3(0.0, 0.02, 0.08) * corona;

    finalColor = vec4(color, clamp(brightness, 0.0, 1.0));
}
