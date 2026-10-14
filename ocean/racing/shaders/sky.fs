#version 330
out vec4 finalColor;
uniform vec2 resolution;
uniform vec3 viewForward;
uniform vec3 viewRight;
uniform vec3 viewUp;
uniform vec3 sunDirection;
uniform float perspectiveScale;
float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p) {
    vec2 i=floor(p), f=fract(p);
    f=f*f*(3.0-2.0*f);
    return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),
        mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);
}
void main() {
    vec2 uv=2.0*gl_FragCoord.xy/resolution-1.0;
    uv.x *= resolution.x/resolution.y;
    vec3 direction=normalize(viewForward+perspectiveScale*(uv.x*viewRight+uv.y*viewUp));
    float height=max(direction.y,0.0);
    vec3 sky=mix(vec3(0.73,0.81,0.88),vec3(0.16,0.39,0.70),pow(height,0.45));
    if(direction.y>0.03) {
        vec2 p=direction.xz/(direction.y+0.18)*2.0;
        float cloud=noise(p)*0.57+noise(p*2.07)*0.28+noise(p*4.13)*0.15;
        cloud=smoothstep(0.55,0.78,cloud)*smoothstep(0.03,0.20,direction.y)*0.65;
        sky=mix(sky,vec3(0.94,0.95,0.96),cloud);
    }
    float sun=max(dot(direction,sunDirection),0.0);
    sky+=vec3(0.30,0.23,0.13)*pow(sun,64.0);
    sky=mix(sky,vec3(1.0,0.96,0.84),smoothstep(0.99990,0.99996,sun));
    sky=mix(sky,vec3(0.48,0.53,0.55),smoothstep(0.0,0.35,-direction.y));
    finalColor=vec4(sky,1.0);
}
