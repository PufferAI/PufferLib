#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;       // Fragment input: vertex attribute: texture coordinates
in vec4 fragColor;          // Fragment input: vertex attribute: color 
in vec3 fragPosition;       // Fragment input: vertex attribute: position

// Input uniform values
uniform sampler2D texture0; // Fragment input: texture

// Output fragment color
out vec4 finalColor;       // Fragment output: pixel color


//Ashima's simplex noise
vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x) {
  return mod289(((x*34.0)+10.0)*x);
}

float snoise(vec2 v)
  {
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
// First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
		+ i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}


float rand(vec2 co){
  return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{
    // Color based on height (e.g., gradient from blue to red)
    float height = fragPosition.y/32.0;
    float delta = 0.0;
    for (int i = -4; i < 4; i++) {
        float scale = pow(2.0, float(i));
        delta += 0.02*snoise((1.0+fragPosition.y)*fragPosition.xz/scale);
    }
    //float scale = pow(2.0, 4.0);
    //float delta = round(2.0*snoise(fragPosition.xz/scale))/10.0;
    float val = 0.5 + height;


    float black = 0.25 - 0.25*fract(20.0*height);
    //height = round(height*5.0)/5.0;
    vec4 start_color = vec4(128.0/255.0, 48.0/255.0, 0.0, 1.0);
    vec4 mid_color = vec4(255.0/255.0, 184.0/255.0, 0.0/255.0, 1.0);
    vec4 end_color = vec4(220.0/255.0, 48.0/255.0, 0.0, 1.0);


    //if (height < 0.5) {
    //    finalColor.rgba = mix(start_color, mid_color, 2.0*height);
    //} else {
    //    finalColor.rgba = mix(mid_color, end_color, 2.0*(height-0.5));
    //}

    finalColor.rgba = mix(mid_color, end_color, height);

    finalColor.rgb -= black;

    if (height < 0.001) {
        finalColor.rgb = start_color.rgb + delta;
    }

    //if (height < 5.0) {
    //    float val = 0.5 + height/10.0;
    //    finalColor.rgba = vec4(val+delta, (val+delta)/2.0, 0.0, 1.0);
    //} else if (height < 15.0) {
    //    float val = 0.5 + (height - 5.0)/5.0;
    //    finalColor.rgba = vec4(val+delta, (val+delta)/3.0, 0.0, 1.0);
    //} else {
    //    float val = 0.5 + (height - 15.0)/(32.0 - 15.0)/2.0;
    //    finalColor.rgba = vec4(val+delta, (val+delta)/4.0, 0.0, 1.0);
    //}
}

