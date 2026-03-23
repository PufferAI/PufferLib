#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;       // Fragment input: vertex attribute: texture coordinates
in vec4 fragColor;          // Fragment input: vertex attribute: color 
in vec3 fragPosition;       // Fragment input: vertex attribute: position
uniform int width;               
uniform int height;              

// Input uniform values
uniform sampler2D texture0; // Fragment input: texture
uniform sampler2D terrain; // Fragment input: texture

// Output fragment color
out vec4 finalColor;       // Fragment output: pixel color

void main()
{
    // Color based on height (e.g., gradient from blue to red)

    float x = fragPosition.x/float(width);
    float y = fragPosition.z/float(height);

    float terrain_height = texture(terrain, vec2(x, y)).r * 32.0;
    float target_height = fragPosition.y;
    float abs_delta = abs(target_height/32.0 - terrain_height/32.0);

    vec4 start_color = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 end_color = vec4(0.0, 1.0, 1.0, 1.0);
    vec4 delta_color = mix(start_color, end_color, abs_delta);

    if (abs_delta > 0.0) {
        abs_delta += 0.25;
    }

    float grid = 16.0;
    float glow = max(exp(-grid * abs(sin(3.14159 * fragPosition.x / grid))), exp(-grid * abs(sin(3.14159 * fragPosition.z / grid))));

    float black = 0.25 - 0.25*fract(20.0*target_height/32.0);
    float r = delta_color.r - black;
    float g = max(glow, delta_color.g - black);
    float b = max(glow, delta_color.b - black);
    float a = max(glow, abs_delta);
    // finalColor.rgba = vec4(r, g, b, a);


    finalColor.rgba = vec4(delta_color.r-black, delta_color.g-black, delta_color.b-black, abs_delta);
}

