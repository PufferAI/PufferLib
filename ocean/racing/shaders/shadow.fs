#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform vec4 colDiffuse;
uniform int alphaMode;
void main() {
    float alpha=texture(texture0,fragTexCoord).a*colDiffuse.a*fragColor.a;
    if (alphaMode!=0 && alpha<0.35) discard;
}
