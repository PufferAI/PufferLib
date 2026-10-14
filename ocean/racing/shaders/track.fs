#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 worldPosition;
in vec3 worldNormal;
out vec4 finalColor;
uniform sampler2D texture0;
uniform sampler2D texture1; // detail albedo
uniform sampler2D texture2; // tangent-space normal
uniform sampler2D shadowTexture;
uniform mat4 lightVP;
uniform int shadowEnabled;
uniform vec3 emission;
uniform vec4 colDiffuse;
uniform int alphaMode;
uniform int groundSurface;
uniform int lit;
uniform vec3 eyePosition;
uniform vec3 sunDirection;
uniform float fogDensity;
uniform float normalStrength, detailScale, detailStrength, detailWorld, roughness, metallic;

vec3 surfaceNormal(vec3 n) {
    vec3 dp1 = dFdx(worldPosition), dp2 = dFdy(worldPosition);
    vec2 uv=groundSurface!=0 && n.y>0.85 ? worldPosition.xz*0.125 : fragTexCoord;
    vec2 duv1 = dFdx(uv), duv2 = dFdy(uv);
    vec3 t = cross(dp2, n)*duv1.x + cross(n, dp1)*duv2.x;
    vec3 b = cross(dp2, n)*duv1.y + cross(n, dp1)*duv2.y;
    float size = max(dot(t,t), dot(b,b));
    if (size < 1e-16 || normalStrength <= 0.0) return n;
    vec3 m = texture(texture2, uv).xyz*2.0-1.0;
    m.xy *= normalStrength;
    return normalize(mat3(t*inversesqrt(size), b*inversesqrt(size), n)*normalize(m));
}
float visibility(float nl) {
    if (shadowEnabled==0 || nl<=0.0) return 1.0;
    vec4 clip=lightVP*vec4(worldPosition,1.0);
    vec3 p=clip.xyz/clip.w*0.5+0.5;
    vec3 dx=dFdx(p),dy=dFdy(p);
    float determinant=dx.x*dy.y-dx.y*dy.x;
    vec2 gradient=abs(determinant)>1e-12
        ? vec2(dx.z*dy.y-dy.z*dx.y,dy.z*dx.x-dx.z*dy.x)/determinant : vec2(0.0);
    gradient=clamp(gradient,vec2(-2.0),vec2(2.0));
    if (any(lessThan(p,vec3(0.0))) || any(greaterThan(p,vec3(1.0)))) return 1.0;
    vec2 texel=1.0/vec2(textureSize(shadowTexture,0));
    float bias=0.00015, visible=0.0;
    for (int x=-1;x<=1;x++) for (int y=-1;y<=1;y++) {
        vec2 uv=p.xy+vec2(x,y)*texel;
        vec2 center=(floor(uv/texel)+0.5)*texel;
        // Compare the receiver plane at each texel centre, avoiding self-shadow stripes.
        float receiver=p.z+dot(center-p.xy,gradient);
        visible+=receiver-bias <= texture(shadowTexture,uv).r ? 1.0 : 0.0;
    }
    float edge=min(min(p.x,1.0-p.x),min(p.y,1.0-p.y));
    return mix(1.0,visible/9.0,smoothstep(0.0,0.06,edge));
}
vec3 skyReflection(vec3 direction,float r) {
    vec3 sky=mix(vec3(0.73,0.81,0.88),vec3(0.16,0.39,0.70),pow(max(direction.y,0.0),0.45));
    sky=mix(sky,vec3(0.18,0.20,0.16),smoothstep(0.0,0.5,-direction.y));
    sky+=vec3(0.8,0.65,0.4)*pow(max(dot(direction,normalize(sunDirection)),0.0),mix(512.0,4.0,r));
    return pow(sky,vec3(2.2));
}
vec3 toneMap(vec3 color) {
    color*=1.15;
    return clamp((color*(2.51*color+0.03))/(color*(2.43*color+0.59)+0.14),0.0,1.0);
}
void main() {
    vec4 c = texture(texture0, fragTexCoord);
    if (groundSurface != 0 && normalize(worldNormal).y > 0.85)
        c = texture(texture1, worldPosition.xz*0.125);
    c.a*=colDiffuse.a*fragColor.a;
    if (alphaMode == 1 && c.a < 0.35) discard;
    if (detailStrength > 0.0) {
        vec2 uv = (detailWorld > 0.5 ? worldPosition.xz : fragTexCoord)*detailScale;
        vec3 detail = texture(texture1, uv).rgb;
        if (detailWorld > 1.5) {
            // Supplied grass photograph supplies fine color; retain the authored broad pattern.
            c.rgb = mix(c.rgb, detail, detailStrength);
        } else {
            vec3 mean = max(textureLod(texture1, uv, 10.0).rgb, vec3(0.03));
            c.rgb *= mix(vec3(1.0), clamp(detail/mean, 0.5, 1.5), detailStrength);
        }
    }
    vec3 color = c.rgb;
    if (lit != 0) {
        vec3 n = normalize(worldNormal);
        if (!gl_FrontFacing) n = -n;
        n = surfaceNormal(n);
        vec3 base = pow(max(c.rgb, vec3(0.0)), vec3(2.2))*colDiffuse.rgb*fragColor.rgb;
        vec3 l = normalize(sunDirection), v = normalize(eyePosition-worldPosition);
        vec3 h = normalize(l+v);
        float nl = max(dot(n,l),0.0), nv = max(dot(n,v),0.001);
        float nh = max(dot(n,h),0.0), vh = max(dot(v,h),0.0);
        float r = clamp(roughness,0.12,1.0), a2 = r*r*r*r;
        float d = a2/(3.14159265*pow(nh*nh*(a2-1.0)+1.0,2.0));
        float k = (r+1.0)*(r+1.0)/8.0;
        float g = nv/(nv*(1.0-k)+k)*nl/(nl*(1.0-k)+k);
        vec3 f = mix(vec3(0.04),base,metallic);
        f += (1.0-f)*pow(1.0-vh,5.0);
        vec3 specular = d*g*f/max(4.0*nv*nl,0.001);
        vec3 ambient = mix(vec3(0.18,0.17,0.14),vec3(0.30,0.34,0.39),n.y*0.5+0.5);
        vec3 light = base*ambient + ((1.0-f)*(1.0-metallic)*base/3.14159265 + specular)
            *nl*vec3(2.5,2.35,2.15)*visibility(max(dot(normalize(worldNormal),l),0.0));
        vec3 reflected=skyReflection(reflect(-v,n),r);
        light+=reflected*f*(1.0-0.65*r)+emission;
        float fog = 1.0-exp(-length(worldPosition-eyePosition)*fogDensity);
        light=mix(light,pow(vec3(0.73,0.81,0.88),vec3(2.2)),min(fog,0.65));
        color=pow(toneMap(max(light,vec3(0.0))),vec3(1.0/2.2));
    } else color*=colDiffuse.rgb*fragColor.rgb;
    finalColor = vec4(color,alphaMode == 2 ? c.a : 1.0);
}
