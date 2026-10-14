#pragma once
#define float3 racing_raymath_float3
#include "raymath.h"
#undef float3
#include "rlgl.h"
#include "car_render.h"
#include "material_render.h"
#include "ghost_view.h"
#include "gate_render.h"
#include "scene_render.h"
#include "lighting_render.h"
#include "race_hud.h"

// Evaluation-only graphics. Training never loads textures or opens a window.
static struct {
    std::vector<RacingEvalSurface> surfaces;
    std::vector<RacingDraw> draw;
    std::vector<RacingRenderBinding> bindings;
    Texture2D *textures;
    unsigned *images, *alpha;
    Color *colors;
    RacingMaterialDetail *details;
    unsigned count, texture_count;
    Material material;
    Shader lighting, sky, shadow_shader;
    RenderTexture2D minimap, shadow;
    Camera3D map_camera, camera;
    BoundingBox bounds;
    Vector3 sun;
    float map_extent;
    int camera_valid, camera_selected, previous_mode, follow_leader, show_hud, diagnostics;
    double camera_check;
    float camera_clearance;
    int shadow_enabled, light_vp, shadow_alpha;
    Texture2D default_texture;
    Shader default_shader;
    RacingCarRender car;
    int opened, camera_mode, manual, lidar, last_serial;
    int show_gates;
    RacingGate gates[128];
    float distance, steering, throttle, brake;
    int alpha_location, ground_location, eye_location, fog_location;
    int sky_resolution, sky_forward, sky_right, sky_up, sky_scale;
} racing_view;

static void racing_eval_bake_minimap();

static void racing_eval_open() {
    auto &v = racing_view;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1440, 960, "Racing | policy evaluation");
    SetWindowMinSize(800,600);
    SetTargetFPS(30);
    racing_hud_font=LoadFontEx("resources/shared/Roboto-Regular.ttf",48,NULL,95);
    SetTextureFilter(racing_hud_font.texture,TEXTURE_FILTER_BILINEAR);
    v.opened = 1;
    v.camera_mode = 1;
    v.distance = 8;
    v.show_hud = 1;
    v.camera_selected = -1;
    v.camera_clearance = 1;
    // Copy the exact counting geometry once; no JSON reconstruction or per-frame GPU transfer.
    assert(trial_gate_count <= 128);
    pf_optix_cuda(cudaMemcpy(v.gates, trial_gates, trial_gate_count * sizeof(RacingGate),
        cudaMemcpyDeviceToHost));
    FILE *map = fopen("ocean/racing/map.bin", "rb");
    FILE *visual = fopen("ocean/racing/map_visual.bin", "rb");
    if (!map || !visual) {
        fprintf(stderr,
                "Missing racing visuals; run bash ocean/racing/view_map.sh --optix first\n");
        exit(1);
    }
    char magic[8];
    unsigned count, materials;
    assert(fread(magic, 1, 8, map) == 8 && memcmp(magic, "PFMAP001", 8) == 0);
    assert(fread(&v.count, 4, 1, map) == 1 && v.count > 0 && v.count < 10000);
    assert(fread(magic, 1, 8, visual) == 8 && memcmp(magic, "PFVIS004", 8) == 0);
    assert(fread(&count, 4, 1, visual) == 1 && count == v.count);
    assert(fread(&v.texture_count, 4, 1, visual) == 1 && v.texture_count <= 1024);
    assert(fread(&materials, 4, 1, visual) == 1 && materials <= 1024);
    unsigned source_count = v.count;
    v.textures = (Texture2D *)calloc(v.texture_count, sizeof(Texture2D));
    v.images = (unsigned *)calloc(materials, sizeof(unsigned));
    v.alpha = (unsigned *)calloc(materials, sizeof(unsigned));
    v.details = (RacingMaterialDetail *)calloc(materials, sizeof(RacingMaterialDetail));
    v.colors = (Color *)calloc(materials, sizeof(Color));
    assert(v.details && v.textures && v.images && v.alpha && v.colors);
    for (unsigned i = 0; i < v.texture_count; i++) {
        unsigned size;
        assert(fread(&size, 4, 1, visual) == 1 && size > 0 && size < 100000000);
        unsigned char *bytes = (unsigned char *)malloc(size);
        assert(bytes && fread(bytes, 1, size, visual) == size);
        Image image = LoadImageFromMemory(".png", bytes, size);
        free(bytes);
        assert(IsImageValid(image));
        v.textures[i] = LoadTextureFromImage(image);
        UnloadImage(image);
        assert(IsTextureValid(v.textures[i]));
        GenTextureMipmaps(&v.textures[i]);
        SetTextureFilter(v.textures[i], TEXTURE_FILTER_TRILINEAR);
        SetTextureFilter(v.textures[i], TEXTURE_FILTER_ANISOTROPIC_8X);
        SetTextureWrap(v.textures[i], TEXTURE_WRAP_REPEAT);
    }
    for (unsigned i = 0; i < materials; i++) {
        float rgba[4];
        assert(fread(&v.images[i], 4, 1, visual) == 1 && v.images[i] < v.texture_count);
        assert(fread(&v.alpha[i], 4, 1, visual) == 1 && v.alpha[i] <= 2);
        assert(fread(&v.details[i].normal, 4, 1, visual) == 1);
        assert(fread(&v.details[i].detail, 4, 1, visual) == 1);
        assert(v.details[i].normal < v.texture_count && v.details[i].detail < v.texture_count);
        assert(fread(rgba, 4, 4, visual) == 4);
        assert(fread(&v.details[i].normal_strength, 4, 6, visual) == 6);
        v.colors[i] = ColorFromNormalized((Vector4){rgba[0], rgba[1], rgba[2], rgba[3]});
    }
    FILE *bindings=fopen("ocean/racing/render_materials.bin","rb");
    if (!bindings) {
        fprintf(stderr,"Run python3 ocean/racing/prepare_render.py to prepare viewer assets\n");
        exit(1);
    }
    assert(fread(magic,1,8,bindings)==8 && memcmp(magic,"PFRMAT01",8)==0);
    assert(fread(&count,4,1,bindings)==1 && count==materials);
    v.bindings.resize(materials);
    assert(fread(v.bindings.data(),sizeof(RacingRenderBinding),materials,bindings)==materials);
    fclose(bindings);
    for (const auto &binding : v.bindings)
        assert(binding.base<v.texture_count && binding.detail<v.texture_count && binding.normal<v.texture_count);
    for (unsigned i = 0; i < source_count; i++) {
        RacingEvalSurface s = {};
        unsigned header[5], vertices;
        char name[160];
        assert(fread(header, 4, 5, map) == 5 && fread(name, 1, 160, map) == 160);
        vertices = header[0];
        s.group = header[1];
        s.material = header[4];
        assert(vertices > 0 && vertices < 10000000 && vertices % 3 == 0);
        assert(s.material < materials);
        s.mesh.vertexCount = vertices;
        s.mesh.triangleCount = vertices / 3;
        s.mesh.vertices = (float *)MemAlloc(vertices * 12);
        s.mesh.texcoords = (float *)MemAlloc(vertices * 8);
        s.mesh.normals = (float *)MemAlloc(vertices * 12);
        assert(s.mesh.vertices && s.mesh.texcoords && s.mesh.normals);
        assert(fread(s.mesh.vertices, 12, vertices, map) == vertices);
        unsigned uv_count;
        assert(fread(&uv_count, 4, 1, visual) == 1 && uv_count == vertices);
        assert(fread(s.mesh.texcoords, 8, vertices, visual) == vertices);
        assert(fread(s.mesh.normals, 12, vertices, visual) == vertices);
        racing_chunk_mesh(s,v.surfaces);
    }
    fclose(map);
    fclose(visual);
    v.count=(unsigned)v.surfaces.size();
    v.draw.reserve(v.count+480);
    BoundingBox bounds = v.surfaces[0].bounds;
    for (unsigned i = 1; i < v.count; i++) {
        Vector3 low = v.surfaces[i].bounds.min, high = v.surfaces[i].bounds.max;
        bounds.min = Vector3Min(bounds.min, low);
        bounds.max = Vector3Max(bounds.max, high);
    }
    v.bounds=bounds;
    float extent = fmaxf(bounds.max.x - bounds.min.x, bounds.max.z - bounds.min.z);
    (void)extent;
    rlSetClipPlanes(0.1,6000);
    v.material = LoadMaterialDefault();
    v.default_shader = v.material.shader;
    v.default_texture = v.material.maps[MATERIAL_MAP_DIFFUSE].texture;
    v.lighting = LoadShader("ocean/racing/shaders/track.vs", "ocean/racing/shaders/track.fs");
    v.sky = LoadShader(NULL, "ocean/racing/shaders/sky.fs");
    assert(IsShaderValid(v.lighting) && IsShaderValid(v.sky));
    v.lighting.locs[SHADER_LOC_MATRIX_MODEL] = GetShaderLocation(v.lighting, "matModel");
    v.lighting.locs[SHADER_LOC_MATRIX_NORMAL] = GetShaderLocation(v.lighting, "matNormal");
    v.alpha_location = GetShaderLocation(v.lighting, "alphaMode");
    v.ground_location = GetShaderLocation(v.lighting, "groundSurface");
    v.eye_location = GetShaderLocation(v.lighting, "eyePosition");
    v.fog_location = GetShaderLocation(v.lighting, "fogDensity");
    v.sky_resolution = GetShaderLocation(v.sky, "resolution");
    v.sky_forward = GetShaderLocation(v.sky, "viewForward");
    v.sky_right = GetShaderLocation(v.sky, "viewRight");
    v.sky_up = GetShaderLocation(v.sky, "viewUp");
    v.sky_scale = GetShaderLocation(v.sky, "perspectiveScale");
    Vector3 sun = Vector3Normalize((Vector3){-0.4f, 0.75f, -0.35f});
    v.sun=sun;
    int lit = 1;
    SetShaderValue(v.lighting, GetShaderLocation(v.lighting, "sunDirection"), &sun,
                   SHADER_UNIFORM_VEC3);
    SetShaderValue(v.lighting, GetShaderLocation(v.lighting, "lit"), &lit, SHADER_UNIFORM_INT);
    SetShaderValue(v.sky, GetShaderLocation(v.sky, "sunDirection"), &sun, SHADER_UNIFORM_VEC3);
    racing_shader_maps(v.lighting);
    v.material.shader = v.lighting;
    racing_car_load(&v.car);
    v.shadow_shader=LoadShader("ocean/racing/shaders/track.vs","ocean/racing/shaders/shadow.fs");
    assert(IsShaderValid(v.shadow_shader));
    v.shadow_shader.locs[SHADER_LOC_MATRIX_MODEL]=GetShaderLocation(v.shadow_shader,"matModel");
    v.shadow_shader.locs[SHADER_LOC_MATRIX_NORMAL]=GetShaderLocation(v.shadow_shader,"matNormal");
    v.shadow_alpha=GetShaderLocation(v.shadow_shader,"alphaMode");
    v.shadow_enabled=GetShaderLocation(v.lighting,"shadowEnabled");
    v.light_vp=GetShaderLocation(v.lighting,"lightVP");
    v.shadow=racing_shadow_target(1024);
    racing_eval_bake_minimap();
    printf("Racing visuals: %u spatial chunks, car LOD at 80m, scene minimap cached\n",v.count);
}


static void racing_eval_collect(Camera3D camera, float aspect,
        bool include_cars, const RacingCarFrame *single, float far_plane=6000) {
    auto &v=racing_view;
    v.draw.clear();
    Matrix clip=racing_camera_matrix(camera,aspect,far_plane);
    Vector3 forward=Vector3Normalize(Vector3Subtract(camera.target,camera.position));
    for (auto &surface : v.surfaces) {
        if (surface.group==6 || !racing_visible(surface.bounds,clip)) continue;
        unsigned id=surface.material;
        const auto &binding=v.bindings[id];
        RacingDraw item={};
        item.mesh=&surface.mesh; item.material=v.material; item.transform=MatrixIdentity();
        item.detail=v.details[id]; item.detail.ground=binding.flags&1;
        item.detail.normal_strength=binding.normal_strength;
        if (binding.flags&2) item.detail.normal_strength=item.detail.detail_strength=0;
        if (item.detail.ground) item.detail.detail_strength=0;
        item.base=v.textures[binding.base]; item.normal=v.textures[binding.normal];
        item.detail_texture=v.textures[binding.detail]; item.color=v.colors[id];
        item.alpha=v.alpha[id]; item.double_sided=(binding.flags&4)!=0;
        Vector3 center=Vector3Scale(Vector3Add(surface.bounds.min,surface.bounds.max),0.5f);
        item.depth=Vector3DotProduct(Vector3Subtract(center,camera.position),forward);
        v.draw.push_back(item);
    }
    if (!include_cars) return;
    int count=racing_ghosts.count ? racing_ghosts.count : single ? 1 : 0;
    for (int car=0;car<count;++car) {
        const RacingCarFrame *f=racing_ghosts.count ? &racing_ghosts.frames[car] : single;
        if (f->task_done || f->crashed) continue;
        Vector3 center={f->position[0],f->position[1],f->position[2]};
        float radius=fmaxf(5,f->wheelbase*1.2f);
        BoundingBox bounds={Vector3Subtract(center,(Vector3){radius,radius,radius}),
                            Vector3Add(center,(Vector3){radius,radius,radius})};
        if (!racing_visible(bounds,clip)) continue;
        bool lod=Vector3DistanceSqr(center,camera.position)>80*80;
        Color tint=racing_ghosts.count ? racing_ghost_colors[car] : WHITE;
        for (int i=0;i<60;++i) {
            Mesh *mesh=lod ? &v.car.lod[i] : &v.car.model.meshes[i];
            if (!mesh->vertexCount) continue;
            Material m=v.car.model.materials[v.car.model.meshMaterial[i]];
            RacingDraw item={}; item.mesh=mesh; item.material=m;
            item.transform=racing_car_part_transform(&v.car,f,i);
            item.detail=v.car.details[i];
            item.base=m.maps[0].texture; item.normal=m.maps[2].texture;
            item.detail_texture=v.default_texture;
            item.color=v.car.flags[i]&1 ? ColorTint(m.maps[0].color,tint) : m.maps[0].color;
            item.alpha=v.car.parts[i][2]; item.double_sided=(v.car.flags[i]&2)!=0;
            Vector3 part=Vector3Transform(Vector3Scale(Vector3Add(v.car.bounds[i].min,v.car.bounds[i].max),0.5f),item.transform);
            item.depth=Vector3DotProduct(Vector3Subtract(part,camera.position),forward);
            v.draw.push_back(item);
        }
    }
}

static void racing_eval_bake_minimap() {
    auto &v=racing_view;
    BoundingBox b=v.bounds;
    if (trial_gate_count) {
        b.min=b.max=(Vector3){v.gates[0].left.x,v.gates[0].left.y,v.gates[0].left.z};
        for (unsigned i=0;i<trial_gate_count;++i) {
            Vector3 left={v.gates[i].left.x,v.gates[i].left.y,v.gates[i].left.z};
            Vector3 right={v.gates[i].right.x,v.gates[i].right.y,v.gates[i].right.z};
            b.min=Vector3Min(b.min,Vector3Min(left,right));
            b.max=Vector3Max(b.max,Vector3Max(left,right));
        }
    }
    Vector3 center=Vector3Scale(Vector3Add(b.min,b.max),0.5f);
    v.map_extent=fmaxf(b.max.x-b.min.x,b.max.z-b.min.z)+160;
    v.map_camera={Vector3Add(center,(Vector3){0,2500,0}),center,(Vector3){0,0,-1},v.map_extent,CAMERA_ORTHOGRAPHIC};
    v.minimap=LoadRenderTexture(512,512);
    int disabled=0; float fog=0;
    SetShaderValue(v.lighting,v.shadow_enabled,&disabled,SHADER_UNIFORM_INT);
    SetShaderValue(v.lighting,v.eye_location,&v.map_camera.position,SHADER_UNIFORM_VEC3);
    SetShaderValue(v.lighting,v.fog_location,&fog,SHADER_UNIFORM_FLOAT);
    BeginTextureMode(v.minimap);
    ClearBackground((Color){35,43,37,255});
    BeginMode3D(v.map_camera);
    racing_eval_collect(v.map_camera,1,false,NULL);
    racing_draw_items(v.draw,v.lighting,v.alpha_location,v.shadow.depth);
    EndMode3D(); EndTextureMode();
}

static void racing_eval_input() {
    auto &v=racing_view;
    if (racing_ghosts.count) {
        int direction=IsKeyPressed(KEY_RIGHT_BRACKET)-IsKeyPressed(KEY_LEFT_BRACKET);
        if (direction) {
            racing_ghosts.selected=racing_next_live(racing_ghosts.selected,direction);
            v.follow_leader=v.manual=0;
        }
        if (IsKeyPressed(KEY_C)) { v.follow_leader=!v.follow_leader; v.manual=0; }
    }
    if (IsKeyPressed(KEY_ONE)) v.camera_mode=0;
    if (IsKeyPressed(KEY_TWO)) v.camera_mode=1;
    if (IsKeyPressed(KEY_THREE)) v.camera_mode=2;
    if (IsKeyPressed(KEY_FOUR)) v.camera_mode=3;
    if (IsKeyPressed(KEY_F1)) v.show_hud=!v.show_hud;
    if (IsKeyPressed(KEY_F2)) v.diagnostics=!v.diagnostics;
    if (IsKeyPressed(KEY_G)) v.show_gates=!v.show_gates;
    if (IsKeyPressed(KEY_L)) v.lidar=!v.lidar;
    if (IsKeyPressed(KEY_TAB) && (!racing_ghosts.count || racing_ghost_active(racing_ghosts.selected))) {
        v.manual=!v.manual; v.follow_leader=0;
    }
    v.distance=Clamp(v.distance-GetMouseWheelMove(),4,20);
    v.steering=(IsKeyDown(KEY_D)||IsKeyDown(KEY_RIGHT))-(IsKeyDown(KEY_A)||IsKeyDown(KEY_LEFT));
    v.throttle=IsKeyDown(KEY_W)||IsKeyDown(KEY_UP);
    v.brake=IsKeyDown(KEY_S)||IsKeyDown(KEY_DOWN);
}

static void racing_eval_select() {
    auto &v=racing_view;
    if (!racing_ghosts.count) return;
    int before=racing_ghosts.selected;
    if (v.follow_leader || !racing_ghost_active(before)) racing_ghosts.selected=racing_live_leader();
    if (before!=racing_ghosts.selected || racing_ghosts.selected<0) v.manual=0;
}

static const char *racing_reset_reason(const RacingEpisodeEnd *end) {
    const char *names[]={"none","collision / rollover","offroad / height",
        "invalid traversal","wrong way","stalled","time limit","lap completed"};
    return end->reason>=0 && end->reason<=7 ? names[end->reason] : "unknown";
}

static Camera3D racing_eval_camera(const RacingCarFrame *frame) {
    auto &v=racing_view;
    int mode=frame ? v.camera_mode : 3;
    if (mode==3) {
        v.camera_valid=0;
        return v.map_camera;
    }
    Vector3 p={frame->position[0],frame->position[1],frame->position[2]};
    Quaternion q={frame->rotation[0],frame->rotation[1],frame->rotation[2],frame->rotation[3]};
    Vector3 forward=Vector3RotateByQuaternion((Vector3){0,0,1},q);
    Vector3 heading=Vector3Normalize((Vector3){forward.x,0,forward.z});
    if (Vector3LengthSqr(heading)<0.5f) heading=(Vector3){0,0,1};
    Camera3D wanted={}; wanted.up=(Vector3){0,1,0}; wanted.fovy=65; wanted.projection=CAMERA_PERSPECTIVE;
    if (mode==0) {
        wanted.position=Vector3Add(p,(Vector3){0,50,0}); wanted.target=p; wanted.up=heading;
        wanted.projection=CAMERA_ORTHOGRAPHIC; wanted.fovy=v.distance*6;
    } else if (mode==1) {
        wanted.position=Vector3Add(Vector3Subtract(p,Vector3Scale(heading,v.distance)),(Vector3){0,3,0});
        wanted.target=Vector3Add(Vector3Add(p,(Vector3){0,0.8f,0}),Vector3Scale(heading,4+fminf(frame->speed*0.15f,8)));
    } else {
        wanted.position=Vector3Add(p,Vector3RotateByQuaternion((Vector3){0.35f,0.65f,0.25f},q));
        wanted.target=Vector3Add(wanted.position,forward); wanted.up=Vector3RotateByQuaternion((Vector3){0,1,0},q);
    }
    bool snap=!v.camera_valid || v.camera_selected!=racing_ghosts.selected || v.previous_mode!=mode
        || Vector3DistanceSqr(v.camera.target,wanted.target)>100*100;
    if (mode!=2 && !snap) {
        float blend=1-expf(-10*Clamp(GetFrameTime(),0,0.1f));
        wanted.position=Vector3Lerp(v.camera.position,wanted.position,blend);
        wanted.target=Vector3Lerp(v.camera.target,wanted.target,blend);
        wanted.up=Vector3Normalize(Vector3Lerp(v.camera.up,wanted.up,blend));
    }
    // Sparse camera ray against nearby chunks, cached at 10 Hz; no whole-track scan of triangles.
    if (mode==1) {
        Vector3 origin=Vector3Add(p,(Vector3){0,1,0});
        Vector3 delta=Vector3Subtract(wanted.position,origin); float distance=Vector3Length(delta);
        if (snap || GetTime()>v.camera_check) {
            v.camera_clearance=distance;
            Ray ray={origin,Vector3Scale(delta,1/fmaxf(distance,0.001f))};
            for (auto &surface : v.surfaces) {
                if (surface.group==6) continue;
                RayCollision bounds=GetRayCollisionBox(ray,surface.bounds);
                if (!bounds.hit || bounds.distance>distance) continue;
                RayCollision hit=GetRayCollisionMesh(ray,surface.mesh,MatrixIdentity());
                if (hit.hit && hit.distance<distance)
                    v.camera_clearance=fminf(v.camera_clearance,fmaxf(0.4f,hit.distance-0.25f));
            }
            v.camera_check=GetTime()+0.1;
        }
        wanted.position=Vector3Add(origin,Vector3Scale(delta,fminf(1,v.camera_clearance/fmaxf(distance,0.001f))));
    }
    v.camera=wanted; v.camera_valid=1; v.camera_selected=racing_ghosts.selected; v.previous_mode=mode;
    return wanted;
}

static void racing_eval_minimap(const RacingCarFrame *single) {
    auto &v=racing_view;
    int size=GetScreenWidth()<1000 ? 180 : 240;
    Rectangle box={(float)GetScreenWidth()-size-12,(float)GetScreenHeight()-size-42,(float)size,(float)size};
    DrawRectangle((int)box.x-3,(int)box.y-3,size+6,size+6,Fade((Color){12,16,23,255},0.9f));
    DrawTexturePro(v.minimap.texture,(Rectangle){0,0,512,-512},box,(Vector2){0,0},0,WHITE);
    int count=racing_ghosts.count ? racing_ghosts.count : single ? 1 : 0;
    for (int i=0;i<count;++i) {
        const RacingCarFrame *f=racing_ghosts.count ? &racing_ghosts.frames[i] : single;
        if (f->task_done || f->crashed) continue;
        Vector2 p=GetWorldToScreenEx((Vector3){f->position[0],f->position[1],f->position[2]},v.map_camera,512,512);
        p=(Vector2){box.x+p.x*size/512,box.y+p.y*size/512};
        if (p.x<box.x || p.x>box.x+size || p.y<box.y || p.y>box.y+size) continue;
        Color color=racing_ghosts.count ? racing_ghost_colors[i] : SKYBLUE;
        DrawCircleV(p,i==racing_ghosts.selected ? 5 : 3,color);
        if (i==racing_ghosts.selected) DrawCircleLines((int)p.x,(int)p.y,7,WHITE);
    }
}

static void racing_eval_draw(RacingCarFrame *frame, const PfOptixRay *rays,
        const PfOptixHit *hits, const RacingEpisodeEnd *last_end) {
    auto &v=racing_view;
    (void)last_end;
    bool live=frame && !frame->task_done && !frame->crashed;
    Camera3D camera=racing_eval_camera(live ? frame : NULL);
    Vector3 focus=live ? (Vector3){frame->position[0],frame->position[1],frame->position[2]} : v.map_camera.target;
    Camera3D light={Vector3Add(focus,Vector3Scale(v.sun,90)),focus,(Vector3){0,1,0},85,CAMERA_ORTHOGRAPHIC};
    Matrix light_vp=racing_camera_matrix(light,1,180);
    if (live && camera.projection!=CAMERA_ORTHOGRAPHIC) {
    rlSetClipPlanes(0.1,180);
    BeginTextureMode(v.shadow);
    ClearBackground(WHITE);
    BeginMode3D(light);
    racing_eval_collect(light,1,true,frame,180);
    // Glass does not cast an opaque shadow; masked foliage and decals keep alpha tests.
    v.draw.erase(std::remove_if(v.draw.begin(),v.draw.end(),[](const RacingDraw &item){ return item.alpha==2; }),v.draw.end());
    racing_draw_items(v.draw,v.shadow_shader,v.shadow_alpha,(Texture2D){});
    EndMode3D(); EndTextureMode();
    rlSetClipPlanes(0.1,6000);
    }
    Vector3 vf=Vector3Normalize(Vector3Subtract(camera.target,camera.position));
    Vector3 vr=Vector3Normalize(Vector3CrossProduct(vf,camera.up)), vu=Vector3CrossProduct(vr,vf);
    Vector2 resolution={(float)GetScreenWidth(),(float)GetScreenHeight()};
    float scale=camera.projection==CAMERA_ORTHOGRAPHIC ? 0 : tanf(camera.fovy*DEG2RAD*0.5f);
    float fog=camera.projection==CAMERA_ORTHOGRAPHIC ? 0 : 0.00012f;
    int shadows=live && camera.projection!=CAMERA_ORTHOGRAPHIC;
    SetShaderValue(v.lighting,v.eye_location,&camera.position,SHADER_UNIFORM_VEC3);
    SetShaderValue(v.lighting,v.fog_location,&fog,SHADER_UNIFORM_FLOAT);
    SetShaderValue(v.lighting,v.shadow_enabled,&shadows,SHADER_UNIFORM_INT);
    SetShaderValueMatrix(v.lighting,v.light_vp,light_vp);
    SetShaderValue(v.sky,v.sky_resolution,&resolution,SHADER_UNIFORM_VEC2);
    SetShaderValue(v.sky,v.sky_forward,&vf,SHADER_UNIFORM_VEC3);
    SetShaderValue(v.sky,v.sky_right,&vr,SHADER_UNIFORM_VEC3);
    SetShaderValue(v.sky,v.sky_up,&vu,SHADER_UNIFORM_VEC3);
    SetShaderValue(v.sky,v.sky_scale,&scale,SHADER_UNIFORM_FLOAT);
    BeginDrawing(); ClearBackground(SKYBLUE);
    BeginShaderMode(v.sky); DrawRectangle(0,0,GetScreenWidth(),GetScreenHeight(),WHITE); EndShaderMode();
    BeginMode3D(camera);
    racing_eval_collect(camera,resolution.x/resolution.y,true,frame);
    racing_draw_items(v.draw,v.lighting,v.alpha_location,v.shadow.depth);
    if (v.show_gates && live) racing_gates_draw(v.gates,trial_gate_count,frame);
    if (v.lidar && live) {
        for (int i=0;i<256;++i) {
            if (hits[i].triangle==-2) continue;
            Vector3 a={rays[i].origin.x,rays[i].origin.y,rays[i].origin.z};
            Vector3 d={rays[i].direction.x,rays[i].direction.y,rays[i].direction.z};
            Vector3 end=Vector3Add(a,Vector3Scale(d,hits[i].distance));
            Color color=hits[i].triangle<0 ? Fade(GRAY,0.15f) : hits[i].material==-1 ? ORANGE : SKYBLUE;
            DrawLine3D(a,end,Fade(color,0.35f));
            if (hits[i].triangle>=0) DrawSphere(end,0.045f,color);
        }
    }
    EndMode3D();
    if (v.show_hud) {
        racing_hud_draw(live ? frame : NULL,v.follow_leader,v.manual,v.diagnostics,v.lidar);
        racing_eval_minimap(frame);
    }
    EndDrawing();
}

static void racing_eval_close() {
    auto &v=racing_view;
    if (!v.opened) return;
    for (auto &surface : v.surfaces) UnloadMesh(surface.mesh);
    v.material.maps[0].texture=v.default_texture; v.material.shader=v.default_shader;
    v.material.maps[1].texture.id=v.material.maps[2].texture.id=0;
    v.material.maps[MATERIAL_MAP_HEIGHT].texture.id=0;
    UnloadMaterial(v.material);
    UnloadShader(v.lighting); UnloadShader(v.sky); UnloadShader(v.shadow_shader);
    UnloadRenderTexture(v.minimap); UnloadRenderTexture(v.shadow);
    racing_car_unload(&v.car);
    for (unsigned i=0;i<v.texture_count;++i) UnloadTexture(v.textures[i]);
    free(v.textures); free(v.images); free(v.alpha); free(v.colors); free(v.details);
    UnloadFont(racing_hud_font);
    CloseWindow(); v={};
}
