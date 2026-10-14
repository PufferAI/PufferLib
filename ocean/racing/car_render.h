#pragma once
#include "car.h"
#include "material_render.h"

// The cooked GLB has body meshes at the COM and wheel meshes centered on their pivots.
typedef struct RacingCarRender {
    Model model;
    int parts[60][3]; // wheel (-1 body), spins, alpha mode
    BoundingBox bounds[60];
    RacingMaterialDetail details[60];
    Mesh lod[60];
    unsigned flags[60]; // paint tint, double sided
    int loaded;
} RacingCarRender;

// Raylib uses 16-bit mesh indices. Merge only compatible parts that fit that limit.
static void racing_car_merge_meshes(Mesh *a, Mesh *b) {
    if (!a->vertexCount || !b->vertexCount || a->vertexCount+b->vertexCount>65535) return;
    Mesh joined={0};
    joined.vertexCount=a->vertexCount+b->vertexCount;
    joined.triangleCount=a->triangleCount+b->triangleCount;
    joined.vertices=(float*)MemAlloc(joined.vertexCount*12);
    joined.normals=(float*)MemAlloc(joined.vertexCount*12);
    joined.texcoords=(float*)MemAlloc(joined.vertexCount*8);
    joined.indices=(unsigned short*)MemAlloc(joined.triangleCount*6);
    Mesh *sources[2]={a,b}; int vertex=0, index=0;
    for (int part=0;part<2;++part) {
        Mesh *m=sources[part];
        memcpy(joined.vertices+vertex*3,m->vertices,m->vertexCount*12);
        if (m->normals) memcpy(joined.normals+vertex*3,m->normals,m->vertexCount*12);
        else memset(joined.normals+vertex*3,0,m->vertexCount*12);
        if (m->texcoords) memcpy(joined.texcoords+vertex*2,m->texcoords,m->vertexCount*8);
        else memset(joined.texcoords+vertex*2,0,m->vertexCount*8);
        for (int j=0;j<m->triangleCount*3;++j)
            joined.indices[index++]=(unsigned short)(vertex+(m->indices ? m->indices[j] : j));
        vertex+=m->vertexCount;
    }
    UnloadMesh(*a); UnloadMesh(*b);
    UploadMesh(&joined,false); *a=joined; memset(b,0,sizeof(*b));
}

static void racing_car_load(RacingCarRender *car) {
    car->model = LoadModel("ocean/racing/car.glb");
    if (car->model.meshCount != 60) {
        fprintf(stderr, "Car import failed: expected 60 meshes; run view_map.sh again\n");
        exit(1);
    }
    FILE *file = fopen("ocean/racing/car_parts.bin", "rb");
    assert(file);
    char magic[8];
    unsigned count;
    assert(fread(magic, 1, 8, file) == 8 && memcmp(magic, "PFCAR001", 8) == 0);
    assert(fread(&count, 4, 1, file) == 1 && count == 60);
    assert(fread(car->parts, sizeof(car->parts), 1, file) == 1);
    fclose(file);
    file = fopen("ocean/racing/car_render.bin", "rb");
    if (!file) {
        fprintf(stderr, "Run python3 ocean/racing/prepare_render.py to prepare viewer assets\n");
        exit(1);
    }
    assert(fread(magic,1,8,file)==8 && memcmp(magic,"PFCRND01",8)==0);
    assert(fread(&count,4,1,file)==1 && count==60);
    for (int i=0;i<60;++i) {
        assert(fread(&car->flags[i],4,1,file)==1);
        assert(fread(car->details[i].emission,4,3,file)==3);
    }
    fclose(file);
    file=fopen("ocean/racing/car_lod.bin","rb");
    assert(file && fread(magic,1,8,file)==8 && memcmp(magic,"PFCLOD01",8)==0);
    assert(fread(&count,4,1,file)==1 && count==60);
    for (int i=0;i<60;++i) {
        unsigned vertices;
        assert(fread(&vertices,4,1,file)==1 && vertices<=10000000 && vertices%3==0);
        Mesh *mesh=&car->lod[i];
        mesh->vertexCount=vertices; mesh->triangleCount=vertices/3;
        if (!vertices) continue;
        mesh->vertices=(float*)MemAlloc(vertices*12);
        mesh->texcoords=(float*)MemAlloc(vertices*8);
        mesh->normals=(float*)MemAlloc(vertices*12);
        assert(fread(mesh->vertices,12,vertices,file)==vertices);
        assert(fread(mesh->texcoords,8,vertices,file)==vertices);
        assert(fread(mesh->normals,12,vertices,file)==vertices);
        UploadMesh(mesh,false);
    }
    fclose(file);
    file = fopen("ocean/racing/car_materials.bin", "rb");
    assert(file);
    assert(fread(magic, 1, 8, file) == 8 && memcmp(magic, "PFCMAT01", 8) == 0);
    assert(fread(&count, 4, 1, file) == 1 && count == 60);
    for (unsigned i = 0; i < count; i++) {
        assert(fread(&car->details[i].normal_strength, 4, 1, file) == 1);
        assert(fread(&car->details[i].roughness, 4, 1, file) == 1);
        assert(fread(&car->details[i].metallic, 4, 1, file) == 1);
    }
    fclose(file);
    for (int i=0;i<60;++i) for (int j=i+1;j<60;++j) {
        if (car->model.meshMaterial[i]!=car->model.meshMaterial[j]
                || memcmp(car->parts[i],car->parts[j],sizeof(car->parts[i]))) continue;
        racing_car_merge_meshes(&car->model.meshes[i],&car->model.meshes[j]);
        racing_car_merge_meshes(&car->lod[i],&car->lod[j]);
    }
    for (int i = 0; i < 60; i++) {
        Mesh *mesh=car->model.meshes[i].vertexCount ? &car->model.meshes[i] : &car->lod[i];
        if (mesh->vertexCount) car->bounds[i]=GetMeshBoundingBox(*mesh);
    }
    for (int i = 0; i < car->model.materialCount; i++) {
        for (int slot = 0; slot <= MATERIAL_MAP_NORMAL; slot++) {
            Texture2D texture = car->model.materials[i].maps[slot].texture;
            if (texture.id > 1) {
                GenTextureMipmaps(&texture);
                SetTextureFilter(texture, TEXTURE_FILTER_TRILINEAR);
                SetTextureFilter(texture, TEXTURE_FILTER_ANISOTROPIC_8X);
                car->model.materials[i].maps[slot].texture = texture;
            }
        }
    }
    car->loaded = 1;
}

static Matrix racing_car_part_transform(const RacingCarRender *car,
        const RacingCarFrame *frame, int i) {
    Quaternion q={frame->rotation[0],frame->rotation[1],frame->rotation[2],frame->rotation[3]};
    Matrix body=MatrixMultiply(QuaternionToMatrix(q),MatrixTranslate(
        frame->position[0],frame->position[1],frame->position[2]));
    int wheel=car->parts[i][0];
    float scale=frame->wheelbase/2.75f;
    Matrix local=MatrixScale(scale,scale,scale);
    if (wheel>=0) {
        scale=frame->radius/0.34f;
        local=MatrixScale(scale,scale,scale);
        if (car->parts[i][1]) local=MatrixMultiply(local,MatrixRotateX(frame->wheel_spin[wheel]));
        local=MatrixMultiply(local,MatrixRotateY(frame->steer[wheel]));
        local=MatrixMultiply(local,MatrixTranslate((wheel%2 ? -0.5f : 0.5f)*frame->track_width,
            frame->wheel_y[wheel],(wheel<2 ? 0.5f : -0.5f)*frame->wheelbase));
    }
    return MatrixMultiply(local,body);
}

static void racing_car_draw_tinted(RacingCarRender *car, RacingCarFrame *frame, Shader shader,
                            int alpha_location, Vector3 eye, Color tint) {
    Quaternion q = {frame->rotation[0], frame->rotation[1], frame->rotation[2], frame->rotation[3]};
    Matrix body =
        MatrixMultiply(QuaternionToMatrix(q),
                       MatrixTranslate(frame->position[0], frame->position[1], frame->position[2]));
    Matrix transforms[60];
    float depths[60];
    int order[60];
    for (int i = 0; i < 60; i++) {
        transforms[i] = racing_car_part_transform(car,frame,i);
        BoundingBox box = car->bounds[i];
        Vector3 center =
            Vector3Transform(Vector3Scale(Vector3Add(box.min, box.max), 0.5f), transforms[i]);
        depths[i] = Vector3DistanceSqr(center, eye);
        order[i] = i;
    }
    for (int i = 1; i < 60; i++) {
        int value = order[i], j = i;
        while (j > 0 && car->parts[order[j - 1]][2] == 2 &&
               (car->parts[value][2] != 2 || depths[value] > depths[order[j - 1]])) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = value;
    }
    rlDisableBackfaceCulling();
    for (int n = 0; n < 60; n++) {
        int i = order[n], mode = car->parts[i][2];
        if (!car->model.meshes[i].vertexCount) continue;
        Material material = car->model.materials[car->model.meshMaterial[i]];
        material.shader = shader;
        racing_material_values(shader, &car->details[i]);
        SetShaderValue(shader, alpha_location, &mode, SHADER_UNIFORM_INT);
        if (mode == 2) {
            rlDisableDepthMask();
        } else {
            rlEnableDepthMask();
        }
        Color original = material.maps[MATERIAL_MAP_DIFFUSE].color;
        material.maps[MATERIAL_MAP_DIFFUSE].color = car->flags[i]&1 ? ColorTint(original, tint) : original;
        DrawMesh(car->model.meshes[i], material, transforms[i]);
        material.maps[MATERIAL_MAP_DIFFUSE].color = original;
    }
    rlEnableDepthMask();
    rlEnableBackfaceCulling();
    Vector3 sensor = Vector3Transform((Vector3){0, frame->lidar_mount_y, 0}, body);
    Vector3 up = Vector3RotateByQuaternion((Vector3){0, 0.08f, 0}, q);
    DrawCylinderEx(Vector3Subtract(sensor, up), Vector3Add(sensor, up), 0.1f, 0.1f, 12, DARKGRAY);
}

static void racing_car_unload(RacingCarRender *car) {
    for (int i=0;i<60;++i) if (car->lod[i].vertexCount) UnloadMesh(car->lod[i]);
    UnloadModel(car->model);
}

static void racing_car_draw(RacingCarRender *car, RacingCarFrame *frame, Shader shader,
                            int alpha_location, Vector3 eye) {
    racing_car_draw_tinted(car, frame, shader, alpha_location, eye, WHITE);
}
