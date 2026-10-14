#pragma once
#include <algorithm>
#include <map>
#include <vector>

struct RacingEvalSurface { Mesh mesh; BoundingBox bounds; unsigned material, group; };
struct RacingDraw {
    Mesh *mesh;
    Material material;
    Matrix transform;
    RacingMaterialDetail detail;
    Texture2D base, normal, detail_texture;
    Color color;
    int alpha, double_sided;
    float depth;
};

static Matrix racing_camera_matrix(Camera3D camera, float aspect, float far_plane=6000) {
    Matrix view = MatrixLookAt(camera.position, camera.target, camera.up);
    float fov = camera.fovy * DEG2RAD;
    Matrix projection = camera.projection == CAMERA_ORTHOGRAPHIC
        ? MatrixOrtho(-camera.fovy*aspect/2, camera.fovy*aspect/2,
                      -camera.fovy/2, camera.fovy/2, 0.1, far_plane)
        : MatrixPerspective(fov, aspect, 0.1, far_plane);
    return MatrixMultiply(view, projection);
}
static bool racing_visible(BoundingBox bounds, Matrix clip) {
    unsigned outside = 63;
    for (int i=0; i<8; ++i) {
        Vector3 p = {i&1 ? bounds.max.x : bounds.min.x,
                     i&2 ? bounds.max.y : bounds.min.y,
                     i&4 ? bounds.max.z : bounds.min.z};
        float x=clip.m0*p.x+clip.m4*p.y+clip.m8*p.z+clip.m12;
        float y=clip.m1*p.x+clip.m5*p.y+clip.m9*p.z+clip.m13;
        float z=clip.m2*p.x+clip.m6*p.y+clip.m10*p.z+clip.m14;
        float w=clip.m3*p.x+clip.m7*p.y+clip.m11*p.z+clip.m15;
        outside &= (x < -w ? 1u : 0u) | (x > w ? 2u : 0u)
            | (y < -w ? 4u : 0u) | (y > w ? 8u : 0u)
            | (z < -w ? 16u : 0u) | (z > w ? 32u : 0u);
    }
    return outside == 0;
}
static void racing_chunk_mesh(RacingEvalSurface source, std::vector<RacingEvalSurface>& chunks) {
    std::map<std::pair<int,int>,std::vector<int>> cells;
    for (int i=0; i<source.mesh.vertexCount; i+=3) {
        float *p=source.mesh.vertices+i*3;
        int x=(int)floorf((p[0]+p[3]+p[6])/300);
        int z=(int)floorf((p[2]+p[5]+p[8])/300);
        cells[{x,z}].push_back(i);
    }
    for (auto &cell : cells) {
        RacingEvalSurface chunk = {};
        chunk.material=source.material; chunk.group=source.group;
        Mesh &mesh=chunk.mesh;
        mesh.vertexCount=(int)cell.second.size()*3;
        mesh.triangleCount=mesh.vertexCount/3;
        mesh.vertices=(float*)MemAlloc(mesh.vertexCount*12);
        mesh.normals=(float*)MemAlloc(mesh.vertexCount*12);
        mesh.texcoords=(float*)MemAlloc(mesh.vertexCount*8);
        for (size_t j=0; j<cell.second.size(); ++j) {
            int i=cell.second[j];
            memcpy(mesh.vertices+j*9,source.mesh.vertices+i*3,36);
            memcpy(mesh.normals+j*9,source.mesh.normals+i*3,36);
            memcpy(mesh.texcoords+j*6,source.mesh.texcoords+i*2,24);
        }
        chunk.bounds=GetMeshBoundingBox(mesh);
        UploadMesh(&mesh,false);
        chunks.push_back(chunk);
    }
    MemFree(source.mesh.vertices); MemFree(source.mesh.normals); MemFree(source.mesh.texcoords);
}
static void racing_draw_items(std::vector<RacingDraw>& items, Shader shader,
        int alpha_location, Texture2D shadow) {
    std::stable_sort(items.begin(),items.end(),[](const RacingDraw &a,const RacingDraw &b) {
        if ((a.alpha==2)!=(b.alpha==2)) return a.alpha!=2;
        if (a.alpha==2) return a.depth>b.depth;
        if (a.base.id!=b.base.id) return a.base.id<b.base.id;
        return a.normal.id<b.normal.id;
    });
    for (auto &item : items) {
        Material m=item.material;
        m.shader=shader;
        MaterialMap saved[4]={m.maps[0],m.maps[1],m.maps[2],m.maps[MATERIAL_MAP_HEIGHT]};
        m.maps[0].texture=item.base; m.maps[0].color=item.color;
        m.maps[1].texture=item.detail_texture; m.maps[2].texture=item.normal;
        m.maps[MATERIAL_MAP_HEIGHT].texture=shadow;
        racing_material_values(shader,&item.detail);
        SetShaderValue(shader,alpha_location,&item.alpha,SHADER_UNIFORM_INT);
        if (item.alpha==2) rlDisableDepthMask(); else rlEnableDepthMask();
        if (item.double_sided) rlDisableBackfaceCulling(); else rlEnableBackfaceCulling();
        DrawMesh(*item.mesh,m,item.transform);
        m.maps[0]=saved[0]; m.maps[1]=saved[1]; m.maps[2]=saved[2];
        m.maps[MATERIAL_MAP_HEIGHT]=saved[3];
    }
    rlEnableDepthMask(); rlEnableBackfaceCulling();
}
