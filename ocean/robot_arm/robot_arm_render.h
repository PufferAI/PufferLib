#pragma once

#define float3 ra_raymath_float3
#include "raymath.h"
#undef float3
#include "rlgl.h"

#define RA_EXPECTED_MESHES 11
#define RA_SHADOW_SIZE 4096
#define RA_DRAW_ITEMS 128

typedef struct RaRenderer {
    Model arm;
    Model cube;
    Model sphere;
    Model cylinder;
    Shader skin_shader;
    Shader depth_shader;
    RenderTexture2D shadow;
    Matrix inverse_bind[RA_LINKS];
    int light_direction_loc;
    int view_position_loc;
    int light_matrix_loc;
    int surface_kind_loc;
    int attempted;
    int loaded;
} RaRenderer;

typedef struct RaRenderHost {
    RaRenderer renderer;
    const char* model_glb;
    Camera3D camera;
    float camera_yaw;
    float camera_pitch;
    float camera_distance;
    int camera_initialized;
    int reset_requested;
} RaRenderHost;

typedef struct RaDrawItem {
    Mesh* mesh;
    Material* material;
    Matrix transform;
    Color color;
    int surface_kind;
} RaDrawItem;

typedef struct RaDrawList {
    RaDrawItem items[RA_DRAW_ITEMS];
    int count;
} RaDrawList;

static Matrix ra_matrix(RaPose pose) {
    Matrix rotation = QuaternionToMatrix(
        (Quaternion){pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w});
    return MatrixMultiply(
        rotation, MatrixTranslate(pose.position.x, pose.position.y, pose.position.z));
}

static Vector3 ra_vector3(RaVec3 value) {
    return (Vector3){value.x, value.y, value.z};
}

static void ra_mesh(RaDrawList* scene, Model* model, Matrix transform, Color color, int surface) {
    assert(scene->count < RA_DRAW_ITEMS);
    scene->items[scene->count++] =
        (RaDrawItem){&model->meshes[0], &model->materials[0], transform, color, surface};
}

static void ra_box(RaDrawList* scene, RaRenderer* renderer, Vector3 center, Vector3 size,
    Color color, int surface) {
    Matrix transform = MatrixMultiply(
        MatrixScale(size.x, size.y, size.z), MatrixTranslate(center.x, center.y, center.z));
    ra_mesh(scene, &renderer->cube, transform, color, surface);
}

static void ra_cylinder(
    RaDrawList* scene, RaRenderer* renderer, Vector3 a, Vector3 b, float radius, Color color) {
    Vector3 direction = Vector3Subtract(b, a);
    float length = Vector3Length(direction);
    Vector3 axis = Vector3CrossProduct((Vector3){0, 1, 0}, direction);
    Matrix rotation = MatrixIdentity();
    if (Vector3Length(axis) > 1.0e-7f) {
        rotation =
            MatrixRotate(Vector3Normalize(axis), acosf(ra_clamp(direction.y / length, -1, 1)));
    } else if (direction.y < 0) {
        rotation = MatrixRotateX(PI);
    }
    Matrix transform = MatrixMultiply(MatrixScale(radius, length, radius), rotation);
    transform = MatrixMultiply(transform, MatrixTranslate(a.x, a.y, a.z));
    ra_mesh(scene, &renderer->cylinder, transform, color, 0);
}
