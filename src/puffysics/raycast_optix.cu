#define PF_OPTIX_DEVICE
#include "raycast_optix.cuh"
#include <optix_device.h>

extern "C" { __constant__ PfOptixQuery pf_optix_query; }
extern "C" __global__ void __raygen__pf_rays() {
    unsigned index = optixGetLaunchIndex().x;
    PfOptixRay ray = pf_optix_query.rays[index];
    float n2 = ray.direction.x*ray.direction.x + ray.direction.y*ray.direction.y +
        ray.direction.z*ray.direction.z;
    if (!isfinite(ray.origin.x) || !isfinite(ray.origin.y) || !isfinite(ray.origin.z) ||
        !isfinite(n2) || fabsf(n2-1) > 1.e-4f || !isfinite(ray.tmin) ||
        !isfinite(ray.tmax) || ray.tmin < 0 || ray.tmax <= ray.tmin) {
        pf_optix_query.hits[index] = {0, {}, -2, -1};
        return;
    }
    pf_optix_query.hits[index] = {ray.tmax, {}, -1, -1};
    optixTrace(pf_optix_query.handle, ray.origin, ray.direction, ray.tmin, ray.tmax,
        0, 255, OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0);
}
extern "C" __global__ void __miss__pf_rays() {}
extern "C" __global__ void __closesthit__pf_rays() {
    unsigned triangle = optixGetPrimitiveIndex();
    unsigned instance = optixGetInstanceId();
    bool dynamic = pf_optix_query.dynamic_vertices && instance != 0;
    const float3* v = (dynamic ? pf_optix_query.dynamic_vertices :
                       pf_optix_query.vertices) + triangle*3;
    float3 a = make_float3(v[1].x-v[0].x, v[1].y-v[0].y, v[1].z-v[0].z);
    float3 b = make_float3(v[2].x-v[0].x, v[2].y-v[0].y, v[2].z-v[0].z);
    float3 n = make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
    if (dynamic) n = optixTransformNormalFromObjectToWorldSpace(n);
    float inv = rsqrtf(fmaxf(n.x*n.x+n.y*n.y+n.z*n.z, 1.e-30f));
    float3 direction = optixGetWorldRayDirection();
    if (n.x*direction.x+n.y*direction.y+n.z*direction.z > 0) {
        inv = -inv;
    }
    unsigned tag = dynamic ?
        ((pf_optix_query.dynamic_object_base+instance) << 12) |
         pf_optix_query.dynamic_materials[triangle] :
         pf_optix_query.materials[triangle];
    pf_optix_query.hits[optixGetLaunchIndex().x] = {optixGetRayTmax(),
        make_float3(n.x*inv, n.y*inv, n.z*inv), (int)triangle,(int)tag};
}
