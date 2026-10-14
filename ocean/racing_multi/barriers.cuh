#pragma once
#include <algorithm>
#include <vector>
#include "rewards.cuh"
#include "../../src/puffysics/collision.cuh"
#include "../../src/puffysics/contact_solver.cuh"

// Immutable map BVH for box/triangle overlap queries (OptiX's GAS is ray-query only).
// Leaves reference the existing GPU map vertices; no per-race geometry copies.
struct RacingBarrierNode { PfVec3 low, high; int first, count, escape; };
struct RacingBarrierMesh {
    RacingBarrierNode *nodes;
    int *indices, count, wall_nodes;
    const float3 *vertices;
    const unsigned *materials;
    PfManifold *contacts;
    PfBody *bodies;
    float crash_speed;
};
static RacingBarrierMesh racing_barriers;
static constexpr int RACING_BARRIER_CONTACTS = 24;
static constexpr int RACING_IMPACT_COOLDOWN = 60; // 0.25 s at 240 Hz.

__device__ static float racing_impact_cost(float impulse, float inverse_mass) {
    float delta_v = impulse * inverse_mass;
    return fminf(fmaxf(delta_v - 0.75f, 0.0f) * 0.008f, 0.08f);
}
__device__ static float racing_wall_impact_cost(float impulse, float inverse_mass) {
    float excess = fmaxf(impulse * inverse_mass - 0.75f, 0.0f);
    return fminf(0.015f * excess * excess, 0.4f);
}

// Explicit collision roles: road/curb side faces must never become chassis walls.
__host__ __device__ static bool racing_wall_material(unsigned m) {
    return m == 0 || m == 1 || m == 2 || m == 8 || m == 13
        || m == 31 || m == 33 || m == 34 || m == 67;
}
__host__ __device__ static bool racing_floor_material(unsigned m) {
    return m == 32 || m == 44 || m == 50 || m == 52 || m == 53
        || m == 55 || m == 56 || m == 58 || m == 59;
}

__host__ __device__ static PfVec3 racing_min(PfVec3 a, PfVec3 b) {
    return pf_v3(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z));
}
__host__ __device__ static PfVec3 racing_max(PfVec3 a, PfVec3 b) {
    return pf_v3(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z));
}
__host__ __device__ static PfVec3 racing_vertex(float3 p) { return pf_v3(p.x,p.y,p.z); }
__device__ static PfVec3 racing_barrier_extent(const PfBody &body) {
    PfVec3 x=pf_quat_rotate(body.rotation,pf_v3(1,0,0));
    PfVec3 y=pf_quat_rotate(body.rotation,pf_v3(0,1,0));
    PfVec3 z=pf_quat_rotate(body.rotation,pf_v3(0,0,1));
    PfVec3 h=body.half_extents;
    return pf_v3(fabsf(x.x)*h.x+fabsf(y.x)*h.y+fabsf(z.x)*h.z,
        fabsf(x.y)*h.x+fabsf(y.y)*h.y+fabsf(z.y)*h.z,
        fabsf(x.z)*h.x+fabsf(y.z)*h.y+fabsf(z.z)*h.z);
}
__device__ static float racing_component(PfVec3 p, int axis) {
    return axis == 0 ? p.x : axis == 1 ? p.y : p.z;
}

// Clip a finite triangle against all six box faces. A plane hit by itself is not contact.
// The centroid of the clipped polygon lies on the triangle and inside the chassis.
__device__ static bool racing_barrier_patch(const PfBody &body, const PfVec3 triangle[3],
    PfVec3 *point) {
    PfQuat inverse = pf_quat_conjugate(body.rotation);
    PfVec3 a[12], b[12];
    int count = 3;
    for (int i = 0; i < 3; i++) a[i] = pf_quat_rotate(inverse, pf_sub(triangle[i],body.position));
    PfVec3 low=racing_min(a[0],racing_min(a[1],a[2]));
    PfVec3 high=racing_max(a[0],racing_max(a[1],a[2]));
    PfVec3 h=pf_add(body.half_extents,pf_v3(0.001f,0.001f,0.001f));
    if (low.x>h.x || high.x<-h.x || low.y>h.y || high.y<-h.y ||
        low.z>h.z || high.z<-h.z) return false;
    for (int face = 0; face < 6; face++) {
        int axis = face / 2, output = 0;
        float sign = face % 2 ? -1 : 1;
        float extent = racing_component(body.half_extents,axis) + 0.001f;
        for (int i = 0; i < count; i++) {
            PfVec3 p = a[i], q = a[(i+1)%count];
            float dp = sign*racing_component(p,axis)-extent;
            float dq = sign*racing_component(q,axis)-extent;
            if (dp <= 0) b[output++] = p;
            if ((dp < 0 && dq > 0) || (dp > 0 && dq < 0))
                b[output++] = pf_add(p,pf_scale(pf_sub(q,p),dp/(dp-dq)));
        }
        count = output;
        if (!count) return false;
        for (int i = 0; i < count; i++) a[i] = b[i];
    }
    PfVec3 local = {};
    for (int i = 0; i < count; i++) local = pf_add(local,a[i]);
    *point = pf_add(body.position,pf_quat_rotate(body.rotation,pf_scale(local,1.0f/count)));
    return true;
}
__device__ static float racing_barrier_radius(const PfBody &body, PfVec3 normal) {
    PfVec3 n = pf_quat_rotate(pf_quat_conjugate(body.rotation),normal);
    return fabsf(n.x)*body.half_extents.x + fabsf(n.y)*body.half_extents.y + fabsf(n.z)*body.half_extents.z;
}
__device__ static PfBody racing_barrier_pose(const PfBody &before, const PfBody &after, float t) {
    PfBody pose = after;
    pose.position = pf_add(before.position,pf_scale(pf_sub(after.position,before.position),t));
    PfQuat a=before.rotation,b=after.rotation;
    float sign = a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z < 0 ? -1 : 1;
    pose.rotation = pf_quat_normalize(PfQuat{a.w*(1-t)+b.w*t*sign,a.x*(1-t)+b.x*t*sign,
        a.y*(1-t)+b.y*t*sign,a.z*(1-t)+b.z*t*sign});
    return pose;
}
__device__ static bool racing_barrier_contact(const PfBody &before, const PfBody &after,
    const PfVec3 triangle[3], unsigned material, PfVec3 *normal, PfContactPoint *contact) {
    PfVec3 n = pf_cross(pf_sub(triangle[1],triangle[0]),pf_sub(triangle[2],triangle[0]));
    float n2 = pf_length_squared(n);
    if (n2 < 1e-12f) return false;
    n = pf_scale(n,rsqrtf(n2));
    if (racing_floor_material(material)) {
        // Upright cars use wheel suspension only, including every curb edge.
        PfVec3 up = pf_quat_rotate(after.rotation,pf_v3(0,1,0));
        if (up.y > 0.25f || fabsf(n.y) < 0.7f) return false;
        if (n.y < 0) n = pf_scale(n,-1);
    } else {
        if (!racing_wall_material(material)) return false;
        if (pf_dot(pf_sub(before.position,triangle[0]),n) < 0) n = pf_scale(n,-1);
    }
    float old_gap = pf_dot(pf_sub(before.position,triangle[0]),n)-racing_barrier_radius(before,n);
    float gap = pf_dot(pf_sub(after.position,triangle[0]),n)-racing_barrier_radius(after,n);
    if (gap > 0.001f) return false;
    PfVec3 point;
    PfBody pose = after;
    if (!racing_barrier_patch(pose,triangle,&point)) {
        // Linear sweep fallback for thin walls crossed during this substep. Finite-triangle
        // clipping at the impact pose prevents the wall's infinite plane blocking empty space.
        if (gap >= old_gap) return false;
        float t = racing_clamp(old_gap/(old_gap-gap),0,1);
        pose = racing_barrier_pose(before,after,t);
        if (!racing_barrier_patch(pose,triangle,&point)) return false;
    }
    // Project the clipped point onto the penetrating box face along -normal.
    PfQuat inverse = pf_quat_conjugate(pose.rotation);
    PfVec3 local = pf_quat_rotate(inverse,pf_sub(point,pose.position));
    PfVec3 direction = pf_quat_rotate(inverse,pf_scale(n,-1));
    float exit = 1e30f;
    for (int axis = 0; axis < 3; axis++) {
        float d = racing_component(direction,axis);
        if (fabsf(d) > 1e-7f) {
            float edge = copysignf(racing_component(pose.half_extents,axis),d);
            exit = fminf(exit,(edge-racing_component(local,axis))/d);
        }
    }
    local = pf_add(local,pf_scale(direction,fmaxf(0,exit)));
    PfVec3 a = pf_add(after.position,pf_quat_rotate(after.rotation,local));
    float separation = pf_dot(pf_sub(a,point),n);
    if (separation > 0.001f) return false;
    *normal = n;
    *contact = {a,point,separation};
    return true;
}

__global__ static void racing_barrier_solve(Env *envs, int count, RacingBarrierMesh mesh) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    Env &e = envs[i];
    e.wall_contact = false;
    if (e.task.done || e.car.crashed) return;
    PfBody *bodies = mesh.bodies+i*2;
    bodies[0] = e.car.body;
    bodies[1] = {}; // Infinite-mass static map.
    bodies[1].rotation = pf_quat_identity();
    if (!pf_vec_valid(bodies[0].position) || !pf_quat_valid(bodies[0].rotation)) return;
    PfBody before = bodies[0];
    before.position = e.race_before;
    before.rotation = e.race_rotation;
    PfVec3 old_extent=racing_barrier_extent(before);
    PfVec3 new_extent=racing_barrier_extent(bodies[0]);
    float dot = before.rotation.w*bodies[0].rotation.w
        + before.rotation.x*bodies[0].rotation.x
        + before.rotation.y*bodies[0].rotation.y
        + before.rotation.z*bodies[0].rotation.z;
    float angle = 2*acosf(fminf(1.0f,fabsf(dot)));
    // Intermediate rotation can extend beyond both endpoint boxes. Its maximum
    // vertex displacement is bounded by radius*angle along the short rotation.
    float rotation_pad = pf_length(before.half_extents)*angle+0.002f;
    PfVec3 pad=pf_v3(rotation_pad,rotation_pad,rotation_pad);
    PfVec3 low=pf_sub(racing_min(pf_sub(before.position,old_extent),
        pf_sub(bodies[0].position,new_extent)),pad);
    PfVec3 high=pf_add(racing_max(pf_add(before.position,old_extent),
        pf_add(bodies[0].position,new_extent)),pad);
    PfManifold *contacts = mesh.contacts+i*RACING_BARRIER_CONTACTS;
    int used = 0;
    float wall_closing_speed = 0;
    bool wall_patch[RACING_BARRIER_CONTACTS] = {};
    PfVec3 up = pf_quat_rotate(bodies[0].rotation,pf_v3(0,1,0));
    int limit = up.y > 0.25f ? mesh.wall_nodes : mesh.count;
    for (int node = 0; node < limit;) {
        RacingBarrierNode b = mesh.nodes[node];
        if (low.x>b.high.x || high.x<b.low.x || low.y>b.high.y || high.y<b.low.y ||
            low.z>b.high.z || high.z<b.low.z) { node=b.escape; continue; }
        node++;
        for (int j = 0; j < b.count; j++) {
            int index = mesh.indices[b.first+j]*3;
            PfVec3 tri[3] = {racing_vertex(mesh.vertices[index]),racing_vertex(mesh.vertices[index+1]),
                racing_vertex(mesh.vertices[index+2])};
            PfVec3 normal;
            PfContactPoint point;
            if (!racing_barrier_contact(before,bodies[0],tri,mesh.materials[index/3],&normal,&point)) continue;
            bool wall = racing_wall_material(mesh.materials[index/3]);
            if (wall) {
                e.wall_contact = true;
                float closing = -pf_dot(pf_point_velocity(&bodies[0],point.point_a),normal);
                wall_closing_speed = fmaxf(wall_closing_speed,closing);
            }
            int target = -1;
            for (int k = 0; k < used; k++) {
                // Merge triangles on the same plane into one contact patch.
                if (pf_dot(contacts[k].normal,normal)>0.9999f && fabsf(pf_dot(normal,
                    pf_sub(contacts[k].points[0].point_b,point.point_b)))<0.005f) { target=k; break; }
            }
            if (target < 0) {
                if (used == RACING_BARRIER_CONTACTS) {
                    // Keep the deepest patches if highly detailed geometry exhausts the budget.
                    e.barrier_overflow++;
                    float shallowest = -1e30f;
                    for (int k = 0; k < used; k++) {
                        float deepest = 0;
                        for (int p = 0; p < contacts[k].point_count; p++)
                            deepest = fminf(deepest,contacts[k].points[p].separation);
                        if (deepest > shallowest) { shallowest=deepest; target=k; }
                    }
                    if (point.separation >= shallowest) continue;
                } else target = used++;
                contacts[target] = {};
                wall_patch[target] = wall;
                contacts[target].body_a = 0;
                contacts[target].body_b = 1;
                contacts[target].normal = normal;
                contacts[target].static_friction = contacts[target].dynamic_friction = 0.35f;
                contacts[target].restitution = 0.05f;
                pf_manifold_tangents(&contacts[target]);
            }
            wall_patch[target] = wall_patch[target] || wall;
            PfManifold &m = contacts[target];
            if (m.point_count < PF_MAX_MANIFOLD_POINTS) m.points[m.point_count++] = point;
            else {
                int shallow = 0;
                for (int k = 1; k < m.point_count; k++)
                    if (m.points[k].separation > m.points[shallow].separation) shallow = k;
                if (point.separation < m.points[shallow].separation) m.points[shallow] = point;
            }
        }
    }
    if (!used) return;
    if (e.wall_contact) {
        float cost = RACING_CONTACT_COST_PER_SECOND * RACING_DT;
        e.race_reward -= cost;
        e.wall_contact_penalty += cost;
        e.wall_contact_seconds += RACING_DT;
    }
    pf_solve_velocity_contacts(bodies,contacts,used);
    float impact = 0;
    for (int k = 0; k < used; k++)
        if (wall_patch[k]) impact = fmaxf(impact, contacts[k].normal_impulse);
    float cost = racing_wall_impact_cost(impact, bodies[0].inverse_mass);
    if (cost > 0 && e.last_wall_impact_tick + RACING_IMPACT_COOLDOWN <= e.task.ticks) {
        e.race_reward -= cost;
        e.wall_impact_penalty += cost;
        e.wall_impacts++;
        e.last_wall_impact_tick = e.task.ticks;
    }
    // Static thin surfaces need full normal recovery, not slow overlap decay. Preserve
    // tangent displacement and rotation. Re-evaluate other planes after each translation.
    PfVec3 shift = {};
    for (int pass = 0; pass < 4; pass++) for (int k = 0; k < used; k++) {
        PfManifold &m = contacts[k];
        float depth = 0;
        for (int p = 0; p < m.point_count; p++)
            depth = fmaxf(depth,-m.points[p].separation-pf_dot(shift,m.normal)-0.001f);
        shift = pf_add(shift,pf_scale(m.normal,depth));
    }
    bodies[0].position = pf_add(bodies[0].position,shift);
    e.car.body = bodies[0];
    if (wall_closing_speed >= mesh.crash_speed) e.car.crashed = 1;
    e.barrier_contacts += used;
}

static void racing_barrier_create(const PfOptixMesh &mesh, int cars, float crash_speed) {
    std::vector<float3> vertices((size_t)mesh.triangles*3);
    pf_optix_cuda(cudaMemcpy(vertices.data(),mesh.vertices,vertices.size()*sizeof(float3),cudaMemcpyDeviceToHost));
    struct Triangle { PfVec3 low, high, centre; int id; };
    std::vector<unsigned> materials(mesh.triangles);
    pf_optix_cuda(cudaMemcpy(materials.data(),mesh.materials,materials.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
    std::vector<Triangle> triangles;
    for (unsigned i = 0; i < mesh.triangles; i++) {
        if (!racing_wall_material(materials[i]) && !racing_floor_material(materials[i])) continue;
        PfVec3 a=racing_vertex(vertices[3*i]),b=racing_vertex(vertices[3*i+1]),c=racing_vertex(vertices[3*i+2]);
        PfVec3 low=racing_min(a,racing_min(b,c)),high=racing_max(a,racing_max(b,c));
        if (racing_floor_material(materials[i])) {
            PfVec3 normal=pf_cross(pf_sub(b,a),pf_sub(c,a));
            if (normal.y*normal.y < 0.49f*pf_length_squared(normal)) continue;
        }
        triangles.push_back({low,high,pf_scale(pf_add(low,high),0.5f),(int)i});
    }
    std::vector<RacingBarrierNode> nodes;
    nodes.reserve(triangles.size()/2);
    assert(!triangles.empty());
    auto wall_end = std::partition(triangles.begin(),triangles.end(),
        [&](const Triangle &t) { return racing_wall_material(materials[t.id]); });
    int wall_count = (int)(wall_end-triangles.begin());
    auto build = [&](auto &&self, int first, int end)->void {
        int index=nodes.size();
        PfVec3 low=triangles[first].low,high=triangles[first].high;
        for (int i=first+1;i<end;i++) { low=racing_min(low,triangles[i].low); high=racing_max(high,triangles[i].high); }
        nodes.push_back({low,high,first,end-first,0});
        if (end-first>8) {
            PfVec3 extent=pf_sub(high,low);
            int axis=extent.x>extent.y ? 0:1;
            if (extent.z>(axis==0?extent.x:extent.y)) axis=2;
            int mid=(first+end)/2;
            std::nth_element(triangles.begin()+first,triangles.begin()+mid,triangles.begin()+end,
                [axis](const Triangle &a,const Triangle &b) {
                    return (axis==0?a.centre.x:axis==1?a.centre.y:a.centre.z)
                        <(axis==0?b.centre.x:axis==1?b.centre.y:b.centre.z);
                });
            nodes[index].count=0;
            self(self,first,mid); self(self,mid,end);
        }
        nodes[index].escape=nodes.size();
    };
    if (wall_count) build(build,0,wall_count);
    int wall_nodes = (int)nodes.size();
    if (wall_count < (int)triangles.size()) build(build,wall_count,(int)triangles.size());
    std::vector<int> indices(triangles.size());
    for (size_t i=0;i<triangles.size();i++) indices[i]=triangles[i].id;
    auto &b=racing_barriers;
    b.count=nodes.size(); b.wall_nodes=wall_nodes;
    b.crash_speed=crash_speed;
    b.vertices=mesh.vertices; b.materials=mesh.materials;
    pf_optix_cuda(cudaMalloc(&b.nodes,nodes.size()*sizeof(RacingBarrierNode)));
    pf_optix_cuda(cudaMalloc(&b.indices,indices.size()*sizeof(int)));
    pf_optix_cuda(cudaMalloc(&b.contacts,(size_t)cars*RACING_BARRIER_CONTACTS*sizeof(PfManifold)));
    pf_optix_cuda(cudaMalloc(&b.bodies,(size_t)cars*2*sizeof(PfBody)));
    pf_optix_cuda(cudaMemcpy(b.nodes,nodes.data(),nodes.size()*sizeof(RacingBarrierNode),cudaMemcpyHostToDevice));
    pf_optix_cuda(cudaMemcpy(b.indices,indices.data(),indices.size()*sizeof(int),cudaMemcpyHostToDevice));
    printf("Barrier contacts: %d BVH nodes, shared map triangles\n",b.count);
}
static void racing_barrier_close() {
    cudaFree(racing_barriers.nodes); cudaFree(racing_barriers.indices);
    cudaFree(racing_barriers.contacts); cudaFree(racing_barriers.bodies);
    racing_barriers={};
}
