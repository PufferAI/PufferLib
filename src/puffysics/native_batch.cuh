#pragma once
#include <cuda_runtime.h>
#include <limits.h>
#include <vector>
#include "articulated_step.cuh"

struct PfNativeBatch {
    PfState* states;
    PfWorkspace* workspaces;
    void* allocations[7];
    int count;
};
static inline void pf_native_batch_destroy(PfNativeBatch* b) {
    if (!b) return;
    cudaFree(b->states); cudaFree(b->workspaces);
    for (void* p:b->allocations) cudaFree(p);
    *b={};
}
__device__ static inline void pf_reset_state(const PfModel& m,PfState& s) {
    for (int i=0;i<m.nq;++i) s.qpos[i]=m.initial_qpos[i];
    for (int i=0;i<m.nv;++i) { s.qvel[i]=0; s.applied[i]=0; }
    for (int i=0;i<m.actuator_count;++i) { s.control[i]=0; s.activation[i]=0; }
    for (int i=0;i<m.link_count;++i) { s.force[i]=pf_v3(0,0,0); s.torque[i]=pf_v3(0,0,0); }
    s.time=0;
}
__global__ static void pf_native_reset_kernel(PfModel m,PfNativeBatch b,const unsigned char* mask) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=b.count || (mask && !mask[i])) return;
    pf_reset_state(m,b.states[i]);
    PfWorkspace& w=b.workspaces[i];
    w.status=PF_OK; w.row_count=0; w.contact_count=0; w.iterations=0; w.residual=0;
    if (!pf_kinematics(m,b.states[i],w)) w.status=PF_INVALID_STATE;
}
__global__ static void pf_native_step_kernel(PfModel m,PfNativeBatch b,PfStepOptions options,
        int substeps,const unsigned char* active) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=b.count || (active && !active[i]) || b.workspaces[i].status!=PF_OK) return;
    for (int step=0;step<substeps;++step)
        // Only reuse within this call: callers may edit state between launches.
        if (!pf_articulated_step(m,b.states[i],b.workspaces[i],options,step!=0)) break;
}
static inline cudaError_t pf_native_reset(PfModel m,PfNativeBatch b,
        const unsigned char* mask=nullptr,cudaStream_t stream=0) {
    if (b.count<=0) return cudaErrorInvalidValue;
    pf_native_reset_kernel<<<(b.count+63)/64,64,0,stream>>>(m,b,mask);
    return cudaGetLastError();
}
static inline cudaError_t pf_native_step(PfModel m,PfNativeBatch b,PfStepOptions options,
        int substeps=1,const unsigned char* active=nullptr,cudaStream_t stream=0) {
    if (b.count<=0 || substeps<=0) return cudaErrorInvalidValue;
    pf_native_step_kernel<<<(b.count+63)/64,64,0,stream>>>(m,b,options,substeps,active);
    return cudaGetLastError();
}
// Allocations are proportional to active-row/contact budgets, not all possible
// pairs. Stepping performs no allocation, synchronization, or host transfer.
static inline bool pf_native_batch_create(const PfModel& m,int count,int row_capacity,
        int contact_capacity,PfNativeBatch* out,bool compact_matrix=true,bool compact_jacobian=true) {
    if (!out || out->states || count<=0 || count>INT_MAX-63 || m.nv<=0 || row_capacity<0 || contact_capacity<0)
        return false;
    size_t n=m.nv,l=m.link_count,r=row_capacity,c=contact_capacity;
    compact_matrix=compact_matrix && m.block_count>0 && m.matrix_size>0;
    compact_jacobian=compact_jacobian && m.link_dof_offsets && m.jacobian_indices;
    size_t jacobian_size=compact_jacobian?(size_t)m.dependency_count:l*n;
    size_t matrix_size=compact_matrix?(size_t)m.matrix_size:n*n;
    if (m.link_count<=0 || m.nq<=0 || m.actuator_count<0 || n*n>INT_MAX
            || l*n>INT_MAX || r*n>INT_MAX) return false;
    size_t state_stride=(size_t)m.nq+2*n+2*(size_t)m.actuator_count;
    size_t work_stride=2*matrix_size+3*n+2*r*n;
    size_t per_world[7]={state_stride*sizeof(float),2*l*sizeof(PfVec3),
        work_stride*sizeof(float),2*jacobian_size*sizeof(PfVec3),l*sizeof(PfLinkState),
        r*sizeof(PfRow),c*sizeof(PfContact)};
    PfNativeBatch b={}; b.count=count;
    for (int i=0;i<7;++i) {
        if (!per_world[i]) continue;
        if (per_world[i]>SIZE_MAX/(size_t)count
                || cudaMalloc(&b.allocations[i],per_world[i]*(size_t)count)!=cudaSuccess) {
            pf_native_batch_destroy(&b); return false;
        }
    }
    std::vector<PfState> states(count);
    std::vector<PfWorkspace> work(count);
    // Inactive Jacobian entries remain zero for the immutable model topology.
    if (cudaMemset(b.allocations[3],0,per_world[3]*(size_t)count)!=cudaSuccess) {
        pf_native_batch_destroy(&b); return false;
    }
    for (int i=0;i<count;++i) {
        float* s=(float*)b.allocations[0]+i*state_stride;
        PfVec3* f=(PfVec3*)b.allocations[1]+(size_t)i*2*l;
        states[i]={s,s+m.nq,s+m.nq+n,s+m.nq+n+m.actuator_count,
            s+m.nq+n+2*m.actuator_count,f,f+l,0};
        float* p=(float*)b.allocations[2]+i*work_stride;
        PfVec3* j=(PfVec3*)b.allocations[3]+(size_t)i*2*jacobian_size;
        PfWorkspace& w=work[i];
        w.links=(PfLinkState*)b.allocations[4]+(size_t)i*l;
        w.linear_jacobian=j; w.angular_jacobian=j+jacobian_size;
        w.compact_matrix=compact_matrix;
        w.sparse_jacobian=m.link_dof_offsets!=nullptr;
        w.compact_jacobian=compact_jacobian;
        w.matrix=p; p+=matrix_size; w.factor=p; p+=matrix_size;
        w.rhs=p; p+=n; w.solution=p; p+=n; w.diagonal=p; p+=n;
        w.jacobian=p; p+=r*n; w.response=p;
        w.rows=r?(PfRow*)b.allocations[5]+(size_t)i*r:nullptr;
        w.contacts=c?(PfContact*)b.allocations[6]+(size_t)i*c:nullptr;
        w.row_capacity=row_capacity; w.contact_capacity=contact_capacity;
    }
    if (cudaMalloc((void**)&b.states,(size_t)count*sizeof(PfState))!=cudaSuccess
            || cudaMalloc((void**)&b.workspaces,(size_t)count*sizeof(PfWorkspace))!=cudaSuccess
            || cudaMemcpy(b.states,states.data(),(size_t)count*sizeof(PfState),cudaMemcpyHostToDevice)!=cudaSuccess
            || cudaMemcpy(b.workspaces,work.data(),(size_t)count*sizeof(PfWorkspace),cudaMemcpyHostToDevice)!=cudaSuccess
            || pf_native_reset(m,b)!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess) {
        pf_native_batch_destroy(&b); return false;
    }
    *out=b; return true;
}
