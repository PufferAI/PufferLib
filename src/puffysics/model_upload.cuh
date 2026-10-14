#pragma once
#include <cuda_runtime.h>
#include "model.cuh"

template<class T> static inline bool pf_upload_array(const T* source,int count,const T** out) {
    *out=nullptr;
    if (!count) return true;
    T* data=nullptr;
    if (count<0 || !source || cudaMalloc((void**)&data,(size_t)count*sizeof(T))!=cudaSuccess) return false;
    if (cudaMemcpy(data,source,(size_t)count*sizeof(T),cudaMemcpyHostToDevice)!=cudaSuccess) {
        cudaFree(data); return false;
    }
    *out=data; return true;
}
static inline void pf_model_destroy(PfModel* m) {
    if (!m) return;
    cudaFree((void*)m->links); cudaFree((void*)m->dofs); cudaFree((void*)m->geoms);
    cudaFree((void*)m->pairs); cudaFree((void*)m->actuators); cudaFree((void*)m->equalities);
    cudaFree((void*)m->sites); cudaFree((void*)m->convexes); cudaFree((void*)m->vertices);
    cudaFree((void*)m->faces); cudaFree((void*)m->heightfields); cudaFree((void*)m->heights);
    cudaFree((void*)m->initial_qpos);
    cudaFree((void*)m->link_dof_offsets); cudaFree((void*)m->link_dofs);
    cudaFree((void*)m->block_offsets); cudaFree((void*)m->block_dofs);
    cudaFree((void*)m->dof_blocks); cudaFree((void*)m->dof_local);
    cudaFree((void*)m->matrix_offsets); cudaFree((void*)m->jacobian_indices); *m={};
}
// source must be the successful output of pf_compile_model. Destination must
// be empty; all model memory is shared across environments and remains const.
static inline bool pf_model_upload(const PfModel& source,PfModel* out) {
    if (!out || out->links || source.nv<=0) return false;
    PfModel m=source;
    m.links=nullptr; m.dofs=nullptr; m.geoms=nullptr; m.pairs=nullptr;
    m.actuators=nullptr; m.equalities=nullptr; m.sites=nullptr; m.convexes=nullptr;
    m.vertices=nullptr; m.faces=nullptr; m.heightfields=nullptr; m.heights=nullptr; m.initial_qpos=nullptr;
    m.link_dof_offsets=nullptr; m.link_dofs=nullptr; m.block_offsets=nullptr; m.block_dofs=nullptr;
    m.dof_blocks=nullptr; m.dof_local=nullptr; m.matrix_offsets=nullptr;
    m.jacobian_indices=nullptr;
    bool ok=pf_upload_array(source.links,source.link_count,&m.links)
        && pf_upload_array(source.dofs,source.nv,&m.dofs)
        && pf_upload_array(source.geoms,source.geom_count,&m.geoms)
        && pf_upload_array(source.pairs,source.pair_count,&m.pairs)
        && pf_upload_array(source.actuators,source.actuator_count,&m.actuators)
        && pf_upload_array(source.equalities,source.equality_count,&m.equalities)
        && pf_upload_array(source.sites,source.site_count,&m.sites)
        && pf_upload_array(source.convexes,source.convex_count,&m.convexes)
        && pf_upload_array(source.vertices,source.vertex_count,&m.vertices)
        && pf_upload_array(source.faces,source.face_count,&m.faces)
        && pf_upload_array(source.heightfields,source.heightfield_count,&m.heightfields)
        && pf_upload_array(source.heights,source.height_count,&m.heights)
        && pf_upload_array(source.initial_qpos,source.nq,&m.initial_qpos)
        && pf_upload_array(source.link_dof_offsets,source.block_count?source.link_count+1:0,&m.link_dof_offsets)
        && pf_upload_array(source.link_dofs,source.dependency_count,&m.link_dofs)
        && pf_upload_array(source.block_offsets,source.block_count?source.block_count+1:0,&m.block_offsets)
        && pf_upload_array(source.block_dofs,source.block_count?source.nv:0,&m.block_dofs)
        && pf_upload_array(source.dof_blocks,source.matrix_size?source.nv:0,&m.dof_blocks)
        && pf_upload_array(source.dof_local,source.matrix_size?source.nv:0,&m.dof_local)
        && pf_upload_array(source.matrix_offsets,source.matrix_size?source.nv:0,&m.matrix_offsets)
        && pf_upload_array(source.jacobian_indices,source.jacobian_indices?source.link_count*source.nv:0,&m.jacobian_indices);
    if (!ok) { pf_model_destroy(&m); return false; }
    *out=m; return true;
}
