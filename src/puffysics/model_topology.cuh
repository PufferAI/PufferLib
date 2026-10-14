#pragma once
#include <algorithm>
#include <vector>
#include "model.cuh"

struct PfTopologyStorage {
    std::vector<int> link_offsets, link_dofs, block_offsets, block_dofs;
    std::vector<int> dof_blocks, dof_local, matrix_offsets;
    std::vector<int> jacobian_indices;
};

// Called after joint validation and mimic coordinate aliasing. Coordinates
// appearing on the same link share a mass block, including across mimic trees.
// Contact/equality connections do not merge mass blocks: their rows couple them.
static inline void pf_compile_topology(const std::vector<PfLink>& links,
        int nv, PfTopologyStorage& t, PfModel& m) {
    t={}; t.link_offsets.push_back(0);
    std::vector<int> parent(nv), ids(nv,-1);
    for (int d=0;d<nv;++d) parent[d]=d;
    auto root=[&](int d) {
        while (parent[d]!=d) { parent[d]=parent[parent[d]]; d=parent[d]; }
        return d;
    };
    for (const PfLink& link:links) {
        std::vector<int> active;
        if (link.parent>=0)
            active.assign(t.link_dofs.begin()+t.link_offsets[link.parent],
                t.link_dofs.begin()+t.link_offsets[link.parent+1]);
        for (int k=0;k<pf_joint_nv(link.joint);++k) {
            int d=link.dof+k;
            auto at=std::lower_bound(active.begin(),active.end(),d);
            if (at==active.end() || *at!=d) active.insert(at,d);
        }
        for (int d:active) parent[root(d)]=root(active.front());
        t.link_dofs.insert(t.link_dofs.end(),active.begin(),active.end());
        t.link_offsets.push_back((int)t.link_dofs.size());
    }
    int blocks=0;
    for (int d=0;d<nv;++d) if (ids[root(d)]<0) ids[root(d)]=blocks++;
    t.block_offsets.assign(blocks+1,0);
    for (int d=0;d<nv;++d) ++t.block_offsets[ids[root(d)]+1];
    for (int b=0;b<blocks;++b) t.block_offsets[b+1]+=t.block_offsets[b];
    std::vector<int> next=t.block_offsets;
    t.block_dofs.resize(nv);
    for (int d=0;d<nv;++d) t.block_dofs[next[ids[root(d)]]++]=d;
    t.dof_blocks.resize(nv); t.dof_local.resize(nv); t.matrix_offsets.resize(nv);
    int size=0;
    for (int b=0;b<blocks;++b) for (int at=t.block_offsets[b];at<t.block_offsets[b+1];++at) {
        int d=t.block_dofs[at], local=at-t.block_offsets[b];
        t.dof_blocks[d]=b; t.dof_local[d]=local; t.matrix_offsets[d]=size;
        size+=local+1;
    }
    m.link_dof_offsets=t.link_offsets.data(); m.link_dofs=t.link_dofs.data();
    m.block_offsets=t.block_offsets.data(); m.block_dofs=t.block_dofs.data();
    m.dependency_count=(int)t.link_dofs.size(); m.block_count=blocks;
    m.dof_blocks=t.dof_blocks.data(); m.dof_local=t.dof_local.data();
    m.matrix_offsets=t.matrix_offsets.data(); m.matrix_size=size;
    t.jacobian_indices.assign(links.size()*(size_t)nv,-1);
    for (int l=0;l<(int)links.size();++l)
        for (int at=t.link_offsets[l];at<t.link_offsets[l+1];++at)
            t.jacobian_indices[l*nv+t.link_dofs[at]]=at;
    m.jacobian_indices=t.jacobian_indices.data();
}
