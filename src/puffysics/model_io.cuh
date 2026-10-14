#pragma once
#include <stdio.h>
#include <string.h>
#include "model_builder.cuh"
#include "state.cuh"

// Versioned native offline cache, not an interchange format: little-endian,
// IEEE float, same structure ABI. Size tags reject mismatched builds. No device
// addresses are serialized. Compile/validate again after loading.
template<class T> static inline bool pf_write_vector(FILE* f,const std::vector<T>& v) {
    uint64_t count=v.size(); uint32_t size=sizeof(T);
    return fwrite(&size,sizeof(size),1,f)==1 && fwrite(&count,sizeof(count),1,f)==1
        && (v.empty() || fwrite(v.data(),sizeof(T),v.size(),f)==v.size());
}
template<class T> static inline bool pf_read_vector(FILE* f,std::vector<T>& v,size_t& budget) {
    uint32_t size=0; uint64_t count=0;
    if (fread(&size,sizeof(size),1,f)!=1 || size!=sizeof(T)
            || fread(&count,sizeof(count),1,f)!=1 || count>INT_MAX || count>budget/sizeof(T)) return false;
    budget-=(size_t)count*sizeof(T); v.resize((size_t)count);
    return v.empty() || fread(v.data(),sizeof(T),v.size(),f)==v.size();
}
static inline bool pf_model_write(const char* path,const PfModelStorage& h,PfStepOptions options) {
    FILE* f=fopen(path,"wb"); if (!f) return false;
    const char magic[8]={'P','F','N','A','T','0','0','2'};
    uint32_t endian=0x01020304,filter=(h.exclude_parent_child?1u:0u)|(h.explicit_pairs?2u:0u);
    bool ok=fwrite(magic,1,8,f)==8 && fwrite(&endian,4,1,f)==1
        && fwrite(&filter,4,1,f)==1 && fwrite(&options,sizeof(options),1,f)==1
        && pf_write_vector(f,h.links) && pf_write_vector(f,h.dofs)
        && pf_write_vector(f,h.geoms) && pf_write_vector(f,h.excluded_links) && pf_write_vector(f,h.pairs)
        && pf_write_vector(f,h.actuators) && pf_write_vector(f,h.equalities)
        && pf_write_vector(f,h.sites) && pf_write_vector(f,h.convexes)
        && pf_write_vector(f,h.vertices) && pf_write_vector(f,h.faces)
        && pf_write_vector(f,h.heightfields) && pf_write_vector(f,h.heights)
        && pf_write_vector(f,h.initial_qpos);
    return fclose(f)==0 && ok;
}
static inline const char* pf_model_read(const char* path,PfModelStorage& storage,
        PfModel& model,PfStepOptions& options) {
    FILE* f=fopen(path,"rb"); if (!f) return "could not open native model";
    char magic[8]; uint32_t endian=0,filter=0; PfStepOptions o={}; PfModelStorage h;
    size_t budget=256u*1024u*1024u;
    bool ok=fread(magic,1,8,f)==8 && memcmp(magic,"PFNAT002",8)==0
        && fread(&endian,4,1,f)==1 && endian==0x01020304
        && fread(&filter,4,1,f)==1 && filter<=3 && fread(&o,sizeof(o),1,f)==1
        && pf_read_vector(f,h.links,budget) && pf_read_vector(f,h.dofs,budget)
        && pf_read_vector(f,h.geoms,budget) && pf_read_vector(f,h.excluded_links,budget) && pf_read_vector(f,h.pairs,budget)
        && pf_read_vector(f,h.actuators,budget) && pf_read_vector(f,h.equalities,budget)
        && pf_read_vector(f,h.sites,budget) && pf_read_vector(f,h.convexes,budget)
        && pf_read_vector(f,h.vertices,budget) && pf_read_vector(f,h.faces,budget)
        && pf_read_vector(f,h.heightfields,budget) && pf_read_vector(f,h.heights,budget)
        && pf_read_vector(f,h.initial_qpos,budget) && fgetc(f)==EOF;
    if (ferror(f)) ok=false;
    fclose(f);
    if (!ok) return "invalid, truncated, oversized, or incompatible native model";
    if (!pf_vec_valid(o.gravity) || !pf_number(o.dt) || o.dt<=0
            || !pf_nonnegative(o.tolerance) || o.iterations<=0) return "invalid step options";
    h.exclude_parent_child=(filter&1)!=0; h.explicit_pairs=(filter&2)!=0;
    PfModel view; const char* error=pf_compile_model(h,view);
    if (error) return error;
    storage=std::move(h); options=o;
    return pf_compile_model(storage,model);
}
