#pragma once
#include <optix.h>
#include <cuda_runtime.h>

// Static triangle queries. Direction must be unit length; distances are world units.
struct PfOptixRay { float3 origin; float tmin; float3 direction; float tmax; };
struct PfOptixHit { float distance; float3 normal; int triangle, material; };
struct PfOptixQuery {
    OptixTraversableHandle handle;
    const float3* vertices;
    const unsigned* materials;
    const PfOptixRay* rays;
    PfOptixHit* hits;
    const float3* dynamic_vertices;
    const unsigned* dynamic_materials;
    unsigned dynamic_object_base;
};

#ifndef PF_OPTIX_DEVICE
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Opt-in synchronization localizes asynchronous failures without slowing normal runs.
static bool pf_cuda_debug_enabled() {
    static bool enabled = getenv("PF_CUDA_DEBUG") && atoi(getenv("PF_CUDA_DEBUG")) != 0;
    return enabled;
}
static void pf_cuda_stage(cudaStream_t stream, const char* stage) {
    if (!pf_cuda_debug_enabled()) return;
    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA stage [%s]: %s (%s)\n", stage,
            cudaGetErrorName(error), cudaGetErrorString(error));
        exit(1);
    }
}
static void pf_optix_log(unsigned level, const char* tag, const char* message, void*) {
    fprintf(stderr, "OptiX [%u][%s]: %s\n", level, tag, message);
}
static void pf_optix_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA ray query: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}
static void pf_optix_check(OptixResult result) {
    if (result != OPTIX_SUCCESS) {
        fprintf(stderr, "OptiX ray query: %s\n", optixGetErrorName(result));
        exit(1);
    }
}
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) PfOptixRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
struct PfOptix {
    OptixDeviceContext context;
    OptixModule module;
    OptixProgramGroup groups[3];
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    PfOptixRecord* records;
};
struct PfOptixMesh {
    OptixTraversableHandle handle;
    float3* vertices;
    unsigned* materials;
    void* accel;
    size_t accel_bytes;
    unsigned triangles;
};

// Include the host implementation in one CUDA translation unit (OptiX function table owner).
static void pf_optix_create(PfOptix* rt, const char* ptx, bool allow_instances = false,
    const char* params_name = "pf_optix_query", size_t params_bytes = sizeof(PfOptixQuery),
    unsigned payload_values = 0) {
    pf_optix_cuda(cudaFree(0));
    pf_optix_check(optixInit());
    OptixDeviceContextOptions context_options = {};
    if (pf_cuda_debug_enabled()) {
        context_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        context_options.logCallbackFunction = pf_optix_log;
        context_options.logCallbackLevel = 3;
    }
    pf_optix_check(optixDeviceContextCreate(0, &context_options, &rt->context));
    FILE* file = fopen(ptx, "rb");
    if (!file) {
        fprintf(stderr, "Missing OptiX PTX: %s\n", ptx);
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    assert(size > 0);
    rewind(file);
    char* code = (char*)malloc(size);
    assert(code && fread(code, 1, size, file) == (size_t)size);
    fclose(file);
    OptixModuleCompileOptions module_options = {};
    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.traversableGraphFlags = allow_instances ?
        (OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING |
         OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS) :
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.usesPrimitiveTypeFlags = (unsigned)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.numPayloadValues = payload_values;
    pipeline_options.pipelineLaunchParamsVariableName = params_name;
    pipeline_options.pipelineLaunchParamsSizeInBytes = params_bytes;
    char log[4096];
    size_t log_size = sizeof(log);
    OptixResult result = optixModuleCreate(rt->context, &module_options, &pipeline_options,
        code, size, log, &log_size, &rt->module);
    if (result != OPTIX_SUCCESS) {
        fprintf(stderr, "%s\n", log);
    }
    pf_optix_check(result);
    free(code);
    OptixProgramGroupDesc desc[3] = {};
    desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc[0].raygen.module = rt->module;
    desc[0].raygen.entryFunctionName = "__raygen__pf_rays";
    desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc[1].miss.module = rt->module;
    desc[1].miss.entryFunctionName = "__miss__pf_rays";
    desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc[2].hitgroup.moduleCH = rt->module;
    desc[2].hitgroup.entryFunctionNameCH = "__closesthit__pf_rays";
    OptixProgramGroupOptions group_options = {};
    log_size = sizeof(log);
    pf_optix_check(optixProgramGroupCreate(rt->context, desc, 3, &group_options,
        log, &log_size, rt->groups));
    OptixPipelineLinkOptions link = {};
    link.maxTraceDepth = 1;
    log_size = sizeof(log);
    pf_optix_check(optixPipelineCreate(rt->context, &pipeline_options, &link, rt->groups, 3,
        log, &log_size, &rt->pipeline));
    OptixStackSizes stack = {};
    for (int i = 0; i < 3; i++) {
        pf_optix_check(optixUtilAccumulateStackSizes(rt->groups[i], &stack, rt->pipeline));
    }
    unsigned traversal, state, continuation;
    pf_optix_check(optixUtilComputeStackSizes(&stack, 1, 0, 0,
        &traversal, &state, &continuation));
    pf_optix_check(optixPipelineSetStackSize(rt->pipeline, traversal, state, continuation,
                                            allow_instances ? 2 : 1));
    PfOptixRecord records[3] = {};
    for (int i = 0; i < 3; i++) {
        pf_optix_check(optixSbtRecordPackHeader(rt->groups[i], &records[i]));
    }
    pf_optix_cuda(cudaMalloc(&rt->records, sizeof(records)));
    pf_optix_cuda(cudaMemcpy(rt->records, records, sizeof(records), cudaMemcpyHostToDevice));
    rt->sbt.raygenRecord = (CUdeviceptr)rt->records;
    rt->sbt.missRecordBase = (CUdeviceptr)(rt->records + 1);
    rt->sbt.missRecordStrideInBytes = sizeof(PfOptixRecord);
    rt->sbt.missRecordCount = 1;
    rt->sbt.hitgroupRecordBase = (CUdeviceptr)(rt->records + 2);
    rt->sbt.hitgroupRecordStrideInBytes = sizeof(PfOptixRecord);
    rt->sbt.hitgroupRecordCount = 1;
}

// Startup operation: upload once, build for traversal speed and compact. Mesh stays immutable.
static void pf_optix_build(PfOptix* rt, PfOptixMesh* mesh, const float3* vertices,
    const unsigned* materials, unsigned triangles, cudaStream_t stream) {
    assert(triangles > 0 && triangles <= 0x1fffffff);
    mesh->triangles = triangles;
    size_t vertex_bytes = (size_t)triangles * 3 * sizeof(float3);
    pf_optix_cuda(cudaMalloc(&mesh->vertices, vertex_bytes));
    pf_optix_cuda(cudaMalloc(&mesh->materials, (size_t)triangles * sizeof(unsigned)));
    pf_optix_cuda(cudaMemcpyAsync(mesh->vertices, vertices, vertex_bytes,
        cudaMemcpyHostToDevice, stream));
    pf_optix_cuda(cudaMemcpyAsync(mesh->materials, materials, triangles * sizeof(unsigned),
        cudaMemcpyHostToDevice, stream));
    CUdeviceptr vertex_buffer = (CUdeviceptr)mesh->vertices;
    unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexBuffers = &vertex_buffer;
    input.triangleArray.numVertices = triangles * 3;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = sizeof(float3);
    input.triangleArray.flags = &flags;
    input.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes;
    pf_optix_check(optixAccelComputeMemoryUsage(rt->context, &options, &input, 1, &sizes));
    void *scratch, *uncompacted;
    unsigned long long* compacted_size;
    pf_optix_cuda(cudaMalloc(&scratch, sizes.tempSizeInBytes));
    pf_optix_cuda(cudaMalloc(&uncompacted, sizes.outputSizeInBytes));
    pf_optix_cuda(cudaMalloc(&compacted_size, sizeof(*compacted_size)));
    OptixAccelEmitDesc emit = {};
    emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit.result = (CUdeviceptr)compacted_size;
    pf_optix_check(optixAccelBuild(rt->context, stream, &options, &input, 1,
        (CUdeviceptr)scratch, sizes.tempSizeInBytes, (CUdeviceptr)uncompacted,
        sizes.outputSizeInBytes, &mesh->handle, &emit, 1));
    unsigned long long compact_bytes;
    pf_optix_cuda(cudaMemcpyAsync(&compact_bytes, compacted_size, sizeof(compact_bytes),
        cudaMemcpyDeviceToHost, stream));
    pf_optix_cuda(cudaStreamSynchronize(stream));
    if (compact_bytes < sizes.outputSizeInBytes) {
        pf_optix_cuda(cudaMalloc(&mesh->accel, compact_bytes));
        pf_optix_check(optixAccelCompact(rt->context, stream, mesh->handle,
            (CUdeviceptr)mesh->accel, compact_bytes, &mesh->handle));
        pf_optix_cuda(cudaStreamSynchronize(stream));
        pf_optix_cuda(cudaFree(uncompacted));
        mesh->accel_bytes = compact_bytes;
    } else {
        mesh->accel = uncompacted;
        mesh->accel_bytes = sizes.outputSizeInBytes;
    }
    pf_optix_cuda(cudaFree(scratch));
    pf_optix_cuda(cudaFree(compacted_size));
}

// Caller owns device parameters/rays/hits and their lifetimes. Queue all dependencies on stream.
// Separate query buffers permit independent batches to share the same static mesh/pipeline.
static void pf_optix_launch(PfOptix* rt, const PfOptixQuery* device_query,
    unsigned count, cudaStream_t stream) {
    if (!count) {
        return;
    }
    cudaStreamCaptureStatus capture;
    pf_optix_cuda(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        fprintf(stderr, "OptiX traversal capture is not supported here; disable CUDA graphs.\n");
        exit(1);
    }
    pf_cuda_stage(stream, "before OptiX traversal");
    OptixResult result = optixLaunch(rt->pipeline, stream, (CUdeviceptr)device_query,
        sizeof(PfOptixQuery), &rt->sbt, count, 1, 1);
    if (result != OPTIX_SUCCESS) {
        cudaError_t error = cudaStreamSynchronize(stream);
        fprintf(stderr, "OptiX launch: rays=%u query=%p, CUDA=%s (%s)\n",
            count, (const void*)device_query, cudaGetErrorName(error), cudaGetErrorString(error));
    }
    pf_optix_check(result);
    pf_cuda_stage(stream, "OptiX traversal");
}
static void pf_optix_mesh_destroy(PfOptixMesh* mesh) {
    pf_optix_cuda(cudaFree(mesh->accel));
    pf_optix_cuda(cudaFree(mesh->vertices));
    pf_optix_cuda(cudaFree(mesh->materials));
    *mesh = {};
}
// Caller must finish outstanding launches before destruction.
static void pf_optix_destroy(PfOptix* rt) {
    pf_optix_cuda(cudaFree(rt->records));
    pf_optix_check(optixPipelineDestroy(rt->pipeline));
    for (int i = 0; i < 3; i++) {
        pf_optix_check(optixProgramGroupDestroy(rt->groups[i]));
    }
    pf_optix_check(optixModuleDestroy(rt->module));
    pf_optix_check(optixDeviceContextDestroy(rt->context));
    *rt = {};
}
#endif
