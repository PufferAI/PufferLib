#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <torch/extension.h>

namespace pufferlib {

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

void compute_puff_advantage_mps(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {

    @autoreleasepool {
        TORCH_CHECK(values.device().is_mps(), "All tensors must be on MPS device");
        TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
        TORCH_CHECK(rewards.is_contiguous(), "rewards must be contiguous");
        TORCH_CHECK(dones.is_contiguous(), "dones must be contiguous");
        TORCH_CHECK(importance.is_contiguous(), "importance must be contiguous");
        TORCH_CHECK(advantages.is_contiguous(), "advantages must be contiguous");
        TORCH_CHECK(values.scalar_type() == torch::kFloat32, "All tensors must be float32");

        int num_steps = values.size(0);
        int horizon = values.size(1);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError* error = nil;

        // probably not all too necessary to cache, but does save like 0.1ms per call
        static id<MTLFunction> function = nil;
        static id<MTLComputePipelineState> pipelineState = nil;
        
        if (function == nil) {
            // read the file & compile the shader
            NSString* sourcePath = [[@(__FILE__) stringByDeletingLastPathComponent]
                stringByAppendingPathComponent:@"pufferlib.metal"];
            NSString* source = [NSString stringWithContentsOfFile:sourcePath
                encoding:NSUTF8StringEncoding error:&error];
            TORCH_CHECK(source, "Failed to read Metal source file: ",
                        error ? [[error localizedDescription] UTF8String] : "unknown error");
        
            id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
            TORCH_CHECK(library, "Failed to compile Metal library: ",
                        [[error localizedDescription] UTF8String]);

            function = [library newFunctionWithName:@"puff_advantage_kernel"];
            TORCH_CHECK(function, "Failed to find puff_advantage_kernel function");

            pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
            TORCH_CHECK(pipelineState, "Failed to create compute pipeline: ",
                        [[error localizedDescription] UTF8String]);
        }


        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^{
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(encoder, "Failed to create compute command encoder");

            [encoder setComputePipelineState:pipelineState];
            [encoder setBuffer:getMTLBufferStorage(values)
                        offset:values.storage_offset() * values.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(rewards)
                        offset:rewards.storage_offset() * rewards.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(dones)
                        offset:dones.storage_offset() * dones.element_size() atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(importance)
                        offset:importance.storage_offset() * importance.element_size() atIndex:3];
            [encoder setBuffer:getMTLBufferStorage(advantages)
                        offset:advantages.storage_offset() * advantages.element_size() atIndex:4];

            float gamma_f = gamma, lambda_f = lambda, rho_clip_f = rho_clip, c_clip_f = c_clip;
            int horizon_i = horizon;

            [encoder setBytes:&gamma_f length:sizeof(float) atIndex:5];
            [encoder setBytes:&lambda_f length:sizeof(float) atIndex:6];
            [encoder setBytes:&rho_clip_f length:sizeof(float) atIndex:7];
            [encoder setBytes:&c_clip_f length:sizeof(float) atIndex:8];
            [encoder setBytes:&horizon_i length:sizeof(int) atIndex:9];

            MTLSize gridSize = MTLSizeMake(num_steps, 1, 1);

            NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > num_steps) {
                threadGroupSize = num_steps;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            torch::mps::commit();
        });
    }
}

TORCH_LIBRARY_IMPL(pufferlib, MPS, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_mps);
}

}
