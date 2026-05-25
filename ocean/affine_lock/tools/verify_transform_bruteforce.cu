#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "generated/affine_lock_transform_table.h"

namespace {

constexpr int kBits = 16;
constexpr int kNumActions = 8;
constexpr int kPermCount = AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERM_COUNT;
constexpr int kMaskCount = 1 << kBits;
constexpr int kKeyCount = kPermCount * kMaskCount;
constexpr uint16_t kMask = static_cast<uint16_t>(kMaskCount - 1);
constexpr unsigned int kInfDepth = 0xffffffffu;
constexpr int kDefaultMaxDepth = 8;
constexpr int kMaxSupportedDepth = 16;
constexpr int kDefaultThreads = 256;
constexpr uint64_t kDefaultChunkSize = 1ull << 26;

static_assert(AFFINE_LOCK_PRECOMPUTED_TRANSFORM_COUNT == 16384,
    "affine_lock transform table size changed");
static_assert(AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERM_COUNT == 32,
    "affine_lock permutation count changed");

struct Options {
    int max_depth = kDefaultMaxDepth;
    bool cpu = false;
    bool allow_huge = false;
    int threads = kDefaultThreads;
    uint64_t chunk_size = kDefaultChunkSize;
    std::string method = "auto";
    std::string json_path;
};

struct TransformEntry {
    uint32_t key;
    uint8_t depth;
};

uint64_t raw_string_count(int depth) {
    return 1ull << (3 * depth);
}

uint64_t raw_string_sum(int max_depth) {
    uint64_t total = 0;
    for (int depth = 0; depth <= max_depth; depth++) {
        total += raw_string_count(depth);
    }
    return total;
}

uint32_t transform_key(int perm_id, uint16_t xor_mask) {
    return (static_cast<uint32_t>(perm_id) << kBits) |
        static_cast<uint32_t>(xor_mask);
}

uint16_t position_mask(int start, int stride) {
    uint16_t mask = 0;
    for (int bit = start; bit < kBits; bit += stride) {
        mask = static_cast<uint16_t>(mask | (1u << bit));
    }
    return mask;
}

uint16_t range_mask(int start, int end) {
    uint16_t mask = 0;
    for (int bit = start; bit < end; bit++) {
        mask = static_cast<uint16_t>(mask | (1u << bit));
    }
    return mask;
}

void build_action_tables(
        uint8_t action_perm[kNumActions][kBits],
        uint16_t action_xor[kNumActions]) {
    for (int action = 0; action < kNumActions; action++) {
        for (int out_bit = 0; out_bit < kBits; out_bit++) {
            action_perm[action][out_bit] = static_cast<uint8_t>(out_bit);
        }
        action_xor[action] = 0;
    }

    action_xor[0] = kMask;

    for (int out_bit = 0; out_bit < kBits; out_bit++) {
        action_perm[1][out_bit] =
            static_cast<uint8_t>((out_bit + 1) % kBits);
        action_perm[2][out_bit] =
            static_cast<uint8_t>((out_bit - 1 + kBits) % kBits);
        action_perm[3][out_bit] =
            static_cast<uint8_t>(kBits - 1 - out_bit);
    }

    action_xor[4] = position_mask(0, 2);
    action_xor[5] = position_mask(1, 2);
    action_xor[6] = range_mask(0, kBits / 2);
    action_xor[7] = range_mask(kBits / 2, kBits);
}

int find_perm_id(const uint8_t perm[kBits]) {
    for (int perm_id = 0; perm_id < kPermCount; perm_id++) {
        bool matches = true;
        for (int bit = 0; bit < kBits; bit++) {
            if (AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERMS[perm_id][bit] !=
                    perm[bit]) {
                matches = false;
                break;
            }
        }
        if (matches) {
            return perm_id;
        }
    }
    return -1;
}

void build_compose_table(
        const uint8_t action_perm[kNumActions][kBits],
        uint8_t compose_perm[kPermCount][kNumActions]) {
    uint8_t next_perm[kBits];
    for (int perm_id = 0; perm_id < kPermCount; perm_id++) {
        for (int action = 0; action < kNumActions; action++) {
            for (int out_bit = 0; out_bit < kBits; out_bit++) {
                int source_bit = action_perm[action][out_bit];
                next_perm[out_bit] =
                    AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERMS[perm_id][source_bit];
            }

            int next_id = find_perm_id(next_perm);
            if (next_id < 0) {
                throw std::runtime_error("action composition left perm set");
            }
            compose_perm[perm_id][action] = static_cast<uint8_t>(next_id);
        }
    }
}

void build_perm_compose_table(uint8_t compose_perm[kPermCount][kPermCount]) {
    uint8_t next_perm[kBits];
    for (int first_id = 0; first_id < kPermCount; first_id++) {
        for (int second_id = 0; second_id < kPermCount; second_id++) {
            for (int out_bit = 0; out_bit < kBits; out_bit++) {
                int source_bit =
                    AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERMS[second_id][out_bit];
                next_perm[out_bit] =
                    AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERMS[first_id][source_bit];
            }

            int next_id = find_perm_id(next_perm);
            if (next_id < 0) {
                throw std::runtime_error("transform composition left perm set");
            }
            compose_perm[first_id][second_id] =
                static_cast<uint8_t>(next_id);
        }
    }
}

uint16_t permute_mask_by_action(
        uint16_t value,
        int action,
        const uint8_t action_perm[kNumActions][kBits]) {
    uint16_t out = 0;
    for (int out_bit = 0; out_bit < kBits; out_bit++) {
        int in_bit = action_perm[action][out_bit];
        if ((value & (1u << in_bit)) != 0u) {
            out = static_cast<uint16_t>(out | (1u << out_bit));
        }
    }
    return out;
}

uint16_t permute_mask_by_perm(uint16_t value, int perm_id) {
    uint16_t out = 0;
    const uint8_t* perm = AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERMS[perm_id];
    for (int out_bit = 0; out_bit < kBits; out_bit++) {
        int in_bit = perm[out_bit];
        if ((value & (1u << in_bit)) != 0u) {
            out = static_cast<uint16_t>(out | (1u << out_bit));
        }
    }
    return out;
}

std::vector<unsigned int> build_generated_depths() {
    std::vector<unsigned int> generated(kKeyCount, kInfDepth);
    for (int i = 0; i < AFFINE_LOCK_PRECOMPUTED_TRANSFORM_COUNT; i++) {
        const AffineLockPrecomputedTransform& record =
            AFFINE_LOCK_PRECOMPUTED_TRANSFORMS[i];
        if (record.perm_id >= kPermCount) {
            throw std::runtime_error("generated transform has invalid perm_id");
        }
        if (record.action_count != record.distance) {
            throw std::runtime_error(
                "generated transform action_count differs from distance");
        }

        uint32_t key = transform_key(record.perm_id, record.xor_mask);
        if (generated[key] != kInfDepth) {
            throw std::runtime_error("duplicate generated transform key");
        }
        generated[key] = record.distance;
    }
    return generated;
}

void update_discovered(
        std::vector<unsigned int>& discovered,
        int perm_id,
        uint16_t xor_mask,
        unsigned int depth) {
    uint32_t key = transform_key(perm_id, xor_mask);
    if (depth < discovered[key]) {
        discovered[key] = depth;
    }
}

std::vector<unsigned int> brute_force_cpu(
        int max_depth,
        const uint8_t action_perm[kNumActions][kBits],
        const uint16_t action_xor[kNumActions],
        const uint8_t compose_perm[kPermCount][kNumActions]) {
    std::vector<unsigned int> discovered(kKeyCount, kInfDepth);

    for (int depth = 0; depth <= max_depth; depth++) {
        uint64_t count = raw_string_count(depth);
        for (uint64_t index = 0; index < count; index++) {
            uint64_t code = index;
            int perm_id = 0;
            uint16_t xor_mask = 0;

            for (int step = 0; step < depth; step++) {
                int action = static_cast<int>(code & 7ull);
                code >>= 3;
                xor_mask = static_cast<uint16_t>(
                    permute_mask_by_action(xor_mask, action, action_perm) ^
                    action_xor[action]);
                perm_id = compose_perm[perm_id][action];
            }

            update_discovered(discovered, perm_id, xor_mask,
                static_cast<unsigned int>(depth));
        }
    }

    return discovered;
}

std::vector<TransformEntry> collect_entries(
        const std::vector<unsigned int>& depths,
        int max_depth) {
    std::vector<TransformEntry> entries;
    for (uint32_t key = 0; key < depths.size(); key++) {
        unsigned int depth = depths[key];
        if (depth <= static_cast<unsigned int>(max_depth)) {
            entries.push_back(
                {key, static_cast<uint8_t>(depth)});
        }
    }
    return entries;
}

uint32_t compose_transform_keys(
        uint32_t first_key,
        uint32_t second_key,
        const uint8_t compose_perm[kPermCount][kPermCount]) {
    int first_perm = first_key >> kBits;
    int second_perm = second_key >> kBits;
    uint16_t first_mask = static_cast<uint16_t>(first_key & kMask);
    uint16_t second_mask = static_cast<uint16_t>(second_key & kMask);
    int next_perm = compose_perm[first_perm][second_perm];
    uint16_t next_mask = static_cast<uint16_t>(
        permute_mask_by_perm(first_mask, second_perm) ^ second_mask);
    return transform_key(next_perm, next_mask);
}

std::vector<unsigned int> compose_entries_cpu(
        int max_depth,
        const std::vector<TransformEntry>& left_entries,
        const std::vector<TransformEntry>& right_entries,
        const uint8_t compose_perm[kPermCount][kPermCount]) {
    std::vector<unsigned int> discovered(kKeyCount, kInfDepth);

    for (const TransformEntry& left : left_entries) {
        for (const TransformEntry& right : right_entries) {
            unsigned int depth =
                static_cast<unsigned int>(left.depth) + right.depth;
            if (depth > static_cast<unsigned int>(max_depth)) {
                continue;
            }
            uint32_t key = compose_transform_keys(
                left.key, right.key, compose_perm);
            if (depth < discovered[key]) {
                discovered[key] = depth;
            }
        }
    }

    return discovered;
}

__device__ uint16_t device_permute_mask_by_action(
        uint16_t value,
        int action,
        const uint8_t* action_perm) {
    uint16_t out = 0;
    const uint8_t* perm = action_perm + action * kBits;
    for (int out_bit = 0; out_bit < kBits; out_bit++) {
        int in_bit = perm[out_bit];
        if ((value & (1u << in_bit)) != 0u) {
            out = static_cast<uint16_t>(out | (1u << out_bit));
        }
    }
    return out;
}

__device__ uint16_t device_permute_mask_by_perm(
        uint16_t value,
        int perm_id,
        const uint8_t* perms) {
    uint16_t out = 0;
    const uint8_t* perm = perms + perm_id * kBits;
    for (int out_bit = 0; out_bit < kBits; out_bit++) {
        int in_bit = perm[out_bit];
        if ((value & (1u << in_bit)) != 0u) {
            out = static_cast<uint16_t>(out | (1u << out_bit));
        }
    }
    return out;
}

__global__ void brute_force_depth_kernel(
        uint64_t start,
        uint64_t count,
        int depth,
        const uint8_t* action_perm,
        const uint16_t* action_xor,
        const uint8_t* compose_perm,
        unsigned int* discovered) {
    uint64_t offset =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= count) {
        return;
    }

    uint64_t code = start + offset;
    unsigned int perm_id = 0;
    uint16_t xor_mask = 0;

    for (int step = 0; step < depth; step++) {
        int action = static_cast<int>(code & 7ull);
        code >>= 3;
        xor_mask = static_cast<uint16_t>(
            device_permute_mask_by_action(xor_mask, action, action_perm) ^
            action_xor[action]);
        perm_id = compose_perm[perm_id * kNumActions + action];
    }

    uint32_t key = (perm_id << kBits) | static_cast<uint32_t>(xor_mask);
    atomicMin(&discovered[key], static_cast<unsigned int>(depth));
}

__global__ void compose_entries_kernel(
        uint64_t start,
        uint64_t count,
        int max_depth,
        const uint32_t* left_keys,
        const uint8_t* left_depths,
        int left_count,
        const uint32_t* right_keys,
        const uint8_t* right_depths,
        int right_count,
        const uint8_t* perms,
        const uint8_t* compose_perm,
        unsigned int* discovered) {
    uint64_t offset =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= count) {
        return;
    }

    uint64_t pair_index = start + offset;
    int left_index = static_cast<int>(pair_index / right_count);
    int right_index = static_cast<int>(pair_index % right_count);
    if (left_index >= left_count) {
        return;
    }

    unsigned int depth =
        static_cast<unsigned int>(left_depths[left_index]) +
        right_depths[right_index];
    if (depth > static_cast<unsigned int>(max_depth)) {
        return;
    }

    uint32_t left_key = left_keys[left_index];
    uint32_t right_key = right_keys[right_index];
    int left_perm = left_key >> kBits;
    int right_perm = right_key >> kBits;
    uint16_t left_mask = static_cast<uint16_t>(left_key & kMask);
    uint16_t right_mask = static_cast<uint16_t>(right_key & kMask);
    unsigned int next_perm = compose_perm[left_perm * kPermCount + right_perm];
    uint16_t next_mask = static_cast<uint16_t>(
        device_permute_mask_by_perm(left_mask, right_perm, perms) ^ right_mask);
    uint32_t key = (next_perm << kBits) | static_cast<uint32_t>(next_mask);
    atomicMin(&discovered[key], depth);
}

void check_cuda(cudaError_t status, const char* call) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(call) + ": " + cudaGetErrorString(status));
    }
}

std::vector<unsigned int> brute_force_cuda(
        int max_depth,
        const Options& options,
        const uint8_t action_perm[kNumActions][kBits],
        const uint16_t action_xor[kNumActions],
        const uint8_t compose_perm[kPermCount][kNumActions]) {
    int device_count = 0;
    cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count <= 0) {
        throw std::runtime_error(
            std::string("no CUDA device available: ") +
            cudaGetErrorString(device_status));
    }

    uint8_t* d_action_perm = nullptr;
    uint16_t* d_action_xor = nullptr;
    uint8_t* d_compose_perm = nullptr;
    unsigned int* d_discovered = nullptr;

    check_cuda(cudaMalloc(&d_action_perm, kNumActions * kBits),
        "cudaMalloc(action_perm)");
    check_cuda(cudaMalloc(&d_action_xor, kNumActions * sizeof(uint16_t)),
        "cudaMalloc(action_xor)");
    check_cuda(cudaMalloc(&d_compose_perm, kPermCount * kNumActions),
        "cudaMalloc(compose_perm)");
    check_cuda(cudaMalloc(&d_discovered, kKeyCount * sizeof(unsigned int)),
        "cudaMalloc(discovered)");

    try {
        check_cuda(cudaMemcpy(d_action_perm, action_perm, kNumActions * kBits,
            cudaMemcpyHostToDevice), "cudaMemcpy(action_perm)");
        check_cuda(cudaMemcpy(d_action_xor, action_xor,
            kNumActions * sizeof(uint16_t), cudaMemcpyHostToDevice),
            "cudaMemcpy(action_xor)");
        check_cuda(cudaMemcpy(d_compose_perm, compose_perm,
            kPermCount * kNumActions, cudaMemcpyHostToDevice),
            "cudaMemcpy(compose_perm)");
        check_cuda(cudaMemset(d_discovered, 0xff,
            kKeyCount * sizeof(unsigned int)), "cudaMemset(discovered)");

        for (int depth = 0; depth <= max_depth; depth++) {
            uint64_t total = raw_string_count(depth);
            for (uint64_t start = 0; start < total;
                    start += options.chunk_size) {
                uint64_t count = std::min(options.chunk_size, total - start);
                uint64_t blocks64 =
                    (count + static_cast<uint64_t>(options.threads) - 1) /
                    static_cast<uint64_t>(options.threads);
                if (blocks64 > 2147483647ull) {
                    throw std::runtime_error(
                        "chunk size requires too many CUDA blocks");
                }
                unsigned int blocks = static_cast<unsigned int>(blocks64);
                brute_force_depth_kernel<<<blocks, options.threads>>>(
                    start, count, depth, d_action_perm, d_action_xor,
                    d_compose_perm, d_discovered);
                check_cuda(cudaGetLastError(), "brute_force_depth_kernel");
                check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            }
        }

        std::vector<unsigned int> discovered(kKeyCount, kInfDepth);
        check_cuda(cudaMemcpy(discovered.data(), d_discovered,
            kKeyCount * sizeof(unsigned int), cudaMemcpyDeviceToHost),
            "cudaMemcpy(discovered)");

        cudaFree(d_action_perm);
        cudaFree(d_action_xor);
        cudaFree(d_compose_perm);
        cudaFree(d_discovered);
        return discovered;
    } catch (...) {
        cudaFree(d_action_perm);
        cudaFree(d_action_xor);
        cudaFree(d_compose_perm);
        cudaFree(d_discovered);
        throw;
    }
}

void split_entries(
        const std::vector<TransformEntry>& entries,
        std::vector<uint32_t>* keys,
        std::vector<uint8_t>* depths) {
    keys->resize(entries.size());
    depths->resize(entries.size());
    for (size_t i = 0; i < entries.size(); i++) {
        (*keys)[i] = entries[i].key;
        (*depths)[i] = entries[i].depth;
    }
}

std::vector<unsigned int> compose_entries_cuda(
        int max_depth,
        const Options& options,
        const std::vector<TransformEntry>& left_entries,
        const std::vector<TransformEntry>& right_entries,
        const uint8_t compose_perm[kPermCount][kPermCount]) {
    int device_count = 0;
    cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count <= 0) {
        throw std::runtime_error(
            std::string("no CUDA device available: ") +
            cudaGetErrorString(device_status));
    }

    std::vector<uint32_t> left_keys;
    std::vector<uint8_t> left_depths;
    std::vector<uint32_t> right_keys;
    std::vector<uint8_t> right_depths;
    split_entries(left_entries, &left_keys, &left_depths);
    split_entries(right_entries, &right_keys, &right_depths);

    uint32_t* d_left_keys = nullptr;
    uint8_t* d_left_depths = nullptr;
    uint32_t* d_right_keys = nullptr;
    uint8_t* d_right_depths = nullptr;
    uint8_t* d_perms = nullptr;
    uint8_t* d_compose_perm = nullptr;
    unsigned int* d_discovered = nullptr;

    check_cuda(cudaMalloc(&d_left_keys,
        left_keys.size() * sizeof(uint32_t)), "cudaMalloc(left_keys)");
    check_cuda(cudaMalloc(&d_left_depths,
        left_depths.size()), "cudaMalloc(left_depths)");
    check_cuda(cudaMalloc(&d_right_keys,
        right_keys.size() * sizeof(uint32_t)), "cudaMalloc(right_keys)");
    check_cuda(cudaMalloc(&d_right_depths,
        right_depths.size()), "cudaMalloc(right_depths)");
    check_cuda(cudaMalloc(&d_perms, kPermCount * kBits), "cudaMalloc(perms)");
    check_cuda(cudaMalloc(&d_compose_perm, kPermCount * kPermCount),
        "cudaMalloc(compose_perm)");
    check_cuda(cudaMalloc(&d_discovered, kKeyCount * sizeof(unsigned int)),
        "cudaMalloc(discovered)");

    try {
        check_cuda(cudaMemcpy(d_left_keys, left_keys.data(),
            left_keys.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy(left_keys)");
        check_cuda(cudaMemcpy(d_left_depths, left_depths.data(),
            left_depths.size(), cudaMemcpyHostToDevice),
            "cudaMemcpy(left_depths)");
        check_cuda(cudaMemcpy(d_right_keys, right_keys.data(),
            right_keys.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy(right_keys)");
        check_cuda(cudaMemcpy(d_right_depths, right_depths.data(),
            right_depths.size(), cudaMemcpyHostToDevice),
            "cudaMemcpy(right_depths)");
        check_cuda(cudaMemcpy(d_perms,
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERMS, kPermCount * kBits,
            cudaMemcpyHostToDevice), "cudaMemcpy(perms)");
        check_cuda(cudaMemcpy(d_compose_perm, compose_perm,
            kPermCount * kPermCount, cudaMemcpyHostToDevice),
            "cudaMemcpy(compose_perm)");
        check_cuda(cudaMemset(d_discovered, 0xff,
            kKeyCount * sizeof(unsigned int)), "cudaMemset(discovered)");

        uint64_t total =
            static_cast<uint64_t>(left_entries.size()) *
            static_cast<uint64_t>(right_entries.size());
        for (uint64_t start = 0; start < total; start += options.chunk_size) {
            uint64_t count = std::min(options.chunk_size, total - start);
            uint64_t blocks64 =
                (count + static_cast<uint64_t>(options.threads) - 1) /
                static_cast<uint64_t>(options.threads);
            if (blocks64 > 2147483647ull) {
                throw std::runtime_error(
                    "chunk size requires too many CUDA blocks");
            }
            unsigned int blocks = static_cast<unsigned int>(blocks64);
            compose_entries_kernel<<<blocks, options.threads>>>(
                start, count, max_depth,
                d_left_keys, d_left_depths, static_cast<int>(left_entries.size()),
                d_right_keys, d_right_depths,
                static_cast<int>(right_entries.size()),
                d_perms, d_compose_perm, d_discovered);
            check_cuda(cudaGetLastError(), "compose_entries_kernel");
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }

        std::vector<unsigned int> discovered(kKeyCount, kInfDepth);
        check_cuda(cudaMemcpy(discovered.data(), d_discovered,
            kKeyCount * sizeof(unsigned int), cudaMemcpyDeviceToHost),
            "cudaMemcpy(discovered)");

        cudaFree(d_left_keys);
        cudaFree(d_left_depths);
        cudaFree(d_right_keys);
        cudaFree(d_right_depths);
        cudaFree(d_perms);
        cudaFree(d_compose_perm);
        cudaFree(d_discovered);
        return discovered;
    } catch (...) {
        cudaFree(d_left_keys);
        cudaFree(d_left_depths);
        cudaFree(d_right_keys);
        cudaFree(d_right_depths);
        cudaFree(d_perms);
        cudaFree(d_compose_perm);
        cudaFree(d_discovered);
        throw;
    }
}

std::vector<unsigned int> brute_force_mitm(
        int max_depth,
        bool use_cpu,
        const Options& options,
        const uint8_t action_perm[kNumActions][kBits],
        const uint16_t action_xor[kNumActions],
        const uint8_t action_compose_perm[kPermCount][kNumActions],
        const uint8_t transform_compose_perm[kPermCount][kPermCount]) {
    int left_depth = max_depth / 2;
    int right_depth = max_depth - left_depth;
    std::vector<unsigned int> left_discovered = brute_force_cpu(
        left_depth, action_perm, action_xor, action_compose_perm);
    std::vector<unsigned int> right_discovered = left_depth == right_depth ?
        left_discovered :
        brute_force_cpu(right_depth, action_perm, action_xor,
            action_compose_perm);
    std::vector<TransformEntry> left_entries =
        collect_entries(left_discovered, left_depth);
    std::vector<TransformEntry> right_entries =
        collect_entries(right_discovered, right_depth);

    if (use_cpu) {
        return compose_entries_cpu(
            max_depth, left_entries, right_entries, transform_compose_perm);
    }
    return compose_entries_cuda(
        max_depth, options, left_entries, right_entries, transform_compose_perm);
}

void verify_depths(
        const std::vector<unsigned int>& generated,
        const std::vector<unsigned int>& discovered,
        int max_depth) {
    int mismatch_count = 0;
    for (int key = 0; key < kKeyCount; key++) {
        unsigned int generated_depth = generated[key];
        unsigned int discovered_depth = discovered[key];
        bool generated_in_scope = generated_depth <=
            static_cast<unsigned int>(max_depth);
        bool discovered_in_scope = discovered_depth <=
            static_cast<unsigned int>(max_depth);
        if (generated_in_scope && generated_depth != discovered_depth) {
            int perm_id = key >> kBits;
            int xor_mask = key & kMask;
            std::fprintf(stderr,
                "missing or wrong generated transform at perm=%d xor=0x%04x: "
                "generated=%u discovered=%u\n",
                perm_id, xor_mask, generated_depth, discovered_depth);
            mismatch_count++;
        } else if (discovered_in_scope && generated_depth != discovered_depth) {
            int perm_id = key >> kBits;
            int xor_mask = key & kMask;
            std::fprintf(stderr,
                "raw actions found non-table transform at perm=%d xor=0x%04x: "
                "generated=%u discovered=%u\n",
                perm_id, xor_mask, generated_depth, discovered_depth);
            mismatch_count++;
        }

        if (mismatch_count >= 20) {
            throw std::runtime_error("too many transform depth mismatches");
        }
    }

    if (mismatch_count != 0) {
        throw std::runtime_error("transform depth mismatch");
    }
}

std::vector<int> discovered_histogram(
        const std::vector<unsigned int>& discovered,
        int max_depth) {
    std::vector<int> histogram(max_depth + 1, 0);
    for (unsigned int depth : discovered) {
        if (depth <= static_cast<unsigned int>(max_depth)) {
            histogram[depth]++;
        }
    }
    return histogram;
}

std::vector<int> generated_histogram() {
    std::vector<int> histogram(
        AFFINE_LOCK_PRECOMPUTED_TRANSFORM_MAX_DISTANCE + 1, 0);
    for (int depth = 0;
            depth <= AFFINE_LOCK_PRECOMPUTED_TRANSFORM_MAX_DISTANCE;
            depth++) {
        histogram[depth] =
            static_cast<int>(
                AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[depth + 1] -
                AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[depth]);
    }
    return histogram;
}

int count_discovered_transforms(
        const std::vector<unsigned int>& discovered,
        int max_depth) {
    int count = 0;
    for (unsigned int depth : discovered) {
        if (depth <= static_cast<unsigned int>(max_depth)) {
            count++;
        }
    }
    return count;
}

uint64_t half_raw_strings_enumerated(int max_depth, const std::string& method) {
    if (method == "raw") {
        return raw_string_sum(max_depth);
    }
    int left_depth = max_depth / 2;
    int right_depth = max_depth - left_depth;
    uint64_t total = raw_string_sum(left_depth);
    if (right_depth != left_depth) {
        total += raw_string_sum(right_depth);
    }
    return total;
}

uint64_t composed_pair_count(
        const std::vector<unsigned int>& discovered,
        int max_depth,
        const std::string& method) {
    if (method == "raw") {
        (void)discovered;
        (void)max_depth;
        return 0;
    }
    int left_depth = max_depth / 2;
    int right_depth = max_depth - left_depth;
    int left_count = count_discovered_transforms(discovered, left_depth);
    int right_count = count_discovered_transforms(discovered, right_depth);
    return static_cast<uint64_t>(left_count) * static_cast<uint64_t>(right_count);
}

void write_histogram_json(FILE* file, const std::vector<int>& histogram) {
    std::fprintf(file, "{");
    for (size_t i = 0; i < histogram.size(); i++) {
        std::fprintf(file, "%s\"%zu\":%d",
            i == 0 ? "" : ",", i, histogram[i]);
    }
    std::fprintf(file, "}");
}

void write_action_ids_json(
        FILE* file,
        const AffineLockPrecomputedTransform& record) {
    std::fprintf(file, "[");
    for (int i = 0; i < record.action_count; i++) {
        int action = static_cast<int>((record.packed_actions >> (3 * i)) & 7ull);
        std::fprintf(file, "%s%d", i == 0 ? "" : ",", action);
    }
    std::fprintf(file, "]");
}

const char* execution_mode(bool used_cpu) {
    return used_cpu ? "cpu" : "cuda";
}

void write_nullable_string(FILE* file, const char* value) {
    if (value == nullptr) {
        std::fprintf(file, "null");
    } else {
        std::fprintf(file, "\"%s\"", value);
    }
}

void write_audit_json(
        const std::string& path,
        const std::vector<unsigned int>& discovered,
        int max_depth,
        bool used_cpu,
        const std::string& method) {
    if (path.empty()) {
        return;
    }

    FILE* file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        throw std::runtime_error(
            "failed to open json output " + path + ": " + std::strerror(errno));
    }

    std::vector<int> brute_histogram =
        discovered_histogram(discovered, max_depth);
    std::vector<int> table_histogram = generated_histogram();
    int discovered_count = count_discovered_transforms(discovered, max_depth);
    int outside_scope =
        AFFINE_LOCK_PRECOMPUTED_TRANSFORM_COUNT - discovered_count;

    std::fprintf(file, "{\n");
    std::fprintf(file, "  \"schema\": \"affine_lock_transform_audit_v1\",\n");
    std::fprintf(file,
        "  \"verification_scope\": "
        "\"generated transforms with minimum depth <= max_depth\",\n");
    std::fprintf(file, "  \"scope_verified\": true,\n");
    std::fprintf(file, "  \"verified_through_depth\": %d,\n", max_depth);
    std::fprintf(file, "  \"all_generated_transforms_verified\": %s,\n",
        outside_scope == 0 ? "true" : "false");
    std::fprintf(file, "  \"bits\": %d,\n", kBits);
    std::fprintf(file, "  \"num_actions\": %d,\n", kNumActions);
    std::fprintf(file, "  \"perm_count\": %d,\n", kPermCount);
    std::fprintf(file, "  \"transform_count\": %d,\n",
        AFFINE_LOCK_PRECOMPUTED_TRANSFORM_COUNT);
    std::fprintf(file, "  \"table_checksum\": \"0x%016llx\",\n",
        static_cast<unsigned long long>(
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_CHECKSUM));
    std::fprintf(file, "  \"table_max_distance\": %d,\n",
        AFFINE_LOCK_PRECOMPUTED_TRANSFORM_MAX_DISTANCE);
    std::fprintf(file, "  \"max_depth\": %d,\n", max_depth);
    std::fprintf(file, "  \"method\": \"%s\",\n", method.c_str());
    std::fprintf(file, "  \"raw_enumeration_mode\": ");
    write_nullable_string(
        file, method == "raw" ? execution_mode(used_cpu) : nullptr);
    std::fprintf(file, ",\n");
    std::fprintf(file, "  \"half_enumeration_mode\": ");
    write_nullable_string(file, method == "mitm" ? "cpu" : nullptr);
    std::fprintf(file, ",\n");
    std::fprintf(file, "  \"pair_composition_mode\": ");
    write_nullable_string(
        file, method == "mitm" ? execution_mode(used_cpu) : nullptr);
    std::fprintf(file, ",\n");
    std::fprintf(file, "  \"raw_action_strings_covered\": %llu,\n",
        static_cast<unsigned long long>(raw_string_sum(max_depth)));
    std::fprintf(file, "  \"raw_action_strings_enumerated\": %llu,\n",
        static_cast<unsigned long long>(
            half_raw_strings_enumerated(max_depth, method)));
    std::fprintf(file, "  \"transform_pairs_composed\": %llu,\n",
        static_cast<unsigned long long>(
            composed_pair_count(discovered, max_depth, method)));
    std::fprintf(file, "  \"generated_transforms_discovered\": %d,\n",
        discovered_count);
    std::fprintf(file, "  \"generated_transforms_outside_scope\": %d,\n",
        outside_scope);
    std::fprintf(file, "  \"distance_histogram_bruteforce\": ");
    write_histogram_json(file, brute_histogram);
    std::fprintf(file, ",\n  \"distance_histogram_generated\": ");
    write_histogram_json(file, table_histogram);
    std::fprintf(file, ",\n  \"transforms\": [\n");

    for (int i = 0; i < AFFINE_LOCK_PRECOMPUTED_TRANSFORM_COUNT; i++) {
        const AffineLockPrecomputedTransform& record =
            AFFINE_LOCK_PRECOMPUTED_TRANSFORMS[i];
        uint32_t key = transform_key(record.perm_id, record.xor_mask);
        unsigned int brute_depth = discovered[key];

        std::fprintf(file, "    {");
        std::fprintf(file, "\"index\":%d", i);
        std::fprintf(file, ",\"perm_id\":%u", record.perm_id);
        std::fprintf(file, ",\"xor_mask\":\"0x%04x\"", record.xor_mask);
        std::fprintf(file, ",\"generated_min_depth\":%u", record.distance);
        std::fprintf(file, ",\"bruteforce_min_depth\":");
        if (brute_depth <= static_cast<unsigned int>(max_depth)) {
            std::fprintf(file, "%u", brute_depth);
        } else {
            std::fprintf(file, "null");
        }
        std::fprintf(file, ",\"generated_packed_actions\":\"0x%016llx\"",
            static_cast<unsigned long long>(record.packed_actions));
        std::fprintf(file, ",\"generated_action_count\":%u",
            record.action_count);
        std::fprintf(file, ",\"generated_action_ids\":");
        write_action_ids_json(file, record);
        std::fprintf(file, "}%s\n",
            i + 1 == AFFINE_LOCK_PRECOMPUTED_TRANSFORM_COUNT ? "" : ",");
    }

    std::fprintf(file, "  ]\n}\n");
    if (std::fclose(file) != 0) {
        throw std::runtime_error(
            "failed to close json output " + path + ": " + std::strerror(errno));
    }
}

void print_summary(
        const std::vector<unsigned int>& discovered,
        int max_depth,
        bool used_cpu,
        const std::string& method) {
    std::vector<int> histogram = discovered_histogram(discovered, max_depth);

    std::printf("affine_lock transform brute-force verifier\n");
    std::printf("method: %s\n", method.c_str());
    if (method == "raw") {
        std::printf("raw_enumeration_mode: %s\n", execution_mode(used_cpu));
    } else {
        std::printf("half_enumeration_mode: cpu\n");
        std::printf("pair_composition_mode: %s\n", execution_mode(used_cpu));
    }
    std::printf("max_depth: %d\n", max_depth);
    std::printf("raw_action_strings_covered: %llu\n",
        static_cast<unsigned long long>(raw_string_sum(max_depth)));
    std::printf("raw_action_strings_enumerated: %llu\n",
        static_cast<unsigned long long>(
            half_raw_strings_enumerated(max_depth, method)));
    std::printf("transform_pairs_composed: %llu\n",
        static_cast<unsigned long long>(
            composed_pair_count(discovered, max_depth, method)));
    std::printf("depth raw_strings unique_shortest generated_shell\n");
    for (int depth = 0; depth <= max_depth; depth++) {
        uint32_t generated_shell = 0;
        if (depth + 1 < AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSET_COUNT) {
            generated_shell =
                AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[depth + 1] -
                AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[depth];
        }
        std::printf("%5d %11llu %15d %15u\n",
            depth,
            static_cast<unsigned long long>(raw_string_count(depth)),
            histogram[depth],
            generated_shell);
    }
}

void print_usage(const char* program) {
    std::fprintf(stderr,
        "usage: %s [--max-depth N] [--method auto|raw|mitm] [--cpu] "
        "[--allow-huge] [--threads N] [--chunk-size N] "
        "[--write-json PATH]\n",
        program);
}

int parse_positive_int(const char* value, const char* name) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 ||
            parsed > 2147483647L) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

int parse_nonnegative_int(const char* value, const char* name) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 ||
            parsed > 2147483647L) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

uint64_t parse_positive_u64(const char* value, const char* name) {
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<uint64_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--cpu") == 0) {
            options.cpu = true;
        } else if (std::strcmp(argv[i], "--allow-huge") == 0) {
            options.allow_huge = true;
        } else if (std::strcmp(argv[i], "--method") == 0) {
            if (++i >= argc) {
                throw std::runtime_error("--method requires a value");
            }
            options.method = argv[i];
        } else if (std::strcmp(argv[i], "--max-depth") == 0) {
            if (++i >= argc) {
                throw std::runtime_error("--max-depth requires a value");
            }
            options.max_depth = parse_nonnegative_int(argv[i], "--max-depth");
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc) {
                throw std::runtime_error("--threads requires a value");
            }
            options.threads = parse_positive_int(argv[i], "--threads");
        } else if (std::strcmp(argv[i], "--chunk-size") == 0) {
            if (++i >= argc) {
                throw std::runtime_error("--chunk-size requires a value");
            }
            options.chunk_size = parse_positive_u64(argv[i], "--chunk-size");
        } else if (std::strcmp(argv[i], "--write-json") == 0) {
            if (++i >= argc) {
                throw std::runtime_error("--write-json requires a path");
            }
            options.json_path = argv[i];
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
        }
    }

    if (options.max_depth < 0 || options.max_depth > kMaxSupportedDepth) {
        throw std::runtime_error("--max-depth must be in [0, 16]");
    }
    if (options.method != "auto" &&
            options.method != "raw" &&
            options.method != "mitm") {
        throw std::runtime_error("--method must be auto, raw, or mitm");
    }
    if (options.method == "raw" && options.max_depth > 8 &&
            !options.allow_huge) {
        throw std::runtime_error(
            "raw --max-depth above 8 requires --allow-huge");
    }
    if (options.method == "raw" && options.cpu && options.max_depth > 8) {
        throw std::runtime_error("CPU mode is capped at --max-depth 8");
    }
    if (options.threads <= 0 || options.threads > 1024) {
        throw std::runtime_error("--threads must be in [1, 1024]");
    }
    if (options.chunk_size <
            static_cast<uint64_t>(options.threads)) {
        throw std::runtime_error("--chunk-size must be at least --threads");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);

        uint8_t action_perm[kNumActions][kBits];
        uint16_t action_xor[kNumActions];
        uint8_t compose_perm[kPermCount][kNumActions];
        uint8_t transform_compose_perm[kPermCount][kPermCount];
        build_action_tables(action_perm, action_xor);
        build_compose_table(action_perm, compose_perm);
        build_perm_compose_table(transform_compose_perm);

        std::string method = options.method;
        if (method == "auto") {
            method = options.max_depth <= 8 ? "raw" : "mitm";
        }

        std::vector<unsigned int> generated = build_generated_depths();
        std::vector<unsigned int> discovered;
        if (method == "raw") {
            discovered = options.cpu ?
                brute_force_cpu(options.max_depth, action_perm, action_xor,
                    compose_perm) :
                brute_force_cuda(options.max_depth, options, action_perm,
                    action_xor, compose_perm);
        } else {
            discovered = brute_force_mitm(options.max_depth, options.cpu,
                options, action_perm, action_xor, compose_perm,
                transform_compose_perm);
        }

        verify_depths(generated, discovered, options.max_depth);
        write_audit_json(options.json_path, discovered, options.max_depth,
            options.cpu, method);
        print_summary(discovered, options.max_depth, options.cpu, method);
        std::printf("verified generated transform depths through d%d\n",
            options.max_depth);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
