// Custom ocean env CUDA. Included by algo.cu.
// Per-env nets live under ocean/<env>/<env>.cu

// Normal(0, std). Used by custom ocean encoders for embeddings.
void puf_normal_init(Prec* dst, float std, ulong seed, cudaStream_t stream) {
    long n = numel(dst->shape);
    assert(n > 0);
    long rand_count = (n % 2 == 0) ? n : n + 1;
    float* buf;
    cudaMalloc(&buf, rand_count * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, buf, rand_count, 0.0f, std);
    curandDestroyGenerator(gen);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst->data, buf, n);
    cudaFree(buf);
}

#ifdef PUFFER_NMMO3
#include "../ocean/nmmo3/nmmo3.cu"
#endif
#ifdef PUFFER_MINIMAL
#include "../ocean/minimal/minimal.cu"
#endif
#ifdef PUFFER_ASTEROIDS
#include "../ocean/asteroids/asteroids.cu"
#endif
#if defined(PUFFER_OSRS_COLOSSEUM) || defined(PUFFER_OSRS_INFERNO) \
    || defined(PUFFER_OSRS_ZULRAH) || defined(PUFFER_OSRS_PVP)
#define PUFFER_OSRS_ENTITY_NET
#endif
#ifdef PUFFER_OSRS_ENTITY_NET
#include "../ocean/osrs/osrs_item_obs_generated.h"
__device__ static const float OSRS_ITEM_OBS_TABLE_DEV
    [OSRS_ITEM_OBS_TABLE_ROWS][OSRS_ITEM_OBS_TABLE_COLS] = {
#include "../ocean/osrs/osrs_item_obs_table.inc"
};
#include "../ocean/osrs/osrs_entity_encoder.cu"
#endif
#ifdef PUFFER_OSRS_COLOSSEUM
#include "../ocean/osrs_colosseum/osrs_colosseum.cu"
#endif
#ifdef PUFFER_OSRS_INFERNO
#include "../ocean/osrs_inferno/osrs_inferno.cu"
#endif
#ifdef PUFFER_NETHACK
#include "../ocean/nethack/nethack.cu"
#include "../ocean/nethack/nethack_policy.cu"
#endif
#ifdef PUFFER_CRAFTAX
#include "../ocean/craftax/craftax.cu"
#endif

// Override encoder vtable when this env has a custom net. No-op otherwise.
static void create_custom_encoder(Encoder* enc) {
#ifdef PUFFER_NETHACK
    create_nethack_encoder(enc);
#elif defined(PUFFER_CRAFTAX)
    create_craftax_encoder(enc);
#elif defined(PUFFER_NMMO3)
#ifdef N3_ATTN
    create_nmmo3_attn_encoder(enc);
#else
    create_nmmo3_conv_encoder(enc);
#endif
#elif defined(PUFFER_MINIMAL)
#ifdef MINIMAL_ATTN
    create_entity_attn_encoder(enc);
#else
    create_minimal_encoder(enc);
#endif
#elif defined(PUFFER_ASTEROIDS)
    create_asteroids_encoder(enc);
#elif defined(PUFFER_OSRS_COLOSSEUM)
    create_osrs_entity_encoder<&OSRS_COLOSSEUM_ENTITY_DESCRIPTOR>(enc);
#elif defined(PUFFER_OSRS_INFERNO)
    create_osrs_entity_encoder<&OSRS_INFERNO_ENTITY_DESCRIPTOR>(enc);
#elif defined(PUFFER_OSRS_ZULRAH) || defined(PUFFER_OSRS_PVP)
    create_osrs_entity_encoder<&OSRS_EQUIPMENT_ENTITY_DESCRIPTOR>(enc);
#else
    (void)enc;
#endif
}

static void create_custom_decoder(Decoder* dec) {
#ifdef PUFFER_NETHACK
    create_nethack_decoder(dec);
#else
    (void)dec;
#endif
}
