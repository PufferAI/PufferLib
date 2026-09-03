// CPU Craftax encoder: bag-of-embeddings + projection, then MinGRU + decoder.
// Weight order matches CUDA create_craftax_encoder + algo decoder/mingru:
//   embed (154 x 16), proj (hidden x 1635), decoder ((ATN_DIM+1) x hidden),
//   mingru layers (3*hidden x hidden each).
// Include puffercpu.c first (Linear / MinGRU / multidiscrete).

#define CX_NUM_CELLS (OBS_ROWS * OBS_COLS)
#define CX_CHANNELS 8
#define CX_MAP_FLOATS (CX_NUM_CELLS * CX_CHANNELS)
#define CX_NUM_SCALARS INVENTORY_OBS_SIZE
#define CX_EMB_DIM 16
#define CX_VOCAB_TOTAL 154
#define CX_MAP_EMB (CX_NUM_CELLS * CX_EMB_DIM)
#define CX_CONCAT (CX_MAP_EMB + CX_NUM_SCALARS)

static const int CX_OFFSETS[CX_CHANNELS] = {
    0, 64, 72, 74, 90, 106, 122, 138
};

typedef struct CraftaxNet {
    int num_agents;
    int hidden;
    int greedy;
    float* embed_w;
    float* concat;
    Linear* proj;
    Linear* decoder;
    MinGRU* mingru;
    Multidiscrete* multidiscrete;
} CraftaxNet;

static inline int puf_cx_align8(int n) {
    return (n + 7) & ~7;
}

static inline int craftax_weight_count(int hidden, int layers) {
    int n = 0;
    n = puf_cx_align8(n + CX_VOCAB_TOTAL * CX_EMB_DIM);
    n = puf_cx_align8(n + hidden * CX_CONCAT);
    n = puf_cx_align8(n + (ATN_DIM + 1) * hidden);
    for (int l = 0; l < layers; l++) {
        n = puf_cx_align8(n + 3 * hidden * hidden);
    }
    return n;
}

static inline CraftaxNet* init_craftax_net(Weights* weights, int num_agents,
        int hidden, int layers) {
    CraftaxNet* net = (CraftaxNet*)calloc(1, sizeof(CraftaxNet));
    net->num_agents = num_agents;
    net->hidden = hidden;
    net->greedy = 1;
    net->embed_w = get_weights_aligned(weights, CX_VOCAB_TOTAL * CX_EMB_DIM);
    net->concat = (float*)calloc((size_t)num_agents * CX_CONCAT, sizeof(float));
    net->proj = make_linear(weights, num_agents, CX_CONCAT, hidden);
    net->decoder = make_linear(weights, num_agents, hidden, ATN_DIM + 1);
    net->mingru = make_mingru(weights, num_agents, hidden, layers);
    int logit_sizes[1] = {ATN_DIM};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

static inline void craftax_encode(CraftaxNet* net, const float* observations) {
    int B = net->num_agents;
    for (int b = 0; b < B; b++) {
        const float* obs = observations + b * OBS_SIZE;
        float* concat = net->concat + b * CX_CONCAT;
        for (int cell = 0; cell < CX_NUM_CELLS; cell++) {
            const float* ids = obs + cell * CX_CHANNELS;
            for (int d = 0; d < CX_EMB_DIM; d++) {
                float sum = 0.0f;
                for (int ch = 0; ch < CX_CHANNELS; ch++) {
                    int row = CX_OFFSETS[ch] + (int)ids[ch];
                    if (row < 0) {
                        row = CX_OFFSETS[ch];
                    }
                    if (row >= CX_VOCAB_TOTAL) {
                        row = CX_VOCAB_TOTAL - 1;
                    }
                    sum += net->embed_w[row * CX_EMB_DIM + d];
                }
                concat[cell * CX_EMB_DIM + d] = sum;
            }
        }
        memcpy(concat + CX_MAP_EMB, obs + CX_MAP_FLOATS,
            CX_NUM_SCALARS * sizeof(float));
    }
    linear(net->proj, net->concat);
}

static inline void forward_craftax(CraftaxNet* net, const float* observations,
        float* terminals, float* actions, const unsigned char* mask) {
    mingru_zero_term(net->mingru, terminals);
    craftax_encode(net, observations);
    mingru(net->mingru, net->proj->output);
    linear(net->decoder, net->mingru->output);
    multidiscrete(net->multidiscrete, net->decoder->output, actions,
        net->greedy, mask);
}

static inline float craftax_value(CraftaxNet* net, int agent) {
    return net->decoder->output[agent * (ATN_DIM + 1) + ATN_DIM];
}
