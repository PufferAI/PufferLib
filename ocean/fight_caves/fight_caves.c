/*
 * fight_caves.c — Standalone Fight Caves manual play, CPU replay and inspection.
 *
 * Compiled with: ./build.sh fight_caves --cpu (add --debug for sanitizers).
 */

#include "fight_caves.h"
#include "puffercpu.c"  /* PufferLib's CPU inference only; not its generic viewer main. */
#include <limits.h>
#include <stdio.h>

/* H is 8-aligned so FP32 and BF16 native tensor layouts have the same padding.
 * Raw files carry no architecture/version metadata: this checks structure only. */
static int fc_policy_weight_count(int hidden, int layers) {
    double count = (double)hidden * (OBS_SIZE + FC_PUFFER_MASK_SIZE + 1)
                 + 3.0 * hidden * hidden * layers;
    if (hidden <= 0 || hidden % 8 || layers <= 0 || count > INT_MAX - 7)
        return 0;
    return (int)count;
}

static Weights* fc_read_checkpoint(const char* path, int count) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "checkpoint rejected: cannot open %s\n", path);
        return NULL;
    }
    long bytes = -1;
    if (fseek(fp, 0, SEEK_END) == 0) bytes = ftell(fp);
    if (bytes < 0 || (size_t)bytes != (size_t)count * sizeof(float) ||
        fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "checkpoint rejected: %s: expected %zu bytes, got %ld\n",
                path, (size_t)count * sizeof(float), bytes);
        fclose(fp);
        return NULL;
    }
    Weights* weights = calloc(1, sizeof(*weights) + ((size_t)count + 7) * sizeof(float));
    if (!weights) {
        fprintf(stderr, "checkpoint rejected: cannot allocate policy weights\n");
        fclose(fp);
        return NULL;
    }
    weights->data = (float*)(weights + 1);
    weights->size = count + 7;
    int valid = fread(weights->data, sizeof(float), count, fp) == (size_t)count;
    valid = valid && fgetc(fp) == EOF && !ferror(fp);
    fclose(fp);
    for (int i = 0; valid && i < count; i++)
        valid = isfinite(weights->data[i]);
    if (!valid) {
        fprintf(stderr, "checkpoint rejected: incomplete or non-finite weights: %s\n", path);
        free(weights);
        return NULL;
    }
    return weights;
}

static int replay(int argc, char** argv, int check_only) {
    if (argc < 3 || argv[2][0] == '-' || strcmp(argv[2], "latest") == 0) {
        fprintf(stderr, "Usage: %s %s CHECKPOINT.bin [--headless] "
                "[--section.key=value ...]\nCPU replay requires an explicit checkpoint path.\n",
                argv[0], check_only ? "check" : "eval");
        return 2;
    }
    int headless = 0, ini_argc = 0;
    char* ini_argv[argc];
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--headless") == 0) headless = 1;
        else ini_argv[ini_argc++] = argv[i];
    }
    Ini ini = {0};
    puf_ini_load_env(&ini, "fight_caves", ini_argc, ini_argv);
    double h = puf_ini_get(&ini, "policy", "hidden_size");
    double l = puf_ini_get(&ini, "policy", "num_layers");
    double limit = puf_ini_get(&ini, "base", "eval_episodes");
    double seed = puf_ini_get(&ini, "base", "seed");
    if (!isfinite(h) || !isfinite(l) || h < 1 || h > INT_MAX || l < 1 ||
        l > INT_MAX || h != floor(h) || l != floor(l) ||
        !isfinite(limit) || limit < (headless ? 1 : 0) || limit > INT_MAX ||
        limit != floor(limit) || !isfinite(seed) || seed < 0 ||
        seed > UINT_MAX || seed != floor(seed)) {
        fprintf(stderr, "invalid CPU replay configuration: positive integer policy dimensions, "
                "nonnegative integer seed and episode limit required "
                "(headless needs at least one episode)\n");
        puf_ini_free(&ini);
        return 2;
    }
    int hidden = (int)h, layers = (int)l;
    int count = fc_policy_weight_count(hidden, layers);
    if (!count) {
        fprintf(stderr, "unsupported CPU policy shape: hidden size must be a multiple of 8 "
                "and the weight count must fit native indexing\n");
        puf_ini_free(&ini);
        return 2;
    }
    Weights* weights = fc_read_checkpoint(argv[2], count);
    if (!weights) {
        puf_ini_free(&ini);
        return 1;
    }
    printf("CPU_META env=fight_caves path=%s file_floats=%d hidden=%d layers=%d\n",
           argv[2], count, hidden, layers);
    if (check_only) {
        free(weights);
        puf_ini_free(&ini);
        return 0;
    }
    int act_sizes[] = ACT_SIZES;
    PufferNet* net = make_puffernet(weights, 1, OBS_SIZE, hidden, layers,
                                  act_sizes, NUM_ATNS);
    Env env = {0};
    float obs[OBS_SIZE] = {0}, actions[NUM_ATNS] = {0}, reward = 0, terminal = 0;
    unsigned char mask[FC_PUFFER_MASK_SIZE] = {0};
    puf_init(&env, puf_ini_section(&ini, "env", 0));
    env.rng = (unsigned)seed;
    srand((unsigned)seed);
    env.agents[0].observations = obs;
    env.agents[0].actions = actions;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    env.agents[0].action_mask = mask;
    puf_reset(&env);
    long steps = 0;
    int episodes = 0;
    /* puf_render owns pacing, pause and single-step; never step its state copy. */
    if (!headless) puf_render(&env);
    while (!limit || episodes < (int)limit) {
        forward_puffernet(net, obs, actions, mask, &terminal);
        puf_step(&env);
        steps++;
        if (terminal > 0.5f) episodes++;
        if (!headless) puf_render(&env);
    }
    float score = env.log.n ? env.log.jad_kill_rate / env.log.n : 0;
    printf("CPU_EVAL env=fight_caves score=%.6f perf=%.6f games=%d steps=%ld params=%d\n",
           score, score, episodes, steps, count);
    puf_close(&env);
    free_puffernet(net);
    free(weights);
    puf_ini_free(&ini);
    return 0;
}

#ifndef FC_CPU_TEST
int main(int argc, char** argv) {
    if (argc > 1 && (strcmp(argv[1], "eval") == 0 || strcmp(argv[1], "check") == 0))
        return replay(argc, argv, strcmp(argv[1], "check") == 0);
    if (argc == 2 && strcmp(argv[1], "--help") == 0)
        printf("CPU tools: %s check CHECKPOINT.bin | "
               "eval CHECKPOINT.bin [--headless] [--section.key=value ...]\n", argv[0]);
    return fc_viewer_main(argc, argv);
}
#endif
