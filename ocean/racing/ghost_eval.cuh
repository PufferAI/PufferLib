#pragma once
#include <random>
#include <utility>
// Included by pufferl.cu after trainer definitions. One existing Policy per independent world.
static int racing_ghost_main(int argc, char **argv) {
    const char *files[RACING_MAX_GHOSTS];
    char *overrides[argc];
    int count = 0, override_count = 0, headless = 0;
    long frames = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--headless")) headless = 1;
        else if (!strncmp(argv[i], "--frames=", 9)) {
            char *end;
            frames = strtol(argv[i] + 9, &end, 10);
            if (*end || frames <= 0) {
                fprintf(stderr, "--frames must be a positive integer\n");
                return 1;
            }
        } else if (!strncmp(argv[i], "--", 2)) overrides[override_count++] = argv[i];
        else {
            if (count == RACING_MAX_GHOSTS) {
                fprintf(stderr, "At most %d ghosts are supported\n", RACING_MAX_GHOSTS);
                return 1;
            }
            files[count++] = argv[i];
        }
    }
    if (count < 2) {
        fprintf(stderr, "Usage: %s %s CHECKPOINT.bin CHECKPOINT.bin [up to 8 files] "
            "[--policy.num_layers=3] [--headless --frames=300]\n", argv[0], argv[1]);
        return 1;
    }
    // Shuffle cosmetic identities independently of CUDA policy sampling and checkpoint order.
    racing_shuffle_names(count);
    if (headless && !frames) frames = 300;
    Ini ini = {};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, override_count, overrides);
    auto &g = racing_ghosts;
    for (int i = 0; i < count; i++) {
        puf_ini_put(&ini, "base.load_model_path", files[i]);
        char resolved[4096];
        const char *path = puf_checkpoint_path_key(&ini, "load_model_path", resolved, sizeof(resolved));
        if (!path || strlen(path) >= sizeof(g.paths[i])) {
            fprintf(stderr, "Cannot resolve checkpoint: %s\n", files[i]);
            puf_ini_free(&ini);
            return 1;
        }
        snprintf(g.paths[i], sizeof(g.paths[i]), "%s", path);
        FILE *file = fopen(path, "rb");
        if (!file) {
            fprintf(stderr, "Cannot open checkpoint: %s\n", path);
            puf_ini_free(&ini);
            return 1;
        }
        fclose(file);
    }
    g.count = count;
    char number[32];
    snprintf(number, sizeof(number), "%d", count);
    puf_ini_put(&ini, "vec.total_agents", number);
#ifdef RACING_MULTI
    puf_ini_put(&ini, "env.race_cars", number);
#endif
    puf_ini_put(&ini, "vec.num_policies", number);
    snprintf(number, sizeof(number), "%d", (int)puf_ini_get(&ini, "policy", "hidden_size"));
    puf_ini_put(&ini, "vec.hist_policy_hidden_size", number);
    snprintf(number, sizeof(number), "%d", (int)puf_ini_get(&ini, "policy", "num_layers"));
    puf_ini_put(&ini, "vec.hist_policy_num_layers", number);
    puf_ini_put(&ini, "vec.num_buffers", "1");
    puf_ini_put(&ini, "vec.num_threads", "1");
    puf_ini_put(&ini, "selfplay.enabled", "0");
    puf_ini_put(&ini, "base.async", "0");
    puf_ini_put(&ini, "base.cudagraphs", "-1");
    puf_ini_put(&ini, "base.profile", "0");
    puf_ini_put(&ini, "base.reset_every_horizon", "0");
    puf_ini_put(&ini, "train.horizon", "1");
    puf_ini_put(&ini, "train.minibatch_size", "1");
    TrainContext ctx = {.world_size = 1, .artifact_owner = 1};
    PuffeRL *p = create_pufferl(&ini, &ctx);
    for (int i = 0; i < count; i++) {
        FILE *file = fopen(g.paths[i], "rb");
        long expected = numel(p->policies[i].master_weights.shape) * sizeof(float);
        long bytes = -1;
        if (file) {
            if (fseek(file, 0, SEEK_END) == 0) bytes = ftell(file);
            fclose(file);
        }
        if (bytes != expected) {
            fprintf(stderr, "Checkpoint %s has %ld bytes; expected %ld for hidden_size=%d, "
                "num_layers=%d. All ghosts must use the same architecture.\n", g.paths[i], bytes,
                expected, p->hypers.hidden_size, p->hypers.num_layers);
            close_pufferl(p);
            puf_ini_free(&ini);
            return 1;
        }
    }
    for (int i = 0; i < count; i++) {
        pufferl_load_policy(p, i, g.paths[i]);
        printf("Ghost %d: %s | %s\n", i + 1, racing_ghost_names[i], g.paths[i]);
    }
    #ifdef RACING_MULTI
    printf("Shared race with car contacts; every car finishes, retires or times out.\n");
#else
    printf("Independent time trials; sampled policy actions. Names are cosmetic contributor labels.\n");
#endif
    if (!headless) racing_eval_open();
    cudaStream_t stream = p->streams[0];
    puf_bind_stream(stream);
    for (long step = 0; !frames || step < frames; step++) {
        if (!headless && WindowShouldClose()) break;
        pufferl_forward_step(p, 0, 0, stream);
        puf_step(p->vec->envs);
        pf_optix_cuda(cudaGetLastError());
        if (!headless) puf_render(p->vec->envs);
        else pf_optix_cuda(cudaStreamSynchronize(stream));
    }
    pf_optix_cuda(cudaStreamSynchronize(stream));
    close_pufferl(p);
    puf_ini_free(&ini);
    return 0;
}
