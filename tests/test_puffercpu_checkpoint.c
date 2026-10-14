// CPU evaluation must recover the tensors saved by either training precision.
// cc -O2 tests/test_puffercpu_checkpoint.c -lm -o /tmp/test_puffercpu_checkpoint
// Repeat with -DPRECISION_FLOAT for --float checkpoints.
#include "../src/puffercpu.c"

// Save identical tensors with the native Allocator's 16-byte alignment.
// Both checkpoint formats contain floats; padding is determined by the
// training parameter type (2 bytes for bf16, 4 for float).
static void write_tensor(FILE* fp, int n, int elem_bytes, int* bytes, int* value) {
    while (*bytes % 16) {
        float padding = 0.0f;
        assert(fwrite(&padding, sizeof(float), 1, fp) == 1);
        *bytes += elem_bytes;
    }
    for (int i = 0; i < n; i++) {
        float w = 0.1f * sinf((float)++(*value));
        assert(fwrite(&w, sizeof(float), 1, fp) == 1);
        *bytes += elem_bytes;
    }
}

static Weights* checkpoint(int input, int hidden, int layers,
        int actions, int continuous, int elem_bytes) {
    char path[] = "/tmp/puffercpu-checkpoint-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE* fp = fdopen(fd, "wb");
    assert(fp);
    int bytes = 0, value = 0;
    write_tensor(fp, input * hidden, elem_bytes, &bytes, &value);
    write_tensor(fp, (actions + 1) * hidden, elem_bytes, &bytes, &value);
    if (continuous) write_tensor(fp, actions, elem_bytes, &bytes, &value);
    for (int i = 0; i < layers; i++) {
        write_tensor(fp, 3 * hidden * hidden, elem_bytes, &bytes, &value);
    }
    assert(fclose(fp) == 0);
    Weights* w = load_weights(path);
    assert(remove(path) == 0);
    return w;
}

static int check(int input, int hidden, int layers, int actions, int continuous) {
    int sizes[8];
    int heads = continuous ? actions : 1;
    for (int i = 0; i < heads; i++) sizes[i] = continuous ? 1 : actions;
#ifdef PRECISION_FLOAT
    int elem_bytes = 4;
#else
    int elem_bytes = 2;
#endif
    Weights* w = checkpoint(input, hidden, layers, actions, continuous, elem_bytes);
    PufferNet* net = make_puffernet(w, 1, input, hidden, layers, sizes, heads);
    float* tensors[3 + layers];
    int counts[3 + layers], n = 0;
    tensors[n] = net->encoder->weights; counts[n++] = input * hidden;
    tensors[n] = net->decoder->weights; counts[n++] = (actions + 1) * hidden;
    if (continuous) {
        tensors[n] = net->log_std; counts[n++] = actions;
    }
    for (int i = 0; i < layers; i++) {
        tensors[n] = net->mingru->proj[i]->weights;
        counts[n++] = 3 * hidden * hidden;
    }
    int value = 0;
    float max_error = 0;
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < counts[t]; i++) {
            float expected = 0.1f * sinf((float)++value);
            float e = fabsf(tensors[t][i] - expected);
            assert(isfinite(e));
            if (e > max_error) max_error = e;
        }
    }
    printf("input=%d hidden=%d layers=%d actions=%d continuous=%d max_error=%.9g\n",
        input, hidden, layers, actions, continuous, max_error);
    free_puffernet(net);
    free(w);
    return max_error != 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    int failures = 0;
    // Drone shape, other continuous heads, and unaligned tensor dimensions.
    failures += check(23, 64, 2, 4, 1);
    failures += check(7, 32, 1, 1, 1);
    failures += check(7, 32, 2, 2, 1);
    failures += check(7, 32, 2, 5, 1);
    failures += check(7, 32, 2, 8, 1);
    failures += check(7, 33, 2, 4, 1);
    failures += check(7, 32, 2, 5, 0);
    failures += check(7, 33, 2, 5, 0);
    return failures != 0;
}
