#include "fc_assets.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int fc_has_prefix(const char* s, const char* prefix) {
    size_t n;
    if (!s || !prefix) return 0;
    n = strlen(prefix);
    return strncmp(s, prefix, n) == 0;
}

static int fc_path_is_absolute(const char* path) {
    return path && path[0] == '/';
}

static int fc_file_exists_path(const char* path) {
    struct stat st;
    return path && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int fc_dir_exists_path(const char* path) {
    struct stat st;
    return path && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int fc_join_path(char* out, size_t cap, const char* a, const char* b) {
    int n;
    if (!out || cap == 0 || !a || !b) return 0;
    if (a[0] == '\0') {
        n = snprintf(out, cap, "%s", b);
    } else if (a[strlen(a) - 1] == '/') {
        n = snprintf(out, cap, "%s%s", a, b);
    } else {
        n = snprintf(out, cap, "%s/%s", a, b);
    }
    return n > 0 && (size_t)n < cap;
}

static void fc_copy_path(char* dst, size_t cap, const char* src) {
    if (!dst || cap == 0) return;
    if (!src) src = "";
    snprintf(dst, cap, "%s", src);
}

static void fc_trim_trailing_slash(char* path) {
    size_t n;
    if (!path) return;
    n = strlen(path);
    while (n > 1 && path[n - 1] == '/') {
        path[n - 1] = '\0';
        n--;
    }
}

static const char* fc_asset_logical_path(const char* path) {
    const char* marker;
    if (!path) return "";
    while (fc_has_prefix(path, "./")) path += 2;
    marker = strstr(path, "/resources/fight_caves/viewer/");
    if (marker) return marker + strlen("/resources/fight_caves/viewer/");
    if (fc_has_prefix(path, "resources/fight_caves/viewer/"))
        return path + strlen("resources/fight_caves/viewer/");
    if (fc_has_prefix(path, "assets/"))
        return path + strlen("assets/");
    return path;
}

static int fc_derive_asset_root(char* out, size_t cap) {
    const char* marker = strstr(__FILE__, "ocean/fight_caves/viewer/src/");
    size_t prefix_len;
    int n;
    if (!marker) return 0;
    prefix_len = (size_t)(marker - __FILE__);
    n = snprintf(out, cap, "%.*sresources/fight_caves/viewer",
                 (int)prefix_len, __FILE__);
    return n > 0 && (size_t)n < cap;
}

static int fc_derive_repo_root(char* out, size_t cap) {
    const char* marker = strstr(__FILE__, "ocean/fight_caves/viewer/src/");
    size_t prefix_len;
    int n;
    if (!marker) return 0;
    prefix_len = (size_t)(marker - __FILE__);
    n = snprintf(out, cap, "%.*s", (int)prefix_len, __FILE__);
    if (!(n > 0 && (size_t)n < cap)) return 0;
    fc_trim_trailing_slash(out);
    return out[0] != '\0';
}

static int fc_derive_repo_root_from_asset_root(char* out, size_t cap,
                                               const char* asset_root) {
    const char* marker;
    size_t prefix_len;
    int n;
    if (!asset_root || !asset_root[0]) return 0;
    marker = strstr(asset_root, "/resources/fight_caves/viewer");
    if (!marker
            && fc_has_prefix(asset_root, "resources/fight_caves/viewer"))
        return snprintf(out, cap, ".") > 0;
    if (!marker) return 0;
    prefix_len = (size_t)(marker - asset_root);
    n = snprintf(out, cap, "%.*s", (int)prefix_len, asset_root);
    if (!(n > 0 && (size_t)n < cap)) return 0;
    fc_trim_trailing_slash(out);
    return out[0] != '\0';
}

static int fc_repo_candidate_valid(const char* root) {
    char path[FC_ASSET_PATH_MAX];
    if (!root || !root[0]) return 0;
    return fc_join_path(path, sizeof(path), root, "ocean/fight_caves")
        && fc_dir_exists_path(path);
}

const char* fc_asset_root(void) {
    static char root[FC_ASSET_PATH_MAX];
    static int initialized;
    const char* env;
    const char* candidates[] = {
        "resources/fight_caves/viewer",
        "../resources/fight_caves/viewer",
        "../../resources/fight_caves/viewer",
        NULL
    };

    env = getenv("FC_ASSET_ROOT");
    if (!env || !env[0]) env = getenv("FC_ASSETS_PATH");
    if (env && env[0]) return env;
    if (initialized) return root;
    initialized = 1;

    if (fc_derive_asset_root(root, sizeof(root)) && fc_dir_exists_path(root))
        return root;
    for (int i = 0; candidates[i]; i++) {
        if (fc_dir_exists_path(candidates[i])) {
            fc_copy_path(root, sizeof(root), candidates[i]);
            return root;
        }
    }
    fc_copy_path(root, sizeof(root), "resources/fight_caves/viewer");
    return root;
}

const char* fc_repo_root(void) {
    static char root[FC_ASSET_PATH_MAX];
    static int initialized;
    const char* env;
    const char* asset_root;
    const char* candidates[] = { ".", "runescape-rl", "..", "../..", NULL };

    env = getenv("FC_REPO_ROOT");
    if (env && env[0]) return env;
    if (initialized) return root;
    initialized = 1;

    if (fc_derive_repo_root(root, sizeof(root)) && fc_repo_candidate_valid(root))
        return root;
    asset_root = fc_asset_root();
    if (fc_derive_repo_root_from_asset_root(root, sizeof(root), asset_root)
            && fc_repo_candidate_valid(root))
        return root;
    for (int i = 0; candidates[i]; i++) {
        if (fc_repo_candidate_valid(candidates[i])) {
            fc_copy_path(root, sizeof(root), candidates[i]);
            return root;
        }
    }
    fc_copy_path(root, sizeof(root), ".");
    return root;
}

int fc_asset_resolve_path(const char* logical_path, char* out, size_t cap) {
    const char* logical = fc_asset_logical_path(logical_path);
    const char* root;
    char joined[FC_ASSET_PATH_MAX];
    if (!out || cap == 0 || !logical_path || !logical[0]) return 0;
    if (fc_path_is_absolute(logical_path) && fc_file_exists_path(logical_path)) {
        fc_copy_path(out, cap, logical_path);
        return 1;
    }
    root = fc_asset_root();
    if (fc_join_path(joined, sizeof(joined), root, logical)
            && fc_file_exists_path(joined)) {
        fc_copy_path(out, cap, joined);
        return 1;
    }
    if (fc_file_exists_path(logical_path)) {
        fc_copy_path(out, cap, logical_path);
        return 1;
    }
    if (fc_join_path(joined, sizeof(joined), root, logical))
        fc_copy_path(out, cap, joined);
    else
        fc_copy_path(out, cap, logical);
    return 0;
}

int fc_repo_resolve_path(const char* logical_path, char* out, size_t cap) {
    char joined[FC_ASSET_PATH_MAX];
    if (!out || cap == 0 || !logical_path || !logical_path[0]) return 0;
    if (fc_path_is_absolute(logical_path) && fc_file_exists_path(logical_path)) {
        fc_copy_path(out, cap, logical_path);
        return 1;
    }
    if (fc_join_path(joined, sizeof(joined), fc_repo_root(), logical_path)
            && fc_file_exists_path(joined)) {
        fc_copy_path(out, cap, joined);
        return 1;
    }
    if (fc_file_exists_path(logical_path)) {
        fc_copy_path(out, cap, logical_path);
        return 1;
    }
    if (fc_join_path(joined, sizeof(joined), fc_repo_root(), logical_path))
        fc_copy_path(out, cap, joined);
    else
        fc_copy_path(out, cap, logical_path);
    return 0;
}

int fc_asset_exists(const char* logical_path) {
    char resolved[FC_ASSET_PATH_MAX];
    return fc_asset_resolve_path(logical_path, resolved, sizeof(resolved));
}

FILE* fc_asset_fopen(const char* logical_path, const char* mode) {
    char resolved[FC_ASSET_PATH_MAX];
    FILE* f;
    if (!fc_asset_resolve_path(logical_path, resolved, sizeof(resolved))) {
        fprintf(stderr, "fc_asset_fopen: missing %s (looked for %s)\n",
                logical_path ? logical_path : "(null)", resolved);
        return NULL;
    }
    f = fopen(resolved, mode);
    if (!f)
        fprintf(stderr, "fc_asset_fopen: cannot open %s: %s\n",
                resolved, strerror(errno));
    return f;
}

int fc_asset_close(FILE* f) {
    return f ? fclose(f) : 0;
}

unsigned char* fc_asset_read_all(const char* logical_path, size_t* out_size) {
    FILE* f = fc_asset_fopen(logical_path, "rb");
    long size;
    unsigned char* data;
    size_t got;

    if (out_size) *out_size = 0;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "%s: seek failed while reading asset\n", logical_path);
        fc_asset_close(f);
        return NULL;
    }
    size = ftell(f);
    if (size <= 0) {
        fprintf(stderr, "%s: empty or unreadable asset\n", logical_path);
        fc_asset_close(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: seek failed while reading asset\n", logical_path);
        fc_asset_close(f);
        return NULL;
    }
    data = malloc((size_t)size);
    if (!data) {
        fprintf(stderr, "%s: out of memory while reading asset\n", logical_path);
        fc_asset_close(f);
        return NULL;
    }
    got = fread(data, 1, (size_t)size, f);
    fc_asset_close(f);
    if (got != (size_t)size) {
        fprintf(stderr, "%s: short read while reading asset\n", logical_path);
        free(data);
        return NULL;
    }
    if (out_size) *out_size = (size_t)size;
    return data;
}
