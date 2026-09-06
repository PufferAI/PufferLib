#pragma once

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PUF_DICT_MAX_KEY 128

typedef struct {
    char key[PUF_DICT_MAX_KEY];
    char* str;
    double value;
    double* values;
    int len;
} DictItem;

typedef struct {
    char* name;
    DictItem* items;
    int size;
    int cap;
} Dict;

static inline char* dict_strdup(const char* s) {
    size_t n = strlen(s) + 1;
    char* out = (char*)malloc(n);
    if (!out) {
        perror("malloc");
        exit(1);
    }
    memcpy(out, s, n);
    return out;
}

static inline DictItem* dict_find(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static inline void dict_reserve(Dict* dict, int cap) {
    if (cap <= dict->cap) {
        return;
    }
    dict->items = (DictItem*)realloc(dict->items, cap * sizeof(DictItem));
    if (!dict->items) {
        perror("realloc");
        exit(1);
    }
    memset(dict->items + dict->cap, 0, (cap - dict->cap) * sizeof(DictItem));
    dict->cap = cap;
}

static inline DictItem* dict_insert(Dict* dict, const char* key) {
    if (strlen(key) >= PUF_DICT_MAX_KEY) {
        fprintf(stderr, "dict key too long: %s\n", key);
        exit(1);
    }
    if (dict->size == dict->cap) {
        dict_reserve(dict, dict->cap ? 2 * dict->cap : 8);
    }

    DictItem* item = &dict->items[dict->size++];
    memset(item, 0, sizeof(*item));
    snprintf(item->key, sizeof(item->key), "%s", key);
    return item;
}

static inline DictItem* dict_item(Dict* dict, const char* key) {
    DictItem* item = dict_find(dict, key);
    if (item) {
        return item;
    }
    return dict_insert(dict, key);
}

static inline void dict_item_clear(DictItem* item) {
    free(item->str);
    free(item->values);
    item->str = NULL;
    item->values = NULL;
    item->len = 0;
}

static inline double dict_get(Dict* dict, const char* key) {
    DictItem* item = dict_find(dict, key);
    if (!item) {
        // Keys come from default.ini (+ env overlays); a miss is a bug, not a normal path.
        fprintf(stderr, "missing key [%s] %s\n", dict->name ? dict->name : "?", key);
        exit(1);
    }
    return item->value;
}

static inline DictItem* dict_set(Dict* dict, const char* key, double value) {
    DictItem* item = dict_item(dict, key);
    dict_item_clear(item);
    item->value = value;
    return item;
}

static inline const char* dict_get_str(Dict* dict, const char* key) {
    DictItem* item = dict_find(dict, key);
    if (!item || !item->str) {
        fprintf(stderr, "missing string [%s] %s\n", dict->name ? dict->name : "?", key);
        exit(1);
    }
    return item->str;
}

static inline DictItem* dict_set_str(Dict* dict, const char* key, const char* value) {
    DictItem* item = dict_item(dict, key);
    dict_item_clear(item);
    item->str = dict_strdup(value);
    return item;
}

static inline void dict_copy(Dict* dst, Dict* src) {
    memset(dst, 0, sizeof(*dst));
    if (src->name) {
        dst->name = dict_strdup(src->name);
    }
    if (src->size) {
        dst->cap = src->size;
        dst->items = (DictItem*)calloc((size_t)dst->cap, sizeof(DictItem));
        if (!dst->items) {
            perror("calloc");
            exit(1);
        }
    }
    for (int i = 0; i < src->size; i++) {
        DictItem* s = &src->items[i];
        DictItem* d = dict_insert(dst, s->key);
        d->value = s->value;
        d->len = s->len;
        if (s->str) {
            d->str = dict_strdup(s->str);
        }
        if (s->values) {
            d->values = (double*)calloc((size_t)s->len, sizeof(double));
            if (!d->values) {
                perror("calloc");
                exit(1);
            }
            memcpy(d->values, s->values, (size_t)s->len * sizeof(double));
        }
    }
}

static inline void dict_clear(Dict* dict) {
    for (int i = 0; i < dict->size; i++) {
        dict_item_clear(&dict->items[i]);
    }
    free(dict->name);
    free(dict->items);
    memset(dict, 0, sizeof(*dict));
}

typedef struct {
    Dict* sections;
    int num_sections;
} Ini;

static char* puf_ini_trim(char* s) {
    while (isspace((unsigned char)*s)) {
        s++;
    }

    char* e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) {
        *--e = 0;
    }
    return s;
}

static int puf_ini_streq_ci(const char* a, const char* b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a++) != tolower((unsigned char)*b++)) {
            return 0;
        }
    }
    return *a == *b;
}

static void puf_ini_strip_comment(char* s) {
    char* prev = NULL;
    char quote = 0;
    for (; *s; s++) {
        if ((*s == '\'' || *s == '"') && (!prev || *prev != '\\')) {
            quote = quote == *s ? 0 : quote ? quote : *s;
        } else if ((*s == '#' || *s == ';') && !quote) {
            *s = 0;
            return;
        }
        prev = s;
    }
}

static void puf_ini_strip_quotes(char* s) {
    size_t n = strlen(s);
    if (n < 2) {
        return;
    }
    if ((s[0] == '\'' && s[n - 1] == '\'') ||
            (s[0] == '"' && s[n - 1] == '"')) {
        memmove(s, s + 1, n - 2);
        s[n - 2] = 0;
    }
}

static int puf_ini_read_line(FILE* fp, char** line, int* cap) {
    int n = 0;
    while (1) {
        int c = fgetc(fp);
        if (c == EOF && n == 0) {
            return 0;
        }
        if (n + 1 >= *cap) {
            *cap = *cap ? 2 * *cap : 256;
            *line = (char*)realloc(*line, (size_t)*cap);
            if (!*line) {
                perror("realloc");
                exit(1);
            }
        }
        if (c == EOF || c == '\n') {
            (*line)[n] = 0;
            return 1;
        }
        (*line)[n++] = (char)c;
    }
}

static int puf_ini_parse_val(const char* raw, double* out) {
    if (puf_ini_streq_ci(raw, "true")) {
        *out = 1.0;
        return 1;
    }
    if (puf_ini_streq_ci(raw, "false")) {
        *out = 0.0;
        return 1;
    }

    char buf[256];
    size_t j = 0;
    for (size_t i = 0; raw[i] && j + 1 < sizeof(buf); i++) {
        if (raw[i] != '_' && !isspace((unsigned char)raw[i])) {
            buf[j++] = raw[i];
        }
    }
    buf[j] = 0;

    char* end = NULL;
    double v = strtod(buf, &end);
    if (!buf[0] || !end || *end) {
        return 0;
    }

    *out = v;
    return 1;
}

static int puf_ini_parse_list(const char* raw, double** out, int* len) {
    int cap = 1;
    for (const char* p = raw; *p; p++) {
        if (*p == ',') {
            cap++;
        }
    }

    double* values = (double*)calloc(cap, sizeof(double));
    if (!values) {
        perror("calloc");
        exit(1);
    }

    int n = 0;
    const char* p = raw;
    while (*p) {
        while (isspace((unsigned char)*p)) {
            p++;
        }
        if (!*p) {
            break;
        }
        char* end = NULL;
        values[n++] = strtod(p, &end);
        if (end == p) {
            free(values);
            return 0;
        }
        p = end;
        while (isspace((unsigned char)*p)) {
            p++;
        }
        if (*p == ',') {
            p++;
        } else if (*p) {
            free(values);
            return 0;
        }
    }

    if (n == 0) {
        free(values);
        return 0;
    }
    *out = values;
    *len = n;
    return 1;
}

static Dict* puf_ini_section(Ini* ini, const char* name, int add) {
    for (int i = 0; i < ini->num_sections; i++) {
        if (strcmp(ini->sections[i].name, name) == 0) {
            return &ini->sections[i];
        }
    }

    if (!add) {
        fprintf(stderr, "config error: missing section [%s]\n", name);
        exit(1);
    }

    ini->sections = (Dict*)realloc(ini->sections,
        (ini->num_sections + 1) * sizeof(Dict));
    if (!ini->sections) {
        perror("realloc");
        exit(1);
    }

    Dict* dict = &ini->sections[ini->num_sections++];
    memset(dict, 0, sizeof(*dict));
    dict->name = dict_strdup(name);
    return dict;
}

static void puf_ini_parse_item(DictItem* item, const char* raw) {
    double value = 0;
    double* values = NULL;
    int len = 0;

    if (puf_ini_parse_val(raw, &value)) {
        item->value = value;
    }
    if (strchr(raw, ',') && puf_ini_parse_list(raw, &values, &len)) {
        item->values = values;
        item->len = len;
        item->value = values[0];
    }
}

static DictItem* puf_ini_set(Dict* dict, const char* key, const char* raw) {
    DictItem* item = dict_set_str(dict, key, raw);
    puf_ini_parse_item(item, raw);
    return item;
}

static inline double puf_ini_get(Ini* ini, const char* section, const char* key) {
    return dict_get(puf_ini_section(ini, section, 0), key);
}

static inline const char* puf_ini_get_str(Ini* ini, const char* section,
        const char* key) {
    return dict_get_str(puf_ini_section(ini, section, 0), key);
}

// Reads a key as a list of numbers. A scalar reads as one element and a missing
// key as none, so an optional list needs no placeholder in default.ini.
static inline int puf_ini_get_list(Ini* ini, const char* section,
        const char* key, double* out, int max) {
    DictItem* item = dict_find(puf_ini_section(ini, section, 0), key);
    if (!item) {
        return 0;
    }
    int n = item->len ? item->len : 1;
    if (n > max) {
        fprintf(stderr, "config error: [%s] %s has %d values, max %d\n",
            section, key, n, max);
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        out[i] = item->len ? item->values[i] : item->value;
    }
    return n;
}

static inline void puf_ini_put(Ini* ini, const char* full_key, const char* raw) {
    const char* split = strrchr(full_key, '.');
    if (!split) {
        fprintf(stderr, "expected section.key, got %s\n", full_key);
        exit(1);
    }

    char section[128];
    char key[PUF_DICT_MAX_KEY];
    snprintf(section, sizeof(section), "%.*s", (int)(split - full_key), full_key);
    snprintf(key, sizeof(key), "%s", split + 1);

    Dict* dict = puf_ini_section(ini, section, 0);
    if (!dict_find(dict, key)) {
        fprintf(stderr, "missing key [%s] %s\n", section, key);
        exit(1);
    }
    puf_ini_set(dict, key, raw);
}

static inline void puf_ini_apply_arg(Ini* ini, const char* default_section,
        const char* arg, int idx) {
    if (arg[0] != '-' || arg[1] != '-') {
        fprintf(stderr, "unexpected argument '%s'\n", arg);
        exit(1);
    }
    char tmp[2048];
    if (strlen(arg) >= sizeof(tmp)) {
        fprintf(stderr, "argv:%d: argument too long\n", idx);
        exit(1);
    }
    snprintf(tmp, sizeof(tmp), "%s", arg);

    char* s = tmp;
    while (*s == '-') {
        s++;
    }

    char* eq = strchr(s, '=');
    const char* value = "true";
    if (eq) {
        *eq = 0;
        value = eq + 1;
    }
    for (char* p = s; *p; p++) {
        if (*p == '-') {
            *p = '_';
        }
    }
    if (!*s) {
        fprintf(stderr, "argv:%d: empty key\n", idx);
        exit(1);
    }

    char full_key[4096];
    if (strchr(s, '.')) {
        snprintf(full_key, sizeof(full_key), "%s", s);
    } else {
        snprintf(full_key, sizeof(full_key), "%s.%s", default_section, s);
    }
    puf_ini_put(ini, full_key, value);
}

static void puf_ini_load_file(Ini* ini, const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "could not open %s: %s\n", path, strerror(errno));
        exit(1);
    }

    Dict* section = NULL;
    char* line = NULL;
    int cap = 0;
    for (int n = 1; puf_ini_read_line(fp, &line, &cap); n++) {
        puf_ini_strip_comment(line);
        char* s = puf_ini_trim(line);
        if (!*s) {
            continue;
        }

        size_t len = strlen(s);
        if (s[0] == '[' && len >= 3 && s[len - 1] == ']') {
            s[len - 1] = 0;
            section = puf_ini_section(ini, puf_ini_trim(s + 1), 1);
            continue;
        }
        if (!section) {
            fprintf(stderr, "%s:%d: expected section before key=value\n", path, n);
            exit(1);
        }

        char* eq = strchr(s, '=');
        if (!eq) {
            fprintf(stderr, "%s:%d: expected key=value\n", path, n);
            exit(1);
        }
        *eq = 0;
        char* key = puf_ini_trim(s);
        char* val = puf_ini_trim(eq + 1);
        puf_ini_strip_quotes(val);
        if (!*key) {
            fprintf(stderr, "%s:%d: empty key\n", path, n);
            exit(1);
        }
        puf_ini_set(section, key, val);
    }

    free(line);
    fclose(fp);
}

static inline void puf_ini_load_env(Ini* ini, const char* env_name,
        int argc, char** argv) {
    puf_ini_load_file(ini, "config/default.ini");

    if (strcmp(env_name, "default") != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "config/%s.ini", env_name);
        puf_ini_load_file(ini, path);
    }

    puf_ini_put(ini, "base.env_name", env_name);
#ifdef PLATFORM_WEB
    {
        char web_path[1024];
        snprintf(web_path, sizeof(web_path), "config/%s_web.ini", env_name);
        FILE* web_fp = fopen(web_path, "r");
        if (web_fp) {
            fclose(web_fp);
            puf_ini_load_file(ini, web_path);
        }
    }
#endif
    for (int i = 0; i < argc; i++) {
        puf_ini_apply_arg(ini, "base", argv[i], i);
    }
}

static inline void puf_ini_write(FILE* fp, Ini* ini) {
    for (int s = 0; s < ini->num_sections; s++) {
        Dict* dict = &ini->sections[s];
        fprintf(fp, "\n[%s]\n", dict->name);
        for (int i = 0; i < dict->size; i++) {
            DictItem* item = &dict->items[i];
            fprintf(fp, "%s = ", item->key);
            if (item->len > 0) {
                for (int j = 0; j < item->len; j++) {
                    fprintf(fp, "%s%.17g", j ? "," : "", item->values[j]);
                }
            } else if (item->str) {
                fprintf(fp, "%s", item->str);
            } else {
                fprintf(fp, "%.17g", item->value);
            }
            fputc('\n', fp);
        }
    }
}

static inline void puf_ini_free(Ini* ini) {
    for (int i = 0; i < ini->num_sections; i++) {
        dict_clear(&ini->sections[i]);
    }
    free(ini->sections);
    memset(ini, 0, sizeof(*ini));
}
