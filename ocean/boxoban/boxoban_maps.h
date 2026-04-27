#ifndef PUFFERLIB_OCEAN_BOXOBAN_MAPS_H
#define PUFFERLIB_OCEAN_BOXOBAN_MAPS_H

#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "generate_maps.h"
#include "parse_maps.h"

/*
Maps are stored in binary files keyed by difficulty.
If the bin does not exist it is created on the fly, then mmapped and shared by envs.
*/

extern uint8_t *MAP_BASE;
extern size_t MAP_FILESIZE;
extern size_t PUZZLE_COUNT;
extern size_t PUZZLE_SIZE;
extern size_t PUZZLE_OBS_BYTES;

int boxoban_prepare_maps_for_difficulty(const char* difficulty, char* out_path, size_t out_cap);
int boxoban_set_map_path(const char *path);
int boxoban_difficulty_id_from_name(const char* difficulty_name);
const char* boxoban_difficulty_name_from_id(int difficulty_id);
void ensure_map_loaded(void);

#ifdef BOXOBAN_MAPS_IMPLEMENTATION

uint8_t *MAP_BASE = NULL;
size_t MAP_FILESIZE = 0;
size_t PUZZLE_COUNT = 0;
size_t PUZZLE_SIZE = BOXOBAN_PUZZLE_BYTES;
size_t PUZZLE_OBS_BYTES = BOXOBAN_PUZZLE_OBS_BYTES;
static char* BOXOBAN_MAP_PATH = NULL;
static const char* BOXOBAN_LEVEL_ROOT = "resources/boxoban/levels";

typedef struct {
    char** items;
    size_t count;
    size_t cap;
} BoxobanPathList;

static int boxoban_cmp_strings(const void* a, const void* b) {
    const char* const* sa = (const char* const*)a;
    const char* const* sb = (const char* const*)b;
    return strcmp(*sa, *sb);
}

static void boxoban_path_list_free(BoxobanPathList* list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->items[i]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->cap = 0;
}

static int boxoban_path_list_append(BoxobanPathList* list, const char* path) {
    if (list->count == list->cap) {
        size_t next_cap = list->cap == 0 ? 64 : list->cap * 2;
        char** next = (char**)realloc(list->items, next_cap * sizeof(char*));
        if (next == NULL) {
            return -1;
        }
        list->items = next;
        list->cap = next_cap;
    }
    char* copied = (char*)malloc(strlen(path) + 1);
    if (copied == NULL) {
        return -1;
    }
    strcpy(copied, path);
    list->items[list->count++] = copied;
    return 0;
}

static int boxoban_has_txt_suffix(const char* name) {
    size_t len = strlen(name);
    return len >= 4 && strcmp(name + len - 4, ".txt") == 0;
}

int boxoban_difficulty_id_from_name(const char* difficulty_name) {
    if (difficulty_name == NULL) {
        return -1;
    }

    if (strcmp(difficulty_name, "basic") == 0) {
        return 0;
    }
    if (strcmp(difficulty_name, "easy") == 0) {
        return 1;
    }
    if (strcmp(difficulty_name, "medium") == 0) {
        return 2;
    }
    if (strcmp(difficulty_name, "hard") == 0) {
        return 3;
    }
    if (strcmp(difficulty_name, "unfiltered") == 0) {
        return 4;
    }

    return -1;
}

const char* boxoban_difficulty_name_from_id(int difficulty_id) {
    switch (difficulty_id) {
    case 0:
        return "basic";
    case 1:
        return "easy";
    case 2:
        return "medium";
    case 3:
        return "hard";
    case 4:
        return "unfiltered";
    default:
        return NULL;
    }
}

static int boxoban_dir_has_txt(const char* dir_path) {
    DIR* dir = opendir(dir_path);
    if (dir == NULL) {
        return 0;
    }
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (boxoban_has_txt_suffix(ent->d_name)) {
            closedir(dir);
            return 1;
        }
    }
    closedir(dir);
    return 0;
}

static int boxoban_collect_sorted_txt_paths_in_dir(const char* dir_path, BoxobanPathList* out_paths) {
    DIR* dir = opendir(dir_path);
    if (dir == NULL) {
        fprintf(stderr, "Missing level directory %s\n", dir_path);
        return -1;
    }

    BoxobanPathList names = {0};
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (!boxoban_has_txt_suffix(ent->d_name)) {
            continue;
        }
        if (boxoban_path_list_append(&names, ent->d_name) != 0) {
            boxoban_path_list_free(&names);
            closedir(dir);
            return -1;
        }
    }
    closedir(dir);

    qsort(names.items, names.count, sizeof(char*), boxoban_cmp_strings);
    for (size_t i = 0; i < names.count; i++) {
        char full_path[1400];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, names.items[i]);
        if (boxoban_path_list_append(out_paths, full_path) != 0) {
            boxoban_path_list_free(&names);
            return -1;
        }
    }
    boxoban_path_list_free(&names);
    return 0;
}

static int boxoban_collect_maps_from_dir(const char* rel_path, BoxobanPathList* out_paths) {
    char level_dir[1400];
    struct stat st;

    snprintf(level_dir, sizeof(level_dir), "%s/%s", BOXOBAN_LEVEL_ROOT, rel_path);
    if (stat(level_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Missing level directory %s\n", level_dir);
        return -1;
    }

    return boxoban_collect_sorted_txt_paths_in_dir(level_dir, out_paths);
}

static int boxoban_collect_maps(const char* difficulty, BoxobanPathList* out_paths) {
    if (strcmp(difficulty, "basic") == 0) {
        return boxoban_collect_maps_from_dir("basic/train", out_paths);
    }
    if (strcmp(difficulty, "easy") == 0) {
        return boxoban_collect_maps_from_dir("easy/train", out_paths);
    }
    if (strcmp(difficulty, "medium") == 0) {
        return boxoban_collect_maps_from_dir("medium/train", out_paths);
    }
    if (strcmp(difficulty, "hard") == 0) {
        return boxoban_collect_maps_from_dir("hard", out_paths);
    }
    if (strcmp(difficulty, "unfiltered") == 0) {
        return boxoban_collect_maps_from_dir("unfiltered/train", out_paths);
    }

    fprintf(stderr, "Invalid difficulty '%s'\n", difficulty);
    return -1;
}

static int boxoban_download_text_maps(const char* difficulty) {
    char zip_url[512];
    snprintf(zip_url, sizeof(zip_url),
        "https://raw.githubusercontent.com/TBBristol/pufferlib_boxoban_levels/main/%s.zip",
        difficulty);
    fprintf(stdout, "[Boxoban] Downloading %s maps from %s\n", difficulty, zip_url);

    char tmp_template[] = "/tmp/boxoban_maps_XXXXXX";
    char* tmp_dir = mkdtemp(tmp_template);
    if (tmp_dir == NULL) {
        return -1;
    }

    char zip_path[1400];
    snprintf(zip_path, sizeof(zip_path), "%s/%s.zip", tmp_dir, difficulty);

    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "curl -L --fail -o '%s' '%s' > /dev/null 2>&1", zip_path, zip_url);
    if (system(cmd) != 0) {
        fprintf(stderr, "Failed to download Boxoban maps with curl\n");
        return -1;
    }

    snprintf(cmd, sizeof(cmd), "unzip -q '%s' -d '%s'", zip_path, tmp_dir);
    if (system(cmd) != 0) {
        fprintf(stderr, "Failed to unzip Boxoban maps archive\n");
        return -1;
    }

    char extracted_root[1400] = {0};
    char find_cmd[4096];
    snprintf(find_cmd, sizeof(find_cmd), "find '%s' -type d -name '%s' | head -n 1", tmp_dir, difficulty);
    FILE* find_pipe = popen(find_cmd, "r");
    if (find_pipe == NULL) {
        return -1;
    }
    if (fgets(extracted_root, sizeof(extracted_root), find_pipe) == NULL) {
        pclose(find_pipe);
        fprintf(stderr, "Downloaded zip missing '%s' directory\n", difficulty);
        return -1;
    }
    pclose(find_pipe);
    extracted_root[strcspn(extracted_root, "\r\n")] = '\0';

    char dest_root[1400];
    snprintf(dest_root, sizeof(dest_root), "%s/%s", BOXOBAN_LEVEL_ROOT, difficulty);
    if (boxoban_mkdir_p(dest_root) != 0) {
        return -1;
    }

    snprintf(cmd, sizeof(cmd), "cp -R '%s/.' '%s/'", extracted_root, dest_root);
    if (system(cmd) != 0) {
        fprintf(stderr, "Failed to copy downloaded maps into %s\n", dest_root);
        return -1;
    }
    return 0;
}

static int boxoban_ensure_text_maps(const char* difficulty) {
    if (strcmp(difficulty, "basic") == 0) {
        char output_dir[1400];
        snprintf(output_dir, sizeof(output_dir), "%s/basic/train", BOXOBAN_LEVEL_ROOT);
        if (boxoban_dir_has_txt(output_dir)) {
            return 0;
        }
        fprintf(stdout, "[Boxoban] Generating basic maps at %s\n", output_dir);
        return boxoban_generate_basic_maps(output_dir, 0);
    }
    if (strcmp(difficulty, "easy") == 0) {
        char output_dir[1400];
        snprintf(output_dir, sizeof(output_dir), "%s/easy/train", BOXOBAN_LEVEL_ROOT);
        if (boxoban_dir_has_txt(output_dir)) {
            return 0;
        }
        fprintf(stdout, "[Boxoban] Generating easy maps at %s\n", output_dir);
        return boxoban_generate_easy_maps(output_dir, 0);
    }
    if (strcmp(difficulty, "medium") == 0) {
        char level_dir[1400];
        snprintf(level_dir, sizeof(level_dir), "%s/medium/train", BOXOBAN_LEVEL_ROOT);
        if (boxoban_dir_has_txt(level_dir)) {
            return 0;
        }
        return boxoban_download_text_maps(difficulty);
    }
    if (strcmp(difficulty, "hard") == 0) {
        char level_dir[1400];
        snprintf(level_dir, sizeof(level_dir), "%s/hard", BOXOBAN_LEVEL_ROOT);
        if (boxoban_dir_has_txt(level_dir)) {
            return 0;
        }
        return boxoban_download_text_maps(difficulty);
    }
    if (strcmp(difficulty, "unfiltered") == 0) {
        char level_dir[1400];
        snprintf(level_dir, sizeof(level_dir), "%s/unfiltered/train", BOXOBAN_LEVEL_ROOT);
        if (boxoban_dir_has_txt(level_dir)) {
            return 0;
        }
        return boxoban_download_text_maps(difficulty);
    }

    return boxoban_download_text_maps(difficulty);
}

static int boxoban_bin_path(const char* difficulty, char* out_path, size_t out_cap) {
    int written = snprintf(out_path, out_cap, "resources/boxoban/boxoban_maps_%s.bin", difficulty);
    if (written <= 0 || (size_t)written >= out_cap) {
        return -1;
    }
    return 0;
}

int boxoban_prepare_maps_for_difficulty(const char* difficulty, char* out_path, size_t out_cap) {
    if (difficulty == NULL || out_path == NULL) {
        return -1;
    }
    if (boxoban_difficulty_id_from_name(difficulty) < 0) {
        return -1;
    }
    if (boxoban_bin_path(difficulty, out_path, out_cap) != 0) {
        return -1;
    }

    if (access(out_path, F_OK) != 0) {
        if (boxoban_ensure_text_maps(difficulty) != 0) {
            return -1;
        }

        BoxobanPathList maps = {0};
        size_t puzzle_count = 0;
        if (boxoban_collect_maps(difficulty, &maps) != 0) {
            boxoban_path_list_free(&maps);
            return -1;
        }

        if (boxoban_write_bin_from_files((const char* const*)maps.items, maps.count, out_path, 0, &puzzle_count) != 0) {
            boxoban_path_list_free(&maps);
            return -1;
        }
        boxoban_path_list_free(&maps);
        fprintf(stdout, "[Boxoban] Generated %zu puzzles for '%s' at %s\n", puzzle_count, difficulty, out_path);
    }

    if (boxoban_set_map_path(out_path) != 0) {
        return -1;
    }
    return 0;
}

static void reset_map_cache(void) {
    if (MAP_BASE != NULL && MAP_BASE != MAP_FAILED && MAP_FILESIZE > 0) {
        munmap(MAP_BASE, MAP_FILESIZE);
    }
    MAP_BASE = NULL;
    MAP_FILESIZE = 0;
    PUZZLE_COUNT = 0;
}

int boxoban_set_map_path(const char *path) {
    if (path == NULL) {
        return -1;
    }
    if (BOXOBAN_MAP_PATH != NULL && strcmp(BOXOBAN_MAP_PATH, path) == 0) {
        return 0;
    }

    char* copied = malloc(strlen(path) + 1);
    if (copied == NULL) {
        return -1;
    }
    strcpy(copied, path);

    reset_map_cache();
    free(BOXOBAN_MAP_PATH);
    BOXOBAN_MAP_PATH = copied;
    return 0;
}

static const char* get_default_map_path(void) {
    const char* env_path = getenv("BOXOBAN_MAP_BIN");
    if (env_path != NULL) {
        return env_path;
    }
    return NULL;
}

void ensure_map_loaded(void) {
    if (MAP_BASE != NULL) {
        return;
    }

    if (BOXOBAN_MAP_PATH == NULL) {
        const char* default_path = get_default_map_path();
        if (default_path != NULL) {
            if (boxoban_set_map_path(default_path) != 0) {
                fprintf(stderr, "Failed to set default Boxoban map path\n");
                abort();
            }
        } else {
            char prepared_path[512];
            if (boxoban_prepare_maps_for_difficulty("basic", prepared_path, sizeof(prepared_path)) != 0) {
                fprintf(stderr, "Failed to prepare default Boxoban maps\n");
                abort();
            }
        }
    }

    int fd = open(BOXOBAN_MAP_PATH, O_RDONLY);
    if (fd < 0) {
        perror("open");
        abort();
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        abort();
    }

    MAP_FILESIZE = st.st_size;
    if (MAP_FILESIZE % PUZZLE_SIZE != 0) {
        fprintf(stderr, "Invalid Boxoban map file size %zu (expected multiple of %zu)\n",
            MAP_FILESIZE, PUZZLE_SIZE);
        abort();
    }
    PUZZLE_COUNT = MAP_FILESIZE / PUZZLE_SIZE;

    MAP_BASE = mmap(NULL, MAP_FILESIZE, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (MAP_BASE == MAP_FAILED) {
        perror("mmap");
        abort();
    }
}

#endif

#endif
