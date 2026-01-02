#include "boxoban.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>


#define Env Boxoban
#include "../env_binding.h"
uint8_t *MAP_BASE = NULL;
char* MAP_PATH = NULL;
size_t MAP_FILESIZE = 0;
size_t PUZZLE_COUNT = 0;
size_t PUZZLE_SIZE = 400;
//100 bytes agent
//100 bytes walls
//100 bytes boxes
//100 bytes targets

static void reset_map_cache(void) {
    if (MAP_BASE != NULL && MAP_BASE != MAP_FAILED && MAP_FILESIZE > 0) {
        munmap(MAP_BASE, MAP_FILESIZE);
    }
    MAP_BASE = NULL;
    MAP_FILESIZE = 0;
    PUZZLE_COUNT = 0;
}

static int update_map_path(PyObject* kwargs) {
    PyObject* map_path_obj = PyDict_GetItemString(kwargs, "map_path");
    if (map_path_obj == NULL || !PyUnicode_Check(map_path_obj)) {
        PyErr_SetString(PyExc_TypeError, "Boxoban requires a string 'map_path' kwarg");
        return -1;
    }

    const char* new_path = PyUnicode_AsUTF8(map_path_obj);
    if (new_path == NULL) {
        return -1;
    }

    if (MAP_PATH != NULL && strcmp(MAP_PATH, new_path) == 0) {
        return 0;
    }

    char* copied = malloc(strlen(new_path) + 1);
    if (copied == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    strcpy(copied, new_path);

    reset_map_cache();
    free(MAP_PATH);
    MAP_PATH = copied;
    return 0;
}

void ensure_map_loaded(void) {
    if (MAP_BASE != NULL)
        return;

    if (MAP_PATH == NULL) {
        fprintf(stderr, "Boxoban map path not set\n");
        abort();
    }

    int fd = open(MAP_PATH, O_RDONLY);
    if (fd <0) {
        perror("open");
        abort();
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        abort();
    }

    MAP_FILESIZE = st.st_size;
    PUZZLE_COUNT = MAP_FILESIZE/PUZZLE_SIZE;

    MAP_BASE = mmap(NULL, MAP_FILESIZE, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (MAP_BASE == MAP_FAILED) {
        perror("mmap");
        abort();
    }

}


static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    if (update_map_path(kwargs) != 0) {
        return -1;
    }
    env->size = (int)unpack(kwargs, "size");
    env->max_steps = (int)unpack(kwargs, "max_steps");
    env->int_r_coeff = (float)unpack(kwargs, "int_r_coeff");
    env->target_loss_pen_coeff = (float)unpack(kwargs, "target_loss_pen_coeff");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "targets_hit", log->n_targets);
    return 0;
}
