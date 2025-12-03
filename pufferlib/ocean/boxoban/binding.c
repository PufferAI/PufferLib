#include "boxoban.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>


#define Env Boxoban
#include "../env_binding.h"
uint8_t *MAP_BASE = NULL;
size_t MAP_FILESIZE = 0;
size_t PUZZLE_COUNT = 0;
size_t PUZZLE_SIZE = 400;
//100 bytes agent
//100 bytes walls
//100 bytes boxes
//100 bytes targets

void ensure_map_loaded(void) {
    if (MAP_BASE != NULL)
        return;

    const char *path = "/Users/ha24583/Documents/GitHub/PufferLib/pufferlib/ocean/boxoban/boxoban_maps.bin";
    int fd = open(path, O_RDONLY);
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
    env->size = (int)unpack(kwargs, "size");
    env->max_steps = (int)unpack(kwargs, "max_steps");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
