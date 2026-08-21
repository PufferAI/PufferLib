// GPU build of mjc_half_cheetah: one thread per env (see backend.h)
#define PUF_BACKEND PUF_GPU
typedef float obs_t;
#include "mjc_half_cheetah.h"
