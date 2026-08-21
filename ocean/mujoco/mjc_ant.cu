// GPU build of mjc_ant: one thread per env (see backend.h)
#define PUF_BACKEND PUF_GPU
typedef float obs_t;
#include "mjc_ant.h"
