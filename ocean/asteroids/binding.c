#include "asteroids.h"

#define Env Asteroids
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->size = unpack(kwargs, "size");
  env->frameskip = unpack(kwargs, "frameskip");
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "perf", log->perf);
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
  return 0;
}
