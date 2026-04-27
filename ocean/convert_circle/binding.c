#include "convert_circle.h"

#define Env ConvertCircle
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->width = unpack(kwargs, "width");
  env->height = unpack(kwargs, "height");
  env->num_agents = unpack(kwargs, "num_agents");
  env->num_factories = unpack(kwargs, "num_factories");
  env->num_resources = unpack(kwargs, "num_resources");
  env->equidistant = unpack(kwargs, "equidistant");
  env->radius = unpack(kwargs, "radius");
  init(env);
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "perf", log->perf);
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
  return 0;
}
