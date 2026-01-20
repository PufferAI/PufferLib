#include "boss_fight.h"

#define Env BossFight
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  // No special init needed for now
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
  return 0;
}
