#ifndef KINEMATIC_BICYCLE_INTEGRAL_H
#define KINEMATIC_BICYCLE_INTEGRAL_H

#include "../kinematic_bicycle_common.h"

void c_reset(KinematicBicycle* env) {
    kb_reset_state(env, 1);
}

void c_step(KinematicBicycle* env) {
    kb_step(env, 1);
}

void c_render(KinematicBicycle* env) {
    (void)env;
}

void c_close(KinematicBicycle* env) {
    free(env->action_buffer);
    env->action_buffer = NULL;
}

#endif
