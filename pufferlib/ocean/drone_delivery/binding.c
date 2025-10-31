#include "drone_delivery.h"

#define Env DroneDelivery
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->num_agents = unpack(kwargs, "num_agents");

    env->grip_k_max = unpack(kwargs, "grip_k_max");

    env->ablation = unpack(kwargs, "ablation");
    env->anneal_min = unpack(kwargs, "anneal_min");

    env->num_envs = unpack(kwargs, "num_envs");
    env->perfect_anneal = unpack(kwargs, "perfect_anneal");
    env->perfect_deadline = unpack(kwargs, "perfect_deadline");

    env->pos_const = unpack(kwargs, "pos_const");
    env->pos_penalty = unpack(kwargs, "pos_penalty");

    env->reward_grip = unpack(kwargs, "reward_grip");
    env->reward_ho_drop = unpack(kwargs, "reward_ho_drop");
    env->reward_hover = unpack(kwargs, "reward_hover");

    env->reward_max_dist = unpack(kwargs, "reward_max_dist");
    env->reward_min_dist = unpack(kwargs, "reward_min_dist");

    env->vel_penalty_clamp = unpack(kwargs, "vel_penalty_clamp");

    env->w_approach = unpack(kwargs, "w_approach");
    env->w_position = unpack(kwargs, "w_position");
    env->w_stability = unpack(kwargs, "w_stability");
    env->w_velocity = unpack(kwargs, "w_velocity");

    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    //assign_to_dict(dict, "collision_rate", log->collision_rate);
    //assign_to_dict(dict, "oob", log->oob);
    //assign_to_dict(dict, "episode_return", log->episode_return);
    //assign_to_dict(dict, "episode_length", log->episode_length);

    //assign_to_dict(dict, "jitter", log->jitter);
    assign_to_dict(dict, "perfect_grip", log->perfect_grip);
    assign_to_dict(dict, "perfect_deliv", log->perfect_deliv);
    //assign_to_dict(dict, "perfect_now", log->perfect_now);
    //assign_to_dict(dict, "to_pickup", log->to_pickup);
    //assign_to_dict(dict, "ho_pickup", log->ho_pickup);
    //assign_to_dict(dict, "de_pickup", log->de_pickup);
    //assign_to_dict(dict, "to_drop", log->to_drop);
    //assign_to_dict(dict, "ho_drop", log->ho_drop);
    //assign_to_dict(dict, "dist", log->dist);
    //assign_to_dict(dict, "dist100", log->dist100);

    assign_to_dict(dict, "episode_num", log->episode_num);
    //assign_to_dict(dict, "tick", log->tick);
    assign_to_dict(dict, "episode_gain", log->episode_gain);
    assign_to_dict(dict, "anneal", log->anneal);

    //assign_to_dict(dict, "n", log->n);
    return 0;
}
