#include "dogfight.h"

#define Env Dogfight

#include <Python.h>

static PyObject* env_force_state(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* env_set_autopilot(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* vec_set_autopilot(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* vec_set_mode_weights(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* vec_set_curriculum_stage(PyObject* self, PyObject* args);
static PyObject* vec_set_curriculum_target(PyObject* self, PyObject* args);
static PyObject* env_get_autopilot_mode(PyObject* self, PyObject* args);
static PyObject* env_get_state(PyObject* self, PyObject* args);
static PyObject* env_set_obs_highlight(PyObject* self, PyObject* args);
static PyObject* env_get_autoace_state(PyObject* self, PyObject* args);
static PyObject* env_set_camera_follow(PyObject* self, PyObject* args);
static PyObject* vec_get_opponent_observations(PyObject* self, PyObject* args);
static PyObject* vec_set_opponent_actions(PyObject* self, PyObject* args);
static PyObject* vec_enable_opponent_override(PyObject* self, PyObject* args);
static PyObject* vec_set_opponent_buffers(PyObject* self, PyObject* args);
static PyObject* vec_set_eval_spawn_mode(PyObject* self, PyObject* args);

#define MY_METHODS \
    {"env_force_state", (PyCFunction)env_force_state, METH_VARARGS | METH_KEYWORDS, "Force environment state"}, \
    {"env_set_autopilot", (PyCFunction)env_set_autopilot, METH_VARARGS | METH_KEYWORDS, "Set opponent autopilot mode"}, \
    {"vec_set_autopilot", (PyCFunction)vec_set_autopilot, METH_VARARGS | METH_KEYWORDS, "Set autopilot for all envs"}, \
    {"vec_set_mode_weights", (PyCFunction)vec_set_mode_weights, METH_VARARGS | METH_KEYWORDS, "Set mode weights for all envs"}, \
    {"vec_set_curriculum_stage", (PyCFunction)vec_set_curriculum_stage, METH_VARARGS, "Set curriculum stage for all envs"}, \
    {"vec_set_curriculum_target", (PyCFunction)vec_set_curriculum_target, METH_VARARGS, "Set curriculum target (float) for all envs"}, \
    {"env_get_autopilot_mode", (PyCFunction)env_get_autopilot_mode, METH_VARARGS, "Get current autopilot mode"}, \
    {"env_get_state", (PyCFunction)env_get_state, METH_VARARGS, "Get raw player state"}, \
    {"env_set_obs_highlight", (PyCFunction)env_set_obs_highlight, METH_VARARGS, "Set observation indices to highlight with red arrows"}, \
    {"env_get_autoace_state", (PyCFunction)env_get_autoace_state, METH_VARARGS, "Get AutoAce opponent state and tactical info"}, \
    {"env_set_camera_follow", (PyCFunction)env_set_camera_follow, METH_VARARGS, "Set camera to follow player (0) or opponent (1)"}, \
    {"vec_get_opponent_observations", (PyCFunction)vec_get_opponent_observations, METH_VARARGS, "Get observations from opponent perspective for self-play"}, \
    {"vec_set_opponent_actions", (PyCFunction)vec_set_opponent_actions, METH_VARARGS, "Set opponent actions from external policy (self-play)"}, \
    {"vec_enable_opponent_override", (PyCFunction)vec_enable_opponent_override, METH_VARARGS, "Enable/disable opponent action override (0=autopilot, 1=external)"}, \
    {"vec_set_opponent_buffers", (PyCFunction)vec_set_opponent_buffers, METH_VARARGS, "Set opponent observation/reward buffers for dual self-play"}, \
    {"vec_set_eval_spawn_mode", (PyCFunction)vec_set_eval_spawn_mode, METH_VARARGS, "Set eval spawn mode (0=random, 1=opponent_advantage)"}

static float get_float(PyObject *kwargs, const char *key, float default_val) {
    if (!kwargs) return default_val;
    PyObject *val = PyDict_GetItemString(kwargs, key);
    if (!val) return default_val;
    if (PyFloat_Check(val)) return (float)PyFloat_AsDouble(val);
    if (PyLong_Check(val)) return (float)PyLong_AsLong(val);
    return default_val;
}

static int get_int(PyObject *kwargs, const char *key, int default_val) {
    if (!kwargs) return default_val;
    PyObject *val = PyDict_GetItemString(kwargs, key);
    if (!val) return default_val;
    if (PyLong_Check(val)) return (int)PyLong_AsLong(val);
    if (PyFloat_Check(val)) return (int)PyFloat_AsDouble(val);
    return default_val;
}

#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->max_steps = unpack(kwargs, "max_steps");
    int obs_scheme = get_int(kwargs, "obs_scheme", 0);

    RewardConfig rcfg = {
        .aim_scale = get_float(kwargs, "reward_aim_scale", 0.05f),
        .closing_scale = get_float(kwargs, "reward_closing_scale", 0.003f),
        .neg_g = get_float(kwargs, "penalty_neg_g", 0.02f),
        .speed_min = get_float(kwargs, "speed_min", 50.0f),
    };

    int curriculum_enabled = get_int(kwargs, "curriculum_enabled", 0);
    int curriculum_randomize = get_int(kwargs, "curriculum_randomize", 0);
    int eval_spawn_mode = get_int(kwargs, "eval_spawn_mode", 0);

    int env_num = get_int(kwargs, "env_num", 0);

    init(env, obs_scheme, &rcfg, curriculum_enabled, curriculum_randomize, env_num);
    env->eval_spawn_mode = eval_spawn_mode;  // Set after init (overrides default 0)
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "sp_player_kills", log->sp_player_kills);
    assign_to_dict(dict, "sp_opp_kills", log->sp_opp_kills);
    assign_to_dict(dict, "shots_fired", log->shots_fired);
    assign_to_dict(dict, "accuracy", log->accuracy);
    assign_to_dict(dict, "stage", log->stage);

    assign_to_dict(dict, "avg_stage_weight", log->total_stage_weight);  // Raw sum → correct avg
    assign_to_dict(dict, "avg_abs_bias", log->total_abs_bias);          // Raw sum → correct avg
    assign_to_dict(dict, "avg_stage", log->stage_sum);                  // Raw sum → correct avg
    assign_to_dict(dict, "base_stage_kills", log->base_stage_kills);   // Raw sum (not averaged)
    assign_to_dict(dict, "base_stage_eps", log->base_stage_eps);       // Raw sum (not averaged)
    assign_to_dict(dict, "ultimate", log->ultimate);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

static PyObject* env_force_state(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "env_force_state requires 1 positional arg (env handle)");
        return NULL;
    }

    Env* env = unpack_env(args);
    if (!env) return NULL;

    float p_px = get_float(kwargs, "p_px", 0.0f);
    float p_py = get_float(kwargs, "p_py", 0.0f);
    float p_pz = get_float(kwargs, "p_pz", 1000.0f);

    float p_vx = get_float(kwargs, "p_vx", 150.0f);
    float p_vy = get_float(kwargs, "p_vy", 0.0f);
    float p_vz = get_float(kwargs, "p_vz", 0.0f);

    float p_ow = get_float(kwargs, "p_ow", 1.0f);
    float p_ox = get_float(kwargs, "p_ox", 0.0f);
    float p_oy = get_float(kwargs, "p_oy", 0.0f);
    float p_oz = get_float(kwargs, "p_oz", 0.0f);

    float p_throttle = get_float(kwargs, "p_throttle", 1.0f);

    float o_px = get_float(kwargs, "o_px", -9999.0f);
    float o_py = get_float(kwargs, "o_py", -9999.0f);
    float o_pz = get_float(kwargs, "o_pz", -9999.0f);

    float o_vx = get_float(kwargs, "o_vx", -9999.0f);
    float o_vy = get_float(kwargs, "o_vy", -9999.0f);
    float o_vz = get_float(kwargs, "o_vz", -9999.0f);

    float o_ow = get_float(kwargs, "o_ow", -9999.0f);
    float o_ox = get_float(kwargs, "o_ox", -9999.0f);
    float o_oy = get_float(kwargs, "o_oy", -9999.0f);
    float o_oz = get_float(kwargs, "o_oz", -9999.0f);

    int tick = get_int(kwargs, "tick", 0);

    int p_cooldown = get_int(kwargs, "p_cooldown", -1);
    int o_cooldown = get_int(kwargs, "o_cooldown", -1);

    force_state(env,
        p_px, p_py, p_pz,
        p_vx, p_vy, p_vz,
        p_ow, p_ox, p_oy, p_oz,
        p_throttle,
        o_px, o_py, o_pz,
        o_vx, o_vy, o_vz,
        o_ow, o_ox, o_oy, o_oz,
        tick,
        p_cooldown,
        o_cooldown
    );

    Py_RETURN_NONE;
}

static PyObject* env_set_autopilot(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "env_set_autopilot requires 1 positional arg (env handle)");
        return NULL;
    }

    Env* env = unpack_env(args);
    if (!env) return NULL;

    int mode = get_int(kwargs, "mode", AP_STRAIGHT);
    if (mode < 0 || mode >= AP_COUNT) mode = AP_STRAIGHT;
    float throttle = get_float(kwargs, "throttle", AP_DEFAULT_THROTTLE);
    float bank_deg = get_float(kwargs, "bank_deg", AP_DEFAULT_BANK_DEG);
    float climb_rate = get_float(kwargs, "climb_rate", AP_DEFAULT_CLIMB_RATE);

    autopilot_set_mode(&env->opponent_ap, (AutopilotMode)mode, throttle, bank_deg, climb_rate);

    Py_RETURN_NONE;
}

static PyObject* vec_set_autopilot(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_set_autopilot requires 1 positional arg (vec handle)");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    int mode = get_int(kwargs, "mode", AP_STRAIGHT);
    if (mode < 0 || mode >= AP_COUNT) mode = AP_STRAIGHT;
    float throttle = get_float(kwargs, "throttle", AP_DEFAULT_THROTTLE);
    float bank_deg = get_float(kwargs, "bank_deg", AP_DEFAULT_BANK_DEG);
    float climb_rate = get_float(kwargs, "climb_rate", AP_DEFAULT_CLIMB_RATE);

    for (int i = 0; i < vec->num_envs; i++) {
        autopilot_set_mode(&vec->envs[i]->opponent_ap, (AutopilotMode)mode,
                          throttle, bank_deg, climb_rate);
    }

    Py_RETURN_NONE;
}

// Set mode weights for curriculum learning (vectorized)
static PyObject* vec_set_mode_weights(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_set_mode_weights requires 1 positional arg (vec handle)");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    // Get weights for each mode (default 0.2 each for modes 1-5)
    float w_level = get_float(kwargs, "level", 0.2f);
    float w_turn_left = get_float(kwargs, "turn_left", 0.2f);
    float w_turn_right = get_float(kwargs, "turn_right", 0.2f);
    float w_climb = get_float(kwargs, "climb", 0.2f);
    float w_descend = get_float(kwargs, "descend", 0.2f);

    // Set weights for all environments
    for (int i = 0; i < vec->num_envs; i++) {
        AutopilotState* ap = &vec->envs[i]->opponent_ap;
        ap->mode_weights[AP_LEVEL] = w_level;
        ap->mode_weights[AP_TURN_LEFT] = w_turn_left;
        ap->mode_weights[AP_TURN_RIGHT] = w_turn_right;
        ap->mode_weights[AP_CLIMB] = w_climb;
        ap->mode_weights[AP_DESCEND] = w_descend;
    }

    Py_RETURN_NONE;
}

// Set curriculum stage for all environments (global curriculum)
static PyObject* vec_set_curriculum_stage(PyObject* self, PyObject* args) {
    PyObject* vec_arg;
    int stage;

    if (!PyArg_ParseTuple(args, "Oi", &vec_arg, &stage)) {
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(vec_arg);
    if (!vec) {
        PyErr_SetString(PyExc_TypeError, "Invalid vec handle");
        return NULL;
    }

    // Set stage for all environments
    for (int i = 0; i < vec->num_envs; i++) {
        set_curriculum_stage(vec->envs[i], stage);
    }

    Py_RETURN_NONE;
}

// Set curriculum target (float 0.0-15.0) for all environments
static PyObject* vec_set_curriculum_target(PyObject* self, PyObject* args) {
    PyObject* vec_arg;
    float target;

    if (!PyArg_ParseTuple(args, "Of", &vec_arg, &target)) {
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(vec_arg);
    if (!vec) {
        PyErr_SetString(PyExc_TypeError, "Invalid vec handle");
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        set_curriculum_target(vec->envs[i], target);
    }

    Py_RETURN_NONE;
}

static PyObject* env_get_autopilot_mode(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env) return NULL;

    return PyLong_FromLong((long)env->opponent_ap.mode);
}

static PyObject* env_get_state(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env) return NULL;

    Plane* p = &env->player;
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;

    PyDict_SetItemString(dict, "px", PyFloat_FromDouble(p->pos.x));
    PyDict_SetItemString(dict, "py", PyFloat_FromDouble(p->pos.y));
    PyDict_SetItemString(dict, "pz", PyFloat_FromDouble(p->pos.z));

    PyDict_SetItemString(dict, "vx", PyFloat_FromDouble(p->vel.x));
    PyDict_SetItemString(dict, "vy", PyFloat_FromDouble(p->vel.y));
    PyDict_SetItemString(dict, "vz", PyFloat_FromDouble(p->vel.z));

    PyDict_SetItemString(dict, "ow", PyFloat_FromDouble(p->ori.w));
    PyDict_SetItemString(dict, "ox", PyFloat_FromDouble(p->ori.x));
    PyDict_SetItemString(dict, "oy", PyFloat_FromDouble(p->ori.y));
    PyDict_SetItemString(dict, "oz", PyFloat_FromDouble(p->ori.z));

    PyDict_SetItemString(dict, "up_x", PyFloat_FromDouble(up.x));
    PyDict_SetItemString(dict, "up_y", PyFloat_FromDouble(up.y));
    PyDict_SetItemString(dict, "up_z", PyFloat_FromDouble(up.z));

    PyDict_SetItemString(dict, "fwd_x", PyFloat_FromDouble(fwd.x));
    PyDict_SetItemString(dict, "fwd_y", PyFloat_FromDouble(fwd.y));
    PyDict_SetItemString(dict, "fwd_z", PyFloat_FromDouble(fwd.z));

    PyDict_SetItemString(dict, "throttle", PyFloat_FromDouble(p->throttle));

    PyDict_SetItemString(dict, "g_force", PyFloat_FromDouble(p->g_force));

    PyDict_SetItemString(dict, "omega_x", PyFloat_FromDouble(p->omega.x));
    PyDict_SetItemString(dict, "omega_y", PyFloat_FromDouble(p->omega.y));
    PyDict_SetItemString(dict, "omega_z", PyFloat_FromDouble(p->omega.z));

    return dict;
}

// Set which observation indices to highlight with red arrows
// Args: env_handle, list of indices (e.g., [4, 5, 6] for pitch, roll, yaw in scheme 0)
static PyObject* env_set_obs_highlight(PyObject* self, PyObject* args) {
    PyObject* env_arg;
    PyObject* indices_list;

    if (!PyArg_ParseTuple(args, "OO", &env_arg, &indices_list)) {
        return NULL;
    }

    // Get env from handle
    Env* env = (Env*)PyLong_AsVoidPtr(env_arg);
    if (!env) {
        PyErr_SetString(PyExc_TypeError, "Invalid env handle");
        return NULL;
    }

    // Clear existing highlights
    memset(env->obs_highlight, 0, sizeof(env->obs_highlight));

    // Parse list of indices
    if (!PyList_Check(indices_list)) {
        PyErr_SetString(PyExc_TypeError, "Second argument must be a list of indices");
        return NULL;
    }

    Py_ssize_t n = PyList_Size(indices_list);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PyList_GetItem(indices_list, i);
        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Indices must be integers");
            return NULL;
        }
        int idx = (int)PyLong_AsLong(item);
        if (idx >= 0 && idx < 16) {
            env->obs_highlight[idx] = 1;
        }
    }

    Py_RETURN_NONE;
}

// Get AutoAce opponent state and tactical info for behavioral tests
static PyObject* env_get_autoace_state(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env) return NULL;

    Plane* opp = &env->opponent;
    Vec3 opp_fwd = quat_rotate(opp->ori, vec3(1, 0, 0));
    Vec3 opp_up = quat_rotate(opp->ori, vec3(0, 0, 1));

    // Compute bank angle (positive = right wing down)
    // Bank = angle between plane's up and world up, signed by up.y
    // Match sign convention from get_current_bank() in autoace.h:
    // positive when right wing down (up.y < 0)
    float opp_bank = acosf(fminf(fmaxf(opp_up.z, -1.0f), 1.0f));
    if (opp_up.y >= 0) opp_bank = -opp_bank;  // Negative when left wing down

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;

    // Opponent plane state
    PyDict_SetItemString(dict, "opp_px", PyFloat_FromDouble(opp->pos.x));
    PyDict_SetItemString(dict, "opp_py", PyFloat_FromDouble(opp->pos.y));
    PyDict_SetItemString(dict, "opp_pz", PyFloat_FromDouble(opp->pos.z));
    PyDict_SetItemString(dict, "opp_vx", PyFloat_FromDouble(opp->vel.x));
    PyDict_SetItemString(dict, "opp_vy", PyFloat_FromDouble(opp->vel.y));
    PyDict_SetItemString(dict, "opp_vz", PyFloat_FromDouble(opp->vel.z));
    PyDict_SetItemString(dict, "opp_fwd_x", PyFloat_FromDouble(opp_fwd.x));
    PyDict_SetItemString(dict, "opp_fwd_y", PyFloat_FromDouble(opp_fwd.y));
    PyDict_SetItemString(dict, "opp_fwd_z", PyFloat_FromDouble(opp_fwd.z));
    PyDict_SetItemString(dict, "opp_bank", PyFloat_FromDouble(opp_bank));

    // Opponent orientation quaternion
    PyDict_SetItemString(dict, "opp_ow", PyFloat_FromDouble(opp->ori.w));
    PyDict_SetItemString(dict, "opp_ox", PyFloat_FromDouble(opp->ori.x));
    PyDict_SetItemString(dict, "opp_oy", PyFloat_FromDouble(opp->ori.y));
    PyDict_SetItemString(dict, "opp_oz", PyFloat_FromDouble(opp->ori.z));

    // Last AutoAce actions (from most recent step)
    PyDict_SetItemString(dict, "opp_throttle", PyFloat_FromDouble(env->last_opp_actions[0]));
    PyDict_SetItemString(dict, "opp_elevator", PyFloat_FromDouble(env->last_opp_actions[1]));
    PyDict_SetItemString(dict, "opp_aileron", PyFloat_FromDouble(env->last_opp_actions[2]));
    PyDict_SetItemString(dict, "opp_rudder", PyFloat_FromDouble(env->last_opp_actions[3]));
    PyDict_SetItemString(dict, "opp_trigger", PyFloat_FromDouble(env->last_opp_actions[4]));

    // Tactical state (from AutoAce)
    TacticalState* ts = &env->opponent_ace.tactical;
    PyDict_SetItemString(dict, "engagement", PyLong_FromLong(env->opponent_ace.engagement));
    PyDict_SetItemString(dict, "mode", PyLong_FromLong(env->opponent_ap.mode));
    PyDict_SetItemString(dict, "aspect_angle", PyFloat_FromDouble(ts->aspect_angle));
    PyDict_SetItemString(dict, "antenna_train", PyFloat_FromDouble(ts->antenna_train));
    PyDict_SetItemString(dict, "range", PyFloat_FromDouble(ts->range));
    PyDict_SetItemString(dict, "closure_rate", PyFloat_FromDouble(ts->closure_rate));
    PyDict_SetItemString(dict, "in_gun_envelope", PyBool_FromLong(ts->in_gun_envelope));

    return dict;
}

// Set camera to follow player (0) or opponent (1)
static PyObject* env_set_camera_follow(PyObject* self, PyObject* args) {
    PyObject* env_arg;
    int follow_opponent;

    if (!PyArg_ParseTuple(args, "Oi", &env_arg, &follow_opponent)) {
        return NULL;
    }

    Env* env = (Env*)PyLong_AsVoidPtr(env_arg);
    if (!env) {
        PyErr_SetString(PyExc_TypeError, "Invalid env handle");
        return NULL;
    }

    env->camera_follow_opponent = follow_opponent;
    Py_RETURN_NONE;
}

// Get opponent observations for all environments (for self-play)
// Returns: numpy array of shape (num_envs, obs_size) with opponent's view of the world
// Currently only supports scheme 0 (OBS_MOMENTUM) - returns 16 obs per env
static PyObject* vec_get_opponent_observations(PyObject* self, PyObject* args) {
    PyObject* vec_arg;

    if (!PyArg_ParseTuple(args, "O", &vec_arg)) {
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(vec_arg);
    if (!vec) {
        PyErr_SetString(PyExc_TypeError, "Invalid vec handle");
        return NULL;
    }

    // Get obs_size from first environment (all envs have same scheme)
    int obs_size = vec->envs[0]->obs_size;

    // Create numpy array of shape (num_envs, obs_size)
    npy_intp dims[2] = {vec->num_envs, obs_size};
    PyObject* arr = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (!arr) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate opponent observations array");
        return NULL;
    }

    // Compute opponent observations for each environment
    float* data = (float*)PyArray_DATA((PyArrayObject*)arr);
    for (int i = 0; i < vec->num_envs; i++) {
        compute_opponent_observations(vec->envs[i], data + i * obs_size);
    }

    return arr;
}

// Set opponent actions for all environments (for self-play)
// Args: vec_handle, actions_array (numpy float32 shape [num_envs, 5])
// Sets opponent_actions_override for each env (used when use_opponent_override=1)
static PyObject* vec_set_opponent_actions(PyObject* self, PyObject* args) {
    PyObject* vec_arg;
    PyObject* actions_arr;

    if (!PyArg_ParseTuple(args, "OO", &vec_arg, &actions_arr)) {
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(vec_arg);
    if (!vec) {
        PyErr_SetString(PyExc_TypeError, "Invalid vec handle");
        return NULL;
    }

    // Verify array shape and type
    if (!PyArray_Check(actions_arr)) {
        PyErr_SetString(PyExc_TypeError, "actions must be a numpy array");
        return NULL;
    }

    PyArrayObject* arr = (PyArrayObject*)actions_arr;
    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "actions must be 2D array (num_envs, 5)");
        return NULL;
    }

    npy_intp* dims = PyArray_DIMS(arr);
    if (dims[0] != vec->num_envs || dims[1] != 5) {
        PyErr_Format(PyExc_ValueError,
                     "actions shape must be (%d, 5), got (%ld, %ld)",
                     vec->num_envs, (long)dims[0], (long)dims[1]);
        return NULL;
    }

    if (PyArray_TYPE(arr) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "actions must be float32 dtype");
        return NULL;
    }

    // Copy actions to each environment's override buffer
    float* data = (float*)PyArray_DATA(arr);
    for (int i = 0; i < vec->num_envs; i++) {
        for (int j = 0; j < 5; j++) {
            vec->envs[i]->opponent_actions_override[j] = data[i * 5 + j];
        }
    }

    Py_RETURN_NONE;
}

// Enable or disable opponent action override for all environments
// Args: vec_handle, enable (0=use autopilot, 1=use external actions)
static PyObject* vec_enable_opponent_override(PyObject* self, PyObject* args) {
    PyObject* vec_arg;
    int enable;

    if (!PyArg_ParseTuple(args, "Oi", &vec_arg, &enable)) {
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(vec_arg);
    if (!vec) {
        PyErr_SetString(PyExc_TypeError, "Invalid vec handle");
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        vec->envs[i]->use_opponent_override = enable ? 1 : 0;
    }

    Py_RETURN_NONE;
}

// Set opponent observation/reward buffers for all environments (for dual self-play)
// Args: vec_handle, opponent_obs_array (numpy float32), opponent_rewards_array (numpy float32)
// These buffers will be written during c_step() enabling Multiprocessing backend
static PyObject* vec_set_opponent_buffers(PyObject* self, PyObject* args) {
    PyObject* vec_arg;
    PyObject* opp_obs_arr;
    PyObject* opp_rew_arr;

    if (!PyArg_ParseTuple(args, "OOO", &vec_arg, &opp_obs_arr, &opp_rew_arr)) {
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(vec_arg);
    if (!vec) {
        PyErr_SetString(PyExc_TypeError, "Invalid vec handle");
        return NULL;
    }

    // Get obs_size from first environment
    int obs_size = vec->envs[0]->obs_size;

    // Set opponent observations buffer
    float* opp_obs_data = NULL;
    if (opp_obs_arr != Py_None) {
        if (!PyArray_Check(opp_obs_arr)) {
            PyErr_SetString(PyExc_TypeError, "opponent_obs must be a numpy array or None");
            return NULL;
        }
        PyArrayObject* arr = (PyArrayObject*)opp_obs_arr;
        if (PyArray_TYPE(arr) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_TypeError, "opponent_obs must be float32 dtype");
            return NULL;
        }
        opp_obs_data = (float*)PyArray_DATA(arr);
    }

    // Set opponent rewards buffer
    float* opp_rew_data = NULL;
    if (opp_rew_arr != Py_None) {
        if (!PyArray_Check(opp_rew_arr)) {
            PyErr_SetString(PyExc_TypeError, "opponent_rewards must be a numpy array or None");
            return NULL;
        }
        PyArrayObject* arr = (PyArrayObject*)opp_rew_arr;
        if (PyArray_TYPE(arr) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_TypeError, "opponent_rewards must be float32 dtype");
            return NULL;
        }
        opp_rew_data = (float*)PyArray_DATA(arr);
    }

    // Set buffers for each environment
    // Each env gets a slice: env[i] -> opp_obs_data + i*obs_size, opp_rew_data + i
    for (int i = 0; i < vec->num_envs; i++) {
        if (opp_obs_data != NULL) {
            vec->envs[i]->opponent_observations = opp_obs_data + i * obs_size;
        } else {
            vec->envs[i]->opponent_observations = NULL;
        }
        if (opp_rew_data != NULL) {
            vec->envs[i]->opponent_rewards = opp_rew_data + i;
        } else {
            vec->envs[i]->opponent_rewards = NULL;
        }
    }

    Py_RETURN_NONE;
}

// Set eval spawn mode for all environments
// Args: vec_handle, mode (0=random, 1=opponent_advantage)
static PyObject* vec_set_eval_spawn_mode(PyObject* self, PyObject* args) {
    PyObject* vec_arg;
    int mode;

    if (!PyArg_ParseTuple(args, "Oi", &vec_arg, &mode)) {
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(vec_arg);
    if (!vec) {
        PyErr_SetString(PyExc_TypeError, "Invalid vec handle");
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        vec->envs[i]->eval_spawn_mode = mode;
    }

    Py_RETURN_NONE;
}
