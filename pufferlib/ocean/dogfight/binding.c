#include "dogfight.h"

#define Env Dogfight

// We need Python.h for the forward declaration, but env_binding.h includes it
// So we'll put the forward decl and MY_METHODS after including env_binding.h
// but we need MY_METHODS defined before... Let's restructure.

// Include Python first to get PyObject type
#include <Python.h>

// Forward declare our custom methods
static PyObject* env_force_state(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* env_set_autopilot(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* vec_set_autopilot(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* vec_set_mode_weights(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* env_get_autopilot_mode(PyObject* self, PyObject* args);
static PyObject* env_get_state(PyObject* self, PyObject* args);
static PyObject* env_set_obs_highlight(PyObject* self, PyObject* args);

// Register custom methods before including the template
#define MY_METHODS \
    {"env_force_state", (PyCFunction)env_force_state, METH_VARARGS | METH_KEYWORDS, "Force environment state"}, \
    {"env_set_autopilot", (PyCFunction)env_set_autopilot, METH_VARARGS | METH_KEYWORDS, "Set opponent autopilot mode"}, \
    {"vec_set_autopilot", (PyCFunction)vec_set_autopilot, METH_VARARGS | METH_KEYWORDS, "Set autopilot for all envs"}, \
    {"vec_set_mode_weights", (PyCFunction)vec_set_mode_weights, METH_VARARGS | METH_KEYWORDS, "Set mode weights for all envs"}, \
    {"env_get_autopilot_mode", (PyCFunction)env_get_autopilot_mode, METH_VARARGS, "Get current autopilot mode"}, \
    {"env_get_state", (PyCFunction)env_get_state, METH_VARARGS, "Get raw player state"}, \
    {"env_set_obs_highlight", (PyCFunction)env_set_obs_highlight, METH_VARARGS, "Set observation indices to highlight with red arrows"}

// Helper to get float from kwargs with default (before env_binding.h since my_init uses it)
static float get_float(PyObject *kwargs, const char *key, float default_val) {
    if (!kwargs) return default_val;
    PyObject *val = PyDict_GetItemString(kwargs, key);
    if (!val) return default_val;
    if (PyFloat_Check(val)) return (float)PyFloat_AsDouble(val);
    if (PyLong_Check(val)) return (float)PyLong_AsLong(val);
    return default_val;
}

// Helper to get int from kwargs with default
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

    float advance_threshold = get_float(kwargs, "advance_threshold", 0.7f);

    int env_num = get_int(kwargs, "env_num", 0);

    int physics_mode = get_int(kwargs, "physics_mode", 0);

    init(env, obs_scheme, &rcfg, physics_mode, curriculum_enabled, curriculum_randomize, advance_threshold, env_num);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "shots_fired", log->shots_fired);
    assign_to_dict(dict, "accuracy", log->accuracy);
    assign_to_dict(dict, "stage", log->stage);
    assign_to_dict(dict, "total_stage_weight", log->total_stage_weight);
    assign_to_dict(dict, "avg_stage_weight", log->avg_stage_weight);
    assign_to_dict(dict, "avg_abs_bias", log->avg_abs_bias);
    assign_to_dict(dict, "ultimate", log->ultimate);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

// Force state wrapper - unpacks kwargs and calls C function
static PyObject* env_force_state(PyObject* self, PyObject* args, PyObject* kwargs) {
    // First arg is env handle
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "env_force_state requires 1 positional arg (env handle)");
        return NULL;
    }

    Env* env = unpack_env(args);
    if (!env) return NULL;

    // Extract all parameters with defaults
    // Player position
    float p_px = get_float(kwargs, "p_px", 0.0f);
    float p_py = get_float(kwargs, "p_py", 0.0f);
    float p_pz = get_float(kwargs, "p_pz", 1000.0f);

    // Player velocity
    float p_vx = get_float(kwargs, "p_vx", 150.0f);
    float p_vy = get_float(kwargs, "p_vy", 0.0f);
    float p_vz = get_float(kwargs, "p_vz", 0.0f);

    // Player orientation (identity quat = wings level, flying +X)
    float p_ow = get_float(kwargs, "p_ow", 1.0f);
    float p_ox = get_float(kwargs, "p_ox", 0.0f);
    float p_oy = get_float(kwargs, "p_oy", 0.0f);
    float p_oz = get_float(kwargs, "p_oz", 0.0f);

    // Player throttle
    float p_throttle = get_float(kwargs, "p_throttle", 1.0f);

    // Opponent position (-9999 = auto: 400m ahead)
    float o_px = get_float(kwargs, "o_px", -9999.0f);
    float o_py = get_float(kwargs, "o_py", -9999.0f);
    float o_pz = get_float(kwargs, "o_pz", -9999.0f);

    // Opponent velocity (-9999 = auto: match player)
    float o_vx = get_float(kwargs, "o_vx", -9999.0f);
    float o_vy = get_float(kwargs, "o_vy", -9999.0f);
    float o_vz = get_float(kwargs, "o_vz", -9999.0f);

    // Opponent orientation (-9999 = auto: match player)
    float o_ow = get_float(kwargs, "o_ow", -9999.0f);
    float o_ox = get_float(kwargs, "o_ox", -9999.0f);
    float o_oy = get_float(kwargs, "o_oy", -9999.0f);
    float o_oz = get_float(kwargs, "o_oz", -9999.0f);

    // Environment tick
    int tick = get_int(kwargs, "tick", 0);

    // Call the C function
    force_state(env,
        p_px, p_py, p_pz,
        p_vx, p_vy, p_vz,
        p_ow, p_ox, p_oy, p_oz,
        p_throttle,
        o_px, o_py, o_pz,
        o_vx, o_vy, o_vz,
        o_ow, o_ox, o_oy, o_oz,
        tick
    );

    Py_RETURN_NONE;
}

// Set autopilot mode for opponent aircraft
static PyObject* env_set_autopilot(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "env_set_autopilot requires 1 positional arg (env handle)");
        return NULL;
    }

    Env* env = unpack_env(args);
    if (!env) return NULL;

    // Get autopilot parameters
    int mode = get_int(kwargs, "mode", AP_STRAIGHT);
    if (mode < 0 || mode >= AP_COUNT) mode = AP_STRAIGHT;  // Bounds check
    float throttle = get_float(kwargs, "throttle", AP_DEFAULT_THROTTLE);
    float bank_deg = get_float(kwargs, "bank_deg", AP_DEFAULT_BANK_DEG);
    float climb_rate = get_float(kwargs, "climb_rate", AP_DEFAULT_CLIMB_RATE);

    // Set the autopilot mode
    autopilot_set_mode(&env->opponent_ap, (AutopilotMode)mode, throttle, bank_deg, climb_rate);

    Py_RETURN_NONE;
}

// Set autopilot mode for all environments (vectorized)
static PyObject* vec_set_autopilot(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_set_autopilot requires 1 positional arg (vec handle)");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    // Get autopilot parameters
    int mode = get_int(kwargs, "mode", AP_STRAIGHT);
    if (mode < 0 || mode >= AP_COUNT) mode = AP_STRAIGHT;  // Bounds check
    float throttle = get_float(kwargs, "throttle", AP_DEFAULT_THROTTLE);
    float bank_deg = get_float(kwargs, "bank_deg", AP_DEFAULT_BANK_DEG);
    float climb_rate = get_float(kwargs, "climb_rate", AP_DEFAULT_CLIMB_RATE);

    // Set autopilot for all environments
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

// Get current autopilot mode (for testing/debugging)
static PyObject* env_get_autopilot_mode(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env) return NULL;

    return PyLong_FromLong((long)env->opponent_ap.mode);
}

// Get raw player state (for physics tests - independent of obs_scheme)
static PyObject* env_get_state(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env) return NULL;

    Plane* p = &env->player;
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;

    // Position
    PyDict_SetItemString(dict, "px", PyFloat_FromDouble(p->pos.x));
    PyDict_SetItemString(dict, "py", PyFloat_FromDouble(p->pos.y));
    PyDict_SetItemString(dict, "pz", PyFloat_FromDouble(p->pos.z));

    // Velocity
    PyDict_SetItemString(dict, "vx", PyFloat_FromDouble(p->vel.x));
    PyDict_SetItemString(dict, "vy", PyFloat_FromDouble(p->vel.y));
    PyDict_SetItemString(dict, "vz", PyFloat_FromDouble(p->vel.z));

    // Orientation quaternion
    PyDict_SetItemString(dict, "ow", PyFloat_FromDouble(p->ori.w));
    PyDict_SetItemString(dict, "ox", PyFloat_FromDouble(p->ori.x));
    PyDict_SetItemString(dict, "oy", PyFloat_FromDouble(p->ori.y));
    PyDict_SetItemString(dict, "oz", PyFloat_FromDouble(p->ori.z));

    // Up vector (derived)
    PyDict_SetItemString(dict, "up_x", PyFloat_FromDouble(up.x));
    PyDict_SetItemString(dict, "up_y", PyFloat_FromDouble(up.y));
    PyDict_SetItemString(dict, "up_z", PyFloat_FromDouble(up.z));

    // Forward vector (derived)
    PyDict_SetItemString(dict, "fwd_x", PyFloat_FromDouble(fwd.x));
    PyDict_SetItemString(dict, "fwd_y", PyFloat_FromDouble(fwd.y));
    PyDict_SetItemString(dict, "fwd_z", PyFloat_FromDouble(fwd.z));

    // Throttle
    PyDict_SetItemString(dict, "throttle", PyFloat_FromDouble(p->throttle));

    // G-force (current G-loading)
    PyDict_SetItemString(dict, "g_force", PyFloat_FromDouble(p->g_force));

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
