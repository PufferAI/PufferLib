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

// Register custom methods before including the template
#define MY_METHODS \
    {"env_force_state", (PyCFunction)env_force_state, METH_VARARGS | METH_KEYWORDS, "Force environment state"}, \
    {"env_set_autopilot", (PyCFunction)env_set_autopilot, METH_VARARGS | METH_KEYWORDS, "Set opponent autopilot mode"}, \
    {"vec_set_autopilot", (PyCFunction)vec_set_autopilot, METH_VARARGS | METH_KEYWORDS, "Set autopilot for all envs"}, \
    {"vec_set_mode_weights", (PyCFunction)vec_set_mode_weights, METH_VARARGS | METH_KEYWORDS, "Set mode weights for all envs"}, \
    {"env_get_autopilot_mode", (PyCFunction)env_get_autopilot_mode, METH_VARARGS, "Get current autopilot mode"}

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
    int obs_scheme = get_int(kwargs, "obs_scheme", 0);  // Default to world frame

    // Build reward config from kwargs (all sweepable via INI)
    RewardConfig rcfg = {
        .kill = get_float(kwargs, "reward_kill", 1.0f),
        .hit = get_float(kwargs, "reward_hit", 0.5f),
        .dist_scale = get_float(kwargs, "reward_dist_scale", 0.0001f),
        .closing_scale = get_float(kwargs, "reward_closing_scale", 0.002f),
        .tail_scale = get_float(kwargs, "reward_tail_scale", 0.05f),
        .tracking = get_float(kwargs, "reward_tracking", 0.05f),
        .firing_solution = get_float(kwargs, "reward_firing_solution", 0.1f),
        .alt_low = get_float(kwargs, "penalty_alt_low", 0.0005f),
        .alt_high = get_float(kwargs, "penalty_alt_high", 0.0002f),
        .stall = get_float(kwargs, "penalty_stall", 0.002f),
        .alt_min = get_float(kwargs, "alt_min", 200.0f),
        .alt_max = get_float(kwargs, "alt_max", 2500.0f),
        .speed_min = get_float(kwargs, "speed_min", 50.0f),
    };

    init(env, obs_scheme, &rcfg);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);  // Kill rate (0-1)
    assign_to_dict(dict, "kills", log->kills);
    assign_to_dict(dict, "deaths", log->deaths);
    assign_to_dict(dict, "shots_fired", log->shots_fired);
    assign_to_dict(dict, "shots_hit", log->shots_hit);
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
