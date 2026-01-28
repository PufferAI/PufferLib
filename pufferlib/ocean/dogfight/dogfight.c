// Standalone C demo for Dogfight environment
// Build: ./scripts/build_ocean.sh dogfight local
// Run:   ./dogfight
//
// Hold LEFT_SHIFT for human control, release for AI autopilot
//
// Flight Stick (Logitech Extreme 3D or similar):
//   Stick X        - Roll (push right = roll right)
//   Stick Y        - Pitch (push forward = nose down)
//   Twist          - Rudder (twist right = yaw right)
//   Throttle       - Throttle (forward = more power)
//   Trigger        - Fire
//
// Keyboard (while holding SHIFT):
//   W/S     - Pitch down/up
//   A/D     - Roll left/right
//   Q/E     - Yaw left/right
//   Up/Down - Throttle up/down
//   Space   - Fire
//
// Global Keys:
//   R       - Restart episode
//   ESC     - Quit

#include <time.h>
#include <math.h>
#include "dogfight.h"
#include "puffernet.h"

// Linux joystick API for flight sticks (bypasses GLFW gamepad abstraction)
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <linux/joystick.h>
#include <sys/ioctl.h>
#include <errno.h>

typedef struct {
    int fd;
    char name[80];
    int num_axes;
    int num_buttons;
    float axes[8];      // Up to 8 axes
    int buttons[16];    // Up to 16 buttons
} LinuxJoystick;

static LinuxJoystick* open_linux_joystick(const char* device) {
    int fd = open(device, O_RDONLY | O_NONBLOCK);
    if (fd < 0) return NULL;

    LinuxJoystick* js = calloc(1, sizeof(LinuxJoystick));
    js->fd = fd;

    ioctl(fd, JSIOCGNAME(80), js->name);
    ioctl(fd, JSIOCGAXES, &js->num_axes);
    ioctl(fd, JSIOCGBUTTONS, &js->num_buttons);

    printf("Joystick found: %s\n", js->name);
    printf("  Axes: %d, Buttons: %d\n", js->num_axes, js->num_buttons);

    return js;
}

static void poll_linux_joystick(LinuxJoystick* js) {
    if (!js) return;

    struct js_event event;
    while (read(js->fd, &event, sizeof(event)) > 0) {
        // Mask off init flag
        event.type &= ~JS_EVENT_INIT;

        if (event.type == JS_EVENT_AXIS) {
            if (event.number < 8) {
                js->axes[event.number] = event.value / 32767.0f;
            }
        } else if (event.type == JS_EVENT_BUTTON) {
            if (event.number < 16) {
                js->buttons[event.number] = event.value;
            }
        }
    }
}

static void close_linux_joystick(LinuxJoystick* js) {
    if (js) {
        close(js->fd);
        free(js);
    }
}
#endif // __linux__

#define DOGFIGHT_OBS_SIZE 17
#define DOGFIGHT_ACTION_SIZE 5
#define DOGFIGHT_HIDDEN_SIZE 128
#define DOGFIGHT_NUM_WEIGHTS 135179

static float apply_deadzone(float value, float deadzone) {
    if (fabsf(value) < deadzone) return 0.0f;
    float sign = value > 0.0f ? 1.0f : -1.0f;
    return sign * (fabsf(value) - deadzone) / (1.0f - deadzone);
}

// Box-Muller transform for sampling from normal distribution
static double randn(double mean, double std) {
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    }

    has_spare = 1;
    double u, v, s;
    do {
        u = 2.0 * rand() / RAND_MAX - 1.0;
        v = 2.0 * rand() / RAND_MAX - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

typedef struct LinearContLSTM LinearContLSTM;
struct LinearContLSTM {
    int num_agents;
    float *obs;
    float *log_std;
    Linear *encoder;
    GELU *gelu1;
    LSTM *lstm;
    Linear *actor;
    Linear *value_fn;
    int num_actions;
};

LinearContLSTM *make_linearcontlstm(Weights *weights, int num_agents, int input_dim,
                                    int logit_sizes[], int num_actions) {
    LinearContLSTM *net = calloc(1, sizeof(LinearContLSTM));
    net->num_agents = num_agents;
    net->obs = calloc(num_agents * input_dim, sizeof(float));
    net->num_actions = logit_sizes[0];
    net->log_std = weights->data;
    weights->idx += net->num_actions;
    net->encoder = make_linear(weights, num_agents, input_dim, DOGFIGHT_HIDDEN_SIZE);
    net->gelu1 = make_gelu(num_agents, DOGFIGHT_HIDDEN_SIZE);
    int atn_sum = 0;
    for (int i = 0; i < num_actions; i++) {
        atn_sum += logit_sizes[i];
    }
    net->actor = make_linear(weights, num_agents, DOGFIGHT_HIDDEN_SIZE, atn_sum);
    net->value_fn = make_linear(weights, num_agents, DOGFIGHT_HIDDEN_SIZE, 1);
    net->lstm = make_lstm(weights, num_agents, DOGFIGHT_HIDDEN_SIZE, DOGFIGHT_HIDDEN_SIZE);
    return net;
}

void free_linearcontlstm(LinearContLSTM *net) {
    free(net->obs);
    free(net->encoder);
    free(net->gelu1);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net);
}

void forward_linearcontlstm(LinearContLSTM *net, float *observations, float *actions) {
    linear(net->encoder, observations);
    gelu(net->gelu1, net->encoder->output);
    lstm(net->lstm, net->gelu1->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);
    for (int i = 0; i < net->num_actions; i++) {
        float std = expf(net->log_std[i]);
        float mean = net->actor->output[i];
        actions[i] = randn(mean, std);
    }
}

void demo() {
    srand(time(NULL));

    Weights *weights = load_weights("resources/dogfight/puffer_dogfight_weights.bin", DOGFIGHT_NUM_WEIGHTS);
    int logit_sizes[1] = {DOGFIGHT_ACTION_SIZE};
    LinearContLSTM *net = make_linearcontlstm(weights, 1, DOGFIGHT_OBS_SIZE, logit_sizes, 1);

    int obs_scheme = OBS_MOMENTUM_BETA;
    int obs_size = OBS_SIZES[obs_scheme];

    Dogfight env = {
        .max_steps = 3000,
    };

    // Allocate buffers
    env.observations = (float*)calloc(obs_size, sizeof(float));
    env.actions = (float*)calloc(5, sizeof(float));  // throttle, elevator, aileron, rudder, trigger
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    RewardConfig rcfg = {
        .aim_scale = 0.05f,
        .closing_scale = 0.003f,
        .neg_g = 0.02f,
        .speed_min = 50.0f,
    };

    // curriculum_enabled=1, curriculum_randomize=1 for variety
    init(&env, obs_scheme, &rcfg, 1, 1, 0);
    c_reset(&env);
    c_render(&env);

    SetTargetFPS(60);

#ifdef __linux__
    LinuxJoystick* linux_js = open_linux_joystick("/dev/input/js0");
    if (!linux_js) linux_js = open_linux_joystick("/dev/input/js1");
    if (!linux_js) {
        printf("No joystick found. Hold SHIFT for keyboard control.\n");
    }
#else
    void* linux_js = NULL;
#endif
    printf("Hold LEFT_SHIFT for human control, release for AI autopilot.\n");
    printf("Press R to restart, ESC to quit.\n");

    while (!WindowShouldClose()) {
        // Restart on R key
        int key = GetKeyPressed();
        if (key == KEY_R || key == 'r' || key == 'R') {
            c_reset(&env);
        }

        // SHIFT = human control, otherwise AI flies
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // ============================================
            // HUMAN CONTROL (hold SHIFT)
            // ============================================
#ifdef __linux__
            poll_linux_joystick(linux_js);
#endif

            env.actions[0] = 0.0f;   // throttle (0 = 50% cruise)
            env.actions[1] = 0.0f;   // elevator
            env.actions[2] = 0.0f;   // ailerons
            env.actions[3] = 0.0f;   // rudder
            env.actions[4] = -1.0f;  // trigger (not firing)

#ifdef __linux__
            if (linux_js) {
                // Logitech Extreme 3D Pro mapping:
                // Axis 0 = Stick X (roll)
                // Axis 1 = Stick Y (pitch)
                // Axis 2 = Twist (rudder)
                // Axis 3 = Throttle slider (forward = -1, back = +1)
                // Button 0 = Trigger

                LinuxJoystick* js = linux_js;

                // Pitch: push forward = nose down = positive (stick Y inverted)
                env.actions[1] = -apply_deadzone(js->axes[1], 0.1f);

                // Roll: push right = roll right = positive
                env.actions[2] = apply_deadzone(js->axes[0], 0.1f);

                // Rudder: twist right = yaw right = negative (action convention)
                env.actions[3] = -apply_deadzone(js->axes[2], 0.1f);

                // Throttle: slider forward = more power = positive action
                // Slider reports -1 at forward, +1 at back, so invert
                env.actions[0] = -js->axes[3];

                // Trigger (button 0)
                if (js->buttons[0]) env.actions[4] = 1.0f;
            }
#endif

            // Keyboard controls (always available when SHIFT held)
            if (IsKeyDown(KEY_W)) env.actions[1] = 1.0f;   // Nose down
            if (IsKeyDown(KEY_S)) env.actions[1] = -1.0f;  // Nose up
            if (IsKeyDown(KEY_A)) env.actions[2] = -1.0f;  // Roll left
            if (IsKeyDown(KEY_D)) env.actions[2] = 1.0f;   // Roll right
            if (IsKeyDown(KEY_Q)) env.actions[3] = 1.0f;   // Yaw left
            if (IsKeyDown(KEY_E)) env.actions[3] = -1.0f;  // Yaw right
            if (IsKeyDown(KEY_UP)) env.actions[0] = 1.0f;  // Full throttle
            if (IsKeyDown(KEY_DOWN)) env.actions[0] = -1.0f;  // Idle
            if (IsKeyDown(KEY_SPACE)) env.actions[4] = 1.0f;  // Fire
        } else {
            forward_linearcontlstm(net, env.observations, env.actions);
        }

        c_step(&env);
        c_render(&env);
    }

#ifdef __linux__
    close_linux_joystick(linux_js);
#endif
    c_close(&env);
    free_linearcontlstm(net);
    free(weights);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
}

int main() {
    demo();
    return 0;
}
