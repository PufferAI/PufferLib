import os
import subprocess
import textwrap


def test_cartpole_step_matches_reference_dynamics(tmp_path):
    fake_raylib = tmp_path / 'raylib.h'
    fake_raylib.write_text(textwrap.dedent('''
        typedef struct Color { unsigned char r, g, b, a; } Color;
        typedef struct Vector2 { float x, y; } Vector2;
        #define KEY_ESCAPE 0
        #define KEY_TAB 1
        static inline void InitWindow(int width, int height, const char* title) {}
        static inline void SetTargetFPS(int fps) {}
        static inline void CloseWindow(void) {}
        static inline int IsKeyDown(int key) { return 0; }
        static inline int IsKeyPressed(int key) { return 0; }
        static inline void ToggleFullscreen(void) {}
        static inline void BeginDrawing(void) {}
        static inline void ClearBackground(Color color) {}
        static inline void DrawLine(int startPosX, int startPosY, int endPosX, int endPosY, Color color) {}
        static inline void DrawRectangle(int posX, int posY, int width, int height, Color color) {}
        static inline void DrawLineEx(Vector2 startPos, Vector2 endPos, float thick, Color color) {}
        static inline void DrawText(const char* text, int posX, int posY, int fontSize, Color color) {}
        static inline const char* TextFormat(const char* text, ...) { return text; }
        static inline void EndDrawing(void) {}
    '''))

    source = tmp_path / 'cartpole_step_test.c'
    source.write_text(textwrap.dedent('''
        #include <math.h>
        #include <stdio.h>
        #include "cartpole.h"

        int main(void) {
            float observations[4] = {0};
            float actions[1] = {1.0f};
            float rewards[1] = {0};
            float terminals[1] = {0};

            Cartpole env = {0};
            env.observations = observations;
            env.actions = actions;
            env.rewards = rewards;
            env.terminals = terminals;
            env.cart_mass = 1.0f;
            env.pole_mass = 0.1f;
            env.pole_length = 0.5f;
            env.gravity = 9.8f;
            env.force_mag = 10.0f;
            env.tau = 0.02f;
            env.continuous = 0;
            env.x = 0.0f;
            env.x_dot = 0.0f;
            env.theta = 0.05f;
            env.theta_dot = 0.1f;

            const float initial_theta_dot = env.theta_dot;
            const float force = env.force_mag;
            const float costheta = cosf(env.theta);
            const float sintheta = sinf(env.theta);
            const float total_mass = env.cart_mass + env.pole_mass;
            const float polemass_length = env.pole_mass * env.pole_length;
            const float temp = (force + polemass_length * initial_theta_dot * initial_theta_dot * sintheta) / total_mass;
            const float thetaacc = (env.gravity * sintheta - costheta * temp) /
                (env.pole_length * (4.0f / 3.0f - env.pole_mass * costheta * costheta / total_mass));
            const float expected_theta_dot = initial_theta_dot + env.tau * thetaacc;

            c_step(&env);

            if (fabsf(env.theta_dot - expected_theta_dot) > 1e-6f) {
                fprintf(stderr, "theta_dot mismatch: got %.9f expected %.9f\\n", env.theta_dot, expected_theta_dot);
                return 1;
            }
            return 0;
        }
    '''))

    binary = tmp_path / 'cartpole_step_test'
    repo_root = os.getcwd()
    subprocess.run([
        os.environ.get('CC', 'cc'),
        '-std=gnu11',
        '-I', str(tmp_path),
        '-I', os.path.join(repo_root, 'ocean', 'cartpole'),
        str(source),
        '-lm',
        '-o', str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
