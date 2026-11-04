// Env-specific tests: tests/*.c next to the env header, compiled into the env's
// test binary in the same translation unit as the env.
//
//   #include "puf_test.h"
//   PUF_TEST(paddle_stays_on_screen) {
//       Env* env = puf_test_env(7);        // fresh env for seed 7, after reset
//       for (int i = 0; i < 100; i++) {
//           env->agents[0].actions[0] = 1;
//           puf_test_step(env);            // puf_step plus the generic checks
//           PUF_CHECK(env->paddle_x <= env->width);
//       }
//       puf_test_close(env);
//   }
//
// A failed PUF_CHECK is reported with its file and line; the test goes on.
#pragma once

typedef void (*PufTestFn)(void);

void puf_test_register(const char* name, PufTestFn fn);
void puf_test_fail(const char* file, int line, const char* what);
Env* puf_test_env(unsigned int seed);
void puf_test_step(Env* env);
void puf_test_close(Env* env);

#define PUF_TEST(name)                                                      \
    static void name(void);                                                 \
    __attribute__((constructor)) static void name##_register(void) {        \
        puf_test_register(#name, name);                                     \
    }                                                                       \
    static void name(void)

#define PUF_CHECK(cond)                                                     \
    do {                                                                    \
        if (!(cond)) {                                                      \
            puf_test_fail(__FILE__, __LINE__, #cond);                       \
        }                                                                   \
    } while (0)
