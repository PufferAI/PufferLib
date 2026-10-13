"""Exercise CPU policy reset selection without requiring a CUDA toolkit."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_policy_reset_includes_every_physical_policy_slot(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the host reset fixture")
    source = (Path(__file__).resolve().parents[1] / "src/pufferl.cu").read_text()
    start = source.index("static void pufferl_reset_policy_envs(")
    end = source.index("\nstatic void rollout_start(", start)
    fixture = (
        r"""
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#define PUF_CPU 0
#define PUF_BACKEND PUF_CPU
using precision_t = float;
struct Prec { long shape[3]; float* data; };
struct Policy { Prec* buffer_states; };
struct Agent { float* rewards; float* terminals; };
struct Env {
    int tag, num_agents;
    Agent* agents;
    int boundary_reached, resets;
};
struct VecEnv {
    int agents_per_buf, size, num_policies, buffers;
    Env* envs;
    float* rewards;
    int* policy_layout;
};
struct PuffeRL { VecEnv* vec; Policy* policies; int default_stream; };
int uploads = 0, syncs = 0;
void puf_reset(Env* env) { env->resets++; }
void cudaMemsetAsync(void* ptr, int value, size_t size, int) {
    memset(ptr, value, size);
}
void cudaDeviceSynchronize() { syncs++; }
void cpu_upload(PuffeRL*, int, int, int) { uploads++; }
"""
        + source[start:end]
        + r"""
int main() {
    float rewards[12], terminals[12] = {};
    for (int i = 0; i < 12; i++) rewards[i] = 9;
    Agent a[] = {{rewards, terminals}, {rewards+2, terminals+2}, {rewards+4, terminals+4}};
    Agent b[] = {{rewards+1, terminals+1}, {rewards+5, terminals+5}};
    Agent c[] = {{rewards+6, terminals+6}, {rewards+8, terminals+8}, {rewards+10, terminals+10}};
    // Each environment has tag 2, but only a and c contain policy 1.
    Env envs[] = {{2,3,a,1,0}, {2,2,b,1,0}, {2,3,c,1,0}};
    int layout[] = {0, 2, 4, 6};
    float states[3][2][12];
    Prec buffers[3][2];
    Policy policies[3];
    for (int policy = 0; policy < 3; policy++) {
        policies[policy].buffer_states = buffers[policy];
        for (int buf = 0; buf < 2; buf++) {
            buffers[policy][buf] = {{2,2,3}, states[policy][buf]};
            for (int i = 0; i < 12; i++) states[policy][buf][i] = 9;
        }
    }
    VecEnv vec = {6, 3, 3, 2, envs, rewards, layout};
    PuffeRL p = {&vec, policies, 0};
    pufferl_reset_policy_envs(&p, 1);
    assert(envs[0].resets == 1 && envs[2].resets == 1);
    assert(envs[1].resets == 0 && envs[1].boundary_reached == 1);
    assert(envs[0].boundary_reached == 0 && envs[2].boundary_reached == 0);
    for (int i = 0; i < 12; i++) {
        bool reset = i % 2 == 0;
        assert(rewards[i] == (reset ? 0 : 9));
        assert(terminals[i] == (reset ? 1 : 0));
    }
    for (int policy = 0; policy < 3; policy++) {
        for (int buf = 0; buf < 2; buf++) {
            for (int layer = 0; layer < 2; layer++) {
                for (int local = 0; local < 2; local++) {
                    for (int hidden = 0; hidden < 3; hidden++) {
                        assert(states[policy][buf][layer*6 + local*3 + hidden] == (local ? 9 : 0));
                    }
                }
            }
        }
    }
    assert(uploads == 2 && syncs == 1);
}
"""
    )
    cpp = tmp_path / "policy_reset.cc"
    executable = tmp_path / "policy_reset"
    cpp.write_text(fixture)
    subprocess.run(
        [compiler, "-std=c++11", str(cpp), "-o", str(executable)], check=True
    )
    subprocess.run([str(executable)], check=True)
