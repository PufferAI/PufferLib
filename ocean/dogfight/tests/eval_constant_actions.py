"""eval_constant_actions.py — Render eval but BYPASS the policy.

Writes constant level-flight actions every step instead of policy output.
If the plane still tumbles, physics is suspect. If it flies fine, the
trained policy is the source of the tumbling, as expected.
"""
import argparse
import os
import sys
import time

REPO = '/home/keith/Git/ml/p4'
sys.path.insert(0, REPO)

import torch
from pufferlib import _C
from pufferlib.torch_pufferl import _CudaPtr, _cpu_tensor

# Reuse build_args from diagnose_right_bias
sys.path.insert(0, os.path.join(REPO, 'ocean/dogfight/tests'))
from diagnose_right_bias import build_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curriculum-enabled', type=int, default=1)
    ap.add_argument('--curriculum-randomize', type=int, default=1)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--throttle', type=float, default=0.7)
    ap.add_argument('--elevator', type=float, default=0.0)
    ap.add_argument('--aileron', type=float, default=0.0)
    ap.add_argument('--rudder', type=float, default=0.0)
    ap.add_argument('--trigger', type=float, default=0.0)
    cli = ap.parse_args()

    args = build_args('dogfight', total_agents=1, model_path=None,
                      max_steps=300)
    args['env']['curriculum_enabled'] = cli.curriculum_enabled
    args['env']['curriculum_randomize'] = cli.curriculum_randomize

    print(f"[const] actions: throttle={cli.throttle} elevator={cli.elevator} "
          f"aileron={cli.aileron} rudder={cli.rudder} trigger={cli.trigger}",
          flush=True)
    print(f"[const] curriculum_enabled={cli.curriculum_enabled} "
          f"curriculum_randomize={cli.curriculum_randomize}", flush=True)

    vec = _C.create_vec(args, _C.gpu)
    device = 'cuda' if _C.gpu else 'cpu'
    if _C.gpu:
        vec_obs = torch.as_tensor(_CudaPtr(vec.gpu_obs_ptr,
            (vec.total_agents, vec.obs_size), torch.float32))
    else:
        vec_obs = _cpu_tensor(vec.obs_ptr,
            (vec.total_agents, vec.obs_size), torch.float32)

    vec.reset()

    # Action layout: [num_atns, total_agents] (transposed)
    consts = torch.tensor([cli.throttle, cli.elevator, cli.aileron,
                            cli.rudder, cli.trigger],
                           dtype=torch.float32, device=device)
    action_buf = consts.unsqueeze(1).repeat(1, vec.total_agents).contiguous()

    t0 = time.time()
    for s in range(cli.steps):
        vec.render(0)
        if _C.gpu:
            vec.gpu_step(action_buf.data_ptr())
            torch.cuda.synchronize()
        else:
            vec.cpu_step(action_buf.data_ptr())
        if s > 0 and s % 200 == 0:
            obs_cpu = torch.as_tensor(vec_obs).detach().cpu().numpy()[0]
            elapsed = time.time() - t0
            # obs[7] = altitude (0..1), obs[15] = range, obs[13] = az
            print(f"[s={s:5d} t={elapsed:5.1f}s] alt={obs_cpu[7]:.3f} "
                  f"az={obs_cpu[13]:+.3f} el={obs_cpu[14]:+.3f} "
                  f"range={obs_cpu[15]:.3f} g={obs_cpu[8]:+.3f}",
                  flush=True)


if __name__ == '__main__':
    main()
