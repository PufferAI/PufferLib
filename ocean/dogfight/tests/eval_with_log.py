"""eval_with_log.py — Render eval with periodic log dumps to stdout.

User watches the raylib window; Claude reads the log dumps from stdout.
Replicates `pufferl eval` but dumps `_C.log(pufferl)` every N rollouts so the
agent's actual per-snapshot metrics (perf, avg_signed_bias, accuracy, etc.)
are visible alongside the rendered behavior.
"""
import argparse
import os
import sys
import json
import time

REPO = '/home/keith/Git/ml/p4'
sys.path.insert(0, REPO)

from pufferlib import _C
from pufferlib.pufferl import load_config

# Replicate load_config without parse_args for clean overrides
def build_args(overrides):
    import ast, configparser
    p = configparser.ConfigParser()
    p.read([os.path.join(REPO, 'config/default.ini'),
            os.path.join(REPO, 'config/dogfight.ini')])
    args = {}
    for s in p.sections():
        args.setdefault(s, {})
        for k in p[s]:
            try:
                args[s][k] = ast.literal_eval(p[s][k])
            except (ValueError, SyntaxError):
                args[s][k] = p[s][k]
    for k, v in args.pop('base', {}).items():
        args[k] = v
    args.setdefault('load_id', None)
    args.setdefault('wandb', False)
    args.setdefault('wandb_project', 'puffer4')
    args.setdefault('wandb_group', 'debug')
    args.setdefault('tag', None)
    args.setdefault('slowly', False)
    args.setdefault('save_frames', 0)
    args.setdefault('gif_path', 'eval.gif')
    args.setdefault('fps', 15)
    args.setdefault('render_mode', 'raylib')
    args.setdefault('reset_state', False)
    args.setdefault('world_size', 1)
    args.setdefault('nccl_id', b'')
    args.setdefault('load_model_path', 'latest')
    args.update(overrides)
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='latest')
    ap.add_argument('--curriculum-enabled', type=int, default=1)
    ap.add_argument('--curriculum-randomize', type=int, default=1)
    ap.add_argument('--log-every', type=int, default=20,
                    help='Print _C.log(pufferl) every N rollouts')
    ap.add_argument('--max-rollouts', type=int, default=2000)
    cli = ap.parse_args()

    args = build_args({'load_model_path': cli.model})
    args['env']['curriculum_enabled'] = cli.curriculum_enabled
    args['env']['curriculum_randomize'] = cli.curriculum_randomize
    args['train']['horizon'] = 1  # render 1 step at a time
    args['vec']['num_buffers'] = 1
    args['reset_state'] = False

    backend = _C
    pufferl = backend.create_pufferl(args)

    # Resolve model path for C backend (.bin in checkpoint_dir/env_name/**/)
    import glob
    load_path = cli.model
    if load_path == 'latest':
        pattern = os.path.join(args['checkpoint_dir'], args['env_name'], '**', '*.bin')
        candidates = glob.glob(pattern, recursive=True)
        load_path = max(candidates, key=os.path.getctime)
    backend.load_weights(pufferl, load_path)
    print(f'[eval] Loaded {load_path}', flush=True)
    print(f'[eval] curriculum_enabled={cli.curriculum_enabled} '
          f'curriculum_randomize={cli.curriculum_randomize}', flush=True)
    print(f'[eval] log every {cli.log_every} rollouts; max {cli.max_rollouts}',
          flush=True)

    t0 = time.time()
    last_log = {}
    for r in range(cli.max_rollouts):
        backend.render(pufferl, 0)
        backend.rollouts(pufferl)
        if r > 0 and r % cli.log_every == 0:
            log = backend.log(pufferl)
            elapsed = time.time() - t0
            # Flatten and keep only env-side numeric scalars
            flat = {}
            for k, v in log.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, (int, float)):
                            flat[f'{k}/{k2}'] = float(v2)
                elif isinstance(v, (int, float)):
                    flat[k] = float(v)
            keep = ['perf', 'score', 'episode_return', 'episode_length',
                    'accuracy', 'shots_fired', 'stage', 'avg_stage',
                    'avg_abs_bias', 'avg_signed_bias', 'avg_control_rate',
                    'player_ground', 'opponent_ground', 'clean_fights', 'n']
            env_log = {k: flat[k] for k in keep if k in flat}
            for k, v in flat.items():
                if k.startswith('environment/') and k not in env_log:
                    env_log[k] = v
            print(f"[r={r:5d} t={elapsed:6.1f}s] " +
                  ' '.join(f"{k}={v:+.3f}" for k, v in env_log.items()),
                  flush=True)
            last_log = env_log

    backend.close(pufferl)


if __name__ == '__main__':
    main()
