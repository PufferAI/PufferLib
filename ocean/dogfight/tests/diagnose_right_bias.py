"""diagnose_right_bias.py — Quantify whether the trained dogfight policy
turns symmetrically based on opponent azimuth, or always banks right.

Outputs (Claude-readable):
  Tier 1 (stdout): summary with VERDICT line.
  Tier 2 (--trace path): JSONL with per-step (obs[13], action[2], full obs/action,
    reward, terminal) for offline grep/analysis.

Usage:
  python ocean/dogfight/tests/diagnose_right_bias.py \
      --model latest --steps 2000 --total-agents 64 \
      --trace /tmp/df_diag_trace.jsonl

Slot 13 in obs is target_az (signed [-1,+1], + = opponent on right).
Slot 2 in action is aileron (signed [-1,+1], + = roll right).
A symmetric policy should have action[2] sign roughly tracking obs[13] sign
(turn toward the opponent), giving a high "sign-match" rate and positive
correlation. The user reports the trained agent always banks right —
this script quantifies that claim.
"""
import argparse
import ast
import configparser
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = '/home/keith/Git/ml/p4'
sys.path.insert(0, REPO)

from pufferlib import _C
from pufferlib.torch_pufferl import PuffeRL, load_policy, sample_logits


def build_args(env_name, total_agents, model_path, max_steps):
    """Re-implement pufferl.load_config without parse_args side effects."""
    p = configparser.ConfigParser()
    default_ini = os.path.join(REPO, 'config/default.ini')
    env_ini = os.path.join(REPO, 'config', f'{env_name}.ini')
    p.read([default_ini, env_ini])
    args = {}
    for section in p.sections():
        if section not in args:
            args[section] = {}
        for key in p[section]:
            try:
                args[section][key] = ast.literal_eval(p[section][key])
            except (ValueError, SyntaxError):
                args[section][key] = p[section][key]
    # Promote 'base' keys (env_name) to top-level dict — matches pufferl flow
    for k, v in args.pop('base', {}).items():
        args[k] = v
    # Defaults that pufferl sets via argparse but aren't in INI
    args.setdefault('load_model_path', model_path)
    args.setdefault('load_id', None)
    args.setdefault('wandb', False)
    args.setdefault('wandb_project', 'puffer4')
    args.setdefault('wandb_group', 'debug')
    args.setdefault('tag', None)
    args.setdefault('slowly', True)
    args.setdefault('save_frames', 0)
    args.setdefault('gif_path', 'eval.gif')
    args.setdefault('fps', 15)
    args.setdefault('render_mode', 'None')
    args.setdefault('reset_state', False)
    args.setdefault('world_size', 1)
    args.setdefault('nccl_id', b'')
    # Override for diagnostic run
    args['vec']['total_agents'] = total_agents
    args['vec']['num_buffers'] = 1
    args['train']['horizon'] = 1
    args['train']['gpus'] = 1
    args['env']['max_steps'] = max_steps
    return args


MIRROR_SIGN = [
    +1, -1, +1, -1, +1, -1, +1, +1, +1, +1,
    +1, -1, +1, -1, +1, +1, +1, +1, +1, +1,
    -1, +1, -1, +1, +1, +1,
]
# Source of truth: ocean/dogfight/dogfight_observations.h:690-732


def probe_mirror(policy, vec, device, n_samples, source_jsonl, out_jsonl):
    """Forward obs pairs (real + y-mirrored) through policy. No env stepping.

    Real obs come from source_jsonl (Phase 0 trace) — we sample distinct ones.
    For each, build the mirrored variant via MIRROR_SIGN and run both through
    forward_eval. Record action means; ideal policy yields aileron_sum ≈ 0.
    """
    # Pull obs samples from existing trace
    obs_samples = []
    if not os.path.exists(source_jsonl):
        print(f"  ERROR: {source_jsonl} not found. Run Phase 0 first.", file=sys.stderr)
        return None
    with open(source_jsonl) as f:
        for ln in f:
            try:
                rec = json.loads(ln)
                if rec.get('agent') == 0:
                    obs_samples.append(rec['obs'])
            except json.JSONDecodeError:
                continue
            if len(obs_samples) >= n_samples:
                break
    obs_arr = np.array(obs_samples, dtype=np.float32)
    print(f"  loaded {len(obs_arr)} obs samples from {source_jsonl}")

    sign = np.array(MIRROR_SIGN, dtype=np.float32)
    obs_mirror = obs_arr * sign[None, :]

    obs_real_t = torch.from_numpy(obs_arr).to(device)
    obs_mirror_t = torch.from_numpy(obs_mirror).to(device)

    # forward_eval expects (batch, obs_size). State per sample.
    state = policy.initial_state(len(obs_arr), device=device)
    if state:
        state = tuple(torch.zeros_like(s) for s in state)

    with torch.no_grad():
        logits_real, _, _ = policy.forward_eval(obs_real_t, state)
        logits_mirror, _, _ = policy.forward_eval(obs_mirror_t, state)

    if isinstance(logits_real, torch.distributions.Normal):
        a_real = logits_real.loc.detach().cpu().numpy()
        a_mirror = logits_mirror.loc.detach().cpu().numpy()
    else:
        a_real, _, _ = sample_logits(logits_real)
        a_mirror, _, _ = sample_logits(logits_mirror)
        a_real = a_real.detach().cpu().numpy()
        a_mirror = a_mirror.detach().cpu().numpy()

    # If policy were y-symmetric: aileron[mirror] = -aileron[real],
    # rudder[mirror] = -rudder[real]; throttle/elevator/trigger stay same.
    # So:  aileron_real + aileron_mirror ≈ 0   if symmetric.
    SLOT_AIL = 2
    SLOT_RUD = 3
    ail_real = a_real[:, SLOT_AIL]
    ail_mirror = a_mirror[:, SLOT_AIL]
    ail_sum = ail_real + ail_mirror
    rud_real = a_real[:, SLOT_RUD]
    rud_mirror = a_mirror[:, SLOT_RUD]
    rud_sum = rud_real + rud_mirror

    with open(out_jsonl, 'w') as f:
        for i in range(len(obs_arr)):
            f.write(json.dumps({
                'sample': i,
                'obs_real': obs_arr[i].tolist(),
                'obs_mirror': obs_mirror[i].tolist(),
                'action_real': a_real[i].tolist(),
                'action_mirror': a_mirror[i].tolist(),
                'aileron_sum': float(ail_sum[i]),
                'rudder_sum': float(rud_sum[i]),
            }) + '\n')

    summary = {
        'n': len(obs_arr),
        'aileron_real_mean': float(ail_real.mean()),
        'aileron_mirror_mean': float(ail_mirror.mean()),
        'aileron_sum_mean': float(ail_sum.mean()),
        'aileron_sum_abs_mean': float(np.abs(ail_sum).mean()),
        'rudder_real_mean': float(rud_real.mean()),
        'rudder_mirror_mean': float(rud_mirror.mean()),
        'rudder_sum_mean': float(rud_sum.mean()),
        'rudder_sum_abs_mean': float(np.abs(rud_sum).mean()),
    }

    print()
    print("=" * 60)
    print(" POLICY MIRROR PROBE (no env stepping)")
    print("=" * 60)
    print(f"  n samples            : {summary['n']}")
    print(f"  aileron_real mean    : {summary['aileron_real_mean']:+.3f}")
    print(f"  aileron_mirror mean  : {summary['aileron_mirror_mean']:+.3f}")
    print(f"  aileron_sum mean     : {summary['aileron_sum_mean']:+.3f}   (0 if policy y-symmetric)")
    print(f"  |aileron_sum| mean   : {summary['aileron_sum_abs_mean']:+.3f}")
    print(f"  rudder_real mean     : {summary['rudder_real_mean']:+.3f}")
    print(f"  rudder_mirror mean   : {summary['rudder_mirror_mean']:+.3f}")
    print(f"  rudder_sum mean      : {summary['rudder_sum_mean']:+.3f}")
    print(f"  |rudder_sum| mean    : {summary['rudder_sum_abs_mean']:+.3f}")
    if abs(summary['aileron_sum_mean']) > 0.1:
        verdict = "POLICY_GLOBALLY_BIASED (aileron does not flip under input mirror)"
    elif summary['aileron_sum_abs_mean'] > 0.2:
        verdict = "POLICY_LOCALLY_ASYMMETRIC (per-sample variance large but mean ~0)"
    else:
        verdict = "POLICY_SYMMETRIC (mirror probe clean)"
    print(f"  VERDICT: {verdict}")
    print("=" * 60)
    print(f"  trace -> {out_jsonl}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='latest', help='Path or "latest"')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--total-agents', type=int, default=64)
    ap.add_argument('--max-steps-per-episode', type=int, default=300)
    ap.add_argument('--trace', default='/tmp/df_diag_trace.jsonl')
    ap.add_argument('--summary', default='/tmp/df_diag_summary.json')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--probe-mirror', action='store_true',
                    help='Forward obs pairs (real, y-mirrored) through policy '
                         'and check action[2] (aileron) and action[3] (rudder) '
                         'mirror-anti-symmetry.')
    ap.add_argument('--probe-source', default='/tmp/df_diag_trace.jsonl',
                    help='JSONL file from a prior run to draw obs samples from')
    ap.add_argument('--probe-out', default='/tmp/df_policy_probe.jsonl')
    ap.add_argument('--probe-n', type=int, default=200)
    ap.add_argument('--heatmap', action='store_true',
                    help='Sweep obs[13] (azimuth) and obs[14] (elevation) '
                         'over [-1,+1] grid with all other slots held at '
                         'a sane "level flight" baseline; report aileron + '
                         'rudder + elevator at each grid point.')
    ap.add_argument('--heatmap-out', default='/tmp/df_policy_heatmap.jsonl')
    ap.add_argument('--heatmap-grid', type=int, default=11)
    ap.add_argument('--heatmap-baseline-source',
                    default='/tmp/df_diag_trace.jsonl',
                    help='Use the median obs from this trace as the baseline '
                         '(everything except slots 13/14). Falls back to '
                         'all-zeros if the file is missing.')
    ap.add_argument('--render', action='store_true',
                    help='Render env 0 each step (raylib window opens). '
                         'Combine with --env.curriculum-enabled 1 / '
                         '--env.curriculum-randomize 1 to watch varied stages.')
    ap.add_argument('--curriculum-enabled', type=int, default=None,
                    help='Override env.curriculum_enabled (0/1)')
    ap.add_argument('--curriculum-randomize', type=int, default=None,
                    help='Override env.curriculum_randomize (0/1)')
    cli = ap.parse_args()

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    args = build_args('dogfight', cli.total_agents, cli.model,
                      cli.max_steps_per_episode)
    if cli.curriculum_enabled is not None:
        args['env']['curriculum_enabled'] = cli.curriculum_enabled
    if cli.curriculum_randomize is not None:
        args['env']['curriculum_randomize'] = cli.curriculum_randomize
    if cli.verbose:
        print(f"  obs_scheme = {args['env']['obs_scheme']}")
        print(f"  total_agents = {args['vec']['total_agents']}")
        print(f"  curriculum_enabled = {args['env']['curriculum_enabled']}")
        print(f"  load_model_path = {args['load_model_path']}")

    # Build vec env (will use GPU if _C built for it; _C.gpu=1 in this build)
    vec = _C.create_vec(args, _C.gpu)
    print(f"  vec.total_agents={vec.total_agents} obs_size={vec.obs_size} "
          f"num_atns={vec.num_atns}")

    # Load policy + checkpoint
    policy = load_policy(args, vec)
    policy.eval()
    device = 'cuda' if _C.gpu else 'cpu'

    if cli.probe_mirror:
        probe_mirror(policy, vec, device, cli.probe_n,
                     cli.probe_source, cli.probe_out)
        return

    if cli.heatmap:
        run_heatmap(policy, vec, device, cli.heatmap_grid,
                    cli.heatmap_baseline_source, cli.heatmap_out)
        return

    # Wire CPU/GPU pointers as torch tensors (zero-copy)
    if _C.gpu:
        from pufferlib.torch_pufferl import _CudaPtr
        vec_obs = torch.as_tensor(_CudaPtr(vec.gpu_obs_ptr,
            (vec.total_agents, vec.obs_size), torch.float32))
        vec_rewards = torch.as_tensor(_CudaPtr(vec.gpu_rewards_ptr,
            (vec.total_agents,), torch.float32))
        vec_terminals = torch.as_tensor(_CudaPtr(vec.gpu_terminals_ptr,
            (vec.total_agents,), torch.float32))
    else:
        from pufferlib.torch_pufferl import _cpu_tensor
        vec_obs = _cpu_tensor(vec.obs_ptr,
            (vec.total_agents, vec.obs_size), torch.float32)
        vec_rewards = _cpu_tensor(vec.rewards_ptr,
            (vec.total_agents,), torch.float32)
        vec_terminals = _cpu_tensor(vec.terminals_ptr,
            (vec.total_agents,), torch.float32)

    vec.reset()
    state = policy.initial_state(vec.total_agents, device=device)
    if state:
        state = tuple(torch.zeros_like(s) for s in state)

    # Allocate action buffer (float32, contiguous: per-atn-slot major as
    # actions_flat.T in rollouts() — see torch_pufferl.py:230).
    action_buf = torch.zeros(vec.num_atns, vec.total_agents,
                             dtype=torch.float32, device=device).contiguous()

    # Stats accumulators
    n = 0
    sum_az = 0.0
    sum_ail = 0.0
    sum_az_ail = 0.0
    sum_az2 = 0.0
    sum_ail2 = 0.0
    sign_match = 0
    bin_left = {'n': 0, 'sum_ail': 0.0, 'sum_az': 0.0}
    bin_right = {'n': 0, 'sum_ail': 0.0, 'sum_az': 0.0}

    # Open trace file
    trace_f = open(cli.trace, 'w')
    print(f"  trace -> {cli.trace}")

    SLOT_AZ = 13
    SLOT_AIL = 2

    for step in range(cli.steps):
        if cli.render:
            vec.render(0)
        obs_t = torch.as_tensor(vec_obs, device=device)
        with torch.no_grad():
            logits, value, state = policy.forward_eval(obs_t, state)
        # Use the mean (deterministic policy output) for analysis,
        # AND for stepping — eliminates sampling noise.
        if isinstance(logits, torch.distributions.Normal):
            action_mean = logits.loc  # [agents, num_atns]
            action_for_step = action_mean.clamp(-1.0, 1.0)
        else:
            # Discrete fallback — sample
            action_for_step, _, _ = sample_logits(logits)
            action_mean = action_for_step.float()

        # Write actions buffer in num_atns-major layout (see torch_pufferl:230)
        action_buf.copy_((action_for_step.T if action_for_step.dim() > 1
                          else action_for_step.unsqueeze(0)).contiguous())

        # Pull obs & action snapshots (cpu numpy) BEFORE stepping
        obs_np = obs_t.detach().cpu().numpy()
        act_np = action_mean.detach().cpu().numpy()

        # Step env
        if _C.gpu:
            vec.gpu_step(action_buf.data_ptr())
            torch.cuda.synchronize()
        else:
            vec.cpu_step(action_buf.data_ptr())

        rew_np = vec_rewards.detach().cpu().numpy().copy()
        term_np = vec_terminals.detach().cpu().numpy().copy()

        # Per-agent accumulators
        az = obs_np[:, SLOT_AZ]
        ail = act_np[:, SLOT_AIL]
        n += len(az)
        sum_az += float(az.sum())
        sum_ail += float(ail.sum())
        sum_az_ail += float((az * ail).sum())
        sum_az2 += float((az * az).sum())
        sum_ail2 += float((ail * ail).sum())
        # Sign-match rate (turn toward opponent ⇔ sign(ail)==sign(az))
        nontrivial = np.abs(az) > 0.05  # ignore near-on-axis
        if nontrivial.any():
            match = (np.sign(az[nontrivial]) == np.sign(ail[nontrivial]))
            sign_match += int(match.sum())
        # Bin by azimuth sign
        left = az < -0.05
        right = az > +0.05
        bin_left['n'] += int(left.sum())
        bin_left['sum_ail'] += float(ail[left].sum())
        bin_left['sum_az'] += float(az[left].sum())
        bin_right['n'] += int(right.sum())
        bin_right['sum_ail'] += float(ail[right].sum())
        bin_right['sum_az'] += float(az[right].sum())

        # Write trace lines (one per agent, lots of data — sample 4 agents
        # per step to keep file size reasonable)
        sample_agents = list(range(min(4, vec.total_agents)))
        for a in sample_agents:
            trace_f.write(json.dumps({
                'step': step,
                'agent': a,
                'obs': obs_np[a].tolist(),
                'action': act_np[a].tolist(),
                'reward': float(rew_np[a]),
                'terminal': bool(term_np[a] > 0.5),
                'az': float(obs_np[a, SLOT_AZ]),
                'ail': float(act_np[a, SLOT_AIL]),
            }) + '\n')

    trace_f.close()

    # ---------- Tier 1 summary -----------------
    n_nontrivial = bin_left['n'] + bin_right['n']
    mean_az = sum_az / n if n else 0.0
    mean_ail = sum_ail / n if n else 0.0
    var_az = max(sum_az2 / n - mean_az**2, 1e-12)
    var_ail = max(sum_ail2 / n - mean_ail**2, 1e-12)
    cov_az_ail = sum_az_ail / n - mean_az * mean_ail
    corr = cov_az_ail / (var_az * var_ail) ** 0.5

    mean_ail_left = (bin_left['sum_ail'] / bin_left['n']) if bin_left['n'] else float('nan')
    mean_ail_right = (bin_right['sum_ail'] / bin_right['n']) if bin_right['n'] else float('nan')
    mean_az_left = (bin_left['sum_az'] / bin_left['n']) if bin_left['n'] else float('nan')
    mean_az_right = (bin_right['sum_az'] / bin_right['n']) if bin_right['n'] else float('nan')

    sign_match_rate = (sign_match / n_nontrivial) if n_nontrivial else float('nan')

    # Verdict logic:
    #   - corr(az, ail) > +0.3   => agent turns toward opponent (healthy)
    #   - corr ≈ 0               => agent ignores opponent direction
    #   - corr < -0.3            => agent turns AWAY from opponent (very bad)
    #   - sign_match < 0.55      => no left/right symmetry
    #   - mean_ail_left > 0      => when opp on left, agent still rolls right (BIAS!)
    if mean_ail_left > 0.05 and mean_ail_right > 0.05:
        verdict = "BIAS_CONFIRMED_RIGHT (both bins produce positive aileron)"
    elif mean_ail_left < -0.05 and mean_ail_right < -0.05:
        verdict = "BIAS_CONFIRMED_LEFT (both bins produce negative aileron)"
    elif sign_match_rate < 0.55:
        verdict = f"NO_DIRECTIONAL_RESPONSE (sign-match {sign_match_rate:.2f} ≈ chance)"
    elif corr > 0.3:
        verdict = "POLICY_SYMMETRIC (turns toward opponent)"
    else:
        verdict = f"AMBIGUOUS (corr={corr:+.3f}, sign-match={sign_match_rate:.2f})"

    summary = {
        'n_steps': cli.steps,
        'n_agents': vec.total_agents,
        'n_total_samples': n,
        'n_nontrivial': n_nontrivial,
        'corr_az_ail': corr,
        'sign_match_rate': sign_match_rate,
        'mean_ail_overall': mean_ail,
        'mean_az_overall': mean_az,
        'opp_LEFT_bin': {
            'n': bin_left['n'],
            'mean_az': mean_az_left,
            'mean_aileron': mean_ail_left,
        },
        'opp_RIGHT_bin': {
            'n': bin_right['n'],
            'mean_az': mean_az_right,
            'mean_aileron': mean_ail_right,
        },
        'verdict': verdict,
    }

    print()
    print("=" * 60)
    print(" TRAINED POLICY DIRECTIONAL RESPONSE DIAGNOSTIC")
    print("=" * 60)
    print(f" total samples : {n}  (across {vec.total_agents} agents x {cli.steps} steps)")
    print(f" non-trivial   : {n_nontrivial}  (|obs[13]|>0.05)")
    print(f" mean aileron  : {mean_ail:+.3f}  (overall)")
    print(f" mean azimuth  : {mean_az:+.3f}  (overall)")
    print(f" corr(az, ail) : {corr:+.3f}    (>+0.3 = healthy, <0 = inverted)")
    print(f" sign-match    : {sign_match_rate:.3f}  (>0.55 = some directional response)")
    print()
    print(" Bin: opp on LEFT (az<-0.05)")
    print(f"    n={bin_left['n']:6d}   mean_az={mean_az_left:+.3f}   mean_aileron={mean_ail_left:+.3f}")
    print(" Bin: opp on RIGHT (az>+0.05)")
    print(f"    n={bin_right['n']:6d}   mean_az={mean_az_right:+.3f}   mean_aileron={mean_ail_right:+.3f}")
    print()
    print(f" VERDICT: {verdict}")
    print("=" * 60)

    with open(cli.summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  trace   -> {cli.trace}")
    print(f"  summary -> {cli.summary}")


if __name__ == '__main__':
    main()
