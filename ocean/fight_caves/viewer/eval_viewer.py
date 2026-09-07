#!/usr/bin/env python3
"""
eval_viewer.py — Watch a trained policy play in the full debug viewer.

Uses PufferLib's CUDA backend for policy inference, pipes actions to the
standalone viewer (fc_viewer --policy-pipe) via stdin/stdout.

Usage:
    python ocean/fight_caves/viewer/eval_viewer.py --ckpt <path_to_.bin>
    python ocean/fight_caves/viewer/eval_viewer.py --ckpt latest

Controls (in viewer window):
    Space       — pause/resume
    Side panel  — replay TPS presets
    Up/Down     — cycle replay speed presets
    Right-drag  — orbit camera
    Scroll      — zoom
    Shift+4/5   — camera presets
    O           — debug overlays
    Q/Esc       — quit
"""

import argparse
import glob
import os
import subprocess
import sys
import sysconfig

import numpy as np


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def repo_root():
    return os.path.abspath(os.path.join(script_dir(), "..", "..", ".."))


def ensure_local_pufferlib_on_path():
    default_puffer_dir = repo_root()
    puffer_dir = os.environ.get("PUFFERLIB_DIR", default_puffer_dir)
    if os.path.isdir(puffer_dir) and puffer_dir not in sys.path:
        sys.path.insert(0, puffer_dir)
    return puffer_dir


def find_compiled_backend(puffer_dir=None):
    override = os.environ.get("FC_COMPILED_BACKEND_PATH")
    if override:
        if os.path.isfile(override):
            return override
        raise RuntimeError(f"FC_COMPILED_BACKEND_PATH is not a file: {override}")

    puffer_dir = puffer_dir or ensure_local_pufferlib_on_path()
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    preferred = os.path.join(puffer_dir, "pufferlib", f"_C{extension_suffix}")
    if os.path.isfile(preferred):
        return preferred
    candidates = []
    for pattern in ("_C*.so", "_C*.dylib", "_C*.pyd"):
        candidates.extend(glob.glob(os.path.join(puffer_dir, "pufferlib", pattern)))
    if not candidates:
        raise RuntimeError(
            f"compiled Puffer backend not found under {puffer_dir}/pufferlib"
        )
    return max(candidates, key=os.path.getmtime)


def load_evaluator_preflight(backend_path):
    from eval_contract import build_verified_preflight

    source_config = os.environ.get(
        "CONFIG_PATH", os.path.join(repo_root(), "config", "fight_caves.ini")
    )
    default_config = os.path.join(repo_root(), "config", "default.ini")
    active_loadout = os.environ.get("FC_ACTIVE_LOADOUT", "FC_LOADOUT_SOTA_TBOW")
    return build_verified_preflight(
        backend_path, source_config, default_config, active_loadout
    )


def expected_parameter_bytes(contract):
    source_config = os.environ.get(
        "CONFIG_PATH", os.path.join(repo_root(), "config", "fight_caves.ini")
    )
    default_config = os.path.join(repo_root(), "config", "default.ini")
    from eval_contract import expected_checkpoint_parameter_bytes

    return expected_checkpoint_parameter_bytes(
        contract, source_config, default_config
    )


def verify_runtime_assets():
    preflight = os.path.join(
        repo_root(), "ocean", "fight_caves", "scripts", "preflight.py"
    )
    result = subprocess.run(
        [sys.executable, preflight, "--mode", "viewer-runtime"],
        cwd=repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Fight Caves runtime/viewer asset preflight failed")


def checkpoint_diagnostic(reason, checkpoint_path, expected_bytes, contract):
    actual_bytes = (
        os.path.getsize(checkpoint_path)
        if checkpoint_path and os.path.isfile(checkpoint_path)
        else "missing"
    )
    return (
        f"checkpoint rejected: {reason}\n"
        f"expected_policy_obs={contract['policy_obs_size']} "
        f"actual_policy_obs={contract['policy_obs_size']}\n"
        f"expected_puffer_obs={contract['puffer_obs_size']} "
        f"actual_puffer_obs={contract['puffer_obs_size']}\n"
        f"expected_action_dims={contract['puffer_action_dims']} "
        f"actual_action_dims={contract['puffer_action_dims']}\n"
        f"expected_parameter_bytes={expected_bytes} "
        f"actual_parameter_bytes={actual_bytes}\n"
        f"observation_version={contract['observation_version']}\n"
        f"action_version={contract['action_version']}\n"
        f"reward_version={contract['reward_version']}\n"
        f"prayer_timing_version={contract['prayer_timing_version']}\n"
        f"state_hash_version={contract['state_hash_version']}"
    )


def latest_source_mtime():
    repo = repo_root()
    patterns = [
        os.path.join(repo, "ocean", "fight_caves", "include", "*.h"),
        os.path.join(repo, "ocean", "fight_caves", "src", "*.c"),
        os.path.join(repo, "ocean", "fight_caves", "viewer", "src", "*.h"),
        os.path.join(repo, "ocean", "fight_caves", "viewer", "src", "*.c"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return max((os.path.getmtime(path) for path in files), default=0.0)


def find_viewer():
    """Find the fc_viewer binary."""
    override = os.environ.get("FC_VIEWER_PATH")
    if override:
        if os.path.isfile(override):
            return override
        raise RuntimeError(f"FC_VIEWER_PATH does not point to a file: {override}")

    repo = repo_root()
    source_mtime = latest_source_mtime()
    preferred = [
        os.path.join(repo, "build", "fight_caves-viewer", "fc_viewer"),
    ]
    candidates = [path for path in preferred if os.path.isfile(path)]

    patterns = [
        os.path.join(repo, "build*", "fight_caves-viewer", "fc_viewer"),
    ]
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    candidates = [path for path in candidates if os.path.isfile(path)]
    if not candidates:
        return None

    seen = set()
    unique = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    for path in unique:
        if os.path.getmtime(path) >= source_mtime:
            return path
    return max(unique, key=os.path.getmtime)


def read_obs_line(proc, total_line_floats):
    """Read one line of space-separated floats from viewer stdout."""
    line = proc.stdout.readline()
    if not line:
        return None
    values = line.strip().split()
    if len(values) != total_line_floats:
        print(f"[eval] Warning: expected {total_line_floats} floats, got {len(values)}",
              file=sys.stderr)
        return None
    return np.array([float(v) for v in values], dtype=np.float32)


def send_actions(proc, actions):
    """Write one action per Puffer action head to viewer stdin."""
    line = " ".join(str(int(a)) for a in actions) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()


def sample_masked(logits_list, mask, act_dims, deterministic=False):
    """Sample actions from logits with mask applied."""
    actions = []
    mask_offset = 0
    for head_idx, (logits, dim) in enumerate(zip(logits_list, act_dims)):
        head_mask = mask[mask_offset:mask_offset + dim]
        mask_offset += dim

        # Apply mask: set invalid actions to -inf
        masked_logits = logits.copy()
        for i in range(dim):
            if head_mask[i] < 0.5:
                masked_logits[i] = -1e9

        if deterministic:
            action = np.argmax(masked_logits)
        else:
            # Softmax + sample
            logits_shifted = masked_logits - np.max(masked_logits)
            probs = np.exp(logits_shifted)
            probs = probs / (probs.sum() + 1e-8)
            action = np.random.choice(dim, p=probs)

        actions.append(action)
    return actions


def load_policy_weights(policy, checkpoint_path, checkpoint_kind, parameter_bytes):
    import torch

    if checkpoint_kind == "pytorch":
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError("PyTorch checkpoint does not contain a state dictionary")
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        expected = policy.state_dict()
        if set(state_dict) != set(expected):
            missing = sorted(set(expected) - set(state_dict))
            extra = sorted(set(state_dict) - set(expected))
            raise RuntimeError(
                "PyTorch checkpoint keys do not match the configured policy: "
                f"missing={missing[:1]}, extra={extra[:1]}"
            )
        for key, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"PyTorch checkpoint value is not a tensor: {key}")
            if tensor.shape != expected[key].shape:
                raise RuntimeError(
                    f"PyTorch checkpoint shape mismatch for {key}: "
                    f"expected={list(expected[key].shape)}, actual={list(tensor.shape)}"
                )
        policy.load_state_dict(state_dict, strict=True)
        print(
            f"[eval] Loaded PyTorch state dictionary ({len(state_dict)} tensors)",
            file=sys.stderr,
        )
        return

    if checkpoint_kind != "raw":
        raise RuntimeError(f"unsupported checkpoint format: {checkpoint_kind!r}")

    # The CUDA trainer saves a flat float32 buffer in this order:
    # encoder.weight, fused decoder/action+value weight, and recurrent layers.
    weights = np.fromfile(checkpoint_path, dtype=np.float32)
    print(f"[eval] Checkpoint: {len(weights)} floats", file=sys.stderr)
    state_dict = policy.state_dict()
    for key in state_dict:
        if "bias" in key:
            state_dict[key] = torch.zeros_like(state_dict[key])

    offset = 0

    def load_tensor(key):
        nonlocal offset
        if key not in state_dict:
            raise KeyError(f"{key} not in model state_dict")
        numel = state_dict[key].numel()
        if offset + numel > len(weights):
            raise RuntimeError(f"weights exhausted at {key}")
        state_dict[key] = torch.from_numpy(
            weights[offset:offset + numel].reshape(state_dict[key].shape).copy()
        )
        offset += numel
        print(
            f"  loaded {key}: {list(state_dict[key].shape)} ({numel})",
            file=sys.stderr,
        )

    load_tensor("encoder.encoder.weight")
    decoder_key = "decoder.decoder.weight"
    value_key = "decoder.value_function.weight"
    if decoder_key not in state_dict or value_key not in state_dict:
        raise KeyError("decoder weights missing from model state_dict")

    decoder_rows = state_dict[decoder_key].shape[0]
    hidden_size = state_dict[decoder_key].shape[1]
    value_rows = state_dict[value_key].shape[0]
    fused_rows = decoder_rows + value_rows
    fused_numel = fused_rows * hidden_size
    if offset + fused_numel > len(weights):
        raise RuntimeError("weights exhausted at fused decoder")
    fused_decoder = weights[offset:offset + fused_numel].reshape(
        fused_rows, hidden_size
    ).copy()
    state_dict[decoder_key] = torch.from_numpy(fused_decoder[:decoder_rows])
    state_dict[value_key] = torch.from_numpy(fused_decoder[decoder_rows:])
    offset += fused_numel
    print(
        f"  loaded fused decoder: {list(fused_decoder.shape)} "
        f"-> {list(state_dict[decoder_key].shape)} + "
        f"{list(state_dict[value_key].shape)}",
        file=sys.stderr,
    )

    network_keys = sorted(
        [
            key for key in state_dict
            if key.startswith("network.layers.") and key.endswith(".weight")
        ],
        key=lambda key: int(key.split(".")[2]),
    )
    model_parameter_floats = (
        state_dict["encoder.encoder.weight"].numel()
        + state_dict[decoder_key].numel()
        + state_dict[value_key].numel()
        + sum(state_dict[key].numel() for key in network_keys)
    )
    model_parameter_bytes = model_parameter_floats * np.dtype(np.float32).itemsize
    if model_parameter_bytes != parameter_bytes:
        raise RuntimeError(
            "constructed model/raw layout mismatch: "
            f"expected_parameter_bytes={parameter_bytes}, "
            f"actual_parameter_bytes={model_parameter_bytes}"
        )
    for key in network_keys:
        load_tensor(key)

    policy.load_state_dict(state_dict)
    if offset != len(weights):
        raise RuntimeError(
            f"unused supplied weights: loaded={offset}, actual={len(weights)}"
        )
    print(f"[eval] Loaded {offset}/{len(weights)} raw weights", file=sys.stderr)


def main():
    # Parse our args FIRST, then clear sys.argv so PufferLib's
    # load_config() doesn't choke on our flags.
    parser = argparse.ArgumentParser(description="Watch trained policy in debug viewer")
    parser.add_argument("--ckpt", type=str, default="latest",
                        help="Path to .bin checkpoint or 'latest'")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use argmax instead of sampling")
    parser.add_argument("--random", action="store_true",
                        help="Use random valid actions (no checkpoint needed)")
    parser.add_argument("--start-wave", type=int, default=0,
                        help="Start at this wave (0 = wave 1)")
    parser.add_argument("--speed", type=int, choices=[1, 2, 4, 10], default=1,
                        help="Initial replay speed multiplier (buttons can switch to TPS presets)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Stop after this many replay episodes (0 = unlimited)")
    parser.add_argument("--max-ticks", type=int, default=0,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Clear sys.argv so PufferLib doesn't see our flags
    sys.argv = [sys.argv[0]]
    puffer_dir = ensure_local_pufferlib_on_path()

    try:
        verify_runtime_assets()
        backend_path = find_compiled_backend(puffer_dir)
        verified_preflight = load_evaluator_preflight(backend_path)
    except Exception as exc:
        print(f"Error: evaluator compiled-contract preflight failed: {exc}", file=sys.stderr)
        return 1
    contract = verified_preflight["contract"]
    policy_obs_size = contract["policy_obs_size"]
    act_dims = contract["puffer_action_dims"]
    mask_size = contract["puffer_mask_size"]
    total_line_floats = contract["puffer_obs_size"]

    # Find viewer binary
    viewer_path = find_viewer()
    if not viewer_path:
        print(
            "Error: fc_viewer binary not found. Build with: "
            "./build.sh fight_caves --viewer",
            file=sys.stderr,
        )
        sys.exit(1)
    if os.path.getmtime(viewer_path) < latest_source_mtime():
        print(
            f"Error: selected fc_viewer is older than current core/viewer sources: {viewer_path}",
            file=sys.stderr,
        )
        print(
            "Rebuild it first with: ./build.sh fight_caves --viewer",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[eval] Viewer: {viewer_path}", file=sys.stderr)
    print(f"[eval] Replay speed: {args.speed}x", file=sys.stderr)
    if args.episodes > 0:
        print(f"[eval] Episode limit: {args.episodes}", file=sys.stderr)
    print(
        f"[eval] Contract: policy_obs={policy_obs_size} mask={mask_size} "
        f"heads={len(act_dims)} total={total_line_floats}",
        file=sys.stderr,
    )

    # Load checkpoint (unless --random)
    policy = None
    if not args.random:
        try:
            parameter_bytes = expected_parameter_bytes(contract)
        except Exception as exc:
            print(f"Error: cannot derive expected checkpoint size: {exc}", file=sys.stderr)
            return 1

        from eval_contract import ContractError, resolve_checkpoint

        request_mode = "latest" if args.ckpt == "latest" else "explicit"
        checkpoint_root = os.environ.get(
            "FC_CHECKPOINT_ROOT",
            os.path.join(repo_root(), "checkpoints"),
        )
        try:
            resolution = resolve_checkpoint(
                request_mode,
                checkpoint_root,
                verified_preflight,
                parameter_bytes,
                checkpoint_path=None if request_mode == "latest" else args.ckpt,
            )
        except ContractError as exc:
            print(
                checkpoint_diagnostic(
                    exc, None if args.ckpt == "latest" else args.ckpt,
                    parameter_bytes, contract,
                ),
                file=sys.stderr,
            )
            return 1

        checkpoint_path = resolution["resolved_path"]
        checkpoint_kind = resolution["format"]
        print(
            f"[eval] Checkpoint: {checkpoint_path} ({checkpoint_kind})",
            file=sys.stderr,
        )

        try:
            import torch
            import pufferlib.models
            from pufferlib.pufferl import load_config

            eval_args = load_config("fight_caves")
            policy_kwargs = eval_args["policy"]
            network_cls = getattr(pufferlib.models, eval_args["torch"]["network"])
            encoder_cls = getattr(pufferlib.models, eval_args["torch"]["encoder"])
            decoder_cls = getattr(pufferlib.models, eval_args["torch"]["decoder"])

            network = network_cls(**policy_kwargs)
            encoder = encoder_cls(total_line_floats, policy_kwargs["hidden_size"])
            decoder = decoder_cls(act_dims, policy_kwargs["hidden_size"])
            policy = pufferlib.models.Policy(encoder, decoder, network)
            policy = policy.cpu()
            load_policy_weights(
                policy, checkpoint_path, checkpoint_kind, parameter_bytes
            )

            policy = policy.cpu()
            policy.eval()
            print("[eval] Policy ready (CPU)", file=sys.stderr)

        except Exception as exc:
            print(
                checkpoint_diagnostic(
                    exc, checkpoint_path, parameter_bytes, contract
                ),
                file=sys.stderr,
            )
            return 1

    # Launch viewer subprocess from repo root so sprite paths resolve
    print("[eval] Launching viewer...", file=sys.stderr)
    viewer_env = os.environ.copy()
    viewer_env.setdefault(
        "FC_ASSET_ROOT",
        os.path.join(repo_root(), "resources", "fight_caves", "viewer"),
    )
    viewer_env.setdefault("FC_REPO_ROOT", repo_root())
    proc = subprocess.Popen(
        [viewer_path, "--policy-pipe", "--speed", str(args.speed)] +
            (["--episodes", str(args.episodes)] if args.episodes > 0 else []) +
            (["--start-wave", str(args.start_wave)] if args.start_wave > 0 else []),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # let viewer stderr pass through to terminal
        text=True,
        bufsize=1,
        cwd=repo_root(),
        env=viewer_env,
    )

    # Hidden state for recurrent policy (MinGRU)
    hidden = None
    if policy is not None:
        import torch
        hidden = policy.initial_state(1, 'cpu')

    try:
        tick = 0
        while True:
            # Read observation from viewer
            obs_data = read_obs_line(proc, total_line_floats)
            if obs_data is None:
                print("[eval] Viewer closed or read error", file=sys.stderr)
                break

            obs = obs_data[:policy_obs_size]
            mask = obs_data[policy_obs_size:]

            if args.random or policy is None:
                # Random valid actions
                actions = sample_masked(
                    [np.zeros(d) for d in act_dims], mask, act_dims, deterministic=False)
            else:
                # Policy inference: feed the same Puffer observation used in training.
                import torch
                with torch.no_grad():
                    full_input = torch.from_numpy(obs_data).unsqueeze(0)
                    output = policy.forward_eval(full_input, hidden)
                    # forward_eval returns (logits, values, state)
                    logits_raw, _values, hidden = output

                    # Extract per-head logits
                    if isinstance(logits_raw, (list, tuple)):
                        logits_list = [l.squeeze(0).numpy() for l in logits_raw]
                    else:
                        # Single tensor — split by action dims
                        lr = logits_raw.squeeze(0).numpy()
                        logits_list = []
                        off = 0
                        for d in act_dims:
                            logits_list.append(lr[off:off+d])
                            off += d

                actions = sample_masked(logits_list, mask, act_dims, args.deterministic)

            # Send actions to viewer
            send_actions(proc, actions)
            tick += 1

            if tick % 100 == 0:
                print(f"[eval] Tick {tick}", file=sys.stderr)
            if args.max_ticks > 0 and tick >= args.max_ticks:
                print(f"[eval] Smoke limit reached at tick {tick}", file=sys.stderr)
                break

    except (BrokenPipeError, KeyboardInterrupt):
        print("[eval] Stopped", file=sys.stderr)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
