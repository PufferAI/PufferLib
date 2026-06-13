import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import pufferlib.pufferl as pufferl
from pufferlib import _C, torch_pufferl


def configure_args(args, hidden_size, max_steps):
    args["slowly"] = True
    args["reset_state"] = False
    args["vec"]["total_agents"] = 1
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 1
    args["train"]["horizon"] = 1
    args["train"]["minibatch_size"] = 1
    args["train"]["total_timesteps"] = 1
    args["policy"]["hidden_size"] = hidden_size
    if max_steps is not None:
        args["env"]["max_steps"] = max_steps
    return args


def native_bin_to_torch_state(raw_path, policy):
    raw = np.fromfile(raw_path, dtype=np.float32)
    state = policy.state_dict()

    hidden = state["encoder.encoder.weight"].shape[0]
    obs = state["encoder.encoder.weight"].shape[1]
    actions = state["decoder.decoder.weight"].shape[0]
    layers = len([k for k in state if k.startswith("network.layers.")])

    expected = hidden * obs + (actions + 1) * hidden + layers * (3 * hidden * hidden)
    if raw.size != expected:
        raise RuntimeError(f"raw checkpoint has {raw.size} floats, expected {expected}")

    idx = 0
    new_state = {}

    n = hidden * obs
    new_state["encoder.encoder.weight"] = torch.from_numpy(
        raw[idx:idx + n].reshape(hidden, obs).copy())
    idx += n
    new_state["encoder.encoder.bias"] = torch.zeros_like(state["encoder.encoder.bias"])

    n = (actions + 1) * hidden
    fused_decoder = raw[idx:idx + n].reshape(actions + 1, hidden)
    idx += n
    new_state["decoder.decoder.weight"] = torch.from_numpy(fused_decoder[:actions].copy())
    new_state["decoder.decoder.bias"] = torch.zeros_like(state["decoder.decoder.bias"])
    new_state["decoder.value_function.weight"] = torch.from_numpy(
        fused_decoder[actions:actions + 1].copy())
    new_state["decoder.value_function.bias"] = torch.zeros_like(
        state["decoder.value_function.bias"])

    for layer in range(layers):
        key = f"network.layers.{layer}.weight"
        n = state[key].numel()
        new_state[key] = torch.from_numpy(raw[idx:idx + n].reshape(state[key].shape).copy())
        idx += n

    policy.load_state_dict(new_state)


def masked_action(logits, mask, sampled):
    masked_logits = logits.clone()
    masked_logits[mask == 0] = -1e9
    if sampled:
        action, _, _ = torch_pufferl.sample_logits(masked_logits)
        return action.to(dtype=torch.float32).contiguous()
    return torch.argmax(masked_logits, dim=-1, keepdim=True).to(dtype=torch.float32).contiguous()


def decode_action(action):
    action = int(action)
    slot = action // 64
    cell = action % 64
    row = cell // 8
    col = cell % 8
    return slot, row, col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="downloads/block_blast_latest/block_blast_1781365132432_0000000499122176.bin",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--delay", type=float, default=0.16)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--print-actions", action="store_true")
    args_cli = parser.parse_args()

    checkpoint = Path(args_cli.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        args = pufferl.load_config("block_blast")
    finally:
        sys.argv = argv

    args = configure_args(args, args_cli.hidden_size, args_cli.max_steps)
    vec = _C.create_vec(args, 0)
    vec.reset()
    vec.log()

    if vec.action_mask_size <= 0:
        raise RuntimeError("block_blast action mask is not exposed by this _C build")

    policy = torch_pufferl.load_policy(args, vec)
    native_bin_to_torch_state(checkpoint, policy)
    policy.eval()
    state = policy.initial_state(1, "cpu")

    mode = "sampled" if args_cli.sampled else "greedy"
    print(f"Rendering {checkpoint} ({mode})")
    if args_cli.no_render:
        print("Render disabled for smoke test.")
    else:
        print("Close the Raylib window or press Esc to stop.")

    step = 0
    try:
        if not args_cli.no_render:
            vec.render(0)
        while True:
            obs = torch_pufferl._cpu_tensor(vec.obs_ptr, (1, vec.obs_size), torch.uint8)
            mask = torch_pufferl._cpu_tensor(
                vec.action_mask_ptr, (1, vec.action_mask_size), torch.uint8)
            with torch.no_grad():
                logits, _, state = policy.forward_eval(obs, state)
                action = masked_action(logits, mask, args_cli.sampled)

            vec.cpu_step(action.data_ptr())
            step += 1

            rewards = torch_pufferl._cpu_tensor(vec.rewards_ptr, (1,), torch.float32)
            terminals = torch_pufferl._cpu_tensor(vec.terminals_ptr, (1,), torch.float32)
            if args_cli.print_actions:
                slot, row, col = decode_action(action.item())
                print(
                    f"step={step:04d} slot={slot} row={row} col={col} "
                    f"reward={rewards[0].item():.3f} terminal={int(terminals[0].item())}"
                )
            if not args_cli.no_render:
                vec.render(0)
                time.sleep(args_cli.delay)
            if terminals[0].item() > 0.0:
                state = policy.initial_state(1, "cpu")
            if args_cli.steps > 0 and step >= args_cli.steps:
                break
    finally:
        vec.close()


if __name__ == "__main__":
    main()
