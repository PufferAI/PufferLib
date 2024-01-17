# One-off demo for pokemon red because there isn't a clean way to put
# the custom map overlay logic into the clean_pufferl file and I want
# to keep that file as minimal as possible
import numpy as np
import torch

def make_pokemon_red_overlay(bg, counts):
    nonzero = np.where(counts > 0, 1, 0)
    scaled = np.clip(counts, 0, 1000) / 1000.0

    # Convert counts to hue map
    hsv = np.zeros((*counts.shape, 3))
    hsv[..., 0] = 2*(1-scaled)/3
    hsv[..., 1] = nonzero
    hsv[..., 2] = nonzero

    # Convert the HSV image to RGB
    import matplotlib.colors as mcolors
    overlay = 255*mcolors.hsv_to_rgb(hsv)

    # Upscale to 16x16
    kernel = np.ones((16, 16, 1), dtype=np.uint8)
    overlay = np.kron(overlay, kernel).astype(np.uint8)
    mask = np.kron(nonzero, kernel[..., 0]).astype(np.uint8)
    mask = np.stack([mask, mask, mask], axis=-1).astype(bool)

    # Combine with background
    render = bg.copy().astype(np.int32)
    render[mask] = 0.2*render[mask] + 0.8*overlay[mask]
    render = np.clip(render, 0, 255).astype(np.uint8)
    return render

def rollout(env_creator, env_kwargs, agent_creator, agent_kwargs, model_path=None, device='cuda', verbose=True):
    env = env_creator(**env_kwargs)
    if model_path is None:
        agent = agent_creator(env, **agent_kwargs)
    else:
        agent = torch.load(model_path, map_location=device)

    terminal = truncated = True

    import cv2
    bg = cv2.imread('kanto_map_dsv.png')

    while True:
        if terminal or truncated:
            if verbose:
                print('---  Reset  ---')

            ob, info = env.reset()
            state = None
            step = 0
            return_val = 0

        ob = torch.tensor(ob).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(agent, 'lstm'):
                action, _, _, _, state = agent.get_action_and_value(ob, state)
            else:
                action, _, _, _ = agent.get_action_and_value(ob)

        ob, reward, terminal, truncated, _ = env.step(action[0].item())
        return_val += reward

        counts_map = env.env.counts_map
        if np.sum(counts_map) > 0 and step % 500 == 0:
            overlay = make_pokemon_red_overlay(bg, counts_map)
            cv2.imshow('Pokemon Red', overlay[1000:][::4, ::4])
            cv2.waitKey(1)

        if verbose:
            print(f'Step: {step} Reward: {reward:.4f} Return: {return_val:.2f}')

        if not env_kwargs['headless']:
            env.render()

        step += 1
