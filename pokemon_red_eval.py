# One-off demo for pokemon red because there isn't a clean way to put
# the custom map overlay logic into the clean_pufferl file and I want
# to keep that file as minimal as possible
import torch
import cv2
import numpy as np
# import pathlib as Path
from checkpoint_file_aggregator import read_checkpoint_logs
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

BG = cv2.imread('kanto_map_dsv.png')

# import io
# from PIL import Image

# from memory_profiler import profile
# import tracemalloc

# tracemalloc.start()
# snapshot1=tracemalloc.take_snapshot()
# bg = cv2.imread('kanto_map_dsv.png') # 142 MiB

# @profile

# Code previously being used to write the aggregated checkpoints data to image using cv2.imwrite
def create_data_image(width, height):
    data_image = np.zeros((height, width, 3), dtype=np.uint8)
    return data_image

def get_text_width(text, fontdict):
    return len(text) * fontdict['fontsize'] * 1.5  # Adjust the multiplier as needed (probably don't)

# snapshot2=tracemalloc.take_snapshot()
# @profile
def make_pokemon_red_overlay(counts):
    nonzero = np.where(counts > 0, 1, 0)
    scaled = np.clip(counts, 0, 1000) / 1000.0

    # Convert counts to hue map
    hsv = np.zeros((*counts.shape, 3))
    hsv[..., 0] = (240.0 / 360) - scaled * (240.0 / 360.0) # bad heatmap with too much icky light green 2*(1-scaled)/3
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
    render = BG.copy().astype(np.int32)
    render[mask] = 0.2*render[mask] + 0.8*overlay[mask]
    render = np.clip(render, 0, 255).astype(np.uint8)
    return render
# snapshot3=tracemalloc.take_snapshot()

# snapshot5=tracemalloc.take_snapshot()
# @profile

def matplotlib_table_map_generate(counts):
    i = 0
    while i < 1:
        # Read the checkpoints data
        try:
            time_checkpoint, stats_checkpoint = read_checkpoint_logs()
        except Exception as e:
            print(f"Failed to read checkpoint logs: {e}")
            time_checkpoint, stats_checkpoint = {}, {}

        # Read the epoch sps data
        try:
            with open("experiments/run_stats.txt", "r") as file:
                epoch_sps = file.readline().strip()
        except Exception as e:
            print(f"Failed to read epoch sps data: {e}")
            epoch_sps = "Unavailable"

        # Assuming there's data in the checkpoints, proceed to create the DataFrame
        if time_checkpoint and stats_checkpoint:
            # Extract data for table
            milestones = list(time_checkpoint.keys())
            times = [time_checkpoint[milestone] for milestone in milestones]
            means = [stats_checkpoint[milestone]['mean'] for milestone in milestones]
            # variances = [stats_checkpoint[milestone]['variance'] for milestone in milestones]
            std_devs = [stats_checkpoint[milestone]['std_dev'] for milestone in milestones]

            data = {
                'Milestone': milestones,
                'Time (min)': times,
                'Mean': means,
                # 'Variance': variances,
                'Std Dev': std_devs
            }
            df = pd.DataFrame(data)
        else:
            print("Checkpoint data is empty. Creating an empty DataFrame.")
            df = pd.DataFrame()

        plt.style.use("dark_background")
        fig, (table_ax, img_ax) = plt.subplots(
            1, 2, figsize=(32, 22), gridspec_kw={'width_ratios': [1, 2]}
        )
        
        # Print the Epoch SPS at the top left of the whole image
        fig.text(0.005, 0.995, f'Epoch SPS: {epoch_sps}', color='0.35', fontsize=40, ha='left', va='top')

        table_ax.axis("off")
        font_size = 30
        
        # fontdict = {'fontsize': 30}
        get_font_dict = lambda x: {'fontsize': x}
        
        # Calculate relative column widths
        widths = []
        # widths_1 = []
        for col in df.columns:
            # max_width_1 = max([get_text_width_1(str(x), get_font_dict(font_size)) for x in df[col].tolist() + [col]])
            max_width = max([get_text_width(str(x), get_font_dict(font_size)) for x in df[col].tolist() + [col]])
            # print(f'max_width={max_width}')
            # print(f'max_width_1={max_width_1}')
            widths.append(max_width)
            # widths_1.append(max_width_1)

        total_width = sum(widths)
        # total_width_1 = sum(widths_1)
        rel_widths = [w / total_width for w in widths]
        # rel_widths_1 = [w / total_width_1 for w in widths_1]
        
        rel_widths[0] = rel_widths[0] * 1.1
        rel_widths[2] = rel_widths[2] * 1.1
        # rel_widths_1[0] = rel_widths_1[0] * 1.1
        
        # print(f'rel_widths = {rel_widths}')
        # print(f'rel_widths_1 = {rel_widths_1}')

        cell_height = 0.035 # Convert font size in points to inches
        
        # Create the table with relative column widths
        the_table = table_ax.table(cellText=df.values, colLabels=df.columns, loc='upper center', colWidths=rel_widths)
       
        # Set table style
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(font_size)

        # Define the colors for headings and different columns
        heading_color = '#ff7f0e'  # Deep blue for the headings 
        column_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b']  # Orange, Green, Purple, Brown

        edge_color = '0.75'

        # Iterate over the cells and set colors
        for (row, col), cell in the_table.get_celld().items():
            if row == 0:  # This is a heading
                cell.get_text().set_color(heading_color)
                cell.set_facecolor('black')  # Heading background color
                cell.set_edgecolor('white')
            else:  # These are data cells
                cell_color = column_colors[col] if col < len(column_colors) else 'black'  # Default to black if no color is defined
                cell.get_text().set_color(cell_color)
                cell.set_facecolor('black')  # Data cell background color
                cell.set_edgecolor(f'{edge_color}')

            cell.set_height(cell_height)

        # Image subplot
        # img = plt.imread("kanto_map_dsv.png")
        img = make_pokemon_red_overlay(counts)
        img_ax.imshow(img)
        img_ax.axis("off")

        fig.tight_layout()

        # Save the figure to a NumPy array
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        table_image_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        table_image_rgba = table_image_rgba.reshape(int(height), int(width), -1)

        plt.close('all')
        cv2.destroyAllWindows()
        return table_image_rgba

def rollout(env_creator, env_kwargs, agent_creator, agent_kwargs, model_path=None, device='cuda', verbose=True):
    env = env_creator(**env_kwargs)
    if model_path is None:
        agent = agent_creator(env, **agent_kwargs)
    else:
        agent = torch.load(model_path, map_location=device)

    terminal = truncated = True

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
            # overlay = make_pokemon_red_overlay(sum(counts_map))
            data_image = matplotlib_table_map_generate(sum(counts_map))
            cv2.imshow('Pokemon Red', data_image[1000:][::4, ::4])
            cv2.waitKey(100)
            cv2.destroyAllWindows()

        if verbose:
            print(f'Step: {step} Reward: {reward:.4f} Return: {return_val:.2f}')

        if not env_kwargs['headless']:
            env.render()

        step += 1
