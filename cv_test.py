import cv2
import numpy as np
# import pathlib as Path
from checkpoint_file_aggregator import read_checkpoint_logs
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib.gridspec import GridSpec
# import io
# from PIL import Image

# from memory_profiler import profile
# import tracemalloc

# tracemalloc.start()
# snapshot1=tracemalloc.take_snapshot()
bg = cv2.imread('kanto_map_dsv.png') # 142 MiB

# @profile

# Code previously being used to write the aggregated checkpoints data to image using cv2.imwrite
def create_data_image(width, height):
    data_image = np.zeros((height, width, 3), dtype=np.uint8)
    return data_image

def get_text_width(text, fontdict):
    # A simple approximation: assume each character is equal width
    # This is a simplification and may not be accurate for all fonts
    # You can replace this with a more sophisticated method if needed
    return len(text) * fontdict['fontsize'] * 1.5  # Adjust the multiplier as needed

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
    render = bg.copy().astype(np.int32)
    render[mask] = 0.2*render[mask] + 0.8*overlay[mask]
    render = np.clip(render, 0, 255).astype(np.uint8)
    return render
# snapshot3=tracemalloc.take_snapshot()

import random
# snapshot5=tracemalloc.take_snapshot()
# @profile
counts_map = np.zeros((444, 436))
# Iterate over the range for x and y separately
for x in range(1, 164):  # Range for x
    for y in range(318, 418):  # Range for y
        # Increment counts_map at position (x, y) by a random number between 10 and 1000
        counts_map[x, y] += random.randint(10, 1000)
        
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

# Run the function and save the image
print_table_image = matplotlib_table_map_generate(counts_map)
cv2.imwrite('image_test_data_image_cv_test_NEW.png', print_table_image)
cv2.waitKey(1)
#         # Attempt to create the figure with the data image and table
#         plt.style.use("dark_background")
#         fig, (table_ax, img_ax) = plt.subplots(
#             1, 2, figsize=(32, 22), width_ratios=[1.1, 3.25], subplot_kw={"anchor": "N"}
#         )

#         # Add the text above the table
#         if epoch_sps:
#             fig.text(0.05, 0.95, f'Epoch SPS: {epoch_sps}', color='white', fontsize=12, ha='left', va='top')

#         # First subplot: Table
#         table_ax.axis("off")

#         # Calculate the offset in figure coordinate space so table isn't edge-magnetic
#         pixel_offset = -50  # The desired pixel offset, positive value to move right
#         dpi = fig.get_dpi()  # Get the DPI of the figure to convert pixels to inches
#         inch_offset = pixel_offset / dpi  # Convert pixel offset to inches
#         # Get the current bounds of the table subplot
#         pos = table_ax.get_position()
#         # Adjust the left bound by the inch_offset
#         new_pos = [pos.x0 + inch_offset / fig.get_figwidth(), pos.y0, pos.width, pos.height]
#         # Set the new position of the table subplot
#         table_ax.set_position(new_pos)
        
#         fontdict = {'fontsize': 20}  # Customize as needed
        
#         # F
#         # font_dict = {'fontsize': 20}
#         text_width = get_text_width('text_here', fontdict)
#         text_width = int(text_width)
        
#         widths = []
#         for col in df.columns:
#             # Get the width of each text entry in the column, including the header
#             text_widths = [get_text_width(str(x), fontdict) for x in df[col].tolist() + [col]]
#             # Find the maximum width for the column
#             max_width = max(text_widths)
#             widths.append(max_width)

#         # Normalize the widths to sum to 1 to get relative column widths
#         total_width = sum(widths)
#         rel_widths = [w / total_width for w in widths]
        
#         # widths = []
#         # for col in df.columns:
#         #     max_width = max([get_text_width(str(x), fontdict) for x in df[col].tolist() + [col]])
#         #     widths.append(max_width)
#         #     widths.append(text_width)
            
#         print(f'text_width: {text_width}')
#         # Normalize the widths to the sum to get relative column widths
#         print(f'widths: {widths}')
#         # total_width = sum(widths)
#         # rel_widths = [w / total_width for w in widths]
#         print(f'rel_widths: {rel_widths}')

                
#         # Calculate relative column widths to size columns to largest cell contents
#         # widths = []
#         # t_widths = []
#         # total_width = 0
#         # t_total_width = 0
#         # for col in df.columns:
#         #     text_width = [max(get_text_width(len(str(x)), font_dict)) for x in df[col]]
#         #     max_width = max([len(str(x)) for x in df[col]])
            
#         #     widths.append(max_width)
#         #     t_widths.append(text_width)
            
#         #     print(f'widths: {widths}')
#         #     print(f't_widths: {t_widths}')
            
#         #     total_width += (max_width)
#         #     t_total_width += int((t_widths))
            
#         # rel_widths = [w/total_width for w in widths]
#         # t_rel_widths = [w/t_total_width for w in t_widths]
        
#         # print(f'rel_widths: {rel_widths}')
#         # print(f't_rel_widths: {t_rel_widths}')
        
        # rel_widths[0] = rel_widths[0] * 0.74
        # t_rel_widths[0] = rel_widths[0] * 1
        
#         # print(f'rel_widths={rel_widths}')
#         # print(f't_rel_widths={t_rel_widths}')
        
#         rel_widths: [0.36106340937668403, 0.2083707562421412, 0.1905873899766481, 0.1904077600143704]
#         # Create the table with relative column widths
#         the_table = table_ax.table(cellText=df.values, colLabels=df.columns, loc="upper center", colWidths=rel_widths)

#         cell_height = 0.025 # Convert font size in points to inches
        
#         # Set table style for white text on black background
#         the_table.auto_set_font_size(False)
#         the_table.set_fontsize(30)
#         for key, cell in the_table.get_celld().items():
#             cell.get_text().set_color('white')
#             cell.set_facecolor('black')
#             cell.set_edgecolor('white')

#         for pos, cell in the_table.get_celld().items():
#             cell.set_height(cell_height)

#         # Adjust the scale of the table to fit the content
#         the_table.scale(1.35, 1.35)

#         # Second subplot: Image
#         img = plt.imread("kanto_map_dsv.png")
#         img_ax.imshow(img)
#         img_ax.axis("off")

#         fig.tight_layout()

#         # Convert the figure to a NumPy array using buffer_rgba
#         fig.canvas.draw()
#         width, height = fig.get_size_inches() * fig.get_dpi()
#         table_image_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#         table_image_rgba = table_image_rgba.reshape(int(height), int(width), -1)

#         i += 1
#         return table_image_rgba

#     plt.close('all')
#     cv2.destroyAllWindows()
#     table_image_rgba = None

# print_table_image = matplotlib_table_map_generate()
# cv2.imwrite(f'image_test_data_image_cv_test_NEW.png', print_table_image)
# cv2.waitKey(1)


    # fig.savefig("image_thatguy.png")


    # import pokemon_red_eval as pre
    # # bg = cv2.imread('kanto_map_dsv.png')
    # counts_map = np.zeros((444, 436))
    # overlay = make_pokemon_red_overlay(counts_map)

    # # BET 
    # di_height = 6976 * 1.5

    # # Make the underlay that contains the dashboard to go to left of map
    # # data_image = np.zeros((int(di_height), 7104, 3), dtype=np.uint8)
    # data_image = create_data_image(int(di_height), 7104)

    # # Define the position where the data image will be rendered
    # x_position = 6976
    # y_position = 7104

    # # Calculate the starting position for rendering the data image
    # start_x = x_position - data_image.shape[1]
    # start_y = y_position - data_image.shape[0]

    # # Ensure the data image fits within the bounds of the background image
    # end_x = min(start_x + data_image.shape[1], overlay.shape[1])
    # end_y = min(start_y + data_image.shape[0], overlay.shape[0])

    # # Calculate the region of interest on the background image
    # roi_start_x = max(start_x, 0)
    # roi_end_x = min(end_x, overlay.shape[1])
    # roi_start_y = max(start_y, 0)
    # roi_end_y = min(end_y, overlay.shape[0])

    # # Calculate the corresponding region on the data image
    # data_roi_start_x = roi_start_x - start_x
    # data_roi_end_x = data_roi_start_x + (roi_end_x - roi_start_x)
    # data_roi_start_y = roi_start_y - start_y
    # data_roi_end_y = data_roi_start_y + (roi_end_y - roi_start_y)

    # # Overlay the background image onto the dashboard underlay
    # # bg[roi_start_y:roi_end_y, roi_start_x:roi_end_x] += data_image[data_roi_start_y:data_roi_end_y, data_roi_start_x:data_roi_end_x]
    # data_image[data_roi_start_y:data_roi_end_y, data_roi_start_x:data_roi_end_x] += overlay[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

    # # Place matplotlib table onto the combined image
    # data_image = matplotlib_table_map_generate(data_image)
    # cv2.imwrite(f'image_test_data_image_cv_test_NEW.png',data_image)
    # cv2.waitKey(1)
    # try:
    #     data_image = matplotlib_table_map_generate(data_image) 
    # except:
    #     pass
    # return data_image