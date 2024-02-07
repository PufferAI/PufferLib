import os
from collections import defaultdict
from pokegym import game_map
import pdb as T
import numpy as np

# from memory_profiler import profile

LOGGABLE_LOCATIONS = {"Viridian City": (1, None),
    "Viridian Forest Entrance": (51, None),
    "Viridian Forest Exit": (47, None),
    "Pewter City": (2, None),
    "Badge 1": (None, None),
    "Route 3": (14, None),
    "Mt Moon Entrance (Route 3)": (59, None),
    "Mt Moon B1F": (60, None),
    "Mt Moon B2F": (61, None),
    "Mt Moon Exit": (59, None),
    "Route 4": (15, None),
    "Cerulean City": (3, None),
    "Badge 2": (None, None),
    "Bill": (88, None),
    "Vermilion City": (5, None),
    "Vermilion Harbor": (94, None),
    "SS Anne Start": (95, None),
    "SS Anne Captains Office": (101, None)}

# @profile
def read_checkpoint_logs():
    checkpoint_data = defaultdict(str)
    stats_data = defaultdict(lambda: {'mean': 0, 'variance': 0, 'std_dev': 0})

    base_dir = "experiments"
    file_path = "experiments/running_experiment.txt"

    with open(file_path, "r") as pathfile:
        exp_uuid8 = pathfile.readline().strip()

    sessions_path = os.path.join(base_dir, exp_uuid8, "sessions")
    all_location_times = defaultdict(list)
    current_location_times = defaultdict(lambda: float('inf'))

    try:
        # Iterate over each log file and append times to all_location_times
        for folder in os.listdir(sessions_path):
            session_path = os.path.join(sessions_path, folder)
            checkpoint_log_file = None
            for file in os.listdir(session_path):
                if file.startswith("checkpoint_log") and file.endswith(".txt"):
                    checkpoint_log_file = os.path.join(session_path, file)
                    break

            if checkpoint_log_file is not None:
                with open(checkpoint_log_file, 'r') as log_file:
                    # lines = log_file.readlines()

                    for line in log_file:
                        line = line.strip()
                        if line.startswith("Location ID:"):
                            current_location_id = line.split(":")[-1].strip()
                            if current_location_id.isdigit():
                                current_location_id = int(current_location_id)
                                if current_location_id in [map_id for location, (map_id, _) in LOGGABLE_LOCATIONS.items() if map_id is not None]:
                                    current_location = current_location_id
                                else:
                                    current_location = 40
                            else:
                                current_location = 40
                        elif line.startswith("Time Visited:") and current_location is not None:
                            time_visited = float(line.split(":")[-1].strip())
                            current_location_times[current_location] = min(current_location_times[current_location], time_visited)
                            all_location_times[current_location].append(time_visited)
                            # print(f'all loc: {all_location_times}')

        # Update checkpoint_data with minimum times
        for location_id, time_visited in current_location_times.items():
            location_name = game_map.get_map_name_from_map_n(location_id)
            formatted_time = '{:.2f}'.format(time_visited / 60)
            checkpoint_data[location_name] = formatted_time

        # Calculate mean, variance, and standard deviation for each location
        for location_id, times in all_location_times.items():
            if times:
                location_name = game_map.get_map_name_from_map_n(location_id)
                
                mean = np.mean(times) / 60  # Convert mean to minutes
                variance = np.var(times) / 3600  # Convert variance to minutes^2
                std_dev = np.sqrt(variance)  # Standard deviation in minutes
                stats_data[location_name]['mean'] = '{:.2f}'.format(mean)
                stats_data[location_name]['variance'] = '{:.2f}'.format(variance)
                stats_data[location_name]['std_dev'] = '{:.2f}'.format(std_dev)



        # # Calculate mean, variance, and standard deviation for each location
        # for location_id, times in all_location_times.items():
        #     if times:
        #         location_name = game_map.get_map_name_from_map_n(location_id)
                
        #         mean = np.mean(times)
        #         variance = np.var(times)
        #         std_dev = np.sqrt(variance)
        #         stats_data[location_name]['mean'] = '{:.2f}'.format(mean)
        #         stats_data[location_name]['variance'] = '{:.2f}'.format(variance)
        #         stats_data[location_name]['std_dev'] = '{:.2f}'.format(std_dev)

        return checkpoint_data, stats_data

    except Exception as e:
        print("An error occurred:", e)
        # Set default values for 'Viridian City'
        viridian_city_id = LOGGABLE_LOCATIONS['Viridian City'][0]
        checkpoint_data[game_map.get_map_name_from_map_n(viridian_city_id)] = '0.00'
        
        stats_data[game_map.get_map_name_from_map_n(viridian_city_id)] = {
            'mean': '0.00',
            'variance': '0.00',
            'std_dev': '0.00'
        }
        return checkpoint_data, stats_data

checkpoint_dict, stats_dict = read_checkpoint_logs()
if checkpoint_dict is not None and stats_dict is not None:
    with open('checkpoint_dict.txt', 'w') as f:
        f.write(str(checkpoint_dict))
        f.write('\n')
        f.write(str(stats_dict))

# if mean_sigma_dict is not None:
#     with open('mean_sigma_dict.txt', 'w') as f:
#         f.write(str(mean_sigma_dict))

# def read_checkpoint_logs():
#     checkpoint_data = defaultdict(str)
#     mean_sigma_data = defaultdict(lambda: {'mean': 0, 'sigma': 0})

#     base_dir = "experiments"
#     file_path = "experiments/running_experiment.txt"

#     with open(file_path, "r") as pathfile:
#         exp_uuid8 = pathfile.readline().strip()

#     sessions_path = os.path.join(base_dir, exp_uuid8, "sessions")
#     output_file_path = os.path.join(base_dir, exp_uuid8, "checkpoints_log.txt")

#     try:
#         for folder in os.listdir(sessions_path):
#             session_path = os.path.join(sessions_path, folder)
#             checkpoint_log_file = None
#             for file in os.listdir(session_path):
#                 if file.startswith("checkpoint_log") and file.endswith(".txt"):
#                     checkpoint_log_file = os.path.join(session_path, file)
#                     break
            
#             if checkpoint_log_file is not None:
#                 with open(checkpoint_log_file, 'r') as log_file:
#                     lines = log_file.readlines()
        
#                 current_location_times = defaultdict(lambda: float('inf'))
#                 current_location_fns = defaultdict(lambda: float('inf')) # added for mean and sigma calc

#                 for line in lines:
#                     line = line.strip()
#                     if line.startswith("Location ID:"):
#                         current_location_id = line.split(":")[-1].strip()
#                         if current_location_id.isdigit() and int(current_location_id) in [map_id for location, (map_id, _) in LOGGABLE_LOCATIONS.items() if map_id is not None]:
#                             current_location = int(current_location_id)
#                         else:
#                             current_location = None  # Skip if not a valid location ID
#                     elif line.startswith("Time Visited:") and current_location is not None:
#                         time_visited = float(line.split(":")[-1].strip())
#                         current_location_times[current_location] = min(current_location_times[current_location], time_visited)
#                         current_location_fns[current_location] = current_location_fns[current_location], time_visited # added for mean and sigma calc

#                 for location_id, time_visited in current_location_times.items():
#                     location_name = game_map.get_map_name_from_map_n(location_id)
#                     formatted_time = '{:.2f}'.format(time_visited)
                    
#                     if location_name in checkpoint_data:
#                         existing_time = float(checkpoint_data[location_name])
#                         if time_visited < existing_time:
#                             checkpoint_data[location_name] = formatted_time
#                     else:
#                         checkpoint_data[location_name] = formatted_time

#                 # Logic in this for loop (uncomment)
#                 # for location_id, time_visited in current_location_fns.items():
#                 #     location_name_fn = game_map.get_map_name_from_map_n(location_id)
#                 #     formatted_time = '{:.2f}'.format(time_visited)
                    
#                 #     if location_name_fn in checkpoint_data:
#                 #         existing_time = float(checkpoint_data[location_name_fn])
#                 #         if time_visited < existing_time:
#                 #             checkpoint_data[location_name_fn] = formatted_time
#                 #     else:
#                 #         checkpoint_data[location_name] = formatted_time

#                 # print("Final checkpoint data:", checkpoint_data)
#         return checkpoint_data

#     except Exception as e:
#         print("An error occurred:", e)
#         return None

# checkpoint_dict = read_checkpoint_logs()
# if checkpoint_dict is not None:
#     with open('checkpoint_dict.txt', 'w') as f:
#         f.write(str(checkpoint_dict))
