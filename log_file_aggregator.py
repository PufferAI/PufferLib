import os
from collections import defaultdict
from copy import deepcopy
# from memory_profiler import profile
# import tracemalloc

# tracemalloc.start()
# snapshots = []
# snapshots.append(tracemalloc.take_snapshot())

# Directories and files
base_dir = "experiments"
file_path = "experiments/running_experiment.txt"

# Read the exp name and strip to string
with open(file_path, "r") as pathfile:
    exp_uuid8 = pathfile.readline().strip()

# Update the sessions_path assignment
sessions_path = os.path.join(base_dir, exp_uuid8, "sessions")
output_file_path = os.path.join(base_dir, exp_uuid8, "pokes_log.txt")                                                                                                       

# Initialize a dictionary to store Pokémon information
pokemon_summary = defaultdict(list)

# Initialize sets to store unique Pokémon names and moves
unique_pokemon = set()
unique_moves = set()

# Variable to store the session folder associated with each log file
current_session_folder = None

def analyze_pokemon_data(data):
    move_counts = {}
    highest_levels = {}
    pokemon_counts = {}
    bag_item_counts = defaultdict(int)

    for session_id, session_data in data.items():
        for attributes in session_data:
            if isinstance(attributes, dict):  # Processing Pokemon data
                if 'name' in attributes and attributes['name'] != '':
                    # Unique Pokemon
                    unique_pokemon.add(attributes['name'])

                    # Moves and move counts
                    moves = attributes['moves']
                    unique_moves.update(moves)
                    for move in moves:
                        move_counts[move] = move_counts.get(move, 0) + 1

                    # Highest level of each Pokemon
                    level = int(attributes['level'])
                    pokemon_name = attributes['name']
                    if pokemon_name not in highest_levels or level > highest_levels[pokemon_name]:
                        highest_levels[pokemon_name] = level

                    # Count of each unique Pokemon
                    pokemon_counts[pokemon_name] = pokemon_counts.get(pokemon_name, 0) + 1
                    
            elif isinstance(attributes, list):  # Processing bag items
                for item in attributes:
                    bag_item_counts[item] += 1

    return {
        'Unique Pokemon': sorted(unique_pokemon),
        'Unique Moves': sorted(unique_moves),
        'Move Counts': {k: v for k, v in sorted(move_counts.items())},
        'Highest Levels': {k: v for k, v in sorted(highest_levels.items())},
        'Pokemon Counts': {k: v for k, v in sorted(pokemon_counts.items())},
        'Bag Item Counts': {k: v for k, v in sorted(bag_item_counts.items())},
    }

for folder in os.listdir(sessions_path):
    session_path = os.path.join(sessions_path, folder)
    current_session_folder = folder  # Update the current session folder
    log_file_path = os.path.join(session_path, "pokemon_party_log.txt")

    if os.path.isfile(log_file_path):
        with open(log_file_path, 'r') as log_file:
            # lines = log_file.readlines()
            collecting_bag_items = False
            current_bag_items = []  # List to store bag items for the current session
            current_pokemon = {}    # Initialize current_pokemon here

            for line in log_file:
                # Skip empty lines
                if not line.strip():
                    continue

                # Check if we have reached the "Bag Items" section
                if line.strip() == "Bag Items:":
                    collecting_bag_items = True
                    continue  # Skip the "Bag Items:" line itself

                # Collect bag items
                if collecting_bag_items:
                    if line.strip():  # If line is not empty, it's a bag item
                        current_bag_items.append(line.strip())
                    else:
                        break  # Stop collecting when reaching an empty line

                # State machine for parsing
                if line.startswith("Slot:"):
                    current_state = "slot"
                    slot = line.strip("\n").split(" ")[-1]
                    current_pokemon["slot"] = slot

                elif line.startswith("Name:"):
                    current_state = "name"
                    current_pokemon["name"] = line.strip("\n").split(" ")[-1]
                    # Add the Pokémon name to the set of unique Pokémon
                    unique_pokemon.add(current_pokemon["name"])

                elif line.startswith("Level:"):
                    current_state = "level"
                    current_pokemon["level"] = line.strip("\n").split(" ")[-1]

                elif line.startswith("Moves:"):
                    current_state = "moves"
                    moves = line.strip("\n").split(":")[-1].split(", ")
                    # Strip spaces from each move
                    moves = [move.strip() for move in moves]
                    current_pokemon["moves"] = moves

                    # Update the set of unique moves
                    unique_moves.update(moves)

                    # Add the current Pokémon to the summary dictionary for the current environment
                    pokemon_summary[current_session_folder].append(deepcopy(current_pokemon))

                    # Reset current_pokemon for the next iteration
                    current_pokemon = {}
                    current_state = None
        
                if line.startswith("Bag Items:"):
                    collecting_bag_items = True
                    continue

                if collecting_bag_items:
                    if line.strip():  # If line is not empty, it's a bag item
                        current_bag_items.append(line.strip())
                    else:
                        collecting_bag_items = False

        # print(f"Session {folder}: Bag Items - {current_bag_items}")
            
        # Add collected bag items to the summary dictionary
        if current_bag_items:
            pokemon_summary[current_session_folder].append(current_bag_items)
            
# Output the aggregated information to the file
with open(output_file_path, 'w') as output_file:
    result = analyze_pokemon_data(pokemon_summary)
    caught = result['Unique Pokemon']
    levels = result['Highest Levels']
    moves = result['Unique Moves']
    incidence = result['Move Counts']

    output_file.write("\nCaught Pokemon  Highest Level  Incidence\n")
    output_file.write("--------------  -------------  ---------\n")
    for pokemon in result['Highest Levels']:
        level = result['Highest Levels'][pokemon]
        count = result['Pokemon Counts'].get(pokemon, 0)
        output_file.write(f"{pokemon.ljust(16)}{str(level).ljust(15)}{str(count)}\n")

    output_file.write("\n\nMoves List      Incidence\n")
    output_file.write("----------      ---------\n")
    for move, count in result['Move Counts'].items():
        output_file.write(f"{move.ljust(15)} {str(count).ljust(18)}\n")
    
    # Output bag item counts
    output_file.write("\n\nBag Items      Incidence\n")
    output_file.write("----------      ---------\n")
    for item, count in result['Bag Item Counts'].items():
        output_file.write(f"{item.ljust(15)} {str(count).ljust(18)}\n")
    output_file.write("\n\n")
    
    # Write each env id's pokemons', levels, moves, and bag items.
    for env_folder, parties in sorted(pokemon_summary.items()):
        output_file.write(f"\n========== {env_folder} ==========\n")
        bag_items_written = False
        for party in parties:
            if isinstance(party, dict) and party.get('name'):
                output_file.write(f"Slot: {party.get('slot', 'empty')}\n")
                output_file.write(f"Name: {party.get('name', 'empty')}\n")
                output_file.write(f"Level: {party.get('level', '0')}\n")
                output_file.write(f"Moves: {', '.join(party.get('moves', ['empty']))}\n")
            elif isinstance(party, list) and party and not bag_items_written:
                output_file.write("\nBag Items:\n")
                for item in party:
                    output_file.write(f"- {item}\n")
                bag_items_written = True  # Ensure bag items are only written once
        output_file.write("\n")

    # Optionally, include a line to separate different environment logs
    output_file.write("=" * 30 + "\n")

# snapshots.append(tracemalloc.take_snapshot())
# for s in range(1,len(snapshots)):
#     print(f'{s} vs {s-1}')
    
#     current = snapshots[s]
#     previous = snapshots[s-1]
#     stats = current.compare_to(previous, 'lineno')
    
#     for stat in stats[:10]:
#         print(f"{stat.traceback}\n Block size: {stat.size}, Count: {stat.count}")
#     print()