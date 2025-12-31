import multiprocessing
from tqdm import tqdm
import time
from pufferlib.ocean.tower_climb import binding

def main(total_maps=1000, output_file="maps.bin", num_processes=4):
    print(f"Generating {total_maps} maps in parallel using {num_processes} processes...")
    start_time = time.time()

    results = []
    # Use a multiprocessing pool to run generate_one_map in parallel
    with multiprocessing.Pool(processes=4) as pool:
        # Create a unique seed for each map
        seeds = range(total_maps)
        for result in tqdm(pool.imap_unordered(binding.generate_one_map, seeds), total=total_maps, desc="Generating maps"):
            results.append(result)

    end_time = time.time()
    print(f"Finished generation in {end_time - start_time:.2f} seconds.")

    print(f"Saving {total_maps} maps to {output_file}...")
    with open(output_file, "wb") as f:
        # Write the header (total number of maps)
        f.write(total_maps.to_bytes(4, 'little'))

        # Write each level's data
        for res in results:
            map_data, rows, cols, size, total_length, goal_loc, spawn_loc = res
            f.write(rows.to_bytes(4, 'little'))
            f.write(cols.to_bytes(4, 'little'))
            f.write(size.to_bytes(4, 'little'))
            f.write(total_length.to_bytes(4, 'little'))
            f.write(goal_loc.to_bytes(4, 'little'))
            f.write(spawn_loc.to_bytes(4, 'little'))
            f.write(map_data)

    print("Done.")

if __name__ == "__main__":
    main(total_maps=20000)
