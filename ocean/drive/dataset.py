"""Convert Waymo Open Motion Dataset (WOMD) JSON maps to binary format for PufferLib drive env.

Step 0: Download the preprocessed JSON scenarios from HuggingFace e.g.,
  https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_train_1000

  uv pip install huggingface_hub
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='daphne-cornelisse/pufferdrive_womd_train_1000', repo_type='dataset', local_dir='drive_data')"

Step 1: Unzip to get folder with .json files
    mkdir -p drive_data/training
    tar xzf drive_data/pufferdrive_womd_train_1000.tar.gz --strip-components=1 -C drive_data/training/

Step 2: Process to map binaries
  python ocean/drive/dataset.py --data_folder drive_data/training --output_dir drive_data/binaries
"""

import json
import struct
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

TRAJECTORY_LENGTH = 91


def calculate_area(p1, p2, p3):
    """Calculate the area of the triangle using the determinant method."""
    return 0.5 * abs(
        (p1["x"] - p3["x"]) * (p2["y"] - p1["y"])
        - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"])
    )


def dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy


def simplify_polyline(geometry, polyline_reduction_threshold, max_segment_length):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)
            if (
                area < polyline_reduction_threshold
                and dist(point1, point3) <= max_segment_length
            ):
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def save_map_binary(map_data, output_file, unique_map_id):
    """Save map data in a binary format readable by C."""
    with open(output_file, "wb") as f:
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))

        # Write objects
        for obj in map_data.get("objects", []):
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", obj_type))
            f.write(struct.pack("i", TRAJECTORY_LENGTH))

            positions = obj.get("position", [])
            for coord in ["x", "y", "z"]:
                for i in range(TRAJECTORY_LENGTH):
                    pos = (
                        positions[i]
                        if i < len(positions)
                        else {"x": 0.0, "y": 0.0, "z": 0.0}
                    )
                    f.write(struct.pack("f", float(pos.get(coord, 0.0))))

            velocities = obj.get("velocity", [])
            for coord in ["x", "y", "z"]:
                for i in range(TRAJECTORY_LENGTH):
                    vel = (
                        velocities[i]
                        if i < len(velocities)
                        else {"x": 0.0, "y": 0.0, "z": 0.0}
                    )
                    f.write(struct.pack("f", float(vel.get(coord, 0.0))))

            headings = obj.get("heading", [])
            f.write(
                struct.pack(
                    f"{TRAJECTORY_LENGTH}f",
                    *[
                        float(headings[i]) if i < len(headings) else 0.0
                        for i in range(TRAJECTORY_LENGTH)
                    ],
                )
            )

            valids = obj.get("valid", [])
            f.write(
                struct.pack(
                    f"{TRAJECTORY_LENGTH}i",
                    *[
                        int(valids[i]) if i < len(valids) else 0
                        for i in range(TRAJECTORY_LENGTH)
                    ],
                )
            )

            f.write(struct.pack("f", float(obj.get("width", 0.0))))
            f.write(struct.pack("f", float(obj.get("length", 0.0))))
            f.write(struct.pack("f", float(obj.get("height", 0.0))))
            goal_pos = obj.get("goalPosition", {"x": 0, "y": 0, "z": 0})
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))
            f.write(struct.pack("i", obj.get("mark_as_expert", 0)))

        # Write roads
        for road in map_data.get("roads", []):
            geometry = road.get("geometry", [])
            road_type = road.get("map_element_id", 0)
            road_type_word = road.get("type", 0)
            if road_type_word == "lane":
                road_type = 2
            elif road_type_word == "road_edge":
                road_type = 15

            if len(geometry) > 10 and road_type <= 16:
                geometry = simplify_polyline(geometry, 0.1, 250)
            size = len(geometry)

            if 0 <= road_type <= 3:
                road_type = 4
            elif 5 <= road_type <= 13:
                road_type = 5
            elif 14 <= road_type <= 16:
                road_type = 6
            elif road_type == 17:
                road_type = 7
            elif road_type == 18:
                road_type = 8
            elif road_type == 19:
                road_type = 9
            elif road_type == 20:
                road_type = 10

            f.write(struct.pack("i", road_type))
            f.write(struct.pack("i", size))

            for coord in ["x", "y", "z"]:
                for point in geometry:
                    f.write(struct.pack("f", float(point.get(coord, 0.0))))

            f.write(struct.pack("f", float(road.get("width", 0.0))))
            f.write(struct.pack("f", float(road.get("length", 0.0))))
            f.write(struct.pack("f", float(road.get("height", 0.0))))
            goal_pos = road.get("goalPosition", {"x": 0, "y": 0, "z": 0})
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))
            f.write(struct.pack("i", road.get("mark_as_expert", 0)))


def _process_single_map(args):
    """Worker function to process a single map file."""
    i, map_path, binary_path = args
    try:
        with open(map_path, "r") as f:
            map_data = json.load(f)
        save_map_binary(map_data, str(binary_path), i)
        return (i, map_path.name, True, None)
    except Exception as e:
        return (i, map_path.name, False, str(e))


def process_all_maps(
    data_folder,
    output_dir,
    max_maps=50_000,
    num_workers=None,
):
    """Process JSON map files into binary format using multiprocessing.

    Args:
        data_folder: Path to the folder containing JSON map files.
        output_dir: Path to the directory where binary files will be written.
        max_maps: Maximum number of maps to process.
        num_workers: Number of parallel workers (defaults to cpu_count()).
    """
    if num_workers is None:
        num_workers = cpu_count()

    data_dir = Path(data_folder)
    binary_dir = Path(output_dir)
    binary_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    tasks = []
    for i, map_path in enumerate(json_files[:max_maps]):
        binary_path = binary_dir / f"map_{i:03d}.bin"
        tasks.append((i, map_path, binary_path))

    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_process_single_map, tasks),
                total=len(tasks),
                desc="Processing maps",
                unit="map",
            )
        )

    successful = sum(1 for _, _, success, _ in results if success)
    failed = sum(1 for _, _, success, _ in results if not success)

    print(f"\nProcessed {successful}/{len(results)} maps successfully.")
    if failed > 0:
        print(f"Failed {failed}/{len(results)} files:")
        for i, name, success, error in results:
            if not success:
                print(f"  {name}: {error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON map files to binary format.")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to folder containing JSON map files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for binary files.")
    parser.add_argument("--max_maps", type=int, default=50_000, help="Maximum number of maps to process.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers.")
    args = parser.parse_args()

    process_all_maps(
        data_folder=args.data_folder,
        output_dir=args.output_dir,
        max_maps=args.max_maps,
        num_workers=args.num_workers,
    )