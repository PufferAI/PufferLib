import numpy as np
import gymnasium
import json
import struct

import pufferlib
from pufferlib.ocean.gpudrive.cy_gpudrive import CyGPUDrive, entity_dtype

class GPUDrive(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1,
            width=1280, height=1024,
            human_agent_idx=0,
            reward_vehicle_collision=-0.1,
            reward_offroad_collision=-0.1,
            buf = None, seed=1):

        # env
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval
        print("Num envs: ", num_envs)
        
        self.num_obs = 6 + 63*7 + 64*7
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
        
        total_agents, agent_offsets =CyGPUDrive.get_total_agent_count(
            num_envs, human_agent_idx, reward_vehicle_collision, reward_offroad_collision)
        
        self.num_agents = total_agents
        print("Num agents: ", self.num_agents)
        super().__init__(buf=buf)
        self.c_envs = CyGPUDrive(self.observations, self.actions, self.rewards, self.masks,
            self.terminals, num_envs, human_agent_idx, reward_vehicle_collision, reward_offroad_collision, offsets = agent_offsets)


    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        self.tick+=1
        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)
                info.append({'total_agents': self.num_agents}) 
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()
        
    def close(self):
        self.c_envs.close() 
def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1['x'] - p3['x']) * (p2['y'] - p1['y']) - (p1['x'] - p2['x']) * (p3['y'] - p1['y']))

def simplify_polyline(geometry, polyline_reduction_threshold):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt, optimized for Python."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry  # Not enough points to simplify

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

            if area < polyline_reduction_threshold:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]

def save_map_binary(map_data, output_file):
    """Saves map data in a binary format readable by C"""
    with open(output_file, 'wb') as f:
        # Count total entities
        print(len(map_data.get('objects', [])))
        print(len(map_data.get('roads', [])))
        num_objects = len(map_data.get('objects', []))
        num_roads = len(map_data.get('roads', []))
        # num_entities = num_objects + num_roads
        f.write(struct.pack('i', num_objects))
        f.write(struct.pack('i', num_roads))
        # f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get('objects', []):
            # Write base entity data
            obj_type = obj.get('type', 1)
            if(obj_type =='vehicle'):
                obj_type = 1
            elif(obj_type == 'pedestrian'):
                obj_type = 2;
            elif(obj_type == 'cyclist'):
                obj_type = 3;
            f.write(struct.pack('i', obj_type))  # type
            f.write(struct.pack('i', obj.get('id', 0)))   # id  
            f.write(struct.pack('i', 91))                  # array_size
            # Write position arrays
            positions = obj.get('position', [])
            for i in range(91):
                pos = positions[i] if i < len(positions) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                f.write(struct.pack('f', float(pos.get('x', 0.0))))
            for i in range(91):
                pos = positions[i] if i < len(positions) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                f.write(struct.pack('f', float(pos.get('y', 0.0))))
            for i in range(91):
                pos = positions[i] if i < len(positions) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                f.write(struct.pack('f', float(pos.get('z', 0.0))))
            
            # Write velocity arrays
            velocities = obj.get('velocity', [])
            for arr, key in [(velocities, 'x'), (velocities, 'y'), (velocities, 'z')]:
                for i in range(91):
                    vel = arr[i] if i < len(arr) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    f.write(struct.pack('f', float(vel.get(key, 0.0))))
            
            # Write heading and valid arrays
            headings = obj.get('heading', [])
            f.write(struct.pack('91f', *[float(headings[i]) if i < len(headings) else 0.0 for i in range(91)]))
            
            valids = obj.get('valid', [])
            f.write(struct.pack('91i', *[int(valids[i]) if i < len(valids) else 0 for i in range(91)]))
            
            # Write scalar fields
            f.write(struct.pack('f', float(obj.get('width', 0.0))))
            f.write(struct.pack('f', float(obj.get('length', 0.0))))
            f.write(struct.pack('f', float(obj.get('height', 0.0))))
            goal_pos = obj.get('goalPosition', {'x': 0, 'y': 0, 'z': 0})  # Get goalPosition object with default
            f.write(struct.pack('f', float(goal_pos.get('x', 0.0))))  # Get x value
            f.write(struct.pack('f', float(goal_pos.get('y', 0.0))))  # Get y value
            f.write(struct.pack('f', float(goal_pos.get('z', 0.0))))  # Get z value
            f.write(struct.pack('i', obj.get('mark_as_expert', 0)))
        
        # Write roads
        for idx, road in enumerate(map_data.get('roads', [])):
            geometry = road.get('geometry', [])
            road_type = road.get('map_element_id', 0)
            # breakpoint()
            if(len(geometry) > 10 and road_type >= 14 and road_type <=16):
                geometry = simplify_polyline(geometry, .1)
            size = len(geometry)
            # breakpoint()
            if(road_type >=0 and road_type <=3):
                road_type = 4
            elif(road_type >=5 and road_type <=13):
                road_type = 5
            elif(road_type >=14 and road_type <=16):
                road_type = 6
            elif(road_type == 17):
                road_type = 7
            elif(road_type == 18):
                road_type = 8
            elif(road_type == 19):
                road_type = 9
            elif(road_type == 20):
                road_type = 10
            # Write base entity data
            f.write(struct.pack('i', road_type))  # type
            f.write(struct.pack('i', road.get('id', 0)))    # id
            f.write(struct.pack('i', size))                 # array_size
            
            # Write position arrays
            for coord in ['x', 'y', 'z']:
                for point in geometry:
                    f.write(struct.pack('f', float(point.get(coord, 0.0))))
            # Write scalar fields
            f.write(struct.pack('f', float(road.get('width', 0.0))))
            f.write(struct.pack('f', float(road.get('length', 0.0))))
            f.write(struct.pack('f', float(road.get('height', 0.0))))
            goal_pos = road.get('goalPosition', {'x': 0, 'y': 0, 'z': 0})  # Get goalPosition object with default
            f.write(struct.pack('f', float(goal_pos.get('x', 0.0))))  # Get x value
            f.write(struct.pack('f', float(goal_pos.get('y', 0.0))))  # Get y value
            f.write(struct.pack('f', float(goal_pos.get('z', 0.0))))  # Get z value
            f.write(struct.pack('i', road.get('mark_as_expert', 0)))
def load_map(map_name, binary_output=None):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, 'r') as f:
        map_data = json.load(f)
    
    if binary_output:
        save_map_binary(map_data, binary_output)
    
    entities = np.zeros(1, dtype=entity_dtype())
    return entities

def process_all_maps():
    """Process all maps and save them as binaries"""
    import os
    from pathlib import Path

    # Create the binaries directory if it doesn't exist
    binary_dir = Path("resources/gpudrive/binaries")
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Path to the training data
    data_dir = Path("data/processed/training")
    
    # Get all JSON files in the training directory
    json_files = sorted(data_dir.glob("*.json"))[0:512]
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    for i, map_path in enumerate(json_files):
        binary_file = f"map_{i:03d}.bin"  # Use zero-padded numbers for consistent sorting
        binary_path = binary_dir / binary_file
        
        print(f"Processing {map_path.name} -> {binary_file}")
        try:
            load_map(str(map_path), str(binary_path))
        except Exception as e:
            print(f"Error processing {map_path.name}: {e}")

def test_performance(timeout=10, atn_cache=1024, num_envs=256):
    import time

    env = GPUDrive(num_envs=num_envs)
    env.reset()
    tick = 0
    num_agents = 1670
    actions = np.stack([
        np.random.randint(0, space.n + 1, (atn_cache, num_agents))
        for space in env.single_action_space
    ], axis=-1)

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]         
        env.step(atn)
        tick += 1

    print(f'SPS: {num_agents * tick / (time.time() - start)}')


if __name__ == '__main__':
    #test_performance()
    process_all_maps()
