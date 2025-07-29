#!/usr/bin/env python3

import torch
import numpy as np
import sys
import os
import re

def extract_weights_from_pt(pt_filepath, output_dir):
    """
    Extract weights from PyTorch .pt file and save as binary file.
    
    Args:
        pt_filepath: Path to the .pt file
        output_dir: Directory to save the binary weights file
    """
    
    # Extract run ID from filename (e.g., puffer_rocket_drone_175376118293.pt -> 175376118293)
    filename = os.path.basename(pt_filepath)
    run_id_match = re.search(r'(\d{8,})', filename)
    if not run_id_match:
        print(f"ERROR: Could not extract run ID from filename: {filename}")
        return False
    
    run_id = run_id_match.group(1)
    
    # Load PyTorch state dict
    try:
        state_dict = torch.load(pt_filepath, map_location='cpu')
        print(f"Loaded state dict with {len(state_dict)} layers")
    except Exception as e:
        print(f"ERROR: Failed to load .pt file: {e}")
        return False
    
    # Extract weights in the correct order for the C network
    weights_list = []
    
    # Network architecture order:
    # 1. log_std (7 values)
    # 2. encoder: Linear(41, 128) -> weight(128,41) + bias(128) = 5376 values
    # 3. actor: Linear(128, 7) -> weight(7,128) + bias(7) = 903 values  
    # 4. value_fn: Linear(128, 1) -> weight(1,128) + bias(1) = 129 values
    # 5. lstm: LSTM(128, 128) -> 4 weight matrices = ~131k values
    
    layer_names = [
        'policy.decoder_logstd',
        'policy.encoder.0.weight', 'policy.encoder.0.bias',
        'policy.decoder_mean.weight', 'policy.decoder_mean.bias', 
        'policy.value.weight', 'policy.value.bias',
        'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 
        'lstm.bias_ih_l0', 'lstm.bias_hh_l0'
    ]
    
    total_params = 0
    for name in layer_names:
        if name in state_dict:
            tensor = state_dict[name]
            flattened = tensor.numpy().flatten()
            weights_list.append(flattened)
            total_params += len(flattened)
            print(f"  {name}: {tensor.shape} -> {len(flattened)} params")
        else:
            print(f"WARNING: Layer {name} not found in state dict")
    
    if not weights_list:
        print("ERROR: No weights extracted!")
        return False
    
    # Concatenate all weights
    all_weights = np.concatenate(weights_list).astype(np.float32)
    print(f"Total extracted weights: {len(all_weights)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binary file
    output_filename = f"drone_weights_{run_id}.bin"
    output_path = os.path.join(output_dir, output_filename)
    
    all_weights.tofile(output_path)
    print(f"Saved {len(all_weights)} weights to: {output_path}")
    
    # Also create a default drone_weights.bin for convenience
    default_path = os.path.join(output_dir, "drone_weights.bin")
    all_weights.tofile(default_path)
    print(f"Also saved as default: {default_path}")
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_pt_to_bin.py <path_to_pt_file>")
        print("Example: python convert_pt_to_bin.py experiments/puffer_rocket_drone_175376118293.pt")
        sys.exit(1)
    
    pt_filepath = sys.argv[1]
    
    if not os.path.exists(pt_filepath):
        print(f"ERROR: File not found: {pt_filepath}")
        sys.exit(1)
    
    if not pt_filepath.endswith('.pt'):
        print(f"ERROR: File must be a .pt file: {pt_filepath}")
        sys.exit(1)
    
    # Determine output directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pufferlib_dir = os.path.dirname(os.path.dirname(script_dir))
    output_dir = os.path.join(pufferlib_dir, "resources", "rocket_drones")
    
    print(f"Converting: {pt_filepath}")
    print(f"Output dir: {output_dir}")
    
    success = extract_weights_from_pt(pt_filepath, output_dir)
    
    if success:
        print("Conversion completed successfully!")
        sys.exit(0)
    else:
        print("Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 