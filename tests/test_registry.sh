#!/bin/bash

# Loop through all folders in the `registry` directory
for folder in pufferlib/registry/*; do
  if [ -d "$folder" ]; then
    # Extract folder name
    folder_name=$(basename $folder)

    if [[ $folder_name == __* ]]; then
      continue
    fi
   
    # Install package with extras
    pip install -e .[$folder_name] > /dev/null 2>&1
    
    # Run tests
    python tests/test_registry.py $folder_name
  fi
done
