#!/bin/bash

# Directory where the files are located
DIRECTORY="/puffertank/pufferlib"

# Delete files starting with "image_test_data" and ending with ".png"
find "$DIRECTORY" -maxdepth 1 -type f -name 'image_test_data*.png' -delete
# find "$DIRECTORY" -maxdepth 1 -type f -name 'image_test_data*.png' -exec echo "Deleting file: {}" \;