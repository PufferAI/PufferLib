#!/bin/bash

environments=(
    "puffer_squared"
    "puffer_password"
    "puffer_stochastic"
    "puffer_memory"
    "puffer_multiagent"
    "puffer_spaces"
    "puffer_bandit"
)

for env in "${environments[@]}"; do
    echo "Training: $env"
    python demo.py --mode train --vec multiprocessing --track --env "$env"
done
