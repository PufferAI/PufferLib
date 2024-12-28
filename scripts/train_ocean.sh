#!/bin/bash
 
environments=(
    "puffer_codeball"
    "puffer_breakout"
    "puffer_connect4"
    "puffer_pong"
    "puffer_snake"
    "puffer_tripletriad"
    "puffer_rware"
    "puffer_go"
    "puffer_tactics"
    "puffer_moba"
    "puffer_codeball"
)

for env in "${environments[@]}"; do
    echo "Training: $env"
    python demo.py --mode train --vec multiprocessing --track --env "$env"
done
