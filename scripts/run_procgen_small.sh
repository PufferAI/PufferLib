COMMON="--train --track --wandb-group baselines"

python demo.py $COMMON --env bigfish
python demo.py $COMMON --env ninja
python demo.py $COMMON --env miner
