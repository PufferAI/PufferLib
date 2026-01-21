curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
git clone https://github.com/frixaco/PufferLib
cd PufferLib
git switch boss-fight
uv venv
source .venv/bin/activate
uv pip install -e .
python setup.py build_boss_fight --inplace --force
puffer train puffer_boss_fight --train.total-timesteps 5000000 --train.device cuda --vec.num-envs 8192 --vec.num-workers 16 --train.minibatch-size 8192 --train.max-minibatch-size 65536

puffer eval puffer*boss_fight --load-model-path $(ls -t experiments/puffer_boss_fight*\_/model\_\_.pt | head -1)
