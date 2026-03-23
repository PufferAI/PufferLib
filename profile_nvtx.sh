nsys profile --capture-range=cudaProfilerApi --sample=none -o profile python -m pufferlib.pufferl train puffer_breakout --train.profile True
nsys stats --report nvtx_sum profile.nsys-repne
