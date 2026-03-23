import torch
from torch import nn
from torch.utils.benchmark import Timer
from torch.utils.flop_counter import FlopCounterMode

from torch.backends import cudnn
cudnn.benchmark = True
cudnn.deterministic = False
cudnn.benchmark_limit = 32

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

INPUT_SIZE = 128
HIDDEN_SIZE1 = 128
HIDDEN_SIZE2 = 512
OUTPUT_SIZE = 128
B = 8192
dtype = torch.bfloat16
inner_loops = 100  # Number of inner iterations to amortize overhead

# Define the model with explicit Kaiming uniform initialization to match JAX
model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE1),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_SIZE1, HIDDEN_SIZE2),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_SIZE2, OUTPUT_SIZE),
).cuda().to(dtype)

# Create input batch
batch = torch.randn(B, INPUT_SIZE).cuda().to(dtype)

# Define a multi-step function to run multiple forwards in one compiled graph
@torch.compile(mode='max-autotune')
def multi_step(model, batch, inner_loops):
    with torch.no_grad():
        carry = torch.tensor(0.0, dtype=torch.float32, device='cuda')
        for i in range(inner_loops):
            y = model(batch)
            carry = carry + y.sum()

        return carry

# Manual FLOPs calculation to match JAX (ignores bias adds and ReLUs as negligible)
flops = (
    2 * B * INPUT_SIZE * HIDDEN_SIZE1 +
    2 * B * HIDDEN_SIZE1 * HIDDEN_SIZE2 +
    2 * B * HIDDEN_SIZE2 * OUTPUT_SIZE
)

# Warmup
for _ in range(10):
    _ = multi_step(model, batch, inner_loops)

# Timing
timer = Timer(
    stmt='multi_step(model, batch, inner_loops)',
    globals={'multi_step': multi_step, 'model': model, 'batch': batch, 'inner_loops': inner_loops}
)
output = timer.timeit(50)

cost = output.mean / inner_loops  # Average time per forward pass (fixed from times[0] to mean)
FLOPS = flops / cost
print(f'TFLOPS: {FLOPS / 1e12:.2f}')
