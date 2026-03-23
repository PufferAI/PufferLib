import jax
import jax.numpy as jnp
from jax import jit, random, lax
import timeit

INPUT_SIZE = 16
HIDDEN_SIZE = 128
OUTPUT_SIZE = 16
B = 2048
dtype = jnp.bfloat16
inner_loops = 100  # Number of inner iterations to amortize overhead

def init_params(key):
    keys = random.split(key, 3)
    # Use uniform initialization to match PyTorch's Kaiming uniform for ReLU
    bound1 = jnp.sqrt(6 / INPUT_SIZE)
    w1 = random.uniform(keys[0], shape=(INPUT_SIZE, HIDDEN_SIZE), minval=-bound1, maxval=bound1, dtype=dtype)
    b1 = jnp.zeros(HIDDEN_SIZE, dtype=dtype)
    bound2 = jnp.sqrt(6 / HIDDEN_SIZE)
    w2 = random.uniform(keys[1], shape=(HIDDEN_SIZE, HIDDEN_SIZE), minval=-bound2, maxval=bound2, dtype=dtype)
    b2 = jnp.zeros(HIDDEN_SIZE, dtype=dtype)
    bound3 = jnp.sqrt(6 / HIDDEN_SIZE)
    w3 = random.uniform(keys[2], shape=(HIDDEN_SIZE, OUTPUT_SIZE), minval=-bound3, maxval=bound3, dtype=dtype)
    b3 = jnp.zeros(OUTPUT_SIZE, dtype=dtype)
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3}

def model(params, x):
    precision = lax.Precision.HIGH  # Use HIGH precision for 4090 to leverage Tensor Cores
    h = jnp.maximum(jnp.dot(x, params['w1'], precision=precision) + params['b1'], 0)
    h = jnp.maximum(jnp.dot(h, params['w2'], precision=precision) + params['b2'], 0)
    return jnp.dot(h, params['w3'], precision=precision) + params['b3']

# Manual FLOPs calculation (ignores bias adds and ReLUs as negligible)
flops_per_forward = (
    2 * B * INPUT_SIZE * HIDDEN_SIZE +      # First matmul
    2 * B * HIDDEN_SIZE * HIDDEN_SIZE +     # Second matmul
    2 * B * HIDDEN_SIZE * OUTPUT_SIZE       # Third matmul
)

# Create concrete inputs
key = random.key(0)
params = init_params(key)
batch = random.normal(random.key(1), (B, INPUT_SIZE), dtype=dtype)

# Define a jitted multi-step function with lax.scan for better optimization
@jit
def multi_step(params, batch):
    def body_fun(carry, _):
        y = model(params, batch)
        carry += y.sum()  # Forces computation without noise
        return carry, None
    carry, _ = lax.scan(body_fun, jnp.array(0.0, dtype=jnp.float32), None, length=inner_loops)
    return carry

# Warmup
for _ in range(10):
    _ = multi_step(params, batch).block_until_ready()

# Timing
def run():
    return multi_step(params, batch).block_until_ready()

t = timeit.timeit(run, number=10)
cost = t / 10 / inner_loops  # Average time per forward pass

FLOPS = flops_per_forward / cost
print(f'TFLOPS: {FLOPS / 1e12:.2f}')
