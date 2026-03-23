import torch
from torch import nn
from torch.utils.benchmark import Timer
from torch.utils.flop_counter import FlopCounterMode
from torch import func

from torch.backends import cudnn
cudnn.benchmark = True
cudnn.deterministic = False
cudnn.benchmark_limit = 32

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

class Default(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
        )
        self.decoder = nn.Linear(hidden_size, output_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observations):
        hidden = self.encode_observations(observations)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1)
        return self.encoder(observations)

    def decode_actions(self, hidden):
        logits = self.decoder(hidden)
        values = self.value(hidden)
        return logits, values


class LSTMWrapper(nn.Module):
    def __init__(self, policy, input_size, hidden_size, output_size):
        super().__init__()
        self.policy = policy
        input_size = hidden_size

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)

    def forward(self, observations, h, c):
        hidden = self.policy.encode_observations(observations)
        hidden, c = self.cell(hidden, (h, c))
        logits, values = self.policy.decode_actions(hidden)
        return logits, values, hidden, c

def get_params_and_buffers(model):
    buffers = dict(model.named_buffers())
    param_names = [k for k, v in model.named_parameters() if v.requires_grad]
    params = [v for k, v in model.named_parameters() if v.requires_grad]
    params_dict = dict(zip(param_names, params))
    return {**buffers, **params_dict}


@torch.compile(fullgraph=True, dynamic=False, mode='reduce-overhead')
def functional_forward(model, params_and_buffers, batch, h, c):
    return func.functional_call(model, params_and_buffers, (batch, h, c))

def rollout(model, params_and_buffers, batch, h, c, seq):
    all_logits = []
    all_values = []
    for i in range(seq):
        logits, values, h, c = functional_forward(model, params_and_buffers, batch[i], h, c)
        all_logits.append(logits)
        all_values.append(values)

    logits = torch.stack(all_logits, dim=0)
    values = torch.stack(all_values, dim=0)

    return logits, values

@torch.compile(fullgraph=True, dynamic=False, mode='reduce-overhead')
def fast_rollout(model, batch, h, c, seq):
    logits = torch.empty(seq, batch.shape[1], OUTPUT_SIZE, device=batch.device, dtype=batch.dtype)
    values = torch.empty(seq, batch.shape[1], 1, device=batch.device, dtype=batch.dtype)
    for i in range(seq):
        l, v, h, c = model(batch[i], h, c)
        logits[i] = l
        values[i] = v

    return logits, values

def evaluate(model, params_and_buffers, batch, h, c, seq):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return fast_rollout(model, batch, h, c, seq)

def compute_loss(params_and_buffers, model, batch, h, c, seq):
    logits, values = rollout(model, params_and_buffers, batch, h, c, seq)
    loss = -torch.log(torch.softmax(logits, dim=-1)).mean() + (values**2).mean()
    return loss

grad_fn = torch.compile(func.grad(compute_loss),
    fullgraph=True, dynamic=False, mode='reduce-overhead')

#grad_fn = func.grad(compute_loss)

def train(model, params_and_buffers, batch, h, c, loops, seq):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(loops):
            grads = grad_fn(params_and_buffers, model, batch, h, c, seq)
            for name in grads:
                params_and_buffers[name].sub_(0.01 * grads[name])

    return params_and_buffers

if __name__ == '__main__':
    INPUT_SIZE = 128
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 4
    B = 256
    SEQ = 64
    LOOPS = 4  
    dtype = torch.bfloat16

    model = LSTMWrapper(
        Default(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE),
        INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
    ).cuda()

    # TODO: carefully test slowdown from this
    params_and_buffers = get_params_and_buffers(model)
    #model = torch.compile(model, mode='reduce-overhead', dynamic=False, fullgraph=True)

    # Create input batch
    batch = torch.randn(SEQ, B, INPUT_SIZE).cuda().to(dtype)

    # Define a multi-step function to run multiple forwards in one compiled graph
    # Manual FLOPs calculation
    I = INPUT_SIZE
    H = HIDDEN_SIZE
    O = OUTPUT_SIZE
    flops = B * (2*I*H + 16*H*H + 2*H*O + 2*H)

    h = torch.zeros(B, HIDDEN_SIZE).cuda().to(dtype)
    c = torch.zeros(B, HIDDEN_SIZE).cuda().to(dtype)

    # Warmup
    for _ in range(3):
        _ = evaluate(model, params_and_buffers, batch, h, c, SEQ)
    # Timing
    timer = Timer(
        stmt='evaluate(model, params_and_buffers, batch, h, c, SEQ)',
        globals={
            'evaluate': evaluate,
            'params_and_buffers': params_and_buffers,
            'model': model,
            'batch': batch,
            'h': h,
            'c': c,
            'SEQ': SEQ,
        }
    )
    output = timer.timeit(LOOPS)

    cost = output.mean / SEQ  # Average time per forward pass (fixed from times[0] to mean)
    FLOPS = flops / cost
    perf_evaluate = f'FLOPS: {FLOPS / 1e12:.2f}T, SPS: {B/cost/1e6:.2f}M'

    # Warmup
    for _ in range(1):
        _ = train(model, params_and_buffers, batch, h, c, LOOPS, SEQ)

    # Timing
    timer = Timer(
        stmt='train(model, params_and_buffers, batch, h, c, LOOPS, SEQ)',
        globals={
            'train': train,
            'params_and_buffers': params_and_buffers,
            'model': model,
            'batch': batch,
            'h': h,
            'c': c,
            'LOOPS': LOOPS,
            'SEQ': SEQ,
        }
    )

    output = timer.timeit(1)
    cost = output.mean / SEQ / LOOPS  # Average time per forward pass (fixed from times[0] to mean)
    FLOPS = 3*flops / cost
    perf_train = f'FLOPS: {FLOPS / 1e12:.2f}T, SPS: {B/cost/1e6:.2f}M'

    print(perf_evaluate)
    print(perf_train)
