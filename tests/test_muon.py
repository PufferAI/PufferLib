
import torch
import torch.utils.cpp_extension
import torch.nn as nn
import warnings
from typing import List

# Suppress heavyball warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module=r'heavyball.*')

# Try importing torch.optim.Muon (from muon-optim package)
from pufferlib.muon import Muon as TorchMuon

# Import heavyball's ForeachMuon
import heavyball
from heavyball import ForeachMuon

# Set compile mode to default
heavyball.utils.compile_mode = "default"

# Reproducibility
torch.manual_seed(42)
torch.set_num_threads(1)

# Config
config = {
    'learning_rate': 1e-3,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_eps': 1e-8,
}

# Model: Linear -> ReLU -> Linear (no biases)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 20, bias=True)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(20, 1, bias=True)

    def forward(self, x):
        return self.l2(self.act(self.l1(x)))

# Initialize model and data
model1 = Net()
model2 = Net()

# Copy weights
for p1, p2 in zip(model1.parameters(), model2.parameters()):
    p2.data.copy_(p1.data)

# Dummy data
x = torch.randn(16, 10)
y = torch.randn(16, 1)

# Optimizers
heavy_optimizer = ForeachMuon(
    model1.parameters(),
    lr=config['learning_rate'],
    betas=(config['adam_beta1'], config['adam_beta2']),
    eps=config['adam_eps'],
    heavyball_momentum=True,
    compile_step=False
)

torch_optimizer = TorchMuon(
    model2.parameters(),
    lr=config['learning_rate'],
    momentum=config['adam_beta1'],
    eps=config['adam_eps'],
    weight_decay=0.0,
)

# Loss function
loss_fn = nn.MSELoss()

# Training loop
n_epochs = 5
print(f"{'Epoch':<6} {'AllClose':<10} {'Max Abs Diff':<15}")
print("-" * 35)

for epoch in range(n_epochs):
    # Zero grads
    heavy_optimizer.zero_grad()
    torch_optimizer.zero_grad()

    # Forward
    loss1 = loss_fn(model1(x), y)
    loss2 = loss_fn(model2(x), y)

    # Backward
    loss1.backward()
    loss2.backward()

    # Step
    heavy_optimizer.step()
    torch_optimizer.step()

    # Compare parameters
    all_close = True
    max_diff = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        diff = (p1.data - p2.data).abs()
        max_diff = max(max_diff, diff.max().item())
        if not torch.allclose(p1.data, p2.data, atol=1e-6, rtol=1e-5):
            all_close = False

    print(f"{epoch+1:<6} {str(all_close):<10} {max_diff:<15.3e}")

    # Optional: break early if divergence
    if not all_close and max_diff > 1e-4:
        print("❗ Significant divergence detected.")
        break

print("\n✅ Test complete.")
