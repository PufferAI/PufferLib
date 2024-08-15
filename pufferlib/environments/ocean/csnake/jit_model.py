import torch

# An instance of your model.
model = torch.load('snake_model.pt', map_location='cpu')

# An example input you would normally provide to your model's forward() method.
obs = torch.randint(0, 8, (16, 11, 11), device='cpu', dtype=torch.int64)
lstm_state = (torch.zeros((1, 16, 128), device='cpu'), torch.zeros((1, 16, 128), device='cpu'))

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, (obs, lstm_state))
torch.jit.save(traced_script_module, 'snake_model_traced.pt')
