import torch
from torch import nn

torch.manual_seed(0)

hidden_size = 256
intermediate_size = 128
print(f"hidden_size: {hidden_size}, intermediate_size: {intermediate_size}")

device = "cuda:3"
print(f"device: {device}")

input = torch.randn((2,16,hidden_size)).to(device)

in_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False).to(device)

# Original output
out = in_proj(input) # [hidden_size, intermediate_size * 2]
hid_orig, gate_orig = out.chunk(2, dim=-1)

# Split layers
in_proj_hid = nn.Linear(hidden_size, intermediate_size, bias=False).to(device)
in_proj_gate = nn.Linear(hidden_size, intermediate_size, bias=False).to(device)

with torch.no_grad():
    in_proj_hid.weight.copy_(in_proj.weight[:intermediate_size, :])
    in_proj_gate.weight.copy_(in_proj.weight[intermediate_size:, :])

hid_new = in_proj_hid(input)
gate_new = in_proj_gate(input)

print(torch.equal(hid_new, hid_orig))   # True
print(torch.equal(gate_new, gate_orig)) # True