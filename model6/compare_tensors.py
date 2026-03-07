import torch

# # Load tensors
# ssm = torch.load("ssm_model_scan_output.pt")
# orig = torch.load("orig_model_scan_output.pt")
ssm = torch.load("ssm_model_hidden_states.pt")
orig = torch.load("orig_model_hidden_states.pt")

# 1️⃣ Check type
print("Type ssm:", type(ssm))
print("Type orig:", type(orig))

# 2️⃣ If they are dicts (very common)
if isinstance(ssm, dict):
    print("Keys ssm:", ssm.keys())
    print("Keys orig:", orig.keys())

# 3️⃣ If they are tensors
if isinstance(ssm, torch.Tensor):
    print("Shape ssm:", ssm.shape)
    print("Shape orig:", orig.shape)

    # Exact match
    print("Exact equal:", torch.equal(ssm, orig))

    # Close match (floating point tolerance)
    print("Allclose:", torch.allclose(ssm, orig, atol=1e-6))

    # Difference statistics
    diff = (ssm - orig).abs()
    print("Max diff:", diff.max().item())
    print("Mean diff:", diff.mean().item())


# 4️⃣ print the first and last elements of the tensors
# First element
print("\nFirst element:")
print("ssm:", ssm[0, 0, 0].item())
print("orig:", orig[0, 0, 0].item())

# Last element
print("\nLast element:")
print("ssm:", ssm[-1, -1, -1].item())
print("orig:", orig[-1, -1, -1].item())

# First row
print("\nFirst row comparison:")
print("ssm:", ssm[0, 0, :5])
print("orig:", orig[0, 0, :5])

# Last row
print("\nLast row comparison:")
print("ssm:", ssm[-1, -1, :5])
print("orig:", orig[-1, -1, :5])