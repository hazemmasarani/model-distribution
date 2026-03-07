import torch

# Load tensors
orig = torch.load("mamba_outputs.pt")
new1 = torch.load("mamba_outputs1_new.pt")
new2 = torch.load("mamba_outputs2_new.pt")

print("Shapes:")
print("orig:", orig.shape)
print("new1:", new1.shape)
print("new2:", new2.shape)
print()

def compare_tensors(a, b, name):
    diff = (a - b).abs()

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    rel_diff = diff / a.abs().clamp_min(1e-8)
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    print(f"---- {name} ----")
    print("Max abs diff:", max_abs)
    print("Mean abs diff:", mean_abs)
    print("Max rel diff:", max_rel)
    print("Mean rel diff:", mean_rel)
    print("Allclose (1e-5):", torch.allclose(a, b, atol=1e-5, rtol=1e-5))
    print("Allclose (1e-4):", torch.allclose(a, b, atol=1e-4, rtol=1e-4))
    print()

# Comparisons
compare_tensors(new1, orig, "new1 vs original")
compare_tensors(new2, orig, "new2 vs original")
compare_tensors(new1, new2, "new1 vs new2")