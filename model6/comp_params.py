from mamba_ssm_modeling import MambaForCausalLM_SSM
from mamba_gate_modeling import MambaForCausalLM_Gate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist

torch.manual_seed(42)

mamba_orig = "state-spaces/mamba-2.8b-hf"
mamba_ssm = "hmasarani/mamba-ssm"
mamba_gate = "hmasarani/mamba-gate"

model_orig = AutoModelForCausalLM.from_pretrained(mamba_orig)
model_ssm = MambaForCausalLM_SSM.from_pretrained(mamba_ssm)
model_gate = MambaForCausalLM_Gate.from_pretrained(mamba_gate)


# Compare parameters
# Function to compare parameters
def compare_shared_params(model_a, model_b, name_a="model_a", name_b="model_b"):
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    shared_keys = set(params_a.keys()).intersection(params_b.keys())
    print(f"\nComparing shared parameters between {name_a} and {name_b}")
    print(f"Number of shared params: {len(shared_keys)}")

    identical_count = 0
    max_diff = 0.0
    for key in shared_keys:
        tensor_a = params_a[key].detach()
        tensor_b = params_b[key].detach()
        if tensor_a.shape != tensor_b.shape:
            print(f"Shape mismatch: {key} -> {tensor_a.shape} vs {tensor_b.shape}")
            continue

        diff = (tensor_a - tensor_b).abs()
        max_diff_key = diff.max().item()
        max_diff = max(max_diff, max_diff_key)

        if torch.allclose(tensor_a, tensor_b, atol=1e-8, rtol=1e-5):
            identical_count += 1
        else:
            print(f"Param differs: {key} | max diff: {max_diff_key}")

    print(f"Shared parameters identical: {identical_count}/{len(shared_keys)}")
    print(f"Maximum absolute difference among shared params: {max_diff}\n")

# Compare original vs SSM
compare_shared_params(model_orig, model_ssm, "original", "SSM")

# Compare original vs Gate
compare_shared_params(model_orig, model_gate, "original", "Gate")


