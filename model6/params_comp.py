import torch
from mamba_ssm_modeling import MambaForCausalLM_SSM
from transformers import MambaForCausalLM


torch.manual_seed(42)
torch.cuda.manual_seed(42)




ssm_model_mane = "HMasarani/mamba-ssm"
model_orig_name = "state-spaces/mamba-2.8b-hf"

model_ssm = MambaForCausalLM_SSM.from_pretrained(ssm_model_mane).to("cuda:0")

model_orig = MambaForCausalLM.from_pretrained(model_orig_name).to("cuda:0")

input_ids = torch.randint(low=0, high=10, size=(2, 16)).to("cuda:0")
outputs_orig = model_orig(input_ids, return_dict=True)
outputs_ssm = model_ssm(input_ids, return_dict=True)

# check if outputs are equal
print(outputs_orig.logits.detach().cpu())
print(outputs_ssm.logits.detach().cpu())
print(torch.allclose(outputs_orig.logits.detach().cpu(), outputs_ssm.logits.detach().cpu(), atol=1e-2))