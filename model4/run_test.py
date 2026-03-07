import torch
from transformers import AutoTokenizer, MambaForCausalLM

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model_name = "state-spaces/mamba-2.8b-hf"
model = MambaForCausalLM.from_pretrained(model_name).to("cuda:3")
tokenizer = AutoTokenizer.from_pretrained(model_name)

vocab_size = model.config.vocab_size

batch_size = 2
seq_len = 128

# # Create input_ids on CPU first
# input_ids = torch.randint(
#     low=0,
#     high=50280,  # Assuming vocab size of 50280, adjust if needed
#     size=(2, 128)
# )

# input_ids = input_ids.to("cuda:3")

input_ids = torch.load("../model6/input_ids.pt").to("cuda:3")

torch.save(input_ids, "../model6/layers_output/orig_mamba/main_input_ids/main_input_ids_layer_0.pt")

outputs = model(input_ids=input_ids, return_dict=True)

# Move to CPU before saving
logits = outputs.logits.detach().cpu()

# Save as PyTorch file
torch.save(logits, "../model6/mamba_outputs.pt")

print("Saved logits to mamba_outputs.pt")
