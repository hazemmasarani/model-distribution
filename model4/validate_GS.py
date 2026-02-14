import torch
from transformers import MambaForCausalLM, MambaConfig

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load pretrained model and tokenizer
# model_name = "state-spaces/mamba-130m-hf"
model_name = "state-spaces/mamba-2.8b-hf"
mamba_config = MambaConfig.from_pretrained(model_name, devices=["cuda:1", "cuda:0"])
model1 = MambaForCausalLM.from_pretrained(model_name, config=mamba_config)
model2 = MambaForCausalLM.from_pretrained(model_name)

# Copy vocab size from model configuration
vocab_size = model1.config.vocab_size

# Define batch size and sequence length
batch_size = 2
seq_len = 16

# Create random input token IDs within vocab range
input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

# Forward pass
outputs1 = model1(
    input_ids,
    return_dict=True
)
outputs2 = model2(
    input_ids,
    return_dict=True
)

out1 = outputs1.logits
out2 = outputs2.logits

print("All close?", torch.allclose(out1, out2, atol=1e-5, rtol=1e-5))
