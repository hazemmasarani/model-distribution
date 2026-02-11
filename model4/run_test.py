import torch
from transformers import AutoTokenizer, MambaForCausalLM

# Load pretrained model and tokenizer
# model_name = "state-spaces/mamba-130m-hf"
model_name = "state-spaces/mamba-2.8b-hf"
model = MambaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Copy vocab size from model configuration
vocab_size = model.config.vocab_size

# Define batch size and sequence length
batch_size = 2
seq_len = 16

# Create random input token IDs within vocab range
input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

# Forward pass
outputs = model(
    input_ids,
    return_dict=True
)

print(outputs)
