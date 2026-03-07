import torch

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf").to("cuda:0")

# Copy vocab size from model configuration
vocab_size = model.config.vocab_size
# print("Vocab size:", vocab_size)

# Define batch size and sequence length
batch_size = 2
seq_len = 256

torch.manual_seed(42)
input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len)).detach().to("cuda:0")

with torch.no_grad():
    embeddings = model.get_input_embeddings()(input_ids)

    embeddings_perturbed = embeddings + 1e-7 * torch.randn_like(embeddings)

    out1 = model(inputs_embeds=embeddings)
    out2 = model(inputs_embeds=embeddings_perturbed)

print("Sensitivity:", (out1.logits - out2.logits).abs().max())