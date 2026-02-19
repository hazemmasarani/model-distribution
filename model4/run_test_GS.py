import torch
from transformers import MambaForCausalLM, MambaConfig

def main():
    # Load pretrained model and tokenizer
    # model_name = "state-spaces/mamba-130m-hf"
    # model_name = "state-spaces/mamba-2.8b-hf"
    model_name = "HMasarani/mamba-ssm"
    mamba_config = MambaConfig.from_pretrained(model_name, devices=["cuda:0", "cuda:1"])
    model = MambaForCausalLM.from_pretrained(model_name, config=mamba_config, ignore_mismatched_sizes=True)
    # model = MambaForCausalLM.from_pretrained(model_name)

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
if __name__ == "__main__":
    main()

