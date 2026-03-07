import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

from mamba_ssm_modeling import MambaForCausalLM_SSM
from mamba_gate_modeling import MambaForCausalLM_Gate
from transformers import AutoModelForCausalLM


def setup(rank, world_size, port_num):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port_num)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def worker(rank, world_size, devices, embeddings, port_num, return_dict):
    setup(rank, world_size, port_num)
    print(f"Rank {rank} initialized")
    device = torch.device(devices[rank])
    embeddings = embeddings.to(device)

    if rank == 0:
        model = MambaForCausalLM_Gate.from_pretrained("HMasarani/mamba-gate")
    else:
        model = MambaForCausalLM_SSM.from_pretrained("HMasarani/mamba-ssm")
    print(f"Rank {rank} model loaded")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, return_dict=True)
        print(f"Rank {rank} logits shape: {outputs.logits.shape}")
        logits = outputs.logits.cpu()
        print("logits copied to cpu.")

    if rank == 0:
        return_dict[rank] = logits
    print(f"Rank {rank} done")

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Run Mamba models on multiple GPUs")
    parser.add_argument("-dev1", type=str, required=True, help="First GPU device, e.g. cuda:0")
    parser.add_argument("-dev2", type=str, required=True, help="Second GPU device, e.g. cuda:1")
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("-port_num", type=int, default=12355, help="Port number")
    args = parser.parse_args()

    torch.manual_seed(42)

    world_size = 2
    devices = [args.dev1, args.dev2]

    # Create input_ids on CPU first
    input_ids = torch.randint(
        low=0,
        high=50280,  # Assuming vocab size of 50280, adjust if needed
        size=(args.batch_size, args.seq_len)
    )

    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")

    embeddings = model.get_input_embeddings()(input_ids).detach()
    embeddings_perturbed = embeddings + 1e-7 * torch.randn_like(embeddings).detach()
    
    del model

    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        worker,
        args=(world_size, devices, embeddings, args.port_num, return_dict),
        nprocs=world_size,
        join=True,
    )

    logits_model1 = return_dict[0]

    print("Model 1 logits shape:", logits_model1.shape)

    mp.spawn(
        worker,
        args=(world_size, devices, embeddings_perturbed, args.port_num, return_dict),
        nprocs=world_size,
        join=True,
    )

    logits_model2 = return_dict[0]

    print("Model 2 logits shape:", logits_model2.shape)

    print("Sensitivity:", (logits_model1 - logits_model2).abs().max())

if __name__ == "__main__":
    main()