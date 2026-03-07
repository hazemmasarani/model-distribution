import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

from mamba_ssm_modeling import MambaForCausalLM_SSM
from mamba_gate_modeling import MambaForCausalLM_Gate


def setup(rank, world_size, port_num):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port_num)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def worker(rank, world_size, devices, input_ids, port_num, return_dict):
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    setup(rank, world_size, port_num)
    print(f"Rank {rank} initialized")
    device = torch.device(devices[rank])
    input_ids = input_ids.to(device)

    if rank == 0:
        model = MambaForCausalLM_Gate.from_pretrained("HMasarani/mamba-gate")
        torch.save(input_ids, "../model6/layers_output/gate_mamba/main_input_ids/main_input_ids_layer_0.pt")
    else:
        model = MambaForCausalLM_SSM.from_pretrained("HMasarani/mamba-ssm")
        torch.save(input_ids, "../model6/layers_output/ssm_mamba/main_input_ids/main_input_ids_layer_0.pt")
    print(f"Rank {rank} model loaded")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
        print(f"Rank {rank} logits shape: {outputs.logits.shape}")
        logits = outputs.logits.cpu()
        print("logits copied to cpu.")

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

    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    world_size = 2
    devices = [args.dev1, args.dev2]

    # # Create input_ids on CPU first
    # input_ids = torch.randint(
    #     low=0,
    #     high=50280,  # Assuming vocab size of 50280, adjust if needed
    #     size=(args.batch_size, args.seq_len)
    # )

    input_ids = torch.load("../model6/input_ids.pt")

    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        worker,
        args=(world_size, devices, input_ids, args.port_num, return_dict),
        nprocs=world_size,
        join=True,
    )

    logits_model1 = return_dict[0]
    logits_model2 = return_dict[1]
    torch.save((logits_model1), "mamba_outputs1_new.pt")
    torch.save((logits_model2), "mamba_outputs2_new.pt")
    print("Model 1 logits shape:", logits_model1.shape)
    print("Model 2 logits shape:", logits_model2.shape)


if __name__ == "__main__":
    main()