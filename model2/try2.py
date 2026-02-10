import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    print(f"Setting up process group for rank {rank}...")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker(rank, world_size, params):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{params[rank]['device']}")

    # Example: create a random tensor on the corresponding GPU
    x = torch.randn(5, 5, device=device)
    y = x.clone()  # Create a copy of the tensor for broadcasting
    print(f"Rank {rank} tensor on device {params[rank]['device']}:\n{x}")
    dist.broadcast(x, src=0)  # Broadcast tensor from rank 0 to all ranks
    print(f"Rank {rank} received tensor:\n{x}")
    if torch.equal(x, y):
        print(f"Rank {rank} successfully broadcasted tensor.")

    cleanup()

if __name__ == "__main__":
    print("Testing Mamba model with random input...")
    world_size = 2  # Number of processes (GPUs)
    
    params = [
        {"device": 1},
        {"device": 0}
    ]
    mp.spawn(worker, args=(world_size,params), nprocs=world_size, join=True)
    print("Test completed.")