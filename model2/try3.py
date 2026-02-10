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

def worker(rank, world_size, matrix):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    matrix = matrix.to(device)
    dist.broadcast(matrix, src=0)  # Broadcast matrix from rank 0 to all ranks
    print(f"Rank {rank} received matrix:\n{matrix}")
    cleanup()

if __name__ == "__main__":
    print("Testing Mamba model with random input...")
    world_size = 2  # Number of processes (GPUs)
    
    # Create a random matrix on the CPU
    matrix = torch.randn(100, 100)
    
    mp.spawn(worker, args=(world_size, matrix), nprocs=world_size, join=True)
    print("Test completed.")
    