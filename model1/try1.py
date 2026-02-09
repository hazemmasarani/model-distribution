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

def test_model_distribution():
    world_size = 2  # Number of processes (GPUs)
    mp.spawn(setup, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    print("Testing Mamba model with random input...")
    test_model_distribution()
    print("Test completed.")

