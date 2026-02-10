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

class ModelTest:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        setup(rank, world_size)

    def run_test(self):
        print(f"Running model test for rank {self.rank}...")
        # Your model code goes here
        # Example: create a random tensor on the corresponding GPU
        device = torch.device(f"cuda:{self.rank}")
        x = torch.randn(5, 5, device=device)
        print(f"Rank {self.rank} tensor:\n{x}")
        cleanup()

def run(rank, world_size):
    # Each spawned process calls this
    test = ModelTest(rank, world_size)
    test.run_test()

def test_model_distribution():
    world_size = 2  # number of processes / GPUs
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    print("Testing Mamba model with random input...")
    test_model_distribution()
    print("Test completed.")