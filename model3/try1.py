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

    if rank == 0:
        x = torch.randn(5, 5, device=device)
        y = torch.zeros_like(x, device=device)
    else:
        y = torch.rand(5, 5, device=device)
        x = torch.zeros_like(y, device=device)
    
    dist.broadcast(x, src=0)  # Broadcast tensor from rank 0 to all ranks
    dist.broadcast(y, src=1)  # Broadcast tensor from rank 1 to all ranks

    ans = x * y
    if rank == 0:
        ans.neg_()

    dist.all_reduce(ans, op=dist.ReduceOp.SUM)

    # check ans if equal zeros
    if torch.all(ans == 0):
        print(f"Rank {rank} successfully computed the result.")
    else:
        print(f"Rank {rank} computed an incorrect result:\n{ans}") 

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