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

def path1(x, iterations=10):
    for _ in range(iterations):
        x = x * 2
    return x

def path2(x, iterations=5):
    for _ in range(iterations):
        x = x + 2
    return x

def worker(rank, world_size, params):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{params[rank]['device']}")

    x = torch.randn(5, 5, device=device)
    y = x.clone()  # Clone x to y for the second path

    if rank == 0:
        x = path1(x, iterations=params[rank]['iterations'])
    else:
        y = path2(y, iterations=params[rank]['iterations'])
    
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
        {"device": 1, "iterations": 10},
        {"device": 0, "iterations": 5}
    ]
    mp.spawn(worker, args=(world_size,params), nprocs=world_size, join=True)
    print("Test completed.")