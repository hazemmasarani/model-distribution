import torch

# Create input_ids on CPU first
input_ids = torch.randint(
    low=0,
    high=50280,  # Assuming vocab size of 50280, adjust if needed
    size=(2, 128)
)


torch.save(input_ids, "../model6/input_ids.pt")