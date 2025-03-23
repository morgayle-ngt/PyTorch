"""
If a GPU device exists on the machine, set the device to use it. If a GPU is not available the set the device to use the CPU.
"""
import torch

# Check if GPU is available to use
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

# Create a tensor and the device configured above
tensor = torch.tensor([4, 90, 90], device=device)

# Print the device of the tensor
print(tensor.device)
