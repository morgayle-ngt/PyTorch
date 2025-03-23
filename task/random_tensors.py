import torch

# Create tensors with different initializations
random_tensor = torch.rand(3, 4, 5)  # Random values between 0 and 1
zeros_tensor = torch.zeros(2, 3)     # All zeros
ones_tensor = torch.ones(2, 3)       # All ones

# Print tensor information
print(f"Random tensor shape: {random_tensor.size()}")
print(f"Zeros tensor shape: {zeros_tensor.size()}")
print(f"Ones tensor shape: {ones_tensor.size()}")

# Print first elements
print(f"First element of random tensor: {random_tensor[0,0,0]}")
print(f"First element of zeros tensor: {zeros_tensor[0,0]}")
print(f"First element of ones tensor: {ones_tensor[0,0]}")