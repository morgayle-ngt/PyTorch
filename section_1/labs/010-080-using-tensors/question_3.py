"""
Randomly create a 3 x 5 x 10 tensor and print its size and dtype attributes
"""
import torch

# Create a random 3x5x10 tensor
tensor = torch.rand(3, 5, 10)

# Print the size and dtype of the tensor
print(tensor.shape, tensor.dtype)
