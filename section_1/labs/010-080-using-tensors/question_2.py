"""
Randomly create a one dimension tensor that consists of 5 values and print its size.
"""
import torch

# Create a random 1D tensor with 5 values
tensor = torch.rand(5)

# Print the size of the tensor
print(tensor.size())
