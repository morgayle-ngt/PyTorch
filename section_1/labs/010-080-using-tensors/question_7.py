"""
Given the two created tensors, join them together on the second dimension (dim 1) and print the size.
"""
import torch

# Create two 2D tensors
tensor_x = torch.tensor([[9, 8, 7, 6], [5, 4, 3, 2]])
tensor_y = torch.tensor([[6, 7, 8, 9], [2, 3, 4, 5]])

# Concatenate tensors along the second dimension
joined_tensor = torch.cat((tensor_x, tensor_y), dim=1)

# Print the size of the resulting tensor
print(joined_tensor.size())
