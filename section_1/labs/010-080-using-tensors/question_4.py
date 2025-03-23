"""
Create a 3 x 3 x 5 tensor consisting of all values of 0. 

Print the size and the device attributes.
"""
import torch

# Create a 3x3x5 tensor filled with zeros
tensor = torch.zeros(3, 3, 5)

# Print the size and device of the tensor
print(tensor.shape, tensor.device)

