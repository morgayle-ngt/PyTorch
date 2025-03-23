"""
Given the created tensor, print the first row of values.
"""
import torch

# Create a 2D tensor
tensor = torch.tensor([[100, 200, 300, 400], [50, 50, 60, 60], [500, 600, 700, 800]])

# Print the first row of the tensor
print(tensor[0])
