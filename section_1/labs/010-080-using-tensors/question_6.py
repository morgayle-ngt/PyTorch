"""
Given the created tensor, print the second row and the second value of the row.
"""
import torch

# Create a 2D tensor
tensor = torch.tensor([[100, 200, 300, 400], [50, 50, 60, 60], [500, 600, 700, 800]])

# Print the second row of the tensor
print(tensor[1])

# Print the second value of the second row
print(tensor[1, 1])
