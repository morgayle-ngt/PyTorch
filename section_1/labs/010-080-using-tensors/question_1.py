"""
Using the tensor function, create a two dimensional tensor using two lists and print the size. 

NOTE: Each list should consist of 4 values.
"""
import torch 

# Create a 2D tensor using two lists
tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

# Print the size of the tensor
print(tensor.size())
