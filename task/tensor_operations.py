import torch

# Create input tensors
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])

# Perform operations
sum_tensor = tensor_a + tensor_b          # Element-wise addition
product_tensor = tensor_a * tensor_b      # Element-wise multiplication
concat_tensor = torch.cat((tensor_a, tensor_b), dim=0)  # Vertical concatenation

# Print results
print(f"Sum:\n{sum_tensor}")
print(f"Product:\n{product_tensor}")
print(f"Concatenated:\n{concat_tensor}")