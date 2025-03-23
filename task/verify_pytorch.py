#!/usr/bin/python3
# File: ~/task/verify_pytorch.py

import torch

# Print PyTorch version
print(torch.__version__)

# Check CUDA support
print(torch.cuda.is_available())

# Create and print a random tensor
print(torch.rand(2, 4))