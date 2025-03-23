"""
Define an optimizer. 

Import the module for optimizers and create using the Adam optimizer using our model weights and a learning rate of 0.001. 

"""
# question_5.py
# Import model from question_2
from question_2 import model
# Import optimization module from PyTorch
import torch.optim as optim

# Create Adam optimizer with learning rate 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)