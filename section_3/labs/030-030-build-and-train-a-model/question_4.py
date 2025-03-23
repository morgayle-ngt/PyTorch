"""
Define a loss function. 

Import the module and create an instance of a loss function called “criterion” using Binary Cross Entropy Loss.

"""
# question_4.py
# Import neural network module from PyTorch
import torch.nn as nn

# Define Binary Cross Entropy loss function
criterion = nn.BCELoss()