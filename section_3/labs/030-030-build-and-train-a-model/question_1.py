"""
Create a Class to build a Neural Network in a PyTorch manner. 

Begin by importing the Neural Network module from PyTorch and name your Class “Neural Network”. 

Then define your layers as follows: 1) A 2D convolutional layer, 2) a 2D max pooling layer and 3) a fully connected layer. 

Once layers have been defined, then define the flow through the layers as follows: 1) pass through conv layer with ReLU activation, then apply max pooling, 2) flatten the output from the convolutional layers, and then 3) pass through fully connected layer with sigmoid.
"""
# Import required PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a Neural Network class that inherits from nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):
        # Initialize the parent class
        super(NeuralNetwork, self).__init__()
        # Define convolutional layer with 3 input channels, 16 output channels and kernel size 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        # Define max pooling layer with kernel size 2 and stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Define fully connected layer with input size 16*16*16 and output size 1
        self.fc1 = nn.Linear(16 * 16 * 16, 1)

    def forward(self, x):
        # Apply convolution, ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 16 * 16 * 16)
        # Apply sigmoid activation to the output
        x = torch.sigmoid(self.fc1(x))
        return x