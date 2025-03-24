"""
Using pre-trained models from torchvision, load resnet18 with default Resnet18 weghts.

Set output layer to 2 classes.

Load our fine tuned model from a checkpoint.

Load the model parameters from the checkpoint. 
"""
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet18 and modify the final layer to output 2 classes
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load checkpoint and update model parameters
checkpoint = torch.load('resnet_checkpoint.tar')
model.load_state_dict(checkpoint['model_state_dict'])