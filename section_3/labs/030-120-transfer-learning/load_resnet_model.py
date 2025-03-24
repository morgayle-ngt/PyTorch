"""
Use the `models.resnet18()` function to load the `resnet18` pre-trained model from torchvision with default weights.
"""
# Import the torchvision models module
from torchvision import models

# Load the ResNet18 model with pre-trained weights
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)