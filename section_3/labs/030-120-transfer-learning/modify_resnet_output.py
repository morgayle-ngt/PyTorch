"""
Modify the `resnet18` model's output layer to classify 2 classes instead of 1000 classes.
"""
# Import the pre-loaded ResNet18 model
from load_resnet_model import model

# Import torch.nn to modify the fully connected (fc) layer
import torch.nn as nn

# Update the fully connected (fc) layer to output 2 classes
model.fc = nn.Linear(512, 2)  # 512 is the input size for ResNet18's fc layer