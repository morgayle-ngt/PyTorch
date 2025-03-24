"""
Set the last layer in the classifier layer to use 2 classes
Freeze all layers.
Unfreeze only the final layer.
"""

from load_mobilenet_v3 import model
import torch.nn as nn


# Modify last layer of the model for 2 classes
model.classifier[-1] = nn.Linear(1280, 2)


# Freeze all layers by setting requires_grad to False
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last layer by setting requires_grad to True
for param in model.classifier[-1].parameters():
    param.requires_grad = True