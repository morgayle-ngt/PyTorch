import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained MobileNet_V3_Large and modify its classifier for 2 classes
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

# Modify the final classifier layer to output 2 classes
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)

# Load checkpoint and update model parameters
checkpoint = torch.load('mobilenet_checkpoint.tar')

# Load the parameters from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# Print the model architecture
print(model)