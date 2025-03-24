"""
Set the optimizer to only update the parameters on the final layer in classifier.


Create a `ExponentialLR` scheduler decays the learning rate of the optimizer by 0.1 every epoch.
"""
from freeze_mobilenet_v3 import model
# Import module
import torch.nn as nn
import torch.optim as optim


# Create Loss function
criterion = nn.CrossEntropyLoss()
# Create Optimizer
optimizer = optim.SGD(model.classifier[-1].parameters(), lr=0.001, momentum=0.9)
# Create Scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)