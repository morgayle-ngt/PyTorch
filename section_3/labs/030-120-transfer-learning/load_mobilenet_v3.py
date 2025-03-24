"""
Load the `mobilenet_v3_large` pre-trained model from `v0.10.0` of the PyTorch vision Github repo.
Be sure to load the pre-trained parameters.
"""
from torch import hub


# Load Model from the hub
model = hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_large', pretrained=True)