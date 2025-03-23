"""
Define `train_transform` and `val_transform` pipelines. 

For the `train_transform` we need the following in our pipeline in the following order: 
Resize of 128 x 128 pixels
Set to Grayscale
30 degrees random rotation
50% chance of a random horizontal flip
converted to a tensor
normalized mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225) for 3 channels

For the `val_transform` we need the following: 
Resize of 128 x 128 pixels
Set to Grayscale
converted to a tensor
normalized mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225) for 3 channels

Use V2 API
"""
from torchvision.transforms import v2
import torch

train_transform = v2.Compose([
    v2.Resize((128, 128)), # Resize to 128 x 128 pixels
    v2.Grayscale(num_output_channels=3), # Convert to grayscale but keep 3 channels
    v2.RandomRotation(30), # Random rotation up to 30 degrees
    v2.RandomHorizontalFlip(0.5), # Random horizontal flip with 50% probability
  v2.ToImage(), # Convert to Image tensor
    v2.ToDtype(torch.float32, scale=True), # Convert to float32 and scale to [0, 1]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with mean and std
])

val_transform = v2.Compose([
    v2.Resize((128, 128)), # Resize to 128 x 128 pixels
    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
    v2.ToImage(), # Convert to tensor
    v2.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale to [0, 1]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with mean and std
])