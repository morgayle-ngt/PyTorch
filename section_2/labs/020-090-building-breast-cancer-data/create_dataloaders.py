"""
Create PyTorch DataLoaders for the train_dataset and the  val_dataset. 

This will define how the data is passed to the model during training and is the last step before you train the model. 

The train_loader should take in the train_dataset with a batch size of 64 and should shuffle. 

The val_loader should take the val_dataset with a batch size of 32 and should not shuffle. 
"""
from torch.utils.data import DataLoader
from create_datasets import train_dataset, val_dataset

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print the DataLoaders
print("train_loader batch size:", train_loader.batch_size)
print("val_loader batch size:", val_loader.batch_size)