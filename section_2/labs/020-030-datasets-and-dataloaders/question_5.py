"""
Create a dataloader from our dataset named cd_dataset called cd_dataloader and then iterate through a batch and print the features and labels shape. 

When creating the dataloader, set the size of the batch to 4.  
"""
from question_3 import cd_dataset
from torch.utils.data import DataLoader

# Create the DataLoader
cd_dataloader = DataLoader(dataset=cd_dataset, batch_size=4, shuffle=True)

# Iterate through a batch
features, labels, urls = next(iter(cd_dataloader))
print(features.shape)
print(labels.shape)