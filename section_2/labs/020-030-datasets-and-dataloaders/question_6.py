"""
Create a new dataset called cd_dataset using the directory structure as classes. Use the images directory.
"""
import torchvision

# Transformation for dataset
transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Create dataset from directory structure
cd_dataset = torchvision.datasets.ImageFolder(root='images', transform=transformations)

# Print the dataset
print(cd_dataset)