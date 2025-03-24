"""
Create a DataLoader named 'test_loader' with a batch size of 64 and with shuffling disabled.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import v2

# Define the custom dataset (Dataset Class)
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = lambda y: target_transform[y]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label

# Label encoding for the dataset
label_encoding = {"malignant": 0, "benign": 1}

# Define image transformations for the test data
test_transform = v2.Compose([
    v2.Resize((128, 128)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

# Determine the current working directory (for locating test_data.csv)
work_dir = os.path.dirname(os.path.abspath(__file__))

# Create the test dataset instance
test_dataset = CustomImageDataset(
    annotations_file=os.path.join(work_dir, 'test_data.csv'),
    img_dir="./",
    transform=test_transform,
    target_transform=label_encoding
)

# Construct the DataLoader with a batch size of 64 and without shuffling
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)