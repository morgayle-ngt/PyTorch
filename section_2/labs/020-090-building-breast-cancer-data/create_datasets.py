"""
Using the custom PyTorch Dataset Class provided, create a training dataset called train_dataset and a validation dataset called val_dataset. 

For each use the proper transformations and the proper annotations file. 

Also be sure to create the label encoding for our 2 labels (benign and malignant) and pass the label_encoder as the target_transform.
"""
from create_transformations import train_transform, val_transform
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageBreastCancerDataSet(Dataset):
    def __init__(self, annotations_file, image_dir, transform, target_transform):
        self.image_labels = pd.read_csv(annotations_file)
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = lambda y: target_transform[y]

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        # Get the image path and label for the given index
        image_path = os.path.join(self.image_dir, self.image_labels.iloc[idx, 0])
        # Open the image file
        image = Image.open(image_path)
        # Get the label for the given index
        label = self.image_labels.iloc[idx, 1]
        # Apply transformations to the image
        image = self.transform(image)
        # Apply label encoding to the label
        label = self.target_transform(label)
        # Return the image and label as a tuple
        return image, label

label_encoding = {'benign': 0, 'malignant': 1}

# Create training dataset
train_dataset = CustomImageBreastCancerDataSet(
    annotations_file='training_data.csv',
    image_dir='data',
    transform=train_transform,
    target_transform=label_encoding
)

# Create validation dataset
val_dataset = CustomImageBreastCancerDataSet(
    annotations_file='validation_data.csv',
    image_dir='data',
    transform=val_transform,
    target_transform=label_encoding
)

# print the lengths of the training and validation datasets
if __name__ == "__main__":
    print("Length of train_dataset:", len(train_dataset))
    print("Length of val_dataset:", len(val_dataset))