"""
Create a custom PyTorch Dataset called initial_dataset that contains the initial data from our annotations file full_image_data.csv. 

Once again we will use Pandas to read in our annotations file. 

We will then return each index with image_path and label.
"""
# Import required libraries
import pandas as pd  # For reading and handling CSV data
from torch.utils.data import Dataset  # Base Dataset class from PyTorch

# Custom Dataset class for breast cancer images
class BreastCancerDataset(Dataset):
    def __init__(self, annotations_file):
        # Read the annotations file into a pandas DataFrame
        self.image_labels = pd.read_csv(annotations_file)

    def __len__(self):
        # Return the number of rows in the annotations file
        return len(self.image_labels)

    def __getitem__(self, idx):
        # Get the image path from first column and label from second column
        image_path = self.image_labels.iloc[idx, 0]
        label = self.image_labels.iloc[idx, 1]
        return image_path, label

# Create an instance of the dataset
initial_dataset = BreastCancerDataset(annotations_file="full_image_data.csv")