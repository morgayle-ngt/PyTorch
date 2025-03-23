from initial_dataset import initial_dataset
from torch.utils.data import random_split

# Calculate the size of the training set (70% of the total dataset)
train_size = int(0.7 * len(initial_dataset))

# Calculate the size of the validation set (20% of the total dataset)
val_size = int(0.2 * len(initial_dataset))

# Calculate the size of the test set (remaining 10% of the dataset)
test_size = len(initial_dataset) - train_size - val_size

# Use random_split to divide the dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(initial_dataset, [train_size, val_size, test_size])