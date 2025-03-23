"""
Create a CSV file named full_image_data.csv that contains 2 columns: file_path and label.

The file_path should reference the path to each image, and label should be derived from the directory name.
"""
import os
import glob
import pandas as pd

# Initialize an empty list to collect file path and label data
data = []

# Iterate over all .jpg files located in subdirectories under the "data" directory
for file_path in glob.glob("data/*/*.jpg"):
    # Extract the label from the directory name in which the image file is located
    label = os.path.basename(os.path.dirname(file_path))
    # Append a dictionary with the file path and corresponding label to the data list
    data.append({"file_path": file_path, "label": label})

# Create a pandas DataFrame and save it to a CSV file
df = pd.DataFrame(data)
df.to_csv("full_image_data.csv", index=False)