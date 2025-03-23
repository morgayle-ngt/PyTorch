import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load and convert image to tensor
img = mpimg.imread("/workspaces/PyTorch/task/images/image.jpg")
image_tensor = torch.from_numpy(img)

# Create visualization
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)

# Red channel visualization
plt.subplot(1, 2, 2)
plt.title("Red Channel")
plt.imshow(image_tensor[:, :, 0], cmap="Reds")

# Save the visualization
#plt.savefig("/root/task/tensor_visualization.png")
#plt.savefig("task/tensor_visualization.png")
plt.savefig("/workspaces/PyTorch/task/tensor_visualization.png")
plt.close()

# Print tensor information
print(f"Image tensor shape: {image_tensor.size()}")
print(f"Image tensor dtype: {image_tensor.dtype}")