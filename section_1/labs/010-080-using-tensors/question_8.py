"""
Using the image module from Pillow, open the image into memory. 

Using the transforms module from torchvision, create a transformation that converts the image into a PyTorch tensor. 

Finally create a tensor from the transformation and print the size and device attributes. 
"""
from PIL import Image
from torchvision import transforms

# Load the image into memory
img = Image.open("/workspaces/PyTorch/images/pytorch-logo.png")

# Create a transformation
transform = transforms.ToTensor()

# Transform our image into a tensor
tensor = transform(img)

# Print the size and device attributes of the tensor
print(tensor.size(), tensor.device)
