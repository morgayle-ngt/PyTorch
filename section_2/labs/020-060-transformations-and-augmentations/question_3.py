from torchvision import transforms
from PIL import Image

# Load the image into memory from the specified path
image = Image.open("images/dog/dog-1.jpg")

# Create the resize transform with target dimensions
resize_transform = transforms.Resize((250, 300))

# Apply the resize transform to the image
resized_image = resize_transform(image)

# Print the resized image object to validate
print(resized_image)