from torchvision.transforms import v2
from PIL import Image

# Load the image into memory from the specified path
image = Image.open("images/dog/dog-1.jpg")

# Create the random horizontal flip transform with a 75% probability
transform = v2.RandomHorizontalFlip(p=0.75)

# Apply the flip transform to the image
rhf_image = transform(image)

# Print the flipped image object to validate
print(rhf_image)