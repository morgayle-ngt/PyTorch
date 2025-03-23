from torchvision.transforms import v2
from PIL import Image

# Load the image to memory
img = Image.open("images/dog/dog-1.jpg")

# Normalize() does not support PIL images; convert to tensor first
tensor_transform = v2.ToTensor()
tensor_img = tensor_transform(img)

# Normalize the image
normalize_transform = v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
normalize_img = normalize_transform(tensor_img)

# Randomly resize the image
rand_resize_transform = v2.RandomResize(min_size=50, max_size=300)
rand_resize_img = rand_resize_transform(normalize_img)

# Print the randomly resized tensor object
print(rand_resize_img)