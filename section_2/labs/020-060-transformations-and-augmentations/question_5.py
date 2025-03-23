from torchvision.transforms import v2
from PIL import Image

# Load image into memory
image = Image.open("images/cat/cat-3.jpg")

# Transform the image to a tensor
tensor_transform = v2.ToTensor()
tensor_image = tensor_transform(image)

# Transform the tensor image by random crop
random_crop_transform = v2.RandomCrop((50, 200))
random_crop_image = random_crop_transform(tensor_image)

# Print the randomly cropped tensor object
print(random_crop_image)