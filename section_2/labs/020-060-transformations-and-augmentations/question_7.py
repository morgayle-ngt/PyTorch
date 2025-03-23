from torchvision.transforms import v2
from PIL import Image

# Load the image into memory
image = Image.open("images/dog/dog-5.jpg")

# Create the Pipeline transform
pipeline = v2.Compose([
   v2.Resize(size=(100, 100)),
   v2.RandomHorizontalFlip(p=0.5),
   v2.RandomPhotometricDistort(
       contrast=(0.7, 1.2))
])

# Apply the pipeline
pipeline_image = pipeline(image)

# Print the pipeline transformed object
print(pipeline_image)