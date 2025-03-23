# Import the transformation module in the first way using version 2 full path
import torchvision.transforms.v2
# Import the transformation module in the second way using alias for version 2
from torchvision.transforms import v2

# Print both imported objects to validate
print(torchvision.transforms.v2)
print(v2)