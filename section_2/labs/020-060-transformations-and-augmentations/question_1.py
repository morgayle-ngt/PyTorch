# Import the transformation module in the first way using full path
import torchvision.transforms

# Import the transformation module in the second way using alias
from torchvision import transforms

# Print both imported objects to validate
print(torchvision.transforms)
print(transforms)