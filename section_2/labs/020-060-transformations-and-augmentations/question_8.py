import torchvision.datasets
from torchvision import transforms

# Set the MNIST mirror to an official source
torchvision.datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/"  # Official PyTorch mirror
]

# Create the pipeline transform
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create the dataset
mnist_ds = torchvision.datasets.MNIST(
    root='mnist',
    train=False,
    download=True,
    transform=transform
)

# Print the transformed dataset object
print(mnist_ds)