"""
Import the modules necessary to create a dataset and to use preloaded image datasets. 

Create a dataset using the "MNIST" preloaded dataset. 

Download the dataset to a directory called "mnist".
"""
from torch.utils.data import DataLoader
import torchvision.datasets
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets

datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",  # Official PyTorch mirror
]

# Create a dataset using the preloaded MNIST dataset
mnist_dataset = torchvision.datasets.MNIST(
    root='mnist',
    train=True,
    download=True,
    transform=ToTensor()
)