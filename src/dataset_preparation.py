# Preparing data

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Function to set up data loaders
def get_data_loaders(data_dir, batch_size=32):

    # Tranform each image
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Tensor normalised to have mean of 0.5 and std dev of 0.5 - [-1, 1] range
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform) # Load train dataset
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform) # Load test dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Initialise a DataLoader object
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Iniitialise a DataLoader object

    return train_loader, test_loader
