# Model for neural network

import torch.nn as nn
import torch.nn.functional as F

# Convolutional Neural Network
class EmotionCNN(nn.Module): # nn.Module is base class for neural network module

    def __init__(self):

        super(EmotionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)  # Assuming 7 classes for 7 emotions

    # Forward pass
    def forward(self, x):

        # First layer + ReLu Activation Function + Max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten tensor
        x = x.view(-1, 128 * 6 * 6)

        # Obtain logits for classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
