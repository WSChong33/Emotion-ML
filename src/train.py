# Script to train and test a neural network model

import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_preparation import get_data_loaders
from model import EmotionCNN

def train_and_test_model(data_dir, num_epochs=25, batch_size=32, learning_rate=0.001):
    train_loader, test_loader = get_data_loaders(data_dir, batch_size)

    model = EmotionCNN()
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizer

    for epoch in range(num_epochs):

        # Training
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Training Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy after Epoch [{epoch+1}/{num_epochs}]: {accuracy:.2f}%')

    torch.save(model.state_dict(), 'emotion_cnn.pth')

if __name__ == '__main__':
    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, 'dataset')
    train_and_test_model(data_dir=data_dir)
