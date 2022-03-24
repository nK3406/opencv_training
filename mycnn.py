#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:29:41 2022


CONV NETWORK

@author: zgn
"""

import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice and easy way
import torchvision.transforms as transforms # Transformations we can perform on our dataset
import torchvision


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size = (3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(8, 16, (3,3),(1,1))
        self.conv3 = nn.Conv2d(16, 64, (3,3), (1,1))
        self.pool = nn.MaxPool2d((2,2),(1,1))
        self.fc1 = nn.Linear(64*19*19, 10) #19 olmasının sebebi, her conv katmandan gectikten sonra 2,her pool katmanından gectikten sonra 1 boyut kücülmesi. 
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv3(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = x.reshape(x.shape[0],-1)
        x= self.fc1(x)
        return x
    
    
batch_size = 64
# input_size = 784 # 28x28 = 784, size of MNIST images (grayscale)
# num_classes = 10
learning_rate = 0.001
num_epochs = 3        
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

 
train_dataset = datasets.MNIST( 
    root="dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_dataset = datasets.MNIST(
    root="dataset/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True
) 


for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy"
            f" {float(num_correct) / float(num_samples) * 100:.2f}"
        )

model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


    