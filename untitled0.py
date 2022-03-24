#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:46:17 2022

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

#hyperparameters
# input_size = 1024
batch_size1 = 64
num_classes = 10
learning_rate = 0.01
num_epochs = 1



train_dataset1 = datasets.CIFAR10(root="dataset/", train=True,transform=transforms.ToTensor(),download=True)
train_loader1 = DataLoader(train_dataset1, batch_size1, shuffle=True)
test_dataset1 = datasets.CIFAR10(root="dataset/", train=False,transform=transforms.ToTensor(),download=True)
test_loader1 = DataLoader(test_dataset1, batch_size1, shuffle=True)



device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    if num_epochs == 5:
        checkpoint = {"model" : model.state_dict() , "optimizer" : optimizer.state_dict()}
        torch.save(checkpoint, "checkpoints.pth.tar")
        
    for batch_idx, (data, targets) in enumerate(train_loader1):
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

check_accuracy(train_loader1, model)
check_accuracy(test_loader1, model)
    



