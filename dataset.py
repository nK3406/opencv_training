#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:55:39 2022

@author: zgn
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# # #Ignore warnings
# # import warnings
# # warnings.filterwarnings("ignore")

# # plt.ion()   # interactive mode

# # csv dosyasindaki bilgileri alıp dataframe olarak kaydetti (69,137)
# landmarks_frame = pd.read_csv('/home/zgn/Desktop/faces/face_landmarks.csv')
# n = 65
# #foto isimlerini veren kod. Yani 0. satır isimler oluyor.
# img_name = landmarks_frame.iloc[n, 0]
# print(img_name)
# #isimler olmadan alıyor(indexliyor) verileri
# landmarks = landmarks_frame.iloc[n, 1:]
# print(landmarks)
# #tek boyutlu numpy arrayine donusturuyor
# landmarks = np.asarray(landmarks)
# print(landmarks)
# # 2.boyuta yani noktasal forma ceviriyor arrayin icindekileri.
# landmarks = landmarks.astype('float').reshape(-1, 2)
# print('Image name: {}'.format(img_name))
# print('Landmarks shape: {}'.format(landmarks.shape))
# print('First 4 Landmarks: {}'.format(landmarks[:4]))

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

# plt.figure()
# show_landmarks(io.imread(os.path.join('/home/zgn/Desktop/faces', img_name)),
#                 landmarks)
# plt.show()

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset = FaceLandmarksDataset(csv_file='/home/zgn/Desktop/faces/face_landmarks.csv',
                                    root_dir='/home/zgn/Desktop/faces')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break
    