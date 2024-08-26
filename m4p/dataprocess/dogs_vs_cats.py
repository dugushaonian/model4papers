#!/usr/bin/env python
# coding=utf-8
"""
https://www.kaggle.com/competitions/dogs-vs-cats/data
"""

import os
import glob

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class DogsVsCatsDataSet(Dataset):
    def __init__(self, file_list, transform=None):
        super().__init__()
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == 'dog' else 0

        return img_transformed, label


def load_train_valid_set(train_path):
    train_list = glob.glob(os.path.join(train_path,'*.jpg'))
    labels = [path.split('/')[-1].split('.')[0] for path in train_list]

    train_list, valid_list = train_test_split(train_list, 
                                              test_size=0.2,
                                              stratify=labels,
                                              random_state=42)
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    train_data = DogsVsCatsDataSet(train_list, train_transforms)
    val_data = DogsVsCatsDataSet(valid_list, val_transforms)
    return train_data, val_data


def load_test_set(test_path):
    test_list = glob.glob(os.path.join(test_path, '*.jpg'))
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    test_data = DogsVsCatsDataSet(test_list, test_transforms)
    return test_data


def dogs_vs_cats_dataloader(train_path, test_path, batch_size):
    train_data, valid_data = load_train_valid_set(train_path)
    test_data = load_test_set(test_path)
    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader
