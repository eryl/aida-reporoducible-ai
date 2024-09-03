from pathlib import Path
from copy import deepcopy
import json

from tqdm import tqdm, trange

from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from torchvision.datasets import oxford_iiit_pet
from torchvision.models.densenet import DenseNet121_Weights, densenet121
from torchvision.io import read_image, ImageReadMode


class ImageDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms
        
    def __getitem__(self, index):
        image_path = self.images[index]
        image = read_image(image_path, mode=ImageReadMode.RGB)
        if self.transforms is not None:
            image = self.transforms(image)
        label_string = self.labels[index]
        float_label = 1. if label_string == 'cat' else 0.
        label = torch.tensor([float_label])
        return image, label
    
    def __len__(self):
        return len(self.images)


with open('data_split_indices.json') as fp:
    data_splits = json.load(fp)

test_images = [Path(s) for s in data_splits['test_images']]
test_labels = data_splits['test_labels']


weights = DenseNet121_Weights.DEFAULT
model = densenet121(weights)
model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 1)) # We set the dimension to 1 since we'll use a sigmoid output

model.load_state_dict(torch.load('models/best_model.pth', weights_only=True))

test_dataset = ImageDataset(test_images, test_labels, transforms=weights.transforms())

device = torch.device('cuda')
loss_fn = nn.BCEWithLogitsLoss()

model.to(device)


