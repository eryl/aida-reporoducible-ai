from pathlib import Path
from copy import deepcopy
import json

from tqdm import tqdm, trange

from sklearn.model_selection import train_test_split

import numpy as np

import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from torchvision.models.densenet import DenseNet121_Weights, densenet121
from torchvision.transforms.v2 import AutoAugment

dataset = Path("datasets") / "oxfordIII"

files = dataset.glob('*.png')

labels = []

indices = np.arange(len(files))

test_split_indices, modeling_split_indices = train_test_split(indices, test_size=0.1, stratify=labels)
modeling_labels = labels[modeling_split_indices]
dev_indices, train_indices = train_test_split(modeling_split_indices, test_size=0.1, stratify=modeling_labels)

test_images = [files[i] for i in test_split_indices]
test_labels = [labels[i] for i in test_split_indices]
dev_images = [files[i] for i in dev_indices]
dev_labels = [labels[i] for i in dev_indices]
train_images = [files[i] for i in train_indices]
train_labels = [files[i] for i in train_indices]

with open("data_split_indices.json", 'w') as fp:
    data_splits = dict(test_images=test_images, test_labels=test_labels, dev_images=dev_images, dev_labels=dev_labels, train_images=train_images, train_labels=train_labels)
    json.dump(data_splits, sort_keys=True, indent=2)

num_classes = 2

weights = DenseNet121_Weights.DEFAULT
model = densenet121(weights)
model.classifier = nn.Sequential(nn.Linear(model.classifier.output_dim, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 1)) # We set the dimension to 1 since we'll use a sigmoid output


training_dataset = ImageDataset(train_images, train_labels)
dev_dataset = ImageDataset(dev_images, dev_labels)


max_epochs = 50

device = torch
loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)

training_dataloader = DataLoader(training_dataset, 
                                 batch_size=16, 
                                 drop_last=True, #When training it can actually be a good idea to drop the last batch. If you're using batch normalization somewhere the training can break if there's just a single example in the batch 
                                 shuffle=True
                                 )

dev_dataloader = DataLoader(dev_dataset, 
                                 batch_size=16, 
                                 drop_last=False,
                                 shuffle=False
                                 )

best_model = None
best_loss = float('inf')

early_stoping_patience = 20
epochs_of_no_progress = 0
for epoch in trange(max_epochs, desc='epoch'):
    model.train()
    for training_batch in tqdm(training_dataloader, desc='training batch', leave=False):
        optimizer.zero_grad()
        x, y = training_batch
        x = x.to(device)
        y = y.to(device)
        prediction = model(training_batch)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        dev_losses = 0
        dev_samples = 0
        for dev_batch in tqdm(dev_dataloader, desc='dev batch', leave=False):
            x, y = training_batch
            x = x.to(device)
            y = y.to(device)
            prediction = model(training_batch)
            loss = loss_fn(prediction, y)
            dev_losses += loss.item()
            # A minor detail, but since we're not dropping any batches we 
            # can't just take the mean of all the losses over all batches. 
            # This would slightly overweight the last batch if it's smaller than the others.
            dev_samples += len(x)  
        mean_dev_performance = dev_losses / dev_samples
        print(f'dev loss: {mean_dev_performance}')  
        
        if mean_dev_performance < best_loss:
            epochs_of_no_progress = 0
            best_loss = mean_dev_performance
            best_model = deepcopy(model)
            model_path = Path('models') / 'best_model.pth'
            model_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), model_path)
        else:
            epochs_of_no_progress += 1
        
    if epochs_of_no_progress >= early_stoping_patience:
        print("Patience has run out, early stopping")
        break
        
print("Training is done")

                
