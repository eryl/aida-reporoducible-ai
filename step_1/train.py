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
from torchvision.transforms.v2 import AutoAugment, Compose
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
    

dataset = Path("data") / "oxfordiii"

files = list(dataset.glob('**/*.jpg'))


# Images are structured like Breedname_number.jpg
#All images with 1st letter as captial are cat images
#images with small first letter are dog images
def get_label(file_path: Path):
    # Check if first letter is a capital letter
    if file_path.name[0].isupper():
        label = 'cat'
    else:
        label = 'dog'
    return label

    
labels = [get_label(f) for f in files]

indices = np.arange(len(files))

modeling_split_indices, test_indices,  = train_test_split(indices, test_size=0.1, stratify=labels)
modeling_labels = [labels[i] for i in modeling_split_indices]
train_indices, dev_indices = train_test_split(modeling_split_indices, test_size=0.1, stratify=modeling_labels)

test_images = [files[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]
dev_images = [files[i] for i in dev_indices]
dev_labels = [labels[i] for i in dev_indices]
train_images = [files[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]

with open("data_split_indices.json", 'w') as fp:
    # JSON can't serialize Path objects, let's make them strings
    data_splits = dict(test_images=[str(p) for p in test_images], 
                       test_labels=test_labels, 
                       dev_images=[str(p) for p in dev_images], 
                       dev_labels=dev_labels, 
                       train_images=[str(p) for p in train_images], 
                       train_labels=train_labels)
    json.dump(data_splits, fp, sort_keys=True, indent=2)

num_classes = 2

weights = DenseNet121_Weights.DEFAULT
model = densenet121(weights)
model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 1)) # We set the dimension to 1 since we'll use a sigmoid output

training_transforms = nn.Sequential(AutoAugment(), weights.transforms())
dev_transforms = weights.transforms()

training_dataset = ImageDataset(train_images, train_labels, transforms=training_transforms)
dev_dataset = ImageDataset(dev_images, dev_labels, transforms=dev_transforms)

device = torch.device('cuda')
model.to(device)


### Start by only adjusting the newly added layers, keeping the rest frozen

max_epochs = 5

# Set requires_grad to False everywhere first
for param in model.parameters():
    param.requires_grad = False

# Now just unfreeze the parameters of the classifier head    
parameters_to_train = list(model.classifier.parameters())
for param in parameters_to_train:
    param.requires_grad = True

loss_fn = nn.BCEWithLogitsLoss()
# Only optimize the classification head
optimizer = AdamW(parameters_to_train, lr=0.003, weight_decay=1e-5)

training_dataloader = DataLoader(training_dataset, 
                                 batch_size=32, 
                                 drop_last=True, #When training it can actually be a good idea to drop the last batch. If you're using batch normalization somewhere the training can break if there's just a single example in the batch 
                                 shuffle=True,
                                 num_workers=6
                                 )

dev_dataloader = DataLoader(dev_dataset, 
                                 batch_size=32, 
                                 drop_last=False,
                                 shuffle=False,
                                 num_workers=6
                                 )

best_model = None
best_loss = float('inf')

early_stoping_patience = 3
epochs_of_no_progress = 0
for epoch in trange(max_epochs, desc='epoch'):
    model.train()
    for training_batch in tqdm(training_dataloader, desc='training batch', leave=False):
        optimizer.zero_grad()
        x, y = training_batch
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        dev_losses = 0
        dev_samples = 0
        for dev_batch in tqdm(dev_dataloader, desc='dev batch', leave=False):
            x, y = dev_batch
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            n_samples = len(x)
            dev_losses += loss.item()*n_samples
            # A minor detail, but since we're not dropping any batches we 
            # can't just take the mean of all the losses over all batches. 
            # This would slightly overweight the last batch if it's smaller than the others.
            dev_samples += n_samples
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
    
#### Now train the whole model, using the best one during the frozen training              
model = best_model

for param in model.parameters():
    param.requires_grad = True

loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=0.000001, weight_decay=1e-5)

training_dataloader = DataLoader(training_dataset, 
                                 batch_size=32, 
                                 drop_last=True, #When training it can actually be a good idea to drop the last batch. If you're using batch normalization somewhere the training can break if there's just a single example in the batch 
                                 shuffle=True,
                                 num_workers=6
                                 )

dev_dataloader = DataLoader(dev_dataset, 
                            batch_size=32, 
                            drop_last=False,
                            shuffle=False,
                            num_workers=6
                            )

best_model = None
best_loss = float('inf')

max_epochs = 50
early_stoping_patience = 5
epochs_of_no_progress = 0
for epoch in trange(max_epochs, desc='epoch'):
    model.train()
    for training_batch in tqdm(training_dataloader, desc='training batch', leave=False):
        optimizer.zero_grad()
        x, y = training_batch
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        dev_losses = 0
        dev_samples = 0
        for dev_batch in tqdm(dev_dataloader, desc='dev batch', leave=False):
            x, y = dev_batch
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            n_samples = len(x)
            dev_losses += loss.item()*n_samples
            # A minor detail, but since we're not dropping any batches we 
            # can't just take the mean of all the losses over all batches. 
            # This would slightly overweight the last batch if it's smaller than the others.
            dev_samples += n_samples
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

                
