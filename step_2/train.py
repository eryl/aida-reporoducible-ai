from pathlib import Path
from copy import deepcopy
import json

from tqdm import tqdm, trange

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np
import pandas as pd


import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from torchvision.models.densenet import DenseNet121_Weights, densenet121
from torchvision.transforms.v2 import AutoAugment, Compose
from torchvision.io import read_image, ImageReadMode

import mlflow

class ImageDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms
    
    def set_transforms(self, transforms):
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
    
    
def setup_data_splits():
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

    # with open("data_split_indices.json", 'w') as fp:
    #     # JSON can't serialize Path objects, let's make them strings
    #     json.dump(data_splits, fp, sort_keys=True, indent=2)

    data_splits = dict(test_images=[str(p) for p in test_images], 
                    test_labels=test_labels, 
                    dev_images=[str(p) for p in dev_images], 
                    dev_labels=dev_labels, 
                    train_images=[str(p) for p in train_images], 
                    train_labels=train_labels)
    
    mlflow.log_dict(data_splits, 'datas_splits.json')
    
    training_dataset = ImageDataset(train_images, train_labels)
    dev_dataset = ImageDataset(dev_images, dev_labels)
    test_dataset = ImageDataset(test_images, test_labels)
    
    return training_dataset, dev_dataset, test_dataset

def train_on_dataloader(model, training_dataloader, optimizer, loss_fn, device, iteration=0):
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
        mlflow.log_metric('training_loss', loss.item(), step=iteration)
        iteration += 1
    return iteration

def evaluate_on_dataloader(model, dataloader, loss_fn, device):
    with torch.no_grad():
        model.eval()
        losses = 0
        n_samples = 0
        logits = []
        probabilities = []
        labels = []
        
        for test_batch in tqdm(dataloader, desc='test batch', leave=False):
            x, y = test_batch
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            probs = torch.sigmoid(prediction)
            logits.append(prediction.cpu().numpy())
            labels.append(y.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            # A minor detail, but since we're not dropping any batches we 
            # can't just take the mean of all the losses over all batches. 
            # This would slightly overweight the last batch if it's smaller than the others.
            batch_n_samples = len(x)
            n_samples += batch_n_samples
            losses += loss.item()*batch_n_samples
            
        mean_test_performance = losses / n_samples
        logits = np.concatenate(logits).flatten()
        labels = np.concatenate(labels).flatten()
        probabilities = np.concatenate(probabilities).flatten()
        roc_auc = roc_auc_score(labels, logits)
        results = {'roc_auc': roc_auc, 'loss': mean_test_performance, 
                   'logits': logits, 'labels': labels, 
                   'probabilities': probabilities}
        return results
        

def train_model(model, training_dataloader, dev_dataloader, optimizer, loss_fn, device, max_epochs, iteration=0):
    best_model = model
    best_roc_auc = float('-inf')

    early_stopping_patience = 3
    epochs_of_no_progress = 0
    for epoch in trange(max_epochs, desc='epoch'):
        iteration = train_on_dataloader(model, training_dataloader, optimizer, loss_fn, device, iteration)
        evaluation_results = evaluate_on_dataloader(model, dev_dataloader, loss_fn, device)
        dev_roc_auc = evaluation_results['roc_auc']
        dev_loss = evaluation_results['loss']
        mlflow.log_metrics({'dev_loss': dev_loss, 'dev_roc_auc': dev_roc_auc}, step=iteration)
        
        if dev_roc_auc > best_roc_auc:
            epochs_of_no_progress = 0
            best_roc_auc = dev_roc_auc
            best_model = deepcopy(model)
            model_path = Path('models') / 'best_model.pth'
            model_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), model_path)
        else:
            epochs_of_no_progress += 1
        
        if epochs_of_no_progress >= early_stopping_patience:
            print("Patience has run out, early stopping")
            break
        
    #### Now train the whole model, using the best one during the frozen training              
    return best_model


def run_training(training_dataset: ImageDataset, dev_dataset: ImageDataset, test_dataset: ImageDataset, device):
    hidden_dim = 1024
    dropout_rate = 0.5
    warmup_learning_rate = 0.003
    warmup_weight_decay = 1e-5
    learning_rate = 1e-6
    weight_decay = 1e-5
    
    mlflow.log_params({'hidden_dim': hidden_dim,
                       'dropout_rate': dropout_rate,
                       'warmup_learning_rate': warmup_learning_rate,
                       'warmup_weight_decay': warmup_weight_decay,
                       'learning_rate': learning_rate,
                       'weight_decay': weight_decay
                       })
        
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights)
    model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1)) # We set the dimension to 1 since we'll use a sigmoid output

    training_transforms = nn.Sequential(AutoAugment(), weights.transforms())
    dev_transforms = weights.transforms()

    training_dataset.set_transforms(training_transforms)
    dev_dataset.set_transforms(dev_transforms)
    test_dataset.set_transforms(dev_transforms)

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
    model.to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()

    ### Start by only adjusting the newly added layers, keeping the rest frozen
    with mlflow.start_run(nested=True, tags={'warmup': True}):
        max_warmup_epochs = 1
        # Set requires_grad to False everywhere first
        for param in model.parameters():
            param.requires_grad = False
        # Now just unfreeze the parameters of the classifier head    
        parameters_to_train = list(model.classifier.parameters())
        for param in parameters_to_train:
            param.requires_grad = True
        # Only optimize the classification head
        optimizer = AdamW(parameters_to_train, lr=warmup_learning_rate, weight_decay=warmup_weight_decay)
        model = train_model(model=model, training_dataloader=training_dataloader, 
                    dev_dataloader=dev_dataloader, optimizer=optimizer, loss_fn=loss_fn, device=device,
                    max_epochs=max_warmup_epochs)
    
    with mlflow.start_run(nested=True, tags={'warmup': False}):
        ## Now we unfreeze the model and train all parameters
        max_epochs = 1
        for param in model.parameters():
            param.requires_grad = True
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model = train_model(model, training_dataloader=training_dataloader, 
                            dev_dataloader=dev_dataloader, optimizer=optimizer, 
                            loss_fn=loss_fn, device=device, max_epochs=max_epochs)
        print("Training is done")
        
    return model


def fit_threshold(targets, predictions):
    # Find the threshold that maximizes the Youden's J statistic (https://en.wikipedia.org/wiki/Youden%27s_J_statistic)
    # See https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/ for a detailed explanation
    # This will essentially pick the threshold which maximizes balanced accuracy
    # In reality you should probably set this depending on how you weight false negatives vs. false positives
    dev_fpr, dev_tpr, dev_threshold = roc_curve(targets, predictions)
    J = dev_tpr - dev_fpr
    ix = np.argmax(J)
    threshold = dev_threshold[ix]
    return threshold


def evaluate_model(model, dataset, tag, device, threshold=None):
    test_dataloader = DataLoader(dataset, 
                                 batch_size=16, 
                                 drop_last=False,
                                 shuffle=False
                                 )

    loss_fn = nn.BCEWithLogitsLoss()
    results = evaluate_on_dataloader(model, test_dataloader, loss_fn, device)
    labels = results['labels']
    logits = results['logits']
    if threshold is None:
        threshold = fit_threshold(labels, logits)
        mlflow.log_param(f'{tag}_threshold', threshold)
        
    class_predictions = results['logits'] >= threshold
    predictions_df = pd.DataFrame(data=dict(files=[str(p) for p in dataset.images],
                                            labels=results['labels'],
                                            logits=results['logits'],
                                            p=results['probabilities'],
                                            class_predictions=class_predictions))
    mlflow.log_table(predictions_df, f'{tag}_predictions.json')
    return threshold
    
        
def main():
    device = torch.device('cuda')
    with mlflow.start_run(tags={'run_level': 'root'}):
        training_dataset, dev_dataset, test_dataset = setup_data_splits()
        model = run_training(training_dataset, dev_dataset, test_dataset, device=device)
        dev_threshold = evaluate_model(model, dev_dataset, 'dev', device=device)
        evaluate_model(model, test_dataset, 'test', device=device, threshold=dev_threshold)


if __name__ == '__main__':
    mlflow.set_experiment(f"step_2")
    main()