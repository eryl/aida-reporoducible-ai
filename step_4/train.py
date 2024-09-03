from dataclasses import dataclass, field
import dataclasses
import functools
from pathlib import Path
from copy import deepcopy
import json
from typing import Dict, List, Tuple, Union

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
import optuna

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
    
def get_label(file_path: Path):
        # Check if first letter is a capital letter
        if file_path.name[0].isupper():
            label = 'cat'
        else:
            label = 'dog'
        return label


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

@dataclass
class TrainingConfig:
    hidden_dim: int
    dropout_rate: float
    warmup_learning_rate: float
    warmup_weight_decay: float
    learning_rate: float
    weight_decay: float
    
    
@dataclass
class TrainingConfigHyperParameters:
    hidden_dim_values: List[int]
    dropout_rate_range: Tuple[float, float]
    warmup_learning_rate_range: Tuple[float, float]
    warmup_weight_decay_range: Tuple[float, float]
    learning_rate_range: Tuple[float, float]
    weight_decay_range: Tuple[float, float]
    
    

def instantiate_config(config: TrainingConfigHyperParameters, trial: optuna.Trial):
    hidden_dim = trial.suggest_categorical('hidden_dim', choices=config.hidden_dim_values)
    dropout_rate = trial.suggest_float('dropout_rate', *config.dropout_rate_range)
    warmup_learning_rate = trial.suggest_float('warmup_learning_rate', *config.warmup_learning_rate_range, log=True)
    warmup_weight_decay = trial.suggest_float('warmup_weight_decay', *config.warmup_weight_decay_range, log=True)
    learning_rate = trial.suggest_float('learning_rate', *config.learning_rate_range, log=True)
    weight_decay = trial.suggest_float('weight_decay', *config.weight_decay_range, log=True)
    instantiated_config = TrainingConfig(hidden_dim=hidden_dim, dropout_rate=dropout_rate, 
                                         warmup_weight_decay=warmup_weight_decay, warmup_learning_rate=warmup_learning_rate,
                                         learning_rate=learning_rate, weight_decay=weight_decay)
    return instantiated_config
    
    
def run_hpo_trial(training_dataset: ImageDataset, dev_dataset: ImageDataset, test_dataset: ImageDataset, device, config: TrainingConfigHyperParameters, trial: optuna.Trial):
    with mlflow.start_run(tags={'hpo_trial': trial.number}, nested=True):
        instantiated_config = instantiate_config(config, trial)
        model = run_training(training_dataset, dev_dataset, test_dataset, device, instantiated_config)
        dev_threshold, roc_auc = evaluate_model(model, dev_dataset, 'dev', device=device)
        return roc_auc


def run_training(training_dataset: ImageDataset, dev_dataset: ImageDataset, test_dataset: ImageDataset, device, config: TrainingConfig):
    params = dataclasses.asdict(config)
    mlflow.log_params(params)
        
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)
    model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, config.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout_rate), nn.Linear(config.hidden_dim, 1)) # We set the dimension to 1 since we'll use a sigmoid output

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
        optimizer = AdamW(parameters_to_train, lr=config.warmup_learning_rate, weight_decay=config.warmup_weight_decay)
        model = train_model(model=model, training_dataloader=training_dataloader, 
                    dev_dataloader=dev_dataloader, optimizer=optimizer, loss_fn=loss_fn, device=device,
                    max_epochs=max_warmup_epochs)
    
    with mlflow.start_run(nested=True, tags={'warmup': False}):
        ## Now we unfreeze the model and train all parameters
        max_epochs = 1
        for param in model.parameters():
            param.requires_grad = True
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
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
    roc_auc = roc_auc_score(labels, logits)
    class_predictions = results['logits'] >= threshold
    predictions_df = pd.DataFrame(data=dict(files=[str(p) for p in dataset.images],
                                            labels=results['labels'],
                                            logits=results['logits'],
                                            p=results['probabilities'],
                                            class_predictions=class_predictions))
    mlflow.log_table(predictions_df, f'{tag}_predictions.json')
    return threshold, roc_auc


def make_dataset_splits(files, train_indices, dev_indices, test_indices):
    labels = [get_label(f) for f in files]
    train_images = [files[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    dev_images = [files[i] for i in dev_indices]
    dev_labels = [labels[i] for i in dev_indices]
    test_images = [files[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    train_dataset = ImageDataset(train_images, train_labels)
    dev_dataset = ImageDataset(dev_images, dev_labels)
    test_dataset = ImageDataset(test_images, test_labels)
    return train_dataset, dev_dataset, test_dataset

        
def main():
    device = torch.device('cuda')
    config = TrainingConfigHyperParameters(hidden_dim_values= [128, 256, 512, 1024, 1024],
                                           dropout_rate_range = (0., 1.),
                                           warmup_learning_rate_range= (1e-6, 1e-2),
                                           warmup_weight_decay_range = (1e-7, 1e-5),
                                           learning_rate_range = (1e-7, 1e-3),
                                           weight_decay_range= (1e-6, 1e-3))
    
    with mlflow.start_run(tags={'run_level': 'root'}):
        dataset = Path("data") / "oxfordiii"
        split_path = dataset / 'dataset_splits.json'
        with open(split_path) as fp:
            dataset_splits = json.load(fp)
        mlflow.log_artifact(split_path, 'dataset_splits.json')
        
        n_root_splits = 2  # We can set this to limit the actual number of splits
        n_nested_splits = 2
        files = [Path(f) for f in dataset_splits['files']]
        
        current_root_split = 0
        for root_split_id, root_splits in dataset_splits['root_splits'].items():
            if current_root_split >= n_root_splits:
                break
            current_root_split += 1
            
            current_nested_split = 0
            for nested_split_id, nested_splits in root_splits['nested_splits'].items():
                if current_nested_split >= n_nested_splits:
                    break
                current_nested_split += 1
                
                train_indices = nested_splits['train_indices']
                dev_indices = nested_splits['dev_indices']
                test_indices = root_splits['test_indices']
                
                with mlflow.start_run(nested=True, tags={'run_level': 'resample', 'root_split_id': root_split_id, 'dataset_split': 'nested', 'nested_split_id': nested_split_id}):
                    n_trials = 3
                    study = optuna.create_study(direction='maximize')
                    training_dataset, dev_dataset, test_dataset = make_dataset_splits(files, train_indices, dev_indices, test_indices)
                    
                    # We use functool.partial to take the function and bound all arguments except the trial value which is the last one
                    bound_criterion = functools.partial(run_hpo_trial, training_dataset, dev_dataset, test_dataset, device, config)
                    study.optimize(bound_criterion, n_trials=n_trials)    
                    best_params = study.best_params
                    best_config = TrainingConfig(hidden_dim=best_params['hidden_dim'],
                                                 dropout_rate=best_params['dropout_rate'],
                                                 warmup_learning_rate=best_params['warmup_learning_rate'],
                                                 warmup_weight_decay=best_params['warmup_weight_decay'],
                                                 learning_rate=best_params['learning_rate'],
                                                 weight_decay=best_params['weight_decay'])
                    # This shorthand generlizes, but might be a bit more obtuse unless you know 
                    # exactly what best_parms is and what double star args does
                    # best_config = TrainingConfig(**best_params)  
                    
                    model = run_training(training_dataset, dev_dataset, test_dataset, device=device, config=best_config)
                    dev_threshold, dev_roc_auc = evaluate_model(model, dev_dataset, 'dev', device=device)
                    evaluate_model(model, test_dataset, 'test', device=device, threshold=dev_threshold)

        
        

if __name__ == '__main__':
    mlflow.set_experiment(f"step_4")
    main()