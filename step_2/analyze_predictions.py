from pathlib import Path
from copy import deepcopy
import json
from io import BytesIO

from tqdm import tqdm, trange

from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import seaborn as sns
from PIL import Image

import mlflow

runs_search_results_df = mlflow.search_runs(filter_string='tags.warmup = "False"', )
run_ids = runs_search_results_df['run_id']
    
for run_id in tqdm(run_ids):
    with mlflow.start_run(run_id) as run:
        
        dev_predictions_df = mlflow.load_table('dev_predictions.json', run_ids=[run_id])
        dev_labels = dev_predictions_df['labels']
        dev_logits = dev_predictions_df['logits']
        
        # Find the threshold that maximizes the Youden's J statistic (https://en.wikipedia.org/wiki/Youden%27s_J_statistic)
        # See https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/ for a detailed explanation
        # This will essentially pick the threshold which maximizes balanced accuracy
        # In reality you should probably set this depending on how you weight false negatives vs. false positives
        dev_fpr, dev_tpr, dev_threshold = roc_curve(dev_labels, dev_logits)
        J = dev_tpr - dev_fpr
        ix = np.argmax(J)
        binary_threshold = dev_threshold[ix]
        mlflow.log_param('threshold', binary_threshold)
        
        test_predictions_df = mlflow.load_table('test_predictions.json', run_ids=[run_id])
        labels = test_predictions_df['labels']
        logits = test_predictions_df['logits']
        class_predictions = test_predictions_df['logits'] >= binary_threshold
        test_predictions_df['class_prediction'] = test_predictions_df['logits'] >= binary_threshold
        evaluation_results = mlflow.evaluate(data=test_predictions_df, targets='labels', predictions='class_prediction', model_type='classifier')
        
        test_roc_auc = roc_auc_score(labels, logits)

        mlflow.log_metric('ROC AUC', test_roc_auc)

        fig = plt.figure()
        fpr, tpr, threshold = roc_curve(labels, logits)
        plt.plot(fpr, tpr)
        mlflow.log_figure(fig, 'roc_curve.png')
        
        fig = plt.figure()
        sns.kdeplot(data=test_predictions_df, x='p', hue='labels')
        mlflow.log_figure(fig, 'predictions_kde.png')

