from pathlib import Path
from copy import deepcopy
import json

from tqdm import tqdm, trange

from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

predictions_df = pd.read_csv('test_prediction.csv')
labels = predictions_df['labels']
logits = predictions_df['logits']

roc_auc = roc_auc_score(labels, logits)

print(f"ROC AUC score:", roc_auc)

fpr, tpr, threshold = roc_curve(labels, logits)
plt.plot(fpr, tpr)
plt.figure()
sns.kdeplot(data=predictions_df, x='p', hue='labels')
plt.show()

