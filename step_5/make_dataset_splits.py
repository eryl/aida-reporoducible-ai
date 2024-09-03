import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


dataset = Path("data") / "oxfordiii"

files = sorted(dataset.glob('**/*.jpg'))


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

k_root_folds = 10
n_root_repeats = 1
k_nested_folds = 10
k_nested_repeats = 1

root_splitter = RepeatedStratifiedKFold(n_splits=k_root_folds, n_repeats=n_root_repeats, random_state=1729)

root_splits = dict()

for i, (modeling_meta_indices, test_meta_indices) in enumerate(root_splitter.split(indices, y=labels)):
    nested_splitter = RepeatedStratifiedKFold(n_splits=k_nested_folds, n_repeats=k_nested_repeats, random_state=4711)
    # Watch out for this mistake. The indices are not the actual values from *modeling_index*, 
    # but rather the indices of those entries. This is not the same as the train_test_split function in scikit-learn
    # We have to use the train and dev "meta" indices to select the actual indices, otherwise we will likely select items 
    # which are _actually_ in the test set
    modeling_indices = [int(indices[i]) for i in modeling_meta_indices]
    test_indices = [int(indices[i]) for i in test_meta_indices]
    nested_splits = dict()
    modeling_labels = [labels[i] for i in modeling_indices]
    for j, (train_meta_indices, dev_meta_indices) in enumerate(nested_splitter.split(modeling_indices, y=modeling_labels)):
        # Have a look at this variable, don't be like me and make this mistake
        train_test_meta_intersection = set(train_meta_indices).intersection(set(test_meta_indices))
        
        train_indices = [int(modeling_indices[i]) for i in train_meta_indices]
        dev_indices = [int(modeling_indices[i]) for i in dev_meta_indices]
        nested_splits[j] = {'train_indices': train_indices, 'dev_indices': dev_indices}
    
    root_splits[i] = {'test_indices': test_indices, 'nested_splits': nested_splits}

stringified_files = [str(p) for p in files]
splits = {'root_splits': root_splits, 'files': stringified_files}

with open(dataset / 'dataset_splits.json', 'w') as fp:
    json.dump(splits, fp)