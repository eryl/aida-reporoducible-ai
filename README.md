# Make ML more reproducible

This repository contains code for the AIDA Technical Workshop on MLFlow and experiment automation.

The code is divided into five steps where we gradually introduce new concepts to make the machine learning experiments more scientifically robust.

For simplicity, we're using the Oxford III dataset of pets and will simply fine tune a neural network to separate the dogs from the cats.

Below is an outline:
 - Step 1: Initial training script. Has many flaws which makes scientific reproducibility difficult.
 - Step 2: Introduce experiment tracking with MLFlow. Focus on logging results.
 - Step 3: Add cross validation to make statistical results.
 - Step 4: Add hyper parameter optimization using Optuna to make results less reliant on humans hyper parameter search.
 - Step 5: Decouples configuraition from the training logic by putting configuration into a separate file. Introduces some advanced python tricks to make hyper parameter specification very flexible.


## Note on structure

The structure of this code base (the whole training logic in a single python module) is chosen so as to make following the evolution of the code easier. When you create your own experiment scripts you should break apart components and have them in separate modules with limited interfaces to make extending and reusing the code base easier. In particular putting dataset code and model code in separate modules often makes it easier to swap these out for other datasets and models.

## Preparations

There are two ways to run the code of this workshop. The best experience is from running the code on your local computer by first cloning this repository and then install the dependencies using conda/mamba. The other is to use the suplied jupyter notebook as a kind of terminal and run the different scripts on Colab. This has the advantage that you don't have to have a computer with a mamba installation and a decent GPU.

### Local installation

**N.b. This option assumes that you have a local NVIDIA GPU installed, for other options you need to consult the pytorch documentation.**

Start by cloning this repository:

```shell
$ git clone this_url
```

If you don't have conda or mamba installed, we suggest you install it using [Miniforge](https://github.com/conda-forge/miniforge). This is a minimal conda distribution with mamba preinstalled. mamba is a drop-in replacement for conda, which has _much_ better dependency resolution, making package installations far less annoying. Be aware that it uses the conda-forge conda channel instead of the official one. This means the packages are more bleeding edge, for good and bad. They are also not curated, and can be a potential security risk (similar to installing packages from PyPI).


After you have a conda installation, create and activate the environment for this workshop by running:

```shell
$ mamba env create -f environment.yml
$ mamba activate reproai
```

or if you don't have mamba installed:

```shell
$ conda env create -f environment.yml
$ conda activate reproai
```

#### Running the scripts

All the scripts used in this workshop are in the separate directories. The main scripts are `train.py` and `analyze_predictions.py`. The scripts typically assume that they are run from the repository root directory, e.g. like `python step_1/train.py`.


### Colab installation
To run these on Google Colab, use the following notebook as an entry point and follow the instructions there: [Colab notebook](https://colab.research.google.com/github/eryl/aida-reporoducible-ai/blob/main/run_on_colab.ipynb).


## Step 1

Before you do anything else, download the data we will use by running:

```shell
$ python step_1/download_data.py
```

This uses the pytorch vision datasets to download the Oxford III dataset into `data/oxfordiii`.

Now you can run the script using

```shell
$ python step_1/train.py
```

Once it's done, you can analyze the trained model and the test predictions by running:

```shell
$ python step_1/analyze_predictions.py
```


In this step we familiarize ourselves with the main code. Look through the script and analyze it from a reproducibility point of view. If you know software engineering, what are some improvements which could make the code easier to maintain, extend and reuse?

 
## Step 2

In this step we extend the code to use MLFlow for tracking important parameters and metrics. Each step will be organized as a different MLFlow experiment, and then broken down the runs into meaningful parts (e.g. the warmup vs. full training). The analyzis script will read data from the MLFlow tracked experiments and use the built in evaluation methods to add a lot of metrics.

You don't need to redownload data, instead just run the same scripts as before:

```shell
$ python step_1/train.py
```

```shell
$ python step_1/analyze_predictions.py
```

The difference now is that instead of just dumping data to text files with hard coded paths (overwriting any previous results), everything is logged into mlflow. To view the results run the command:

```shell
$ mlflow ui
```

This creates a new  mlflow viewer server (similar to Tensorboard) at [127.0.0.1:5000](http://127.0.0.1:5000)
Here you can inspect the results from the training and the analysis script. We've organized the training into nested runs. The root run is where the main results are logged, while the nested runs mainly contain information on training statistics.


## Step 3

We've only trained on a single data split previously, that is very prone to sampling bias (is the result you get because of a particular draw as a test set?). In this step we add cross validation. The cross validation used a nested strategy, where the root level samples different test sets and the nested level samples different training and development (validation) sets. This allows us to statistically estimate the effect data sampling has on the trained models.

Start by creating the data splits we will use:

```shell
$ python step_3/make_dataset_splits.py
```
This creates a nested dictionary saved to `data/oxfordiii/dataset_splits.json`. The dictionary has two keys at the top level: `files` and `root_splits`. The `files` value is a list of the files we use. The order of these files is what the splits will refer to so is considered the canonical order.
The `root_splits` value is another dictionary where the keys are fold indices which maps to nested dictionaries with the keys `test_indices` and `nested_splits`. The test indices are the indices used for the heldout (test) set of this fold, while the nested splits in turn is a dictionary with keys being subfold indices and the values being a dictionary with `train_indices` and `dev_indices` keys and their respective indices.

The reason for creating this external split definition is that it makes experimentation more robust. If you want to compare how well different models perform over different folds, it makes the comparison more straight forward if the different models have the same sampling biases.

In the training script, we can train on only subsets of all the folds to quickly test things by setting the variables 
`n_root_splits` and `n_nested_splits` to different values. By default they are both set to $2$, so we will perform $4=2*2$ experiments.

To perform all the splits ($10*10$ with the default split script) you can set these variables to `None`.

Run the training like before with:

```shell
$ python step_3/train.py
```
```shell
$ python step_1/analyze_predictions.py
```

You can now compare the results of the different splits using `mlflow ui`.


## Step 4

Manually setting hyper parameters is tricky, and makes the method reliant on you, the experimenter. Ideally, another person should be able to take your code and run in on another dataset for the same problem and get (statistically) similar results. To make this more likely, we need to make hyper parameter search automatic. In this step we've added search using Optuna to perform the hyper parameter search for you.

Hyper parameter search is also iterative, so doing it for each nested search adds another factor to the number of experiments. In this case the default number of hyper parameter iterations is low (3), and once we've done all the hyper parameter runs we will make a final training with the best hyper parameters. This means that the we will do a total of $16 = 2*2*(3+1)$ training runs! 

For real world problems, the higher you set the number of hyper parameter trials, the better. Here the limiting factor is really the amount of compute you have available. A practical alternative can be to do the hyper parameter search using only one split of the modeling set, and then use the same set of hyper parameters for all the nested cross validation folds, with the risk of biasing the hyper parameter selection to a specific development set giving you worse performance than you might have had.


```shell
$ python step_4/train.py
```
```shell
$ python step_4/analyze_predictions.py
```


## Step 5

This far we've used hard coded configuration variables inside the training logic. In this step we will take the configuration parameters of our model and move them to a different file. We will still use python for configuring everything, but since the configuration is in an external file, we can easily try different configurations and persist the ones used in our experiment for later re-run (just use the logged config file in the other experiment). We will also illustrate how we can make hyper paremeter specification very flexible by wrapping the optuna arguments in a special object.
The main difference between this step and the previous one is mostly one of software engineering; we've change how the code is organized and not the experiment it actually performs.


```shell
$ python step_5/train.py
```
```shell
$ python step_5/analyze_predictions.py
```
