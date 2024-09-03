# Automate your machine learning experiments

This repository contains code for the AIDA Technical Workshop on MLFlow and experiment automation.

The code is divided into five steps where we gradually introduce new concepts to make the machine learning experiments more scientifically robust.

For simplicity, we're using the Oxford III dataset of pets and will simply fine tune a neural network to separate the dogs from the cats.

Below is an outline:
 - Step 1: Initial training script. Has many flaws which makes scientific reproducibility difficult.
 - Step 2: Introduce experiment tracking with MLFlow. Focus on logging results.
 - Step 3: Add cross validation to make statistical results.
 - Step 4: Add hyper parameter optimization using Optuna to make results less reliant on humans hyper parameter search.
 - Step 5: Decouples configuraition from the training logic by putting configuration into a separate file. Introduces some advanced python tricks to make hyper parameter specification very flexible.


## Preparations

There are two ways to run the code of this workshop. The best experience is from running the code on your local computer by first cloning this repository and then install the dependencies using conda/mamba. The other is to use the suplied jupyter notebook as a kind of terminal and run the different scripts on Colab. This has the advantage that you don't have to have a computer with a mamba installation and a decent GPU.

### Local installation
Start by cloning this repository:

```shell
$ git clone this_url
```

If you don't have conda or mamba installed, we suggest you install it using [Miniforge](https://github.com/conda-forge/miniforge). This is a minimal conda distribution with mamba preinstalled. mamba is a drop-in replacement for conda, which has _much_ better dependency resolution, making package installations far less annoying. Be aware that it uses the conda-forge conda channel instead of the official one. This means the packages are more bleeding edge, for good and bad. They are also not curated, and can be a potential security risk (similar to installing packages from PyPI).


After you have a conda installation, create and activate the environment for this workshop by running:

```shell
$ conda env create -f environment.yml
$ conda activate reproai
```

#### Running the scripts

All the scripts used in this workshop are in the separate directories. The main scripts are `train.py` and `analyze_predictions.py`. The scripts typically assume that they are run from the repository root directory, e.g. like `python step_1/train.py`.


### Colab installation
To run these on Google Colab, use the following notebook as an entry point and follow the instructions there: [Colab notebook]().


## Step 1

In this step we familiarize ourselves with the main code. Look through the script and analyze it from a reproducibility point of view. If you know software engineering, what are some improvements which could make the code easier to maintain, extend and reuse?

 
## Step 2

Extend the code to use MLFlow for tracking important parameters and metrics. Organize each step as a different MLFlow experiment, and then break down the runs into meaningful parts (e.g. the warmup vs. full training). The analyzis script will read data from the MLFlow tracked experiments and use the built in evaluation methods to add a lot of metrics.


## Step 3

We've only trained on a single data split previously, that is very prone to sampling bias (is the result you get because of a particular draw as a test set?). In this step you will need to perform cross validation. The cross validation should support nested cross validation, where the outer loop puts away a test set and the inner loop resamples different training and development (validation) sets.


## Step 4

Manually setting hyper parameters is tricky, and makes the method reliant on you, the experimenter. Ideally, another person should be able to take your code and run in on another dataset for the same problem and get (statistically) similar results. To make this more likely, we need to make hyper parameter search automatic. In this step you will use Optuna to perform the hyper parameter search for you.

## Step 5

This far we've used hard coded configuration variables inside the training logic. In this step we will take the configuration parameters of our model and move them to a different file. We will still use python for configuring everything, but since the configuration is in an external file, we can easily try different configurations and persist the ones used in our experiment for later re-run (just use the logged config file in the other experiment). We will also illustrate how we can make hyper paremeter specification very flexible by wrapping the optuna arguments in a special object.