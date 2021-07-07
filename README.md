# Reinforced Graph Neural Networks for Collaborative Filtering

## 1. Description
This is a project of the Computational Intelligence Lab 2021 ([Course Website](http://da.inf.ethz.ch/teaching/2021/CIL/)). The project implements a recommender system of user-item ratings between 1 and 5 ([Kaggle Competition](https://www.kaggle.com/c/cil-collaborative-filtering-2021)).

Our team name for the Kaggle competition is `Our Team`.

## 2. Structure of Repository
The following provides a high-level overview of the repository structure. More detail on the different directories is provided in the README corresponding to the directories.

- `/data`: This directory contains all the data used for training and testing our models.
- `/src`: The source code for all the models is contained in this directory.
- `init_leonhard.sh`: Sets up the environment for the Leonhard Cluster. More detail on setting up the environment is provided in Section 3.
- `requirements.txt`: Specifies the required dependencies to run the models.

## 3. Setup
### 3.1 Clone the project
Cloning the project downloads all models used for this project including the data needed for training and prediction.
```
git clone https://github.com/walkersilas/CIL-Project.git
cd CIL-Project
```

### 3.2 Setting up the Python Environment
All the experiments for this project were run using Python version 3.7.7.

First, a python virtual environment needs to be created in the `CIL-Project` directory. This is done using the following command in the `CIL-Project` directory:
```
python3 -m venv ./venv
```
Executing the `init_leonhard.sh` script installs the required dependencies and adds the source directory to the PYTHONPATH:
```
source ./init_leonhard.sh
```

### 3.3 Setting up the Comet Logger
In order to log the experiments and retrieve the predictions, we have used [Comet](https://www.comet.ml/). This logger enables tracking different metrics during training. Additionally, the predictions of our models are logged directly as an asset using Comet. Access to these predictions is then granted on the Comet website.

As a first step, a Comet account needs to be created on the [Comet website](https://www.comet.ml/). Afterwards, a new project can be added to the workspace of Comet. When viewing the project, the API key is accessible.

To use Comet in our experiments, we need to create a JSON file containing the necessary information for the Comet API. The file should be of the following structure:
```json
{
  "api_key": "API-KEY",
  "project_name": "PROJECT-NAME",
  "workspace": "WORKSPACE"
}
```
Hereby, the corresponding values from the Comet project should be inserted instead of the placeholder values API-KEY, PROJECT-NAME, and WORKSPACE.

By default, the file is called `comet.json` and is located in the `CIL-Project` directory. This default location can be changed by providing the corresponding command-line option `--comet-key path-to-comet-key`. More detail on the command-line options is provided in Section 4.3.

## 4. Executing Models
After setting up the environment, executing different models is done in a fairly generic fashion. First, we need to change to the directory containing the main methods, i.e. `cd /src/mains`.

If we want to execute the Reinforced GNN with NCF model, we need to create the reinforcements first. Thus, the model is executed as follows:
```
python3 reinforcement_generator.py
python3 main_reinforced_gnn_ncf.py
```
Conversely, if we want to execute one of the other models, we simply execute the following command where MODEL is replaced by the corresponding model name:
```
python3 main_MODEL.py
```

### 4.1 Ensemble Learning
In order to perform ensemble learning of the Reinforced GNN with NCF model, one needs to perform the following steps:

1. Similarly to before, change to the directory containing the main methods `cd /src/mains`.

2. Execute the shell script for ensemble learning of the Reinforced GNN with NCF model:
```
./ensemble_reinforced_gnn_ncf.sh
```

### 4.2 Configurations
Each of the models has a default configuration used for training and testing. To change this default configuration, one can provide a partial configuration as a JSON file. This is done using the command-line option `--config path-to-config-json`. The model then internally combines the default configuration with the provided new configuration. The final configuration used for running the model is then simply the default configuration where all values which are specified in the new configuration have been overwritten.

As an example the following partial configuration overwrites the learning rate of the default configuration while keeping the rest of the default configuration as it is:
```json
{
  "learning_rate": 1e-5
}
```

### 4.3 Command-Line Options
To provide some flexibility of the execution, all of the main methods support the same command-line options. These options are briefly described in the following:

- `--data-dir`: This specifies the relative path from `CIL-Project/src/mains` to the data directory containing the unprocessed data. The default path is `../../data` which points to `CIL-Project/data`.

- `--train-data`: This specifies the name of the training data file. The default name is `data_train.csv`.

- `--test-data`: This specifies the name of the testing data file. The default name is `data_test.csv`.

- `--random-seed`: This specifies the random seed used during the execution of the models. The default random seed is `7`.

- `--disable-logging`: This specifies whether the Comet logger should be disabled. By default, this value is set to `false`.

- `--comet-key`: This specifies the relative path from `CIL-Project/src/mains` to the Comet API key and credentials as described in Section 3.3. The default path is `../../comet.json` which points to `CIL-Project/comet.json`.

- `--comet-directory`: This specifies the log directory when Comet can not be run in online mode. The default log directory is `./logs`.

- `--dataloader-workers`: This specifies the number of worker threads used to load the data in the models. By default this value is set to `8`.

- `--config`: This specifies the path to an optional configuration used for training and testing the models. The default value for this option is `None`.

- `--ensemble-learning`: This option specifies whether the model is run with ensemble learning or not. By default, this option is set to `false`. Note that this option is only supported for the Reinforced GNN with NCF model.

- `--ensemble-directory`: This option is used to specify the directory from which the mean predictions should be computed. By default, this option is set to `None`. Note that this option is only supported for `CIL-Project/src/utilities/get_mean_predictions.py` which combines predictions of multiple models.

## 5. Reproducing Experiments
TODO

## 6. Resource Requirements
All of the experiments were run on the [Leonhard Cluster](https://scicomp.ethz.ch/wiki/Leonhard) using 1 GPU and 64 GB of RAM. All the standard models (without ensemble learning) finish execution in roughly 1 hour. Conversely, the ensemble learning model takes roughly 7.5 hours.

## 7. Acknowledgements
We would like to give credit to the following two libraries used during the project work:

- [Pytorch Lightning](https://www.pytorchlightning.ai/) which provides a high-level interface for PyTorch.

- [Surprise Library](https://github.com/NicolasHug/Surprise) which provides implementations of recommender system algorithms used in some of our baselines.
