# Reinforced Graph Neural Networks for Collaborative Filtering

## 1. Description
This is a project of the Computational Intelligence Lab 2021 ([Course Website](http://da.inf.ethz.ch/teaching/2021/CIL/)). The project implements a recommender system of user-item ratings between 1 and 5 ([Kaggle Competition](https://www.kaggle.com/c/cil-collaborative-filtering-2021)).

Our team name for the Kaggle competition is `Our Team`.

## 2. Structure of Repository
The following provides a high-level overview of the repository structure. More detail on the different directories is provided in the README corresponding to the directories.

- `/data`: This directory contains all the data used for training and testing our models.
- `/src`: The source code for all the models is contained in this directory.
- `init_leonhard.sh`: Sets up the environment for the Leonhard Cluster. More detail on the setup is provided in Section 3.
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

To use Comet in our experiments, we need to create a file called `comet.json` containing the necessary information for the Comet API. The file should be of the following structure:
```json
{
  "api_key": "API-KEY",
  "project_name": "PROJECT-NAME",
  "workspace": "WORKSPACE"
}
```
Hereby, the corresponding values from the Comet project should be inserted instead of the placeholder values API-KEY, PROJECT-NAME, and WORKSPACE.

By default, the `comet.json` file is located in the `CIL-Project` directory. This default location can be changed by providing the corresponding command-line option `--comet-key path-to-comet-key`. More detail on the command-line options is provided in Section 4.2.

## 4. Executing Models
TODO

### 4.1 Ensemble Learning
TODO

### 4.2 Command-Line Options
TODO

## 5. Recreating Experiments
TODO

## 6. Resource requirements
TODO

## 7. Acknowledgements
TODO
