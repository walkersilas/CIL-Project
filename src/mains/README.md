## Main Methods
This directory serves as entry point to start invoking the different models. Each of the different models has an own main method which can directly be used to train and test the model with the default hyperparameters. Additionally, this directory includes the shell script used for ensemble learning of the Reinforced GNN with NCF model.

Note that the execution of different Python files may generate additional temporary directories within this directory. Whenever a neural network based model is executed, an additional directory `/checkpoints` is created containing the model with the best validation score during training. This best model is later used for the predictions. On invocation of the reinforcement generator, a directory `/cache` is generated containing all the generated reinforcements. When ensemble learning is used, a directory `/ensemble` is created containing the predictions of each of the single models. The predictions in this directory are later combined into a single prediction used for the final evaluation.

## Structure of Directory
- `ensemble_reinforced_gnn_ncf.sh`: Invokes ensemble learning of the Reinforced Graph Neural Network with Neural Collaborative Filtering
- `main_gnn_baseline.py`: Invokes the Graph Neural Network baseline
- `main_gnn_ncf.py`: Invokes the Graph Neural Network with Neural Collaborative Filtering
- `main_ncf_baseline.py`: Invokes the Neural Collaborative Filtering baseline
- `main_nmf.py`: Invokes the Non-Negative Matrix Factorization baseline
- `main_reinforced_gnn_ncf.py`: Invokes the Reinforced Graph Neural Network with Neural Collaborative Filtering
- `main_slopeone.py`: Invokes the SlopeOne baseline
- `main_svd_unbiased.py`: Invokes the Unbiased Singular Value Decomposition baseline
- `main_svdpp.py`: Invokes the Singular Value Decomposition++
- `reinforcement_generator.py`: Invokes the Reinforcement Generator
