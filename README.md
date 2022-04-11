# Implementation of "Variationally Regularized Graph-based Representation Learning for Electronic Health Records" model (VGNN)

This repository contains an implementation of VGNN model described in the [Variationally Regularized Graph-based Representation Learning for Electronic Health Records](https://arxiv.org/abs/1912.03761) paper.
The intent of this work is to attempt to reproduce the results.

## Installation
Preferably, create a new python environment via conda or venv. To install required dependencies, run 
```shell
pip install -r requirements.txt
```

To use the model, request MIMIC and/or eICU dataset access from [physionet](https://physionet.org).

Run the following scripts to preprocess the data and obtain files for training and evaluation of the model:
```shell
python preprocess_eicu.py --input_path {dataset_path} --output_path {storage_path}

python preprocess_mimic.py --input_path {dataset_path} --output_path {storage_path}
```

## Training
To train the model, run

```shell
python train.py --data_path /path/to/dataset --embedding_size 128 --result_path {model_path}
```

If you are training on eICU dataset, it is important to also pass --none_graph_features=1 flag since the first feature (previous readmission) is processed differently.

Explanation of command line arguments:

* --result_path. Output path of model checkpoints. Defaults to '.'.
* --data_path. Input path of processed dataset. Defaults to ./mimc.
* --embedding_size. Embedding size. Defaults to 256.
* --num_of_layers. Number of graph layers. Default to 2
* --num_of_heads. Number of attention heads. Defaults to 1
* --lr. Learning rate. Default to 1e-4
* --batch_size. Batch size. Defaults to 32
* --dropout. Dropout rate. Defaults to 0.4
* --reg. Whether to apply variational regularization. Defaults to True
* --lbd. Variational regularization loss weight. Defaults to 1.0
* --model_path. Path to checkpoint of trained model. Defaults to None
* --none_graph_features. Number of first features that should not be included in graph processing. Default to 0.

### Commands to train the current best models

eICU:
```shell
python train.py --data_path path/to/mimic --embedding_size 128 --batch_size 32 --result_path path/to/out --num_of_layers=2 --none_graph_features=1 --lbd=0.01
```

MIMIC:
```shell
python train.py --data_path path/to/mimic --embedding_size 256 --batch_size 16 --result_path path/to/out --num_of_layers=2 --lbd=0.003
```

## Evaluation
To evaluate a trained model on test set, run

```shell
python3 evaluate.py --model_path=path/to/checkpoint --data_path /path/to/dataset --embedding_size 128 --batch_size 32 --num_of_layers=2
```

Note that as of right now, the parameters provided must match the trained model.

## Training in Google Colab
Open train.ipynb in google colab and follow instruction in the notebook
The notebook just installs dependencies, clones the repo, and runs train.py. The model code is located in vgnn_models.py with some layers defined in layers.py.
