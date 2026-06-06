# FinalNet

Click-through rate (CTR) prediction is one of the fundamental tasks for online advertising and recommendation. Inspired by factorization machines, in this paper, we propose FINAL, a factorized interaction layer that extends the widely-used linear layer and is capable of learning quadratic feature interactions. Similar to MLPs, multiple FINAL layers can be stacked into a FINAL block, yielding feature interactions with an exponential degree growth. We unify feature interactions and MLPs into a single FINAL block and empirically show its effectiveness as a replacement for the MLP block. 

> Jieming Zhu, Qinglin Jia, Guohao Cai, Quanyu Dai, Jingjie Li, Zhenhua Dong, Ruiming Tang, Rui Zhang. [FINAL: Factorized Interaction Layer for CTR Prediction](https://dl.acm.org/doi/10.1145/3539618.3591988), in SIGIR 2023.


## Requirements

Please install the requirements as follows:

```bash
pip install -r requirements.txt
```

## Configuration Guide

  
The `dataset_config.yaml` file contains all the dataset settings as follows.
  
| Params                        | Type | Default | Description                                                                                                                             |
| ----------------------------- | ---- | ------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| data_root                     | str  |         | the root directory to load and save data data                                                                                                          |
| data_format                   | str  |         | input data format, "h5", "csv", or "tfrecord" supported                                                                                 |
| train_data                    | str  | None    | training data path                                                                                                                      |
| valid_data                    | str  | None    | validation data path                                                                                                                    |
| test_data                     | str  | None    | test data path                                                                                                                          |
| min_categr_count              | int  | 1       | min count to filter category features,                                                                                                  |
| feature_cols                  | list |         | a list of features with the following dict keys                                                                                         |
| feature_cols::name            | str\|list  |         | feature column name in csv. A list is allowed in which the features have the same feature type and will be expanded accordingly.                                                                                                               |
| feature_cols::active          | bool |         | whether to use the feature                                                                                                              |
| feature_cols::dtype           | str  |         | the input data dtype, "int"\|"str"                                                                                                       |
| feature_cols::type            | str  |         | feature type "numeric"\|"categorical"\|"sequence"\|"meta"                                                                                  |
| label_col                     | dict |         | specify label column                                                                                                                    |
| label_col::name               | str  |         | label column name in csv                                                                                                                |
| label_col::dtype              | str  |         | label data dtype                                                                                                                        |



The `model_config.yaml` file contains all the model hyper-parameters as follows.
  
| Params                  | Type            | Default                 | Description                                                                                                                                                                                                       |
| ----------------------- | --------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model                   | str             | "FinalNet"                 | model name,  which should be same with model class name                                                                                                                                                           |
| dataset_id              | str             | "TBD"                   | dataset_id to be determined                                                                                                                                                                                       |
| loss                    | str             | "binary_crossentropy"   | loss function                                                                                                                                                                                                     |
| metrics                 | list            | ['logloss', 'AUC']      | a list of metrics for evaluation                                                                                                                                                                                  |
| task                    | str             | "binary_classification" | task type supported: ```"regression"```, ```"binary_classification"```                                                                                                                                            |
| optimizer               | str             | "adam"                  | optimizer used for training                                                                                                                                                                                       |
| learning_rate           | float           | 1.0e-3                  | learning rate                                                                                                                                                                                                     |
| embedding_regularizer   | float\|str       | 0                       | regularization weight for embedding matrix: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                  |
| net_regularizer         | float\|str       | 0                       | regularization weight for network parameters: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                |
| batch_size              | int             | 10000                   | batch size, usually a large number for CTR prediction task                                                                                                                                                        |
| embedding_dim           | int             | 32                      | embedding dimension of features. Note that field-wise embedding_dim can be specified in ```feature_specs```.                                                                                                      |
| block1_hidden_units       | list            | [64, 64, 64]          | hidden units of block1                                                                                                                                                                                              |
| block1_hidden_activations | str\|list        | "relu"               | activation function in block1. Particularly, layer-wise activations can be specified as a list, e.g., ["relu",  "leakyrelu", "sigmoid"]                                                                             |
| block2_hidden_units       | list            | [64, 64, 64]          | hidden units of block2                                                                                                                                                                                              |
| block2_hidden_activations | str             | "relu"                | activation function in block2. Particularly, layer-wise activations can be specified as a list, e.g., ["relu", "leakyrelu", "sigmoid"]                                                                              |
| block1_dropout            | float           | 0                       | dropout rate in block1                                                                                                                                                                                              |
| block2_dropout            | float           | 0                       | dropout rate in block2                                                                                                                                                                                              |
| block_type         | str            | "2B"                   | block type: ```"1B"```, ```"2B"```                                                                                                                                                                               |
| batch_norm         | bool            | True                   | whether to use BN                                                                                                                                                                                       |
| use_feature_gating                  | bool            | True                    | whether to use feature gating                                                                                                                                                                                   |
| residual_type         | str            | "concat"                |  residual type in FINAL: ```"concat"```, ```"sum"```                                                                                                                                                          |
| epochs                  | int             | 100                     | the max number of epochs for training, which can early stop via monitor metrics.                                                                                                                                  |
| shuffle                 | bool            | True                    | whether shuffle the data samples for each epoch of training                                                                                                                                                       |
| seed                    | int             | 2021                    | the random seed used for reproducibility                                                                                                                                                                          |
| monitor                 | str\|dict        | 'AUC'                   | the monitor metrics for early stopping. It supports a single metric, e.g., ```"AUC"```. It also supports multiple metrics using a dict, e.g., {"AUC": 2, "logloss": -1} means ```2*AUC - logloss```.              |
| monitor_mode            | str             | 'max'                   | ```"max"``` means that the higher the better, while ```"min"``` denotes that the lower the better.                                                                                                                |
| model_root              | str             | './checkpoints/'        | the dir to save model checkpoints and running logs                                                                                                                                                                |
| early_stop_patience     | int             | 2                       | training is stopped when monitor metric fails to become better for ```early_stop_patience=2```consective evaluation intervals.                                                                                    |
| save_best_only          | bool            | True                    | whether to save the best model checkpoint only                                                                                                                                                                    |
| eval_steps              | int\|None        | None                    | evaluate the model on validation data every ```eval_steps```. By default, ```None``` means evaluation every epoch.                                                                                                |


## Results

The evaluation results on AUC:

|   Model  |   Criteo  |   Avazu   | MovieLens |   Frappe  |
|:--------:|:---------:|:---------:|:---------:|:---------:|
|    DCN   |   81.39   |   76.47   |   96.87   |   98.39   |
| AutoInt+ |   81.39   |   76.45   |   96.92   |   98.48   |
|   AFN+   |   81.43   |   76.48   |   96.42   |   98.26   |
|  MaskNet |   81.39   |   76.49   |   96.87   |   98.43   |
|   DCNv2  |   81.42   |   76.54   |   96.91   |   98.45   |
|   EDCN   |   81.47   |   76.52   |   96.71   |   98.50   |
| FinalMLP |   81.49   |   76.66   |   97.20   | **98.61** |
| FinalNet_1B |   81.44   |   76.60   |   97.06   |   98.52   |
| FinalNet_2B | **81.54** | **76.71** | **97.25** |   98.57   |


For reproducing the results, please refer to https://github.com/reczoo/RecZoo/tree/main/ranking/ctr/FinalNet
