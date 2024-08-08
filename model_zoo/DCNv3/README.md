# DCNv3

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-criteo)](https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-kdd12)](https://paperswithcode.com/sota/click-through-rate-prediction-on-kdd12?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-kkbox)](https://paperswithcode.com/sota/click-through-rate-prediction-on-kkbox?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-ipinyou)](https://paperswithcode.com/sota/click-through-rate-prediction-on-ipinyou?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-avazu)](https://paperswithcode.com/sota/click-through-rate-prediction-on-avazu?p=dcnv3-towards-next-generation-deep-cross)

We introduces the next generation deep cross networks, called DCNv3, which uses sub-networks LCN and ECN to capture both low-order and high-order feature interactions without relying on the less interpretable DNN. LCN uses a linearly growing interaction method for low-order interactions, while ECN employs an exponentially increasing method for high-order interactions. The Self-Mask filters interaction noise and further improves DCNv3’s computational efficiency. Tri-BCE helped the two sub-networks in DCNv3 obtain more suitable supervision signals for themselves. Comprehensive experiments on six datasets demonstrated the effectiveness, efficiency, and interpretability of DCNv3.
> Li, Honghao and Zhang, Yiwen and Zhang, Yi and Li, Hanwei and Sang, Lei and Zhu, Jieming. [DCNv3: Towards Next Generation Deep Cross Network for Click-Through Rate Prediction](https://arxiv.org/abs/2407.13349).

## Model Overview

<div align="center">
    <img src="https://github.com/user-attachments/assets/6f0479ce-edb2-4ad1-92a0-7ec9aeeb5f2d" alt="DCNv3" />
</div>


## Requirements

We have tested FinalMLP with the following requirements.

```python
python: 3.8
pytorch: 1.10
fuxictr: 2.0.1
```

## DCNv3 Configuration Guide

  
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
| model                   | str             | "FinalMLP"              | model name,  which should be same with model class name                                                                                                                                                           |
| dataset_id              | str             | "TBD"                   | dataset_id to be determined                                                                                                                                                                                       |
| loss                    | str             | "binary_crossentropy"   | loss function                                                                                                                                                                                                     |
| metrics                 | list            | ['logloss', 'AUC']      | a list of metrics for evaluation                                                                                                                                                                                  |
| task                    | str             | "binary_classification" | task type supported: ```"regression"```, ```"binary_classification"```                                                                                                                                            |
| optimizer               | str             | "adam"                  | optimizer used for training                                                                                                                                                                                       |
| learning_rate           | float           | 1.0e-3                  | learning rate                                                                                                                                                                                                     |
| embedding_regularizer   | float\|str       | 0                       | regularization weight for embedding matrix: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                  |
| net_regularizer         | float\|str       | 0                       | regularization weight for network parameters: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                |
| batch_size              | int             | 10000                   | batch size, usually a large number for CTR prediction task                                                                                                                                                        |
| embedding_dim           | int             | 16                      | embedding dimension of features. Note that field-wise embedding_dim can be specified in ```feature_specs```.                                                                                                      |
| num_deep_cross_layers   | int             | 3                       | number of layers in DCNv3            
| num_shallow_cross_layers| int             | 3                       | number of layers in SCNv3                                                                                                                                                                                             |
| deep_net_dropout        | float           | 0                       | dropout rate in DCNv3                                                                                                                                                                                              |
| shallow_net_dropout     | float           | 0                       | dropout rate in SCNv3                                                                                                                                                                                              |
| layer_norm              | bool            | False                   | whether using LN in Self-Mask                                                                                                                                                                                          |
| batch_norm              | bool            | False                   | whether using BN in SDCNv3                                                                                                                                                                                          |
| num_heads               | int             | 1                       | number of heads used for embeddding layer                                                                                                                                                                         |
| epochs                  | int             | 100                     | the max number of epochs for training, which can early stop via monitor metrics.                                                                                                                                  |
| shuffle                 | bool            | True                    | whether shuffle the data samples for each epoch of training                                                                                                                                                       |
| seed                    | int             | 2021                    | the random seed used for reproducibility                                                                                                                                                                          |
| monitor                 | str\|dict        | 'AUC'                   | the monitor metrics for early stopping. It supports a single metric, e.g., ```"AUC"```. It also supports multiple metrics using a dict, e.g., {"AUC": 2, "logloss": -1} means ```2*AUC - logloss```.              |
| monitor_mode            | str             | 'max'                   | ```"max"``` means that the higher the better, while ```"min"``` denotes that the lower the better.                                                                                                                |
| model_root              | str             | './checkpoints/'        | the dir to save model checkpoints and running logs                                                                                                                                                                |
| early_stop_patience     | int             | 2                       | training is stopped when monitor metric fails to become better for ```early_stop_patience=2```consective evaluation intervals.                                                                                    |
| save_best_only          | bool            | True                    | whether to save the best model checkpoint only                                                                                                                                                                    |
| eval_steps              | int\|None        | None                    | evaluate the model on validation data every ```eval_steps```. By default, ```None``` means evaluation every epoch.                                                                                                |

## ECN Configuration Guide

  
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
| model                   | str             | "FinalMLP"              | model name,  which should be same with model class name                                                                                                                                                           |
| dataset_id              | str             | "TBD"                   | dataset_id to be determined                                                                                                                                                                                       |
| loss                    | str             | "binary_crossentropy"   | loss function                                                                                                                                                                                                     |
| metrics                 | list            | ['logloss', 'AUC']      | a list of metrics for evaluation                                                                                                                                                                                  |
| task                    | str             | "binary_classification" | task type supported: ```"regression"```, ```"binary_classification"```                                                                                                                                            |
| optimizer               | str             | "adam"                  | optimizer used for training                                                                                                                                                                                       |
| learning_rate           | float           | 1.0e-3                  | learning rate                                                                                                                                                                                                     |
| embedding_regularizer   | float\|str       | 0                       | regularization weight for embedding matrix: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                  |
| net_regularizer         | float\|str       | 0                       | regularization weight for network parameters: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                |
| batch_size              | int             | 10000                   | batch size, usually a large number for CTR prediction task                                                                                                                                                        |
| embedding_dim           | int             | 16                      | embedding dimension of features. Note that field-wise embedding_dim can be specified in ```feature_specs```.                                                                                                      |
| num_cross_layers        | int             | 3                       | number of layers in DCNv3            
| net_dropout             | float           | 0                       | dropout rate in DCNv3                                                                                                                                                                                              |
| layer_norm              | bool            | False                   | whether using LN in Self-Mask                                                                                                                                                                                          |
| batch_norm              | bool            | False                   | whether using BN in DCNv3                                                                                                                                                                                          |
| num_heads               | int             | 1                       | number of heads used for embeddding layer                                                                                                                                                                         |
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

For reproducing the results, please refer to https://github.com/salmon1802/DCNv3/tree/master/checkpoints
