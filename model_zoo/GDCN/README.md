## GDCN

| [Overview](#Overview) | [Configuration](#Configuration) | [Implementation](#Implementation) | [Discussion](#Discussion) |
| :--: | :--: | :--: | :--: |

### Overview

GDCN is a CTR prediction model that learns explicit and bounded-degree cross features. The model is published in the following paper:

+ [Towards Deeper, Lighter and Interpretable Cross Network for
CTR Prediction](https://dl.acm.org/doi/pdf/10.1145/3583780.3615089), in CIKM 2023.

**Key components:**

+ *CrossNet*: The component provides explicit feature crossing with bounded degree.
  
  $$x_{l+1} = x_0x_l^Tw + b + x_l$$

+ *Dynamic embedding size*: It provides a formula to compute the embedding size of each feature field. 
  
  $$emb\_dim = 6\times(vocab\_size)^{1/4}$$

### Configuration

The `model_config.yaml` file contains all the model hyper-parameters as follows.

| Params                 | Type            | Default                   | Description                                                                                                                                                                                                       |
| ---------------------- | --------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model                  | str             | "DCN"                     | model name,  which should be same with model class name                                                                                                                                                           |
| dataset_id             | str             | "TBD"                     | dataset_id to be determined                                                                                                                                                                                       |
| loss                   | str             | "binary_crossentropy"     | loss function                                                                                                                                                                                                     |
| metrics                | list            | ['logloss', 'AUC']        | a list of metrics for evaluation                                                                                                                                                                                  |
| task                   | str             | "binary_classification"   | task type supported: ```"regression"```, ```"binary_classification"```                                                                                                                                            |
| optimizer              | str             | "adam"                    | optimizer used for training                                                                                                                                                                                       |
| learning_rate          | float           | 1.0e-3                    | learning rate                                                                                                                                                                                                     |
| embedding_regularizer  | float/str       | 0                         | regularization weight for embedding matrix: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                  |
| net_regularizer        | float/str       | 0                         | regularization weight for network parameters: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                |
| batch_size             | int             | 10000                     | batch size, usually a large number for CTR prediction task                                                                                                                                                        |
| embedding_dim          | int             | 32                        | embedding dimension of features. Note that field-wise embedding_dim can be specified in ```feature_specs```.                                                                                                      |
| dnn_hidden_units       | list            | [1024, 512, 256]          | hidden units in DNN                                                                                                                                                                                               |
| dnn_activations        | str/list        | "relu"                    | activation function in DNN. Particularly, layer-wise activations can be specified as a list, e.g., ["relu",  "leakyrelu", "sigmoid"]                                                                              |
| num_cross_layers       | int             | 3                         | number of cross layers in CrossNet                                                                                                                                                                                |
| net_dropout            | float           | 0                         | dropout rate in DNN                                                                                                                                                                                               |
| batch_norm             | bool            | False                     | whether using BN in DNN                                                                                                                                                                                           |
| epochs                 | int             | 100                       | the max number of epochs for training, which can early stop via monitor metrics.                                                                                                                                  |
| shuffle                | bool            | True                      | whether shuffle the data samples for each epoch of training                                                                                                                                                       |
| seed                   | int             | 20222023                  | the random seed used for reproducibility                                                                                                                                                                          |
| monitor                | str/dict        | {'AUC': 1, 'logloss': -1} | the monitor metrics for early stopping. It supports a single metric, e.g., ```"AUC"```. It also supports multiple metrics using a dict, e.g., {"AUC": 2, "logloss": -1} means ```2*AUC - logloss```.              |
| monitor_mode           | str             | 'max'                     | ```"max"``` means that the higher the better, while ```"min"``` denotes that the lower the better.                                                                                                                |
| model_root             | str             | './checkpoints/'          | the dir to save model checkpoints and running logs                                                                                                                                                                |
| num_workers            | int             | 3                         | the number of workers for data loader                                                                                                                                                                             |
| verbose                | int             | 1                         | 0 for salience while 1 for verbose logging with tqdm                                                                                                                                                              |
| early_stop_patience    | int             | 2                         | training is stopped when monitor metric fails to become better for ```early_stop_patience=2```consective evaluation intervals.                                                                                    |
| pickle_feature_encoder | bool            | True                      | whether to pickle the feature encoder during preprocessing. It is used when input ```data_format="csv"```.                                                                                                        |
| save_best_only         | bool            | True                      | whether to save the best model checkpoint only                                                                                                                                                                    |
| eval_steps             | int/None        | None                      | evaluate the model on validation data every ```eval_steps```. By default, ```None``` means evaluation every epoch.                                                                                                |
| debug_mode             | bool            | False                     | used for code testing. When setting it to ```True```, the ```experiment_id``` will be randomly generated to avoid interleaving when running multiple processes for parameter tunning by ```run_param_tuner.py```. |
| group_id               | None (optional) | None                      | required for metrics like ```gAUC```, ```NDCG```.                                                                                                                                                                 |
| use_features       | None (optional) | None                      | used for feature selection, i.e., only selecting an ordered subset of features as model input                                                                                                              |
| feature_specs          | dict (optional) | None                      | used for specifying field-wise configurations, such as ```embedding_dim```, ```feature_encoder``` for a specific field.                                                                                           |


### Implementation

**Code structure:**

```
├── config                        # 配置文件夹
│   ├── dataset_config.yaml       # 数据集配置文件
│   └── model_config.yaml         # 模型配置文件
├── src                           # 模型代码文件夹
│   └── GDCN.py                    # 模型代码
├── fuxictr_version.py            # fuxictr加载及版本检查文件
├── README.md                     # 使用说明
├── requirements.txt              # 依赖文件
└── run_expid.py                  # 执行脚本文件
```

**Requirements:** 

The model is tested with the following dependencies.

+ fuxictr==2.0.0

+ pytorch==1.11

**Get started:**

Running the model on the tiny data:

```
python run_expid.py --expid GDCNP_test --gpu 0
```
