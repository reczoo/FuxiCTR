## Shared-Bottom

| [Overview](#Overview) | [Configuration](#Configuration) | [Implementation](#Implementation) | 


### Overview

Shared-Bottom is the classic multitasking model based on a shared bottom. The model is published in the following paper:

+ Rich Caruana. [Multitask Learning](https://link.springer.com/article/10.1023/A:1007379606734), in Machine learning 1997.

**Model structure:**

<div align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/xue-pai/FuxiCTR@main/docs/img/SharedBottom.jpg">
</div>

This figure comes from the paper of [MMoE (KDD 2018)](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-).

**Key components:**

+ *Bottom*: All tasks share a (DNN) Bottom.. 
  
  $$h=f_{DNN}(x)$$

+ *Task-specific Tower*: Each task has a separate tower. 
  
  $$y^{k}=f^{t}_{Tower}(h)$$

### Configuration

The `model_config.yaml` file contains all the model hyper-parameters as follows.

| Params                 | Type            | Default                   | Description                                                                                                                                                                                                       |
| ---------------------- | --------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model                  | str             | "SharedBottom"                     | model name,  which should be same with model class name                                                                                                                                                           |
| dataset_id             | str             | "TBD"                     | dataset_id to be determined                                                                                                                                                                                       |
| loss                   | list             | ["binary_crossentropy","binary_crossentropy"]     | loss function for each task                                                                                                                                                                                                     |
| metrics                | list            | ['logloss', 'AUC']        | a list of metrics for evaluation                                                                                                                                                                                  |
| task                   | list             | ["binary_crossentropy","binary_crossentropy"]   | task type supported: ```"regression"```, ```"binary_classification"```                                                                                                                                            |
| num_tasks       | int             | 2                         | number of tasks                                                                                                                                                                                |
| optimizer              | str             | "adam"                    | optimizer used for training                                                                                                                                                                                       |
| learning_rate          | float           | 1.0e-3                    | learning rate                                                                                                                                                                                                     |
| embedding_regularizer  | float/str       | 1.e-6                         | regularization weight for embedding matrix: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                  |
| net_regularizer        | float/str       | 0                         | regularization weight for network parameters: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                |
| batch_size             | int             | 128                     | batch size                                                                                                                                                        |
| embedding_dim          | int             | 128                        | embedding dimension of features. Note that field-wise embedding_dim can be specified in ```feature_specs```.                                                                                                      |
| bottom_hidden_units       | list            | [512, 256, 128]          | hidden units in expert networks                                                                                                                                                                                               |
| tower_hidden_units       | list            | [128, 64]          | hidden units in tower networks                                                                                                                                                                                               |
| hidden_activations        | str/list        | "relu"                    | activation function in DNN. Particularly, layer-wise activations can be specified as a list, e.g., ["relu",  "leakyrelu", "sigmoid"]                                                                              |
| net_dropout            | float           | 0                         | dropout rate in DNN                                                                                                                                                                                               |
| batch_norm             | bool            | False                     | whether using BN in DNN                                                                                                                                                                                           |
| epochs                 | int             | 50                       | the max number of epochs for training, which can early stop via monitor metrics.                                                                                                                                  |
| shuffle                | bool            | True                      | whether shuffle the data samples for each epoch of training                                                                                                                                                       |
| seed                   | int             | 2023                  | the random seed used for reproducibility                                                                                                                                                                          |
| monitor                | str/dict        | 'AUC' | the monitor metrics for early stopping. It supports a single metric, e.g., ```"AUC"```. It also supports multiple metrics using a dict, e.g., {"AUC": 2, "logloss": -1} means ```2*AUC - logloss```.              |
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
│   └── SharedBottom.py           # 模型代码
├── fuxictr_version.py            # fuxictr加载及版本检查文件
├── README.md                     # 使用说明
└── run_expid.py                  # 执行脚本文件
```

**Requirements:** 

The model is tested with the following dependencies.

+ fuxictr==2.0.1

+ pytorch==1.10

**Get started:**

Running the model on the tiny data:

```
python run_expid.py --expid SharedBottom_test --gpu 0 
```

