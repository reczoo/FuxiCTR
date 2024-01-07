# Configuration CheetSheet

```{note}
Tutorials for FuxiCTR v2 only.
```

This tutorial presents the details of how to use the YAML config files.


The `dataset_config` contains the following keys:

+ **dataset_id**: the key used to denote a dataset split, e.g., taobao_tiny_data
+ **data_root**: the directory to save or load the h5 dataset files
+ **data_format**: csv | h5
+ **train_data**: training data file path
+ **valid_data**: validation data file path
+ **test_data**: test data file path
+ **min_categr_count**: the default threshold used to filter rare features
+ **feature_cols**: a list of feature columns, each containing the following keys
  - **name**: feature name, i.e., column header name.
  - **active**: True | False, whether to use the feature.
  - **dtype**: the data type of this column.
  - **type**: categorical | numeric | sequence, which type of features.
  - **source**: optional, which feature source, such as user/item/context.
  - **share_embedding**: optional, specify which feature_name to share embedding.
  - **embedding_dim**: optional, embedding dim of a specific field, overriding the default embedding_dim if used.
  - **pretrained_emb**: optional, filepath of pretrained embedding, which should be a h5 file with two columns (id, emb).
  - **freeze_emb**: optional, True | False, whether to freeze embedding is pretrained_emb is used.
  - **encoder**: optional, "MaskedAveragePooling" | "MaskedSumPooling" | "null", specify how to pool the sequence feature. "MaskedAveragePooling" is used by default. "null" means no pooling is required.
  - **splitter**: optional, how to split the sequence feature during preprocessing; the space " " is used by default. 
  - **max_len**: optional, the max length set to pad or truncate the sequence features. If not specified, the max length of all the training samples will be used. 
  - **padding**: optional, "pre" | "post", either pre padding or post padding the sequence. If "pre" is applied, the sequence will be chunked or padded from the left of the sequence.
  - **na_value**: optional, what value used to fill the missing entries of a field; "" is used by default.
  - **preprocess**': optional, which defines the function to preprocess the feature column. The function needs to be defined as a class method of features.FeatureEncoder. See the example: https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/datasets/criteo.py#L25
  - **normalizer**: optional, "StandardScaler" | "MinMaxScaler", which is applied for numeric features. 
+ **label_col**: label name, i.e., the column header of the label
  - **name**: the column header name for label
  - **dtype**: the data type
  
The `model_config` contains the following keys:

+ **expid**: the key used to denote an experiment id, e.g., DeepFM_test. Each expid corresponds to a dataset_id and the model hyper-parameters used for experiment.
+ **model_root**: the directory to save or load the model checkpoints and running logs.
+ **workers**: the number of processes used for data generator.
+ **verbose**: 0 for disabling tqdm progress bar; 1 for enabling tqdm progress bar.
+ **patience**: how many epochs to stop training if no improvments are made.
+ **pickle_feature_encoder**: True | False, whether pickle feature_encoder
+ **use_hdf5**: True | False, whether reuse h5 data if available
+ **save_best_only**: True | False, whether to save the best model weights only.
+ **every_x_epochs**: how many epochs to evaluate the model on valiadtion set, float supported. For example, 0.5 denotes to evaluate every half epoch.
+ **debug**: True | False, whether to enable debug mode. If enabled, every run will generate a new expid to avoid conflicted runs on two code versions. 
+ **partition_block_size**: the number of samples in a data block, -1 by default. Set the value when you want to into break a large h5 file into data blocks.
+ **model**: model name used to load the specific model class
+ **dataset_id**: the dataset_id used for the experiment
+ **loss**: currently support "binary_crossentropy" only.
+ **metrics**: list, currently support ['logloss', 'AUC'] only
+ **task**: currently support "binary_classification" only
+ **optimizer**: "adam" is used by default
+ **learning_rate**: the initial learning rate
+ **batch_size**: the batch size for model training
+ **embedding_dim**: the default embedding dim for all feature fields. It will be ignored if a feature has embedding_dim value.
+ **epochs**: the max number of epochs for model training
+ **shuffle**: True | False, whether to shuffle data for each epoch
+ **seed**: int, fix the random seed for reproduciblity
+ **monitor**: 'AUC' | 'logloss' | {'AUC': 1, 'logloss': -1}, the metric used to determine early stopping. The dict can be used for combine multiple metrics. E.g., {'AUC': 2, 'logloss': -1} means 2 * AUC - logloss and the larger the better. 
+ **monitor_mode**: 'max' | 'min', the mode of the metric. E.g., 'max' for AUC and 'min' for logloss.

There are also some model-specific hyper-parameters. E.g., DeepFM has the following specific hyper-parameters:
+ **hidden_units**: list, hidden units of MLP
+ **hidden_activations**: str or list, e.g., 'relu' or ['relu', 'tanh']. When each layer has the same activation, one could use str; otherwise use list to set activations for each layer.
+ **net_regularizer**: regularizaiton weight for MLP, supporting different types such as 1.e-8 | l2(1.e-8) | l1(1.e-8) | l1_l2(1.e-8, 1.e-8). l2 norm is used by default.
+ **embedding_regularizer**: regularizaiton weight for feature embeddings, supporting different types such as 1.e-8 | l2(1.e-8) | l1(1.e-8) | l1_l2(1.e-8, 1.e-8). l2 norm is used by default.
+ **net_dropout**: dropout rate for MLP, e.g., 0.1 denotes that hidden values are dropped randomly with 10% probability. 
+ **batch_norm**: False | True, whether to apply batch normalizaiton on MLP.


Many config files are available at https://github.com/xue-pai/FuxiCTR/tree/main/config for your reference. Here, we take the config [demo/demo_config](https://github.com/xue-pai/FuxiCTR/tree/main/demo/demo_config) as an example. The dataset_config.yaml and model_config.yaml are as follows. 

```
# dataset_config.yaml
taobao_tiny: # dataset_id
    data_root: ../data/
    data_format: csv
    train_data: ../data/tiny_data/train_sample.csv
    valid_data: ../data/tiny_data/valid_sample.csv
    test_data: ../data/tiny_data/test_sample.csv
    min_categr_count: 1
    feature_cols:
        - {name: ["userid","adgroup_id","pid","cate_id","campaign_id","customer","brand","cms_segid",
                  "cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"], 
                  active: True, dtype: str, type: categorical}
    label_col: {name: clk, dtype: float}
```

Note that we merge the feature_cols with the same config settings for compactness. But we also could expand them as shown below.

```
taobao_tiny:
    data_root: ../data/
    data_format: csv
    train_data: ../data/tiny_data/train_sample.csv
    valid_data: ../data/tiny_data/valid_sample.csv
    test_data: ../data/tiny_data/test_sample.csv
    min_categr_count: 1
    feature_cols:
        [{name: "userid", active: True, dtype: str, type: categorical},
         {name: "adgroup_id", active: True, dtype: str, type: categorical},
         {name: "pid", active: True, dtype: str, type: categorical},
         {name: "cate_id", active: True, dtype: str, type: categorical},
         {name: "campaign_id", active: True, dtype: str, type: categorical},
         {name: "customer", active: True, dtype: str, type: categorical},
         {name: "brand", active: True, dtype: str, type: categorical},
         {name: "cms_segid", active: True, dtype: str, type: categorical},
         {name: "cms_group_id", active: True, dtype: str, type: categorical},
         {name: "final_gender_code", active: True, dtype: str, type: categorical},
         {name: "age_level", active: True, dtype: str, type: categorical},
         {name: "pvalue_level", active: True, dtype: str, type: categorical},
         {name: "shopping_level", active: True, dtype: str, type: categorical},
         {name: "occupation", active: True, dtype: str, type: categorical}]
    label_col: {name: clk, dtype: float}
```

The following model config contains two parts. When `Base` is available, the base settings will be shared by all expids. The base settings can be also overridden in expid with the same key. This design is for compactness when a large group of model configs are available, as shown in `./config` folder. `Base` and expid `DeepFM_test` can be either put in the same `model_config.yaml` file or the same `model_config` directory. Note that in any case, each expid should be unique among all the expids.

```
# model_config.yaml
Base: 
    model_root: '../checkpoints/'
    workers: 3
    verbose: 1
    patience: 2
    pickle_feature_encoder: True
    use_hdf5: True
    save_best_only: True
    every_x_epochs: 1
    debug: False
    partition_block_size: -1

DeepFM_test:
    model: DeepFM
    dataset_id: taobao_tiny_data # each expid corresponds to a dataset_id
    loss: 'binary_crossentropy'
    metrics: ['logloss', 'AUC']
    task: binary_classification
    optimizer: adam
    hidden_units: [64, 32]
    hidden_activations: relu
    net_regularizer: 0
    embedding_regularizer: 1.e-8
    learning_rate: 1.e-3
    batch_norm: False
    net_dropout: 0
    batch_size: 128
    embedding_dim: 4
    epochs: 1
    shuffle: True
    seed: 2019
    monitor: 'AUC'
    monitor_mode: 'max'
```

The `load_config` method will automatically merge the above two parts. If you prefer, it is also flexible to remove `Base` and declare all the settings using only one dict as below.

```
DeepFM_test:
    model_root: '../checkpoints/'
    workers: 3
    verbose: 1
    patience: 2
    pickle_feature_encoder: True
    use_hdf5: True
    save_best_only: True
    every_x_epochs: 1
    debug: False
    partition_block_size: -1
    model: DeepFM
    dataset_id: taobao_tiny_data
    loss: 'binary_crossentropy'
    metrics: ['logloss', 'AUC']
    task: binary_classification
    optimizer: adam
    hidden_units: [64, 32]
    hidden_activations: relu
    net_regularizer: 0
    embedding_regularizer: 1.e-8
    learning_rate: 1.e-3
    batch_norm: False
    net_dropout: 0
    batch_size: 128
    embedding_dim: 4
    epochs: 1
    shuffle: True
    seed: 2019
    monitor: 'AUC'
    monitor_mode: 'max'
```
