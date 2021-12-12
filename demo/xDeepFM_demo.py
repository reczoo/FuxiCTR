# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys
sys.path.append('../')
import os
import logging
from datetime import datetime
from fuxictr import datasets
from fuxictr.datasets.taobao import FeatureEncoder
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.pytorch.models import xDeepFM
from fuxictr.pytorch.torch_utils import seed_everything

if __name__ == '__main__':
    feature_cols = [{'name': ["userid","adgroup_id","pid","cate_id","campaign_id","customer","brand","cms_segid",
                              "cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"],
                     'active': True, 'dtype': 'str', 'type': 'categorical'}]
    label_col = {'name': 'clk', 'dtype': float}

    params = {'model_id': 'xDeepFM_demo',
              'dataset_id': 'taobao_tiny',
              'train_data': '../data/tiny_data/train_sample.csv',
              'valid_data': '../data/tiny_data/valid_sample.csv',
              'test_data': '../data/tiny_data/test_sample.csv',
              'model_root': '../checkpoints/',
              'data_root': '../data/',
              'feature_cols': feature_cols,
              'label_col': label_col,
              'embedding_regularizer': 0,
              'net_regularizer': 0,
              'dnn_hidden_units': [64, 64],
              'dnn_activations': "relu",
              'learning_rate': 1e-3,
              'net_dropout': 0,
              'cin_layer_units': [16, 16, 16],
              'batch_norm': False,
              'optimizer': 'adam',
              'task': 'binary_classification',
              'loss': 'binary_crossentropy',
              'metrics': ['logloss', 'AUC'],
              'min_categr_count': 1,
              'embedding_dim': 10,
              'batch_size': 16,
              'epochs': 3,
              'shuffle': True,
              'seed': 2019,
              'monitor': 'AUC',
              'monitor_mode': 'max',
              'use_hdf5': True,
              'pickle_feature_encoder': True,
              'save_best_only': True,
              'every_x_epochs': 1,
              'patience': 2,
              'num_workers': 1,
              'partition_block_size': -1,
              'verbose': 0,
              'version': 'pytorch',
              'gpu': -1}

    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # Set feature_encoder that defines how to preprocess data
    feature_encoder = FeatureEncoder(feature_cols, 
                                     label_col, 
                                     dataset_id=params['dataset_id'], 
                                     data_root=params["data_root"])

    # Build dataset from csv to h5
    datasets.build_dataset(feature_encoder, 
                           train_data=params["train_data"], 
                           valid_data=params["valid_data"], 
                           test_data=params["test_data"])
    
    # Get feature_map that defines feature specs
    feature_map = feature_encoder.feature_map

    # Get train and validation data generator from h5
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    train_gen, valid_gen = datasets.h5_generator(feature_map, 
                                                 stage='train', 
                                                 train_data=os.path.join(data_dir, 'train.h5'),
                                                 valid_data=os.path.join(data_dir, 'valid.h5'),
                                                 batch_size=params['batch_size'],
                                                 shuffle=params['shuffle'])
    
    # Model initialization and fitting                                                  
    model = xDeepFM(feature_encoder.feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    model.fit_generator(train_gen, 
                        validation_data=valid_gen, 
                        epochs=params['epochs'],
                        verbose=params['verbose'])
    model.load_weights(model.checkpoint) # reload the best checkpoint
    
    logging.info('***** validation results *****')
    model.evaluate_generator(valid_gen)

    logging.info('***** validation results *****')
    test_gen = datasets.h5_generator(feature_map, 
                                     stage='test',
                                     test_data=os.path.join(data_dir, 'test.h5'),
                                     batch_size=params['batch_size'],
                                     shuffle=False)
    model.evaluate_generator(test_gen)


