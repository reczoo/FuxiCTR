# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys
sys.path.append('../')
import os
from fuxictr.datasets import data_generator
from fuxictr.datasets.taobao import FeatureEncoder
from datetime import datetime
from fuxictr.utils import set_logger, print_to_json
import logging
from fuxictr.pytorch.models import DCN
from fuxictr.pytorch.utils import seed_everything

if __name__ == '__main__':
    feature_cols = [{'name': ["userid","adgroup_id","pid","cate_id","campaign_id","customer","brand","cms_segid",
                              "cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"],
                     'active': True, 'dtype': 'str', 'type': 'categorical'}]
    label_col = {'name': 'clk', 'dtype': float}

    params = {'model_id': 'DCN_demo',
              'dataset_id': 'tiny_data_demo',
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
              'crossing_layers': 3,
              'learning_rate': 1e-3,
              'net_dropout': 0,
              'batch_norm': False,
              'optimizer': 'adam',
              'task': 'binary_classification',
              'loss': 'binary_crossentropy',
              'metrics': ['logloss', 'AUC'],
              'min_categr_count': 1,
              'embedding_dim': 10,
              'batch_size': 64,
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
              'workers': 1,
              'verbose': 0,
              'version': 'pytorch',
              'gpu': -1}

    set_logger(params)
    logging.info('Start the demo...')
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    feature_encoder = FeatureEncoder(feature_cols, 
                                     label_col, 
                                     dataset_id=params['dataset_id'], 
                                     data_root=params["data_root"],
                                     version=params['version'])
    feature_encoder.fit(train_data=params['train_data'], 
                        min_categr_count=params['min_categr_count'])

    train_gen, valid_gen, test_gen = data_generator(feature_encoder,
                                                    train_data=params['train_data'],
                                                    valid_data=params['valid_data'],
                                                    test_data=params['test_data'],
                                                    batch_size=params['batch_size'],
                                                    shuffle=params['shuffle'],
                                                    use_hdf5=params['use_hdf5'])
    model = DCN(feature_encoder.feature_map, **params)
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=params['epochs'],
                        verbose=params['verbose'])
    model.load_weights(model.checkpoint)
    
    logging.info('***** validation/test results *****')
    model.evaluate_generator(valid_gen)
    model.evaluate_generator(test_gen)


