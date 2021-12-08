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
from fuxictr.utils import set_logger, print_to_json, load_config
import logging
from fuxictr.pytorch.models import DeepFM
from fuxictr.pytorch.utils import seed_everything

if __name__ == '__main__':
    # Load params from config files
    config_dir = 'demo_config'
    experiment_id = 'DeepFM_test'
    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params)
    logging.info('Start the demo...')
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # Set up feature encoder
    feature_encoder = FeatureEncoder(**params)
    feature_encoder.fit(train_data=params['train_data'], 
                        min_categr_count=params['min_categr_count'])

    # Build train/validation/test data generators
    train_gen, valid_gen, test_gen = data_generator(feature_encoder,
                                                    train_data=params['train_data'],
                                                    valid_data=params['valid_data'],
                                                    test_data=params['test_data'],
                                                    batch_size=params['batch_size'],
                                                    shuffle=params['shuffle'],
                                                    use_hdf5=params['use_hdf5'])
    # Build a DeepFM model
    model = DeepFM(feature_encoder.feature_map, **params)
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=params['epochs'],
                        verbose=params['verbose'])
   
    # Reloading weights of the best checkpoint
    model.load_weights(model.checkpoint)

    # Evalution on validation
    logging.info('***** validation results *****')
    model.evaluate_generator(valid_gen)

    # Evalution on test
    logging.info('***** test results *****')
    model.evaluate_generator(test_gen)

