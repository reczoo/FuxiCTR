# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('../')
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
import gc
import argparse
import logging
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
    parser.add_argument('--config', type=str, default='../config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='LR_avazu_test', help='The experiment_id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    
    args = vars(parser.parse_args())
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    if params.get('version'):
        if params.get('version') != args['version']:
            raise RuntimeError('The config experiment_id={} does not support {}!'\
                               .format(experiment_id, args['version']))
    else:
        params['version'] = args['version']
    if args['version'] == 'tensorflow':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
        import tensorflow as tf
        from tensorflow.python.keras import backend as K
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=config))
        from fuxictr.tensorflow import models
        from fuxictr.tensorflow.utils import seed_everything
    elif args['version'] == 'pytorch':
        from fuxictr.pytorch import models
        from fuxictr.pytorch.utils import seed_everything
        params['gpu'] = args['gpu']

    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    dataset = params['dataset_id'].split('_')[0].lower()
    try:
        ds = getattr(datasets, dataset)
    except:
        raise RuntimeError('Dataset={} not exist!'.format(dataset))

    feature_encoder = ds.FeatureEncoder(**params)
    if params.get("data_format") == 'h5':
        if os.path.exists(feature_encoder.json_file):
            feature_encoder.feature_map.load(feature_encoder.json_file)
        else:
            raise RuntimeError('feature_map not exist!')
    elif params.get('pickle_feature_encoder') and os.path.exists(feature_encoder.pickle_file):
        feature_encoder = feature_encoder.load_pickle(feature_encoder.pickle_file)
    else:
        feature_encoder.fit(**params)

    model_class = getattr(models, params['model'])
    model = model_class(feature_encoder.feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    train_gen, valid_gen = datasets.data_generator(feature_encoder, stage='train', **params)
    model.fit_generator(train_gen, validation_data=valid_gen, **params)
    model.load_weights(model.checkpoint)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate_generator(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    logging.info('******** Test evaluation ********')
    test_gen = datasets.data_generator(feature_encoder, stage='test', **params)
    test_result = model.evaluate_generator(test_gen)
    
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))

