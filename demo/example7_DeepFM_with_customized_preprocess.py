# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import sys
sys.path.append('../')
import os
import logging
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
from model_zoo import DeepFM
import polars as pl
from datetime import datetime


class CustomizedFeatureProcessor(FeatureProcessor):
    """
    This is a demo for implementing customized feature processing functions.

    In the config/example7_config/dataset_config.yaml file, the 'convert_weekday' and 'convert_hour'
    processors are called. Hence, it is necessary to implement the two functions by inheriting from
    'fuxictr.preprocess.FeatureProcessor'. Some concrete examples can be found in 'fuxictr.datasets'.

    Each processor function ONLY accepts one argument: col_name, and returns an expression based on 
    polars. We use polars instead of pandas for speedup.
    """

    def convert_weekday(self, col_name=None):
        def _convert_weekday(timestamp):
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            return int(dt.strftime('%w'))
        return pl.col("time_stamp").apply(_convert_weekday)

    def convert_hour(self, col_name=None):
        def _convert_hour(timestamp):
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            return int(dt.hour)
        return pl.col("time_stamp").apply(_convert_hour)

if __name__ == '__main__':
    # Load params from config files
    config_dir = './config/example7_config'
    experiment_id = 'DeepFM_test_csv'
    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # Set feature_encoder that defines how to preprocess data
    feature_encoder = CustomizedFeatureProcessor(feature_cols=params["feature_cols"],
                                                 label_col=params["label_col"],
                                                 dataset_id=params["dataset_id"], 
                                                 data_root=params["data_root"])

    # Build dataset
    params["train_data"], params["valid_data"], params["test_data"] = \
        build_dataset(feature_encoder, 
                      train_data=params["train_data"],
                      valid_data=params["valid_data"],
                      test_data=params["test_data"])
    
    # Get feature_map that defines feature specs
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # Get train and validation data generators
    train_gen, valid_gen = RankDataLoader(feature_map, 
                                          stage='train', 
                                          train_data=params['train_data'],
                                          valid_data=params['valid_data'],
                                          batch_size=params['batch_size'],
                                          data_format=params["data_format"],
                                          shuffle=params['shuffle']).make_iterator()

    # Model initialization and fitting
    model = DeepFM(feature_map, **params)
    model.fit(train_gen, validation_data=valid_gen, epochs=params['epochs'])
    
    logging.info('***** Validation evaluation *****')
    model.evaluate(valid_gen)

    logging.info('***** Test evaluation *****')
    test_gen = RankDataLoader(feature_map, 
                              stage='test',
                              test_data=params['test_data'],
                              batch_size=params['batch_size'],
                              data_format=params["data_format"],
                              shuffle=False).make_iterator()
    model.evaluate(test_gen)
