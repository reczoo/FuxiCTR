# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys
sys.path.append('../')
import os
from fuxictr import datasets
from fuxictr.datasets.taobao import FeatureEncoder
from fuxictr.utils import set_logger, print_to_json, load_dataset_config
import logging

if __name__ == '__main__':
    # Load params from config files
    config_dir = 'demo_config'
    dataset_id = 'taobao_tiny'
    params = load_dataset_config(config_dir, dataset_id)

    # set up logger and random seed
    set_logger(params, log_file='./demo.log')
    logging.info(print_to_json(params))

    # Set up feature encoder
    feature_encoder = FeatureEncoder(feature_cols=params["feature_cols"],
                                     label_col=params["label_col"],
                                     dataset_id=dataset_id, 
                                     data_root=params["data_root"])
    datasets.build_dataset(feature_encoder, 
                           train_data=params["train_data"], 
                           valid_data=params["valid_data"], 
                           test_data=params["test_data"])

