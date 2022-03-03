# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

import os
import logging
import logging.config
import yaml
import glob
import json
from collections import OrderedDict


def load_config(config_dir, experiment_id):
    params = dict()
    model_configs = glob.glob(os.path.join(config_dir, 'model_config.yaml'))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, 'model_config/*.yaml'))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    for config in model_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    if experiment_id not in found_params:
        raise ValueError("expid={} not found in config".format(experiment_id))
    # Update base settings first so that values can be overrided when conflict 
    # with experiment_id settings
    params.update(found_params.get('Base', {}))
    params.update(found_params.get(experiment_id))
    params['model_id'] = experiment_id
    dataset_params = load_dataset_config(config_dir, params['dataset_id'])
    params.update(dataset_params)
    return params


def load_dataset_config(config_dir, dataset_id):
    dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config.yaml'))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config/*.yaml'))
    for config in dataset_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                return config_dict[dataset_id]
    raise RuntimeError('dataset_id={} is not found in config.'.format(dataset_id))


def set_logger(params, log_file=None):
    if log_file is None:
        dataset_id = params['dataset_id']
        model_id = params['model_id']
        log_dir = os.path.join(params['model_root'], dataset_id)
        log_file = os.path.join(log_dir, model_id + '.log')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])

def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)

def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())

class Monitor(object):
    def __init__(self, kv):
        if isinstance(kv, str):
            kv = {kv: 1}
        self.kv_pairs = kv

    def get_value(self, logs):
        value = 0
        for k, v in self.kv_pairs.items():
            value += logs.get(k, 0) * v
        return value