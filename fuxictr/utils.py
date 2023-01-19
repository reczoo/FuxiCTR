# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
import h5py


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
    # Update base setting first so that values can be overrided when conflict 
    # with settings in experiment_id
    params.update(found_params.get('Base', {}))
    if experiment_id in found_params:
        params.update(found_params.get(experiment_id))
    if experiment_id not in found_params or 'dataset_id' not in params:
        raise RuntimeError('expid={} is not valid in config.'.format(experiment_id))
    params['model_id'] = experiment_id
    dataset_id = params['dataset_id']
    dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config.yaml'))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config/*.yaml'))
    dataset_id_found = False
    for config in dataset_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                dataset_id_found = True
                break
    assert dataset_id_found, "dataset_id={} is not found in config.".format(dataset_id)
    return params

def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params['model_id']
    log_dir = os.path.join(params['model_root'], dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + '.log')

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

    def get_metrics(self):
        return list(self.kv_pairs.keys())

def load_h5(data_path, verbose=0):
    if verbose == 0:
        logging.info('Loading data from h5: ' + data_path)
    data_dict = dict()
    with h5py.File(data_path, 'r') as hf:
        for key in hf.keys():
            data_dict[key] = hf[key][:]
    return data_dict