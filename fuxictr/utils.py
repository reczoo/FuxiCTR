# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import h5py
import numpy as np
import pandas as pd
from collections import OrderedDict
import fuxictr


def load_config(config_dir, experiment_id):
    """Load merged model and dataset configuration for an experiment.

    Args:
        config_dir (str): Directory containing YAML configuration files.
        experiment_id (str): Experiment identifier to look up in model config.

    Returns:
        dict: Merged parameters from model and dataset configs.
    """
    params = load_model_config(config_dir, experiment_id)
    data_params = load_dataset_config(config_dir, params['dataset_id'])
    params.update(data_params)
    return params

def load_model_config(config_dir, experiment_id):
    """Load model configuration from YAML files.

    Searches for ``model_config.yaml`` or files under ``model_config/`` directory.
    Merges ``Base`` settings with experiment-specific settings.

    Args:
        config_dir (str): Directory containing model configuration YAML files.
        experiment_id (str): Experiment identifier to look up.

    Returns:
        dict: Model parameters with ``dataset_id`` and ``model_id`` set.

    Raises:
        RuntimeError: If no valid configuration files are found.
    """
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
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
    # Update base and exp_id settings consectively to allow overwritting when conflicts exist
    params = found_params.get('Base', {})
    params.update(found_params.get(experiment_id, {}))
    assert "dataset_id" in params, f'expid={experiment_id} is not valid in config.'
    params["model_id"] = experiment_id
    return params

def load_dataset_config(config_dir, dataset_id):
    """Load dataset configuration from YAML files.

    Searches for ``dataset_config.yaml`` or files under ``dataset_config/`` directory.

    Args:
        config_dir (str): Directory containing dataset configuration YAML files.
        dataset_id (str): Dataset identifier to look up.

    Returns:
        dict: Dataset parameters with ``dataset_id`` set.

    Raises:
        RuntimeError: If the dataset_id is not found in any config file.
    """
    params = {"dataset_id": dataset_id}
    dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config/*.yaml"))
    for config in dataset_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                return params
    raise RuntimeError(f'dataset_id={dataset_id} is not found in config.')

def set_logger(params):
    """Configure logging to file and stdout for a training run.

    Removes existing root handlers, then sets up a new logger writing to
    ``<model_root>/<dataset_id>/<model_id>.log``.

    Args:
        params (dict): Must contain ``dataset_id``. May contain ``model_id``
            and ``model_root``.
    """
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + '.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])
    logging.info("FuxiCTR version: " + fuxictr.__version__)

def print_to_json(data, sort_keys=True):
    """Convert a dictionary to a pretty-printed JSON string.

    All values are converted to strings for serialization.

    Args:
        data (dict): Dictionary to serialize.
        sort_keys (bool): If ``True``, sort keys alphabetically. Default: ``True``.

    Returns:
        str: JSON-formatted string with 4-space indentation.
    """
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)

def print_to_list(data):
    """Format a dictionary into a human-readable key-value list string.

    Args:
        data (dict): Dictionary with numeric values.

    Returns:
        str: ``"key1: value1 - key2: value2"`` formatted string.
    """
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())


class Monitor(object):
    """Monitor a weighted combination of metrics for early stopping.

    Tracks one or more metric names, each with an optional weight.
    If a single string is passed, it is treated as a metric with weight 1.

    Args:
        kv (str or dict): Metric name(s) and their weights.

    Example::

        monitor = Monitor({"AUC": 1.0, "logloss": -1.0})
        score = monitor.get_value({"AUC": 0.8, "logloss": 0.4})
    """

    def __init__(self, kv):
        if isinstance(kv, str):
            kv = {kv: 1}
        self.kv_pairs = kv

    def get_value(self, logs):
        """Compute the weighted sum of monitored metrics.

        Args:
            logs (dict): Dictionary of metric names to values.

        Returns:
            float: Weighted sum of metric values.
        """
        value = 0
        for k, v in self.kv_pairs.items():
            value += logs.get(k, 0) * v
        return value

    def get_metrics(self):
        """Get the list of monitored metric names.

        Returns:
            list: List of metric name strings.
        """
        return list(self.kv_pairs.keys())


def not_in_whitelist(element, whitelist=[]):
    """Check whether an element is absent from a whitelist.

    Args:
        element: Item to test.
        whitelist (list or scalar): Allowed values. If empty, always returns ``False``.
            If a non-list scalar, performs equality comparison.

    Returns:
        bool: ``True`` if the element is *not* in the whitelist.
    """
    if not whitelist:
        return False
    elif type(whitelist) == list:
        return element not in whitelist
    else:
        return element != whitelist
