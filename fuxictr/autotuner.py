# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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


import itertools
import subprocess
import yaml
import os
import numpy as np
import time
import glob
import hashlib
from .utils import print_to_json, load_model_config, load_dataset_config

# add this line to avoid weird characters in yaml files
yaml.Dumper.ignore_aliases = lambda *args : True

def enumerate_params(config_file, exclude_expid=[]):
    with open(config_file, "r") as cfg:
        config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
    # tuning space
    tune_dict = config_dict["tuner_space"]
    for k, v in tune_dict.items():
        if not isinstance(v, list):
            tune_dict[k] = [v]
    experiment_id = config_dict["base_expid"]
    if "model_config" in config_dict:
        model_dict = config_dict["model_config"][experiment_id]
    else:
        base_config_dir = config_dict.get("base_config", os.path.dirname(config_file))
        model_dict = load_model_config(base_config_dir, experiment_id)

    dataset_id = config_dict.get("dataset_id", model_dict["dataset_id"])
    if "dataset_config" in config_dict:
        dataset_dict = config_dict["dataset_config"][dataset_id]
    else:
        dataset_dict = load_dataset_config(base_config_dir, dataset_id)
        
    if model_dict["dataset_id"] == "TBD": # rename base expid
        model_dict["dataset_id"] = dataset_id
        experiment_id = model_dict["model"] + "_" + dataset_id
        
    # key checking
    tuner_keys = set(tune_dict.keys())
    base_keys = set(model_dict.keys()).union(set(dataset_dict.keys()))
    if len(tuner_keys - base_keys) > 0:
        raise RuntimeError("Invalid params in tuner config: {}".format(tuner_keys - base_keys))

    config_dir = config_file.replace(".yaml", "")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # enumerate dataset para combinations
    dataset_dict = {k: tune_dict[k] if k in tune_dict else [v] for k, v in dataset_dict.items()}
    dataset_para_keys = list(dataset_dict.keys())
    dataset_para_combs = dict()
    for idx, values in enumerate(itertools.product(*map(dataset_dict.get, dataset_para_keys))):
        dataset_params = dict(zip(dataset_para_keys, values))
        if (dataset_params["data_format"] == "npz" or
           (dataset_params["data_format"] == "parquet" and 
            dataset_params.get("rebuild_dataset") == False)):
            dataset_para_combs[dataset_id] = dataset_params
        else:
            hash_id = hashlib.md5("".join(sorted(print_to_json(dataset_params))).encode("utf-8")).hexdigest()[0:8]
            dataset_para_combs[dataset_id + "_{}".format(hash_id)] = dataset_params

    # dump dataset para combinations to config file
    dataset_config = os.path.join(config_dir, "dataset_config.yaml")
    with open(dataset_config, "w") as fw:
        yaml.dump(dataset_para_combs, fw, default_flow_style=None, indent=4)

    # enumerate model para combinations
    model_dict = {k: tune_dict[k] if k in tune_dict else [v] for k, v in model_dict.items()}
    model_para_keys = list(model_dict.keys())
    model_param_combs = dict()
    for idx, values in enumerate(itertools.product(*map(model_dict.get, model_para_keys))):
        model_param_combs[idx + 1] = dict(zip(model_para_keys, values))
        
    # update dataset_id into model params
    merged_param_combs = dict()
    for idx, item in enumerate(itertools.product(model_param_combs.values(),
                                                 dataset_para_combs.keys())):
        para_dict = item[0]
        para_dict["dataset_id"] = item[1]
        del para_dict["model_id"]
        random_str = ""
        if para_dict["debug_mode"]:
            random_str = "{:06d}".format(np.random.randint(1e6)) # add a random number to avoid duplicate during debug
        hash_id = hashlib.md5(("".join(sorted(print_to_json(para_dict))) + random_str).encode("utf-8")).hexdigest()[0:8]
        hash_expid = experiment_id + "_{:03d}_{}".format(idx + 1, hash_id)
        if hash_expid not in exclude_expid:
            merged_param_combs[hash_expid] = para_dict.copy()

    # dump model para combinations to config file
    model_config = os.path.join(config_dir, "model_config.yaml")
    with open(model_config, "w") as fw:
        yaml.dump(merged_param_combs, fw, default_flow_style=None, indent=4)
    print("Enumerate all tuner configurations done.")    
    return config_dir

def load_experiment_ids(config_dir):
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    experiment_id_list = []
    for config in model_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            experiment_id_list += config_dict.keys()
    return sorted(experiment_id_list)

def grid_search(config_dir, gpu_list, expid_tag=None, script='run_expid.py'):
    experiment_id_list = load_experiment_ids(config_dir)
    if expid_tag is not None:
        experiment_id_list = [expid for expid in experiment_id_list if str(expid_tag) in expid]
        assert len(experiment_id_list) > 0, "tag={} does not match any expid."
    gpu_list = list(gpu_list)
    idle_queue = list(range(len(gpu_list)))
    processes = dict()
    while len(experiment_id_list) > 0:
        if len(idle_queue) > 0:
            idle_idx = idle_queue.pop(0)
            gpu_id = gpu_list[idle_idx]
            expid = experiment_id_list.pop(0)
            cmd = "python -u {} --config {} --expid {} --gpu {}"\
                    .format(script, config_dir, expid, gpu_id)
            p = subprocess.Popen(cmd.split())
            processes[idle_idx] = p
        else:
            time.sleep(3)
            for idle_idx, p in processes.items():
                if p.poll() is not None: # terminated
                    idle_queue.append(idle_idx)
    [p.wait() for p in processes.values()]
