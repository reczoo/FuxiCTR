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


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src
import gc
import argparse
import os
from pathlib import Path

def safe_eval(key, val):
    """Safely parse stringified dict/list values from YAML."""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception as e:
            print(f"Failed to parse '{key}': {e}")
            sys.exit(1)
    return val

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    params.setdefault('verbose', 1)
    params.setdefault('model_root', './model_ckpt/')

    # Safe parsing for potential YAML stringified fields
    params['feature_specs'] = safe_eval('feature_specs', params.get('feature_specs', {}))
    params['label_col'] = safe_eval('label_col', params.get('label_col', []))
    params['use_features'] = safe_eval('use_features', params.get('use_features', []))

    # Process features for FeatureProcessor
    feature_cols_for_processor = []
    for feat in params['use_features']:
        if feat in params['feature_specs']:
            col = params['feature_specs'][feat].copy()
            col['name'] = feat
            feature_cols_for_processor.append(col)
        else:
            print(f"WARNING: Feature '{feat}' missing from feature_specs.")

    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")

    if params["data_format"] == "csv":
        feature_processor = FeatureProcessor(
            feature_cols=feature_cols_for_processor,
            label_col=params['label_col'],
            dataset_id=params['dataset_id'],
            data_root=params['data_root']
        )
        params["train_data"], params["valid_data"], params["test_data"] = build_dataset(
            feature_encoder=feature_processor,
            train_data=params["train_data"],
            valid_data=params["valid_data"],
            test_data=params["test_data"],
            data_format="csv",
            feature_map_path=feature_map_json
        )

    params["feature_specs"] = feature_cols_for_processor  # Overwrite for internal consistency

    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)

    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters()

    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    test_result = {}
    if params["test_data"]:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)

    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n'.format(
            datetime.now().strftime('%Y%m%d-%H%M%S'),
            ' '.join(sys.argv), experiment_id, params['dataset_id'],
            "N.A.", print_to_list(valid_result), print_to_list(test_result)
        ))
