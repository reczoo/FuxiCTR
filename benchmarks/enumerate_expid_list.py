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


import sys
sys.path.append('../')
from datetime import datetime
import gc
import pandas as pd
import argparse
from fuxictr import autotuner 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        default='../config/tuner_config.yaml', 
                        help='The config file for para tuning.')
    parser.add_argument('--exclude', type=str, 
                        default='', 
                        help='The experiment_result.csv file to exclude finished expid.')
    args = vars(parser.parse_args())
    exclude_expid = []
    if args['exclude'] != '':
        result_df = pd.read_csv(args['exclude'], header=None)
        expid_df = result_df.iloc[:, 2].map(lambda x: x.replace('[exp_id] ', ''))
        exclude_expid = expid_df.tolist()
    # enumerate parameter space combinations
    config_dir = autotuner.enumerate_params(args['config'], exclude_expid=exclude_expid)

