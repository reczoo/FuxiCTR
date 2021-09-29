# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys
sys.path.append('../')
from datetime import datetime
import gc
import argparse
from fuxictr import autotuner 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
    parser.add_argument('--config', type=str, help='The config file for param tuning.')
    parser.add_argument('--tag', type=str, default=None, help='Which expid to run (e.g. 001 for the first expid).')
    parser.add_argument('--gpu', nargs='+', default=[-1], help='The list of gpu indexes, -1 for cpu.')
    args = vars(parser.parse_args())
    gpu_list = args['gpu']
    version = args['version']
    tag = args['tag']

    # generate parameter space combinations
    config_dir = autotuner.enumerate_params(args['config'])
    autotuner.run_all(version, config_dir, gpu_list, tag)

