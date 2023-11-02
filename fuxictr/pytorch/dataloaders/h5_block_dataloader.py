# =========================================================================
# Copyright (C) 2023. FuxiCTR Authors. All rights reserved.
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


import numpy as np
from fuxictr.utils import load_h5
import h5py
from itertools import chain
import torch
from torch.utils import data
import logging
import glob


class BlockIterDataPipe(data.IterDataPipe):
    def __init__(self, block_datapipe, feature_map, verbose=0):
        self.feature_map = feature_map
        self.block_datapipe = block_datapipe
        self.verbose = verbose
        
    def load_data(self, data_path):
        data_dict = load_h5(data_path, verbose=self.verbose)
        data_arrays = []
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        for col in all_cols:
            array = data_dict[col]
            if array.ndim == 1:
                data_arrays.append(array.reshape(-1, 1))
            else:
                data_arrays.append(array)
        data_tensor = torch.from_numpy(np.hstack(data_arrays))
        return data_tensor

    def read_block(self, data_block):
        darray = self.load_data(data_block)
        for idx in range(darray.shape[0]):
            yield darray[idx, :]

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None: # single-process data loading
            block_list = self.block_datapipe
        else: # in a worker process
            block_list = [
                block
                for idx, block in enumerate(self.block_datapipe)
                if idx % worker_info.num_workers == worker_info.id
            ]
        return chain.from_iterable(map(self.read_block, block_list))


class DataLoader(data.DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, verbose=0, buffer_size=100000, **kwargs):
        data_blocks = glob.glob(data_path + "/*.h5")
        assert len(data_blocks) > 0, f"invalid data_path: {data_path}"
        if len(data_blocks) > 1:
            data_blocks.sort(key=lambda x: int(x.split("_")[-1].split(".")[0])) # e.g. "part_1.h5"
        self.data_blocks = data_blocks
        self.num_blocks = len(self.data_blocks)
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.num_batches, self.num_samples = self.count_batches_and_samples()
        datapipe = BlockIterDataPipe(data_blocks, feature_map, verbose)
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=buffer_size)
        super(DataLoader, self).__init__(dataset=datapipe, batch_size=batch_size, num_workers=num_workers)

    def __len__(self):
        return self.num_batches

    def count_batches_and_samples(self):
        num_samples = 0
        num_batches = 0
        for block_path in self.data_blocks:
            with h5py.File(block_path, 'r') as hf:
                y = hf[self.feature_map.labels[0]][:]
                num_samples += len(y)
                num_batches += int(np.ceil(len(y) * 1.0 / self.batch_size))
        return num_batches, num_samples


class H5BlockDataLoader(object):
    def __init__(self, feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, verbose=0, **kwargs):
        logging.info("Loading data...")
        train_gen = None
        valid_gen = None
        test_gen = None
        self.stage = stage
        if stage in ["both", "train"]:
            train_gen = DataLoader(feature_map, train_data, batch_size=batch_size, shuffle=shuffle, verbose=verbose, **kwargs)
            logging.info("Train samples: total/{:d}, blocks/{:d}".format(train_gen.num_samples, train_gen.num_blocks))     
            if valid_data:
                valid_gen = DataLoader(feature_map, valid_data, batch_size=batch_size, shuffle=False, verbose=verbose, **kwargs)
                logging.info("Validation samples: total/{:d}, blocks/{:d}".format(valid_gen.num_samples, valid_gen.num_blocks))

        if stage in ["both", "test"]:
            if test_data:
                test_gen = DataLoader(feature_map, test_data, batch_size=batch_size, shuffle=False, verbose=verbose, **kwargs)
                logging.info("Test samples: total/{:d}, blocks/{:d}".format(test_gen.num_samples, test_gen.num_blocks))
        self.train_gen, self.valid_gen, self.test_gen = train_gen, valid_gen, test_gen

    def make_iterator(self):
        if self.stage == "train":
            logging.info("Loading train and validation data done.")
            return self.train_gen, self.valid_gen
        elif self.stage == "test":
            logging.info("Loading test data done.")
            return self.test_gen
        else:
            logging.info("Loading data done.")
            return self.train_gen, self.valid_gen, self.test_gen
