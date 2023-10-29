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


import numpy as np
from fuxictr.utils import load_h5
import h5py
from itertools import chain
import torch
from torch.utils import data
import logging
import glob


class DataBlockDataset(data.IterableDataset):
    def __init__(self, feature_map, block_file_list, shuffle=False, verbose=0):
        # block_file_list: path list of data blocks
        self.feature_map = feature_map
        self.data_blocks = block_file_list
        self.shuffle = shuffle
        self.verbose = verbose
        
    def load_data_array(self, data_path):
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
    
    def iter_block(self, data_block):
        darray = self.load_data_array(data_block)
        block_size = darray.shape[0]
        indexes = list(range(block_size))
        if self.shuffle:
            np.random.shuffle(indexes)
        for idx in indexes:
            yield darray[idx, :]

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None: # single-process data loading
            chunk_list = self.data_blocks
        else: # in a worker process
            worker_id = worker_info.id
            chunk_list = np.array_split(self.data_blocks, worker_info.num_workers)
            sub_list = chunk_list[worker_id].tolist()
            if self.shuffle:
                np.random.shuffle(sub_list)
        return chain.from_iterable(map(self.iter_block, sub_list))


class DataLoader(data.DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, verbose=0, **kwargs):
        data_blocks = glob.glob(data_path + "/*.h5")
        assert len(data_blocks) > 0, f"invalid data_path: {data_path}"
        if len(data_blocks) > 1:
            data_blocks.sort(key=lambda x: int(x.split("_")[-1].split(".")[0])) # e.g. "part_1.h5"
        self.data_blocks = data_blocks
        self.num_blocks = len(self.data_blocks)
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.num_batches, self.num_samples = self.count_batches_and_samples()
        self.dataset = DataBlockDataset(feature_map, data_blocks, shuffle=shuffle, verbose=verbose)
        super(DataLoader, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                         num_workers=num_workers)

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
