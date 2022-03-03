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


import numpy as np
from torch.utils import data
from ..datasets.data_utils import load_hdf5
import h5py
from itertools import chain
import torch

class Dataset(data.Dataset):
    def __init__(self, darray):
        self.darray = darray
        
    def __getitem__(self, index):
        X = self.darray[index, 0:-1]
        y = self.darray[index, -1]
        return X, y
    
    def __len__(self):
        return self.darray.shape[0]


class DataGenerator(data.DataLoader):
    def __init__(self, data_path, batch_size=32, shuffle=False, num_workers=1, **kwargs):
        if type(data_path) == list:
            data_path = data_path[0]
        data_array = load_hdf5(data_path)
        self.dataset = Dataset(data_array)
        super(DataGenerator, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(len(self.dataset) * 1.0 / self.batch_size))
        self.num_samples = len(data_array)
        self.num_positives = data_array[:, -1].sum()
        self.num_negatives = self.num_samples - self.num_positives

    def __len__(self):
        return self.num_batches


class DataBlockGenerator(object):
    def __init__(self, data_block_list, batch_size=32, shuffle=False, **kwargs):
        # data_block_list: path list of data blocks
        self.data_blocks = data_block_list
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_blocks = len(self.data_blocks)
        self.num_batches, self.num_samples, self.num_positives, self.num_negatives \
            = self.count_batches_and_samples()

    def iter_block(self, data_block):
        darray = load_hdf5(data_block, verbose=False)
        X = torch.from_numpy(darray[:, 0:-1])
        y = torch.from_numpy(darray[:, -1])
        block_size = len(y)
        indexes = list(range(block_size))
        if self.shuffle:
            np.random.shuffle(indexes)
        for idx in range(0, block_size, self.batch_size):
            batch_index = indexes[idx:(idx + self.batch_size)]
            yield X[batch_index], y[batch_index]

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.data_blocks)
        return chain.from_iterable(map(self.iter_block, self.data_blocks))

    def __len__(self):
        return self.num_batches

    def count_batches_and_samples(self):
        num_samples = 0
        num_positives = 0
        num_batches = 0
        for block_path in self.data_blocks:
            with h5py.File(block_path, 'r') as hf:
                data_array = hf[list(hf.keys())[0]][:]
                num_samples += len(data_array)
                num_positives += np.sum(data_array[:, -1])
                num_batches += int(np.ceil(len(data_array) * 1.0 / self.batch_size))
        num_negatives = num_samples - num_positives
        return num_batches, num_samples, num_positives, num_negatives

