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
from torch.utils import data
from fuxictr.utils import load_h5
import torch
import logging


class Dataset(data.Dataset):
    def __init__(self, feature_map, data_path):
        self.feature_map = feature_map
        self.darray = self.load_data_array(data_path)
        
    def __getitem__(self, index):
        return self.darray[index, :]
    
    def __len__(self):
        return self.darray.shape[0]

    def load_data_array(self, data_path):
        data_dict = load_h5(data_path) # dict of arrays from h5
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


class DataLoader(data.DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False, num_workers=1, **kwargs):
        if not data_path.endswith(".h5"):
            data_path += ".h5"
        self.dataset = Dataset(feature_map, data_path)
        super(DataLoader, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)
        self.num_samples = len(self.dataset)
        self.num_batches = int(np.ceil(self.num_samples * 1.0 / self.batch_size))

    def __len__(self):
        return self.num_batches


class H5DataLoader(object):
    def __init__(self, feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, **kwargs):
        logging.info("Loading data...")
        train_gen = None
        valid_gen = None
        test_gen = None
        self.stage = stage
        if stage in ["both", "train"]:
            train_gen = DataLoader(feature_map, train_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
            logging.info("Train samples: total/{:d}, blocks/{:d}".format(train_gen.num_samples, 1))
            if valid_data:  
                valid_gen = DataLoader(feature_map, valid_data, batch_size=batch_size, shuffle=False, **kwargs)
                logging.info("Validation samples: total/{:d}, blocks/{:d}".format(valid_gen.num_samples, 1))
        if stage in ["both", "test"]:
            if test_data:
                test_gen = DataLoader(feature_map, test_data, batch_size=batch_size, shuffle=False, **kwargs)
                logging.info("Test samples: total/{:d}, blocks/{:d}".format(test_gen.num_samples, 1))
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
