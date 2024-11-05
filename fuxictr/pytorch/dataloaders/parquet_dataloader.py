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


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import polars as pl


class ParquetDataset(Dataset):
    def __init__(self, feature_map, data_path):
        self.feature_map = feature_map
        self.darray = self.load_data(data_path)
        
    def __getitem__(self, index):
        return self.darray[index, :]
    
    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path):
        df = pl.read_parquet(data_path)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        data_arrays = []
        for col in all_cols:
            if df[col].dtype != pl.List:
                array = np.array(df[col])
            else:
                array = df[col].explode().to_numpy().reshape(df.shape[0], -1)
            data_arrays.append(array)
        return np.column_stack(data_arrays)


class ParquetDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(feature_map, data_path)
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches


class BatchCollator(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        batch_dict = dict()
        for col in all_cols:
            batch_dict[col] = batch_tensor[:, self.feature_map.get_column_index(col)]
        return batch_dict
