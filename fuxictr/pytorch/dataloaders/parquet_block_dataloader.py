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
from itertools import chain
from torch.utils.data.dataloader import default_collate
from torch.utils.data import IterDataPipe, DataLoader, get_worker_info
import glob
import polars as pl
import pandas as pd
import os


class ParquetIterDataPipe(IterDataPipe):
    """Iterable data pipe that yields individual rows from Parquet data blocks.

    Args:
        data_blocks (list[str]): List of paths to Parquet files.
        feature_map (FeatureMap): Feature map that defines columns and labels.
    """

    def __init__(self, data_blocks, feature_map):
        self.feature_map = feature_map
        self.data_blocks = data_blocks

    def load_data(self, data_path):
        """Load data from a Parquet file and stack columns into a 2-D array.

        Args:
            data_path (str): Path to the Parquet file.

        Returns:
            np.ndarray: Stacked array of shape (num_samples, num_columns).
        """
        df = pd.read_parquet(data_path)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        data_arrays = []
        for col in all_cols:
            if df[col].dtype == "object":
                array = np.array(df[col].to_list())
            else:
                array = df[col].to_numpy()
            data_arrays.append(array)
        return np.column_stack(data_arrays)

    def read_block(self, data_block):
        """Yield individual rows from a single Parquet block.

        Args:
            data_block (str): Path to the Parquet file.

        Yields:
            np.ndarray: A single row from the data block.
        """
        darray = self.load_data(data_block)
        for idx in range(darray.shape[0]):
            yield darray[idx, :]

    def __iter__(self):
        """Iterate over all data blocks with optional multi-worker sharding.

        Yields:
            np.ndarray: A single row from the dataset.
        """
        worker_info = get_worker_info()
        if worker_info is None: # single-process data loading
            block_list = self.data_blocks
        else: # in a worker process
            block_list = [
                block
                for idx, block in enumerate(self.data_blocks)
                if idx % worker_info.num_workers == worker_info.id
            ]
        return chain.from_iterable(map(self.read_block, block_list))


class ParquetBlockDataLoader(DataLoader):
    """DataLoader for Parquet data blocks with optional shuffling and multi-worker support.

    Args:
        feature_map (FeatureMap): Feature map that defines columns and labels.
        data_path (str): Path to a single Parquet file or a directory of Parquet files.
        split (str, optional): Data split, one of ``"train"`` or ``"test"``.
            Default: ``"train"``.
        batch_size (int, optional): Number of samples per batch. Default: ``32``.
        shuffle (bool, optional): Whether to shuffle the data. Default: ``False``.
        num_workers (int, optional): Number of worker processes. Default: ``1``.
        buffer_size (int, optional): Shuffle buffer size when ``shuffle=True``.
            Default: ``100000``.
        **kwargs: Additional arguments passed to ``DataLoader``.
    """

    def __init__(self, feature_map, data_path, split="train", batch_size=32, shuffle=False,
                 num_workers=1, buffer_size=100000, **kwargs):
        if not data_path.endswith("parquet"):
            data_path = os.path.join(data_path, "*.parquet")
        data_blocks = sorted(glob.glob(data_path)) # sort by part name
        assert len(data_blocks) > 0, f"invalid data_path: {data_path}"
        self.data_blocks = data_blocks
        self.num_blocks = len(self.data_blocks)
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.num_batches, self.num_samples = self.count_batches_and_samples()
        datapipe = ParquetIterDataPipe(self.data_blocks, feature_map)
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=buffer_size)
        elif split == "test":
            num_workers = 1 # multiple workers cannot keep the order of data reading
        super().__init__(dataset=datapipe,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map))

    def __len__(self):
        """Return the number of batches per epoch.

        Returns:
            int: Number of batches.
        """
        return self.num_batches

    def count_batches_and_samples(self):
        """Count total samples and derive the number of batches.

        Returns:
            tuple: ``(num_batches, num_samples)``.
        """
        num_samples = 0
        for data_block in self.data_blocks:
            df = pl.scan_parquet(data_block)
            num_samples += df.select(pl.count()).collect().item()
        num_batches = int(np.ceil(num_samples / self.batch_size))
        return num_batches, num_samples


class BatchCollator(object):
    """Collate a batch of rows into a dictionary of column tensors.

    Args:
        feature_map (FeatureMap): Feature map that defines columns and labels.
    """

    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, batch):
        """Collate a list of rows into a batched dictionary.

        Args:
            batch (list[np.ndarray]): List of rows.

        Returns:
            dict[str, torch.Tensor]: Mapping from column name to batched tensor.
        """
        batch_tensor = default_collate(batch)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        batch_dict = dict()
        for col in all_cols:
            batch_dict[col] = batch_tensor[:, self.feature_map.get_column_index(col)]
        return batch_dict
