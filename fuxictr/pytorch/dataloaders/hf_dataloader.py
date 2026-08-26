# =========================================================================
# Copyright (C) 2026. The FuxiCTR Library. All rights reserved.
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
import glob
import logging
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
from datasets import load_dataset
import huggingface_hub.constants
huggingface_hub.constants.HF_HUB_OFFLINE = True


class HFDataLoader(DataLoader):
    """DataLoader backed by HuggingFace datasets + torch DataLoader for batching,
    shuffling, and multi-worker loading.

    Args:
        feature_map (FeatureMap): Feature map that defines columns and labels.
        data_path (str): Path to a parquet file or directory of parquet files.
        split (str, optional): Data split, one of ``"train"`` or ``"test"``.
            Default: ``"train"``.
        batch_size (int, optional): Number of samples per batch. Default: ``32``.
        shuffle (bool, optional): Whether to shuffle the data. Default: ``False``.
            When ``streaming=True``, shuffling is performed by HuggingFace
            buffered shuffle (see ``buffer_size``) and torch-level
            shuffle is disabled (torch does not support ``shuffle=True`` for
            iterable datasets).
        num_workers (int, optional): Number of DataLoader worker processes.
            Default: ``1``. With ``streaming=True``, HuggingFace automatically
            shards the iterable across workers to avoid duplicate samples.
            Forced to ``1`` when ``split == "test"`` for deterministic order.
        streaming (bool, optional): Whether to load data as a streaming
            ``IterableDataset`` (no full in-memory materialization). The number
            of samples is obtained from the parquet metadata via the HuggingFace
            dataset builder. Default: ``False``.
        buffer_size (int, optional): Buffer size for buffered shuffle
            when ``streaming=True`` and ``shuffle=True``. Ignored otherwise.
            Default: ``100000``.
        **kwargs: Additional arguments passed to ``DataLoader``.
    """

    def __init__(self, feature_map, data_path, split="train", batch_size=32,
                 shuffle=False, num_workers=1, buffer_size=100000, 
                 streaming=False, **kwargs):
        self.feature_map = feature_map
        self.streaming = streaming
        self.batch_size = batch_size
        dataset = self.load_data(data_path, streaming=streaming)
        if split == "test":
            num_workers = 1  # keep deterministic order for test
        if streaming and shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
            shuffle = False
        super().__init__(dataset=dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers)

    def load_data(self, data_path, streaming=False):
        """Load parquet file(s) into a HuggingFace dataset.

        Args:
            data_path (str): Path to a parquet file or directory of parquet files.
            streaming (bool, optional): Whether to return a streaming
                ``IterableDataset`` instead of a memory-mapped ``Dataset``.
                Default: ``False``.

        Returns:
            datasets.Dataset or datasets.IterableDataset: HuggingFace dataset
            formatted as torch tensors.
        """
        if not data_path.endswith(".parquet"):
            data_path = os.path.join(data_path, "*.parquet")
        parquet_files = sorted(glob.glob(data_path))  # sort by part name
        assert len(parquet_files) > 0, f"invalid data_path: {data_path}"
        self.num_blocks = len(parquet_files)
        self.parquet_files = parquet_files
        logging.info(f"Loading parquet files: {parquet_files}")
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        dataset = load_dataset("parquet", data_files=parquet_files, split="train",
                               columns=all_cols, streaming=streaming).with_format("torch")
        self.num_samples = self._count_samples(parquet_files) if streaming else len(dataset)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        return dataset

    def _count_samples(self, data_files):
        """Return the number of samples by reading parquet footer metadata.

        Used in streaming mode, where the ``IterableDataset`` does not expose
        ``__len__``. Reads only the parquet footer (no data, no cache).

        Returns:
            int: Number of samples across the given parquet files.
        """
        return sum(pq.ParquetFile(f).metadata.num_rows for f in data_files)

    def __len__(self):
        """Return the number of batches per epoch.

        Returns:
            int: Number of batches.
        """
        return self.num_batches
