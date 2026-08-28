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
import ray


class ParquetDataLoader(object):
    """DataLoader backed by Ray Data for distributed parquet loading.

    Ray Data handles parallel I/O, streaming execution (data larger than
    memory), and optional block-level shuffle internally — no manual
    multi-worker sharding or buffered shuffle needed.

    Args:
        feature_map (FeatureMap): Feature map that defines columns and labels.
        data_path (str): Path to a parquet file or directory of parquet files.
        split (str, optional): Data split, one of ``"train"`` or ``"test"``.
            Default: ``"train"``.
        batch_size (int, optional): Number of samples per batch. Default: ``32``.
        shuffle (bool, optional): Whether to shuffle data each epoch. Default:
            ``False``. When ``True``, a local in-memory shuffle buffer is used
            so that every ``__iter__`` call yields a different row order
            (similar to ``torch.utils.data.DataLoader(shuffle=True)``).
            Ignored when ``split == "test"``.
        num_workers (int, optional): Indicates how many batches to prefetch.
            Default: 1
        buffer_size (int, optional): Local shuffle buffer size (number of rows)
            used when ``shuffle=True``. Default: ``100000``.
        **kwargs: Additional arguments (e.g. ``streaming``)
            accepted for backward compatibility but ignored.
    """

    def __init__(self, feature_map, data_path, split="train", batch_size=32,
                 shuffle=False, num_workers=1, buffer_size=100000, **kwargs):
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.split = split
        self.shuffle = shuffle and split != "test"
        self.buffer_size = buffer_size
        self.num_workers = num_workers

        # Initialize Ray if not already running
        if not ray.is_initialized():
            # Suppress Ray's own logs so they don't mix with FuxiCTR logs.
            ray.init(ignore_reinit_error=True,
                     logging_level=logging.ERROR,
                     log_to_driver=False)
            ctx = ray.data.DataContext.get_current()
            ctx.enable_progress_bars = False
            ctx.enable_rich_progress_bars = False
            logging.info("Ray initialized for data loading.")

        self.dataset = self.load_data(data_path)

    def load_data(self, data_path):
        """Load parquet file(s) into a Ray Dataset.

        Args:
            data_path (str): Path to a parquet file, directory, or glob pattern.

        Returns:
            ray.data.Dataset: Ray Dataset with selected columns.
        """
        if not data_path.endswith(".parquet"):
            data_path = os.path.join(data_path, "*.parquet")
        parquet_files = sorted(glob.glob(data_path))  # sort by part name
        assert len(parquet_files) > 0, f"invalid data_path: {data_path}"
        self.parquet_files = parquet_files
        logging.info(f"Loading parquet files: {parquet_files}")

        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        dataset = ray.data.read_parquet(parquet_files, columns=all_cols)
        self.num_blocks = len(parquet_files)
        self.num_samples = dataset.count()
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        return dataset

    def __iter__(self):
        """Iterate over batches, yielding ``Dict[str, torch.Tensor]``.

        Each batch is a dictionary mapping column names to torch tensors,
        compatible with ``RankModel.forward(batch_data)``.
        """
        kwargs = dict(batch_size=self.batch_size, 
                      drop_last=False,
                      prefetch_batches=self.num_workers)
        if self.shuffle:
            kwargs["local_shuffle_buffer_size"] = self.buffer_size
        return iter(self.dataset.iter_torch_batches(**kwargs))

    def __len__(self):
        """Return the number of batches per epoch.

        Returns:
            int: Number of batches.
        """
        return self.num_batches
