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


import os
from .npz_dataloader import NpzDataLoader
from .npz_block_dataloader import NpzBlockDataLoader
from .parquet_dataloader import ParquetDataLoader
import logging


def RankDataLoader(feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                   batch_size=32, shuffle=True, data_format="npz", **kwargs):
    """Create train/validation/test generators for ranking tasks.

    Selects the appropriate underlying ``DataLoader`` based on the
    ``data_format`` (``npz`` or ``parquet``). For ``npz`` format, a directory
    path selects ``NpzBlockDataLoader`` for block-wise loading, while a file path
    selects ``NpzDataLoader``.

    Args:
        feature_map (FeatureMap): Feature map that defines columns and labels.
        stage (str, optional): Stage to load, one of ``"both"``, ``"train"``, or ``"test"``.
            Default: ``"both"``.
        train_data (str, optional): Path to training data. Default: ``None``.
        valid_data (str, optional): Path to validation data. Default: ``None``.
        test_data (str, optional): Path to test data. Default: ``None``.
        batch_size (int, optional): Number of samples per batch. Default: ``32``.
        shuffle (bool, optional): Whether to shuffle training data. Default: ``True``.
        data_format (str, optional): Data format, one of ``"npz"`` or ``"parquet"``.
        **kwargs: Additional arguments passed to the underlying ``DataLoader``.

    Returns:
        tuple or DataLoader: Depending on ``stage``:
            - ``"train"``: ``(train_gen, valid_gen)``
            - ``"test"``: ``test_gen``
            - ``"both"``: ``(train_gen, valid_gen, test_gen)``
    """
    logging.info("Loading datasets...")
    train_gen = None
    valid_gen = None
    test_gen = None
    if kwargs.get("data_loader"):
        DataLoader = kwargs["data_loader"]
    else:
        if data_format == "npz":
            data_path = train_data or valid_data or test_data
            if data_path and os.path.isdir(data_path):
                DataLoader = NpzBlockDataLoader
            else:
                DataLoader = NpzDataLoader
        elif data_format in ("parquet", "csv"):
            DataLoader = ParquetDataLoader
        else:
            raise ValueError(f"data_format={data_format} not supported.")
    if stage in ["both", "train"]:
        train_gen = DataLoader(feature_map, train_data, split="train", batch_size=batch_size,
                               shuffle=shuffle, **kwargs)
        logging.info(
            "Train samples: total/{:d}, blocks/{:d}"
            .format(train_gen.num_samples, train_gen.num_blocks)
        )
        if valid_data:
            valid_gen = DataLoader(feature_map, valid_data, split="valid",
                                   batch_size=batch_size, shuffle=False, **kwargs)
            logging.info(
                "Validation samples: total/{:d}, blocks/{:d}"
                .format(valid_gen.num_samples, valid_gen.num_blocks)
            )

    if stage in ["both", "test"]:
        if test_data:
            test_gen = DataLoader(feature_map, test_data, split="test", batch_size=batch_size,
                                  shuffle=False, **kwargs)
            logging.info(
                "Test samples: total/{:d}, blocks/{:d}"
                .format(test_gen.num_samples, test_gen.num_blocks)
            )

    if stage == "train":
        logging.info("Loading train and validation data done.")
        return train_gen, valid_gen
    elif stage == "test":
        logging.info("Loading test data done.")
        return test_gen
    else:
        logging.info("Loading data done.")
        return train_gen, valid_gen, test_gen
