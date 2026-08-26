# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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


import os
import logging
import numpy as np
import gc
import polars as pl
import pyarrow.dataset as pads
from datasets import Dataset, Features, Value, Sequence
import multiprocessing as mp


def split_train_test(train_ddf=None, valid_ddf=None, test_ddf=None, valid_size=0,
                     test_size=0, split_type="sequential"):
    """Split a training DataFrame into train/validation/test sets.

    Supports sequential (by index) or random splitting.

    Args:
        train_ddf (pd.DataFrame): Full training data.
        valid_ddf (pd.DataFrame, optional): Pre-existing validation data.
        test_ddf (pd.DataFrame, optional): Pre-existing test data.
        valid_size (int or float): Validation set size. If ``< 1``, treated as
            a fraction. Default: ``0``.
        test_size (int or float): Test set size. If ``< 1``, treated as a
            fraction. Default: ``0``.
        split_type (str): ``"sequential"`` or ``"random"``. Default: ``"sequential"``.

    Returns:
        tuple: ``(train_ddf, valid_ddf, test_ddf)``.
    """
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()
    return train_ddf, valid_ddf, test_ddf


def transform(feature_encoder, ddf, split="train", block_size=0):
    """Transform features to integer IDs via ``feature_encoder.transform`` and write parquet.

    Args:
        feature_encoder (FeatureProcessor): Fitted feature processor.
        ddf (polars.LazyFrame): Input lazy frame.
        split (str): Output name relative to ``data_dir`` (file prefix when
            ``block_size == 0``, directory name when ``block_size > 0``).
        block_size (int): Rows per parquet part. ``0`` writes a single file.
            Default: ``0``.
    """
    logging.info("Transform features to integer IDs...")
    ds = Dataset.from_polars(ddf)
    num_proc = max(1, mp.cpu_count() // 2)
    ds = ds.map(feature_encoder.transform, batched=True, num_proc=num_proc)
    feature_schema = dict(ds.features)
    for feature, spec in feature_encoder.feature_map.features.items():
        if feature in feature_schema:
            ftype = spec["type"]
            if ftype in ("categorical", "meta"):
                feature_schema[feature] = Value("int64")
            elif ftype == "numeric":
                feature_schema[feature] = Value("float64")
            elif ftype == "sequence":
                feature_schema[feature] = Sequence(Value("int64"))
    ds = ds.cast(Features(feature_schema))
    if block_size > 0:
        data_path = os.path.join(feature_encoder.data_dir, split)
        os.makedirs(data_path, exist_ok=True)
        pads.write_dataset(
            ds.data.table,
            base_dir=data_path,
            basename_template="part-{i}.parquet",
            format="parquet",
            max_rows_per_file=block_size,
            max_rows_per_group=block_size,
            use_threads=True,
        )
        logging.info("Saved parquet files to: " + data_path)
    else:
        data_path = os.path.join(feature_encoder.data_dir, split + ".parquet")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        ds.to_parquet(data_path)
        logging.info("Saved parquet file to: " + data_path)


def merge_meta_ddf(feature_encoder, train_ddf, valid_ddf=None, test_ddf=None):
    """Build a lazy frame of meta columns from all splits (train/valid/test).

    Meta features (e.g. ``group_id``) serve as unique keys for grouped metrics
    such as NDCG/gAUC, so their vocabulary must cover every split to avoid
    OOV collisions. Other feature types only fit on the train split.

    Args:
        feature_encoder (FeatureProcessor): Fitted feature processor.
        train_ddf (pl.LazyFrame): Preprocessed train lazy frame.
        valid_ddf (pl.LazyFrame or None): Preprocessed valid lazy frame.
        test_ddf (pl.LazyFrame or None): Preprocessed test lazy frame.

    Returns:
        pl.LazyFrame or None: Concatenated meta columns, or None if there
        are no active meta features.
    """
    meta_cols = [col["name"] for col in feature_encoder.feature_cols
                 if col["type"] == "meta" and col.get("active", True) != False]
    if not meta_cols:
        return None
    parts = [ddf.select(meta_cols)
             for ddf in (train_ddf, valid_ddf, test_ddf) if ddf is not None]
    if len(parts) == 1:
        return parts[0]
    return pl.concat(parts)


def build_dataset(feature_encoder, train_data=None, valid_data=None, test_data=None,
                  valid_size=0, test_size=0, split_type="sequential", data_block_size=0,
                  rebuild_dataset=True, **kwargs):
    """ Build feature_map and transform data """
    if rebuild_dataset:
        feature_map_path = os.path.join(feature_encoder.data_dir, "feature_map.json")
        if os.path.exists(feature_map_path):
            logging.warn(f"Skip rebuilding {feature_map_path}. "
                + "Please delete it manually if rebuilding is required.")
        else:
            # Load data files
            train_ddf = feature_encoder.read_data(train_data, **kwargs)
            valid_ddf = None
            test_ddf = None

            # Split data for train/validation/test
            if valid_size > 0 or test_size > 0:
                valid_ddf = feature_encoder.read_data(valid_data, **kwargs)
                test_ddf = feature_encoder.read_data(test_data, **kwargs)
                # TODO: check split_train_test in lazy mode
                train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf, valid_ddf, test_ddf, 
                                                                valid_size, test_size, split_type)
            
            # fit and transform train_ddf
            train_ddf = feature_encoder.preprocess(train_ddf)
            # Ensure valid/test are loaded and preprocessed so meta vocab covers all splits
            if valid_ddf is None and valid_data is not None:
                valid_ddf = feature_encoder.read_data(valid_data, **kwargs)
            if test_ddf is None and test_data is not None:
                test_ddf = feature_encoder.read_data(test_data, **kwargs)
            if valid_ddf is not None:
                valid_ddf = feature_encoder.preprocess(valid_ddf)
            if test_ddf is not None:
                test_ddf = feature_encoder.preprocess(test_ddf)
            meta_ddf = merge_meta_ddf(feature_encoder, train_ddf, valid_ddf, test_ddf)
            feature_encoder.fit(train_ddf, rebuild_dataset=True, meta_ddf=meta_ddf, **kwargs)
            transform(feature_encoder, train_ddf, split='train', block_size=data_block_size)
            del train_ddf
            gc.collect()

            # Transfrom valid_ddf
            if valid_ddf is not None:
                transform(feature_encoder, valid_ddf, split='valid', block_size=data_block_size)
                del valid_ddf
                gc.collect()

            # Transfrom test_ddf
            if test_ddf is not None:
                transform(feature_encoder, test_ddf, split='test', block_size=data_block_size)
                del test_ddf
                gc.collect()
            logging.info("Transform csv data to parquet done.")

        if data_block_size > 0:
            train_set, valid_set, test_set = "train/", "valid/", "test/"
        else:
            train_set, valid_set, test_set = "train.parquet", "valid.parquet", "test.parquet"
        train_data, valid_data, test_data = (
            os.path.join(feature_encoder.data_dir, train_set), \
            os.path.join(feature_encoder.data_dir, valid_set), \
            os.path.join(feature_encoder.data_dir, test_set) if (
                test_data or test_size > 0) else None
        )
    
    else: # skip rebuilding data but only compute feature_map.json
        feature_encoder.fit(train_ddf=None, rebuild_dataset=False, **kwargs)
    
    # Return processed data splits
    return train_data, valid_data, test_data
