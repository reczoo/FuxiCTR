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
import ray
from ray.data.datasource.filename_provider import FilenameProvider
from ray.data import SaveMode, TaskPoolStrategy


class SimpleFilenameProvider(FilenameProvider):
    def __init__(self, file_format="parquet"):
        self.file_format = file_format
    
    def get_filename_for_task(self, write_uuid, task_index):
        return f"part-{task_index:06d}.{self.file_format}"


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


def init_ray():
    # Initialize Ray if not already running
    if not ray.is_initialized():
        # Ensure Ray workers can import fuxictr by adding the project root to PYTHONPATH
        project_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        ray.init(ignore_reinit_error=True,
                 logging_level=logging.ERROR,
                 log_to_driver=True,
                 runtime_env={"env_vars": {"PYTHONPATH": project_root}})
        logging.info("Ray initialized for data transformation.")
        ctx = ray.data.DataContext.get_current()
        ctx.enable_progress_bars = True
        

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
    logging.info(f"Transform {split} features to integer IDs...")

    # Convert polars LazyFrame to Ray Dataset via Arrow
    init_ray()
    row_index = "row_index"
    if block_size == 0:
        ddf = ddf.with_row_index(name=row_index)
    table = ddf.collect().to_arrow()
    ds = ray.data.from_arrow(table)
    del table; gc.collect()

    # Parallel batched transform via Ray Data map_batches
    num_blocks = ds.count() // block_size if block_size > 0 else os.cpu_count() // 2
    num_workers = min(num_blocks, os.cpu_count() // 2)
    ds = ds.repartition(num_blocks=num_blocks).map_batches(
        feature_encoder.transform,
        batch_size=100000,
        batch_format="numpy",
        num_cpus=1,
        compute=TaskPoolStrategy(size=num_workers)
    )

    # Write parquet outputss
    if block_size > 0:
        data_path = os.path.join(feature_encoder.data_dir, split)
        os.makedirs(data_path, exist_ok=True)
        ds.write_parquet(
            data_path, filename_provider=SimpleFilenameProvider(), mode=SaveMode.OVERWRITE
        )
        logging.info(f"Saved {num_blocks} parquet files to: " + data_path)
    else:
        data_path = os.path.join(feature_encoder.data_dir, split + ".parquet")
        os.makedirs(feature_encoder.data_dir, exist_ok=True)
        ds = ds.sort(row_index).drop_columns(row_index)
        ds.to_pandas().to_parquet(data_path)
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
            if valid_ddf is not None:
                valid_ddf = feature_encoder.preprocess(valid_ddf)
            if test_ddf is None and test_data is not None:
                test_ddf = feature_encoder.read_data(test_data, **kwargs)
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
