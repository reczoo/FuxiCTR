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
import multiprocessing as mp
import polars as pl


def split_train_test(train_ddf=None, valid_ddf=None, test_ddf=None, valid_size=0, 
                     test_size=0, split_type="sequential"):
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


def transform_block(feature_encoder, df_block, filename):
    df_block = feature_encoder.transform(df_block)
    data_path = os.path.join(feature_encoder.data_dir, filename)
    logging.info("Saving data to parquet: " + data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df_block.to_parquet(data_path, index=False, engine="pyarrow")


def transform(feature_encoder, ddf, filename, block_size=0):
    ddf = ddf.collect().to_pandas()
    if block_size > 0:
        pool = mp.Pool(mp.cpu_count() // 2)
        block_id = 0
        for idx in range(0, len(ddf), block_size):
            df_block = ddf.iloc[idx:(idx + block_size)]
            pool.apply_async(
                transform_block,
                args=(feature_encoder, df_block,
                      '{}/part_{:05d}.parquet'.format(filename, block_id))
            )
            block_id += 1
        pool.close()
        pool.join()
    else:
        transform_block(feature_encoder, ddf, filename + ".parquet")


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
            feature_encoder.fit(train_ddf, rebuild_dataset=True, **kwargs)
            transform(feature_encoder, train_ddf, 'train', block_size=data_block_size)
            del train_ddf
            gc.collect()

            # Transfrom valid_ddf
            if valid_ddf is None and (valid_data is not None):
                valid_ddf = feature_encoder.read_data(valid_data, **kwargs)
            if valid_ddf is not None:
                valid_ddf = feature_encoder.preprocess(valid_ddf)
                transform(feature_encoder, valid_ddf, 'valid', block_size=data_block_size)
                del valid_ddf
                gc.collect()

            # Transfrom test_ddf
            if test_ddf is None and (test_data is not None):
                test_ddf = feature_encoder.read_data(test_data, **kwargs)
            if test_ddf is not None:
                test_ddf = feature_encoder.preprocess(test_ddf)
                transform(feature_encoder, test_ddf, 'test', block_size=data_block_size)
                del test_ddf
                gc.collect()
            logging.info("Transform csv data to parquet done.")

        train_data, valid_data, test_data = (
            os.path.join(feature_encoder.data_dir, "train"), \
            os.path.join(feature_encoder.data_dir, "valid"), \
            os.path.join(feature_encoder.data_dir, "test") if (
                test_data or test_size > 0) else None
        )
    
    else: # skip rebuilding data but only compute feature_map.json
        feature_encoder.fit(train_ddf=None, rebuild_dataset=False, **kwargs)
    
    # Return processed data splits
    return train_data, valid_data, test_data
