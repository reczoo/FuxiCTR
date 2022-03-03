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

import h5py
import os
import logging
import numpy as np
import gc
import glob

    
def save_hdf5(data_array, data_path, key="data"):
    logging.info("Saving data to h5: " + data_path)
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))
    with h5py.File(data_path, 'w') as hf:
        hf.create_dataset(key, data=data_array)


def load_hdf5(data_path, key=None, verbose=True):
    if verbose:
        logging.info('Loading data from h5: ' + data_path)
    with h5py.File(data_path, 'r') as hf:
        if key is not None:
            data_array = hf[key][:]
        else:
            data_array = hf[list(hf.keys())[0]][:]
    return data_array


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


def build_dataset(feature_encoder, train_data=None, valid_data=None, test_data=None, valid_size=0, 
                  test_size=0, split_type="sequential", **kwargs):
    """ Build feature_map and transform h5 data """
    # Load csv data
    train_ddf = feature_encoder.read_csv(train_data)
    valid_ddf = feature_encoder.read_csv(valid_data) if valid_data else None
    test_ddf = feature_encoder.read_csv(test_data) if test_data else None
    
    # Split data for train/validation/test
    if valid_size > 0 or test_size > 0:
        train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf, valid_ddf, test_ddf, 
                                                          valid_size, test_size, split_type)
    # fit and transform train_ddf
    train_ddf = feature_encoder.preprocess(train_ddf)
    train_array = feature_encoder.fit_transform(train_ddf, **kwargs)
    block_size = int(kwargs.get("data_block_size", 0))
    if block_size > 0:
        block_id = 0
        for idx in range(0, len(train_array), block_size):
            save_hdf5(train_array[idx:(idx + block_size), :], os.path.join(feature_encoder.data_dir, 'train_part_{}.h5'.format(block_id)))
            block_id += 1
    else:
        save_hdf5(train_array, os.path.join(feature_encoder.data_dir, 'train.h5'))
    del train_array, train_ddf
    gc.collect()

    # Transfrom valid_ddf
    if valid_ddf is not None:
        valid_ddf = feature_encoder.preprocess(valid_ddf)
        valid_array = feature_encoder.transform(valid_ddf)
        if block_size > 0:
            block_id = 0
            for idx in range(0, len(valid_array), block_size):
                save_hdf5(valid_array[idx:(idx + block_size), :], os.path.join(feature_encoder.data_dir, 'valid_part_{}.h5'.format(block_id)))
                block_id += 1
        else:
            save_hdf5(valid_array, os.path.join(feature_encoder.data_dir, 'valid.h5'))
        del valid_array, valid_ddf
        gc.collect()

    # Transfrom test_ddf
    if test_ddf is not None:
        test_ddf = feature_encoder.preprocess(test_ddf)
        test_array = feature_encoder.transform(test_ddf)
        if block_size > 0:
            block_id = 0
            for idx in range(0, len(test_array), block_size):
                save_hdf5(test_array[idx:(idx + block_size), :], os.path.join(feature_encoder.data_dir, 'test_part_{}.h5'.format(block_id)))
                block_id += 1
        else:
            save_hdf5(test_array, os.path.join(feature_encoder.data_dir, 'test.h5'))
        del test_array, test_ddf
        gc.collect()
    logging.info("Transform csv data to h5 done.")


def h5_generator(feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, **kwargs):
    logging.info("Loading data...")
    if kwargs.get("data_block_size", 0) > 0: 
        from ..pytorch.data_generator import DataBlockGenerator as DataGenerator
    else:
        from ..pytorch.data_generator import DataGenerator

    train_gen = None
    valid_gen = None
    test_gen = None
    if stage in ["both", "train"]:
        train_blocks = glob.glob(train_data)
        valid_blocks = glob.glob(valid_data)
        assert len(train_blocks) > 0 and len(valid_blocks) > 0, "invalid data files or paths."
        if len(train_blocks) > 1:
            train_blocks.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(valid_blocks) > 1:
            valid_blocks.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        train_gen = DataGenerator(train_blocks, batch_size=batch_size, shuffle=shuffle, **kwargs)
        valid_gen = DataGenerator(valid_blocks, batch_size=batch_size, shuffle=False, **kwargs)
        logging.info("Train samples: total/{:d}, pos/{:.0f}, neg/{:.0f}, ratio/{:.2f}%, blocks/{:.0f}" \
                     .format(train_gen.num_samples, train_gen.num_positives, train_gen.num_negatives,
                             100. * train_gen.num_positives / train_gen.num_samples, train_gen.num_blocks))
        logging.info("Validation samples: total/{:d}, pos/{:.0f}, neg/{:.0f}, ratio/{:.2f}%, blocks/{:.0f}" \
                     .format(valid_gen.num_samples, valid_gen.num_positives, valid_gen.num_negatives,
                             100. * valid_gen.num_positives / valid_gen.num_samples, valid_gen.num_blocks))
        if stage == "train":
            logging.info("Loading train data done.")
            return train_gen, valid_gen

    if stage in ["both", "test"]:
        test_blocks = glob.glob(test_data)
        if len(test_blocks) > 0:
            if len(test_blocks) > 1:
                test_blocks.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            test_gen = DataGenerator(test_blocks, batch_size=batch_size, shuffle=False, **kwargs)
            logging.info("Test samples: total/{:d}, pos/{:.0f}, neg/{:.0f}, ratio/{:.2f}%, blocks/{:.0f}" \
                         .format(test_gen.num_samples, test_gen.num_positives, test_gen.num_negatives,
                                 100. * test_gen.num_positives / test_gen.num_samples, test_gen.num_blocks))
        if stage == "test":
            logging.info("Loading test data done.")
            return test_gen

    logging.info("Loading data done.")
    return train_gen, valid_gen, test_gen


def tfrecord_generator():
    raise NotImplementedError()


