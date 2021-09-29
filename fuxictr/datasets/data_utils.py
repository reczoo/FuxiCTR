# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import h5py
import os
import logging
import numpy as np

class DataIO(object):
    def __init__(self, feature_encoder, data_format='csv'):
        self.feature_encoder = feature_encoder
        self.data_format = data_format

    def load_data(self, data_path, use_hdf5=True):
        if self.data_format == 'h5':
            data_array = self.load_hdf5(data_path)
            return data_array
        elif self.data_format == 'csv':
            hdf5_file = os.path.join(self.feature_encoder.data_dir, 
                                     os.path.splitext(os.path.basename(data_path))[0] + '.h5')
            if use_hdf5 and os.path.exists(hdf5_file):
                try:
                    data_array = self.load_hdf5(hdf5_file)
                    return data_array
                except:
                    logging.info('Loading h5 file failed, reloading from {}'.format(data_path))
            ddf = self.feature_encoder.read_csv(data_path)
            data_array = self.feature_encoder.transform(ddf)
            if use_hdf5:
                self.save_hdf5(data_array, hdf5_file)
        return data_array

    def save_hdf5(self, data_array, data_path, key="data"):
        logging.info("Saving data to h5: " + data_path)
        if not os.path.exists(os.path.dirname(data_path)):
            os.makedirs(os.path.dirname(data_path))
        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset(key, data=data_array)

    def load_hdf5(self, data_path, key="data"):
        logging.info('Loading data from h5: ' + data_path)
        with h5py.File(data_path, 'r') as hf:
            data_array = hf[key][:]
        return data_array

def data_generator(feature_encoder, stage="both", train_data=None, valid_data=None, test_data=None,
                   validation_samples=0, split_type="sequential", batch_size=32, shuffle=True, 
                   use_hdf5=True, neg_samples=-1, data_format='csv', **kwargs):
    logging.info("Loading data...")
    # Choose different DataGenerator versions
    if feature_encoder.version == 'tensorflow':
        from ..tensorflow.data_generator import DataGenerator
    elif feature_encoder.version == 'pytorch':
        from ..pytorch.data_generator import DataGenerator
    train_gen = None
    valid_gen = None
    test_gen = None
    data_io = DataIO(feature_encoder, data_format)
    if stage in ["both", "train"]:
        train_array =  data_io.load_data(train_data, use_hdf5=use_hdf5)
        num_samples = len(train_array)
        if valid_data:
            valid_array = data_io.load_data(valid_data, use_hdf5=use_hdf5)
            validation_samples = len(valid_array)
            train_samples = num_samples
        else:
            if validation_samples < 1:
                validation_samples = int(num_samples * validation_samples)
            train_samples = num_samples - validation_samples
            instance_IDs = np.arange(num_samples)
            if split_type == "random":
                np.random.shuffle(instance_IDs)
            valid_array = train_array[instance_IDs[train_samples:], :]
            train_array = train_array[instance_IDs[0:train_samples], :]
        train_gen = DataGenerator(train_array, batch_size=batch_size, shuffle=shuffle, neg_samples=neg_samples, **kwargs)
        valid_gen = DataGenerator(valid_array, batch_size=batch_size, shuffle=False, neg_samples=-1, **kwargs)
        logging.info("Train samples: total/{:d}, pos/{:.0f}, neg/{:.0f}, ratio/{:.2f}%" \
                     .format(train_samples, train_array[:, -1].sum(), train_samples-train_array[:, -1].sum(),
                        100 * train_array[:, -1].sum() / train_samples))
        logging.info("Validation samples: total/{:d}, pos/{:.0f}, neg/{:.0f}, ratio/{:.2f}%" \
                     .format(validation_samples, valid_array[:, -1].sum(), validation_samples-valid_array[:, -1].sum(),
                             100 * valid_array[:, -1].sum() / validation_samples))
        if stage == "train":
            logging.info("Loading train data done.")
            return train_gen, valid_gen

    if stage in ["both", "test"]:
        test_array = data_io.load_data(test_data, use_hdf5=use_hdf5)
        test_samples = len(test_array)
        test_gen = DataGenerator(test_array, batch_size=batch_size, shuffle=False, neg_samples=-1, **kwargs)
        logging.info("Test samples: total/{:d}, pos/{:.0f}, neg/{:.0f}, ratio/{:.2f}%" \
                     .format(test_samples, test_array[:, -1].sum(), test_samples-test_array[:, -1].sum(),
                             100 * test_array[:, -1].sum() / test_samples))
        if stage == "test":
            logging.info("Loading test data done.")
            return test_gen

    logging.info("Loading data done.")
    return train_gen, valid_gen, test_gen

def hdf5_generator(feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                   validation_samples=0, split_type="sequential", batch_size=32, shuffle=True, 
                   neg_samples=-1, data_format='h5', **kwargs):
    return data_generator(feature_map, stage=stage, train_data=train_data, valid_data=valid_data, 
                          test_data=test_data, validation_samples=validation_samples, 
                          split_type=split_type, batch_size=batch_size, shuffle=shuffle, 
                          use_hdf5=use_hdf5, neg_samples=neg_samples, data_format=data_format, **kwargs)

def tf_record_generator():
    raise NotImplementedError()