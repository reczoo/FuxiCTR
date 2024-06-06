# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import tensorflow as  tf
import logging


class TFRecordDataLoader(object):
    def __init__(self, feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, drop_remainder=False, **kwargs):
        logging.info("Loading data...")
        self.stage = stage
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.schema = dict()
        for feat, feat_spec in feature_map.features.items():
            if feat_spec["type"] == "numeric":
                self.schema[feat] = tf.io.FixedLenFeature(dtype=tf.float32, shape=1)
            elif feat_spec["type"] in ["categorical", "meta"]:
                self.schema[feat] = tf.io.FixedLenFeature(dtype=tf.int64, shape=1)
            elif feat_spec["type"] == "sequence":
                self.schema[feat] = tf.io.FixedLenFeature(dtype=tf.int64, shape=feat_spec["max_len"])
        for label in feature_map.labels:
            self.schema[label] = tf.io.FixedLenFeature(dtype=tf.float32, shape=1)

    def input_fn(self, filenames, batch_size=32, shuffle=True):
        def parse_example(example):
            example_dict = tf.io.parse_single_example(example, features=self.schema)
            return example_dict
        dataset = tf.data.TFRecordDataset(filenames).map(parse_example, num_parallel_calls=1)
        dataset = dataset.prefetch(buffer_size=1).batch(batch_size, drop_remainder=self.drop_remainder)
        if shuffle:
            dataset = dataset.shuffle(batch_size * 10)
        return dataset

    def make_iterator(self):
        if self.stage == "train":
            logging.info("Loading train and validation data done.")
            return self.input_fn(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle), \
                   self.input_fn(self.valid_data, batch_size=self.batch_size, shuffle=False)
        elif self.stage == "test":
            logging.info("Loading test data done.")
            return self.input_fn(self.test_data, batch_size=self.batch_size, shuffle=False)
        else:
            logging.info("Loading data done.")
            return self.input_fn(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle), \
                   self.input_fn(self.valid_data, batch_size=self.batch_size, shuffle=False), \
                   self.input_fn(self.test_data, batch_size=self.batch_size, shuffle=False)

