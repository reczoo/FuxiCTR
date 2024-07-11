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


import h5py
import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from fuxictr.tensorflow.tf_utils import get_initializer, get_regularizer
from fuxictr.tensorflow import layers
from tensorflow.keras.layers import Layer, Embedding


class FeatureEmbedding(Layer):
    def __init__(self, 
                 feature_map,
                 embedding_dim,
                 embedding_initializer="random_normal(stddev=1e-4)",
                 embedding_regularizer=None,
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 name_prefix="emb_"):
        super(FeatureEmbedding, self).__init__()
        self.embedding_layer = FeatureEmbeddingDict(feature_map, 
                                                    embedding_dim,
                                                    embedding_initializer=embedding_initializer,
                                                    embedding_regularizer=embedding_regularizer,
                                                    required_feature_columns=required_feature_columns,
                                                    not_required_feature_columns=not_required_feature_columns,
                                                    use_pretrain=use_pretrain,
                                                    use_sharing=use_sharing,
                                                    name_prefix=name_prefix)

    def call(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        return feature_emb


class FeatureEmbeddingDict(Layer):
    def __init__(self, 
                 feature_map,
                 embedding_dim,
                 embedding_initializer="random_normal(stddev=1e-4)",
                 embedding_regularizer=None,
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 name_prefix="emb_"):
        super(FeatureEmbeddingDict, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.use_pretrain = use_pretrain
        self.embedding_initializer = embedding_initializer
        self.embedding_layers = OrderedDict()
        self.feature_encoders = OrderedDict()

        for feature, feature_spec in self._feature_map.features.items():
            if self.is_required(feature):
                if not (use_pretrain and use_sharing) and embedding_dim == 1:
                    feat_emb_dim = 1 # in case for LR
                    if feature_spec["type"] == "sequence":
                        self.feature_encoders[feature] = layers.MaskedSumPooling()
                else:
                    feat_emb_dim = feature_spec.get("embedding_dim", embedding_dim)
                    if feature_spec.get("feature_encoder", None):
                        self.feature_encoders[feature] = self.get_feature_encoder(feature_spec["feature_encoder"])

                # Set embedding_layer according to share_embedding
                if use_sharing and feature_spec.get("share_embedding") in self.embedding_layers:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec["share_embedding"]]
                    continue

                if feature_spec["type"] == "numeric":
                    self.embedding_layers[feature] = tf.keras.layers.Dense(feat_emb_dim, use_bias=False)
                elif feature_spec["type"] == "categorical":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = Embedding(feature_spec["vocab_size"], 
                                                 feat_emb_dim,
                                                 embeddings_initializer=get_initializer(embedding_initializer),
                                                 embeddings_regularizer=get_regularizer(embedding_regularizer),
                                                 mask_zero=True if padding_idx == 0 else False,
                                                 input_length=1,
                                                 name=name_prefix + feature)
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix,
                                                                          feature_map, 
                                                                          feature, 
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix
                elif feature_spec["type"] == "sequence":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = Embedding(feature_spec["vocab_size"], 
                                                 feat_emb_dim,
                                                 embeddings_initializer=get_initializer(embedding_initializer),
                                                 embeddings_regularizer=get_regularizer(embedding_regularizer),
                                                 mask_zero=True if padding_idx == 0 else False,
                                                 input_length=feature_spec["max_len"],
                                                 name=name_prefix + feature)
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix, 
                                                                          feature_map, 
                                                                          feature,
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix

    def get_feature_encoder(self, encoder):
        try:
            if type(encoder) == list:
                encoder_list = []
                for enc in encoder:
                    encoder_list.append(eval(enc))
                encoder_layer = tf.keras.Sequential(*encoder_list)
            else:
                encoder_layer = eval(encoder)
            return encoder_layer
        except:
            raise ValueError("feature_encoder={} is not supported.".format(encoder))
                              
    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.features[feature]
        if feature_spec["type"] == "meta":
            return False
        elif self.required_feature_columns and (feature not in self.required_feature_columns):
            return False
        elif self.not_required_feature_columns and (feature in self.not_required_feature_columns):
            return False
        else:
            return True

    def get_pretrained_embedding(self, pretrained_path, feature_name):
        with h5py.File(pretrained_path, 'r') as hf:
            embeddings = hf[feature_name][:]
        return embeddings

    def load_pretrained_embedding(self, embedding_matrix, feature_map, feature_name, freeze=False, padding_idx=None):
        pretrained_path = os.path.join(feature_map.data_dir, feature_map.features[feature_name]["pretrained_emb"])
        embeddings = self.get_pretrained_embedding(pretrained_path, feature_name)
        if padding_idx is not None:
            embeddings[padding_idx] = np.zeros(embeddings.shape[-1])
        assert embeddings.shape[-1] == embedding_matrix.embedding_dim, \
            "{}\'s embedding_dim is not correctly set to match its pretrained_emb shape".format(feature_name)
        embedding_matrix.set_weights([embeddings])
        if freeze:
            embedding_matrix.trainable = False
        return embedding_matrix

    def dict2tensor(self, embedding_dict, feature_list=[], feature_source=[], feature_type=[], flatten_emb=False):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_list = []
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec["source"] not in feature_source:
                continue
            if feature_type and feature_spec["type"] not in feature_type:
                continue
            if feature_list and feature not in feature_list:
                continue
            if feature in embedding_dict:
                feature_emb_list.append(embedding_dict[feature])
        if flatten_emb:
            feature_emb = tf.squeeze(tf.concat(feature_emb_list, axis=-1), axis=1)
        else:
            feature_emb = tf.concat(feature_emb_list, axis=1)
        return feature_emb

    def call(self, inputs, feature_source=[], feature_type=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec["source"] not in feature_source:
                continue
            if feature_type and feature_spec["type"] not in feature_type:
                continue
            if feature in self.embedding_layers:
                if feature_spec["type"] == "numeric":
                    inp = tf.reshape(inputs[feature], (-1, 1))
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = inputs[feature]
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = inputs[feature]
                    embeddings = self.embedding_layers[feature](inp)
                else:
                    raise NotImplementedError
                if feature in self.feature_encoders:
                    embeddings = self.feature_encoders[feature](embeddings)
                feature_emb_dict[feature] = embeddings
        return feature_emb_dict

