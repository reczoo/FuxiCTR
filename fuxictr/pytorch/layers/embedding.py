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

import torch
from torch import nn
import h5py
import os
import numpy as np
from collections import OrderedDict
from . import sequence


class EmbeddingLayer(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim,
                 embedding_dropout=0,
                 load_pretrain=True,
                 required_feature_columns=[],
                 not_required_feature_columns=[]):
        super(EmbeddingLayer, self).__init__()
        self.embedding_layer = EmbeddingDictLayer(feature_map, 
                                                  embedding_dim,
                                                  load_pretrain=load_pretrain,
                                                  required_feature_columns=required_feature_columns,
                                                  not_required_feature_columns=not_required_feature_columns)
        self.dropout = nn.Dropout2d(embedding_dropout) if embedding_dropout > 0 else None

    def forward(self, X):
        feature_emb_dict = self.embedding_layer(X)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        if self.dropout is not None:
            feature_emb = self.dropout(feature_emb)
        return feature_emb


class EmbeddingDictLayer(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim, 
                 embedding_dim_dict={},
                 load_pretrain=True,
                 required_feature_columns=[],
                 not_required_feature_columns=[]):
        super(EmbeddingDictLayer, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.embedding_layer = nn.ModuleDict()
        self.seq_encoder_layer = nn.ModuleDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if self.is_required(feature):
                # Set embedding_layer according to share_embedding
                if "share_embedding" in feature_spec:
                    self.embedding_layer[feature] = self.embedding_layer[feature_spec["share_embedding"]]
                feat_emb_dim = embedding_dim_dict.get(feature, embedding_dim)
                if embedding_dim == 1:
                    feat_emb_dim = embedding_dim # keep embedding_dim=1 for LR
                if feature_spec["type"] == "numeric":
                    if feature not in self.embedding_layer:
                        self.embedding_layer[feature] = nn.Linear(1, feat_emb_dim, bias=False)
                elif feature_spec["type"] == "categorical":
                    if feature not in self.embedding_layer:
                        padding_idx = feature_spec.get("padding_idx", None)
                        embedding_matrix = nn.Embedding(feature_spec["vocab_size"], 
                                                        feat_emb_dim, 
                                                        padding_idx=padding_idx)
                        if load_pretrain and "pretrained_emb" in feature_spec:
                            embeddings = self.get_pretrained_embedding(feature_map.data_dir, feature, feature_spec)
                            embedding_matrix = self.set_pretrained_embedding(embedding_matrix, embeddings, 
                                                                             freeze=feature_spec["freeze_emb"],
                                                                             padding_idx=padding_idx)
                        self.embedding_layer[feature] = embedding_matrix
                elif feature_spec["type"] == "sequence":
                    if feature not in self.embedding_layer:
                        padding_idx = feature_spec["vocab_size"] - 1
                        embedding_matrix = nn.Embedding(feature_spec["vocab_size"], 
                                                        feat_emb_dim, 
                                                        padding_idx=padding_idx)
                        if load_pretrain and "pretrained_emb" in feature_spec:
                            embeddings = self.get_pretrained_embedding(feature_map.data_dir, feature, feature_spec)
                            embedding_matrix = self.set_pretrained_embedding(embedding_matrix, embeddings, 
                                                                             freeze=feature_spec["freeze_emb"],
                                                                             padding_idx=padding_idx)
                        self.embedding_layer[feature] = embedding_matrix
                    self.set_sequence_encoder(feature, feature_spec.get("encoder", None))

    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.feature_specs[feature]
        if len(self.required_feature_columns) > 0 and (feature not in self.required_feature_columns):
            return False
        elif feature in self.not_required_feature_columns:
            return False
        else:
            return True

    def set_sequence_encoder(self, feature, encoder):
        if encoder is None or encoder in ["none", "null"]:
            self.seq_encoder_layer.update({feature: None})
        elif encoder == "MaskedAveragePooling":
            self.seq_encoder_layer.update({feature: sequence.MaskedAveragePooling()})
        elif encoder == "MaskedSumPooling":
            self.seq_encoder_layer.update({feature: sequence.MaskedSumPooling()})
        else:
            raise RuntimeError("Sequence encoder={} is not supported.".format(encoder))

    def get_pretrained_embedding(self, data_dir, feature_name, feature_spec):
        pretrained_path = os.path.join(data_dir, feature_spec["pretrained_emb"])
        with h5py.File(pretrained_path, 'r') as hf:
            embeddings = hf[feature_name][:]
        return embeddings

    def set_pretrained_embedding(self, embedding_matrix, embeddings, freeze=False, padding_idx=None):
        if padding_idx is not None:
            embeddings[padding_idx] = np.zeros(embeddings.shape[-1])
        embeddings = torch.from_numpy(embeddings).float()
        embedding_matrix.weight = torch.nn.Parameter(embeddings)
        if freeze:
            embedding_matrix.weight.requires_grad = False
        return embedding_matrix

    def dict2tensor(self, embedding_dict, feature_source=None, feature_type=None):
        if feature_source is not None:
            if not isinstance(feature_source, list):
                feature_source = [feature_source]
            feature_emb_list = []
            for feature, feature_spec in self._feature_map.feature_specs.items():
                if feature_spec["source"] in feature_source:
                    feature_emb_list.append(embedding_dict[feature])
            return torch.stack(feature_emb_list, dim=1)
        elif feature_type is not None:
            if not isinstance(feature_type, list):
                feature_type = [feature_type]
            feature_emb_list = []
            for feature, feature_spec in self._feature_map.feature_specs.items():
                if feature_spec["type"] in feature_type:
                    feature_emb_list.append(embedding_dict[feature])
            return torch.stack(feature_emb_list, dim=1)
        else:
            return torch.stack(list(embedding_dict.values()), dim=1)

    def forward(self, X):
        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if feature in self.embedding_layer:
                if feature_spec["type"] == "numeric":
                    inp = X[:, feature_spec["index"]].float().view(-1, 1)
                    embedding_vec = self.embedding_layer[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = X[:, feature_spec["index"]].long()
                    embedding_vec = self.embedding_layer[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = X[:, feature_spec["index"]].long()
                    seq_embed_matrix = self.embedding_layer[feature](inp)
                    if self.seq_encoder_layer[feature] is not None:
                        embedding_vec = self.seq_encoder_layer[feature](seq_embed_matrix)
                    else:
                        embedding_vec = seq_embed_matrix
                feature_emb_dict[feature] = embedding_vec
        return feature_emb_dict




