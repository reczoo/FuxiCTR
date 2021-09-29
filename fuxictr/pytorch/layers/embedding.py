# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

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
                 feature_types=["numeric", "categorical", "sequence"]):
        super(EmbeddingLayer, self).__init__()
        self._feature_map = feature_map
        self._feature_types = feature_types
        self.embedding_layer = nn.ModuleDict()
        self.seq_encoder_layer = nn.ModuleDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if feature_spec["type"] == "numeric" and "numeric" in feature_types:
                self.embedding_layer.update({feature: nn.Linear(1, embedding_dim, bias=False)})
            elif feature_spec["type"] == "categorical" and "categorical" in feature_types:
                self.embedding_layer.update({feature: nn.Embedding(feature_spec["vocab_size"], 
                                                                   embedding_dim, 
                                                                   padding_idx=feature_spec["vocab_size"] - 1)})
            elif feature_spec["type"] == "sequence" and "sequence" in feature_types:
                self.embedding_layer.update({feature: nn.Embedding(feature_spec["vocab_size"], 
                                                                   embedding_dim, 
                                                                   padding_idx=feature_spec["vocab_size"] - 1)})
                if feature_spec["encoder"] != "":
                    try:
                        self.seq_encoder_layer.update({feature: getattr(sequence, feature_spec["encoder"])()})
                    except:
                        raise RuntimeError("Sequence encoder={} is not supported.".format(feature_spec["encoder"]))
                else:
                    self.seq_encoder_layer.update({feature: sequence.MaskedAveragePooling()})

    def forward(self, X):
        feature_emb_list = [] 
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if feature_spec["type"] == "numeric" and "numeric" in self._feature_types:
                inp = X[:, feature_spec["index"]].float().view(-1, 1)
                embedding_vec = self.embedding_layer[feature](inp)
            elif feature_spec["type"] == "categorical" and "categorical" in self._feature_types:
                inp = X[:, feature_spec["index"]].long()
                embedding_vec = self.embedding_layer[feature](inp)
            elif feature_spec["type"] == "sequence" and "sequence" in self._feature_types:   
                inp = X[:, feature_spec["index"]].long()
                seq_embed_matrix = self.embedding_layer[feature](inp)
                embedding_vec = self.seq_encoder_layer[feature](seq_embed_matrix)
            feature_emb_list.append(embedding_vec)
        return feature_emb_list


class EmbeddingLayer_v2(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim,
                 embedding_dropout=0,
                 feature_types=["numeric", "categorical", "sequence"]):
        super(EmbeddingLayer_v2, self).__init__()
        self._feature_map = feature_map
        self._feature_types = feature_types
        self.embedding_layer = nn.ModuleDict()
        self.seq_encoder_layer = nn.ModuleDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if feature_spec["type"] == "numeric" and "numeric" in feature_types:
                self.embedding_layer.update({feature: nn.Linear(1, embedding_dim, bias=False)})
            elif feature_spec["type"] == "categorical" and "categorical" in feature_types:
                self.embedding_layer.update({feature: nn.Embedding(feature_spec["vocab_size"], 
                                                                   embedding_dim, 
                                                                   padding_idx=feature_spec["vocab_size"] - 1)})
            elif feature_spec["type"] == "sequence" and "sequence" in feature_types:
                self.embedding_layer.update({feature: nn.Embedding(feature_spec["vocab_size"], 
                                                                   embedding_dim, 
                                                                   padding_idx=feature_spec["vocab_size"] - 1)})
                if feature_spec["encoder"] != "":
                    try:
                        self.seq_encoder_layer.update({feature: getattr(sequence, feature_spec["encoder"])()})
                    except:
                        raise RuntimeError("Sequence encoder={} is not supported.".format(feature_spec["encoder"]))
                else:
                    self.seq_encoder_layer.update({feature: sequence.MaskedAveragePooling()})
        self.dropout = nn.Dropout2d(embedding_dropout) if embedding_dropout > 0 else None

    def forward(self, X):
        feature_emb_list = []
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if feature_spec["type"] == "numeric" and "numeric" in self._feature_types:
                inp = X[:, feature_spec["index"]].float().view(-1, 1)
                embedding_vec = self.embedding_layer[feature](inp)
            elif feature_spec["type"] == "categorical" and "categorical" in self._feature_types:
                inp = X[:, feature_spec["index"]].long()
                embedding_vec = self.embedding_layer[feature](inp)
            elif feature_spec["type"] == "sequence" and "sequence" in self._feature_types:   
                inp = X[:, feature_spec["index"]].long()
                seq_embed_matrix = self.embedding_layer[feature](inp)
                embedding_vec = self.seq_encoder_layer[feature](seq_embed_matrix)
            feature_emb_list.append(embedding_vec)
        feature_emb = torch.stack(feature_emb_list, dim=1)
        if self.dropout is not None:
            feature_emb = self.dropout(feature_emb)
        return feature_emb


class EmbeddingLayer_v3(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim,
                 embedding_dropout=0,
                 required_feature_columns=[],
                 not_required_feature_columns=[]):
        super(EmbeddingLayer_v3, self).__init__()
        self.embedding_layer = EmbeddingDictLayer(feature_map, 
                                                  embedding_dim,
                                                  required_feature_columns,
                                                  not_required_feature_columns)
        self.dropout = nn.Dropout2d(embedding_dropout) if embedding_dropout > 0 else None

    def forward(self, X):
        feature_emb_dict = self.embedding_layer(X)
        feature_emb = torch.stack(self.embedding_layer.dict2list(feature_emb_dict), dim=1)
        if self.dropout is not None:
            feature_emb = self.dropout(feature_emb)
        return feature_emb


class EmbeddingDictLayer(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim, 
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
                feat_emb_dim = feature_spec.get("embedding_dim", embedding_dim)
                if feature_spec["type"] == "numeric":
                    if feature not in self.embedding_layer:
                        self.embedding_layer[feature] = nn.Linear(1, feat_emb_dim, bias=False)
                elif feature_spec["type"] == "categorical":
                    if feature not in self.embedding_layer:
                        padding_idx = feature_spec.get("padding_idx", None)
                        embedding_matrix = nn.Embedding(feature_spec["vocab_size"], 
                                                        feat_emb_dim, 
                                                        padding_idx=padding_idx)
                        if "pretrained_emb" in feature_spec:
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
                        if "pretrained_emb" in feature_spec:
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

    def dict2list(self, embedding_dict):
        return list(embedding_dict.values())

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


class SENET_Layer(nn.Module):
    def __init__(self, num_fields, reduction_ratio=3):
        super(SENET_Layer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                        nn.ReLU(),
                                        nn.Linear(reduced_size, num_fields, bias=False),
                                        nn.ReLU())

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V

