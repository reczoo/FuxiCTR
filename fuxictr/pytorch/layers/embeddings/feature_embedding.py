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


import torch
from torch import nn
import os
import numpy as np
from collections import OrderedDict
from .pretrained_embedding import PretrainedEmbedding
from fuxictr.pytorch.torch_utils import get_initializer
from fuxictr.utils import not_in_whitelist
from fuxictr.pytorch import layers


class FeatureEmbedding(nn.Module):
    """Feature embedding layer that converts raw features into dense embeddings.

    ``FeatureEmbedding`` wraps ``FeatureEmbeddingDict`` to produce a unified embedding tensor
    from a dictionary of raw feature inputs.

    Args:
        feature_map (FeatureMap): A ``FeatureMap`` instance that provides feature metadata.
        embedding_dim (int): Dimension of the embedding vectors.
        embedding_initializer (str, optional): Initializer for embedding weights. Default:
            ``"partial(nn.init.normal_, std=1e-4)"``.
        required_feature_columns (list, optional): List of feature names to include. If None,
            all non-meta features are included. Default: ``None``.
        not_required_feature_columns (list, optional): List of feature names to exclude.
            Default: ``None``.
        use_pretrain (bool, optional): Whether to use pretrained embeddings if available.
            Default: ``True``.
        use_sharing (bool, optional): Whether to share embeddings among features that specify
            ``share_embedding``. Default: ``True``.

    Example::

        embedding_layer = FeatureEmbedding(feature_map, embedding_dim=16)
        feature_emb = embedding_layer(X, flatten_emb=False)
    """

    def __init__(self,
                 feature_map,
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True):
        super(FeatureEmbedding, self).__init__()
        self.embedding_layer = FeatureEmbeddingDict(feature_map,
                                                    embedding_dim,
                                                    embedding_initializer=embedding_initializer,
                                                    required_feature_columns=required_feature_columns,
                                                    not_required_feature_columns=not_required_feature_columns,
                                                    use_pretrain=use_pretrain,
                                                    use_sharing=use_sharing)

    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        """Forward pass to compute feature embeddings.

        Args:
            X (dict): Raw feature inputs.
            feature_source (list, optional): Filter features by source. Default: ``[]``.
            feature_type (list, optional): Filter features by type. Default: ``[]``.
            flatten_emb (bool, optional): If True, concatenates embeddings along the last dimension;
                otherwise, stacks them along a new field dimension. Default: ``False``.

        Returns:
            torch.Tensor: Feature embeddings.
        """
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        return feature_emb


class FeatureEmbeddingDict(nn.Module):
    """Dictionary-based feature embedding layer that maps each feature to its embedding module.

    ``FeatureEmbeddingDict`` manages individual embedding layers for each feature in the
    ``FeatureMap``, supports pretrained embeddings, embedding sharing, and feature encoders.

    Args:
        feature_map (FeatureMap): A ``FeatureMap`` instance that provides feature metadata.
        embedding_dim (int): Dimension of the embedding vectors.
        embedding_initializer (str, optional): Initializer for embedding weights. Default:
            ``"partial(nn.init.normal_, std=1e-4)"``.
        required_feature_columns (list, optional): List of feature names to include. If None,
            all non-meta features are included. Default: ``None``.
        not_required_feature_columns (list, optional): List of feature names to exclude.
            Default: ``None``.
        use_pretrain (bool, optional): Whether to use pretrained embeddings if available.
            Default: ``True``.
        use_sharing (bool, optional): Whether to share embeddings among features that specify
            ``share_embedding``. Default: ``True``.

    Example::

        embedding_dict = FeatureEmbeddingDict(feature_map, embedding_dim=16)
        feature_emb_dict = embedding_dict(X)
    """

    def __init__(self,
                 feature_map,
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True):
        super(FeatureEmbeddingDict, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.use_pretrain = use_pretrain
        self.embedding_initializer = get_initializer(embedding_initializer)
        self.embedding_layers = nn.ModuleDict()
        self.feature_encoders = nn.ModuleDict()
        for feature, feature_spec in self._feature_map.features.items():
            if self.is_required(feature):
                if not (use_pretrain and use_sharing) and embedding_dim == 1:
                    feat_dim = 1 # in case for LR
                    if feature_spec["type"] == "sequence":
                        self.feature_encoders[feature] = layers.MaskedSumPooling()
                else:
                    feat_dim = feature_spec.get("embedding_dim", embedding_dim)
                    if feature_spec.get("feature_encoder", None):
                        self.feature_encoders[feature] = self.get_feature_encoder(feature_spec["feature_encoder"])
                    else:
                        if feature_spec["type"] == "embedding": # add embedding projection
                            pretrain_dim = feature_spec.get("pretrain_dim", feat_dim)
                            self.feature_encoders[feature] = nn.Linear(pretrain_dim, feat_dim, bias=False)

                # Set embedding_layer according to share_embedding
                if use_sharing and feature_spec.get("share_embedding") in self.embedding_layers:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec["share_embedding"]]
                    continue

                if feature_spec["type"] == "numeric":
                    self.embedding_layers[feature] = nn.Linear(1, feat_dim, bias=False)
                elif feature_spec["type"] in ["categorical", "sequence"]:
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        pretrain_path = os.path.join(feature_map.data_dir,
                                                     feature_spec["pretrained_emb"])
                        vocab_path = os.path.join(feature_map.data_dir,
                                                  "feature_vocab.json")
                        pretrain_dim = feature_spec.get("pretrain_dim", feat_dim)
                        pretrain_usage = feature_spec.get("pretrain_usage", "init")
                        self.embedding_layers[feature] = PretrainedEmbedding(feature,
                                                                             feature_spec,
                                                                             pretrain_path,
                                                                             vocab_path,
                                                                             feat_dim,
                                                                             pretrain_dim,
                                                                             pretrain_usage,
                                                                             embedding_initializer)
                    else:
                        padding_idx = feature_spec.get("padding_idx", None)
                        self.embedding_layers[feature] = nn.Embedding(feature_spec["vocab_size"],
                                                                      feat_dim,
                                                                      padding_idx=padding_idx)
                elif feature_spec["type"] == "embedding":
                    self.embedding_layers[feature] = nn.Identity()
        self.init_weights()

    def get_feature_encoder(self, encoder):
        """Build a feature encoder from a configuration string or list.

        Args:
            encoder (str or list): Encoder configuration. Can be a module string or a list of
                module strings to be wrapped in ``nn.Sequential``.

        Returns:
            nn.Module: The constructed encoder module.

        Raises:
            ValueError: If the encoder configuration is not supported.
        """
        try:
            if type(encoder) == list:
                encoder_list = []
                for enc in encoder:
                    encoder_list.append(eval(enc))
                encoder_layer = nn.Sequential(*encoder_list)
            else:
                encoder_layer = eval(encoder)
            return encoder_layer
        except:
            raise ValueError("feature_encoder={} is not supported.".format(encoder))

    def init_weights(self):
        """Initialize embedding weights for all non-shared, non-pretrained embedding layers."""
        for k, v in self.embedding_layers.items():
            if "share_embedding" in self._feature_map.features[k]:
                continue
            if type(v) == PretrainedEmbedding: # skip pretrained
                v.init_weights()
            elif type(v) == nn.Embedding:
                if v.padding_idx is not None:
                    self.embedding_initializer(v.weight[1:, :]) # set padding_idx to zero
                else:
                    self.embedding_initializer(v.weight)

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

    def dict2tensor(self, embedding_dict, flatten_emb=False, feature_list=[], feature_source=[],
                    feature_type=[]):
        """Convert an embedding dictionary to a tensor.

        Args:
            embedding_dict (OrderedDict): Dictionary mapping feature names to embedding tensors.
            flatten_emb (bool, optional): If True, concatenates embeddings along the last dimension;
                otherwise, stacks them along a new field dimension. Default: ``False``.
            feature_list (list, optional): Filter by feature names. Default: ``[]``.
            feature_source (list, optional): Filter by feature source. Default: ``[]``.
            feature_type (list, optional): Filter by feature type. Default: ``[]``.

        Returns:
            torch.Tensor: Stacked or concatenated feature embeddings.
        """
        feature_emb_list = []
        for feature, feature_spec in self._feature_map.features.items():
            if feature_list and not_in_whitelist(feature, feature_list):
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            if feature_type and not_in_whitelist(feature_spec["type"], feature_type):
                continue
            if feature in embedding_dict:
                feature_emb_list.append(embedding_dict[feature])
        if flatten_emb:
            feature_emb = torch.cat(feature_emb_list, dim=-1)
        else:
            feature_emb = torch.stack(feature_emb_list, dim=1)
        return feature_emb

    def forward(self, inputs, feature_source=[], feature_type=[]):
        """Forward pass to compute a dictionary of feature embeddings.

        Args:
            inputs (dict): Raw feature inputs.
            feature_source (list, optional): Filter features by source. Default: ``[]``.
            feature_type (list, optional): Filter features by type. Default: ``[]``.

        Returns:
            OrderedDict: Dictionary mapping feature names to their embeddings.
        """
        feature_emb_dict = OrderedDict()
        for feature in inputs.keys():
            feature_spec = self._feature_map.features[feature]
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            if feature_type and not_in_whitelist(feature_spec["type"], feature_type):
                continue
            if feature in self.embedding_layers:
                if feature_spec["type"] == "numeric":
                    inp = inputs[feature].float().view(-1, 1)
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "embedding":
                    inp = inputs[feature].float()
                    embeddings = self.embedding_layers[feature](inp)
                else:
                    raise NotImplementedError
                if feature in self.feature_encoders:
                    embeddings = self.feature_encoders[feature](embeddings)
                feature_emb_dict[feature] = embeddings
        return feature_emb_dict
