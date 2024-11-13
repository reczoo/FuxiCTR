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


import torch
from torch import nn
import os
import io
import json
import numpy as np
import logging
from fuxictr.pytorch.torch_utils import get_initializer
from fuxictr.preprocess.tokenizer import load_pretrain_emb


class PretrainedEmbedding(nn.Module):
    def __init__(self,
                 feature_name,
                 feature_spec,
                 pretrain_path,
                 vocab_path,
                 embedding_dim,
                 pretrain_dim,
                 pretrain_usage="init",
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)"):
        """
        Fusion pretrained embedding with ID embedding
        :param: fusion_type: init/sum/concat
        """
        super().__init__()
        assert pretrain_usage in ["init", "sum", "concat"]
        self.pretrain_usage = pretrain_usage
        self.embedding_initializer = get_initializer(embedding_initializer)
        padding_idx = feature_spec.get("padding_idx", None)
        self.oov_idx = feature_spec["oov_idx"]
        self.freeze_emb = feature_spec["freeze_emb"]
        self.pretrain_embedding = self.load_pretrained_embedding(feature_spec["vocab_size"],
                                                                 pretrain_dim,
                                                                 pretrain_path,
                                                                 vocab_path,
                                                                 feature_name,
                                                                 freeze=self.freeze_emb,
                                                                 padding_idx=padding_idx)
        if pretrain_usage != "init":
            self.id_embedding = nn.Embedding(feature_spec["vocab_size"],
                                             embedding_dim,
                                             padding_idx=padding_idx)
        self.proj = None
        if pretrain_usage in ["init", "sum"] and embedding_dim != pretrain_dim:
            self.proj = nn.Linear(pretrain_dim, embedding_dim, bias=False)
        if pretrain_usage == "concat":
            self.proj = nn.Linear(pretrain_dim + embedding_dim, embedding_dim, bias=False)

    def init_weights(self):
        if self.pretrain_usage in ["sum", "concat"]:
            nn.init.zeros_(self.id_embedding.weight) # set oov token embeddings to zeros
            self.embedding_initializer(self.id_embedding.weight[1:self.oov_idx, :])

    def load_feature_vocab(self, vocab_path, feature_name):
        with io.open(vocab_path, "r", encoding="utf-8") as fd:
            vocab = json.load(fd)
            vocab_type = type(list(vocab.items())[1][0]) # get key dtype
        return vocab[feature_name], vocab_type

    def load_pretrained_embedding(self, vocab_size, pretrain_dim, pretrain_path, vocab_path,
                                  feature_name, freeze=False, padding_idx=None):
        embedding_layer = nn.Embedding(vocab_size,
                                       pretrain_dim,
                                       padding_idx=padding_idx)
        if freeze:
            embedding_matrix = np.zeros((vocab_size, pretrain_dim))
        else:
            embedding_matrix = np.random.normal(loc=0, scale=1.e-4, size=(vocab_size, pretrain_dim))
            if padding_idx:
                embedding_matrix[padding_idx, :] = np.zeros(pretrain_dim) # set as zero for PAD
        logging.info("Loading pretrained_emb: {}".format(pretrain_path))
        keys, embeddings = load_pretrain_emb(pretrain_path, keys=["key", "value"])
        assert embeddings.shape[-1] == pretrain_dim, f"pretrain_dim={pretrain_dim} not correct."
        vocab, vocab_type = self.load_feature_vocab(vocab_path, feature_name)
        keys = keys.astype(vocab_type) # ensure the same dtype between pretrained keys and vocab keys
        for idx, word in enumerate(keys):
            if word in vocab:
                embedding_matrix[vocab[word]] = embeddings[idx]
        embedding_layer.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).float())
        if freeze:
            embedding_layer.weight.requires_grad = False
        return embedding_layer

    def forward(self, inputs):
        mask = (inputs <= self.oov_idx).float()
        pretrain_emb = self.pretrain_embedding(inputs)
        if not self.freeze_emb:
            pretrain_emb = pretrain_emb * mask.unsqueeze(-1)
        if self.pretrain_usage == "init":
            if self.proj is not None:
                feature_emb = self.proj(pretrain_emb)
            else:
                feature_emb = pretrain_emb
        else:
            id_emb = self.id_embedding(inputs)
            id_emb = id_emb * mask.unsqueeze(-1)
            if self.pretrain_usage == "sum":
                if self.proj is not None:
                    feature_emb = self.proj(pretrain_emb) + id_emb
                else:
                    feature_emb = pretrain_emb + id_emb
            elif self.pretrain_usage == "concat":
                feature_emb = torch.cat([pretrain_emb, id_emb], dim=-1)
                feature_emb = self.proj(feature_emb)
        return feature_emb
