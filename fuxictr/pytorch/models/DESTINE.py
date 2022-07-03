# =========================================================================
# Copyright (C) 2022 Huawei Technologies Co., Ltd. All rights reserved.
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
#
# Copyright (C) 2021 Big Data and Multi-modal Computing Group, CRIPAC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# =========================================================================

import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, MLP_Layer, LR_Layer


class DESTINE(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DESTINE", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 attention_dim=16,
                 num_heads=2,
                 attention_layers=2,
                 dnn_hidden_units=[], 
                 dnn_activations="ReLU",
                 net_dropout=0.1,
                 att_dropout=0.1,
                 relu_before_att=False,
                 batch_norm=False, 
                 use_scale=False,
                 use_wide=True,
                 residual_mode="each_layer", # ['last_layer', 'each_layer', None]
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DESTINE, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.lr = LR_Layer(feature_map, output_activation=None) if use_wide else None
        self.dnn = MLP_Layer(input_dim=feature_map.num_fields * embedding_dim,
                             output_dim=1, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case that only DESTINE is used
        self.self_attns = nn.ModuleList([
            DisentangledSelfAttention(embedding_dim if i == 0 else attention_dim,
                                      attention_dim, 
                                      num_heads,
                                      att_dropout,
                                      residual_mode=="each_layer",
                                      use_scale,
                                      relu_before_att) \
            for i in range(attention_layers)])
        self.attn_fc = nn.Linear(feature_map.num_fields * attention_dim, 1)
        if residual_mode == "last_layer":
            self.W_res = nn.Linear(embedding_dim, attention_dim)
        else:
            self.W_res = None
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        cross_X = feature_emb
        for self_attn in self.self_attns:
            cross_X = self_attn(cross_X, cross_X, cross_X)
        if self.W_res is not None:
            cross_X += self.W_res(feature_emb)
        y_pred = self.attn_fc(cross_X.flatten(start_dim=1))
        if self.lr is not None:
            y_pred += self.lr(X)
        if self.dnn is not None:
            y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

        
class DisentangledSelfAttention(nn.Module):
    """ Disentangle self-attention for DESTINE. The implementation is a bit different from what is 
        described in the paper, but exactly follows the code from the authors:
        https://github.com/CRIPAC-DIG/DESTINE/blob/c68e182aa220b444df73286e5e928e8a072ba75e/layers/activation.py#L90
    """
    def __init__(self, embedding_dim, attention_dim=64, num_heads=1, dropout_rate=0.1,
                 use_residual=True, use_scale=False, relu_before_att=False):
        super(DisentangledSelfAttention, self).__init__()
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_scale = use_scale
        self.relu_before_att = relu_before_att

        self.W_q = nn.Linear(embedding_dim, self.attention_dim)
        self.W_k = nn.Linear(embedding_dim, self.attention_dim)
        self.W_v = nn.Linear(embedding_dim, self.attention_dim)
        self.W_unary = nn.Linear(embedding_dim, num_heads)

        if use_residual:
            self.W_res = nn.Linear(embedding_dim, self.attention_dim)
        else:
            self.W_res = None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, query, key, value):
        residual = query
        unary = self.W_unary(key) # [batch, num_fields, num_heads]
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        if self.relu_before_att:
            query = query.relu()
            key = key.relu()
            value = value.relu()

        # split heads to [batch * num_heads, num_fields, head_dim]
        batch_size = query.size(0)
        query = torch.cat(query.split(split_size=self.head_dim, dim=2), dim=0)
        key = torch.cat(key.split(split_size=self.head_dim, dim=2), dim=0)
        value = torch.cat(value.split(split_size=self.head_dim, dim=2), dim=0)

        # whiten
        mu_query = query - query.mean(dim=1, keepdim=True)
        mu_key = key - key.mean(dim=1, keepdim=True)
        pair_weights = torch.bmm(mu_query, mu_key.transpose(1, 2))
        if self.use_scale:
            pair_weights /= self.head_dim ** 0.5
        pair_weights = F.softmax(pair_weights, dim=2) # [num_heads * batch, num_fields, num_fields]

        unary_weights = F.softmax(unary, dim=1)
        unary_weights = unary_weights.view(batch_size * self.num_heads, -1, 1)
        unary_weights = unary_weights.transpose(1, 2) # [num_heads * batch, 1, num_fields]
        
        attn_weights = pair_weights + unary_weights
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, value)
        output = torch.cat(output.split(batch_size, dim=0), dim=2)

        if self.W_res is not None:
            output += self.W_res(residual)
        return output