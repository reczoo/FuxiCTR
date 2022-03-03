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
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import MLP_Layer, EmbeddingLayer, ScaledDotProductAttention


class InterHAt(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="InterHAt", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_dim=None,
                 order=2,
                 num_heads=1,
                 attention_dim=10,
                 hidden_units=[64, 64],
                 hidden_activations="relu",
                 batch_norm=False,
                 layer_norm=True,
                 use_residual=True,
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(InterHAt, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.order = order
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.multi_head_attention = MultiHeadSelfAttention(embedding_dim, 
                                                           attention_dim, 
                                                           num_heads,
                                                           dropout_rate=net_dropout,
                                                           use_residual=use_residual,
                                                           use_scale=True,
                                                           layer_norm=layer_norm)
        self.feedforward = FeedForwardNetwork(embedding_dim, 
                                              hidden_dim=hidden_dim,
                                              layer_norm=layer_norm, 
                                              use_residual=use_residual)
        self.aggregation_layers = nn.ModuleList([AttentionalAggregation(embedding_dim, hidden_dim) 
                                                 for _ in range(order)])
        self.attentional_score = AttentionalAggregation(embedding_dim, hidden_dim)
        self.mlp = MLP_Layer(input_dim=embedding_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        X0 = self.embedding_layer(X)
        X1 = self.feedforward(self.multi_head_attention(X0))
        X_p = X1
        agg_u = []
        for p in range(self.order):
            u_p = self.aggregation_layers[p](X_p) # b x emb
            agg_u.append(u_p)
            if p != self.order - 1:
                X_p = u_p.unsqueeze(1) * X1 + X_p
        U = torch.stack(agg_u, dim=1) # b x order x emb
        u_f = self.attentional_score(U)
        y_pred = self.mlp(u_f)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class MultiHeadAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False):
        super(MultiHeadAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim // num_heads
        self.attention_dim = attention_dim
        self.output_dim = num_heads * attention_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = attention_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, self.output_dim, bias=False)
        if input_dim != self.output_dim:
            self.W_res = nn.Linear(self.output_dim, input_dim, bias=False)
        else:
            self.W_res = None
        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, query, key, value, mask=None):
        residual = query
        
        # linear projection
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.attention_dim)
        key = key.view(batch_size * self.num_heads, -1, self.attention_dim)
        value = value.view(batch_size * self.num_heads, -1, self.attention_dim)
        if mask:
            mask = mask.repeat(self.num_heads, 1, 1)
        # scaled dot product attention
        output, attention = self.dot_product_attention(query, key, value, self.scale, mask)
        # concat heads
        output = output.view(batch_size, -1, self.output_dim)
        # final linear projection
        if self.W_res is not None:
            output = self.W_res(output)
        output = output.relu()
        if self.dropout is not None:
            output = self.dropout(output)
        if self.use_residual:
            output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        return output, attention


class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, X):
        output, attention = super(MultiHeadSelfAttention, self).forward(X, X, X)
        return output


class AttentionalAggregation(nn.Module):
    '''
    agg attention for InterHAt
    '''
    def __init__(self, embedding_dim, hidden_dim=None):
        super(AttentionalAggregation, self).__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim
        self.agg = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), 
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1, bias=False),
                                 nn.Softmax(dim=1))

    def forward(self, X):
        # X: b x f x emb
        attentions = self.agg(X) # b x f x 1
        attention_out = (attentions * X).sum(dim=1) # b x emb
        return attention_out


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, layer_norm=True, use_residual=True):
        super(FeedForwardNetwork, self).__init__()
        self.use_residual = use_residual
        if hidden_dim is None:
            hidden_dim = 4 * input_dim
        self.ffn = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, input_dim))
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None

    def forward(self, X):
        output = self.ffn(X)
        if self.use_residual:
            output += X
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        return output

