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

from torch import nn
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import MLP_Layer, EmbeddingLayer, LR_Layer, ScaledDotProductAttention


class AutoInt(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(AutoInt, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=False) \
                        if use_wide else None
        self.dnn = MLP_Layer(input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case no DNN used
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embedding_dim if i == 0 else attention_dim,
                                     attention_dim=attention_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=net_dropout, 
                                     use_residual=use_residual, 
                                     use_scale=use_scale, 
                                     layer_norm=layer_norm) \
             for i in range(attention_layers)])
        self.fc = nn.Linear(feature_map.num_fields * attention_dim, 1)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        attention_out = self.self_attention(feature_emb)
        attention_out = torch.flatten(attention_out, start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(X)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False):
        super(MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None

    def forward(self, X):
        residual = X
        
        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output