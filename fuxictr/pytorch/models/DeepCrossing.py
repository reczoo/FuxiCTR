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
from fuxictr.pytorch.layers import EmbeddingLayer
from fuxictr.pytorch.torch_utils import get_activation


class DeepCrossing(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepCrossing", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 residual_blocks=[64, 64, 64],
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DeepCrossing, self).__init__(feature_map, 
                                           model_id=model_id, 
                                           gpu=gpu, 
                                           embedding_regularizer=embedding_regularizer, 
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(residual_blocks)
        layers = []
        input_dim = feature_map.num_fields * embedding_dim
        for hidden_dim, hidden_activation in zip(residual_blocks, hidden_activations):
            layers.append(ResidualBlock(input_dim, 
                                        hidden_dim,
                                        hidden_activation,
                                        net_dropout,
                                        use_residual,
                                        batch_norm))
        layers.append(nn.Linear(input_dim, 1))
        self.crossing_layer = nn.Sequential(*layers) # * used to unpack list
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.crossing_layer(feature_emb.flatten(start_dim=1))
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class ResidualBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_activation="ReLU",
                 dropout_rate=0,
                 use_residual=True,
                 batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.activation_layer = get_activation(hidden_activation)
        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   self.activation_layer,
                                   nn.Linear(hidden_dim, input_dim))
        self.use_residual = use_residual
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, X):
        X_out = self.layer(X)
        if self.use_residual:
            X_out = X_out + X
        if self.batch_norm is not None:
            X_out = self.batch_norm(X_out)
        output = self.activation_layer(X_out)
        if self.dropout is not None:
            output = self.dropout(output)
        return output



