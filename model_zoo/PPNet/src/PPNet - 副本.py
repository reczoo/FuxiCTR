# =========================================================================
# Copyright (C) 2022. FuxiCTR Authors. All rights reserved.
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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation


class PPNet(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="PPNet", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10,
                 gate_features=[],
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(PPNet, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.gate_embed_layer = FeatureEmbedding(feature_map, embedding_dim, 
                                                 required_feature_columns=gate_features)
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        gate_input_dim = feature_map.sum_emb_out_dim() + len(gate_features) * embedding_dim
        self.ppn = PPNetBlock(input_dim=feature_map.sum_emb_out_dim(),
                              gate_input_dim=gate_input_dim,
                              hidden_units=hidden_units,
                              hidden_activations=hidden_activations,
                              dropout_rates=net_dropout,
                              batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        gate_emb = self.gate_embed_layer(X)
        y_aux = self.dnn(feature_emb.flatten(start_dim=1))
        y_pred = self.ppn(feature_emb.detach().flatten(start_dim=1), 
                          gate_emb.flatten(start_dim=1))
        y_aux = self.output_activation(y_aux)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred, "y_aux": y_aux}
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        loss_aux = self.loss_fn(return_dict["y_aux"], y_true, reduction='mean')
        return loss + loss_aux


class GateNN(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidden_dim=None,
                 hidden_activation="ReLU",
                 dropout=0.0,
                 batch_norm=False):
        super(GateNN, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        gate_layers = [nn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            gate_layers.append(nn.BatchNorm1d(hidden_dim))
        gate_layers.append(get_activation(hidden_activation))
        if dropout > 0:
            gate_layers.append(nn.Dropout(dropout))
        gate_layers.append(nn.Linear(hidden_dim, output_dim))
        self.gate = nn.Sequential(*gate_layers)

    def forward(self, inputs):
        return self.gate(inputs).sigmoid() * 2


class PPNetBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 gate_input_dim,
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 dropout_rates=0.0,
                 batch_norm=False,
                 use_bias=True):
        super(PPNetBlock, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        self.gate_layers = nn.ModuleList()
        self.dnn_layers = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers = []
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
            self.gate_layers.append(GateNN(gate_input_dim, output_dim=hidden_units[idx]))
            self.dnn_layers.append(nn.Sequential(*dense_layers))
        self.gate_layers.append(GateNN(gate_input_dim, output_dim=hidden_units[-1]))
        self.dnn_layers.append(nn.Linear(hidden_units[-1], 1, bias=use_bias))
    
    def forward(self, detached_feat_emb, flat_gate_emb):
        gate_input = torch.cat([detached_feat_emb, flat_gate_emb], dim=-1)
        h = detached_feat_emb
        for i in range(len(self.dnn_layers)):
            g = self.gate_layers[i](gate_input)
            h = self.dnn_layers[i](h * g)
        return h

