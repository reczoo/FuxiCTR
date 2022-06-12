# =========================================================================
# Copyright (C) 2022 FuxiCTR Authors. All rights reserved.
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
from fuxictr.pytorch.layers import EmbeddingLayer, MLP_Layer
from fuxictr.pytorch.torch_utils import get_activation


class MaskNet(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="MaskNet",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[64,64,64],
                 dnn_hidden_activations="ReLU",
                 model_type="SerialMaskNet",
                 parallel_num_blocks=1,
                 parallel_block_dim=64,
                 reduction_ratio=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 net_dropout=0,
                 emb_layernorm=True,
                 net_layernorm=True,
                 **kwargs):
        super(MaskNet, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        if model_type == "SerialMaskNet":
            self.mask_net = SerialMaskNet(input_dim=feature_map.num_fields * embedding_dim,
                                          output_dim=1,
                                          output_activation=self.get_output_activation(task),
                                          hidden_units=dnn_hidden_units,
                                          hidden_activations=dnn_hidden_activations,
                                          reduction_ratio=reduction_ratio,
                                          dropout_rates=net_dropout,
                                          layer_norm=net_layernorm)
        elif model_type == "ParallelMaskNet":
            self.mask_net = ParallelMaskNet(input_dim=feature_map.num_fields * embedding_dim,
                                            output_dim=1,
                                            output_activation=self.get_output_activation(task),
                                            num_blocks=parallel_num_blocks, 
                                            block_dim=parallel_block_dim, 
                                            hidden_units=dnn_hidden_units,
                                            hidden_activations=dnn_hidden_activations,
                                            reduction_ratio=reduction_ratio,
                                            dropout_rates=net_dropout,
                                            layer_norm=net_layernorm)
        self.num_fields = feature_map.num_fields
        if emb_layernorm:
            self.emb_norm = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(self.num_fields))
        else:
            self.emb_norm = None
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        if self.emb_norm is not None:
            feat_list = feature_emb.chunk(self.num_fields, dim=1)
            V_hidden = torch.cat([self.emb_norm[i](feat) for i, feat in enumerate(feat_list)], dim=1)
        else:
            V_hidden = feature_emb
        y_pred = self.mask_net(feature_emb.flatten(start_dim=1), V_hidden.flatten(start_dim=1))
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
        

class SerialMaskNet(nn.Module):
    def __init__(self, input_dim, output_dim=None, output_activation=None, hidden_units=[], 
                 hidden_activations="ReLU", reduction_ratio=1, dropout_rates=0, layer_norm=True):
        super(SerialMaskNet, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.hidden_units = [input_dim] + hidden_units
        self.mask_blocks = nn.ModuleList()
        for idx in range(len(self.hidden_units) - 1):
            self.mask_blocks.append(MaskBlock(input_dim, 
                                              self.hidden_units[idx], 
                                              self.hidden_units[idx + 1], 
                                              hidden_activations[idx], 
                                              reduction_ratio, 
                                              dropout_rates[idx],
                                              layer_norm))
        fc_layers = []
        if output_dim is not None:
            fc_layers.append(nn.Linear(self.hidden_units[-1], output_dim))
        if output_activation is not None:
            fc_layers.append(get_activation(output_activation))
        self.fc = None
        if len(fc_layers) > 0:
            self.fc = nn.Sequential(*fc_layers)

    def forward(self, V_emb, V_hidden):
        v_out = V_hidden
        for idx in range(len(self.hidden_units) - 1):
            v_out = self.mask_blocks[idx](V_emb, v_out)
        if self.fc is not None:
            v_out = self.fc(v_out)
        return v_out


class ParallelMaskNet(nn.Module):
    def __init__(self, input_dim, output_dim=None, output_activation=None, num_blocks=1, block_dim=64, 
                 hidden_units=[], hidden_activations="ReLU", reduction_ratio=1, dropout_rates=0, 
                 layer_norm=True):
        super(ParallelMaskNet, self).__init__()
        self.num_blocks = num_blocks
        self.mask_blocks = nn.ModuleList([MaskBlock(input_dim, 
                                                    input_dim, 
                                                    block_dim, 
                                                    hidden_activations, 
                                                    reduction_ratio, 
                                                    dropout_rates,
                                                    layer_norm) for _ in range(num_blocks)])

        self.dnn = MLP_Layer(input_dim=block_dim * num_blocks,
                              output_dim=output_dim, 
                              hidden_units=hidden_units,
                              hidden_activations=hidden_activations,
                              output_activation=output_activation,
                              dropout_rates=dropout_rates)

    def forward(self, V_emb, V_hidden):
        block_out = []
        for i in range(self.num_blocks):
            block_out.append(self.mask_blocks[i](V_emb, V_hidden))
        concat_out = torch.cat(block_out, dim=-1)
        v_out = self.dnn(concat_out)
        return v_out


class MaskBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation="ReLU", reduction_ratio=1, 
                 dropout_rate=0, layer_norm=True):
        super(MaskBlock, self).__init__()
        self.mask_layer = nn.Sequential(nn.Linear(input_dim, int(hidden_dim * reduction_ratio)),
                                        nn.ReLU(),
                                        nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim))
        hidden_layers = [nn.Linear(hidden_dim, output_dim, bias=False)]
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            hidden_layers.append(nn.Dropout(p=dropout_rate))
        self.hidden_layer = nn.Sequential(*hidden_layers)

    def forward(self, V_emb, V_hidden):
        V_mask = self.mask_layer(V_emb)
        v_out = self.hidden_layer(V_mask * V_hidden)
        return v_out
