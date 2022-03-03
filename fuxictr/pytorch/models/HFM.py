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
from fuxictr.pytorch.layers import MLP_Layer, EmbeddingLayer, LR_Layer, HolographicInteractionLayer


class HFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="HFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 interaction_type="circular_convolution",
                 use_dnn=True,
                 hidden_units=[64, 64],
                 hidden_activations=["relu", "relu"],
                 batch_norm=False,
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(HFM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.lr_layer = LR_Layer(feature_map, output_activation=None)
        self.hfm_layer = HolographicInteractionLayer(feature_map.num_fields, interaction_type=interaction_type)
        self.use_dnn = use_dnn
        if self.use_dnn:
            input_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2) * embedding_dim
            self.dnn = MLP_Layer(input_dim=input_dim,
                                 output_dim=1, 
                                 hidden_units=hidden_units,
                                 hidden_activations=hidden_activations,
                                 output_activation=None,
                                 dropout_rates=net_dropout, 
                                 batch_norm=batch_norm)
        else:
            self.proj_h = nn.Linear(embedding_dim, 1, bias=False)
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
        interact_out = self.hfm_layer(feature_emb)
        if self.use_dnn:
            hfm_out = self.dnn(torch.flatten(interact_out, start_dim=1))
        else:
            hfm_out = self.proj_h(interact_out.sum(dim=1))
        y_pred = hfm_out + self.lr_layer(X)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

