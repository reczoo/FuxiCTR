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

""" 
    [WWW2021] FM2: Field-matrixed Factorization Machines for Recommender Systems
"""
import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, LR_Layer


class FmFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FmFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 regularizer=None, 
                 field_interaction_type="matrixed",
                 **kwargs):
        super(FmFM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=regularizer, 
                                   net_regularizer=regularizer,
                                   **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_fields = feature_map.num_fields
        self.interact_dim = int(self.num_fields * (self.num_fields - 1) / 2)
        self.field_interaction_type = field_interaction_type
        if self.field_interaction_type == "vectorized":
            self.interaction_weight = nn.Parameter(torch.Tensor(self.interact_dim, embedding_dim))
        elif self.field_interaction_type == "matrixed":
            self.interaction_weight = nn.Parameter(torch.Tensor(self.interact_dim, embedding_dim, embedding_dim))
        nn.init.xavier_normal_(self.interaction_weight)
        self.triu_index = torch.triu(torch.ones(self.num_fields, self.num_fields), 1).nonzero().to(self.device)
        self.lr_layer = LR_Layer(feature_map, output_activation=None)
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
        left_emb = torch.index_select(feature_emb, 1, self.triu_index[:, 0])
        right_emb = torch.index_select(feature_emb, 1, self.triu_index[:, 1])
        if self.field_interaction_type == "vectorized":
            left_emb = left_emb * self.interaction_weight
        elif self.field_interaction_type == "matrixed":
            left_emb = torch.matmul(left_emb.unsqueeze(2), self.interaction_weight).squeeze(2)
        y_pred = (left_emb * right_emb).sum(dim=-1).sum(dim=-1, keepdim=True)
        y_pred += self.lr_layer(X)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

