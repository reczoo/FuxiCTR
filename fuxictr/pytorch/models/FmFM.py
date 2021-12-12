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
from .base_model import BaseModel
from ..layers import EmbeddingLayer, LR_Layer

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
        self.lr_layer = LR_Layer(feature_map, final_activation=None)
        self.upper_triange_mask = torch.triu(torch.ones(self.num_fields, self.num_fields - 1), 0).byte().to(self.device)
        self.lower_triange_mask = torch.tril(torch.ones(self.num_fields, self.num_fields - 1), -1).byte().to(self.device)
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.apply(self.init_weights)

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        field_wise_emb = feature_emb.unsqueeze(2).repeat(1, 1, self.num_fields - 1, 1)
        upper_tensor = torch.masked_select(field_wise_emb, self.upper_triange_mask.unsqueeze(-1)) \
                            .view(-1, self.interact_dim, self.embedding_dim)
        if self.field_interaction_type == "vectorized":
            upper_tensor = upper_tensor * self.interaction_weight
        elif self.field_interaction_type == "matrixed":
            upper_tensor = torch.matmul(upper_tensor.unsqueeze(2), self.interaction_weight).squeeze(2)
        lower_tensor = torch.masked_select(field_wise_emb.transpose(1, 2), self.lower_triange_mask.t().unsqueeze(-1)) \
                            .view(-1, self.interact_dim, self.embedding_dim)
        y_pred = (upper_tensor * lower_tensor).flatten(start_dim=1).sum(dim=-1, keepdim=True)
        y_pred += self.lr_layer(X)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

