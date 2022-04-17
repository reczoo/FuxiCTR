# =========================================================================
# Copyright (C) 2022 FuxiCTR Authors. All rights reserved.
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


class SAM(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="SAM",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 interaction_type="SAM2E", # option in ["SAM2A", "SAM2E", "SAM3A", "SAM3E"]
                 aggregation="concat",
                 num_interaction_layers=3,
                 use_residual=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 net_dropout=0,
                 **kwargs):
        super(SAM, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.block = SAMBlock(num_interaction_layers, feature_map.num_fields, embedding_dim, use_residual, 
                              interaction_type, aggregation, net_dropout)
        if aggregation == "concat":
            if interaction_type in ["SAM2A", "SAM2E"]:
                self.fc = nn.Linear(embedding_dim * (feature_map.num_fields ** 2), 1)
            else:
                self.fc = nn.Linear(feature_map.num_fields * embedding_dim, 1)
        else:
            self.fc = nn.Linear(embedding_dim, 1)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        interact_out = self.block(feature_emb)
        y_pred = self.fc(interact_out)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
        

class SAMBlock(nn.Module):
    def __init__(self, num_layers, num_fields, embedding_dim, use_residual=False, 
                 interaction_type="SAM2E", aggregation="concat", dropout=0):
        super(SAMBlock, self).__init__()
        assert aggregation in ["concat", "weighted_pooling", "mean_pooling", "sum_pooling"]
        self.aggregation = aggregation
        if self.aggregation == "weighted_pooling":
            self.weight = nn.Parameter(torch.ones(num_fields, 1))
        if interaction_type == "SAM2A":
            assert aggregation == "concat", "Only aggregation=concat is supported for SAM2A."
            self.layers = nn.ModuleList([SAM2A(num_fields, embedding_dim, dropout)])
        elif interaction_type == "SAM2E":
            assert aggregation == "concat", "Only aggregation=concat is supported for SAM2E."
            self.layers = nn.ModuleList([SAM2E(embedding_dim, dropout)])
        elif interaction_type == "SAM3A":
            self.layers = nn.ModuleList([SAM3A(num_fields, embedding_dim, use_residual, dropout) \
                                         for _ in range(num_layers)])
        elif interaction_type == "SAM3E":
            self.layers = nn.ModuleList([SAM3E(embedding_dim, use_residual, dropout) \
                                         for _ in range(num_layers)])
        else:
            raise ValueError("interaction_type={} not supported.".format(interaction_type))

    def forward(self, F):
        for layer in self.layers:
            F = layer(F)
        if self.aggregation == "concat":
            out = F.flatten(start_dim=1)
        elif self.aggregation == "weighted_pooling":
            out = (F * self.weight).sum(dim=1)
        elif self.aggregation == "mean_pooling":
            out = F.mean(dim=1)
        elif self.aggregation == "sum_pooling":
            out = F.sum(dim=1)
        return out


class SAM2A(nn.Module):
    def __init__(self, num_fields, embedding_dim, dropout=0):
        super(SAM2A, self).__init__()
        self.W = nn.Parameter(torch.ones(num_fields, num_fields, embedding_dim)) # f x f x d
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, F.transpose(1, 2)) # b x f x f
        out = S.unsqueeze(-1) * self.W # b x f x f x d
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM2E(nn.Module):
    def __init__(self, embedding_dim, dropout=0):
        super(SAM2E, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, F.transpose(1, 2)) # b x f x f
        U = torch.einsum("bnd,bmd->bnmd", F, F) # b x f x f x d
        out = S.unsqueeze(-1) * U # b x f x f x d
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM3A(nn.Module):
    def __init__(self, num_fields, embedding_dim, use_residual=True, dropout=0):
        super(SAM3A, self).__init__()
        self.W = nn.Parameter(torch.ones(num_fields, num_fields, embedding_dim)) # f x f x d
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_residual = use_residual
        if use_residual:
            self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, self.K(F).transpose(1, 2)) # b x f x f
        out = (S.unsqueeze(-1) * self.W).sum(dim=2) # b x f x d
        if self.use_residual:
            out += self.Q(F)
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM3E(nn.Module):
    def __init__(self, embedding_dim, use_residual=True, dropout=0):
        super(SAM3E, self).__init__()
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_residual = use_residual
        if use_residual:
            self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, self.K(F).transpose(1, 2)) # b x f x f
        U = torch.einsum("bnd,bmd->bnmd", F, F) # b x f x f x d
        out = (S.unsqueeze(-1) * U).sum(dim=2) # b x f x d
        if self.use_residual:
            out += self.Q(F)
        if self.dropout:
            out = self.dropout(out)
        return out

