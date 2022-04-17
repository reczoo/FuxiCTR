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


class AOANet(BaseModel):
    """ The AOANet model
        References:
          - Lang Lang, Zhenlong Zhu, Xuanye Liu, Jianxin Zhao, Jixing Xu, Minghui Shan: 
            Architecture and Operation Adaptive Network for Online Recommendations, KDD 2021.
          - [PDF] https://dl.acm.org/doi/pdf/10.1145/3447548.3467133
    """
    def __init__(self, 
                 feature_map, 
                 model_id="AOANet", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64],
                 dnn_hidden_activations="ReLU",
                 num_interaction_layers=3,
                 num_subspaces=4,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(AOANet, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.dnn = MLP_Layer(input_dim=feature_map.num_fields * embedding_dim,
                             output_dim=None, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.gin = GeneralizedInteractionNet(num_interaction_layers, 
                                             num_subspaces, 
                                             feature_map.num_fields, 
                                             embedding_dim)
        self.fc = nn.Linear(dnn_hidden_units[-1] + num_subspaces * embedding_dim, 1)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feat_emb = self.embedding_layer(X)
        dnn_out = self.dnn(feat_emb.flatten(start_dim=1))
        interact_out = self.gin(feat_emb).flatten(start_dim=1)
        y_pred = self.fc(torch.cat([dnn_out, interact_out], dim=-1))
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class GeneralizedInteractionNet(nn.Module):
    def __init__(self, num_layers, num_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteractionNet, self).__init__()
        self.layers = nn.ModuleList([GeneralizedInteraction(num_fields if i == 0 else num_subspaces, 
                                                           num_subspaces, 
                                                           num_fields, 
                                                           embedding_dim) \
                                     for i in range(num_layers)])
    
    def forward(self, B_0):
        B_i = B_0
        for layer in self.layers:
            B_i = layer(B_0, B_i)
        return B_i
            

class GeneralizedInteraction(nn.Module):
    def __init__(self, input_subspaces, output_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteraction, self).__init__()
        self.input_subspaces = input_subspaces
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.eye(embedding_dim, embedding_dim).unsqueeze(0).repeat(output_subspaces, 1, 1))
        self.alpha = nn.Parameter(torch.ones(input_subspaces * num_fields, output_subspaces))
        self.h = nn.Parameter(torch.ones(output_subspaces, embedding_dim, 1))

    def forward(self, B_0, B_i):
        outer_product = torch.einsum("bnh,bnd->bnhd",
                                     B_0.repeat(1, self.input_subspaces, 1), 
                                     B_i.repeat(1, 1, self.num_fields).view(B_i.size(0), -1, self.embedding_dim)) # b x (field*in) x d x d 
        fusion = torch.matmul(outer_product.permute(0, 2, 3, 1), self.alpha) # b x d x d x out
        fusion = self.W * fusion.permute(0, 3, 1, 2) # b x out x d x d
        B_i = torch.matmul(fusion, self.h).squeeze(-1) # b x out x d
        return B_i
