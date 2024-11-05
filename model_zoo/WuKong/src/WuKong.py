# =========================================================================
# Copyright (C) 2024. XiaoLongtao. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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
""" This model implements the paper: Zhang et al., Wukong: Towards a Scaling Law for 
    Large-Scale Recommendation, Arxiv 2024.
    [PDF] https://arxiv.org/abs/2403.02545
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class WuKong(BaseModel):
    """
    The WuKong model class that implements factorization machines-based model.

    Args:
        feature_map: A FeatureMap instance used to store feature specs (e.g., vocab_size).
        model_id: Equivalent to model class name by default, which is used in config to determine 
            which model to call.
        gpu: gpu device used to load model. -1 means cpu (default=-1).
        learning_rate: learning rate for training (default=1e-3).
        embedding_dim: embedding dimension of features (default=64).
        num_layers: number of WuKong layers (default=3).
        compression_dim: dimension of compressed features in LCB (default=40).
        mlp_hidden_units: hidden units of MLP on top of WuKong (default=[32,32]).
        fmb_units: hidden units of FMB (default=[32,32]).
        fmb_dim: dimension of FMB output (default=40).
        project_dim: dimension of projection matrix in FMB (default=8).
        dropout_rate: dropout rate used in LCB (default=0.2).
        embedding_regularizer: regularization term used for embedding parameters (default=0).
        net_regularizer: regularization term used for network parameters (default=0).
    """
    def __init__(self,
                 feature_map,
                 model_id="WuKong",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 num_layers=3,
                 compression_dim=40,
                 mlp_hidden_units=[32,32],
                 fmb_units=[32,32],
                 fmb_dim=40,
                 project_dim=8,
                 dropout_rate=0.2,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(WuKong, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.interaction_layers = nn.ModuleList([
            WuKongLayer(feature_map.num_fields, embedding_dim, project_dim, fmb_units, fmb_dim, compression_dim,dropout_rate) for _ in range(num_layers)
            ])
        self.final_mlp = MLP_Block(input_dim=feature_map.num_fields*embedding_dim,
                                   output_dim=1,
                                   hidden_units=mlp_hidden_units,
                                   hidden_activations='relu',
                                   output_activation=None)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        for layer in self.interaction_layers:
            feature_emb = layer(feature_emb)
        y_pred = self.final_mlp(feature_emb)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class FactorizationMachineBlock(nn.Module):
    def __init__(self, num_features=14, embedding_dim=16, project_dim=8):
        super(FactorizationMachineBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.project_dim = project_dim
        self.num_features = num_features
        self.projection_matrix = nn.Parameter(torch.randn(self.num_features, self.project_dim))
    
    def forward(self, x):
        batch_size = x.size(0)
        x_fm = x.view(batch_size, self.num_features, self.embedding_dim)
        projected = torch.matmul(x_fm.transpose(1, 2), self.projection_matrix)
        fm_matrix = torch.matmul(x_fm, projected)
        return fm_matrix.view(batch_size, -1)


class FMB(nn.Module):
    def __init__(self, num_features=14, embedding_dim=16, fmb_units=[32,32], fmb_dim=40, project_dim=8):
        super(FMB, self).__init__()
        self.fm_block = FactorizationMachineBlock(num_features, embedding_dim, project_dim)
        self.layer_norm = nn.LayerNorm(num_features * project_dim)
        model_layers = [nn.Linear(num_features * project_dim, fmb_units[0]), nn.ReLU()]
        for i in range(1, len(fmb_units)):
            model_layers.append(nn.Linear(fmb_units[i-1], fmb_units[i]))
            model_layers.append(nn.ReLU())
        model_layers.append(nn.Linear(fmb_units[-1], fmb_dim))
        self.mlp = nn.Sequential(*model_layers)
    
    def forward(self, x):
        y = self.fm_block(x)
        y = self.layer_norm(y)
        y = self.mlp(y)
        y = F.relu(y)
        return y


class LinearCompressionBlock(nn.Module):
    """ Linear Compression Block (LCB) """
    def __init__(self, num_features=14, embedding_dim=16, compressed_dim=8,dropout_rate=0.2):
        super(LinearCompressionBlock, self).__init__()
        self.linear = nn.Linear(num_features * embedding_dim, compressed_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, x):
        return self.dropout(self.linear(x.view(x.size(0), -1)))


class WuKongLayer(nn.Module):
    def __init__(self, num_features=14, embedding_dim=16, project_dim=4, fmb_units=[40,40,40], fmb_dim=40, compressed_dim=40, dropout_rate=0.2):
        super(WuKongLayer, self).__init__()
        self.fmb = FMB(num_features, embedding_dim, fmb_units, fmb_dim, project_dim)
        self.lcb = LinearCompressionBlock(num_features, embedding_dim, compressed_dim, dropout_rate)
        self.layer_norm = nn.LayerNorm(num_features * embedding_dim)
        self.transform = nn.Linear(fmb_dim + compressed_dim, num_features*embedding_dim)
    
    def forward(self, x):
        fmb_out = self.fmb(x)
        lcb_out = self.lcb(x)
        concat_out = torch.cat([fmb_out, lcb_out], dim=1)
        concat_out = self.transform(concat_out)
        add_norm_out = self.layer_norm(concat_out+x.view(x.size(0), -1))
        return add_norm_out
