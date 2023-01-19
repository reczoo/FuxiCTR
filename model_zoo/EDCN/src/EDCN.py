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
from fuxictr.pytorch.layers import FeatureEmbedding, CrossInteraction, MLP_Block


class EDCN(BaseModel):
    """ The EDCN model
        References:
          - Bo Chen, Yichao Wang, Zhirong Liu, Ruiming Tang, Wei Guo, Hongkun Zheng, Weiwei Yao, Muyu Zhang, 
            Xiuqiang He: Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel 
            Deep CTR Models, CIKM 2021.
          - [PDF] https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf
          - [Code] https://github.com/mindspore-ai/models/blob/master/research/recommend/EDCN/src/edcn.py 
    """
    def __init__(self, 
                 feature_map, 
                 model_id="EDCN", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 num_cross_layers=3,
                 hidden_activations="ReLU",
                 bridge_type="hadamard_product",
                 temperature=1,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(EDCN, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        hidden_dim = feature_map.num_fields * embedding_dim
        self.dense_layers = nn.ModuleList([MLP_Block(input_dim=hidden_dim, # MLP = linear + activation + dropout
                                                     output_dim=None, 
                                                     hidden_units=[hidden_dim],
                                                     hidden_activations=hidden_activations,
                                                     output_activation=None,
                                                     dropout_rates=net_dropout, 
                                                     batch_norm=False) \
                                           for _ in range(num_cross_layers)])
        self.cross_layers = nn.ModuleList([CrossInteraction(hidden_dim) for _ in range(num_cross_layers)])
        self.bridge_modules = nn.ModuleList([BridgeModule(hidden_dim, bridge_type) for _ in range(num_cross_layers)])
        self.regulation_modules = nn.ModuleList([RegulationModule(feature_map.num_fields, 
                                                                  embedding_dim,
                                                                  tau=temperature,
                                                                  use_bn=batch_norm) \
                                                 for _ in range(num_cross_layers)])
        self.fc = nn.Linear(hidden_dim * 3, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feat_emb = self.embedding_layer(X)
        cross_i, deep_i = self.regulation_modules[0](feat_emb.flatten(start_dim=1))
        cross_0 = cross_i
        for i in range(len(self.cross_layers)):
            cross_i = self.cross_layers[i](cross_0, cross_i)
            deep_i = self.dense_layers[i](deep_i)
            bridge_i = self.bridge_modules[i](cross_i, deep_i)
            if i + 1 < len(self.cross_layers):
                cross_i, deep_i = self.regulation_modules[i + 1](bridge_i)
        y_pred = self.fc(torch.cat([cross_i, deep_i, bridge_i], dim=-1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class BridgeModule(nn.Module):
    def __init__(self, hidden_dim, bridge_type="hadamard_product"):
        super(BridgeModule, self).__init__()
        assert bridge_type in ["hadamard_product", "pointwise_addition", "concatenation", "attention_pooling"],\
               "bridge_type={} is not supported.".format(bridge_type)
        self.bridge_type = bridge_type
        if bridge_type == "concatenation":
            self.concat_pooling = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), 
                                                nn.ReLU())
        elif bridge_type == "attention_pooling":
            self.attention1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim, bias=False),
                                            nn.Softmax(dim=-1))
            self.attention2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim, bias=False),
                                            nn.Softmax(dim=-1))
    
    def forward(self, X1, X2):
        out = None
        if self.bridge_type == "hadamard_product":
            out = X1 * X2
        elif self.bridge_type == "pointwise_addition":
            out = X1 + X2
        elif self.bridge_type == "concatenation":
            out = self.concat_pooling(torch.cat([X1, X2], dim=-1))
        elif self.bridge_type == "attention_pooling":
            out = self.attention1(X1) * X1 + self.attention1(X2) * X2
        return out
            

class RegulationModule(nn.Module):
    def __init__(self, num_fields, embedding_dim, tau=1, use_bn=False):
        super(RegulationModule, self).__init__()
        self.tau = tau
        self.embedding_dim = embedding_dim
        self.use_bn = use_bn
        self.g1 = nn.Parameter(torch.ones(num_fields))
        self.g2 = nn.Parameter(torch.ones(num_fields))
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(num_fields * embedding_dim)
            self.bn2 = nn.BatchNorm1d(num_fields * embedding_dim)
    
    def forward(self, X):
        g1 = (self.g1 / self.tau).softmax(dim=-1).unsqueeze(-1).repeat(1, self.embedding_dim).view(1, -1)
        g2 = (self.g2 / self.tau).softmax(dim=-1).unsqueeze(-1).repeat(1, self.embedding_dim).view(1, -1)
        out1, out2 = g1 * X, g2 * X
        if self.use_bn:
            out1, out2 = self.bn1(out1), self.bn2(out2)
        return out1, out2
