# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
from fuxictr.pytorch.layers import FeatureEmbedding, LogisticRegression


class FmFM(BaseModel):
    """ The FmFM model
        Reference:
        - FM2: Field-matrixed Factorization Machines for Recommender Systems, WWW'2021.
    """
    def __init__(self, 
                 feature_map, 
                 model_id="FmFM", 
                 gpu=-1, 
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
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        interact_dim = int(num_fields * (num_fields - 1) / 2)
        self.field_interaction_type = field_interaction_type
        if self.field_interaction_type == "vectorized":
            self.interaction_weight = nn.Parameter(torch.Tensor(interact_dim, embedding_dim))
        elif self.field_interaction_type == "matrixed":
            self.interaction_weight = nn.Parameter(torch.Tensor(interact_dim, embedding_dim, embedding_dim))
        else:
            raise ValueError("field_interaction_type={} is not supported.".format(self.field_interaction_type))
        nn.init.xavier_normal_(self.interaction_weight)
        self.lr_layer = LogisticRegression(feature_map)
        self.triu_index = torch.triu_indices(num_fields, num_fields, offset=1).to(self.device)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        left_emb = torch.index_select(feature_emb, 1, self.triu_index[0])
        right_emb = torch.index_select(feature_emb, 1, self.triu_index[1])
        if self.field_interaction_type == "vectorized":
            left_emb = left_emb * self.interaction_weight
        elif self.field_interaction_type == "matrixed":
            left_emb = torch.matmul(left_emb.unsqueeze(2), self.interaction_weight).squeeze(2)
        y_pred = (left_emb * right_emb).sum(dim=-1).sum(dim=-1, keepdim=True)
        y_pred += self.lr_layer(X)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

