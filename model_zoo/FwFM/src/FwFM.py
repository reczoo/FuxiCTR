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
from fuxictr.pytorch.layers import FeatureEmbedding, InnerProductInteraction


class FwFM(BaseModel):
    """ The FwFM model
        Reference:
          - Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising, WWW'2018.
    """
    def __init__(self, 
                 feature_map, 
                 model_id="FwFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 regularizer=None, 
                 linear_type="FiLV", 
                 **kwargs):
        """ 
        linear_type: `LW`, `FeLV`, or `FiLV`
        """
        super(FwFM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=regularizer, 
                                   net_regularizer=regularizer,
                                   **kwargs) 
        interact_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2)
        self.interaction_weight = nn.Linear(interact_dim, 1)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.inner_product_layer = InnerProductInteraction(feature_map.num_fields, output="inner_product")
        self._linear_type = linear_type
        if linear_type == "LW":
            self.linear_weight_layer = FeatureEmbedding(feature_map, 1, use_pretrain=False)
        elif linear_type == "FeLV":
            self.linear_weight_layer = FeatureEmbedding(feature_map, embedding_dim)
        elif linear_type == "FiLV":
            self.linear_weight_layer = nn.Linear(feature_map.num_fields * embedding_dim, 1, bias=False)
        else:
            raise NotImplementedError("linear_type={} is not supported.".format(linear_type))
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        inner_product_vec = self.inner_product_layer(feature_emb)
        poly2_part = self.interaction_weight(inner_product_vec)
        if self._linear_type == "LW":
            linear_weights = self.linear_weight_layer(X)
            linear_part = linear_weights.sum(dim=1)
        elif self._linear_type == "FeLV":
            linear_weights = self.linear_weight_layer(X)
            linear_part = (feature_emb * linear_weights).sum((1, 2)).view(-1, 1)
        elif self._linear_type == "FiLV":
            linear_part = self.linear_weight_layer(feature_emb.flatten(start_dim=1))
        y_pred = poly2_part + linear_part # bias added in poly2_part
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

