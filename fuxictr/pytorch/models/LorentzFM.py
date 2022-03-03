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
from itertools import combinations
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, InnerProductLayer


class LorentzFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="LorentzFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 embedding_dropout=0,
                 regularizer=None, 
                 **kwargs):
        super(LorentzFM, self).__init__(feature_map, 
                                        model_id=model_id, 
                                        gpu=gpu, 
                                        embedding_regularizer=regularizer, 
                                        net_regularizer=regularizer,
                                        **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.inner_product_layer = InnerProductLayer(feature_map.num_fields, output="inner_product")
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X) # bs x field x dim
        inner_product = self.inner_product_layer(feature_emb) # bs x (field x (field - 1) / 2)
        zeroth_components = self.get_zeroth_components(feature_emb) # batch * field
        y_pred = self.triangle_pooling(inner_product, zeroth_components) 
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def get_zeroth_components(self, feature_emb):
        '''
        compute the 0th component
        '''
        sum_of_square = torch.sum(feature_emb ** 2, dim=-1) # batch * field
        zeroth_components = torch.sqrt(sum_of_square + 1) # beta = 1
        return zeroth_components # batch * field

    def triangle_pooling(self, inner_product, zeroth_components):
        '''
        T(u,v) = (1 - <u, v>L - u0 - v0) / (u0 * v0)
               = (1 + u0 * v0 - inner_product - u0 - v0) / (u0 * v0)
               = 1 + (1 - inner_product - u0 - v0) / (u0 * v0)
        '''
        num_fields = zeroth_components.size(1)
        p, q = zip(*list(combinations(range(num_fields), 2)))
        u0, v0 = zeroth_components[:, p], zeroth_components[:, q]  # batch * (f(f-1)/2)
        score_tensor = 1 + torch.div(1 - inner_product - u0 - v0, u0 * v0) # batch * (f(f-1)/2)
        output = torch.sum(score_tensor, dim=1) # batch * 1
        return output