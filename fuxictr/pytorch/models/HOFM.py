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

from torch import nn
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import LR_Layer, EmbeddingLayer, InnerProductLayer
from itertools import combinations


class HOFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="HOFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 order=3,
                 embedding_dim=10,
                 reuse_embedding=False,
                 embedding_dropout=0,
                 regularizer=None, 
                 **kwargs):
        super(HOFM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=regularizer, 
                                   net_regularizer=regularizer,
                                   **kwargs)
        self.order = order
        assert order >= 2, "order >= 2 is required in HOFM!"
        self.reuse_embedding = reuse_embedding
        if reuse_embedding:
            assert isinstance(embedding_dim, int), "embedding_dim should be an integer when reuse_embedding=True."
            self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        else:
            if not isinstance(embedding_dim, list):
                embedding_dim = [embedding_dim] * (order - 1)
            self.embedding_layers = nn.ModuleList([EmbeddingLayer(feature_map, embedding_dim[i]) \
                                                   for i in range(order - 1)])
        self.inner_product_layer = InnerProductLayer(feature_map.num_fields)
        self.lr_layer = LR_Layer(feature_map, use_bias=True)
        self.output_activation = self.get_output_activation(task)
        self.field_conjunction_dict = dict()
        for order_i in range(3, self.order + 1):
            order_i_conjunction = zip(*list(combinations(range(feature_map.num_fields), order_i)))
            for k, field_index in enumerate(order_i_conjunction):
                self.field_conjunction_dict[(order_i, k)] = torch.LongTensor(field_index)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        y_pred = self.lr_layer(X)
        if self.reuse_embedding:
            feature_emb = self.embedding_layer(X)
        for i in range(2, self.order + 1):
            order_i_out = self.high_order_interaction(feature_emb if self.reuse_embedding \
                                                      else self.embedding_layers[i - 2](X), order_i=i)
            y_pred += order_i_out
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def high_order_interaction(self, feature_emb, order_i):
        if order_i == 2:
            interaction_out = self.inner_product_layer(feature_emb)
        elif order_i > 2:
            index = self.field_conjunction_dict[(order_i, 0)].to(self.device)
            hadamard_product = torch.index_select(feature_emb, 1, index)
            for k in range(1, order_i):
                index = self.field_conjunction_dict[(order_i, k)].to(self.device)
                hadamard_product = hadamard_product * torch.index_select(feature_emb, 1, index)
            interaction_out = hadamard_product.sum((1, 2)).view(-1, 1)
        return interaction_out
