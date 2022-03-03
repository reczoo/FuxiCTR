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
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, InnerProductLayer, LR_Layer


class AFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 attention_dropout=[0, 0],
                 attention_dim=10,
                 use_attention=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(AFM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.use_attention = use_attention
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.product_layer = InnerProductLayer(feature_map.num_fields, output="elementwise_product")
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=True)
        self.attention = nn.Sequential(nn.Linear(embedding_dim, attention_dim),
                                       nn.ReLU(),
                                       nn.Linear(attention_dim, 1, bias=False),
                                       nn.Softmax(dim=1))
        self.weight_p = nn.Linear(embedding_dim, 1, bias=False)
        self.dropout1 = nn.Dropout(attention_dropout[0])
        self.dropout2 = nn.Dropout(attention_dropout[1])
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        elementwise_product = self.product_layer(feature_emb) # bs x f(f-1)/2 x dim
        if self.use_attention:
            attention_weight = self.attention(elementwise_product)
            attention_weight = self.dropout1(attention_weight)
            attention_sum = torch.sum(attention_weight * elementwise_product, dim=1)
            attention_sum = self.dropout2(attention_sum)
            afm_out = self.weight_p(attention_sum)
        else:
            afm_out = torch.flatten(elementwise_product, start_dim=1).sum(dim=-1).unsqueeze(-1)
        y_pred = self.lr_layer(X) + afm_out
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
