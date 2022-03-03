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
from fuxictr.pytorch.layers import LR_Layer, EmbeddingLayer


class FFMv2(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FFMv2", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=2, 
                 regularizer=None, 
                 **kwargs):
        super(FFMv2, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=regularizer, 
                                     net_regularizer=regularizer,
                                     **kwargs) 
        self.num_fields = feature_map.num_fields
        self.embedding_dim = embedding_dim
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=True)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim * (self.num_fields - 1)) # b x f x dim*(f-1)
        self.output_activation = self.get_output_activation(task)
        self.upper_triange_mask = torch.triu(torch.ones(self.num_fields, self.num_fields - 1), 0).byte().to(self.device)
        self.lower_triange_mask = torch.tril(torch.ones(self.num_fields, self.num_fields - 1), -1).byte().to(self.device)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        lr_out = self.lr_layer(X)
        field_wise_embedding = self.embedding_layer(X).view(-1, self.num_fields, self.num_fields - 1, self.embedding_dim)
        ffm_out = self.ffm_interaction(field_wise_embedding)
        y_pred = lr_out + ffm_out
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def ffm_interaction(self, field_wise_embedding):
        upper_tensor = torch.masked_select(field_wise_embedding, self.upper_triange_mask.unsqueeze(-1))
        lower_tensor = torch.masked_select(field_wise_embedding.transpose(1, 2), self.lower_triange_mask.t().unsqueeze(-1))
        out = (upper_tensor * lower_tensor).view(self.batch_size, -1).sum(dim=-1, keepdim=True)
        return out
