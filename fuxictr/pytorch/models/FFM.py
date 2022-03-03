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


class FFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=2, 
                 regularizer=None, 
                 **kwargs):
        super(FFM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=regularizer, 
                                  net_regularizer=regularizer,
                                  **kwargs)
        self.num_fields = feature_map.num_fields
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=True)
        self.embedding_layers = nn.ModuleList([EmbeddingLayer(feature_map, embedding_dim) 
                                               for x in range(self.num_fields - 1)]) # (F - 1) x F x bs x dim
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        lr_out = self.lr_layer(X)
        field_wise_emb_list = [each_layer(X) for each_layer in self.embedding_layers] # (F - 1) list of bs x F x d
        ffm_out = self.ffm_interaction(field_wise_emb_list)
        y_pred = lr_out + ffm_out
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def ffm_interaction(self, field_wise_emb_list):
        dot = 0
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                v_ij = field_wise_emb_list[j - 1][:, i, :]
                v_ji = field_wise_emb_list[i][:, j, :]
                dot += torch.sum(v_ij * v_ji, dim=1, keepdim=True)
        return dot
