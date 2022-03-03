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
from fuxictr.pytorch.layers import EmbeddingLayer, MLP_Layer


class ONNv2(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="ONNv2", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=2, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU",
                 embedding_dropout=0,
                 net_dropout = 0,
                 batch_norm = False,
                 **kwargs):
        super(ONNv2, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.num_fields = feature_map.num_fields
        self.embedding_dim = embedding_dim
        self.interact_units = int(self.num_fields * (self.num_fields - 1) / 2)
        self.dnn = MLP_Layer(input_dim=embedding_dim * self.num_fields + self.interact_units,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim * self.num_fields) # b x f x dim*f
        self.diag_mask = torch.eye(self.num_fields).byte().to(self.device)
        self.upper_triange_mask = torch.triu(torch.ones(self.num_fields, self.num_fields), 1).byte().to(self.device)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        field_wise_embedding = self.embedding_layer(X).view(-1, self.num_fields, self.num_fields, self.embedding_dim)
        copy_embedding = torch.masked_select(field_wise_embedding, self.diag_mask.unsqueeze(-1)).view(self.batch_size, -1)
        ffm_out = self.ffm_interaction(field_wise_embedding)
        dnn_input = torch.cat([copy_embedding, ffm_out], dim=1)
        y_pred = self.dnn(dnn_input)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def ffm_interaction(self, field_wise_embedding):
        out = (field_wise_embedding.transpose(1, 2) * field_wise_embedding).sum(dim=-1)
        out = torch.masked_select(out, self.upper_triange_mask).view(-1, self.interact_units)
        return out
