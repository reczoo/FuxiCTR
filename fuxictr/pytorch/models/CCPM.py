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
from .base_model import BaseModel
from ..layers import EmbeddingLayer, CCPM_ConvLayer
from ...pytorch.torch_utils import set_activation

class CCPM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="CCPM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 channels=[4, 4, 2],
                 kernel_heights=[6, 5, 3],
                 activation="Tanh",
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(CCPM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs) 
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.conv_layer = CCPM_ConvLayer(feature_map.num_fields, 
                                         channels=channels, 
                                         kernel_heights=kernel_heights, 
                                         activation=activation)
        conv_out_dim = 3 * embedding_dim * channels[-1] # 3 is k-max-pooling size of the last layer
        self.fc = nn.Linear(conv_out_dim, 1)
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.apply(self.init_weights)
            
    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        conv_in = torch.unsqueeze(feature_emb, 1) # shape (bs, 1, field, emb)
        conv_out = self.conv_layer(conv_in)
        flatten_out = torch.flatten(conv_out, start_dim=1)
        y_pred = self.fc(flatten_out)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict





