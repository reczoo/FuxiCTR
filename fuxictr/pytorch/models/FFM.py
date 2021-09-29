# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from torch import nn
import torch
from .base_model import BaseModel
from ..layers import LR_Layer, EmbeddingLayer


class FFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=2, 
                 regularizer=None, 
                 embedding_dropout=0,
                 **kwargs):
        super(FFM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=regularizer, 
                                  net_regularizer=regularizer,
                                  **kwargs) 
        self.num_fields = feature_map.num_fields
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=True)
        self.embedding_layers = nn.ModuleList([EmbeddingLayer(feature_map, embedding_dim, embedding_dropout) 
                                               for x in range(self.num_fields - 1)]) # (F - 1) x F x bs x dim
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        lr_out = self.lr_layer(X)
        field_aware_emb_list = [each_layer(X) for each_layer in self.embedding_layers] # (F - 1) x F x bs x d
        ffm_out = self.field_aware_interaction(field_aware_emb_list)
        y_pred = lr_out + ffm_out
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"y_pred": y_pred, "loss": loss}
        return return_dict

    def field_aware_interaction(self, field_aware_emb_list):
        dot = 0
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                v_ij = field_aware_emb_list[j - 1][i]
                v_ji = field_aware_emb_list[i][j]
                dot += torch.sum(torch.mul(v_ij, v_ji), dim=1, keepdim=True)
        return dot
