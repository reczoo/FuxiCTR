# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from torch import nn
import torch
from .base_model import BaseModel
from ..layers import LR_Layer, EmbeddingLayer, DNN_Layer


class ONN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="ONN", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=2, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU",
                 embedding_dropout=0,
                 net_dropout = 0,
                 batch_norm = False,
                 **kwargs):
        super(ONN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs) 
        self.num_fields = feature_map.num_fields
        input_dim = embedding_dim * self.num_fields + int(self.num_fields * (self.num_fields - 1) / 2)
        self.dnn = DNN_Layer(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             final_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True)
        self.embedding_layers = nn.ModuleList([EmbeddingLayer(feature_map, 
                                                              embedding_dim, 
                                                              embedding_dropout) 
                                               for _ in range(self.num_fields)]) # f x f x bs x dim
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        field_aware_emb_list = [each_layer(X) for each_layer in self.embedding_layers] # list of list (bs x d)
        # copy_embedding
        copy_embedding = torch.cat(field_aware_emb_list[0], dim=1)
        # field-aware interaction
        ffm_out = self.field_aware_interaction(field_aware_emb_list[1:])
        dnn_input = torch.cat([copy_embedding, ffm_out], dim=1)
        y_pred = self.dnn(dnn_input)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"y_pred": y_pred, "loss": loss}
        return return_dict

    def field_aware_interaction(self, field_aware_emb_list):
        interaction = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                v_ij = field_aware_emb_list[j - 1][i]
                v_ji = field_aware_emb_list[i][j]
                dot = torch.sum(torch.mul(v_ij, v_ji), dim=1, keepdim=True)
                interaction.append(dot)
        return torch.cat(interaction, dim=1)
