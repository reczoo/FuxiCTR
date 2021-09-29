# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v3, DNN_Layer

class DCN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCN", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_dim=10, 
                 dnn_hidden_units=[], 
                 dnn_activations="ReLU",
                 crossing_layers=3, 
                 embedding_dropout=0,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DCN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer_v3(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim
        self.dnn = DNN_Layer(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             final_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, crossing_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1) # [cross_part, dnn_part] -> logit
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(flat_feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(flat_feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(CrossInteractionLayer(input_dim)
                                       for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i


class CrossInteractionLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out
