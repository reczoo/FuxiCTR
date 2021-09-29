# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingLayer
from ...pytorch.utils import set_activation

class DeepCrossing(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepCrossing", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 residual_blocks=[64, 64, 64],
                 hidden_activations="ReLU", 
                 embedding_dropout=0,
                 net_dropout=0, 
                 batch_norm=False, 
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DeepCrossing, self).__init__(feature_map, 
                                           model_id=model_id, 
                                           gpu=gpu, 
                                           embedding_regularizer=embedding_regularizer, 
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, 
                                              embedding_dim, 
                                              embedding_dropout)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(residual_blocks)
        layers = []
        input_dim = feature_map.num_fields * embedding_dim
        for hidden_dim, hidden_activation in zip(residual_blocks, hidden_activations):
            layers.append(ResidualBlock(input_dim, 
                                        hidden_dim,
                                        hidden_activation,
                                        net_dropout,
                                        use_residual,
                                        batch_norm))
        layers.append(nn.Linear(input_dim, 1))
        self.crossing_layer = nn.Sequential(*layers) # * used to unpack list
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb_list = self.embedding_layer(X)
        concate_feature_emb = torch.cat(feature_emb_list, dim=1)
        y_pred = self.crossing_layer(concate_feature_emb)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict


class ResidualBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_activation="ReLU",
                 dropout_rate=0,
                 use_residual=True,
                 batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.activation_layer = set_activation(hidden_activation)
        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   self.activation_layer,
                                   nn.Linear(hidden_dim, input_dim))
        self.use_residual = use_residual
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, X):
        X_out = self.layer(X)
        if self.use_residual:
            X_out = X_out + X
        if self.batch_norm is not None:
            X_out = self.batch_norm(X_out)
        output = self.activation_layer(X_out)
        if self.dropout is not None:
            output = self.dropout(output)
        return output



