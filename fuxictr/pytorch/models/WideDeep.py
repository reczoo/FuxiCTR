# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v3, DNN_Layer, LR_Layer


class WideDeep(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="WideDeep", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(WideDeep, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.embedding_layer = EmbeddingLayer_v3(feature_map, embedding_dim)
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=False)
        self.dnn = DNN_Layer(input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             final_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True)
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.lr_layer(X)
        y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
