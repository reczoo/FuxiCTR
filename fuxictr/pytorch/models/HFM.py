# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import DNN_Layer, EmbeddingLayer_v2, LR_Layer, HolographicInteractionLayer
from itertools import combinations


class HFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="HFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 interaction_type="circular_convolution",
                 use_dnn=True,
                 hidden_units=[64, 64],
                 hidden_activations=["relu", "relu"],
                 batch_norm=False,
                 embedding_dropout=0,
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(HFM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer_v2(feature_map, 
                                                 embedding_dim, 
                                                 embedding_dropout)
        self.lr_layer = LR_Layer(feature_map, final_activation=None)
        self.hfm_layer = HolographicInteractionLayer(feature_map.num_fields, interaction_type=interaction_type)
        self.use_dnn = use_dnn
        if self.use_dnn:
            input_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2) * embedding_dim
            self.dnn = DNN_Layer(input_dim=input_dim,
                                 output_dim=1, 
                                 hidden_units=hidden_units,
                                 hidden_activations=hidden_activations,
                                 final_activation=None,
                                 dropout_rates=net_dropout, 
                                 batch_norm=batch_norm)
        else:
            self.proj_h = nn.Linear(embedding_dim, 1, bias=False)
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)
            
    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        interact_out = self.hfm_layer(feature_emb)
        if self.use_dnn:
            hfm_out = self.dnn(torch.flatten(interact_out, start_dim=1))
        else:
            hfm_out = self.proj_h(interact_out.sum(dim=1))
        y_pred = hfm_out + self.lr_layer(X)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

