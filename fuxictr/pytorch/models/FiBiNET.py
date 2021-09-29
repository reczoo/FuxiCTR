# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import DNN_Layer, EmbeddingLayer, SENET_Layer, BilinearInteractionLayer, LR_Layer


class FiBiNET(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FiBiNET", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_dim=10, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 reduction_ratio=3,
                 bilinear_type="field_interaction",
                 embedding_dropout=0,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(FiBiNET, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim, embedding_dropout)
        num_fields = feature_map.num_fields
        self.senet_layer = SENET_Layer(num_fields, reduction_ratio)
        self.bilinear_interaction = BilinearInteractionLayer(num_fields, embedding_dim, bilinear_type)
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=False)
        input_dim = num_fields * (num_fields - 1) * embedding_dim
        self.dnn = DNN_Layer(input_dim=input_dim,
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
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb_list = self.embedding_layer(X) # list of b x embedding_dim
        feature_emb = torch.stack(feature_emb_list, dim=1)
        senet_emb = self.senet_layer(feature_emb)
        bilinear_p = self.bilinear_interaction(feature_emb)
        bilinear_q = self.bilinear_interaction(senet_emb)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        dnn_out = self.dnn(comb_out)
        y_pred = self.lr_layer(X) + dnn_out
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {'y_pred': y_pred, 'loss':loss}
        return return_dict