# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from torch import nn
import torch
from .base_model import BaseModel
from ..layers import DNN_Layer, EmbeddingLayer, LR_Layer, InnerProductLayer


class NFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="NFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 embedding_dropout=0,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(NFM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs) 
        self.embedding_layer = EmbeddingLayer(feature_map, 
                                              embedding_dim, 
                                              embedding_dropout=embedding_dropout)
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=False)
        self.inner_product_layer = InnerProductLayer(output="bi_vector")
        self.dnn = DNN_Layer(input_dim=embedding_dim,
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
        y_pred = self.lr_layer(X)
        feature_emb_list = self.embedding_layer(X)
        inner_product_tensor = self.inner_product_layer(feature_emb_list)
        bi_pooling_tensor = inner_product_tensor.view(self.batch_size, -1)
        y_pred += self.dnn(bi_pooling_tensor)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict