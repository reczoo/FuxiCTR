# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from torch import nn
import torch
from .base_model import BaseModel
from ..layers import DNN_Layer, EmbeddingLayer, InnerProductLayer


class PNN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="PNN", 
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
                 product_type="inner", 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(PNN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs) 
        self.embedding_layer = EmbeddingLayer(feature_map, 
                                              embedding_dim, 
                                              embedding_dropout=embedding_dropout)
        if product_type != "inner":
            raise NotImplementedError("product_type={} has not been implemented.".format(product_type))
        self.inner_product_layer = InnerProductLayer(output="dot_vector")
        input_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2) \
                  + feature_map.num_fields * embedding_dim
        self.dnn = DNN_Layer(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             final_activation=self.get_final_activation(task),
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) 
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)
            
    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb_list = self.embedding_layer(X)
        inner_product_vec = self.inner_product_layer(feature_emb_list)
        dense_input = torch.cat(feature_emb_list + [inner_product_vec], dim=1)
        y_pred = self.dnn(dense_input)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict